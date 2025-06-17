/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use controller::bootstrap::bootstrap_controller;
use hyperactor::ActorHandle;
use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::ProcId;
use hyperactor::WorldId;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::sim::AddressProxyPair;
use hyperactor::channel::sim::SimAddr;
use hyperactor::simnet;
use hyperactor::simnet::OperationalMessage;
use hyperactor::simnet::SpawnMesh;
use hyperactor::simnet::simnet_handle;
use hyperactor_multiprocess::System;
use hyperactor_multiprocess::proc_actor::ProcActor;
use hyperactor_multiprocess::proc_actor::spawn;
use hyperactor_multiprocess::system::ServerHandle;
use hyperactor_multiprocess::system_actor::ProcLifecycleMode;
use monarch_messages::worker::Factory;
use tokio::sync::Mutex;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::task::JoinHandle;
use torch_sys::Layout;
use torch_sys::ScalarType;

use crate::simulator::Simulator;
use crate::worker::Fabric;
use crate::worker::MockWorkerParams;
use crate::worker::WorkerActor;

/// spawns the system.
#[tracing::instrument("spawn_system")]
pub async fn spawn_system(system_addr: ChannelAddr) -> Result<ServerHandle> {
    // TODO: pass in as args
    let supervision_update_timeout = Duration::from_secs(120);
    let world_eviction_timeout = Duration::from_secs(120);

    let handle = System::serve(
        system_addr.clone(),
        supervision_update_timeout,
        world_eviction_timeout,
    )
    .await?;
    Ok(handle)
}

/// Spawns the controller proc and actor.
#[tracing::instrument("spawn_controller")]
pub async fn spawn_controller(
    bootstrap_addr: ChannelAddr,
    controller_actor_id: ActorId,
    worker_world_id: WorldId,
    world_size: usize,
) -> anyhow::Result<ActorHandle<ProcActor>> {
    let listen_addr = ChannelAddr::any(bootstrap_addr.transport());
    let ChannelAddr::Sim(bootstrap_addr) = bootstrap_addr else {
        panic!("bootstrap_addr must be a SimAddr");
    };
    let bootstrap_addr = ChannelAddr::Sim(
        SimAddr::new_with_src(
            AddressProxyPair {
                address: listen_addr.clone(),
                proxy: bootstrap_addr.proxy().clone(),
            },
            bootstrap_addr.addr().clone(),
            bootstrap_addr.proxy().clone(),
        )
        .unwrap(),
    );
    tracing::info!(
        "controller listen addr: {}, bootstrap addr: {}",
        &listen_addr,
        &bootstrap_addr
    );

    let worker_name = "worker";
    let supervision_query_interval = Duration::from_secs(2);
    let supervision_update_interval = Duration::from_secs(2);
    let worker_progress_check_interval = Duration::from_secs(10);
    let operation_timeout = Duration::from_secs(120);
    let operations_per_worker_progress_request = 100;
    let proc_actor_handle = bootstrap_controller(
        bootstrap_addr,
        Some(listen_addr),
        controller_actor_id,
        world_size,
        worker_world_id.clone(),
        worker_name.to_string(),
        supervision_query_interval,
        supervision_update_interval,
        worker_progress_check_interval,
        operation_timeout,
        operations_per_worker_progress_request,
        None,  /* extra_controller_labels */
        false, /* fail_on_worker_timeout  */
    )
    .await?;

    Ok(proc_actor_handle)
}

/// Spawns workers. Right now, only one mocked worker is spawned. TODO: spawn multiple workers.
#[tracing::instrument("spawn_worker")]
pub async fn spawn_sim_worker(
    bootstrap_addr: ChannelAddr,
    worker_world_id: WorldId,
    controller_actor_id: ActorId,
    world_size: usize,
    rank: usize,
) -> anyhow::Result<ActorHandle<ProcActor>> {
    let listen_addr = ChannelAddr::any(bootstrap_addr.transport());
    let worker_proc_id = ProcId(worker_world_id.clone(), rank);
    let worker_actor_id = ActorId(worker_proc_id.clone(), "worker".into(), 0);

    let ChannelAddr::Sim(bootstrap_addr) = bootstrap_addr else {
        panic!("bootstrap_addr must be a SimAddr");
    };
    let bootstrap_addr = ChannelAddr::Sim(
        SimAddr::new_with_src(
            AddressProxyPair {
                address: listen_addr.clone(),
                proxy: bootstrap_addr.proxy().clone(),
            },
            bootstrap_addr.addr().clone(),
            bootstrap_addr.proxy().clone(),
        )
        .unwrap(),
    );
    tracing::info!(
        "worker {} listen addr: {}, bootstrap addr: {}",
        &worker_actor_id,
        &listen_addr,
        &bootstrap_addr
    );

    let supervision_update_interval = Duration::from_secs(10);
    let bootstrap = ProcActor::bootstrap(
        worker_proc_id,
        worker_world_id,
        listen_addr,
        bootstrap_addr.clone(),
        supervision_update_interval,
        HashMap::new(),
        ProcLifecycleMode::ManagedBySystem,
    )
    .await?;
    let mut system = hyperactor_multiprocess::System::new(bootstrap_addr);
    let client = system.attach().await?;
    let fabric = Arc::new(Fabric::new());
    let factory = Factory {
        size: vec![2, 3],
        dtype: ScalarType::Float,
        layout: Layout::Strided,
        device: "cpu".try_into().unwrap(),
    };
    let controller_actor_ref = ActorRef::attest(controller_actor_id);
    let params = MockWorkerParams::new(
        worker_actor_id.rank(),
        worker_actor_id.clone(),
        fabric.clone(),
        factory.clone(),
        2,
        controller_actor_ref,
    );
    let _worker_actor_ref = spawn::<WorkerActor>(
        &client,
        &bootstrap.proc_actor.bind(),
        worker_actor_id.name(),
        &params,
    )
    .await?;
    Ok(bootstrap.proc_actor)
}

/// Bootstrap the simulation. Spawns the system, controllers, and workers.
/// Args:
///    system_addr: The address of the system actor.
pub async fn bootstrap(
    system_addr: ChannelAddr,
    proxy_addr: ChannelAddr,
    world_size: usize,
) -> Result<JoinHandle<()>> {
    // TODO: enable supervision events.
    let mut operational_message_rx = simnet::start(system_addr.clone(), proxy_addr, 1000)?;
    let simulator = Arc::new(Mutex::new(Simulator::new(system_addr).await?));
    let operational_listener_handle = {
        let simulator = simulator.clone();
        tokio::spawn(async move {
            handle_operational_message(&mut operational_message_rx, simulator, world_size).await
        })
    };

    Ok(operational_listener_handle)
}

async fn handle_operational_message(
    operational_message_rx: &mut UnboundedReceiver<OperationalMessage>,
    simulator: Arc<Mutex<Simulator>>,
    world_size: usize,
) {
    while let Some(msg) = operational_message_rx.recv().await {
        tracing::info!("received operational message: {:?}", msg);
        match msg {
            OperationalMessage::SpawnMesh(SpawnMesh {
                system_addr,
                controller_actor_id,
                worker_world,
            }) => {
                if let Err(e) = simulator
                    .lock()
                    .await
                    .spawn_mesh(system_addr, controller_actor_id, worker_world, world_size)
                    .await
                {
                    tracing::error!("failed to spawn mesh: {:?}", e);
                }
            }
            OperationalMessage::KillWorld(world_id) => {
                if let Err(e) = simulator.lock().await.kill_world(&world_id) {
                    tracing::error!("failed to kill world: {:?}", e);
                }
            }
            OperationalMessage::SetTrainingScriptState(state) => match simnet_handle() {
                Ok(handle) => handle.set_training_script_state(state),
                Err(e) => {
                    tracing::error!("failed to set training script state: {:?}", e);
                }
            },
        }
    }
}
