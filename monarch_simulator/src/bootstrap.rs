use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use controller::bootstrap::bootstrap_controller;
use hyperactor::ActorHandle;
use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::ProcId;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::sim::SimAddr;
use hyperactor::channel::sim::operational_message_receiver;
use hyperactor::simnet::OperationalMessage;
use hyperactor::simnet::SpawnMesh;
use hyperactor_multiprocess::System;
use hyperactor_multiprocess::proc_actor::ProcActor;
use hyperactor_multiprocess::proc_actor::spawn;
use hyperactor_multiprocess::system::ServerHandle;
use hyperactor_multiprocess::system_actor::ProcLifecycleMode;
use hyperactor_multiprocess::system_actor::SystemActorParams;
use monarch_messages::worker::Factory;
use tokio::sync::Mutex;
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinHandle;
use torch_sys::Layout;
use torch_sys::ScalarType;

use crate::SimulatorError;
use crate::controller::SimControllerActor;
use crate::controller::SimControllerParams;
use crate::simulator::Simulator;
use crate::worker::Fabric;
use crate::worker::MockWorkerParams;
use crate::worker::WorkerActor;

/// Given a system ChannelAddr, return the bootstrap and listen addresses.
/// reuse_system_proxy = False will generate a new proxy address that is different from the system.
/// The new proxy address will have the same transport as the system proxy.
/// proxy address. reuse_system_proxy = True will reuse the system proxy address for both bootstrap
/// and listen addresses.
pub fn bootstrap_and_listen_address(
    system_addr: &ChannelAddr,
    reuse_system_proxy: bool,
) -> Result<(ChannelAddr, ChannelAddr), SimulatorError> {
    let listen_addr;
    let bootstrap_addr;
    if let ChannelAddr::Sim(system_sim_addr) = system_addr {
        // When we are getting the bootstrap address and listen address from a sim address,
        // it is important to set up the correct proxy addresses.
        // The bootstrap address and listen address should have reversed src and dst addresses.
        // bootstrap_addr's dst proxy should be the same as system_sim_addr's dst proxy.
        // listen_addr's dst proxy should have the same transport as system_sim_addr's dst proxy.
        let addr: ChannelAddr = ChannelAddr::any(system_sim_addr.dst().transport());
        let proxy_addr: ChannelAddr = if reuse_system_proxy {
            system_sim_addr.dst_proxy().clone()
        } else {
            ChannelAddr::any(system_sim_addr.dst_proxy().transport())
        };
        let sim_addr = SimAddr::new(
            addr.clone(),
            proxy_addr.clone(),
            system_sim_addr.dst().clone(),
            system_sim_addr.dst_proxy().clone(),
        )?;
        bootstrap_addr = ChannelAddr::Sim(sim_addr.clone());
        listen_addr = ChannelAddr::Sim(sim_addr);
    } else {
        bootstrap_addr = system_addr.clone();
        listen_addr = ChannelAddr::any(bootstrap_addr.transport());
    };
    Ok((bootstrap_addr, listen_addr))
}

/// spawns the system.
#[tracing::instrument("spawn_system")]
pub async fn spawn_system(system_addr: ChannelAddr) -> Result<ServerHandle> {
    // TODO: pass in as args
    let supervision_update_timeout = Duration::from_secs(120);
    let world_eviction_timeout = Duration::from_secs(120);

    let handle = System::serve(
        system_addr.clone(),
        SystemActorParams::new(supervision_update_timeout, world_eviction_timeout),
    )
    .await?;
    Ok(handle)
}

/// Spawns the controller proc and actor.
#[tracing::instrument("spawn_controller")]
pub async fn spawn_controller(
    system_addr: ChannelAddr,
    controller_actor_id: ActorId,
    worker_actor_id: ActorId,
) -> anyhow::Result<ActorHandle<ProcActor>> {
    let (bootstrap_addr, listen_addr) = bootstrap_and_listen_address(&system_addr, true)?;
    tracing::info!(
        "controller listen addr: {}, bootstrap addr: {}",
        &listen_addr,
        &bootstrap_addr
    );

    let worker_world_id = worker_actor_id.proc_id().world_id();
    let worker_name = worker_actor_id.name();
    let supervision_query_interval = Duration::from_secs(2);
    let supervision_update_interval = Duration::from_secs(2);
    let worker_progress_check_interval = Duration::from_secs(10);
    let operation_timeout = Duration::from_secs(120);
    let operations_per_worker_progress_request = 100;
    let proc_actor_handle = bootstrap_controller(
        bootstrap_addr,
        Some(listen_addr),
        controller_actor_id,
        1, /* num_procs */
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

// TODO(lky): delete spawn_sim_controller.
/// Spawns the sim controller proc and actor.
#[tracing::instrument("spawn_sim_controller")]
pub async fn spawn_sim_controller(
    system_addr: ChannelAddr,
    controller_actor_id: ActorId,
    worker_actor_id: ActorId,
) -> anyhow::Result<ActorHandle<ProcActor>> {
    let (bootstrap_addr, listen_addr) = bootstrap_and_listen_address(&system_addr, true)?;
    tracing::info!(
        "controller listen addr: {}, bootstrap addr: {}",
        &listen_addr,
        &bootstrap_addr
    );

    let (proc_actor_handle, controller_actor_ref) = SimControllerActor::bootstrap(
        controller_actor_id,
        listen_addr,
        bootstrap_addr,
        SimControllerParams::new(worker_actor_id),
        Duration::from_secs(1),
    )
    .await?;
    tracing::info!(
        "controller starts with id: {}",
        controller_actor_ref.actor_id()
    );

    Ok(proc_actor_handle)
}

/// Spawns workers. Right now, only one mocked worker is spawned. TODO: spawn multiple workers.
#[tracing::instrument("spawn_worker")]
pub async fn spawn_sim_worker(
    system_addr: ChannelAddr,
    worker_actor_id: ActorId,
    controller_actor_id: ActorId,
) -> anyhow::Result<ActorHandle<ProcActor>> {
    let (bootstrap_addr, listen_addr) = bootstrap_and_listen_address(&system_addr, true)?;
    let world_id = worker_actor_id.proc_id().world_id();
    tracing::info!(
        "worker {} listen addr: {}, bootstrap addr: {}",
        &worker_actor_id,
        &listen_addr,
        &bootstrap_addr
    );

    let supervision_update_interval = Duration::from_secs(1);
    let worker_proc_id = worker_actor_id.proc_id();
    let bootstrap = ProcActor::bootstrap(
        worker_proc_id.clone(),
        world_id.clone(),
        listen_addr,
        bootstrap_addr,
        supervision_update_interval,
        HashMap::new(),
        ProcLifecycleMode::ManagedBySystem,
    )
    .await?;
    let mut system = hyperactor_multiprocess::System::new(system_addr.clone());
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
pub async fn boostrap(system_addr: ChannelAddr) -> Result<JoinHandle<()>> {
    // TODO: enable supervision events.
    let mut operational_message_rx = operational_message_receiver().await?;
    let simulator = Arc::new(Mutex::new(Simulator::new(system_addr.clone()).await?));
    let operational_listener_handle = {
        let simulator = simulator.clone();
        tokio::spawn(async move {
            handle_operational_message(&mut operational_message_rx, simulator).await
        })
    };

    Ok(operational_listener_handle)
}

async fn handle_operational_message(
    operational_message_rx: &mut Receiver<OperationalMessage>,
    simulator: Arc<Mutex<Simulator>>,
) {
    while let Some(msg) = operational_message_rx.recv().await {
        tracing::info!("received operational message: {:?}", msg);
        match msg {
            OperationalMessage::SpawnMesh(SpawnMesh {
                system_addr,
                controller_actor_id,
                worker_world,
            }) => {
                let worker_actor_id = ActorId(ProcId(worker_world, 0), "root".into(), 0);
                if let Err(e) = simulator
                    .lock()
                    .await
                    .spawn_mesh(system_addr, controller_actor_id, worker_actor_id)
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
        }
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::channel::ChannelTransport;

    use super::*;

    #[tokio::test]
    async fn test_local_bootstrap_and_listen_address() {
        let system_addr = ChannelAddr::any(ChannelTransport::Local);
        let (bootstrap_addr, listen_addr) =
            bootstrap_and_listen_address(&system_addr, false).unwrap();
        assert_eq!(system_addr.clone(), bootstrap_addr);
        assert_eq!(system_addr, listen_addr);
    }

    #[tokio::test]
    async fn test_sim_bootstrap_and_listen_address() {
        let system_sim_addr = "sim!unix!system,unix!system_proxy,unix!system,unix!system_proxy"
            .parse::<ChannelAddr>()
            .unwrap();
        let (bootstrap_addr, listen_addr) =
            bootstrap_and_listen_address(&system_sim_addr, false).unwrap();

        let ChannelAddr::Sim(bootstrap_addr) = bootstrap_addr else {
            panic!("bootstrap_addr is not a sim address");
        };
        let ChannelAddr::Sim(listen_addr) = listen_addr else {
            panic!("listen_addr is not a sim address");
        };
        let ChannelAddr::Sim(system_addr) = &system_sim_addr else {
            panic!("system_sim_addr is not a sim address");
        };

        assert_eq!(bootstrap_addr.dst(), system_addr.dst());
        assert_eq!(bootstrap_addr.dst_proxy(), system_addr.dst_proxy());

        assert_eq!(listen_addr.dst(), system_addr.dst());
        assert_eq!(listen_addr.dst_proxy(), system_addr.dst_proxy());

        assert_eq!(bootstrap_addr.src_proxy(), listen_addr.src_proxy());
        assert_eq!(bootstrap_addr.src(), listen_addr.src());
        assert_eq!(
            bootstrap_addr.src_proxy().transport(),
            system_addr.dst_proxy().transport()
        );

        let (bootstrap_addr, listen_addr) =
            bootstrap_and_listen_address(&system_sim_addr, true).unwrap();
        let ChannelAddr::Sim(bootstrap_addr) = bootstrap_addr else {
            panic!("bootstrap_addr is not a sim address");
        };
        let ChannelAddr::Sim(listen_addr) = listen_addr else {
            panic!("listen_addr is not a sim address");
        };
        assert_eq!(bootstrap_addr.dst_proxy(), bootstrap_addr.src_proxy());
        assert_eq!(listen_addr.dst_proxy(), listen_addr.src_proxy());
        assert_eq!(bootstrap_addr.dst_proxy(), system_addr.dst_proxy());
        assert_eq!(listen_addr.dst_proxy(), system_addr.dst_proxy());
    }
}
