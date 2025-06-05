/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::future::IntoFuture;

use anyhow::Result;
use futures::FutureExt;
use futures::future::BoxFuture;
use hyperactor::ActorHandle;
use hyperactor::ActorId;
use hyperactor::WorldId;
use hyperactor::channel::ChannelAddr;
use hyperactor_multiprocess::proc_actor::ProcActor;
use hyperactor_multiprocess::system::ServerHandle;

use crate::SimulatorError;
use crate::bootstrap::spawn_controller;
use crate::bootstrap::spawn_sim_worker;
use crate::bootstrap::spawn_system;

/// The simulator manages all of the meshes and the system handle.
#[derive(Debug)]
pub struct Simulator {
    /// A map from world name to actor handles in that world.
    worlds: HashMap<String, Vec<ActorHandle<ProcActor>>>,
    system_handle: ServerHandle,
}

impl Simulator {
    pub async fn new(system_addr: ChannelAddr) -> Result<Self> {
        Ok(Self {
            worlds: HashMap::new(),
            system_handle: spawn_system(system_addr).await?,
        })
    }

    pub async fn spawn_mesh(
        &mut self,
        system_addr: ChannelAddr,
        controller_actor_id: ActorId,
        worker_world_id: WorldId,
        world_size: usize,
    ) -> Result<()> {
        let controller = spawn_controller(
            system_addr.clone(),
            controller_actor_id.clone(),
            worker_world_id.clone(),
            world_size,
        )
        .await?;
        self.worlds.insert(
            controller_actor_id.world_name().to_string(),
            vec![controller],
        );

        for rank in 0..world_size {
            let worker = spawn_sim_worker(
                system_addr.clone(),
                worker_world_id.clone(),
                controller_actor_id.clone(),
                world_size,
                rank,
            )
            .await?;
            self.worlds
                .entry(worker_world_id.name().to_string())
                .or_insert(vec![])
                .push(worker);
        }
        Ok(())
    }

    /// Kills the actors within the given world.
    /// Returns error if there's no world found in the current simulator.
    pub fn kill_world(&mut self, world_name: &str) -> Result<(), SimulatorError> {
        let actors = self
            .worlds
            .remove(world_name)
            .ok_or(SimulatorError::WorldNotFound(world_name.to_string()))?;
        for actor in actors {
            actor.drain_and_stop()?;
        }
        Ok(())
    }
}

/// IntoFuture allows users to await the handle. The future resolves when
/// the simulator itself has all of the actor handles stopped.
impl IntoFuture for Simulator {
    type Output = ();
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let future = async move {
            self.system_handle.await;
            for actors in self.worlds.into_values() {
                for actor in actors {
                    actor.await;
                }
            }
        };

        future.boxed()
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::ActorId;
    use hyperactor::ProcId;
    use hyperactor::WorldId;
    use hyperactor::channel::ChannelAddr;
    use hyperactor::channel::ChannelTransport;
    use hyperactor::simnet;
    use rand::Rng;
    use rand::distributions::Alphanumeric;

    #[cfg(target_os = "linux")]
    fn random_str() -> String {
        rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(24)
            .map(char::from)
            .collect::<String>()
    }

    #[tracing_test::traced_test]
    #[tokio::test]
    async fn test_spawn_and_kill_mesh() {
        let proxy = format!("unix!@{}", random_str());

        simnet::start(
            ChannelAddr::any(ChannelTransport::Unix),
            proxy.parse().unwrap(),
            1000,
        )
        .unwrap();

        let system_addr = format!("sim!unix!@system,{}", &proxy)
            .parse::<ChannelAddr>()
            .unwrap();
        let mut simulator = super::Simulator::new(system_addr.clone()).await.unwrap();
        let mut controller_actor_ids = vec![];
        let mut worker_actor_ids = vec![];
        let n_meshes = 2;
        for i in 0..n_meshes {
            let controller_world_name = format!("controller_world_{}", i);
            let worker_world_name = format!("worker_world_{}", i);
            controller_actor_ids.push(ActorId(
                ProcId(WorldId(controller_world_name), 0),
                "root".into(),
                0,
            ));
            worker_actor_ids.push(ActorId(
                ProcId(WorldId(worker_world_name.clone()), 0),
                "root".into(),
                0,
            ));
            simulator
                .spawn_mesh(
                    system_addr.clone(),
                    controller_actor_ids.last().unwrap().clone(),
                    WorldId(worker_world_name),
                    1,
                )
                .await
                .unwrap();
        }

        assert_eq!(simulator.worlds.len(), n_meshes * 2);
        let world_name = controller_actor_ids[0].world_name();
        let controller_actor_handle = simulator.worlds.get(world_name).unwrap().first().unwrap();
        assert_eq!(controller_actor_handle.actor_id().world_name(), world_name);

        simulator.kill_world(world_name).unwrap();
        assert_eq!(simulator.worlds.len(), n_meshes * 2 - 1);
    }
}
