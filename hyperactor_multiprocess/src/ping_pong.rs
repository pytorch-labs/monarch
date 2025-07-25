/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use hyperactor::ActorRef;
    use hyperactor::Mailbox;
    use hyperactor::channel::ChannelAddr;
    use hyperactor::channel::sim;
    use hyperactor::channel::sim::SimAddr;
    use hyperactor::id;
    use hyperactor::reference::Index;
    use hyperactor::reference::WorldId;
    use hyperactor::simnet;
    use hyperactor::simnet::NetworkConfig;
    use hyperactor::test_utils::pingpong::PingPongActor;
    use hyperactor::test_utils::pingpong::PingPongActorParams;
    use hyperactor::test_utils::pingpong::PingPongMessage;

    use crate::System;
    use crate::proc_actor::ProcActor;
    use crate::proc_actor::spawn;
    use crate::system_actor::ProcLifecycleMode;

    #[tracing_test::traced_test]
    #[tokio::test]
    async fn test_sim_ping_pong() {
        let system_addr = "local!1".parse::<ChannelAddr>().unwrap();

        simnet::start();

        let system_sim_addr = SimAddr::new(system_addr.clone()).unwrap();
        let server_handle = System::serve(
            ChannelAddr::Sim(system_sim_addr.clone()),
            Duration::from_secs(10),
            Duration::from_secs(10),
        )
        .await
        .unwrap();

        let server_local_addr = server_handle.local_addr();
        let mut system = System::new(server_local_addr.clone());

        // Create a client on its individual proc that can talk to the system.
        let sys_mailbox = system.attach().await.unwrap();

        let world_id = id!(world);

        let ping_actor_ref = spawn_proc_actor(
            2,
            system_sim_addr.clone(),
            sys_mailbox.clone(),
            world_id.clone(),
        )
        .await;

        let pong_actor_ref =
            spawn_proc_actor(3, system_sim_addr, sys_mailbox.clone(), world_id.clone()).await;

        // Configure the simulation network.
        let simnet_config_yaml = r#"
edges:
  - src: local!1
    dst: local!2
    metadata:
      latency: 1
  - src: local!2
    dst: local!1
    metadata:
      latency: 1
  - src: local!1
    dst: local!3
    metadata:
      latency: 2
  - src: local!3
    dst: local!1
    metadata:
      latency: 2
"#;
        let simnet_config = NetworkConfig::from_yaml(simnet_config_yaml).unwrap();
        sim::update_config(simnet_config).await.unwrap();

        // Kick start the ping pong game by sending a message to the ping actor. The message will ask the
        // ping actor to deliver a message to the pong actor with TTL - 1. The pong actor will then
        // deliver a message to the ping actor with TTL - 2. This will continue until the TTL reaches 0.
        // The ping actor will then send a message to the done channel to indicate that the game is over.
        let (done_tx, done_rx) = sys_mailbox.open_once_port();
        let ping_pong_message = PingPongMessage(4, pong_actor_ref.clone(), done_tx.bind());
        ping_actor_ref
            .send(&sys_mailbox, ping_pong_message)
            .unwrap();

        assert!(done_rx.recv().await.unwrap());

        let records = simnet::simnet_handle().unwrap().close().await.unwrap();
        eprintln!(
            "records: {}",
            serde_json::to_string_pretty(&records).unwrap()
        );
    }

    async fn spawn_proc_actor(
        actor_index: Index,
        system_addr: SimAddr,
        sys_mailbox: Mailbox,
        world_id: WorldId,
    ) -> ActorRef<PingPongActor> {
        let proc_addr = format!("local!{}", actor_index)
            .parse::<ChannelAddr>()
            .unwrap();

        let proc_sim_addr = SimAddr::new(proc_addr.clone()).unwrap();
        let proc_listen_addr = ChannelAddr::Sim(proc_sim_addr);
        let proc_id = world_id.proc_id(actor_index);
        let proc_to_system = ChannelAddr::Sim(
            SimAddr::new_with_src(proc_addr.clone(), system_addr.addr().clone()).unwrap(),
        );
        let bootstrap = ProcActor::bootstrap(
            proc_id,
            world_id.clone(),
            proc_listen_addr,
            proc_to_system,
            Duration::from_secs(3),
            HashMap::new(),
            ProcLifecycleMode::ManagedBySystem,
        )
        .await
        .unwrap();

        let params = PingPongActorParams::new(None, None);
        spawn::<PingPongActor>(
            &sys_mailbox,
            &bootstrap.proc_actor.bind(),
            actor_index.to_string().as_str(),
            &params,
        )
        .await
        .unwrap()
    }
}
