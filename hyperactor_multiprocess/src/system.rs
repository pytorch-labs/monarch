use std::collections::HashMap;
use std::future::IntoFuture;

use futures::FutureExt;
use futures::future::BoxFuture;
use hyperactor::actor::ActorError;
use hyperactor::actor::ActorHandle;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::clock::ClockKind;
use hyperactor::id;
use hyperactor::mailbox::BoxedMailboxSender;
use hyperactor::mailbox::Mailbox;
use hyperactor::mailbox::MailboxClient;
use hyperactor::mailbox::MailboxSender;
use hyperactor::mailbox::MailboxServer;
use hyperactor::mailbox::MailboxServerHandle;
use hyperactor::mailbox::MessageEnvelope;
use hyperactor::mailbox::Undeliverable;
use hyperactor::mailbox::monitored_return_handle;
use hyperactor::proc::Proc;
use system_actor::SystemActor;
use system_actor::SystemActorParams;
use system_actor::SystemMessageClient;
use tokio::join;

use crate::proc_actor::ProcMessage;
use crate::system_actor;
use crate::system_actor::ProcLifecycleMode;

/// Multiprocess system implementation.
#[derive(Debug)]
pub struct System {
    addr: ChannelAddr,
}

impl System {
    /// Spawns a system actor and serves it at the provided channel
    /// address. This becomes a well-known address with which procs
    /// can bootstrap.
    pub async fn serve(
        addr: ChannelAddr,
        params: SystemActorParams,
    ) -> Result<ServerHandle, anyhow::Error> {
        let clock = ClockKind::for_channel_addr(&addr);
        let (actor_handle, system_proc) = SystemActor::bootstrap_with_clock(params, clock).await?;
        actor_handle.bind::<SystemActor>();

        // Undeliverable messages encountered by the mailbox server
        // are to be returned to the system actor.
        let system_return_handle = actor_handle.port::<Undeliverable<MessageEnvelope>>();
        let (local_addr, rx) = channel::serve(addr).await?;
        // `MailboxServer::serve()`.
        let mailbox_handle = system_proc.clone().serve(rx, system_return_handle);

        Ok(ServerHandle {
            actor_handle,
            mailbox_handle,
            local_addr,
        })
    }

    /// Connect to the system at the provided address.
    pub fn new(addr: ChannelAddr) -> Self {
        Self { addr }
    }

    /// A sender capable of routing all messages to actors in the system.
    pub async fn sender(&self) -> Result<impl MailboxSender, anyhow::Error> {
        let tx = channel::dial(self.addr.clone())?;
        Ok(MailboxClient::new(tx))
    }

    /// Join the system ephemerally. This allocates an actor id, and returns the
    /// corresponding mailbox.
    ///
    /// TODO: figure out lifecycle management: e.g., should this be
    /// alive until all ports are deallocated and the receiver is dropped?
    pub async fn attach(&mut self) -> Result<Mailbox, anyhow::Error> {
        // TODO: just launch a proc actor here to handle the local
        // proc management.
        let world_id = id!(user);
        let proc = Proc::new(
            world_id.random_user_proc(),
            BoxedMailboxSender::new(self.sender().await?),
        );

        let (proc_addr, proc_rx) = channel::serve(ChannelAddr::any(self.addr.transport()))
            .await
            .unwrap();

        let _proc_serve_handle: MailboxServerHandle =
            proc.clone().serve(proc_rx, monitored_return_handle());

        // Now, pretend we are the proc actor, and use this to join the system.
        let proc_inst = proc.attach("proc")?;
        let (proc_tx, mut proc_rx) = proc_inst.open_port();

        system_actor::SYSTEM_ACTOR_REF
            .join(
                &proc_inst,
                world_id,
                /*proc_id=*/ proc.proc_id().clone(),
                /*proc_message_port=*/ proc_tx.bind(),
                proc_addr,
                HashMap::new(),
                ProcLifecycleMode::Detached,
            )
            .await
            .unwrap();
        let timeout = tokio::time::Duration::from_secs(
            std::env::var("MONARCH_MESSAGE_DELIVERY_TIMEOUT_SECS")
                .ok()
                .and_then(|val| val.parse::<u64>().ok())
                .unwrap_or(10),
        );
        loop {
            let result = tokio::time::timeout(timeout, proc_rx.recv()).await?;
            match result? {
                ProcMessage::Joined() => break,
                message => tracing::info!("proc message while joining: {:?}", message),
            }
        }

        proc.attach("user")
    }
}

/// Handle for a running system server.
#[derive(Debug)]
pub struct ServerHandle {
    actor_handle: ActorHandle<SystemActor>,
    mailbox_handle: MailboxServerHandle,
    local_addr: ChannelAddr,
}

impl ServerHandle {
    /// Stop the server. The user should join the handle after calling stop.
    pub async fn stop(&self) -> Result<(), ActorError> {
        // TODO: this needn't be async
        self.actor_handle.drain_and_stop()?;
        self.mailbox_handle.stop();
        Ok(())
    }

    /// The local (bound) address of the server.
    pub fn local_addr(&self) -> &ChannelAddr {
        &self.local_addr
    }

    /// The system actor handle.
    pub fn system_actor_handle(&self) -> &ActorHandle<SystemActor> {
        &self.actor_handle
    }
}

/// A future implementation for actor handle used for joining. It is
/// forwarded to the underlying join handles.
impl IntoFuture for ServerHandle {
    type Output = ();
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let future = async move {
            let _ = join!(self.actor_handle.into_future(), self.mailbox_handle);
        };
        future.boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::time::Duration;

    use anyhow::Context;
    use anyhow::Result;
    use async_trait::async_trait;
    use hyperactor::Actor;
    use hyperactor::ActorHandle;
    use hyperactor::ActorId;
    use hyperactor::ActorRef;
    use hyperactor::GangId;
    use hyperactor::Handler;
    use hyperactor::Instance;
    use hyperactor::Named;
    use hyperactor::PortRef;
    use hyperactor::ProcId;
    use hyperactor::WorldId;
    use hyperactor::actor::ActorStatus;
    use hyperactor::channel::ChannelAddr;
    use hyperactor::channel::ChannelTransport;
    use hyperactor::clock::Clock;
    use hyperactor::clock::RealClock;
    use hyperactor::id;
    use hyperactor::mailbox::BoxedMailboxSender;
    use hyperactor::mailbox::MailboxRouter;
    use hyperactor::mailbox::open_port;
    use hyperactor::message::Bind;
    use hyperactor::message::Bindings;
    use hyperactor::message::IndexedErasedUnbound;
    use hyperactor::message::Unbind;
    use hyperactor::message::Unbound;
    use hyperactor::proc::Proc;
    use hyperactor::test_utils::proc_supervison::ProcSupervisionCoordinator;
    use hyperactor::test_utils::tracing::set_tracing_env_filter;
    use hyperactor_mesh::comm::CommActor;
    use hyperactor_mesh::comm::CommActorParams;
    use hyperactor_mesh::comm::multicast::CastMessage;
    use hyperactor_mesh::comm::multicast::CastMessageEnvelope;
    use hyperactor_mesh::comm::multicast::DestinationPort;
    use hyperactor_mesh::comm::multicast::Uslice;
    use hyperactor_telemetry::env::execution_id;
    use maplit::hashset;
    use ndslice::Slice;
    use ndslice::selection;
    use serde::Deserialize;
    use serde::Serialize;
    use timed_test::async_timed_test;
    use tracing::Level;

    use super::*;
    use crate::System;
    use crate::proc_actor::Environment;
    use crate::proc_actor::ProcActor;
    use crate::proc_actor::ProcMessageClient;
    use crate::supervision::ProcSupervisor;
    use crate::system_actor::ProcLifecycleMode;
    use crate::system_actor::SYSTEM_ACTOR_REF;
    use crate::system_actor::Shape;
    use crate::system_actor::SystemActorParams;
    use crate::system_actor::SystemMessageClient;
    use crate::system_actor::SystemSnapshot;
    use crate::system_actor::SystemSnapshotFilter;
    use crate::system_actor::WorldSnapshot;
    use crate::system_actor::WorldSnapshotProcInfo;
    use crate::system_actor::WorldStatus;

    #[tokio::test]
    async fn test_join() {
        for transport in ChannelTransport::all() {
            // TODO: make ChannelAddr::any work even without
            #[cfg(not(target_os = "linux"))]
            if matches!(transport, ChannelTransport::Unix) {
                continue;
            }

            let system_handle = System::serve(
                ChannelAddr::any(transport),
                SystemActorParams::new(Duration::from_secs(10), Duration::from_secs(10)),
            )
            .await
            .unwrap();

            let mut system = System::new(system_handle.local_addr().clone());
            let client1 = system.attach().await.unwrap();
            let client2 = system.attach().await.unwrap();

            let (port, mut port_rx) = client2.open_port();

            port.bind().send(&client1, 123u64).unwrap();
            assert_eq!(port_rx.recv().await.unwrap(), 123u64);

            system_handle.stop().await.unwrap();
            system_handle.await;
        }
    }

    #[tokio::test]
    async fn test_system_snapshot() {
        let system_handle = System::serve(
            ChannelAddr::any(ChannelTransport::Local),
            SystemActorParams::new(Duration::from_secs(10), Duration::from_secs(10)),
        )
        .await
        .unwrap();

        let mut system = System::new(system_handle.local_addr().clone());
        let client = system.attach().await.unwrap();

        let sys_actor_handle = system_handle.system_actor_handle();
        // Check the inital state.
        {
            let snapshot = sys_actor_handle
                .snapshot(&client, SystemSnapshotFilter::all())
                .await
                .unwrap();
            assert_eq!(
                snapshot,
                SystemSnapshot {
                    worlds: HashMap::new(),
                    execution_id: execution_id(),
                }
            );
        }

        // Create a world named foo, and join a non-worker proc to it.
        let foo_world = {
            let foo_world_id = WorldId("foo_world".to_string());
            sys_actor_handle
                .upsert_world(
                    &client,
                    foo_world_id.clone(),
                    Shape::Definite(vec![2]),
                    5,
                    Environment::Local,
                    HashMap::new(),
                )
                .await
                .unwrap();
            {
                let snapshot = sys_actor_handle
                    .snapshot(&client, SystemSnapshotFilter::all())
                    .await
                    .unwrap();
                let time = snapshot
                    .worlds
                    .get(&foo_world_id)
                    .unwrap()
                    .status
                    .as_unhealthy()
                    .unwrap()
                    .clone();
                assert_eq!(
                    snapshot,
                    SystemSnapshot {
                        worlds: HashMap::from([(
                            foo_world_id.clone(),
                            WorldSnapshot {
                                host_procs: HashSet::new(),
                                procs: HashMap::new(),
                                status: WorldStatus::Unhealthy(time),
                                labels: HashMap::new(),
                            }
                        ),]),
                        execution_id: execution_id(),
                    }
                );
            }

            // Join a non-worker proc to the "foo" world.
            {
                let test_labels =
                    HashMap::from([("test_name".to_string(), "test_value".to_string())]);
                let listen_addr = ChannelAddr::any(ChannelTransport::Local);
                let proc_id = ProcId(foo_world_id.clone(), 1);
                ProcActor::try_bootstrap(
                    proc_id.clone(),
                    foo_world_id.clone(),
                    listen_addr,
                    system_handle.local_addr().clone(),
                    ActorRef::attest(proc_id.actor_id("supervision", 0)),
                    Duration::from_secs(30),
                    test_labels.clone(),
                    ProcLifecycleMode::ManagedBySystem,
                )
                .await
                .unwrap();

                let snapshot = sys_actor_handle
                    .snapshot(&client, SystemSnapshotFilter::all())
                    .await
                    .unwrap();
                let time = snapshot
                    .worlds
                    .get(&foo_world_id)
                    .unwrap()
                    .status
                    .as_unhealthy()
                    .unwrap()
                    .clone();
                let foo_world = (
                    foo_world_id.clone(),
                    WorldSnapshot {
                        host_procs: HashSet::new(),
                        procs: HashMap::from([(
                            proc_id.clone(),
                            WorldSnapshotProcInfo {
                                labels: test_labels.clone(),
                            },
                        )]),
                        status: WorldStatus::Unhealthy(time),
                        labels: HashMap::new(),
                    },
                );

                assert_eq!(
                    snapshot,
                    SystemSnapshot {
                        worlds: HashMap::from([foo_world.clone(),]),
                        execution_id: execution_id(),
                    },
                );

                // check snapshot world filters
                let snapshot = sys_actor_handle
                    .snapshot(
                        &client,
                        SystemSnapshotFilter {
                            worlds: vec![WorldId("none".to_string())],
                            world_labels: HashMap::new(),
                            proc_labels: HashMap::new(),
                        },
                    )
                    .await
                    .unwrap();
                assert!(snapshot.worlds.is_empty());
                // check actor filters
                let snapshot = sys_actor_handle
                    .snapshot(
                        &client,
                        SystemSnapshotFilter {
                            worlds: vec![],
                            world_labels: HashMap::new(),
                            proc_labels: test_labels.clone(),
                        },
                    )
                    .await
                    .unwrap();
                assert_eq!(snapshot.worlds.get(&foo_world_id).unwrap(), &foo_world.1);
                foo_world
            }
        };

        // Create a worker world from host procs.
        {
            let worker_world_id = WorldId("worker_world".to_string());
            let host_world_id = WorldId(("hostworker_world").to_string());
            let listen_addr: ChannelAddr = ChannelAddr::any(ChannelTransport::Local);
            // Join a host proc to the system first with no worker_world yet.
            let host_proc_id_1 = ProcId(host_world_id.clone(), 1);
            ProcActor::try_bootstrap(
                host_proc_id_1.clone(),
                host_world_id.clone(),
                listen_addr.clone(),
                system_handle.local_addr().clone(),
                ActorRef::attest(host_proc_id_1.actor_id("supervision", 0)),
                Duration::from_secs(30),
                HashMap::new(),
                ProcLifecycleMode::ManagedBySystem,
            )
            .await
            .unwrap();
            {
                let snapshot = sys_actor_handle
                    .snapshot(&client, SystemSnapshotFilter::all())
                    .await
                    .unwrap();
                assert_eq!(
                    snapshot,
                    SystemSnapshot {
                        worlds: HashMap::from([
                            foo_world.clone(),
                            (
                                worker_world_id.clone(),
                                WorldSnapshot {
                                    host_procs: HashSet::from([host_proc_id_1.clone()]),
                                    procs: HashMap::new(),
                                    status: WorldStatus::AwaitingCreation,
                                    labels: HashMap::new(),
                                }
                            ),
                        ]),
                        execution_id: execution_id(),
                    },
                );
            }

            // Upsert the worker world.
            sys_actor_handle
                .upsert_world(
                    &client,
                    worker_world_id.clone(),
                    // 12 worker procs in total, 8 per host. That means one
                    // host spawn 8 procs, and another host spawn 4 procs.
                    Shape::Definite(vec![3, 4]),
                    8,
                    Environment::Local,
                    HashMap::new(),
                )
                .await
                .unwrap();
            // Wait for the worker procs being spawned.
            RealClock.sleep(Duration::from_secs(2)).await;
            {
                let snapshot = sys_actor_handle
                    .snapshot(&client, SystemSnapshotFilter::all())
                    .await
                    .unwrap();
                let time = snapshot
                    .worlds
                    .get(&worker_world_id)
                    .unwrap()
                    .status
                    .as_unhealthy()
                    .unwrap()
                    .clone();
                assert_eq!(
                    snapshot,
                    SystemSnapshot {
                        worlds: HashMap::from([
                            foo_world.clone(),
                            (
                                worker_world_id.clone(),
                                WorldSnapshot {
                                    host_procs: HashSet::from([host_proc_id_1.clone()]),
                                    procs: (8..12)
                                        .map(|i| (
                                            ProcId(worker_world_id.clone(), i),
                                            WorldSnapshotProcInfo {
                                                labels: HashMap::new()
                                            }
                                        ))
                                        .collect(),
                                    status: WorldStatus::Unhealthy(time),
                                    labels: HashMap::new(),
                                }
                            ),
                        ]),
                        execution_id: execution_id(),
                    },
                );
            }

            let host_proc_id_0 = ProcId(host_world_id.clone(), 0);
            ProcActor::try_bootstrap(
                host_proc_id_0.clone(),
                host_world_id.clone(),
                listen_addr,
                system_handle.local_addr().clone(),
                ActorRef::attest(host_proc_id_0.actor_id("supervision", 0)),
                Duration::from_secs(30),
                HashMap::new(),
                ProcLifecycleMode::ManagedBySystem,
            )
            .await
            .unwrap();

            // Wait for the worker procs being spawned.
            RealClock.sleep(Duration::from_secs(2)).await;
            {
                let snapshot = sys_actor_handle
                    .snapshot(&client, SystemSnapshotFilter::all())
                    .await
                    .unwrap();
                assert_eq!(
                    snapshot,
                    SystemSnapshot {
                        worlds: HashMap::from([
                            foo_world,
                            (
                                worker_world_id.clone(),
                                WorldSnapshot {
                                    host_procs: HashSet::from([host_proc_id_0, host_proc_id_1]),
                                    procs: HashMap::from_iter((0..12).map(|i| (
                                        ProcId(worker_world_id.clone(), i),
                                        WorldSnapshotProcInfo {
                                            labels: HashMap::new()
                                        }
                                    ))),
                                    // We have 12 procs ready to serve a 3 X 4 world.
                                    status: WorldStatus::Live,
                                    labels: HashMap::new(),
                                }
                            ),
                        ]),
                        execution_id: execution_id(),
                    }
                );
            }
        }
    }

    // The test consists of 2 steps:
    // 1. spawn a system with 2 host procs, and 8 worker procs. For each worker
    //    proc, spawn a root actor with a children tree.
    // 2. Send a Stop message to system actor, and verify everything will be
    //    shut down.
    #[tracing_test::traced_test]
    #[async_timed_test(timeout_secs = 60)]
    async fn test_system_shutdown() {
        let system_handle = System::serve(
            ChannelAddr::any(ChannelTransport::Local),
            SystemActorParams::new(Duration::from_secs(10), Duration::from_secs(10)),
        )
        .await
        .unwrap();
        let system_supervision_ref: ActorRef<ProcSupervisor> =
            ActorRef::attest(SYSTEM_ACTOR_REF.actor_id().clone());

        let mut system = System::new(system_handle.local_addr().clone());
        let client = system.attach().await.unwrap();

        let sys_actor_handle = system_handle.system_actor_handle();

        // Create a worker world from host procs.
        let worker_world_id = WorldId("worker_world".to_string());
        let shape = vec![2, 2, 4];
        let host_proc_actors = {
            let host_world_id = WorldId(("hostworker_world").to_string());
            // Upsert the worker world.
            sys_actor_handle
                .upsert_world(
                    &client,
                    worker_world_id.clone(),
                    // 2 worker procs in total, 8 per host.
                    Shape::Definite(shape.clone()),
                    8,
                    Environment::Local,
                    HashMap::new(),
                )
                .await
                .unwrap();

            // Bootstrap the host procs, which will lead to work procs being spawned.
            let futs = (0..2).map(|i| {
                let host_proc_id = ProcId(host_world_id.clone(), i);
                ProcActor::try_bootstrap(
                    host_proc_id.clone(),
                    host_world_id.clone(),
                    ChannelAddr::any(ChannelTransport::Local),
                    system_handle.local_addr().clone(),
                    system_supervision_ref.clone(),
                    Duration::from_secs(30),
                    HashMap::new(),
                    ProcLifecycleMode::ManagedBySystem,
                )
            });
            futures::future::try_join_all(futs).await.unwrap()
        };
        // Wait for the worker procs being spawned.
        RealClock.sleep(Duration::from_secs(2)).await;

        // Create a world named foo, and directly join procs to it.
        let foo_proc_actors = {
            let foo_world_id = WorldId("foo_world".to_string());
            sys_actor_handle
                .upsert_world(
                    &client,
                    foo_world_id.clone(),
                    Shape::Definite(vec![2]),
                    2,
                    Environment::Local,
                    HashMap::new(),
                )
                .await
                .unwrap();
            // Join a non-worker proc to the "foo" world.
            let foo_futs = (0..2).map(|i| {
                let listen_addr = ChannelAddr::any(ChannelTransport::Local);
                let proc_id = ProcId(foo_world_id.clone(), i);
                ProcActor::try_bootstrap(
                    proc_id.clone(),
                    foo_world_id.clone(),
                    listen_addr,
                    system_handle.local_addr().clone(),
                    system_supervision_ref.clone(),
                    Duration::from_secs(30),
                    HashMap::new(),
                    ProcLifecycleMode::ManagedBySystem,
                )
            });
            futures::future::try_join_all(foo_futs).await.unwrap()
        };

        let (port, receiver) = client.open_once_port::<()>();
        // Kick off the shutdown.
        sys_actor_handle
            .stop(&client, None, Duration::from_secs(5), port.bind())
            .await
            .unwrap();
        receiver.recv().await.unwrap();
        RealClock.sleep(Duration::from_secs(5)).await;

        // // Verify all the host actors are stopped.
        for bootstrap in host_proc_actors {
            bootstrap.proc_actor.into_future().await;
        }

        // Verify all the foo actors are stopped.
        for bootstrap in foo_proc_actors {
            bootstrap.proc_actor.into_future().await;
        }
        // Verify the system actor is stopped.
        system_handle.actor_handle.into_future().await;

        // Since we do not have the worker actor handles, verify the worker procs
        // are stopped by checking the logs.
        for m in 0..(shape.iter().product()) {
            let proc_id = worker_world_id.proc_id(m);
            assert!(logs_contain(format!("{proc_id}: proc stopped",).as_str()));
        }
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_single_world_shutdown() {
        let system_handle = System::serve(
            ChannelAddr::any(ChannelTransport::Local),
            SystemActorParams::new(Duration::from_secs(10), Duration::from_secs(10)),
        )
        .await
        .unwrap();
        let system_supervision_ref: ActorRef<ProcSupervisor> =
            ActorRef::attest(SYSTEM_ACTOR_REF.actor_id().clone());

        let mut system = System::new(system_handle.local_addr().clone());
        let client = system.attach().await.unwrap();

        let sys_actor_handle = system_handle.system_actor_handle();

        let host_world_id = WorldId(("host_world").to_string());
        let worker_world_id = WorldId("worker_world".to_string());
        let foo_world_id = WorldId("foo_world".to_string());

        // Create a worker world from host procs.
        let shape = vec![2, 2, 4];
        let host_proc_actors = {
            // Upsert the worker world.
            sys_actor_handle
                .upsert_world(
                    &client,
                    worker_world_id.clone(),
                    // 2 worker procs in total, 8 per host.
                    Shape::Definite(shape.clone()),
                    8,
                    Environment::Local,
                    HashMap::new(),
                )
                .await
                .unwrap();

            // Bootstrap the host procs, which will lead to work procs being spawned.
            let futs = (0..2).map(|i| {
                let host_proc_id = ProcId(host_world_id.clone(), i);
                ProcActor::try_bootstrap(
                    host_proc_id.clone(),
                    host_world_id.clone(),
                    ChannelAddr::any(ChannelTransport::Local),
                    system_handle.local_addr().clone(),
                    system_supervision_ref.clone(),
                    Duration::from_secs(30),
                    HashMap::new(),
                    ProcLifecycleMode::ManagedBySystem,
                )
            });
            futures::future::try_join_all(futs).await.unwrap()
        };
        // Wait for the worker procs being spawned.
        RealClock.sleep(Duration::from_secs(2)).await;

        // Create a world named foo, and directly join procs to it.
        let foo_proc_actors = {
            sys_actor_handle
                .upsert_world(
                    &client,
                    foo_world_id.clone(),
                    Shape::Definite(vec![2]),
                    2,
                    Environment::Local,
                    HashMap::new(),
                )
                .await
                .unwrap();
            // Join a non-worker proc to the "foo" world.
            let foo_futs = (0..2).map(|i| {
                let listen_addr = ChannelAddr::any(ChannelTransport::Local);
                let proc_id = ProcId(foo_world_id.clone(), i);
                ProcActor::try_bootstrap(
                    proc_id.clone(),
                    foo_world_id.clone(),
                    listen_addr,
                    system_handle.local_addr().clone(),
                    system_supervision_ref.clone(),
                    Duration::from_secs(30),
                    HashMap::new(),
                    ProcLifecycleMode::ManagedBySystem,
                )
            });
            futures::future::try_join_all(foo_futs).await.unwrap()
        };

        {
            let snapshot = sys_actor_handle
                .snapshot(&client, SystemSnapshotFilter::all())
                .await
                .unwrap();
            let snapshot_world_ids: HashSet<WorldId> = snapshot.worlds.keys().cloned().collect();
            assert_eq!(
                snapshot_world_ids,
                hashset! {worker_world_id.clone(), foo_world_id.clone(), WorldId("_world".to_string())}
            );
        }

        let (port, receiver) = client.open_once_port::<()>();
        // Kick off the shutdown.
        sys_actor_handle
            .stop(
                &client,
                Some(vec![WorldId("foo_world".into())]),
                Duration::from_secs(5),
                port.bind(),
            )
            .await
            .unwrap();
        receiver.recv().await.unwrap();
        RealClock.sleep(Duration::from_secs(5)).await;

        // Verify all the foo actors are stopped.
        for bootstrap in foo_proc_actors {
            bootstrap.proc_actor.into_future().await;
        }

        // host actors should still be running.
        for bootstrap in host_proc_actors {
            match tokio::time::timeout(Duration::from_secs(5), bootstrap.proc_actor.into_future())
                .await
            {
                Ok(_) => {
                    panic!("foo actor shouldn't be stopped");
                }
                Err(_) => {}
            }
        }

        // Verify the system actor not stopped.
        match tokio::time::timeout(
            Duration::from_secs(3),
            system_handle.actor_handle.clone().into_future(),
        )
        .await
        {
            Ok(_) => {
                panic!("system actor shouldn't be stopped");
            }
            Err(_) => {}
        }

        {
            let snapshot = sys_actor_handle
                .snapshot(&client, SystemSnapshotFilter::all())
                .await
                .unwrap();
            let snapshot_world_ids: HashSet<WorldId> = snapshot.worlds.keys().cloned().collect();
            // foo_world_id is no longer in the snapshot.
            assert_eq!(
                snapshot_world_ids,
                hashset! {worker_world_id, WorldId("_world".to_string())}
            );
        }
    }

    // Test our understanding of when & where channel addresses are
    // dialed.
    #[tracing_test::traced_test]
    #[tokio::test]
    async fn test_channel_dial_count() {
        let system_handle = System::serve(
            ChannelAddr::any(ChannelTransport::Tcp),
            SystemActorParams::new(Duration::from_secs(10), Duration::from_secs(10)),
        )
        .await
        .unwrap();

        let system_addr = system_handle.local_addr();
        let mut system = System::new(system_addr.clone());
        // `system.attach()` calls `system.send()` which
        // `channel::dial()`s the system address for a `MailboxClient`
        // for the `EnvelopingMailboxSender` to be the forwarding
        // sender for `client1`s proc (+1 dial).
        //
        // The forwarding sender will be used to send a join message
        // to the system actor that uses the `NetTx` just dialed so no
        // new `channel::dial()` for that (+0 dial). However, the
        // system actor will respond to the join message by using the
        // proc address (given in the join message) for the new proc
        // when it sends from its `DialMailboxRouter` so we expect to
        // see a `channel::dial()` there (+1 dial).
        let client1 = system.attach().await.unwrap();

        // `system.attach()` calls `system.send()` which
        // `channel::dial()`s the system address for a `MailboxClient`
        // for the `EnvelopingMailboxSender` to be the forwarding
        // sender for `client2`s proc (+1 dial).
        //
        // The forwarding sender will be used to send a join message
        // to the system actor that uses the `NetTx` just dialed so no
        // new `channel::dial()` for that (+0 dial). However, the
        // system actor will respond to the join message by using the
        // proc address (given in the join message) for the new proc
        // when it sends from its `DialMailboxRouter` so we expect to
        // see a `channel::dial()` there (+1 dial).
        let client2 = system.attach().await.unwrap();

        // Send a message to `client2` from `client1`. This will
        // involve forwarding to the system actor using `client1`'s
        // proc's forwarder already dialied `NetTx` (+0 dial). The
        // system actor will relay to `client2`'s proc. The `NetTx` to
        // that proc was cached in the system actor's
        // `DialmailboxRouter` when responding to `client2`'s join (+0
        // dial).
        let (port, mut port_rx) = client2.open_port();
        port.bind().send(&client1, 123u64).unwrap();
        assert_eq!(port_rx.recv().await.unwrap(), 123u64);

        // In summary we expect to see 4 dials.
        logs_assert(|logs| {
            let dial_count = logs
                .iter()
                .filter(|log| log.contains("dialing channel tcp"))
                .count();
            if dial_count == 4 {
                Ok(())
            } else {
                Err(format!("unexpected tcp channel dial count: {}", dial_count))
            }
        });

        system_handle.stop().await.unwrap();
        system_handle.await;
    }

    #[derive(Debug, Named, Serialize, Deserialize, PartialEq)]
    #[named(dump = false)]
    enum TestMessage {
        Forward(String),
    }

    // TODO(pzhang) add macro to auto implement these traits.
    impl Unbind for TestMessage {
        fn unbind(self) -> anyhow::Result<Unbound<Self>> {
            Ok(Unbound::new(self, Bindings::default()))
        }
    }

    impl Bind for TestMessage {
        fn bind(self, _bindings: &Bindings) -> anyhow::Result<Self> {
            Ok(self)
        }
    }

    #[derive(Debug)]
    #[hyperactor::export_spawn(TestMessage, IndexedErasedUnbound<TestMessage>)]
    struct TestActor {
        // Forward the received message to this port, so it can be inspected by
        // the unit test.
        forward_port: PortRef<TestMessage>,
    }

    #[derive(Debug, Clone, Named, Serialize, Deserialize)]
    struct TestActorParams {
        forward_port: PortRef<TestMessage>,
    }

    #[async_trait]
    impl Actor for TestActor {
        type Params = TestActorParams;

        async fn new(params: Self::Params) -> Result<Self> {
            let Self::Params { forward_port } = params;
            Ok(Self { forward_port })
        }
    }

    #[async_trait]
    impl Handler<TestMessage> for TestActor {
        async fn handle(&mut self, this: &Instance<Self>, msg: TestMessage) -> anyhow::Result<()> {
            self.forward_port.send(this, msg)?;
            Ok(())
        }
    }

    async fn spawn_comm_actors(num: usize) -> Result<Vec<(Proc, ActorHandle<CommActor>)>> {
        let mut comms = vec![];
        let router = MailboxRouter::new();
        for idx in 0..num {
            let proc_id = ProcId(id!(local), idx);
            let proc = Proc::new(proc_id, BoxedMailboxSender::new(router.clone()));
            router.bind(proc.proc_id().clone().into(), proc.clone());
            ProcSupervisionCoordinator::set(&proc).await?;

            let comm = proc.spawn::<CommActor>("comm", CommActorParams {}).await?;
            comm.bind::<CommActor>();
            comms.push((proc, comm));
        }

        Ok(comms)
    }

    #[tokio::test]
    async fn test_comm_actor_cast() -> Result<()> {
        set_tracing_env_filter(Level::INFO);
        let comm_actors = spawn_comm_actors(2).await?;

        let world = id!(foo);

        let mut queues = vec![];
        let mut test_actors = vec![];
        for (proc, _) in comm_actors.iter() {
            let client = proc.attach("user")?;
            let (tx, rx) = open_port(&client);
            queues.push(rx);
            let test = proc
                .spawn::<TestActor>(
                    "actor",
                    TestActorParams {
                        forward_port: tx.bind(),
                    },
                )
                .await?;
            test.bind::<TestActor>();
            test_actors.push(test);
        }

        for (_, comm) in &comm_actors {
            comm.send(CastMessage {
                dest: Uslice {
                    slice: Slice::new(0, vec![2], vec![1])?,
                    selection: selection::dsl::true_(),
                },
                message: CastMessageEnvelope::new(
                    ActorId(world.random_user_proc(), "user".into(), 0),
                    DestinationPort::new::<TestActor, TestMessage>(GangId(
                        comm.actor_id().proc_id().world_id().clone(),
                        "actor".into(),
                    )),
                    TestMessage::Forward("abc".to_string()),
                )?,
            })?;
        }

        for mut queue in queues.into_iter() {
            let msg = queue.recv().await.context("missing")?;
            assert_eq!(msg, TestMessage::Forward("abc".to_string()));
            let msg = queue.recv().await.context("missing")?;
            assert_eq!(msg, TestMessage::Forward("abc".to_string()));
        }

        for ((mut proc, handle), test) in comm_actors.into_iter().zip(test_actors.into_iter()) {
            handle.drain_and_stop()?;
            assert_eq!(handle.await, ActorStatus::Stopped);
            test.drain_and_stop()?;
            assert_eq!(test.await, ActorStatus::Stopped);
            proc.destroy_and_wait(Duration::from_secs(10), None).await?;
        }
        Ok(())
    }

    #[async_timed_test(timeout_secs = 30)]
    async fn test_comm_actor_cast_system() -> Result<()> {
        let system_handle = System::serve(
            ChannelAddr::any(ChannelTransport::Local),
            SystemActorParams::new(Duration::from_secs(10), Duration::from_secs(10)),
        )
        .await?;
        let system_supervision_ref: ActorRef<ProcSupervisor> =
            ActorRef::attest(SYSTEM_ACTOR_REF.actor_id().clone());
        let mut system = System::new(system_handle.local_addr().clone());
        let client = system.attach().await?;
        let sys_actor_handle = system_handle.system_actor_handle();
        let world_id = id!(world);

        // Create world.
        let shape = vec![4, 4, 4];
        let host_proc_actors = {
            // Upsert the worker world.
            sys_actor_handle
                .upsert_world(
                    &client,
                    world_id.clone(),
                    // 64 worker procs in total, 1 per host.
                    Shape::Definite(shape.clone()),
                    1,
                    Environment::Local,
                    HashMap::new(),
                )
                .await
                .unwrap();

            // Bootstrap the host procs, which will lead to work procs being spawned.
            let futs = (0..64).map(|i| {
                let proc_id = ProcId(world_id.clone(), i);
                ProcActor::try_bootstrap(
                    proc_id.clone(),
                    world_id.clone(),
                    ChannelAddr::any(ChannelTransport::Local),
                    system_handle.local_addr().clone(),
                    system_supervision_ref.clone(),
                    Duration::from_secs(30),
                    HashMap::new(),
                    ProcLifecycleMode::ManagedBySystem,
                )
            });
            futures::future::try_join_all(futs).await.unwrap()
        };

        // Spawn comm and test actors.
        let (spawned_test_tx, mut spawned_test) = open_port(&client);
        let world_size = host_proc_actors.len();
        let mut queues = vec![];
        for bootstrap in host_proc_actors.iter() {
            let (tx, rx) = open_port(&client);
            queues.push(rx);
            bootstrap
                .proc_actor
                .spawn(
                    &client,
                    // Use explicit actor type to avoid the WorkActor dependency.
                    "hyperactor_multiprocess::system::tests::TestActor".to_owned(),
                    "actor".into(),
                    bincode::serialize(&TestActorParams {
                        forward_port: tx.bind(),
                    })?,
                    spawned_test_tx.bind(),
                )
                .await?;
        }
        for _ in 0..world_size {
            spawned_test.recv().await?;
        }

        // Send cast messages to everyeone.
        for bootstrap in &host_proc_actors {
            let comm = bootstrap.comm_actor.bind::<CommActor>();
            comm.send(
                &client,
                CastMessage {
                    // Destination is every node in the world.
                    dest: Uslice {
                        slice: Slice::new(0, vec![4, 4, 4], vec![16, 4, 1])?,
                        selection: selection::dsl::true_(),
                    },
                    message: CastMessageEnvelope::new(
                        ActorId(world_id.random_user_proc(), "user".into(), 0),
                        DestinationPort::new::<TestActor, TestMessage>(GangId(
                            comm.actor_id().proc_id().world_id().clone(),
                            "actor".into(),
                        )),
                        TestMessage::Forward("abc".to_string()),
                    )?,
                },
            )?;
        }

        // Check that test actors received the forwarded messages.
        for mut queue in queues.into_iter() {
            for _ in 0..world_size {
                let msg = queue.recv().await.context("missing")?;
                assert_eq!(msg, TestMessage::Forward("abc".to_string()));
            }
        }
        Ok(())
    }
}
