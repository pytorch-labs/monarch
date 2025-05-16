use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use hyperactor::Named;
use hyperactor::WorldId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::Rx;
use hyperactor::channel::Tx;
use hyperactor::channel::TxStatus;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use hyperactor::mailbox::DialMailboxRouter;
use hyperactor::mailbox::MailboxServer;
use hyperactor::mailbox::monitored_return_handle;
use hyperactor::reference::Reference;
use hyperactor::serde_json;
use ndslice::Shape;
use serde::Deserialize;
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::WatchStream;
use tokio_util::sync::CancellationToken;

use crate::alloc::Alloc;
use crate::alloc::AllocSpec;
use crate::alloc::Allocator;
use crate::alloc::ProcState;
use crate::alloc::ProcessAllocator;

/// Control messages sent from remote allocator to local allocator.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub enum RemoteProcessAllocatorMessage {
    /// Create allocation with given spec and send updates to bootstrap_addr.
    Allocate {
        /// Allocation spec. Shape is a slice of the original shape with
        /// correct rank offset.
        spec: AllocSpec,
        /// Bootstrap address to be used for sending updates.
        bootstrap_addr: ChannelAddr,
        /// Ordered list of hosts in this allocation. Can be used to
        /// pre-populate the any local configurations such as torch.dist.
        hosts: Vec<String>,
        /// How often to send heartbeat messages to check if client is alive.
        heartbeat_interval: Duration,
    },

    /// Stop allocation.
    Stop,
}

/// Control message sent from local allocator to remote allocator
/// relaying process state updates.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub enum RemoteProcessProcStateMessage {
    /// Allocation successful and Update, Done messages will follow.
    Allocated { world_id: WorldId, shape: Shape },
    /// ProcState updates.
    Update(ProcState),
    /// Underlying Alloc is done.
    Done(WorldId),
    /// Heartbeat message to check if client is alive.
    HeartBeat,
}

/// Allocator with a service frontend that wraps ProcessAllocator.
pub struct RemoteProcessAllocator {
    cancel_token: CancellationToken,
}

impl RemoteProcessAllocator {
    /// Create a new allocator. It will not start until start() is called.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            cancel_token: CancellationToken::new(),
        })
    }

    /// Stop the allocator. This will stop any ongoing allocations.
    pub fn terminate(&self) {
        self.cancel_token.cancel();
    }

    /// Start a remote process allocator with given cmd listening for
    /// RemoteProcessAllocatorMessage on serve_addr. Call will block until cancelled.
    /// The implementation is simple such that it can only handle one Alloc at
    /// a time. Generally that's the most common use-case.
    /// Flow works as follows:
    /// 1. Client sends Allocate message to serve_addr.
    /// 2. Allocator connects to bootstrap_addr, creates Alloc and sends Allocated message.
    /// 3. Allocator streams one or more Update messages to bootstrap_addr as Alloc progresses
    ///    making the following changes:
    ///    * Remap mesh_agent listen address to our own forwarder actor address.
    /// 4. Allocator sends Done message to bootstrap_addr when Alloc is done.
    ///
    /// At any point, client can send Stop message to serve_addr to stop the allocator.
    pub async fn start(&self, cmd: Command, serve_addr: ChannelAddr) -> Result<(), anyhow::Error> {
        let process_allocator = ProcessAllocator::new(cmd);
        self.start_with_allocator(serve_addr, process_allocator)
            .await
    }

    /// Start a remote process allocator with given allocator listening for
    /// RemoteProcessAllocatorMessage on serve_addr.
    /// Used for testing.
    pub async fn start_with_allocator<A: Allocator + Send + Sync + 'static>(
        &self,
        serve_addr: ChannelAddr,
        mut process_allocator: A,
    ) -> Result<(), anyhow::Error>
    where
        <A as Allocator>::Alloc: Send,
        <A as Allocator>::Alloc: Sync,
    {
        tracing::info!("starting remote allocator on: {}", serve_addr);
        let (_, mut rx) = channel::serve(serve_addr)
            .await
            .map_err(anyhow::Error::from)?;

        struct ActiveAllocation {
            handle: JoinHandle<()>,
            cancel_token: CancellationToken,
        }
        async fn ensure_previous_alloc_stopped(active_allocation: &mut Option<ActiveAllocation>) {
            if let Some(active_allocation) = active_allocation.take() {
                tracing::info!("previous alloc found, stopping");
                active_allocation.cancel_token.cancel();
                // should be ok to wait even if original caller has gone since heartbeat
                // will eventually timeout and exit the loop.
                if let Err(e) = active_allocation.handle.await {
                    tracing::error!("allocation handler failed: {}", e);
                }
            }
        }

        let mut active_allocation: Option<ActiveAllocation> = None;
        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Ok(RemoteProcessAllocatorMessage::Allocate {
                            spec,
                            bootstrap_addr,
                            hosts,
                            heartbeat_interval,
                        }) => {
                            tracing::info!("received allocation request: {:?}", spec);

                            ensure_previous_alloc_stopped(&mut active_allocation).await;

                            match process_allocator.allocate(spec.clone()).await {
                                Ok(alloc) => {
                                    let cancel_token = CancellationToken::new();
                                    active_allocation = Some(ActiveAllocation {
                                        cancel_token: cancel_token.clone(),
                                        handle: tokio::spawn(Self::handle_allocation_request(
                                        Box::new(alloc) as Box<dyn Alloc + Send + Sync>,
                                        bootstrap_addr,
                                        hosts,
                                        heartbeat_interval,
                                        cancel_token,
                                    )),
                                })
                                }
                                Err(e) => {
                                    tracing::error!("allocation for {:?} failed: {}", spec, e);
                                    continue;
                                }
                            }
                        }
                        Ok(RemoteProcessAllocatorMessage::Stop) => {
                            tracing::info!("received stop request");

                            ensure_previous_alloc_stopped(&mut active_allocation).await;
                        }
                        Err(e) => {
                            tracing::error!("upstream channel error: {}", e);
                            continue;
                        }
                    }
                }
                _ = self.cancel_token.cancelled() => {
                    tracing::info!("main loop cancelled");

                    ensure_previous_alloc_stopped(&mut active_allocation).await;

                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_allocation_request(
        alloc: Box<dyn Alloc + Send + Sync>,
        bootstrap_addr: ChannelAddr,
        hosts: Vec<String>,
        heartbeat_interval: Duration,
        cancel_token: CancellationToken,
    ) {
        // start proc message forwarder
        let router = DialMailboxRouter::new();
        let (forwarder_addr, forwarder_rx) =
            match channel::serve(ChannelAddr::any(bootstrap_addr.transport())).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!("failed to to bootstrap forwarder actor: {}", e);
                    return;
                }
            };
        let mailbox_handle = router
            .clone()
            .serve(forwarder_rx, monitored_return_handle());
        tracing::info!("started forwarder on: {}", forwarder_addr);

        // Check if we need to write TORCH_ELASTIC_CUSTOM_HOSTNAMES_LIST_FILE
        // See: https://github.com/fairinternal/xlformers/blob/llama4_monarch/tools/launching/torchx/entrypoint/generate_ranks.py
        if let Ok(hosts_file) = std::env::var("TORCH_ELASTIC_CUSTOM_HOSTNAMES_LIST_FILE") {
            tracing::info!("writing hosts to {}", hosts_file);
            #[derive(Serialize)]
            struct Hosts {
                hostnames: Vec<String>,
            }
            match serde_json::to_string(&Hosts { hostnames: hosts }) {
                Ok(json) => match tokio::fs::File::create(&hosts_file).await {
                    Ok(mut file) => {
                        if file.write_all(json.as_bytes()).await.is_err() {
                            tracing::error!("failed to write hosts to {}", hosts_file);
                            return;
                        }
                    }
                    Err(e) => {
                        tracing::error!("failed to open hosts file {}: {}", hosts_file, e);
                        return;
                    }
                },
                Err(e) => {
                    tracing::error!("failed to serialize hosts: {}", e);
                    return;
                }
            }
        }

        Self::handle_allocation_loop(
            alloc,
            bootstrap_addr,
            router,
            forwarder_addr,
            heartbeat_interval,
            cancel_token,
        )
        .await;

        mailbox_handle.stop();
        if let Err(e) = mailbox_handle.await {
            tracing::error!("failed to join forwarder: {}", e);
        }
    }

    async fn handle_allocation_loop(
        mut alloc: Box<dyn Alloc + Send + Sync>,
        bootstrap_addr: ChannelAddr,
        router: DialMailboxRouter,
        forward_addr: ChannelAddr,
        heartbeat_interval: Duration,
        cancel_token: CancellationToken,
    ) {
        let tx = match channel::dial(bootstrap_addr) {
            Ok(tx) => tx,
            Err(err) => {
                tracing::error!("failed to dial bootstrap address: {}", err);
                return;
            }
        };
        if let Err(e) = tx
            .send(RemoteProcessProcStateMessage::Allocated {
                world_id: alloc.world_id().clone(),
                shape: alloc.shape().clone(),
            })
            .await
        {
            tracing::error!("failed to send Allocated message: {}", e);
            return;
        }

        let mut mesh_agents_by_proc_id = HashMap::new();
        let mut running = true;
        let tx_status = tx.status().clone();
        let mut tx_watcher = WatchStream::new(tx_status);
        loop {
            tokio::select! {
                _ = cancel_token.cancelled(), if running => {
                    tracing::info!("cancelled, stopping allocation");
                    running = false;
                    if let Err(e) = alloc.stop().await {
                        tracing::error!("stop failed: {}", e);
                        break;
                    }
                }
                status = tx_watcher.next(), if running => {
                    match status  {
                        Some(TxStatus::Closed) => {
                            tracing::error!("upstream channel state closed");
                            break;
                        },
                        _ => {
                            tracing::debug!("got channel event: {:?}", status.unwrap());
                            continue;
                        }
                    }
                }
                e = alloc.next() => {
                    match e {
                        Some(event) => {
                            tracing::debug!("got event: {:?}", event);
                            let event = match event {
                                ProcState::Created { .. } => event,
                                ProcState::Running { proc_id, mesh_agent, addr } => {
                                    tracing::debug!("remapping mesh_agent {}: addr {} -> {}", mesh_agent, addr, forward_addr);
                                    mesh_agents_by_proc_id.insert(proc_id.clone(), mesh_agent.clone());
                                    router.bind(mesh_agent.actor_id().proc_id().clone().into(), addr);
                                    ProcState::Running { proc_id, mesh_agent, addr: forward_addr.clone() }
                                },
                                  ProcState::Stopped { proc_id, reason } => {
                                    match mesh_agents_by_proc_id.remove(&proc_id) {
                                        Some(mesh_agent) => {
                                            tracing::debug!("unmapping mesh_agent {}", mesh_agent);
                                            let agent_ref: Reference = mesh_agent.actor_id().proc_id().clone().into();
                                            router.unbind(&agent_ref);
                                        },
                                        None => {
                                            tracing::warn!("mesh_agent not found for proc_id: {}", proc_id);
                                        }
                                    }
                                    ProcState::Stopped { proc_id, reason }
                                },
                            };
                            tracing::debug!("sending event: {:?}", event);
                            tx.post(RemoteProcessProcStateMessage::Update(event));
                        }
                        None => {
                            tracing::debug!("sending done");
                            tx.post(RemoteProcessProcStateMessage::Done(alloc.world_id().clone()));
                            running = false;
                            break;
                        }
                    }
                }
                _ = RealClock.sleep(heartbeat_interval) => {
                    tracing::trace!("sending heartbeat");
                    tx.post(RemoteProcessProcStateMessage::HeartBeat);
                }
            }
        }
        tracing::debug!("allocation handler loop exited");
        if running {
            tracing::info!("stopping processes");
            if let Err(e) = alloc.stop_and_wait().await {
                tracing::error!("stop failed: {}", e);
                return;
            }
            tracing::info!("stop finished");
        }
    }
}

#[cfg(test)]
mod test {
    use std::assert_matches::assert_matches;

    use hyperactor::ActorRef;
    use hyperactor::channel::ChannelRx;
    use hyperactor::id;
    use ndslice::shape;
    use tokio::sync::oneshot;

    use super::*;
    use crate::alloc::AllocConstraints;
    use crate::alloc::ChannelTransport;
    use crate::alloc::MockAlloc;
    use crate::alloc::MockAllocWrapper;
    use crate::alloc::MockAllocator;
    use crate::alloc::ProcStopReason;
    use crate::proc_mesh::mesh_agent::MeshAgent;

    async fn read_all_created(rx: &mut ChannelRx<RemoteProcessProcStateMessage>, alloc_len: usize) {
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Created { .. }) => i += 1,
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
    }

    async fn read_all_running(rx: &mut ChannelRx<RemoteProcessProcStateMessage>, alloc_len: usize) {
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Running { .. }) => i += 1,
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
    }

    async fn read_all_stopped(rx: &mut ChannelRx<RemoteProcessProcStateMessage>, alloc_len: usize) {
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Stopped { .. }) => i += 1,
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
    }

    fn set_procstate_execptations(alloc: &mut MockAlloc, shape: Shape) {
        let alloc_len = shape.slice().len();
        alloc.expect_shape().return_const(shape.clone());
        for i in 0..alloc_len {
            let proc_id = format!("test[{}]", i).parse().unwrap();
            let coords = shape.slice().coordinates(i).unwrap();
            alloc
                .expect_next()
                .times(1)
                .return_once(|| Some(ProcState::Created { proc_id, coords }));
        }
        for i in 0..alloc_len {
            let proc_id = format!("test[{}]", i).parse().unwrap();
            let mesh_agent = ActorRef::<MeshAgent>::attest(
                format!("test[{}].mesh_agent[{}]", i, i).parse().unwrap(),
            );
            alloc.expect_next().times(1).return_once(|| {
                Some(ProcState::Running {
                    proc_id,
                    addr: ChannelAddr::Unix("/proc0".parse().unwrap()),
                    mesh_agent,
                })
            });
        }
        for i in 0..alloc_len {
            let proc_id = format!("test[{}]", i).parse().unwrap();
            alloc.expect_next().times(1).return_once(|| {
                Some(ProcState::Stopped {
                    proc_id,
                    reason: ProcStopReason::Unknown,
                })
            });
        }
    }

    #[timed_test::async_timed_test(timeout_secs = 5)]
    async fn test_simple() {
        hyperactor_telemetry::initialize_logging();
        let serve_addr = ChannelAddr::any(ChannelTransport::Unix);
        let bootstrap_addr = ChannelAddr::any(ChannelTransport::Unix);
        let (_, mut rx) = channel::serve(bootstrap_addr.clone()).await.unwrap();

        let spec = AllocSpec {
            shape: shape!(host = 1, gpu = 2),
            constraints: AllocConstraints::none(),
        };
        let tx = channel::dial(serve_addr.clone()).unwrap();

        let alloc_len = spec.shape.slice().len();

        let world_id: WorldId = id!(test_world_id);
        let mut alloc = MockAlloc::new();
        alloc.expect_world_id().return_const(world_id.clone());
        alloc.expect_shape().return_const(spec.shape.clone());

        set_procstate_execptations(&mut alloc, spec.shape.clone());

        // final none
        alloc.expect_next().return_const(None);

        let mut allocator = MockAllocator::new();
        let total_messages = alloc_len * 3 + 1;
        let mock_wrapper = MockAllocWrapper::new_block_next(
            alloc,
            // block after create, running, stopped and done.
            total_messages,
        );
        allocator
            .expect_allocate()
            .times(1)
            .return_once(move |_| Ok(mock_wrapper));

        let remote_allocator = RemoteProcessAllocator::new();
        let handle = tokio::spawn({
            let remote_allocator = remote_allocator.clone();
            async move {
                remote_allocator
                    .start_with_allocator(serve_addr, allocator)
                    .await
            }
        });

        tx.send(RemoteProcessAllocatorMessage::Allocate {
            spec: spec.clone(),
            bootstrap_addr,
            hosts: vec![],
            heartbeat_interval: Duration::from_secs(1),
        })
        .await
        .unwrap();

        // Allocated
        let m = rx.recv().await.unwrap();
        assert!(
            matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape)
        );

        // All Created events
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Created { proc_id, coords }) => {
                    let expected_proc_id = format!("test[{}]", i).parse().unwrap();
                    let expected_coords = spec.shape.slice().coordinates(i).unwrap();
                    assert_eq!(proc_id, expected_proc_id);
                    assert_eq!(coords, expected_coords);
                    i += 1;
                }
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
        // All Running events
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Running {
                    proc_id,
                    mesh_agent,
                    addr: _,
                }) => {
                    let expected_proc_id = format!("test[{}]", i).parse().unwrap();
                    let expected_mesh_agent = ActorRef::<MeshAgent>::attest(
                        format!("test[{}].mesh_agent[{}]", i, i).parse().unwrap(),
                    );
                    assert_eq!(proc_id, expected_proc_id);
                    assert_eq!(mesh_agent, expected_mesh_agent);
                    i += 1;
                }
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
        // All Stopped events
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Stopped {
                    proc_id,
                    reason: ProcStopReason::Unknown,
                }) => {
                    let expected_proc_id = format!("test[{}]", i).parse().unwrap();
                    assert_eq!(proc_id, expected_proc_id);
                    i += 1;
                }
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
        // Done
        loop {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Done(id) => {
                    assert_eq!(id, world_id);
                    break;
                }
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }

        remote_allocator.terminate();
        handle.await.unwrap().unwrap();
    }

    #[timed_test::async_timed_test(timeout_secs = 15)]
    async fn test_normal_stop() {
        hyperactor_telemetry::initialize_logging();
        let serve_addr = ChannelAddr::any(ChannelTransport::Unix);
        let bootstrap_addr = ChannelAddr::any(ChannelTransport::Unix);
        let (_, mut rx) = channel::serve(bootstrap_addr.clone()).await.unwrap();

        let spec = AllocSpec {
            shape: shape!(host = 1, gpu = 2),
            constraints: AllocConstraints::none(),
        };
        let tx = channel::dial(serve_addr.clone()).unwrap();

        let alloc_len = spec.shape.slice().len();

        let world_id: WorldId = id!(test_world_id);
        let mut alloc = MockAllocWrapper::new_block_next(
            MockAlloc::new(),
            // block after all created, all running
            alloc_len * 2,
        );
        let next_tx = alloc.notify_tx();
        alloc.alloc.expect_world_id().return_const(world_id.clone());
        alloc.alloc.expect_shape().return_const(spec.shape.clone());

        set_procstate_execptations(&mut alloc.alloc, spec.shape.clone());

        alloc.alloc.expect_next().return_const(None);
        alloc.alloc.expect_stop().times(1).return_once(|| Ok(()));

        let mut allocator = MockAllocator::new();
        allocator
            .expect_allocate()
            .times(1)
            .return_once(|_| Ok(alloc));

        let remote_allocator = RemoteProcessAllocator::new();
        let handle = tokio::spawn({
            let remote_allocator = remote_allocator.clone();
            async move {
                remote_allocator
                    .start_with_allocator(serve_addr, allocator)
                    .await
            }
        });

        tx.send(RemoteProcessAllocatorMessage::Allocate {
            spec: spec.clone(),
            bootstrap_addr,
            hosts: vec![],
            heartbeat_interval: Duration::from_millis(200),
        })
        .await
        .unwrap();

        // Allocated
        let m = rx.recv().await.unwrap();
        assert_matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape);

        read_all_created(&mut rx, alloc_len).await;
        read_all_running(&mut rx, alloc_len).await;

        // allocation finished. now we stop it.
        tracing::info!("stopping allocation");
        tx.send(RemoteProcessAllocatorMessage::Stop).await.unwrap();
        // receive all stops
        next_tx.send(()).unwrap();

        read_all_stopped(&mut rx, alloc_len).await;

        remote_allocator.terminate();
        handle.await.unwrap().unwrap();
    }

    #[timed_test::async_timed_test(timeout_secs = 15)]
    async fn test_realloc() {
        hyperactor_telemetry::initialize_logging();
        let serve_addr = ChannelAddr::any(ChannelTransport::Unix);
        let bootstrap_addr = ChannelAddr::any(ChannelTransport::Unix);
        let (_, mut rx) = channel::serve(bootstrap_addr.clone()).await.unwrap();

        let spec = AllocSpec {
            shape: shape!(host = 1, gpu = 2),
            constraints: AllocConstraints::none(),
        };
        let tx = channel::dial(serve_addr.clone()).unwrap();

        let alloc_len = spec.shape.slice().len();

        let world_id: WorldId = id!(test_world_id);
        let mut alloc1 = MockAllocWrapper::new_block_next(
            MockAlloc::new(),
            // block after all created, all running
            alloc_len * 2,
        );
        let next_tx1 = alloc1.notify_tx();
        alloc1
            .alloc
            .expect_world_id()
            .return_const(world_id.clone());
        alloc1.alloc.expect_shape().return_const(spec.shape.clone());

        set_procstate_execptations(&mut alloc1.alloc, spec.shape.clone());
        alloc1.alloc.expect_next().return_const(None);
        alloc1.alloc.expect_stop().times(1).return_once(|| Ok(()));
        // second allocation
        let mut alloc2 = MockAllocWrapper::new_block_next(
            MockAlloc::new(),
            // block after all created, all running
            alloc_len * 2,
        );
        let next_tx2 = alloc2.notify_tx();
        alloc2
            .alloc
            .expect_world_id()
            .return_const(world_id.clone());
        alloc2.alloc.expect_shape().return_const(spec.shape.clone());
        set_procstate_execptations(&mut alloc2.alloc, spec.shape.clone());
        alloc2.alloc.expect_next().return_const(None);
        alloc2.alloc.expect_stop().times(1).return_once(|| Ok(()));

        let mut allocator = MockAllocator::new();
        allocator
            .expect_allocate()
            .times(1)
            .return_once(|_| Ok(alloc1));
        // second alloc
        allocator
            .expect_allocate()
            .times(1)
            .return_once(|_| Ok(alloc2));

        let remote_allocator = RemoteProcessAllocator::new();
        let handle = tokio::spawn({
            let remote_allocator = remote_allocator.clone();
            async move {
                remote_allocator
                    .start_with_allocator(serve_addr, allocator)
                    .await
            }
        });

        tx.send(RemoteProcessAllocatorMessage::Allocate {
            spec: spec.clone(),
            bootstrap_addr: bootstrap_addr.clone(),
            hosts: vec![],
            heartbeat_interval: Duration::from_millis(200),
        })
        .await
        .unwrap();

        // Allocated
        let m = rx.recv().await.unwrap();
        assert_matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape);

        read_all_created(&mut rx, alloc_len).await;
        read_all_running(&mut rx, alloc_len).await;

        // allocation finished now we request a new one
        tx.send(RemoteProcessAllocatorMessage::Allocate {
            spec: spec.clone(),
            bootstrap_addr,
            hosts: vec![],
            heartbeat_interval: Duration::from_millis(200),
        })
        .await
        .unwrap();
        // unblock next for the first allocation
        next_tx1.send(()).unwrap();
        // we expect a stop(), then Stopped proc states, then a new Allocated
        read_all_stopped(&mut rx, alloc_len).await;
        let m = rx.recv().await.unwrap();
        assert_matches!(m, RemoteProcessProcStateMessage::Done(_));
        let m = rx.recv().await.unwrap();
        assert_matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape);
        // ProcStates for the new allocation
        read_all_created(&mut rx, alloc_len).await;
        read_all_running(&mut rx, alloc_len).await;
        // finally stop
        tracing::info!("stopping allocation");
        tx.send(RemoteProcessAllocatorMessage::Stop).await.unwrap();
        // receive all stops
        next_tx2.send(()).unwrap();

        read_all_stopped(&mut rx, alloc_len).await;

        remote_allocator.terminate();
        handle.await.unwrap().unwrap();
    }

    #[timed_test::async_timed_test(timeout_secs = 15)]
    async fn test_upstream_closed() {
        std::env::set_var("MONARCH_MESSAGE_DELIVERY_TIMEOUT_SECS", "1");

        hyperactor_telemetry::initialize_logging();
        let serve_addr = ChannelAddr::any(ChannelTransport::Unix);
        let bootstrap_addr = ChannelAddr::any(ChannelTransport::Unix);
        let (_, mut rx) = channel::serve(bootstrap_addr.clone()).await.unwrap();

        let spec = AllocSpec {
            shape: shape!(host = 1, gpu = 2),
            constraints: AllocConstraints::none(),
        };
        let tx = channel::dial(serve_addr.clone()).unwrap();

        let alloc_len = spec.shape.slice().len();

        let world_id: WorldId = id!(test_world_id);
        let mut alloc = MockAllocWrapper::new_block_next(
            MockAlloc::new(),
            // block after all created, all running
            alloc_len * 2,
        );
        let next_tx = alloc.notify_tx();
        alloc.alloc.expect_world_id().return_const(world_id.clone());
        alloc.alloc.expect_shape().return_const(spec.shape.clone());

        set_procstate_execptations(&mut alloc.alloc, spec.shape.clone());

        alloc.alloc.expect_next().return_const(None);
        // we expect a stop due to the failure
        // synchronize test with the stop
        let (stop_tx, stop_rx) = oneshot::channel();
        alloc.alloc.expect_stop().times(1).return_once(|| {
            stop_tx.send(()).unwrap();
            Ok(())
        });

        let mut allocator = MockAllocator::new();
        allocator
            .expect_allocate()
            .times(1)
            .return_once(|_| Ok(alloc));

        let remote_allocator = RemoteProcessAllocator::new();
        let handle = tokio::spawn({
            let remote_allocator = remote_allocator.clone();
            async move {
                remote_allocator
                    .start_with_allocator(serve_addr, allocator)
                    .await
            }
        });

        tx.send(RemoteProcessAllocatorMessage::Allocate {
            spec: spec.clone(),
            bootstrap_addr,
            hosts: vec![],
            heartbeat_interval: Duration::from_millis(200),
        })
        .await
        .unwrap();

        // Allocated
        let m = rx.recv().await.unwrap();
        assert_matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape);

        read_all_created(&mut rx, alloc_len).await;
        read_all_running(&mut rx, alloc_len).await;

        // allocation finished. terminate connection.
        tracing::info!("closing upstream");
        drop(rx);
        // wait for the heartbeat to expire
        #[allow(clippy::disallowed_methods)]
        tokio::time::sleep(Duration::from_secs(2)).await;
        // wait for the stop to be called
        stop_rx.await.unwrap();
        // unblock next
        next_tx.send(()).unwrap();
        remote_allocator.terminate();
        handle.await.unwrap().unwrap();
    }
}
