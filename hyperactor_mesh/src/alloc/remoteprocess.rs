use std::sync::Arc;

use hyperactor::Named;
use hyperactor::WorldId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelRx;
use hyperactor::channel::Rx;
use hyperactor::channel::Tx;
use hyperactor::serde_json;
use ndslice::Shape;
use serde::Deserialize;
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::alloc::Alloc;
use crate::alloc::AllocSpec;
use crate::alloc::Allocator;
use crate::alloc::ProcState;
use crate::alloc::ProcessAllocator;

/// Control messages sent from from remote allocator to local allocator.
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
    /// 3. Allocator streams one or more Update messages to bootstrap_addr as Alloc progresses.
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
    async fn start_with_allocator<A: Allocator + Send + Sync + 'static>(
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

        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Ok(RemoteProcessAllocatorMessage::Allocate {
                            spec,
                            bootstrap_addr,
                            hosts,
                        }) => {
                            tracing::info!("received allocation request: {:?}", spec);
                            match process_allocator.allocate(spec.clone()).await {
                                Ok(alloc) => {
                                    self.handle_allocation_request(
                                        &mut rx,
                                        Box::new(alloc) as Box<dyn Alloc + Send + Sync>,
                                        bootstrap_addr,
                                        hosts,
                                    )
                                    .await;
                                }
                                Err(e) => {
                                    tracing::error!("allocation for {:?} failed: {}", spec, e);
                                    continue;
                                }
                            }
                        }
                        Ok(RemoteProcessAllocatorMessage::Stop) => {
                            tracing::info!("received stop request");
                        }
                        Err(e) => {
                            tracing::error!("upstream channel error: {}", e);
                            continue;
                        }
                    }
                }
                _ = self.cancel_token.cancelled() => {
                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_allocation_request(
        &self,
        rx: &mut ChannelRx<RemoteProcessAllocatorMessage>,
        mut alloc: Box<dyn Alloc + Send + Sync>,
        bootstrap_addr: ChannelAddr,
        hosts: Vec<String>,
    ) {
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

        let mut running = true;
        loop {
            tokio::select! {
                biased;
                _ = self.cancel_token.cancelled() => {
                    tracing::info!("cancelled");
                    break;
                }
                m = rx.recv(), if running => {
                    match m {
                        Ok(RemoteProcessAllocatorMessage::Stop) => {
                            tracing::info!("received stop request");
                            if let Err(e) = alloc.stop().await {
                                tracing::error!("stop failed: {}", e);
                                return;
                            }
                            running = false;
                        }
                        Ok(_) => {
                            tracing::error!("unexpected message: {:?}", m);
                            return;
                        }
                        Err(e) => {
                            tracing::error!("upstream channel error: {}", e);
                            return;
                        }
                    }
                }
                e = alloc.next() => {
                    match e {
                        Some(event) => {
                            tracing::debug!("sending event: {:?}", event);
                            if let Err(e) = tx.send(RemoteProcessProcStateMessage::Update(event)).await {
                                tracing::error!("failed to send event to bootstrap address: {}", e);
                                return;
                            }
                        }
                        None => {
                            tracing::debug!("sending done");
                            if let Err(e) = tx.send(RemoteProcessProcStateMessage::Done(alloc.world_id().clone())).await {
                                tracing::error!("failed to send Done to bootstrap address: {}", e);
                            }
                            return;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::thread::sleep;

    use hyperactor::ActorRef;
    use hyperactor::id;
    use ndslice::shape;

    use super::*;
    use crate::alloc::AllocConstraints;
    use crate::alloc::ChannelTransport;
    use crate::alloc::MockAlloc;
    use crate::alloc::MockAllocator;
    use crate::proc_mesh::mesh_agent::MeshAgent;

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
        for i in 0..alloc_len {
            let proc_id = format!("test[{}]", i).parse().unwrap();
            let coords = spec.shape.slice().coordinates(i).unwrap();
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
            alloc
                .expect_next()
                .times(1)
                .return_once(|| Some(ProcState::Stopped(proc_id)));
        }
        // final none
        alloc.expect_next().times(1).return_once(|| None);
        // hack to block next() until we drain and issue cancellation
        // as mockall doesn't support returning futures.
        alloc.expect_next().times(..1).return_once(|| {
            #[allow(clippy::disallowed_methods)]
            sleep(Duration::from_secs(1));
            None
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
        })
        .await
        .unwrap();

        // Allocated
        let m = rx.recv().await.unwrap();
        assert!(
            matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape)
        );

        // All Created events
        for i in 0..alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Created { proc_id, coords }) => {
                    let expected_proc_id = format!("test[{}]", i).parse().unwrap();
                    let expected_coords = spec.shape.slice().coordinates(i).unwrap();
                    assert_eq!(proc_id, expected_proc_id);
                    assert_eq!(coords, expected_coords);
                }
                _ => panic!("unexpected message: {:?}", m),
            }
        }
        // All Running events
        for i in 0..alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Running {
                    proc_id,
                    mesh_agent,
                    addr,
                }) => {
                    let expected_proc_id = format!("test[{}]", i).parse().unwrap();
                    let expected_mesh_agent = ActorRef::<MeshAgent>::attest(
                        format!("test[{}].mesh_agent[{}]", i, i).parse().unwrap(),
                    );
                    assert_eq!(proc_id, expected_proc_id);
                    assert_eq!(mesh_agent, expected_mesh_agent);
                    assert_eq!(addr, ChannelAddr::Unix("/proc0".parse().unwrap()))
                }
                _ => panic!("unexpected message: {:?}", m),
            }
        }
        // All Stopped events
        for i in 0..alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Stopped(proc_id)) => {
                    let expected_proc_id = format!("test[{}]", i).parse().unwrap();
                    assert_eq!(proc_id, expected_proc_id);
                }
                _ => panic!("unexpected message: {:?}", m),
            }
        }
        // Done
        let m = rx.recv().await.unwrap();
        assert!(matches!(
            m,
            RemoteProcessProcStateMessage::Done(id) if id == world_id
        ));

        remote_allocator.terminate();
        handle.await.unwrap().unwrap();
    }
}
