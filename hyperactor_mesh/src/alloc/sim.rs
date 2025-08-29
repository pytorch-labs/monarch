/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Support for allocating procs in the local process with simulated channels.

#![allow(dead_code)] // until it is used outside of testing

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use hyperactor::ProcId;
use hyperactor::WorldId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use hyperactor::mailbox::MailboxServer;
use hyperactor::mailbox::MailboxServerHandle;
use hyperactor::proc::Proc;
use hyperactor::simnet::BetaDistribution;
use hyperactor::simnet::Event;
use hyperactor::simnet::SimNetError;
use ndslice::view::Extent;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::Mutex;

use crate::alloc::Alloc;
use crate::alloc::AllocSpec;
use crate::alloc::Allocator;
use crate::alloc::AllocatorError;
use crate::alloc::ProcState;
use crate::alloc::ProcStopReason;
use crate::proc_mesh::mesh_agent::MeshAgent;
use crate::shortuuid::ShortUuid;

/// An allocator that runs procs in the local process with network traffic going through simulated channels.
pub struct SimAllocator;

const CHAOS_MONKEY_TIMEOUT: tokio::time::Duration = tokio::time::Duration::from_secs(10);

const PROC_STOP_TIMEOUT: tokio::time::Duration = tokio::time::Duration::from_secs(10);

#[async_trait]
impl Allocator for SimAllocator {
    type Alloc = SimAlloc;

    async fn allocate(&mut self, spec: AllocSpec) -> Result<Self::Alloc, AllocatorError> {
        Ok(SimAlloc::new(spec))
    }
}

impl SimAllocator {
    #[cfg(test)]
    pub(crate) fn new_and_start_simnet() -> Self {
        hyperactor::simnet::start();
        Self
    }
}

struct SimProc {
    proc: Proc,
    addr: ChannelAddr,
    handle: MailboxServerHandle,
}

/// A simulated allocation. It is a collection of procs that are running in the local process.
pub struct SimAlloc {
    spec: AllocSpec,
    name: String,
    world_id: WorldId,
    proc_event_rx:
        tokio::sync::mpsc::UnboundedReceiver<(Option<ProcState>, Option<(usize, LocalProc)>)>,
    proc_event_tx:
        tokio::sync::mpsc::UnboundedSender<(Option<ProcState>, Option<(usize, LocalProc)>)>,
    procs: Arc<Mutex<HashMap</*rank:*/ usize, LocalProc>>>,
    stopped: bool,
}

impl SimAlloc {
    fn new(spec: AllocSpec) -> Self {
        let mut rng = {
            let seed: u64 = spec
                .constraints
                .match_labels
                .get("alloc_seed")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            let mut seed_bytes = [0u8; 32];
            seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
            StdRng::from_seed(seed_bytes)
        };

        let min_allocation_ms = spec
            .constraints
            .match_labels
            .get("min_allocation_ms")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(10);
        let max_allocation_ms = spec
            .constraints
            .match_labels
            .get("max_allocation_ms")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1000);
        let allocation_time_alpha = spec
            .constraints
            .match_labels
            .get("allocation_time_alpha")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(2.0);
        let allocation_time_beta = spec
            .constraints
            .match_labels
            .get("allocation_time_beta")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1.0);

        let name = ShortUuid::generate().to_string();
        let client_proc_id = ProcId::Ranked(WorldId(format!("{}_manager", name)), 0);
        let world_id = WorldId(name.clone());

        let handle = hyperactor::simnet::simnet_handle().expect("simnet event loop not running");

        handle.register_proc(
            client_proc_id.clone(),
            spec.extent
                .point(spec.extent.sizes().iter().map(|_| 0).collect())
                .expect("should be valid point"),
        );

        let (proc_event_tx, proc_event_rx) = tokio::sync::mpsc::unbounded_channel();

        for rank in 0..spec.extent.num_ranks() {
            let _ = handle.send_event(Box::new(AllocateProcEvent {
                tx: proc_event_tx.clone(),
                rank,
                world_id: world_id.clone(),
                spec: spec.clone(),
                duration: BetaDistribution::new(
                    tokio::time::Duration::from_millis(min_allocation_ms),
                    tokio::time::Duration::from_millis(max_allocation_ms),
                    allocation_time_alpha,
                    allocation_time_beta,
                )
                .expect("valid parameters")
                .sample(&mut rng),
            }));
        }

        Self {
            spec,
            name,
            world_id,
            proc_event_rx,
            proc_event_tx,
            procs: Arc::new(Mutex::new(HashMap::new())),
            stopped: false,
        }
    }

    /// A chaos monkey that can be used to stop procs at random.
    pub(crate) fn chaos_monkey(
        &self,
    ) -> impl Fn(usize) -> Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + 'static>
    {
        let procs = self.procs.clone();
        move |rank| {
            let procs = procs.clone();
            Box::new(async move {
                let proc = {
                    let mut guard = procs.lock().await;
                    guard.remove(&rank)
                };

                match proc {
                    Some(mut proc) => {
                        proc.handle.stop("received Action::Stop");
                        proc.proc
                            .destroy_and_wait(CHAOS_MONKEY_TIMEOUT, None)
                            .await
                            .map_err(|e| {
                                anyhow::anyhow!("failed to destroy proc {}: {}", rank, e)
                            })?;
                        Ok(())
                    }
                    None => Err(anyhow::anyhow!("proc {} not found", rank)),
                }
            })
        }
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}

struct LocalProc {
    proc: Proc,
    handle: MailboxServerHandle,
}

#[derive(Debug)]
struct AllocateProcEvent {
    tx: tokio::sync::mpsc::UnboundedSender<(Option<ProcState>, Option<(usize, LocalProc)>)>,
    world_id: WorldId,
    rank: usize,
    spec: AllocSpec,
    duration: tokio::time::Duration,
}

#[async_trait]
impl Event for AllocateProcEvent {
    async fn handle(&mut self) -> Result<(), SimNetError> {
        let proc_id = ProcId::Ranked(self.world_id.clone(), self.rank);
        let bspan = tracing::info_span!("mesh_agent_bootstrap");
        let (proc, mesh_agent) = match MeshAgent::bootstrap(proc_id.clone()).await {
            Ok(proc_and_agent) => proc_and_agent,
            Err(err) => {
                let message = format!("failed spawn mesh agent for {}: {}", self.rank, err);
                tracing::error!(message);
                // It's unclear if this is actually recoverable in a practical sense,
                // so we give up.
                self.tx
                    .send((
                        Some(ProcState::Failed {
                            world_id: self.world_id.clone(),
                            description: message,
                        }),
                        None,
                    ))
                    .expect("should be able to send");
                return Ok(());
            }
        };
        drop(bspan);

        let (addr, proc_rx) = channel::serve(ChannelAddr::any(ChannelTransport::Sim(Box::new(
            ChannelTransport::Unix,
        ))))
        .await
        .expect("should be able to serve sim channel");

        // Undeliverable messages get forwarded to the mesh agent.
        let handle = proc.clone().serve(proc_rx);

        let point = match self.spec.extent.point_of_rank(self.rank) {
            Ok(point) => point,
            Err(err) => {
                tracing::error!("failed to get point for rank {}: {}", self.rank, err);
                self.tx.send((None, None)).expect("should be able to send");
                return Ok(());
            }
        };

        self.tx
            .send((
                Some(ProcState::Created {
                    proc_id: proc_id.clone(),
                    point,
                    pid: std::process::id(),
                }),
                Some((self.rank, LocalProc { proc, handle })),
            ))
            .expect("should be able to send");

        self.tx
            .send((
                Some(ProcState::Running {
                    proc_id,
                    mesh_agent: mesh_agent.bind(),
                    addr,
                }),
                None,
            ))
            .expect("should be able to send");

        Ok(())
    }

    fn duration(&self) -> tokio::time::Duration {
        self.duration
    }

    fn summary(&self) -> String {
        let proc_id = ProcId::Ranked(self.world_id.clone(), self.rank);
        format!("allocating proc {}", proc_id)
    }
}

#[async_trait]
impl Alloc for SimAlloc {
    async fn next(&mut self) -> Option<ProcState> {
        if self.stopped {
            return None;
        }

        if let Some((proc_state, new_proc)) = self.proc_event_rx.recv().await {
            match &proc_state {
                Some(ProcState::Created { proc_id, point, .. }) => {
                    hyperactor::simnet::simnet_handle()
                        .expect("simnet event loop not running")
                        .register_proc(proc_id.clone(), point.clone());
                }
                None => {
                    self.stopped = true;
                }
                _ => {}
            }
            if let Some((rank, proc)) = new_proc {
                let mut guard = self.procs.lock().await;
                guard.insert(rank, proc);
            }
            proc_state
        } else {
            self.stopped = true;
            None
        }
    }

    fn extent(&self) -> &Extent {
        &self.spec.extent
    }

    fn world_id(&self) -> &WorldId {
        &self.world_id
    }

    fn transport(&self) -> ChannelTransport {
        ChannelTransport::Sim(Box::new(ChannelTransport::Unix))
    }

    async fn stop(&mut self) -> Result<(), AllocatorError> {
        let mut guard = self.procs.lock().await;
        let mut procs_to_stop = Vec::new();
        for rank in 0..self.spec.extent.num_ranks() {
            if let Some(proc) = guard.remove(&rank) {
                proc.handle.stop("received Action::Stop");
                procs_to_stop.push((proc, rank));
            }
        }
        futures::future::join_all(procs_to_stop.into_iter().map(|(mut proc, rank)| {
            let tx = self.proc_event_tx.clone();
            async move {
                if let Err(e) = proc.proc.destroy_and_wait(PROC_STOP_TIMEOUT, None).await {
                    tracing::error!("failed to destroy proc {}: {}", rank, e);
                } else {
                    tx.send((
                        Some(ProcState::Stopped {
                            proc_id: proc.proc.proc_id().clone(),
                            reason: ProcStopReason::Stopped,
                        }),
                        None,
                    ))
                    .expect("should be able to send");
                }
            }
        }))
        .await;
        self.proc_event_tx
            .send((None, None))
            .expect("should be able to send");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use hyperactor::simnet::BetaDistribution;
    use hyperactor::simnet::LatencyConfig;
    use hyperactor::simnet::LatencyDistribution;
    use ndslice::extent;

    use super::*;
    use crate::ProcMesh;
    use crate::RootActorMesh;
    use crate::actor_mesh::ActorMesh;
    use crate::alloc::AllocConstraints;
    use crate::alloc::test_utils::TestActor;

    #[tokio::test]
    async fn test_allocator_basic() {
        hyperactor::simnet::start();
        crate::alloc::testing::test_allocator_basic(SimAllocator).await;
    }

    #[tokio::test]
    async fn test_allocator_registers_resources() {
        hyperactor::simnet::start_with_config(LatencyConfig {
            inter_zone_distribution: LatencyDistribution::Beta(
                BetaDistribution::new(
                    tokio::time::Duration::from_millis(999),
                    tokio::time::Duration::from_millis(999),
                    1.0,
                    1.0,
                )
                .unwrap(),
            ),
            ..Default::default()
        });

        let alloc = SimAllocator
            .allocate(AllocSpec {
                extent: extent!(region = 1, dc = 1, zone = 10, rack = 1, host = 1, gpu = 1),
                constraints: AllocConstraints {
                    match_labels: HashMap::new(),
                },
            })
            .await
            .unwrap();

        let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();

        let handle = hyperactor::simnet::simnet_handle().unwrap();
        let actor_mesh: RootActorMesh<TestActor> = proc_mesh.spawn("echo", &()).await.unwrap();
        let actors = actor_mesh.iter_actor_refs().collect::<Vec<_>>();
        assert_eq!(
            handle.sample_latency(
                actors[0].actor_id().proc_id(),
                actors[1].actor_id().proc_id()
            ),
            tokio::time::Duration::from_millis(999)
        );
        assert_eq!(
            handle.sample_latency(
                actors[2].actor_id().proc_id(),
                actors[9].actor_id().proc_id()
            ),
            tokio::time::Duration::from_millis(999)
        );
        assert_eq!(
            handle.sample_latency(
                proc_mesh.client().actor_id().proc_id(),
                actors[1].actor_id().proc_id()
            ),
            tokio::time::Duration::from_millis(999)
        );
    }
}
