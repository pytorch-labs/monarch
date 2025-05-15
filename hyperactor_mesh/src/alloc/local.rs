//! Support for allocating procs in the local process.

#![allow(dead_code)] // until it is used outside of testing

use std::collections::VecDeque;
use std::future;
use std::time::Duration;

use async_trait::async_trait;
use hyperactor::ProcId;
use hyperactor::WorldId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use hyperactor::mailbox;
use hyperactor::mailbox::MailboxServer;
use hyperactor::mailbox::MailboxServerHandle;
use hyperactor::proc::Proc;
use ndslice::Shape;
use tokio::time::sleep;

use super::ProcStopReason;
use crate::alloc::Alloc;
use crate::alloc::AllocSpec;
use crate::alloc::Allocator;
use crate::alloc::AllocatorError;
use crate::alloc::ProcState;
use crate::proc_mesh::mesh_agent::MeshAgent;
use crate::shortuuid::ShortUuid;

/// An allocator that runs procs in the local process. It is primarily useful for testing,
/// or small meshes that can run entirely locally.
///
/// Currently, the allocator will allocate all procs, but each is treated as infallible,
/// since they share fault domain with the client of the alloc.
pub struct LocalAllocator;

#[async_trait]
impl Allocator for LocalAllocator {
    type Alloc = LocalAlloc;

    async fn allocate(&mut self, spec: AllocSpec) -> Result<Self::Alloc, AllocatorError> {
        Ok(LocalAlloc::new(spec))
    }
}

struct LocalProc {
    rank: usize,
    proc: Proc,
    addr: ChannelAddr,
    handle: MailboxServerHandle,
}

/// A local allocation. It is a collection of procs that are running in the local process.
pub struct LocalAlloc {
    spec: AllocSpec,
    name: ShortUuid,
    world_id: WorldId, // to provide storage
    procs: Vec<LocalProc>,
    queue: VecDeque<ProcState>,
    running: bool,
}

impl LocalAlloc {
    fn new(spec: AllocSpec) -> Self {
        let name = ShortUuid::generate();
        Self {
            spec,
            name: name.clone(),
            world_id: WorldId(name.to_string()),
            procs: Vec::new(),
            queue: VecDeque::new(),
            running: true,
        }
    }

    pub(crate) fn name(&self) -> &ShortUuid {
        &self.name
    }

    fn size(&self) -> usize {
        self.spec.shape.slice().len()
    }
}

#[async_trait]
impl Alloc for LocalAlloc {
    async fn next(&mut self) -> Option<ProcState> {
        if let state @ Some(_) = self.queue.pop_front() {
            return state;
        }

        if self.running {
            if self.procs.len() == self.size() {
                future::pending::<()>().await;
                unreachable!("future::pending completed");
            }

            // Perform an allocation:
            let rank = self.procs.len();

            let proc_id = ProcId(self.world_id.clone(), rank);
            let (proc, mesh_agent) = match MeshAgent::bootstrap(proc_id.clone()).await {
                Ok(proc_and_agent) => proc_and_agent,
                Err(err) => {
                    tracing::error!("failed spawn mesh agent for {}: {}", rank, err);
                    // It's unclear if this is actually recoverable in a practical sense,
                    // so we give up.
                    return None;
                }
            };

            let (addr, proc_rx) = loop {
                match channel::serve(ChannelAddr::any(self.transport())).await {
                    Ok(addr_and_proc_rx) => break addr_and_proc_rx,
                    Err(err) => {
                        tracing::error!("failed to create channel for rank {}: {}", rank, err);
                        #[allow(clippy::disallowed_methods)]
                        sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                }
            };

            let handle = proc
                .clone()
                .serve(proc_rx, mailbox::monitored_return_handle());

            self.procs.push(LocalProc {
                rank,
                proc,
                addr: addr.clone(),
                handle,
            });

            // Adjust for shape slice offset for non-zero shapes (sub-shapes).
            let rank = rank + self.spec.shape.slice().offset();
            let coords = match self.spec.shape.slice().coordinates(rank) {
                Ok(coords) => coords,
                Err(err) => {
                    tracing::error!("failed to get coords for rank {}: {}", rank, err);
                    return None;
                }
            };
            let created = ProcState::Created {
                proc_id: proc_id.clone(),
                coords,
            };
            self.queue.push_back(ProcState::Running {
                proc_id,
                mesh_agent: mesh_agent.bind(),
                addr,
            });
            Some(created)
        } else {
            let mut proc_to_stop = self.procs.pop()?;
            if let Err(err) = proc_to_stop
                .proc
                .destroy_and_wait(Duration::from_millis(10), None)
                .await
            {
                tracing::error!("error while stopping proc {}: {}", proc_to_stop.rank, err);
            }
            Some(ProcState::Stopped {
                reason: ProcStopReason::Stopped,
                proc_id: proc_to_stop.proc.proc_id().clone(),
            })
        }
    }

    fn shape(&self) -> &Shape {
        &self.spec.shape
    }

    fn world_id(&self) -> &WorldId {
        &self.world_id
    }

    fn transport(&self) -> ChannelTransport {
        ChannelTransport::Local
    }

    async fn stop(&mut self) -> Result<(), AllocatorError> {
        self.running = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::alloc_test_suite!(LocalAllocator);
}
