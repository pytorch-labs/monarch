/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Support for allocating procs in the local process.

#![allow(dead_code)] // until it is used outside of testing

use std::collections::HashMap;
use std::collections::VecDeque;
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
use tokio::sync::mpsc;
use tokio::time::sleep;

use super::ProcStopReason;
use crate::alloc::Alloc;
use crate::alloc::AllocSpec;
use crate::alloc::Allocator;
use crate::alloc::AllocatorError;
use crate::alloc::ProcState;
use crate::proc_mesh::mesh_agent::MeshAgent;
use crate::shortuuid::ShortUuid;

enum Action {
    Start(usize),
    Stop(usize, ProcStopReason),
    Stopped,
}

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
    proc: Proc,
    addr: ChannelAddr,
    handle: MailboxServerHandle,
}

/// A local allocation. It is a collection of procs that are running in the local process.
pub struct LocalAlloc {
    spec: AllocSpec,
    name: ShortUuid,
    world_id: WorldId, // to provide storage
    procs: HashMap<usize, LocalProc>,
    queue: VecDeque<ProcState>,
    todo_tx: mpsc::UnboundedSender<Action>,
    todo_rx: mpsc::UnboundedReceiver<Action>,
    stopped: bool,
    failed: bool,
}

impl LocalAlloc {
    fn new(spec: AllocSpec) -> Self {
        let name = ShortUuid::generate();
        let (todo_tx, todo_rx) = mpsc::unbounded_channel();
        for rank in 0..spec.shape.slice().len() {
            todo_tx.send(Action::Start(rank)).unwrap();
        }
        Self {
            spec,
            name: name.clone(),
            world_id: WorldId(name.to_string()),
            procs: HashMap::new(),
            queue: VecDeque::new(),
            todo_tx,
            todo_rx,
            stopped: false,
            failed: false,
        }
    }

    /// A chaos monkey that can be used to stop procs at random.
    pub(crate) fn chaos_monkey(&self) -> impl Fn(usize, ProcStopReason) + 'static {
        let todo_tx = self.todo_tx.clone();
        move |rank, reason| {
            todo_tx.send(Action::Stop(rank, reason)).unwrap();
        }
    }

    /// A function to shut down the alloc for testing purposes.
    pub(crate) fn stopper(&self) -> impl Fn() + 'static {
        let todo_tx = self.todo_tx.clone();
        let size = self.size();
        move || {
            for rank in 0..size {
                todo_tx
                    .send(Action::Stop(rank, ProcStopReason::Stopped))
                    .unwrap();
            }
            todo_tx.send(Action::Stopped).unwrap();
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
        if self.stopped {
            return None;
        }
        if self.failed && !self.stopped {
            // Failed alloc. Wait for stop().
            futures::future::pending::<()>().await;
            unreachable!("future::pending completed");
        }
        let event = loop {
            if let state @ Some(_) = self.queue.pop_front() {
                break state;
            }

            match self.todo_rx.recv().await? {
                Action::Start(rank) => {
                    let proc_id = ProcId(self.world_id.clone(), rank);
                    let bspan = tracing::info_span!("mesh_agent_bootstrap");
                    let (proc, mesh_agent) = match MeshAgent::bootstrap(proc_id.clone()).await {
                        Ok(proc_and_agent) => proc_and_agent,
                        Err(err) => {
                            let message = format!("failed spawn mesh agent for {}: {}", rank, err);
                            tracing::error!(message);
                            // It's unclear if this is actually recoverable in a practical sense,
                            // so we give up.
                            self.failed = true;
                            break Some(ProcState::Failed {
                                world_id: self.world_id.clone(),
                                description: message,
                            });
                        }
                    };
                    drop(bspan);

                    let (addr, proc_rx) = loop {
                        match channel::serve(ChannelAddr::any(self.transport())).await {
                            Ok(addr_and_proc_rx) => break addr_and_proc_rx,
                            Err(err) => {
                                tracing::error!(
                                    "failed to create channel for rank {}: {}",
                                    rank,
                                    err
                                );
                                #[allow(clippy::disallowed_methods)]
                                sleep(Duration::from_secs(1)).await;
                                continue;
                            }
                        }
                    };

                    let handle = proc
                        .clone()
                        .serve(proc_rx, mailbox::monitored_return_handle());

                    self.procs.insert(
                        rank,
                        LocalProc {
                            proc,
                            addr: addr.clone(),
                            handle,
                        },
                    );

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
                    break Some(created);
                }
                Action::Stop(rank, reason) => {
                    let Some(mut proc_to_stop) = self.procs.remove(&rank) else {
                        continue;
                    };
                    if let Err(err) = proc_to_stop
                        .proc
                        .destroy_and_wait(Duration::from_millis(10), None)
                        .await
                    {
                        tracing::error!("error while stopping proc {}: {}", rank, err);
                    }
                    break Some(ProcState::Stopped {
                        reason,
                        proc_id: proc_to_stop.proc.proc_id().clone(),
                    });
                }
                Action::Stopped => break None,
            }
        };
        self.stopped = event.is_none();
        event
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
        for rank in 0..self.size() {
            self.todo_tx
                .send(Action::Stop(rank, ProcStopReason::Stopped))
                .unwrap();
        }
        self.todo_tx.send(Action::Stopped).unwrap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::alloc_test_suite!(LocalAllocator);
}
