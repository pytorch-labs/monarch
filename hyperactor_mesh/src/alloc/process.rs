#![allow(dead_code)] // some things currently used only in tests

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use enum_as_inner::EnumAsInner;
use hyperactor::ProcId;
use hyperactor::WorldId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelError;
use hyperactor::channel::ChannelTransport;
use hyperactor::channel::ChannelTx;
use hyperactor::channel::Rx;
use hyperactor::channel::Tx;
use hyperactor::channel::TxStatus;
use hyperactor::sync::monitor;
use ndslice::Shape;
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use super::Alloc;
use super::AllocSpec;
use super::Allocator;
use super::AllocatorError;
use super::ProcState;
use crate::assign::Ranks;
use crate::bootstrap;
use crate::bootstrap::Allocator2Process;
use crate::bootstrap::Process2Allocator;
use crate::bootstrap::Process2AllocatorMessage;
use crate::shortuuid::ShortUuid;

/// An allocator that allocates procs by executing managed (local)
/// processes. ProcessAllocator is configured with a [`Command`] (template)
/// to spawn external processes. These processes must invoke [`hyperactor_mesh::bootstrap`] or
/// [`hyperactor_mesh::bootstrap_or_die`], which is responsible for coordinating
/// with the allocator.
pub struct ProcessAllocator {
    cmd: Arc<Mutex<Command>>,
}

impl ProcessAllocator {
    /// Create a new allocator using the provided command (template).
    /// The command is used to spawn child processes that host procs.
    /// The binary should yield control to [`hyperactor_mesh::bootstrap`]
    /// or [`hyperactor_mesh::bootstrap_or_die`] or after initialization.
    pub fn new(cmd: Command) -> Self {
        Self {
            cmd: Arc::new(Mutex::new(cmd)),
        }
    }
}

#[async_trait]
impl Allocator for ProcessAllocator {
    type Alloc = ProcessAlloc;

    async fn allocate(&mut self, spec: AllocSpec) -> Result<ProcessAlloc, AllocatorError> {
        let (bootstrap_addr, rx) = channel::serve(ChannelAddr::any(ChannelTransport::Unix))
            .await
            .map_err(anyhow::Error::from)?;

        let name = ShortUuid::generate();
        let n = spec.shape.slice().len();
        Ok(ProcessAlloc {
            name: name.clone(),
            world_id: WorldId(name.to_string()),
            spec: spec.clone(),
            bootstrap_addr,
            rx,
            index: 0,
            active: HashMap::new(),
            ranks: Ranks::new(n),
            cmd: Arc::clone(&self.cmd),
            children: JoinSet::new(),
            running: true,
        })
    }
}

/// An allocation produced by [`ProcessAllocator`].
pub struct ProcessAlloc {
    name: ShortUuid,
    world_id: WorldId, // to provide storage
    spec: AllocSpec,
    bootstrap_addr: ChannelAddr,
    rx: channel::ChannelRx<Process2Allocator>,
    index: usize,
    active: HashMap<usize, Child>,
    // Maps process index to its rank.
    ranks: Ranks<usize>,
    cmd: Arc<Mutex<Command>>,
    children: JoinSet<usize>,
    running: bool,
}

#[derive(EnumAsInner)]
enum ChannelState {
    NotConnected,
    Connected(ChannelTx<Allocator2Process>),
    Failed(ChannelError),
}

struct Child {
    channel: ChannelState,
    group: monitor::Group,
}

impl Child {
    fn monitored(mut process: tokio::process::Child) -> (Self, impl Future) {
        let (group, handle) = monitor::group();

        let child = Self {
            channel: ChannelState::NotConnected,
            group,
        };

        let monitor = async move {
            tokio::select! {
                _ = handle => {
                    match process.kill().await {
                        Err(err) => {
                            tracing::error!("error killing process: {}", err);
                            // In this cased, we're left with little choice but to
                            // orphan the process.
                        },
                        Ok(_) => {
                            let _ = process.wait().await;
                        }
                    }
                }
                _ = process.wait() => (),
            }
        };

        (child, monitor)
    }

    fn kill(&self) {
        self.group.fail();
    }

    fn connect(&mut self, addr: ChannelAddr) -> bool {
        if !self.channel.is_not_connected() {
            return false;
        }

        let cloned_addr = addr.clone();
        match channel::dial(addr) {
            Ok(channel) => {
                let mut status = channel.status().clone();
                self.channel = ChannelState::Connected(channel);
                // Monitor the channel, killing the process if it becomes unavailable
                // (fails keepalive).
                self.group.spawn(async move {
                    let _ = status
                        .wait_for(|status| matches!(status, TxStatus::Closed))
                        .await;
                    Result::<(), ()>::Err(())
                });
            }
            Err(err) => {
                self.channel = ChannelState::Failed(err);
                self.kill();
            }
        };
        true
    }

    fn post(&mut self, message: Allocator2Process) {
        // We're here simply assuming that if we're not connected, we're about to
        // be killed.
        if let ChannelState::Connected(channel) = &mut self.channel {
            channel.post(message);
        } else {
            self.kill();
        }
    }
}

impl ProcessAlloc {
    // Also implement exit (for graceful exit)

    // Currently procs and processes are 1:1, so this just fully exits
    // the process.
    fn kill(&mut self, proc_id: &ProcId) -> Result<(), anyhow::Error> {
        self.get_mut(proc_id)?.kill();
        Ok(())
    }

    fn get(&self, proc_id: &ProcId) -> Result<&Child, anyhow::Error> {
        self.active.get(&self.index(proc_id)?).ok_or_else(|| {
            anyhow::anyhow!(
                "proc {} not currently active in alloc {}",
                proc_id,
                self.name
            )
        })
    }

    fn get_mut(&mut self, proc_id: &ProcId) -> Result<&mut Child, anyhow::Error> {
        self.active.get_mut(&self.index(proc_id)?).ok_or_else(|| {
            anyhow::anyhow!(
                "proc {} not currently active in alloc {}",
                &proc_id,
                self.name
            )
        })
    }

    fn index(&self, proc_id: &ProcId) -> Result<usize, anyhow::Error> {
        anyhow::ensure!(
            proc_id.world_name().parse::<ShortUuid>()? == self.name,
            "proc {} does not belong to alloc {}",
            proc_id,
            self.name
        );
        Ok(proc_id.rank())
    }

    async fn maybe_spawn(&mut self) -> Option<ProcState> {
        if self.active.len() >= self.spec.shape.slice().len() {
            return None;
        }
        let mut cmd = self.cmd.lock().await;
        let index = self.index;
        self.index += 1;

        cmd.env(
            bootstrap::BOOTSTRAP_ADDR_ENV,
            self.bootstrap_addr.to_string(),
        );
        cmd.env(bootstrap::BOOTSTRAP_INDEX_ENV, index.to_string());

        // Opt-in to signal handling (`PR_SET_PDEATHSIG`) so that the
        // spawned subprocess will automatically exit when the parent
        // process dies.
        cmd.env("HYPERACTOR_MANAGED_SUBPROCESS", "1");

        let proc_id = ProcId(WorldId(self.name.to_string()), index);
        match cmd.spawn() {
            Err(err) => {
                // Should we proactively retry here, or do we always just
                // wait for another event request?
                tracing::error!("spawn {}: {}", index, err);
                None
            }
            Ok(mut process) => match self.ranks.assign(index) {
                Err(_index) => {
                    tracing::info!("could not assign rank to {}", proc_id);
                    let _ = process.kill().await;
                    None
                }
                Ok(rank) => {
                    let (handle, monitor) = Child::monitored(process);
                    self.children.spawn(async move {
                        monitor.await;
                        index
                    });
                    self.active.insert(index, handle);
                    // Adjust for shape slice offset for non-zero shapes (sub-shapes).
                    let rank = rank + self.spec.shape.slice().offset();
                    let coords = self.spec.shape.slice().coordinates(rank).unwrap();
                    Some(ProcState::Created { proc_id, coords })
                }
            },
        }
    }

    fn remove(&mut self, index: usize) {
        self.ranks.unassign(index);
        self.active.remove(&index);
    }
}

#[async_trait]
impl Alloc for ProcessAlloc {
    async fn next(&mut self) -> Option<ProcState> {
        if !self.running && self.active.is_empty() {
            return None;
        }

        loop {
            if self.running {
                if let state @ Some(_) = self.maybe_spawn().await {
                    return state;
                }
            }

            let transport = self.transport().clone();

            tokio::select! {
                Ok(Process2Allocator(index, message)) = self.rx.recv() => {
                    let child = match self.active.get_mut(&index) {
                        None => {
                            tracing::info!("message {:?} from zombie {}", message, index);
                            continue;
                        }
                        Some(child) => child,
                    };

                    match message {
                        Process2AllocatorMessage::Hello(addr) => {
                            if !child.connect(addr.clone()) {
                                tracing::error!("received multiple hellos from {}", index);
                                continue;
                            }

                            child.post(Allocator2Process::StartProc(
                                ProcId(WorldId(self.name.to_string()), index),
                                transport,
                            ));
                        }

                        Process2AllocatorMessage::StartedProc(proc_id, mesh_agent, addr) => {
                            break Some(ProcState::Running {
                                proc_id,
                                mesh_agent,
                                addr,
                            });
                        }
                    }
                },

                Some(Ok(index)) = self.children.join_next() => {
                    self.remove(index);
                    break Some(ProcState::Stopped(ProcId(WorldId(self.name.to_string()), index)));
                },
            }
        }
    }

    fn shape(&self) -> &Shape {
        &self.spec.shape
    }

    fn world_id(&self) -> &WorldId {
        &self.world_id
    }

    fn transport(&self) -> ChannelTransport {
        ChannelTransport::Unix
    }

    async fn stop(&mut self) -> Result<(), AllocatorError> {
        // We rely on the teardown here, and that the process should
        // exit on its own. We shoudl have a hard timeout here as well,
        // so that we never rely on the system functioning correctly
        // for liveness.
        for (_index, child) in self.active.iter_mut() {
            child.post(Allocator2Process::StopAndExit(0));
        }

        self.running = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(fbcode_build)] // we use an external binary, produced by buck
    crate::alloc_test_suite!(ProcessAllocator::new(Command::new(
        buck_resources::get("monarch/hyperactor_mesh/bootstrap").unwrap()
    )));
}
