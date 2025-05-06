#![allow(dead_code)] // some things currently used only in tests

use std::collections::HashMap;
use std::mem::take;
use std::sync::Arc;
use std::sync::OnceLock;

use async_trait::async_trait;
use hyperactor::ProcId;
use hyperactor::WorldId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelError;
use hyperactor::channel::ChannelTransport;
use hyperactor::channel::Rx;
use hyperactor::channel::SendError;
use hyperactor::channel::Tx;
use ndslice::Shape;
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

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

        let (reap_tx, reap_rx) = mpsc::channel(1);

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
            reap_tx,
            reap_rx,
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
    reap_tx: mpsc::Sender<(usize, std::process::ExitStatus)>,
    reap_rx: mpsc::Receiver<(usize, std::process::ExitStatus)>,
    running: bool,
}

struct Child {
    addr: OnceLock<ChannelAddr>,
    kill: Option<oneshot::Sender<()>>,
}

impl Child {
    // (Apologies)
    fn new(
        mut process: tokio::process::Child,
        index: usize,
        reap: mpsc::Sender<(usize, std::process::ExitStatus)>,
    ) -> Self {
        let (kill_tx, mut kill_rx) = oneshot::channel();

        tokio::spawn(async move {
            let status = tokio::select! {
                _ = &mut kill_rx => {
                    let _ = process.kill().await;
                    process.wait().await
                }

                status = process.wait() => {
                    status
                }
            }
            .unwrap_or(Default::default());
            let _ = reap.send((index, status)).await;
        });

        Self {
            addr: OnceLock::new(),
            kill: Some(kill_tx),
        }
    }

    fn kill(&mut self) {
        if let Some(kill) = take(&mut self.kill) {
            let _ = kill.send(());
        }
    }

    async fn exit(&self, code: i32) -> Result<(), anyhow::Error> {
        self.send(Allocator2Process::Exit(code)).await?;
        Ok(())
    }

    async fn send(&self, message: Allocator2Process) -> Result<(), SendError<Allocator2Process>> {
        let addr = match self.addr.get() {
            Some(addr) => addr.clone(),
            None => {
                let err = ChannelError::from(anyhow::anyhow!(
                    "attempted to send on client for which no address is defined"
                ));
                return Err(SendError(err, message));
            }
        };
        let tx = match channel::dial(addr) {
            Ok(tx) => tx,
            Err(err) => return Err(SendError(err, message)),
        };
        tx.send(message).await
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
                    self.active
                        .insert(index, Child::new(process, index, self.reap_tx.clone()));
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
                            if child.addr.set(addr.clone()).is_err() {
                                tracing::error!("received multiple hellos from {}", index);
                                continue;
                            }

                            if let Err(err) = child
                                .send(Allocator2Process::StartProc(
                                    ProcId(WorldId(self.name.to_string()), index),
                                    transport,
                                ))
                                .await
                            {
                                tracing::error!("failed to send StartProc message to {addr}: {err}");
                                // We now consider this failed:
                                self.remove(index);
                            }
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

                Some((index, _status)) = self.reap_rx.recv() => {
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
        for (index, child) in self.active.iter_mut() {
            if let Err(err) = child.send(Allocator2Process::StopAndExit(0)).await {
                tracing::error!("failed to send StopAndExit message to {index}: {err}");
                // Make sure the child is actually killed.
                child.kill();
            }
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
