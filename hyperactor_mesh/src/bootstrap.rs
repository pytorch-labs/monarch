/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Duration;

use futures::StreamExt;
use hyperactor::ActorRef;
use hyperactor::Named;
use hyperactor::ProcId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use hyperactor::channel::Rx;
use hyperactor::channel::Tx;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use hyperactor::mailbox::MailboxServer;
use serde::Deserialize;
use serde::Serialize;
use signal_hook::consts::signal::SIGTERM;

use crate::proc_mesh::mesh_agent::MeshAgent;

pub(crate) const BOOTSTRAP_ADDR_ENV: &str = "HYPERACTOR_MESH_BOOTSTRAP_ADDR";
pub(crate) const BOOTSTRAP_INDEX_ENV: &str = "HYPERACTOR_MESH_INDEX";

/// Messages sent from the process to the allocator. This is an envelope
/// containing the index of the process (i.e., its "address" assigned by
/// the allocator), along with the control message in question.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub(crate) struct Process2Allocator(pub usize, pub Process2AllocatorMessage);

/// Control messages sent from processes to the allocator.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub(crate) enum Process2AllocatorMessage {
    /// Initialize a process2allocator session. The process is
    /// listening on the provided channel address, to which
    /// [`Allocator2Process`] messages are sent.
    Hello(ChannelAddr),

    /// A proc with the provided ID was started. Its mailbox is
    /// served at the provided channel address. Procs are started
    /// after instruction by the allocator through the corresponding
    /// [`Allocator2Process`] message.
    StartedProc(ProcId, ActorRef<MeshAgent>, ChannelAddr),

    Heartbeat,
}

/// Messages sent from the allocator to a process.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub(crate) enum Allocator2Process {
    /// Request to start a new proc with the provided ID, listening
    /// to an address on the indicated channel transport.
    StartProc(ProcId, ChannelTransport),

    /// A request for the process to shut down its procs and exit the
    /// process with the provided code.
    StopAndExit(i32),

    /// A request for the process to immediately exit with the provided
    /// exit code
    Exit(i32),
}

async fn exit_if_missed_heartbeat(bootstrap_index: usize, bootstrap_addr: ChannelAddr) {
    let tx = match channel::dial(bootstrap_addr.clone()) {
        Ok(tx) => tx,

        Err(err) => {
            tracing::error!(
                "Failed to establish heartbeat connection to allocator, exiting! (addr: {:?}): {}",
                bootstrap_addr,
                err
            );
            std::process::exit(1);
        }
    };
    tracing::info!(
        "Heartbeat connection established to allocator (idx: {bootstrap_index}, addr: {bootstrap_addr:?})",
    );
    loop {
        RealClock.sleep(Duration::from_secs(5)).await;

        let result = tx
            .send(Process2Allocator(
                bootstrap_index,
                Process2AllocatorMessage::Heartbeat,
            ))
            .await;

        if let Err(err) = result {
            tracing::error!(
                "Heartbeat failed to allocator, exiting! (addr: {:?}): {}",
                bootstrap_addr,
                err
            );
            std::process::exit(1);
        }
    }
}

/// Entry point to processes managed by hyperactor_mesh. This advertises the process
/// to a bootstrap server, and receives instructions to manage the lifecycle(s) of
/// procs within this process.
///
/// If bootstrap returns any error, it is defunct from the point of view of hyperactor_mesh,
/// and the process should likely exit:
///
/// ```ignore
/// let err = hyperactor_mesh::bootstrap().await;
/// tracing::error("could not bootstrap mesh process: {}", err);
/// std::process::exit(1);
/// ```
///
/// Use [`bootstrap_or_die`] to implement this behavior directly.
pub async fn bootstrap() -> anyhow::Error {
    pub async fn go() -> Result<(), anyhow::Error> {
        let mut signals = signal_hook_tokio::Signals::new([SIGTERM])?;

        let bootstrap_addr: ChannelAddr = std::env::var(BOOTSTRAP_ADDR_ENV)
            .map_err(|err| anyhow::anyhow!("read `{}`: {}", BOOTSTRAP_ADDR_ENV, err))?
            .parse()?;
        let bootstrap_index: usize = std::env::var(BOOTSTRAP_INDEX_ENV)
            .map_err(|err| anyhow::anyhow!("read `{}`: {}", BOOTSTRAP_INDEX_ENV, err))?
            .parse()?;
        let listen_addr = ChannelAddr::any(bootstrap_addr.transport());
        let (serve_addr, mut rx) = channel::serve(listen_addr).await?;
        let tx = channel::dial(bootstrap_addr.clone())?;

        tx.send(Process2Allocator(
            bootstrap_index,
            Process2AllocatorMessage::Hello(serve_addr),
        ))
        .await?;

        tokio::spawn(exit_if_missed_heartbeat(bootstrap_index, bootstrap_addr));

        let mut procs = Vec::new();

        loop {
            let _ = hyperactor::tracing::info_span!("wait_for_next_message_from_mesh_agent");
            tokio::select! {
                msg = rx.recv() => {
                    match msg? {
                        Allocator2Process::StartProc(proc_id, listen_transport) => {
                            let (proc, mesh_agent) = MeshAgent::bootstrap(proc_id.clone()).await?;
                            let (proc_addr, proc_rx) =
                                channel::serve(ChannelAddr::any(listen_transport)).await?;
                            // Undeliverable messages get forwarded to the mesh agent.
                            let handle = proc.clone().serve(proc_rx, mesh_agent.port());
                            drop(handle); // linter appeasement; it is safe to drop this future
                            tx.send(Process2Allocator(
                                bootstrap_index,
                                Process2AllocatorMessage::StartedProc(
                                    proc_id.clone(),
                                    mesh_agent.bind(),
                                    proc_addr,
                                ),
                            ))
                            .await?;
                            procs.push(proc);
                        }
                        Allocator2Process::StopAndExit(code) => {
                            tracing::info!("stopping procs with code {code}");
                            for mut proc_to_stop in procs {
                                if let Err(err) = proc_to_stop
                                    .destroy_and_wait(Duration::from_millis(10), None)
                                    .await
                                {
                                    tracing::error!(
                                        "error while stopping proc {}: {}",
                                        proc_to_stop.proc_id(),
                                        err
                                    );
                                }
                            }
                            tracing::info!("exiting with {code}");
                            std::process::exit(code);
                        }
                        Allocator2Process::Exit(code) => {
                            tracing::info!("exiting with {code}");
                            std::process::exit(code);
                        }
                    }
                }
                signal = signals.next() => {
                    if signal.is_some_and(|sig| sig == SIGTERM) {
                        tracing::info!("received SIGTERM, stopping procs");
                        for mut proc_to_stop in procs {
                            if let Err(err) = proc_to_stop
                                .destroy_and_wait(Duration::from_millis(10), None)
                                .await
                            {
                                tracing::error!(
                                    "error while stopping proc {}: {}",
                                    proc_to_stop.proc_id(),
                                    err
                                );
                            }
                        }
                        // SAFETY: We're setting the handle to SigDfl (defautl system behaviour)
                        if let Err(err) = unsafe {
                            nix::sys::signal::signal(nix::sys::signal::SIGTERM, nix::sys::signal::SigHandler::SigDfl)
                        } {
                            tracing::error!("failed to signal SIGTERM: {}", err);
                        }
                        if let Err(err) = nix::sys::signal::raise(nix::sys::signal::SIGTERM) {
                            tracing::error!("failed to raise SIGTERM: {}", err);
                        }
                        std::process::exit(128 + SIGTERM);
                    }
                }
            }
        }
    }

    go().await.unwrap_err()
}

/// A variant of [`bootstrap`] that logs the error and exits the process
/// if bootstrapping fails.
pub async fn bootstrap_or_die() -> ! {
    let err = bootstrap().await;
    tracing::error!("failed to bootstrap mesh process: {}", err);
    std::process::exit(1)
}
