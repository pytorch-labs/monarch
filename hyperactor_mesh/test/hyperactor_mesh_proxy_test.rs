/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

use anyhow::Result;
use async_trait::async_trait;
use clap::Parser;
use hyperactor::Actor;
use hyperactor::Context;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor::PortRef;
use hyperactor_mesh::Mesh;
use hyperactor_mesh::ProcMesh;
use hyperactor_mesh::RootActorMesh;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::ProcessAllocator;
use ndslice::extent;
use serde::Deserialize;
use serde::Serialize;
use tokio::process::Command;

pub fn initialize() {
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("failed to set subscriber");

    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| {
        #[cfg(target_os = "linux")]
        linux::initialize();
    });
}

#[cfg(target_os = "linux")]
mod linux {
    use std::backtrace::Backtrace;
    use std::process;

    use nix::sys::signal::SigHandler;
    use nix::unistd::getpid;
    use tokio::signal::unix::SignalKind;
    use tokio::signal::unix::signal;

    pub(crate) fn initialize() {
        // Safety: Because I want to
        unsafe {
            extern "C" fn handle_fatal_signal(signo: libc::c_int) {
                let bt = Backtrace::force_capture();
                let signame = nix::sys::signal::Signal::try_from(signo).expect("unknown signal");
                tracing::error!("stacktrace"= %bt, "fatal signal {signo}:{signame} received");
                std::process::exit(1);
            }
            nix::sys::signal::signal(
                nix::sys::signal::SIGABRT,
                SigHandler::Handler(handle_fatal_signal),
            )
            .expect("unable to register signal handler");
            nix::sys::signal::signal(
                nix::sys::signal::SIGSEGV,
                SigHandler::Handler(handle_fatal_signal),
            )
            .expect("unable to register signal handler");
        }

        // Set up the async signal handler FIRST
        let rt = tokio::runtime::Handle::current();
        rt.spawn(async {
            // Set up signal handler before prctl
            let mut sigusr1 = match signal(SignalKind::user_defined1()) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("failed to set up SIGUSR1 signal handler: {:?}", err);
                    return;
                }
            };

            // SAFETY: Now set PDEATHSIG after handler is ready. This
            // is unsafe.
            unsafe {
                if libc::prctl(
                    libc::PR_SET_PDEATHSIG,
                    nix::sys::signal::SIGUSR1 as libc::c_ulong,
                ) != 0
                {
                    eprintln!(
                        "prctl(PR_SET_PDEATHSIG) failed: {}",
                        std::io::Error::last_os_error()
                    );
                    return;
                }

                // Close the race: if parent already died, we are now orphaned.
                if libc::getppid() == 1 {
                    tracing::error!(
                        "hyperactor[{}]: parent already dead on startup; exiting",
                        getpid()
                    );
                    std::process::exit(1);
                }
            }

            // Wait for the signal
            sigusr1.recv().await;
            tracing::error!(
                "hyperactor[{}]: parent process died (SIGUSR1 received); exiting",
                getpid()
            );
            process::exit(1);
        });
    }
}

#[derive(Parser)]
struct Args {
    /// Run bootstrap logic
    #[arg(long)]
    bootstrap: bool,
}

// -- TestActor

#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [
        Echo,
    ],
)]
pub struct TestActor;

#[async_trait]
impl Actor for TestActor {
    type Params = ();

    async fn new(_params: Self::Params) -> Result<Self, anyhow::Error> {
        Ok(Self)
    }
}

#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct Echo(pub String, pub PortRef<String>);

#[async_trait]
impl Handler<Echo> for TestActor {
    async fn handle(&mut self, cx: &Context<Self>, message: Echo) -> Result<(), anyhow::Error> {
        let Echo(message, reply_port) = message;
        reply_port.send(cx, message)?;
        Ok(())
    }
}

// -- ProxyActor

#[hyperactor::export(
    spawn = true,
    handlers = [
        Echo,
    ],
)]
pub struct ProxyActor {
    #[allow(dead_code)]
    proc_mesh: Arc<ProcMesh>,
    actor_mesh: RootActorMesh<'static, TestActor>,
}

impl fmt::Debug for ProxyActor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProxyActor")
            .field("proc_mesh", &"...")
            .field("actor_mesh", &"...")
            .finish()
    }
}

#[async_trait]
impl Actor for ProxyActor {
    type Params = String;

    async fn new(exe_path: Self::Params) -> anyhow::Result<Self, anyhow::Error> {
        let mut cmd = Command::new(PathBuf::from(&exe_path));
        cmd.arg("--bootstrap");

        let mut allocator = ProcessAllocator::new(cmd);

        let alloc = allocator
            .allocate(AllocSpec {
                extent: extent! { replica = 1 },
                constraints: Default::default(),
            })
            .await
            .unwrap();
        let proc_mesh = Arc::new(ProcMesh::allocate(alloc).await.unwrap());
        let leaked: &'static Arc<ProcMesh> = Box::leak(Box::new(proc_mesh));
        let actor_mesh: RootActorMesh<'static, TestActor> =
            leaked.spawn("echo", &()).await.unwrap();
        Ok(Self {
            proc_mesh: Arc::clone(leaked),
            actor_mesh,
        })
    }
}

#[async_trait]
impl Handler<Echo> for ProxyActor {
    async fn handle(&mut self, cx: &Context<Self>, message: Echo) -> Result<(), anyhow::Error> {
        let actor = self.actor_mesh.get(0).unwrap();

        let (tx, mut rx) = cx.open_port();
        actor.send(cx, Echo(message.0, tx.bind()))?;
        message.1.send(cx, rx.recv().await.unwrap())?;

        Ok(())
    }
}

async fn run_client(exe_path: PathBuf) -> Result<(), anyhow::Error> {
    let mut cmd = Command::new(PathBuf::from(&exe_path));
    cmd.arg("--bootstrap");

    let mut allocator = ProcessAllocator::new(cmd);
    let alloc = allocator
        .allocate(AllocSpec {
            extent: extent! { replica = 1 },
            constraints: Default::default(),
        })
        .await
        .unwrap();

    let mut proc_mesh = ProcMesh::allocate(alloc).await?;
    let actor_mesh: RootActorMesh<'_, ProxyActor> = proc_mesh
        .spawn("proxy", &exe_path.to_str().unwrap().to_string())
        .await?;
    let proxy_actor = actor_mesh.get(0).unwrap();
    let (tx, mut rx) = actor_mesh.open_port::<String>();
    proxy_actor.send(proc_mesh.client(), Echo("hello!".to_owned(), tx.bind()))?;

    let msg = rx.recv().await?;
    println!("{}", msg);
    assert_eq!(msg, "hello!");

    let mut alloc = proc_mesh.events().unwrap().into_alloc();
    alloc.stop_and_wait().await?;
    drop(alloc);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Logs are written to /tmp/$USER/monarch_log*.
    initialize();

    let args = Args::parse();
    if args.bootstrap {
        hyperactor_mesh::bootstrap_or_die().await;
    } else {
        let exe_path: PathBuf = env::current_exe().unwrap_or_else(|e| {
            eprintln!("Failed to get current executable path: {}", e);
            std::process::exit(1);
        });
        run_client(exe_path).await?;
    }

    Ok(())
}
