/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::io::BufRead;
use std::io::Write;
use std::os::fd::RawFd;
use std::time::Duration;

use anyhow::Result;
use anyhow::anyhow;
use hyperactor::ActorHandle;
use hyperactor::ProcId;
use hyperactor::WorldId;
use hyperactor::actor::ActorStatus;
use hyperactor::channel::ChannelAddr;
use hyperactor_multiprocess::proc_actor::ProcActor;
use hyperactor_multiprocess::system_actor::ProcLifecycleMode;
use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::Deserialize;
use serde::Serialize;

use crate::pipe::OutOfProcessSetupParams;
use crate::pipe::Pipe;
use crate::pipe::StreamPipe;
use crate::py_pipe::PyPipe;
use crate::py_pipe::run_py_pipe;

#[derive(clap::Parser)]
pub enum BinaryArgs {
    /// Starts running a worker server.
    WorkerServer { rd: RawFd, wr: RawFd },
    /// Starts running a worker.
    Worker(WorkerBootstrapArgs),
    /// Starts running a pipe.
    Pipe,
}

/// Bootstrap arguments used for bootstrapping workers for both Python and Rust entry points.
// TODO: We might want to convert these arguments to environment variables depending on
// how we launch the processes in the end.
#[derive(Debug, clap::Args)]
pub struct WorkerBootstrapArgs {
    /// The world id of the launched worker.
    #[arg(long)]
    pub world_id: WorldId,

    /// The proc id of the launched worker.
    #[arg(long)]
    pub proc_id: ProcId,

    /// The system address for the worker to connect to.
    #[arg(long)]
    pub bootstrap_addr: ChannelAddr,

    /// The supervision update interval for worker proc actor.
    #[arg(long, default_value_t = 5)]
    pub supervision_update_interval_in_sec: u64,

    /// Proc metadata which will be available through system.
    /// Keys are not allowed to contain '='.
    #[clap(long, value_parser=parse_key_val)]
    extra_proc_labels: Option<Vec<(String, String)>>,
}

/// Bootstrap the worker proc and join the system at `bootstrap_addr`.
/// The actual worker actor is spawned by the corresponding controller.
/// The worker Python dependencies need to be packaged
/// separately and loaded during the runtime.
pub async fn bootstrap_worker_proc(
    args: WorkerBootstrapArgs,
) -> Result<ActorHandle<ProcActor>, anyhow::Error> {
    let labels: HashMap<String, String> = match args.extra_proc_labels {
        Some(extra_lables) => extra_lables.into_iter().collect(),
        _ => HashMap::new(),
    };

    tracing::info!(
        "bootstrap worker proc {} in world {} with labels: {:?}",
        args.proc_id,
        args.world_id,
        labels
    );
    let bootstrap = ProcActor::bootstrap(
        args.proc_id,
        args.world_id,
        ChannelAddr::any(args.bootstrap_addr.transport()),
        args.bootstrap_addr.clone(),
        Duration::from_secs(args.supervision_update_interval_in_sec),
        labels,
        ProcLifecycleMode::ManagedBySystem,
    )
    .await?;

    Ok(bootstrap.proc_actor)
}

pub fn bootstrap_pipe() -> Result<(), anyhow::Error> {
    // Use a temp pipe just to ship the init params.
    // Value of 4 is arbitrary as our side does not need to do buffering.
    let mut pipe = StreamPipe::new(std::io::stdin(), std::io::stdout(), 4);
    let init: OutOfProcessSetupParams = pipe.recv()?;
    // Create a PyPipe that allows unsafe object conversion. This allows the pipe to
    // receive tensors, which we know is safe because StreamPipe receives the serialized
    // tensors from out-of-process, and they therefore can't be owned by anything except
    // the pipe's python code.
    run_py_pipe(
        PyPipe::new(Box::new(pipe), init.ranks, init.sizes, true),
        init.function,
        init.args,
        init.kwargs,
    )?;

    Ok(())
}

fn parse_key_val(s: &str) -> anyhow::Result<(String, String)> {
    match s.split_once('=') {
        None => Err(anyhow::anyhow!("invalid KEY=value: no `=` found in `{s}`")),
        Some((a, b)) => Ok((a.to_owned(), b.to_owned())),
    }
}

#[pyclass(
    frozen,
    module = "monarch._rust_bindings.monarch_tensor_worker.bootstrap"
)]
#[derive(Debug, Serialize, Deserialize)]
pub enum WorkerServerRequest {
    Run {
        world_id: String,
        proc_id: String,
        bootstrap_addr: String,
        labels: Vec<(String, String)>,
    },
    Exit(),
}

#[pymethods]
impl WorkerServerRequest {
    fn to_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(self).map_err(|e| anyhow!(e))?)
    }

    fn __str__(&self) -> String {
        format!("{:?}", self)
    }
}

#[pyclass(
    frozen,
    module = "monarch._rust_bindings.monarch_tensor_worker.bootstrap"
)]
#[derive(Debug, Serialize, Deserialize)]
pub enum WorkerServerResponse {
    Finished { error: Option<String> },
}

#[pymethods]
impl WorkerServerResponse {
    #[classmethod]
    fn from_json(_: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        Ok(serde_json::from_str(json).map_err(|e| anyhow!(e))?)
    }

    fn __str__(&self) -> String {
        format!("{:?}", self)
    }
}

pub fn worker_server(inp: impl BufRead, mut outp: impl Write) -> Result<()> {
    hyperactor::initialize();

    tracing::info!("running worker server on {}", std::process::id());

    for line in inp.lines() {
        let line = line?;
        let request: WorkerServerRequest = serde_json::from_str(&line)?;
        tracing::info!("got worker request: {:?}", request);
        let response = match serde_json::from_str(&line)? {
            WorkerServerRequest::Run {
                world_id,
                proc_id,
                bootstrap_addr,
                labels,
            } => {
                let args = WorkerBootstrapArgs {
                    world_id: world_id.parse()?,
                    proc_id: proc_id.parse()?,
                    bootstrap_addr: bootstrap_addr.parse()?,
                    supervision_update_interval_in_sec: 5,
                    extra_proc_labels: Some(labels),
                };
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()?;
                let res = rt
                    .block_on(async move { anyhow::Ok(bootstrap_worker_proc(args).await?.await) });
                WorkerServerResponse::Finished {
                    error: match res {
                        Err(err) => Some(format!("{}", err)),
                        Ok(ActorStatus::Stopped) => None,
                        Ok(status) => Some(format!("unexpected actor status: {}", status)),
                    },
                }
            }
            WorkerServerRequest::Exit() => break,
        };
        tracing::info!("sending worker response: {:?}", response);
        writeln!(outp, "{}", &serde_json::to_string(&response)?)?;
    }

    tracing::info!("finished running worker server");

    // TODO(agallagher): Forcing an exit here saves 700ms on shutdown for some
    // reasons -- does this avoid some slow Python shutdown code?
    //Ok(())
    std::process::exit(0);
}

pub fn register_python_bindings(worker_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    worker_mod.add_class::<WorkerServerRequest>()?;
    worker_mod.add_class::<WorkerServerResponse>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use hyperactor::channel::ChannelTransport;
    use hyperactor::id;
    use hyperactor_multiprocess::System;
    use timed_test::async_timed_test;

    use super::*;

    #[async_timed_test(timeout_secs = 60)]
    async fn test_worker_bootstrap() {
        let server_handle = System::serve(
            ChannelAddr::any(ChannelTransport::Local),
            Duration::from_secs(10),
            Duration::from_secs(10),
        )
        .await
        .unwrap();

        let world_id = id!(worker);
        let proc_id = world_id.proc_id(0);
        let proc_handle = bootstrap_worker_proc(WorkerBootstrapArgs {
            world_id,
            proc_id,
            bootstrap_addr: server_handle.local_addr().clone(),
            supervision_update_interval_in_sec: 5,
            extra_proc_labels: None,
        })
        .await
        .unwrap();

        proc_handle.drain_and_stop().unwrap();
    }

    #[test]
    fn test_parse_key_val_valid_input() {
        let s = "key=value";
        assert_eq!(
            parse_key_val(s).unwrap(),
            ("key".to_string(), "value".to_string())
        );
    }

    #[test]
    fn test_parse_key_val_extra_equals() {
        let s = "key=value=3";
        assert_eq!(
            parse_key_val(s).unwrap(),
            ("key".to_string(), "value=3".to_string())
        );
    }

    #[test]
    fn test_parse_key_val_invalid() {
        let s = "invalid";
        assert!(parse_key_val(s).is_err());
    }
}
