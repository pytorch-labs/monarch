/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A binary to launch the system, host, or controller actors. Controller actors should be
//! launched through a separate binary that is defined in crate [`monarch_tensor_worker`]
//! due to Python dependency for workers.

use std::os::fd::FromRawFd;
use std::os::fd::RawFd;

use anyhow::Result;
use clap::Parser;
use controller::bootstrap::ControllerServerRequest;
use controller::bootstrap::ControllerServerResponse;
use controller::bootstrap::RunCommand;
use tokio::io::AsyncBufRead;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;

/// Bootstrap commands and arguments used for system, proc, and controller actors.
// TODO: The logic to spawn the hyperactor part of this can probably live in hyperactor
// itself, and we can just call that from here.
#[derive(Parser)]
enum BootstrapCommand {
    #[command(flatten)]
    Run(RunCommand),
    Serve {
        read: RawFd,
        write: RawFd,
    },
}

async fn serve(inp: impl AsyncBufRead + Unpin, mut outp: impl AsyncWrite + Unpin) -> Result<()> {
    tracing::info!("running controller server on {}", std::process::id());

    let mut lines = inp.lines();
    while let Some(line) = lines.next_line().await? {
        let request: ControllerServerRequest = serde_json::from_str(&line)?;
        tracing::info!("got controller request: {:?}", request);
        let response = match serde_json::from_str(&line)? {
            ControllerServerRequest::Run(cmd) => {
                let res = controller::bootstrap::run(cmd)?.await?;
                ControllerServerResponse::Finished {
                    error: match res {
                        Err(err) => Some(format!("{}", err)),
                        Ok(()) => None,
                    },
                }
            }
            ControllerServerRequest::Exit() => break,
        };
        tracing::info!("sending controller response: {:?}", response);
        outp.write_all(format!("{}\n", serde_json::to_string(&response)?).as_bytes())
            .await?;
        outp.flush().await?;
    }

    tracing::info!("finished running controller server");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    hyperactor::initialize();

    match BootstrapCommand::try_parse()? {
        BootstrapCommand::Run(cmd) => controller::bootstrap::run(cmd)?.await??,
        BootstrapCommand::Serve { read, write } => {
            serve(
                // SAFETY: Raw FD passed in from parent.
                BufReader::new(unsafe { tokio::fs::File::from_raw_fd(read) }),
                // SAFETY: Raw FD passed in from parent.
                unsafe { tokio::fs::File::from_raw_fd(write) },
            )
            .await?
        }
    }

    Ok(())
}
