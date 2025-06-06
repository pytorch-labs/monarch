/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::result::Result;

use clap::Parser;
use clap::command;
use hyperactor::channel::ChannelAddr;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAllocator;
use tokio::process::Command;

#[derive(Parser, Debug)]
#[command(about = "Runs hyperactor's process allocator")]
pub struct Args {
    #[arg(
        long,
        default_value_t = 26600,
        help = "The port to bind to on [::] (all network interfaces on this host). Same as specifying `--addr=[::]:{port}`"
    )]
    pub port: u16,

    #[arg(
        long,
        help = "The address to bind to in the form: \
                `{transport}!{address}:{port}` (e.g. `tcp!127.0.0.1:26600`). \
                If specified, `--port` argument is ignored"
    )]
    pub addr: Option<String>,

    #[arg(
        long,
        default_value = "monarch_bootstrap",
        help = "The path to the binary that this process allocator spawns on an `allocate` request"
    )]
    pub program: String,
}

pub fn main_impl(
    serve_address: ChannelAddr,
    program: String,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    tracing::info!("bind address is: {}", serve_address);
    tracing::info!("program to spawn on allocation request: [{}]", &program);

    tokio::spawn(async {
        RemoteProcessAllocator::new()
            .start(Command::new(program), serve_address)
            .await
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use clap::Parser;
    use hyperactor::WorldId;
    use hyperactor::channel::ChannelTransport;
    use hyperactor_mesh::alloc;
    use hyperactor_mesh::alloc::Alloc;
    use hyperactor_mesh::alloc::remoteprocess;
    use ndslice::shape;

    use super::*;

    #[tokio::test]
    async fn test_args_defaults() -> Result<(), anyhow::Error> {
        let args = vec!["process_allocator"];

        let parsed_args = Args::parse_from(args);

        assert_eq!(parsed_args.port, 26600);
        assert_eq!(parsed_args.addr, None);
        assert_eq!(parsed_args.program, "monarch_bootstrap");
        Ok(())
    }

    #[tokio::test]
    async fn test_args() -> Result<(), anyhow::Error> {
        let args = vec![
            "process_allocator",
            "--addr=tcp!127.0.0.1:29501",
            "--program=/bin/echo",
        ];

        let parsed_args = Args::parse_from(args);

        assert_eq!(parsed_args.addr, Some("tcp!127.0.0.1:29501".to_string()));
        assert_eq!(parsed_args.program, "/bin/echo");
        Ok(())
    }

    #[tokio::test]
    async fn test_main_impl() -> Result<(), anyhow::Error> {
        hyperactor::initialize();

        let serve_address = ChannelAddr::any(ChannelTransport::Unix);
        let program = String::from("/bin/date"); // date is usually a unix built-in command
        let server_handle = main_impl(serve_address.clone(), program);

        let spec = alloc::AllocSpec {
            // NOTE: x cannot be more than 1 since we created a single process-allocator server instance!
            shape: shape! { x=1, y=4 },
            constraints: Default::default(),
        };

        let mut initializer = remoteprocess::MockRemoteProcessAllocInitializer::new();
        initializer.expect_initialize_alloc().return_once(move || {
            Ok(vec![remoteprocess::RemoteProcessAllocHost {
                hostname: serve_address.to_string(),
                id: serve_address.to_string(),
            }])
        });

        let heartbeat = std::time::Duration::from_millis(100);
        let world_id = WorldId("__unused__".to_string());

        let mut alloc = remoteprocess::RemoteProcessAlloc::new(
            spec.clone(),
            world_id,
            ChannelTransport::Unix,
            0,
            heartbeat,
            initializer,
        )
        .await
        .unwrap();

        // make sure we accounted for `world_size` number of Created and Stopped proc states
        let world_size = spec.shape.slice().iter().count();
        let mut created_ranks: HashSet<usize> = HashSet::new();
        let mut stopped_ranks: HashSet<usize> = HashSet::new();

        while created_ranks.len() < world_size || stopped_ranks.len() < world_size {
            let proc_state = alloc.next().await.unwrap();
            match proc_state {
                alloc::ProcState::Created { proc_id, coords: _ } => {
                    // alloc.next() will keep creating procs and incrementing rank id
                    // so we mod the rank by world_size to map it to its logical rank
                    created_ranks.insert(proc_id.rank() % world_size);
                }
                alloc::ProcState::Stopped { proc_id, reason: _ } => {
                    stopped_ranks.insert(proc_id.rank() % world_size);
                }
                _ => {}
            }
        }

        let expected_ranks: HashSet<usize> = (0..world_size).collect();
        assert_eq!(created_ranks, expected_ranks);
        assert_eq!(stopped_ranks, expected_ranks);

        server_handle.abort();
        Ok(())
    }
}
