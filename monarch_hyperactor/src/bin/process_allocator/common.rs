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
use tokio::time::Duration;

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

    #[arg(
        long,
        help = "If specified, a timeout for the allocator to wait before exiting. Unspecified means no timeout"
    )]
    pub timeout_sec: Option<u64>,
}

pub fn main_impl(
    serve_address: ChannelAddr,
    program: Command,
    timeout: Option<Duration>,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    tracing::info!("bind address is: {}", serve_address);
    tracing::info!("program to spawn on allocation request: [{:?}]", &program);

    tokio::spawn(async move {
        RemoteProcessAllocator::new()
            .start(program, serve_address, timeout)
            .await
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use clap::Parser;
    use hyperactor::WorldId;
    use hyperactor::channel::ChannelTransport;
    use hyperactor::clock::Clock;
    use hyperactor::clock::RealClock;
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
        hyperactor::initialize_with_current_runtime();

        let serve_address = ChannelAddr::any(ChannelTransport::Unix);
        let program = Command::new("/bin/date"); // date is usually a unix built-in command
        let server_handle = main_impl(serve_address.clone(), program, None);

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
                alloc::ProcState::Created { proc_id, .. } => {
                    // alloc.next() will keep creating procs and incrementing rank id
                    // so we mod the rank by world_size to map it to its logical rank
                    created_ranks.insert(proc_id.rank() % world_size);
                }
                alloc::ProcState::Stopped { proc_id, .. } => {
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

    /// Tests that an allocator with a timeout and no messages will exit and not
    /// finish allocating.
    #[tokio::test]
    async fn test_timeout() -> Result<(), anyhow::Error> {
        hyperactor::initialize_with_current_runtime();

        let serve_address = ChannelAddr::any(ChannelTransport::Unix);
        let program = Command::new("/bin/date"); // date is usually a unix built-in command
        // 1 second quick timeout to check that it fails.
        let timeout = Duration::from_millis(500);
        let server_handle = main_impl(serve_address.clone(), program, Some(timeout));

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

        // Wait at least as long as the timeout before sending any messages.
        RealClock.sleep(timeout * 2).await;

        // Attempt to allocate, it should fail because a timeout happens before
        let mut alloc = remoteprocess::RemoteProcessAlloc::new(
            spec.clone(),
            world_id.clone(),
            ChannelTransport::Unix,
            0,
            heartbeat,
            initializer,
        )
        .await
        .unwrap();
        let res = alloc.next().await.unwrap();
        // Should fail because the allocator timed out.
        if let alloc::ProcState::Failed {
            world_id: msg_world_id,
            description,
        } = res
        {
            assert_eq!(msg_world_id, world_id);
            assert!(description.contains("no process has ever been allocated"));
        } else {
            panic!("Unexpected ProcState: {:?}", res);
        }

        server_handle.abort();
        Ok(())
    }

    /// Tests that an allocator with a timeout and some messages will still exit
    /// after the allocation finishes.
    #[tokio::test]
    async fn test_timeout_after_message() -> Result<(), anyhow::Error> {
        hyperactor::initialize_with_current_runtime();

        let serve_address = ChannelAddr::any(ChannelTransport::Unix);
        let program = Command::new("/bin/date"); // date is usually a unix built-in command
        // Slower timeout so we can send a message in time.
        let timeout = Duration::from_millis(1500);
        let server_handle = main_impl(serve_address.clone(), program, Some(timeout));

        let spec = alloc::AllocSpec {
            // NOTE: x cannot be more than 1 since we created a single process-allocator server instance!
            shape: shape! { x=1, y=4 },
            constraints: Default::default(),
        };

        let mut initializer = remoteprocess::MockRemoteProcessAllocInitializer::new();
        let alloc_host = remoteprocess::RemoteProcessAllocHost {
            hostname: serve_address.to_string(),
            id: serve_address.to_string(),
        };
        let alloc_host_clone = alloc_host.clone();
        initializer
            .expect_initialize_alloc()
            .return_once(move || Ok(vec![alloc_host_clone]));

        let heartbeat = std::time::Duration::from_millis(100);
        let world_id = WorldId("__unused__".to_string());

        // Attempt to allocate, it should succeed because a timeout happens before
        let mut alloc = remoteprocess::RemoteProcessAlloc::new(
            spec.clone(),
            world_id.clone(),
            ChannelTransport::Unix,
            0,
            heartbeat,
            initializer,
        )
        .await
        .unwrap();
        // Ensure the process starts.
        alloc.next().await.unwrap();
        // Now stop the alloc and wait for a timeout to ensure the allocator exited.
        alloc.stop_and_wait().await.unwrap();

        // Wait at least as long as the timeout before sending any messages.
        RealClock.sleep(timeout * 2).await;

        // Allocate again to see the error.
        let mut initializer = remoteprocess::MockRemoteProcessAllocInitializer::new();
        initializer
            .expect_initialize_alloc()
            .return_once(move || Ok(vec![alloc_host]));
        let mut alloc = remoteprocess::RemoteProcessAlloc::new(
            spec.clone(),
            world_id.clone(),
            ChannelTransport::Unix,
            0,
            heartbeat,
            initializer,
        )
        .await
        .unwrap();
        let res = alloc.next().await.unwrap();
        // Should fail because the allocator timed out.
        if let alloc::ProcState::Failed {
            world_id: msg_world_id,
            description,
        } = res
        {
            assert_eq!(msg_world_id, world_id);
            assert!(description.contains("no process has ever been allocated"));
        } else {
            panic!("Unexpected ProcState: {:?}", res);
        }

        server_handle.abort();
        Ok(())
    }

    /// Tests that an allocator with a timeout, that has a process running and
    /// receives no messages, will keep running as long as the processes do.
    #[tokio::test]
    async fn test_timeout_not_during_execution() -> Result<(), anyhow::Error> {
        hyperactor::initialize_with_current_runtime();

        let serve_address = ChannelAddr::any(ChannelTransport::Unix);
        let mut program = Command::new("/usr/bin/sleep"); // use a command that waits for a while
        program.arg("3");
        let timeout = Duration::from_millis(500);
        let server_handle = main_impl(serve_address.clone(), program, Some(timeout));

        let spec = alloc::AllocSpec {
            // NOTE: x cannot be more than 1 since we created a single process-allocator server instance!
            shape: shape! { x=1, y=4 },
            constraints: Default::default(),
        };

        let mut initializer = remoteprocess::MockRemoteProcessAllocInitializer::new();
        let alloc_host = remoteprocess::RemoteProcessAllocHost {
            hostname: serve_address.to_string(),
            id: serve_address.to_string(),
        };
        initializer
            .expect_initialize_alloc()
            .return_once(move || Ok(vec![alloc_host]));

        let heartbeat = std::time::Duration::from_millis(100);
        let world_id = WorldId("__unused__".to_string());

        // Attempt to allocate, it should succeed because a timeout happens before
        let mut alloc = remoteprocess::RemoteProcessAlloc::new(
            spec.clone(),
            world_id.clone(),
            ChannelTransport::Unix,
            0,
            heartbeat,
            initializer,
        )
        .await
        .unwrap();
        // Ensure the process starts. Since the command is "sleep", it should
        // start without stopping.
        // make sure we accounted for `world_size` number of Created and Stopped proc states
        let world_size = spec.shape.slice().iter().count();
        let mut created_ranks: HashSet<usize> = HashSet::new();

        while created_ranks.len() < world_size {
            let proc_state = alloc.next().await.unwrap();
            match proc_state {
                alloc::ProcState::Created { proc_id, .. } => {
                    created_ranks.insert(proc_id.rank());
                }
                _ => {
                    panic!("Unexpected message: {:?}", proc_state)
                }
            }
        }
        // Now that all procs have started, wait at least as long as the timeout
        // before sending any messages. This way we ensure the remote allocator
        // stays alive as long as the child processes stay alive.
        RealClock.sleep(timeout * 2).await;
        // Now wait for more events and ensure they are ProcState::Stopped
        let mut stopped_ranks: HashSet<usize> = HashSet::new();
        while stopped_ranks.len() < world_size {
            let proc_state = alloc.next().await.unwrap();
            match proc_state {
                alloc::ProcState::Created { .. } => {
                    // ignore
                }
                alloc::ProcState::Stopped { proc_id, .. } => {
                    stopped_ranks.insert(proc_id.rank() % world_size);
                }
                _ => {
                    panic!("Unexpected message: {:?}", proc_state)
                }
            }
        }
        server_handle.abort();
        Ok(())
    }
}
