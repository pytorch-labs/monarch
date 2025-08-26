/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Main running script for parameter server example.
//!
//! This script needs to be kept separate to avoid buck naming collisions.
//!
//! Specifically, parameter_server::run uses ProcAllocator, which spawns
//! the binary defined in //monarch/examples/rdma/bootstrap.rs.
//!
//! If this main script was kept in the same file as parameter_server.rs, then
//! spawning the actors defined in parameter_server would be named e.g.
//! "parameter_server_example::ParameterServerActor", whereas the bootstrap binary
//! expects this to be named "parameter_server::ParameterServerActor".
//!
//! Keeping this file separate allows us to avoid this naming collision.
use parameter_server::run;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    run(4, 5).await
}
