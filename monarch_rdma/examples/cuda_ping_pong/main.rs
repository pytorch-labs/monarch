/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Main running script for CUDA ping pong example.
//!
//! This script needs to be kept separate to avoid buck naming collisions.
//!
//! Specifically, cuda_ping_pong::run uses ProcAllocator, which spawns
//! the binary defined in //monarch/monarch_rdma/examples/cuda_ping_pong/cuda_ping_pong_bootstrap.rs.
//!
//! If this main script was kept in the same file as cuda_ping_pong_example.rs, then
//! spawning the actors defined in cuda_ping_pong_example would be named e.g.
//! "cuda_ping_pong::CudaRdmaActor", whereas the bootstrap binary
//! expects this to be named "cuda_ping_pong::CudaRdmaActor".
//!
//! Keeping this file separate allows us to avoid this naming collision.
use cuda_ping_pong::run;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    run().await
}
