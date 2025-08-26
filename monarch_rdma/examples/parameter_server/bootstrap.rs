/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unused)]
use std::hint::black_box;

use monarch_rdma::RdmaManagerActor;
use parameter_server::ParameterServerActor;
use parameter_server::WorkerActor;

/// This is an "empty shell" bootstrap process,
/// simply invoking [`hyperactor_mesh::bootstrap_or_die`].
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    // The following black_box lines force-link the actors needed for the parameter server
    // example to run. Relying on side-effects for actor registration is not consistent across
    // all build modes.
    let _ = black_box::<Option<RdmaManagerActor>>(None);
    let _ = black_box::<Option<ParameterServerActor>>(None);
    let _ = black_box::<Option<WorkerActor>>(None);
    hyperactor_mesh::bootstrap_or_die().await;
}
