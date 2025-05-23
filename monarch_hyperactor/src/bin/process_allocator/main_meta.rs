/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// std::time::SystemTime::now is used by #[cli::main] suppress the warning since
// it is used around the CLI not in hyperactor core code
#![allow(clippy::disallowed_methods)]

mod common;

use anyhow::Result;
use cli::ExitCode;
use common::Args;
use common::main_impl;
use fbinit::FacebookInit;
use hyperactor::channel::ChannelAddr;
use hyperactor_meta_lib::system_resolution::canonicalize_hostname;
use hyperactor_telemetry::env::Env;
use hyperactor_telemetry::env::HYPERACTOR_EXECUTION_ID_ENV;
use hyperactor_telemetry::env::MAST_HPC_JOB_NAME_ENV;

fn hostname() -> String {
    canonicalize_hostname(
        hostname::get()
            .ok()
            .and_then(|hostname| hostname.into_string().ok())
            .expect("failed to retrieve hostname")
            .as_str(),
    )
}

#[cli::main("process_allocator")]
async fn main(_fb: FacebookInit, args: Args) -> Result<ExitCode> {
    match Env::current() {
        Env::Mast => {
            let job_name =
                std::env::var(MAST_HPC_JOB_NAME_ENV).expect("MAST_HPC_JOB_NAME not set in MAST");
            std::env::set_var(HYPERACTOR_EXECUTION_ID_ENV, job_name);
        }
        _ => {}
    }
    hyperactor::initialize();

    let current_host = hostname();
    tracing::info!(
        "NOTE: argument `--addr` is ignored when running internally at Meta! \
        Process allocator runs on current host: `{}:{}` using Meta-TLS over TCP",
        &current_host,
        args.port
    );
    let serve_address = ChannelAddr::MetaTls(current_host, args.port);
    let result = main_impl(serve_address, args.program).await;

    match result {
        Ok(_) => Ok(ExitCode::SUCCESS),
        Err(e) => {
            tracing::error!("Error running process allocator: {:?}", e);
            Ok(ExitCode::FAILURE)
        }
    }
}
