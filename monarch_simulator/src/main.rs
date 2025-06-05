/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A binary to launch the simulated Monarch controller along with necessary environment.
use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;
use hyperactor::channel::ChannelAddr;
use monarch_simulator_lib::bootstrap::bootstrap;

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long)]
    system_addr: ChannelAddr,
    #[arg(short, long)]
    proxy_addr: ChannelAddr,
}

const TITLE: &str = r#"
******************************************************
*                                                    *
* ____ ___ __  __ _   _ _        _  _____ ___  ____  *
*/ ___|_ _|  \/  | | | | |      / \|_   _/ _ \|  _ \ *
*\___ \| || |\/| | | | | |     / _ \ | || | | | |_) |*
* ___) | || |  | | |_| | |___ / ___ \| || |_| |  _ < *
*|____/___|_|  |_|\___/|_____/_/   \_\_| \___/|_| \_\*
*                                                    *
******************************************************
"#;

#[tokio::main]
async fn main() -> Result<ExitCode> {
    eprintln!("{}", TITLE);
    hyperactor::initialize();
    let args = Args::parse();

    let system_addr = args.system_addr.clone();
    let proxy_addr = args.proxy_addr.clone();
    tracing::info!("starting Monarch simulation");

    let operational_listener_handle = bootstrap(system_addr, proxy_addr, 1).await?;

    operational_listener_handle
        .await
        .expect("simulator exited unexpectedly");

    Ok(ExitCode::SUCCESS)
}
