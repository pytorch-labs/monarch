/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod common;

use clap::Parser;
use common::Args;
use common::main_impl;
use hyperactor::channel::ChannelAddr;

#[tokio::main]
async fn main() {
    let args = Args::parse();
    hyperactor::initialize();

    let bind = format!("{}:{}", args.addr, args.port);
    let socket_addr: std::net::SocketAddr = bind.parse().unwrap();
    let serve_address = ChannelAddr::Tcp(socket_addr);

    let _ = main_impl(serve_address, args.program).await.unwrap();
}
