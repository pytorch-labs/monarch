/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use bytes::Bytes;
use hyperactor::Named;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelRx;
use hyperactor::channel::ChannelTransport;
use hyperactor::channel::Rx;
use hyperactor::channel::Tx;
use serde::Deserialize;
use serde::Serialize;
use tokio::time::Duration;
use tokio::time::Instant;

#[derive(Clone, Debug, Named, Serialize, Deserialize)]
struct Message(serde_multipart::Part);

async fn server(
    mut server_rx: ChannelRx<Message>,
    client_addr: ChannelAddr,
) -> Result<(), anyhow::Error> {
    let client_tx = channel::dial(client_addr)?;
    loop {
        let message = server_rx.recv().await?;
        client_tx.post(message);
    }
}

// Analog of https://www.internalfb.com/phabricator/paste/view/P1903314366, using Channel APIs.
// Possibly we should create separate threads for the client and server to also make the OS-level
// setup equivalent.
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), anyhow::Error> {
    let transport = ChannelTransport::Tcp;
    // let transport = ChannelTransport::Local;
    let message_size = 1_000_000;
    let num_iter = 1000;

    let (client_addr, mut client_rx) =
        channel::serve::<Message>(ChannelAddr::any(transport.clone()))
            .await
            .unwrap();
    let (server_addr, server_rx) = channel::serve::<Message>(ChannelAddr::any(transport.clone()))
        .await
        .unwrap();

    let _server_handle = tokio::spawn(server(server_rx, client_addr));

    let server_tx = channel::dial(server_addr)?;

    let message = Message(serde_multipart::Part::from(vec![0u8; message_size]));

    for _ in 0..10 {
        // Warmup
        let t = Instant::now();
        server_tx.post(message.clone() /*cheap */);
        client_rx.recv().await?;
    }

    let mut latencies = vec![];
    let mut total_bytes_sent = 0usize;
    let mut total_bytes_received = 0usize;

    let start = Instant::now();
    for _ in 0..num_iter {
        total_bytes_sent += message.0.len();
        let start = Instant::now();
        server_tx.post(message.clone() /*cheap */);
        total_bytes_received += client_rx.recv().await?.0.len();
        latencies.push(start.elapsed());
    }
    let elapsed = start.elapsed();

    let avg_latency = ((latencies.iter().sum::<Duration>().as_micros() as f64) / 1000f64)
        / (latencies.len() as f64);
    let min_latency = (latencies.iter().min().unwrap().as_micros() as f64) / 1000f64;
    let max_latency = (latencies.iter().max().unwrap().as_micros() as f64) / 1000f64;

    let total_bytes_transferred = total_bytes_sent + total_bytes_received;
    let bandwidth_bytes_per_sec =
        (total_bytes_transferred as f64) / ((elapsed.as_millis() as f64) / 1000f64);
    let bandwidth_mbps = (bandwidth_bytes_per_sec * 8f64) / (1024f64 * 1024f64);

    println!("Results:");
    println!("Average latency: {} ms", avg_latency);
    println!("Min latency: {} ms", min_latency);
    println!("Max latency: {} ms", max_latency);
    println!("Total iterations: {}", latencies.len());
    println!("Total time: {} seconds", elapsed.as_secs());
    println!("Bytes sent: {} bytes", total_bytes_sent);
    println!("Bytes received: {} bytes", total_bytes_received);
    println!("Total bytes transferred: {} bytes", total_bytes_transferred);
    println!(
        "Bandwidth: {} bytes/sec ({} Mbps)",
        bandwidth_bytes_per_sec, bandwidth_mbps
    );

    Ok(())
}
