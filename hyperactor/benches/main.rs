/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use futures::future::join_all;
use hyperactor::Named;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use hyperactor::channel::Rx;
use hyperactor::channel::Tx;
use hyperactor::channel::dial;
use hyperactor::channel::serve;
use serde::Deserialize;
use serde::Serialize;
use tokio::runtime::Runtime;
use tokio::select;
use tokio::sync::oneshot;

#[derive(Debug, Clone, Serialize, Deserialize, Named, PartialEq)]
struct Message {
    id: u64,
    payload: Vec<u8>,
}

impl Message {
    fn new(id: u64, size: usize) -> Self {
        Self {
            id,
            payload: vec![0; size],
        }
    }
}

// Benchmark message sizes
fn bench_message_sizes(c: &mut Criterion) {
    let transports = vec![
        ("local", ChannelTransport::Local),
        ("tcp", ChannelTransport::Tcp),
        ("unix", ChannelTransport::Unix),
    ];

    for (transport_name, transport) in &transports {
        for size in [10_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000] {
            let mut group = c.benchmark_group(format!("send_receive/{}", transport_name));
            let transport = transport.clone();
            group.throughput(Throughput::Bytes(size as u64));
            group.sampling_mode(criterion::SamplingMode::Flat);
            group.sample_size(10);
            group.bench_function(BenchmarkId::from_parameter(size), move |b| {
                let mut b = b.to_async(Runtime::new().unwrap());
                let tt = &transport;
                b.iter_custom(|iters| async move {
                    let addr = ChannelAddr::any(tt.clone());
                    let (listen_addr, mut rx) = serve::<Message>(addr).await.unwrap();
                    let tx = dial::<Message>(listen_addr).unwrap();
                    let msg = Message::new(0, size);
                    let start = Instant::now();
                    for _ in 0..iters {
                        tx.post(msg.clone());
                        rx.recv().await.unwrap();
                    }
                    start.elapsed()
                });
            });
            group.finish();
        }
    }
}

// Benchmark message rates with a single client
fn bench_message_rates(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_rates");

    let transports = vec![
        ("local", ChannelTransport::Local),
        ("tcp", ChannelTransport::Tcp),
        ("unix", ChannelTransport::Unix),
        //TODO Add TLS once it is able to run in Sandcastle
    ];

    let rates = vec![100, 1000, 5000];

    let payload_size = 1024; // 1KB payload

    for rate in &rates {
        for (transport_name, transport) in &transports {
            let rate = *rate;

            group.bench_function(format!("rate_{}_{}mps", transport_name, rate), move |b| {
                let mut b = b.to_async(Runtime::new().unwrap());
                b.iter_custom(|iters| async move {
                    let total_msgs = iters * rate;
                    let addr = ChannelAddr::any(transport.clone());
                    let (listen_addr, mut rx) = serve::<Message>(addr).await.unwrap();
                    tokio::spawn(async move {
                        let mut received_count = 0;

                        while received_count < total_msgs {
                            match rx.recv().await {
                                Ok(_) => received_count += 1,
                                Err(e) => {
                                    panic!("Error receiving message: {}", e);
                                }
                            }
                        }
                    });

                    let tx = dial::<Message>(listen_addr).unwrap();
                    let message = Message::new(0, payload_size);
                    let start = Instant::now();

                    for _ in 0..iters {
                        let mut response_handlers: Vec<tokio::task::JoinHandle<()>> =
                            Vec::with_capacity(rate as usize);
                        for _ in 0..rate {
                            let (return_sender, return_receiver) = oneshot::channel();
                            if let Err(e) = tx.try_post(message.clone(), return_sender) {
                                panic!("Failed to send message: {:?}", e);
                            }

                            let handle = tokio::spawn(async move {
                                select! {
                                    _ = return_receiver => {},
                                    _ = tokio::time::sleep(Duration::from_millis(5000)) => {
                                        panic!("Did not get ack within timeout");

                                    }
                                }
                            });

                            response_handlers.push(handle);

                            let delay_ms = if rate > 0 { 1000 / rate } else { 0 };
                            let elapsed = start.elapsed().as_millis();
                            let effective_delay = (delay_ms as u128).saturating_sub(elapsed);
                            if effective_delay > 0 {
                                tokio::time::sleep(Duration::from_millis(effective_delay as u64))
                                    .await;
                            }
                        }
                        join_all(response_handlers).await;
                    }

                    start.elapsed()
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_message_sizes, bench_message_rates);

criterion_main!(benches);
