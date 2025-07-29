/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
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
        ("metatls", ChannelTransport::MetaTls),
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

criterion_group!(benches, bench_message_sizes);

criterion_main!(benches);
