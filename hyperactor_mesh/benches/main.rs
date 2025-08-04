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
use criterion::criterion_group;
use criterion::criterion_main;
use hyperactor_mesh::ProcMesh;
use hyperactor_mesh::actor_mesh::ActorMesh;
use hyperactor_mesh::actor_mesh::RootActorMesh;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::LocalAllocator;
use hyperactor_mesh::selection::dsl::all;
use hyperactor_mesh::selection::dsl::true_;
use hyperactor_mesh::shape;

mod bench_actor;
use bench_actor::BenchActor;
use bench_actor::BenchMessage;
use tokio::runtime::Runtime;

// Benchmark how long does it take to process 1KB message on 1, 10, 100, 1K hosts with 8 GPUs each
fn bench_actor_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("actor_scaling");
    let host_counts = vec![1, 10, 100, 1000];
    let message_size = 1024; // Fixed message size (1KB)
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);

    for host_count in host_counts {
        group.bench_function(BenchmarkId::from_parameter(host_count), |b| {
            let mut b = b.to_async(Runtime::new().unwrap());
            b.iter_custom(|iters| async move {
                let shape = shape! {  hosts=host_count, gpus=8 };
                let alloc = LocalAllocator
                    .allocate(AllocSpec {
                        shape: shape.clone(),
                        constraints: Default::default(),
                    })
                    .await
                    .unwrap();

                let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();
                let trainer_mesh: RootActorMesh<BenchActor> =
                    proc_mesh.spawn("trainer", &()).await.unwrap();
                let client = proc_mesh.client();

                let start = Instant::now();
                for i in 0..iters {
                    let (tx, mut rx) = client.open_port();
                    let payload = vec![0u8; message_size];

                    trainer_mesh
                        .cast(
                            all(true_()),
                            BenchMessage {
                                step: i as usize,
                                reply: tx.bind(),
                                payload,
                            },
                        )
                        .unwrap();

                    let mut msg_rcv = 0;
                    while msg_rcv < host_count {
                        let _ = rx.recv().await.unwrap();
                        msg_rcv += 1;
                    }
                }

                start.elapsed()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_actor_scaling);
criterion_main!(benches);
