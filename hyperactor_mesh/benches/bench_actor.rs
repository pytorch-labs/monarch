/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::Bind;
use hyperactor::Context;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor::PortRef;
use hyperactor::Unbind;
use hyperactor::clock::Clock;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Deserialize, Named, Bind, Unbind)]
pub struct BenchMessage {
    pub step: usize,
    pub reply: PortRef<usize>,
    #[serde(with = "serde_bytes")]
    pub payload: Vec<u8>,
}

#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [
        BenchMessage { cast = true },
    ],
)]
pub struct BenchActor {}

#[async_trait]
impl Actor for BenchActor {
    type Params = ();

    async fn new(_: Self::Params) -> Result<Self, anyhow::Error> {
        Ok(Self {})
    }
}

#[async_trait]
impl Handler<BenchMessage> for BenchActor {
    async fn handle(
        &mut self,
        ctx: &Context<Self>,
        msg: BenchMessage,
    ) -> Result<(), anyhow::Error> {
        hyperactor::clock::ClockKind::default()
            .sleep(Duration::from_millis(100))
            .await;

        let _ = msg.reply.send(ctx, msg.step);
        Ok(())
    }
}
