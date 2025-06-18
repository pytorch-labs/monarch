/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::IndexedErasedUnbound;
use hyperactor::message::Unbind;
use serde::Deserialize;
use serde::Serialize;

use crate::actor_mesh::Cast;

/// Message that can be sent to an EmptyActor.
#[derive(Serialize, Deserialize, Debug, Named, Clone)]
pub struct EmptyMessage();

// TODO(pzhang) replace the boilerplate Bind/Unbind impls with a macro.
impl Bind for EmptyMessage {
    fn bind(&mut self, _bindings: &mut Bindings) -> anyhow::Result<()> {
        Ok(())
    }
}

impl Unbind for EmptyMessage {
    fn unbind(&self, _bindings: &mut Bindings) -> anyhow::Result<()> {
        Ok(())
    }
}

/// No-op actor.
#[derive(Debug, PartialEq)]
#[hyperactor::export(
    handlers = [
        EmptyMessage,
        Cast<EmptyMessage>,
        IndexedErasedUnbound<Cast<EmptyMessage>>,
    ],
)]
pub struct EmptyActor();

#[async_trait]
impl Actor for EmptyActor {
    type Params = ();

    async fn new(_: ()) -> Result<Self, anyhow::Error> {
        Ok(Self())
    }
}

#[async_trait]
impl Handler<EmptyMessage> for EmptyActor {
    async fn handle(&mut self, _: &Instance<Self>, _: EmptyMessage) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

#[async_trait]
impl Handler<Cast<EmptyMessage>> for EmptyActor {
    async fn handle(
        &mut self,
        _: &Instance<Self>,
        _: Cast<EmptyMessage>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }
}
hyperactor::remote!(EmptyActor);
