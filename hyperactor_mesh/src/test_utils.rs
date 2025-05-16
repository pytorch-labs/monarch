use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use serde::Deserialize;
use serde::Serialize;

use crate::actor_mesh::Cast;

/// Message that can be sent to an EmptyActor.
#[derive(Serialize, Deserialize, Debug, Named, Clone)]
pub struct EmptyMessage();

/// No-op actor.
#[derive(Debug, PartialEq)]
#[hyperactor::export(EmptyMessage, Cast<EmptyMessage>)]
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
