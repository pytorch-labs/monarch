use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;

use crate as hyperactor; // for macros
use crate::Actor;
use crate::ActorRef;
use crate::Handler;
use crate::Instance;
use crate::Named;
use crate::OncePortRef;
use crate::PortRef;
use crate::mailbox::MessageEnvelope;
use crate::mailbox::Undeliverable;

/// A message that can be passed around. It contains
/// 0. the TTL of this PingPong game
/// 1. the next actor to send the message to
/// 2. a port to send a true value to when TTL = 0.
#[derive(Serialize, Deserialize, Debug, Named)]
pub struct PingPongMessage(pub u64, pub ActorRef<PingPongActor>, pub OncePortRef<bool>);

/// Initialization parameters for `PingPongActor`s.
#[derive(Debug, Named, Serialize, Deserialize, Clone)]
pub struct PingPongActorParams {
    /// A port to send undeliverable messages to.
    undeliverable_port_ref: PortRef<Undeliverable<MessageEnvelope>>,
    /// The TTL at which the actor will exit with error.
    error_ttl: Option<u64>,
}

impl PingPongActorParams {
    /// Create a new set of initialization parameters.
    pub fn new(
        undeliverable_port_ref: PortRef<Undeliverable<MessageEnvelope>>,
        error_ttl: Option<u64>,
    ) -> Self {
        Self {
            undeliverable_port_ref,
            error_ttl,
        }
    }
}

/// A PingPong actor that can play the PingPong game by sending messages around.
#[derive(Debug)]
#[hyperactor::export(PingPongMessage)]
pub struct PingPongActor {
    params: PingPongActorParams,
}

#[async_trait]
impl Actor for PingPongActor {
    type Params = PingPongActorParams;

    async fn new(params: Self::Params) -> Result<Self, anyhow::Error> {
        Ok(Self { params })
    }

    // This is an override of the default actor behavior. It is used
    // for testing the mechanism for returning undeliverable messages to
    // their senders.
    async fn handle_undeliverable_message(
        &mut self,
        this: &Instance<Self>,
        undelivered: crate::mailbox::Undeliverable<crate::mailbox::MessageEnvelope>,
    ) -> Result<(), anyhow::Error> {
        // Forward this undelivered message to the port ref given on
        // construction.
        self.params
            .undeliverable_port_ref
            .send(this, undelivered)
            .unwrap();

        // For the purposes of testing we don't return `Err` here as
        // we normally would for an arbitrary actor. If we did the
        // actor would stop running, transition to the `Failed` state
        // and thereafter no longer receive any undelivered messages.
        Ok(())
    }
}

#[async_trait]
impl Handler<PingPongMessage> for PingPongActor {
    /// Handles the PingPong Message. It will send the message to th actor specified in the
    /// PingPongMessage if TTL > 0. And deliver a true to the done port if TTL = 0.
    /// It also panics if TTL == 66 for testing purpose.
    async fn handle(
        &mut self,
        this: &Instance<Self>,
        PingPongMessage(ttl, pong_actor, done_port): PingPongMessage,
    ) -> anyhow::Result<()> {
        // PingPongActor sends the messages back and forth. When it's ttl = 0, it will stop.
        // User can set a preconfigured TTL that can cause mocked problem: such as an error.
        if Some(ttl) == self.params.error_ttl {
            anyhow::bail!("PingPong handler encountered an Error");
        }
        if ttl == 0 {
            done_port.send(this, true)?;
        } else {
            let next_message = PingPongMessage(ttl - 1, this.bind(), done_port);
            pong_actor.send(this, next_message)?;
        }
        Ok(())
    }
}

hyperactor::remote!(PingPongActor);
