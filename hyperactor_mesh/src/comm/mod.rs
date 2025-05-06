pub mod multicast;

use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::ControlFlow;

use anyhow::Result;
use anyhow::ensure;
use async_trait::async_trait;
use futures::future::try_join_all;
use hyperactor::Actor;
use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use ndslice::selection::NormalizedSelectionKey;
use ndslice::selection::routing::RoutingFrame;
use ndslice::selection::routing::RoutingFrameKey;
use serde::Deserialize;
use serde::Serialize;

use crate::comm::multicast::CastMessage;
use crate::comm::multicast::CastMessageEnvelope;
use crate::comm::multicast::ForwardMessage;
use crate::selection::routing::RoutingStep;

/// Parameters to initialize the CommActor
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub struct CommActorParams {}

/// This is the comm actor used for efficient and scalable message multicasting
/// and result accumulation.
#[derive(Debug, Clone)]
#[hyperactor::export(CastMessage, ForwardMessage)]
pub struct CommActor {}

#[async_trait]
impl Actor for CommActor {
    type Params = CommActorParams;

    async fn new(_params: Self::Params) -> Result<Self> {
        Ok(Self {})
    }
}

impl CommActor {
    /// Forward the message to the comm actor on the given peer rank.
    async fn forward_single(
        &self,
        this: &Instance<Self>,
        rank: usize,
        dest: RoutingFrame,
        message: CastMessageEnvelope,
    ) -> Result<()> {
        let world_id = message.dest_port.gang_id.world_id();
        let proc_id = world_id.proc_id(rank);
        let actor_id = ActorId::root(proc_id, this.self_id().name().to_string());
        let comm_actor = ActorRef::<CommActor>::attest(actor_id);
        let port = comm_actor.port::<ForwardMessage>();
        port.send(this, ForwardMessage { dest, message })?;
        Ok(())
    }

    /// Forward a message to a set of peer nodes.
    async fn forward(
        &self,
        this: &Instance<Self>,
        ranks: impl IntoIterator<Item = RoutingFrame> + Debug,
        message: &CastMessageEnvelope,
    ) -> Result<()> {
        try_join_all(
            ranks
                .into_iter()
                .map(async |dest| {
                    self.forward_single(this, dest.location().unwrap(), dest, message.clone())
                        .await
                })
                // Don't short-circuit, but still propagate errors.
                .collect::<Vec<_>>(),
        )
        .await?;
        Ok(())
    }
}

// TODO(T218630526): reliable casting for mutable topology
#[async_trait]
impl Handler<CastMessage> for CommActor {
    async fn handle(&mut self, this: &Instance<Self>, cast_message: CastMessage) -> Result<()> {
        // Always forward the message to the root rank of the slice, casting starts from there.
        let slice = cast_message.dest.slice.clone();
        let selection = cast_message.dest.selection.clone();
        let frame = RoutingFrame::root(selection, slice);

        self.forward(this, [frame].into_iter(), &cast_message.message)
            .await?;
        Ok(())
    }
}

#[async_trait]
impl Handler<ForwardMessage> for CommActor {
    async fn handle(&mut self, this: &Instance<Self>, fwd_message: ForwardMessage) -> Result<()> {
        // Ensure we're handling a message for the correct rank
        let rank = fwd_message
            .dest
            .slice
            .location(&fwd_message.dest.here)
            .unwrap();
        ensure!(
            fwd_message.message.dest_port.gang_id.world_id() == this.self_id().proc_id().world_id()
        );
        ensure!(rank == this.self_id().proc_id().rank());

        if fwd_message.dest.deliver_here() {
            this.post(
                fwd_message
                    .message
                    .dest_port
                    .port_id(this.self_id().proc_id().rank()),
                fwd_message.message.data.clone(),
            );
        } else {
            let mut seen = HashSet::new();
            let mut unique_hops = vec![];

            fwd_message.dest.next_steps(
                &mut |_| panic!("Choice encountered in CommActor routing"),
                &mut |step| {
                    if let RoutingStep::Forward(frame) = step {
                        let key = RoutingFrameKey::new(
                            frame.here.clone(),
                            frame.dim,
                            NormalizedSelectionKey::new(&frame.selection),
                        );
                        if seen.insert(key) {
                            unique_hops.push(frame);
                        }
                    }
                    ControlFlow::Continue(())
                },
            );

            self.forward(this, unique_hops.into_iter(), &fwd_message.message)
                .await?;
        }

        Ok(())
    }
}

// Tests are located in mod hyperactor_multiprocess/system.rs
