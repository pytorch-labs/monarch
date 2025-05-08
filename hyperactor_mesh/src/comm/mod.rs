pub mod multicast;

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::ControlFlow;

use anyhow::Result;
use async_trait::async_trait;
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
    fn forward_single(
        &self,
        this: &Instance<Self>,
        rank: usize,
        dests: Vec<RoutingFrame>,
        message: CastMessageEnvelope,
    ) -> Result<()> {
        let world_id = message.dest_port.gang_id.world_id();
        let proc_id = world_id.proc_id(rank);
        let actor_id = ActorId::root(proc_id, this.self_id().name().to_string());
        let comm_actor = ActorRef::<CommActor>::attest(actor_id);
        let port = comm_actor.port::<ForwardMessage>();
        port.send(this, ForwardMessage { dests, message })?;
        Ok(())
    }

    fn get_next_hops(&self, dest: RoutingFrame) -> Result<Vec<RoutingFrame>> {
        let mut seen = HashSet::new();
        let mut unique_hops = vec![];
        dest.next_steps(
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
        Ok(unique_hops)
    }

    // Recursively resolve next hops for the given routing frame for the given
    // rank, returning a tuple of whether to deliver to the current node and
    // frames to forward to peer ranks.
    fn resolve_routing_one(
        &self,
        rank: usize,
        frame: RoutingFrame,
        deliver_here: &mut bool,
        next_hops: &mut HashMap<usize, Vec<RoutingFrame>>,
    ) -> Result<()> {
        let frame_rank = frame.slice.location(&frame.here)?;
        if frame_rank == rank {
            if frame.deliver_here() {
                *deliver_here = true;
            } else {
                for frame in self.get_next_hops(frame)?.into_iter() {
                    self.resolve_routing_one(rank, frame, deliver_here, next_hops)?;
                }
            }
        } else {
            next_hops.entry(frame_rank).or_default().push(frame);
        }
        Ok(())
    }

    fn resolve_routing(
        &self,
        rank: usize,
        frames: impl IntoIterator<Item = RoutingFrame> + Debug,
    ) -> Result<(bool, HashMap<usize, Vec<RoutingFrame>>)> {
        let mut deliver_here = false;
        let mut next_hops = HashMap::new();
        for frame in frames.into_iter() {
            self.resolve_routing_one(rank, frame, &mut deliver_here, &mut next_hops)?;
        }
        Ok((deliver_here, next_hops))
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
        let rank = frame.slice.location(&frame.here)?;
        self.forward_single(this, rank, vec![frame], cast_message.message)?;
        Ok(())
    }
}

#[async_trait]
impl Handler<ForwardMessage> for CommActor {
    async fn handle(&mut self, this: &Instance<Self>, fwd_message: ForwardMessage) -> Result<()> {
        let ForwardMessage { dests, message } = fwd_message;

        // Resolve/dedup routing frames.
        let rank = this.self_id().proc_id().rank();
        let (deliver_here, next_hops) = self.resolve_routing(rank, dests)?;

        // Deliever message here, if necessary.
        if deliver_here {
            this.post(
                message.dest_port.port_id(this.self_id().proc_id().rank()),
                message.data.clone(),
            );
        }

        // Forward to peers.
        next_hops
            .into_iter()
            .map(|(peer, dests)| self.forward_single(this, peer, dests, message.clone()))
            .collect::<Result<Vec<_>>>()?;

        Ok(())
    }
}

// Tests are located in mod hyperactor_multiprocess/system.rs
