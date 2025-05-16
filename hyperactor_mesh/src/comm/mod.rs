pub mod multicast;

use std::cmp::Ordering;
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
use ndslice::Slice;
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

/// A message buffered due to out-of-order delivery.
#[derive(Debug)]
struct Buffered {
    /// Sequence number of this message.
    seq: usize,
    /// Whether to deliver this message to this comm-actors actors.
    deliver_here: bool,
    /// Peer comm actors to forward message to.
    next_hops: HashMap<usize, Vec<RoutingFrame>>,
    /// The message to deliver.
    message: CastMessageEnvelope,
}

/// Bookkeeping to handle sequence numbers and in-order delivery for messages
/// sent to and through this comm actor.
#[derive(Debug, Default)]
struct ReceiveState {
    /// The sequence of the last received message.
    seq: usize,
    /// A buffer storing messages we received out-of-order, indexed by the seq
    /// that should precede it.
    buffer: HashMap<usize, Buffered>,
    /// A map of the last sequence number we sent to next hops, indexed by rank.
    last_seqs: HashMap<usize, usize>,
}

/// This is the comm actor used for efficient and scalable message multicasting
/// and result accumulation.
#[derive(Debug)]
#[hyperactor::export(CastMessage, ForwardMessage)]
pub struct CommActor {
    /// Each world will use it's own seq num from this caster.
    send_seq: HashMap<Slice, usize>,
    /// Each world/castor uses it's own stream.
    recv_state: HashMap<(Slice, ActorId), ReceiveState>,
}

#[async_trait]
impl Actor for CommActor {
    type Params = CommActorParams;

    async fn new(_params: Self::Params) -> Result<Self> {
        Ok(Self {
            send_seq: HashMap::new(),
            recv_state: HashMap::new(),
        })
    }
}

impl CommActor {
    /// Forward the message to the comm actor on the given peer rank.
    fn forward(this: &Instance<Self>, rank: usize, message: ForwardMessage) -> Result<()> {
        let world_id = message.message.dest_port().gang_id().world_id();
        let proc_id = world_id.proc_id(rank);
        let actor_id = ActorId::root(proc_id, this.self_id().name().to_string());
        let comm_actor = ActorRef::<CommActor>::attest(actor_id);
        let port = comm_actor.port::<ForwardMessage>();
        port.send(this, message)?;
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

    fn handle_message(
        this: &Instance<Self>,
        deliver_here: bool,
        next_hops: HashMap<usize, Vec<RoutingFrame>>,
        sender: ActorId,
        message: CastMessageEnvelope,
        seq: usize,
        last_seqs: &mut HashMap<usize, usize>,
    ) -> Result<()> {
        // Deliever message here, if necessary.
        if deliver_here {
            this.post(
                message.dest_port().port_id(this.self_id().proc_id().rank()),
                message.data().clone(),
            );
        }

        // Forward to peers.
        next_hops
            .into_iter()
            .map(|(peer, dests)| {
                let last_seq = last_seqs.entry(peer).or_default();
                Self::forward(
                    this,
                    peer,
                    ForwardMessage {
                        dests,
                        sender: sender.clone(),
                        message: message.clone(),
                        seq,
                        last_seq: *last_seq,
                    },
                )?;
                *last_seq = seq;
                Ok(())
            })
            .collect::<Result<Vec<_>>>()?;

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
        let rank = frame.slice.location(&frame.here)?;
        let seq = self
            .send_seq
            .entry(frame.slice.as_ref().clone())
            .or_default();
        let last_seq = *seq;
        *seq += 1;
        Self::forward(
            this,
            rank,
            ForwardMessage {
                dests: vec![frame],
                sender: this.self_id().clone(),
                message: cast_message.message,
                seq: *seq,
                last_seq,
            },
        )?;
        Ok(())
    }
}

#[async_trait]
impl Handler<ForwardMessage> for CommActor {
    async fn handle(&mut self, this: &Instance<Self>, fwd_message: ForwardMessage) -> Result<()> {
        let ForwardMessage {
            sender,
            dests,
            message,
            seq,
            last_seq,
        } = fwd_message;

        // Resolve/dedup routing frames.
        let rank = this.self_id().proc_id().rank();
        let slice = dests[0].slice.as_ref().clone();
        let (deliver_here, next_hops) = self.resolve_routing(rank, dests)?;

        let recv_state = self.recv_state.entry((slice, sender.clone())).or_default();
        match recv_state.seq.cmp(&last_seq) {
            // We got the expected next message to deliver to this host.
            Ordering::Equal => {
                // We got an in-order operation, so handle it now.
                Self::handle_message(
                    this,
                    deliver_here,
                    next_hops,
                    sender.clone(),
                    message,
                    seq,
                    &mut recv_state.last_seqs,
                )?;
                recv_state.seq = seq;

                // Also deliver any pending operations from the recv buffer that
                // were received out-of-order that are now unblocked.
                while let Some(Buffered {
                    seq,
                    deliver_here,
                    next_hops,
                    message,
                }) = recv_state.buffer.remove(&recv_state.seq)
                {
                    Self::handle_message(
                        this,
                        deliver_here,
                        next_hops,
                        sender.clone(),
                        message,
                        seq,
                        &mut recv_state.last_seqs,
                    )?;
                    recv_state.seq = seq;
                }
            }
            // We got an out-of-order operation, so buffer it for now, until we
            // recieved the onces sequenced before it.
            Ordering::Less => {
                tracing::warn!(
                    "buffering out-of-order message with seq {} (last {}), expected {}: {:?}",
                    seq,
                    last_seq,
                    recv_state.seq,
                    message
                );
                recv_state.buffer.insert(
                    last_seq,
                    Buffered {
                        seq,
                        deliver_here,
                        next_hops,
                        message,
                    },
                );
            }
            // We already got this message -- just drop it.
            Ordering::Greater => {
                tracing::warn!("received duplicate message with seq {}: {:?}", seq, message);
            }
        }

        Ok(())
    }
}

// Tests are located in mod hyperactor_multiprocess/system.rs
