/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::comm::multicast::CAST_ORIGINATING_SENDER;
use crate::reference::ActorMeshId;
pub mod multicast;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::Context;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::PortRef;
use hyperactor::WorldId;
use hyperactor::data::Serialized;
use hyperactor::mailbox::DeliveryError;
use hyperactor::mailbox::MailboxSender;
use hyperactor::mailbox::Undeliverable;
use hyperactor::mailbox::UndeliverableMailboxSender;
use hyperactor::mailbox::UndeliverableMessageError;
use hyperactor::mailbox::monitored_return_handle;
use hyperactor::reference::UnboundPort;
use ndslice::selection::routing::RoutingFrame;
use serde::Deserialize;
use serde::Serialize;

use crate::comm::multicast::CastMessage;
use crate::comm::multicast::CastMessageEnvelope;
use crate::comm::multicast::ForwardMessage;
use crate::comm::multicast::set_cast_info_on_headers;

/// Parameters to initialize the CommActor
#[derive(Debug, Clone, Serialize, Deserialize, Named, Default)]
pub struct CommActorParams {}

/// A message buffered due to out-of-order delivery.
#[derive(Debug)]
struct Buffered {
    /// Sequence number of this message.
    seq: usize,
    /// Whether to deliver this message to this comm-actors actors.
    deliver_here: bool,
    /// Peer comm actors to forward message to.
    next_steps: HashMap<usize, Vec<RoutingFrame>>,
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
    /// A map of the last sequence number we sent to next steps, indexed by rank.
    last_seqs: HashMap<usize, usize>,
}

/// This is the comm actor used for efficient and scalable message multicasting
/// and result accumulation.
#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [
        CommActorMode,
        CastMessage,
        ForwardMessage,
    ],
)]
pub struct CommActor {
    /// Sequence numbers are maintained for each (actor mesh id, sender).
    send_seq: HashMap<(ActorMeshId, ActorId), usize>,
    /// Each sender is a unique stream.
    recv_state: HashMap<(ActorMeshId, ActorId), ReceiveState>,

    /// The comm actor's mode.
    mode: CommActorMode,
}

/// Configuration for how a `CommActor` determines its own rank and locates peers.
///
/// - In `Mesh` mode, the comm actor is assigned an explicit rank and a mapping to each peer by rank.
/// - In `Implicit` mode, the comm actor infers its rank and peers from its own actor ID.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub enum CommActorMode {
    /// When configured as a mesh, the comm actor is assigned a rank
    /// and a set of references for each peer rank.
    Mesh(usize, HashMap<usize, ActorRef<CommActor>>),

    /// In an implicit mode, the comm actor derives its rank and
    /// peers from its own ID.
    Implicit,

    /// Like `Implicit`, but override the destination world id.
    /// This is useful for setups where comm actors may not reside
    /// in the destination world. It is meant as a temporary bridge
    /// until we are fully onto ActorMeshes.
    // TODO: T224926642 Remove this once we are fully onto ActorMeshes.
    ImplicitWithWorldId(WorldId),
}

impl Default for CommActorMode {
    fn default() -> Self {
        Self::Implicit
    }
}

impl CommActorMode {
    /// Return the peer comm actor for the given rank, given a self id,
    /// destination port, and rank.
    fn peer_for_rank(&self, self_id: &ActorId, rank: usize) -> Result<ActorRef<CommActor>> {
        match self {
            Self::Mesh(_self_rank, peers) => peers
                .get(&rank)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no peer for rank {}", rank)),
            Self::Implicit => {
                let world_id = self_id
                    .proc_id()
                    .world_id()
                    .ok_or_else(|| anyhow::anyhow!("comm actor must be on a ranked proc"))?;
                let proc_id = world_id.proc_id(rank);
                let actor_id = ActorId::root(proc_id, self_id.name().to_string());
                Ok(ActorRef::<CommActor>::attest(actor_id))
            }
            Self::ImplicitWithWorldId(world_id) => {
                let proc_id = world_id.proc_id(rank);
                let actor_id = ActorId::root(proc_id, self_id.name().to_string());
                Ok(ActorRef::<CommActor>::attest(actor_id))
            }
        }
    }

    /// Return the rank of the comm actor, given a self id.
    fn self_rank(&self, self_id: &ActorId) -> Result<usize> {
        match self {
            Self::Mesh(rank, _) => Ok(*rank),
            Self::Implicit | Self::ImplicitWithWorldId(_) => self_id
                .proc_id()
                .rank()
                .ok_or_else(|| anyhow::anyhow!("comm actor must be on a ranked proc")),
        }
    }
}

#[async_trait]
impl Actor for CommActor {
    type Params = CommActorParams;

    async fn new(_params: Self::Params) -> Result<Self> {
        Ok(Self {
            send_seq: HashMap::new(),
            recv_state: HashMap::new(),
            mode: Default::default(),
        })
    }

    // This is an override of the default actor behavior.
    async fn handle_undeliverable_message(
        &mut self,
        cx: &Instance<Self>,
        undelivered: hyperactor::mailbox::Undeliverable<hyperactor::mailbox::MessageEnvelope>,
    ) -> Result<(), anyhow::Error> {
        let Undeliverable(mut message_envelope) = undelivered;

        // 1. Case delivery failure at a "forwarding" step.
        if let Ok(ForwardMessage { message, .. }) =
            message_envelope.deserialized::<ForwardMessage>()
        {
            let sender = message.sender();
            let return_port = PortRef::attest_message_port(sender);
            message_envelope.set_error(DeliveryError::Multicast(format!(
                "comm actor {} failed to forward the cast message; return to \
                its original sender's port {}",
                cx.self_id(),
                return_port.port_id(),
            )));
            return_port
                .send(cx, Undeliverable(message_envelope.clone()))
                .map_err(|err| {
                    let error = DeliveryError::BrokenLink(format!(
                        "error occured when returning ForwardMessage to the original \
                        sender's port {}; error is: {}",
                        return_port.port_id(),
                        err,
                    ));
                    message_envelope.set_error(error);
                    UndeliverableMessageError::return_failure(&message_envelope)
                })?;
            return Ok(());
        }

        // 2. Case delivery failure at a "deliver here" step.
        if let Some(sender) = message_envelope.headers().get(CAST_ORIGINATING_SENDER) {
            let return_port = PortRef::attest_message_port(sender);
            message_envelope.set_error(DeliveryError::Multicast(format!(
                "comm actor {} failed to deliver the cast message to the dest \
                actor; return to its original sender's port {}",
                cx.self_id(),
                return_port.port_id(),
            )));
            return_port
                .send(cx, Undeliverable(message_envelope.clone()))
                .map_err(|err| {
                    let error = DeliveryError::BrokenLink(format!(
                        "error occured when returning cast message to the original \
                        sender's port {}; error is: {}",
                        return_port.port_id(),
                        err,
                    ));
                    message_envelope.set_error(error);
                    UndeliverableMessageError::return_failure(&message_envelope)
                })?;
            return Ok(());
        }

        // 3. A return of an undeliverable message was itself returned.
        UndeliverableMailboxSender
            .post(message_envelope, /*unused */ monitored_return_handle());
        Ok(())
    }
}

impl CommActor {
    /// Forward the message to the comm actor on the given peer rank.
    fn forward(
        cx: &Instance<Self>,
        mode: &CommActorMode,
        rank: usize,
        message: ForwardMessage,
    ) -> Result<()> {
        let child = mode.peer_for_rank(cx.self_id(), rank)?;
        child.send(cx, message)?;
        Ok(())
    }

    fn handle_message(
        cx: &Context<Self>,
        mode: &CommActorMode,
        deliver_here: bool,
        next_steps: HashMap<usize, Vec<RoutingFrame>>,
        sender: ActorId,
        mut message: CastMessageEnvelope,
        seq: usize,
        last_seqs: &mut HashMap<usize, usize>,
    ) -> Result<()> {
        // Split ports, if any, and update message with new ports. In this
        // way, children actors will reply to this comm actor's ports, instead
        // of to the original ports provided by parent.
        message
            .data_mut()
            .visit_mut::<UnboundPort>(|UnboundPort(port_id, reducer_spec)| {
                let split = port_id.split(cx, reducer_spec.clone())?;

                #[cfg(test)]
                tests::collect_split_port(port_id, &split, deliver_here);

                *port_id = split;
                Ok(())
            })?;

        // Deliver message here, if necessary.
        if deliver_here {
            let rank_on_root_mesh = mode.self_rank(cx.self_id())?;
            let cast_rank = message.relative_rank(rank_on_root_mesh)?;
            let cast_shape = message.shape();
            let mut headers = cx.headers().clone();
            set_cast_info_on_headers(
                &mut headers,
                cast_rank,
                cast_shape.clone(),
                message.sender().clone(),
            );
            cx.post(
                cx.self_id()
                    .proc_id()
                    .actor_id(message.dest_port().actor_name(), 0)
                    .port_id(message.dest_port().port()),
                headers,
                Serialized::serialize(message.data())?,
            );
        }

        // Forward to peers.
        next_steps
            .into_iter()
            .map(|(peer, dests)| {
                let last_seq = last_seqs.entry(peer).or_default();
                Self::forward(
                    cx,
                    mode,
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

#[async_trait]
impl Handler<CommActorMode> for CommActor {
    async fn handle(&mut self, _cx: &Context<Self>, mode: CommActorMode) -> Result<()> {
        self.mode = mode;
        Ok(())
    }
}

// TODO(T218630526): reliable casting for mutable topology
#[async_trait]
impl Handler<CastMessage> for CommActor {
    async fn handle(&mut self, cx: &Context<Self>, cast_message: CastMessage) -> Result<()> {
        // Always forward the message to the root rank of the slice, casting starts from there.
        let slice = cast_message.dest.slice.clone();
        let selection = cast_message.dest.selection.clone();
        let frame = RoutingFrame::root(selection, slice);
        let rank = frame.slice.location(&frame.here)?;
        let seq = self
            .send_seq
            .entry(cast_message.message.stream_key())
            .or_default();
        let last_seq = *seq;
        *seq += 1;
        Self::forward(
            cx,
            &self.mode,
            rank,
            ForwardMessage {
                dests: vec![frame],
                sender: cx.self_id().clone(),
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
    async fn handle(&mut self, cx: &Context<Self>, fwd_message: ForwardMessage) -> Result<()> {
        let ForwardMessage {
            sender,
            dests,
            message,
            seq,
            last_seq,
        } = fwd_message;

        // Resolve/dedup routing frames.
        let rank = self.mode.self_rank(cx.self_id())?;
        let (deliver_here, next_steps) =
            ndslice::selection::routing::resolve_routing(rank, dests, &mut |_| {
                panic!("Choice encountered in CommActor routing")
            })?;

        let recv_state = self.recv_state.entry(message.stream_key()).or_default();
        match recv_state.seq.cmp(&last_seq) {
            // We got the expected next message to deliver to this host.
            Ordering::Equal => {
                // We got an in-order operation, so handle it now.
                Self::handle_message(
                    cx,
                    &self.mode,
                    deliver_here,
                    next_steps,
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
                    next_steps,
                    message,
                }) = recv_state.buffer.remove(&recv_state.seq)
                {
                    Self::handle_message(
                        cx,
                        &self.mode,
                        deliver_here,
                        next_steps,
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
                        next_steps,
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

pub mod test_utils {
    use anyhow::Result;
    use async_trait::async_trait;
    use hyperactor::Actor;
    use hyperactor::ActorId;
    use hyperactor::Bind;
    use hyperactor::Context;
    use hyperactor::Handler;
    use hyperactor::Named;
    use hyperactor::PortRef;
    use hyperactor::Unbind;
    use serde::Deserialize;
    use serde::Serialize;

    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Named)]
    pub struct MyReply {
        pub sender: ActorId,
        pub value: u64,
    }

    #[derive(Debug, Named, Serialize, Deserialize, PartialEq, Clone, Bind, Unbind)]
    pub enum TestMessage {
        Forward(String),
        CastAndReply {
            arg: String,
            // Intentionally not including 0. As a result, this port will not be
            // split.
            // #[binding(include)]
            reply_to0: PortRef<String>,
            #[binding(include)]
            reply_to1: PortRef<u64>,
            #[binding(include)]
            reply_to2: PortRef<MyReply>,
        },
    }

    #[derive(Debug)]
    #[hyperactor::export(
        spawn = true,
        handlers = [
            TestMessage { cast = true },
        ],
    )]
    pub struct TestActor {
        // Forward the received message to this port, so it can be inspected by
        // the unit test.
        forward_port: PortRef<TestMessage>,
    }

    #[derive(Debug, Clone, Named, Serialize, Deserialize)]
    pub struct TestActorParams {
        pub forward_port: PortRef<TestMessage>,
    }

    #[async_trait]
    impl Actor for TestActor {
        type Params = TestActorParams;

        async fn new(params: Self::Params) -> Result<Self> {
            let Self::Params { forward_port } = params;
            Ok(Self { forward_port })
        }
    }

    #[async_trait]
    impl Handler<TestMessage> for TestActor {
        async fn handle(&mut self, cx: &Context<Self>, msg: TestMessage) -> anyhow::Result<()> {
            self.forward_port.send(cx, msg)?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::collections::HashSet;
    use std::fmt::Display;
    use std::hash::Hash;
    use std::ops::DerefMut;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    use hyperactor::PortId;
    use hyperactor::PortRef;
    use hyperactor::accum;
    use hyperactor::accum::Accumulator;
    use hyperactor::accum::ReducerSpec;
    use hyperactor::clock::Clock;
    use hyperactor::clock::RealClock;
    use hyperactor::config;
    use hyperactor::mailbox::PortReceiver;
    use hyperactor::mailbox::open_port;
    use hyperactor::reference::Index;
    use hyperactor_mesh_macros::sel;
    use maplit::btreemap;
    use maplit::hashmap;
    use ndslice::Selection;
    use ndslice::extent;
    use ndslice::selection::test_utils::collect_commactor_routing_tree;
    use test_utils::*;
    use timed_test::async_timed_test;
    use tokio::time::Duration;

    use super::*;
    use crate::ProcMesh;
    use crate::actor_mesh::ActorMesh;
    use crate::actor_mesh::RootActorMesh;
    use crate::alloc::AllocSpec;
    use crate::alloc::Allocator;
    use crate::alloc::LocalAllocator;
    use crate::proc_mesh::SharedSpawnable;

    struct Edge<T> {
        from: T,
        to: T,
        is_leaf: bool,
    }

    impl<T> From<(T, T, bool)> for Edge<T> {
        fn from((from, to, is_leaf): (T, T, bool)) -> Self {
            Self { from, to, is_leaf }
        }
    }

    // The relationship between original ports and split ports. The elements in
    // the tuple are (original port, split port, deliver_here).
    static SPLIT_PORT_TREE: OnceLock<Mutex<Vec<Edge<PortId>>>> = OnceLock::new();

    // Collect the relationships between original ports and split ports into
    // SPLIT_PORT_TREE. This is used by tests to verify that ports are split as expected.
    pub(crate) fn collect_split_port(original: &PortId, split: &PortId, deliver_here: bool) {
        let mutex = SPLIT_PORT_TREE.get_or_init(|| Mutex::new(vec![]));
        let mut tree = mutex.lock().unwrap();

        tree.deref_mut().push(Edge {
            from: original.clone(),
            to: split.clone(),
            is_leaf: deliver_here,
        });
    }

    // A representation of a tree.
    //   * Map's keys are the tree's leafs;
    //   * Map's values are the path from the root to that leaf.
    #[derive(PartialEq)]
    struct PathToLeaves<T>(BTreeMap<T, Vec<T>>);

    // Add a custom Debug trait impl so the result from assert_eq! is readable.
    impl<T: Display> Debug for PathToLeaves<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            fn vec_to_string<T: Display>(v: &[T]) -> String {
                v.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
                    .join(", ")
            }

            for (src, path) in &self.0 {
                write!(f, "{} -> {}\n", src, vec_to_string(path))?;
            }
            Ok(())
        }
    }

    fn build_paths<T: Clone + Eq + Hash + Ord>(edges: &[Edge<T>]) -> PathToLeaves<T> {
        let mut child_parent_map = HashMap::new();
        let mut all_nodes = HashSet::new();
        let mut parents = HashSet::new();
        let mut children = HashSet::new();
        let mut dests = HashSet::new();

        // Build parent map and track all nodes and children
        for Edge { from, to, is_leaf } in edges {
            child_parent_map.insert(to.clone(), from.clone());
            all_nodes.insert(from.clone());
            all_nodes.insert(to.clone());
            parents.insert(from.clone());
            children.insert(to.clone());
            if *is_leaf {
                dests.insert(to.clone());
            }
        }

        // For each leaf, reconstruct path back to root
        let mut result = BTreeMap::new();
        for dest in dests {
            let mut path = vec![dest.clone()];
            let mut current = dest.clone();
            while let Some(parent) = child_parent_map.get(&current) {
                path.push(parent.clone());
                current = parent.clone();
            }
            path.reverse();
            result.insert(dest, path);
        }

        PathToLeaves(result)
    }

    #[test]
    fn test_build_paths() {
        // Given the tree:
        //     0
        //    / \
        //   1   4
        //  / \   \
        // 2   3   5
        let edges: Vec<_> = [
            (0, 1, false),
            (1, 2, true),
            (1, 3, true),
            (0, 4, true),
            (4, 5, true),
        ]
        .into_iter()
        .map(|(from, to, is_leaf)| Edge { from, to, is_leaf })
        .collect();

        let paths = build_paths(&edges);

        let expected = btreemap! {
            2 => vec![0, 1, 2],
            3 => vec![0, 1, 3],
            4 => vec![0, 4],
            5 => vec![0, 4, 5],
        };

        assert_eq!(paths.0, expected);
    }

    //  Given a port tree,
    //     * remove the client port, i.e. the 1st element of the path;
    //     * verify all remaining ports are comm actor ports;
    //     * remove the actor information and return a rank-based tree representation.
    //
    //  The rank-based tree representation is what [collect_commactor_routing_tree] returns.
    //  This conversion enables us to compare the path against [collect_commactor_routing_tree]'s result.
    //
    //      For example, for a 2x2 slice, the port tree could look like:
    //      dest[0].comm[0][1028] -> [client[0].client_user[0][1025], dest[0].comm[0][1028]]
    //      dest[1].comm[0][1028] -> [client[0].client_user[0][1025], dest[0].comm[0][1028], dest[1].comm[0][1028]]
    //      dest[2].comm[0][1028] -> [client[0].client_user[0][1025], dest[0].comm[0][1028], dest[2].comm[0][1028]]
    //      dest[3].comm[0][1028] -> [client[0].client_user[0][1025], dest[0].comm[0][1028], dest[2].comm[0][1028], dest[3].comm[0][1028]]
    //
    //     The result should be:
    //     0 -> 0
    //     1 -> 0, 1
    //     2 -> 0, 2
    //     3 -> 0, 2, 3
    fn get_ranks(paths: PathToLeaves<PortId>, client_reply: &PortId) -> PathToLeaves<Index> {
        let ranks = paths
            .0
            .into_iter()
            .map(|(dst, mut path)| {
                let first = path.remove(0);
                // The first PortId is the client's reply port.
                assert_eq!(&first, client_reply);
                // Other ports's actor ID must be dest[?].comm[0], where ? is
                // the rank we want to extract here.
                assert_eq!(dst.actor_id().name(), "comm");
                let actor_path = path
                    .into_iter()
                    .map(|p| {
                        assert_eq!(p.actor_id().name(), "comm");
                        p.actor_id().rank()
                    })
                    .collect();
                (dst.into_actor_id().rank(), actor_path)
            })
            .collect();
        PathToLeaves(ranks)
    }

    struct MeshSetup {
        actor_mesh: RootActorMesh<'static, TestActor>,
        reply1_rx: PortReceiver<u64>,
        reply2_rx: PortReceiver<MyReply>,
        reply_tos: Vec<(PortRef<u64>, PortRef<MyReply>)>,
    }

    struct NoneAccumulator;

    impl Accumulator for NoneAccumulator {
        type State = u64;
        type Update = u64;

        fn accumulate(
            &self,
            _state: &mut Self::State,
            _update: Self::Update,
        ) -> anyhow::Result<()> {
            unimplemented!()
        }

        fn reducer_spec(&self) -> Option<ReducerSpec> {
            unimplemented!()
        }
    }

    async fn setup_mesh<A>(accum: Option<A>) -> MeshSetup
    where
        A: Accumulator<Update = u64, State = u64> + Send + Sync + 'static,
    {
        let extent = extent!(replica = 4, host = 4, gpu = 4);
        let alloc = LocalAllocator
            .allocate(AllocSpec {
                extent: extent.clone(),
                constraints: Default::default(),
                proc_name: None,
            })
            .await
            .unwrap();

        let proc_mesh = Arc::new(ProcMesh::allocate(alloc).await.unwrap());
        let dest_actor_name = "dest_actor";
        let (tx, mut rx) = hyperactor::mailbox::open_port(proc_mesh.client());
        let params = TestActorParams {
            forward_port: tx.bind(),
        };
        let actor_mesh = proc_mesh
            .clone()
            .spawn::<TestActor>(dest_actor_name, &params)
            .await
            .unwrap();

        let (reply_port_handle0, _) = open_port::<String>(proc_mesh.client());
        let reply_port_ref0 = reply_port_handle0.bind();
        let (reply_port_handle1, reply1_rx) = match accum {
            Some(a) => proc_mesh.client().open_accum_port(a),
            None => open_port(proc_mesh.client()),
        };
        let reply_port_ref1 = reply_port_handle1.bind();
        let (reply_port_handle2, reply2_rx) = open_port::<MyReply>(proc_mesh.client());
        let reply_port_ref2 = reply_port_handle2.bind();
        let message = TestMessage::CastAndReply {
            arg: "abc".to_string(),
            reply_to0: reply_port_ref0.clone(),
            reply_to1: reply_port_ref1.clone(),
            reply_to2: reply_port_ref2.clone(),
        };

        let selection = sel!(*);
        actor_mesh
            .cast(proc_mesh.client(), selection.clone(), message)
            .unwrap();

        let mut reply_tos = vec![];
        for _ in extent.points() {
            let msg = rx.recv().await.expect("missing");
            match msg {
                TestMessage::CastAndReply {
                    arg,
                    reply_to0,
                    reply_to1,
                    reply_to2,
                } => {
                    assert_eq!(arg, "abc");
                    // port 0 is still the same as the original one because it
                    // is not included in MutVisitor.
                    assert_eq!(reply_to0, reply_port_ref0);
                    // ports have been replaced by comm actor's split ports.
                    assert_ne!(reply_to1, reply_port_ref1);
                    assert_eq!(reply_to1.port_id().actor_id().name(), "comm");
                    assert_ne!(reply_to2, reply_port_ref2);
                    assert_eq!(reply_to2.port_id().actor_id().name(), "comm");
                    reply_tos.push((reply_to1, reply_to2));
                }
                _ => {
                    panic!("unexpected message: {:?}", msg);
                }
            }
        }

        // Verify the split port paths are the same as the casting paths.
        {
            // Get the paths used in casting
            let sel_paths = PathToLeaves(
                collect_commactor_routing_tree(&selection, &extent.to_slice())
                    .delivered
                    .into_iter()
                    .collect(),
            );

            // Get the split port paths collected in SPLIT_PORT_TREE during casting
            let (reply1_paths, reply2_paths) = {
                let tree = SPLIT_PORT_TREE.get().unwrap();
                let edges = tree.lock().unwrap();
                let (reply1, reply2): (BTreeMap<_, _>, BTreeMap<_, _>) = build_paths(&edges)
                    .0
                    .into_iter()
                    .partition(|(_dst, path)| &path[0] == reply_port_ref1.port_id());
                (
                    get_ranks(PathToLeaves(reply1), reply_port_ref1.port_id()),
                    get_ranks(PathToLeaves(reply2), reply_port_ref2.port_id()),
                )
            };

            // split port paths should be the same as casting paths
            assert_eq!(sel_paths, reply1_paths);
            assert_eq!(sel_paths, reply2_paths);
        }

        MeshSetup {
            actor_mesh,
            reply1_rx,
            reply2_rx,
            reply_tos,
        }
    }

    #[async_timed_test(timeout_secs = 30)]
    async fn test_cast_and_reply() {
        let MeshSetup {
            actor_mesh,
            mut reply1_rx,
            mut reply2_rx,
            reply_tos,
            ..
        } = setup_mesh::<NoneAccumulator>(None).await;
        let proc_mesh_client = actor_mesh.proc_mesh().client();

        // Reply from each dest actor. The replies should be received by client.
        {
            for (dest_actor, (reply_to1, reply_to2)) in
                actor_mesh.ranks.iter().zip(reply_tos.iter())
            {
                let rank = dest_actor.actor_id().rank() as u64;
                reply_to1.send(proc_mesh_client, rank).unwrap();
                let my_reply = MyReply {
                    sender: dest_actor.actor_id().clone(),
                    value: rank,
                };
                reply_to2.send(proc_mesh_client, my_reply.clone()).unwrap();

                assert_eq!(reply1_rx.recv().await.unwrap(), rank);
                assert_eq!(reply2_rx.recv().await.unwrap(), my_reply);
            }
        }

        tracing::info!("the 1st updates from all dest actors were receivered by client");

        // Now send multiple replies from the dest actors. They should all be
        // received by client. Replies sent from the same dest actor should
        // be received in the same order as they were sent out.
        {
            let n = 100;
            let mut expected2: HashMap<usize, Vec<MyReply>> = hashmap! {};
            for (dest_actor, (_reply_to1, reply_to2)) in
                actor_mesh.ranks.iter().zip(reply_tos.iter())
            {
                let rank = dest_actor.actor_id().rank();
                let mut sent2 = vec![];
                for i in 0..n {
                    let value = (rank * 100 + i) as u64;
                    let my_reply = MyReply {
                        sender: dest_actor.actor_id().clone(),
                        value,
                    };
                    reply_to2.send(proc_mesh_client, my_reply.clone()).unwrap();
                    sent2.push(my_reply);
                }
                assert!(
                    expected2.insert(rank, sent2).is_none(),
                    "duplicate rank {rank} in map"
                );
            }

            let mut received2: HashMap<usize, Vec<MyReply>> = hashmap! {};

            for _ in 0..(n * actor_mesh.ranks.len()) {
                let my_reply = reply2_rx.recv().await.unwrap();
                received2
                    .entry(my_reply.sender.rank())
                    .or_default()
                    .push(my_reply);
            }
            assert_eq!(received2, expected2);
        }
    }

    async fn wait_for_with_timeout(
        receiver: &mut PortReceiver<u64>,
        expected: u64,
        dur: Duration,
    ) -> anyhow::Result<()> {
        // timeout wraps the entire async block
        RealClock
            .timeout(dur, async {
                loop {
                    let msg = receiver.recv().await.unwrap();
                    if msg == expected {
                        break;
                    }
                }
            })
            .await?;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 30)]
    async fn test_cast_and_accum() -> Result<()> {
        let config = config::global::lock();
        // Use temporary config for this test
        let _guard1 = config.override_key(config::SPLIT_MAX_BUFFER_SIZE, 1);

        let MeshSetup {
            actor_mesh,
            mut reply1_rx,
            reply_tos,
            ..
        } = setup_mesh(Some(accum::sum::<u64>())).await;
        let proc_mesh_client = actor_mesh.proc_mesh().client();

        // Now send multiple replies from the dest actors. They should all be
        // received by client. Replies sent from the same dest actor should
        // be received in the same order as they were sent out.
        {
            let mut sum = 0;
            let n = 100;
            for (dest_actor, (reply_to1, _reply_to2)) in
                actor_mesh.ranks.iter().zip(reply_tos.iter())
            {
                let rank = dest_actor.actor_id().rank();
                for i in 0..n {
                    let value = (rank + i) as u64;
                    reply_to1.send(proc_mesh_client, value).unwrap();
                    sum += value;
                }
            }
            wait_for_with_timeout(&mut reply1_rx, sum, Duration::from_secs(2))
                .await
                .unwrap();
            // no more messages
            RealClock.sleep(Duration::from_secs(2)).await;
            let msg = reply1_rx.try_recv().unwrap();
            assert_eq!(msg, None);
        }
        Ok(())
    }
}
