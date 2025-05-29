/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod multicast;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::PortId;
use hyperactor::data::Serialized;
use ndslice::Slice;
use ndslice::selection::routing::RoutingFrame;
use serde::Deserialize;
use serde::Serialize;

use crate::comm::multicast::CastMessage;
use crate::comm::multicast::CastMessageEnvelope;
use crate::comm::multicast::ForwardMessage;

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
#[hyperactor::export_spawn(CastMessage, ForwardMessage)]
pub struct CommActor {
    /// Each world will use its own seq num from this caster.
    send_seq: HashMap<Slice, usize>,
    /// Each world/caster uses its own stream.
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

    fn handle_message(
        this: &Instance<Self>,
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
        let reply_ports = message.data().get::<PortId>()?;
        if !reply_ports.is_empty() {
            let split_ports = reply_ports
                .iter()
                .map(|p| p.split(this, message.reducer_typehash().clone()))
                .collect::<Vec<_>>();
            message.data_mut().replace::<PortId>(split_ports.iter())?;

            #[cfg(test)]
            tests::collect_split_ports(&reply_ports, &split_ports, deliver_here);
        }

        // Deliever message here, if necessary.
        if deliver_here {
            this.post(
                message.dest_port().port_id(this.self_id().proc_id().rank()),
                Serialized::serialize(message.data())?,
            );
        }

        // Forward to peers.
        next_steps
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
        let (deliver_here, next_steps) =
            ndslice::selection::routing::resolve_routing(rank, dests, &mut |_| {
                panic!("Choice encountered in CommActor routing")
            })?;

        let recv_state = self.recv_state.entry((slice, sender.clone())).or_default();
        match recv_state.seq.cmp(&last_seq) {
            // We got the expected next message to deliver to this host.
            Ordering::Equal => {
                // We got an in-order operation, so handle it now.
                Self::handle_message(
                    this,
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
                        this,
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

// Some tests are located in mod hyperactor_multiprocess/system.rs

// Need to be public since it is used in hyperactor_multiprocess/system.rs
pub mod test_utils {
    // use std::collections::HashMap;
    // use std::collections::HashSet;
    // use std::time::Duration;

    // use anyhow::Context;
    use anyhow::Result;
    use async_trait::async_trait;
    use hyperactor::Actor;
    use hyperactor::ActorHandle;
    use hyperactor::ActorId;
    use hyperactor::Handler;
    use hyperactor::Instance;
    use hyperactor::Named;
    use hyperactor::PortId;
    use hyperactor::PortRef;
    use hyperactor::ProcId;
    use hyperactor::WorldId;
    use hyperactor::id;
    use hyperactor::mailbox::BoxedMailboxSender;
    use hyperactor::mailbox::MailboxRouter;
    use hyperactor::message::Bind;
    use hyperactor::message::Bindings;
    use hyperactor::message::IndexedErasedUnbound;
    use hyperactor::message::Unbind;
    use hyperactor::message::Unbound;
    use hyperactor::proc::Proc;
    use hyperactor::test_utils::proc_supervison::ProcSupervisionCoordinator;
    use serde::Deserialize;
    use serde::Serialize;

    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Named)]
    pub struct MyReply {
        pub sender: ActorId,
        pub value: u64,
    }

    #[derive(Debug, Named, Serialize, Deserialize, PartialEq)]
    #[named(dump = false)]
    pub enum TestMessage {
        Forward(String),
        CastAndReply {
            arg: String,
            reply_to0: PortRef<String>,
            reply_to1: PortRef<u64>,
            reply_to2: PortRef<MyReply>,
        },
    }

    // TODO(pzhang) add macro to auto implement these traits.
    impl Unbind for TestMessage {
        fn unbind(self) -> anyhow::Result<Unbound<Self>> {
            match &self {
                TestMessage::Forward(_) => Ok(Unbound::new(self, Bindings::default())),
                TestMessage::CastAndReply {
                    reply_to1,
                    reply_to2,
                    ..
                } => {
                    let mut bindings = Bindings::default();
                    let ports = [
                        // Intentionally not visiting 0. As a result, this port
                        // will not be split.
                        // reply_to0.port_id().clone(),
                        reply_to1.port_id(),
                        reply_to2.port_id(),
                    ];
                    bindings.insert::<PortId>(ports.into_iter())?;
                    Ok(Unbound::new(self, bindings))
                }
            }
        }
    }

    impl Bind for TestMessage {
        fn bind(mut self, bindings: &Bindings) -> anyhow::Result<Self> {
            match &mut self {
                TestMessage::Forward(_) => Ok(self),
                TestMessage::CastAndReply {
                    reply_to1,
                    reply_to2,
                    ..
                } => {
                    let mut_ports = [
                        // Intentionally not visiting 0. As a result, this port
                        // will not be split.
                        // reply_to0.port_id_mut(),
                        reply_to1.port_id_mut(),
                        reply_to2.port_id_mut(),
                    ];
                    bindings.bind_to(mut_ports.into_iter())?;
                    Ok(self)
                }
            }
        }
    }

    #[derive(Debug)]
    #[hyperactor::export_spawn(TestMessage, IndexedErasedUnbound<TestMessage>)]
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
        async fn handle(&mut self, this: &Instance<Self>, msg: TestMessage) -> anyhow::Result<()> {
            self.forward_port.send(this, msg)?;
            Ok(())
        }
    }

    pub async fn spawn_comm_actors(num: usize) -> Result<Vec<(Proc, ActorHandle<CommActor>)>> {
        let router = MailboxRouter::new();
        spawn_comm_actors_with_router(id!(local), num, &router).await
    }

    pub async fn spawn_comm_actors_with_router(
        world_id: WorldId,
        num: usize,
        router: &MailboxRouter,
    ) -> Result<Vec<(Proc, ActorHandle<CommActor>)>> {
        let mut comms = vec![];
        for idx in 0..num {
            let proc_id = ProcId(world_id.clone(), idx);
            let proc = Proc::new(proc_id, BoxedMailboxSender::new(router.clone()));
            router.bind(proc.proc_id().clone().into(), proc.clone());
            ProcSupervisionCoordinator::set(&proc).await?;

            let comm = proc.spawn::<CommActor>("comm", CommActorParams {}).await?;
            comm.bind::<CommActor>();
            comms.push((proc, comm));
        }

        Ok(comms)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::collections::HashSet;
    use std::fmt::Display;
    use std::hash::Hash;
    use std::ops::DerefMut;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    use hyperactor::ActorHandle;
    use hyperactor::GangId;
    use hyperactor::Mailbox;
    use hyperactor::PortId;
    use hyperactor::PortRef;
    use hyperactor::WorldId;
    use hyperactor::accum;
    use hyperactor::accum::Accumulator;
    use hyperactor::accum::CommReducer;
    use hyperactor::clock::Clock;
    use hyperactor::clock::RealClock;
    use hyperactor::id;
    use hyperactor::mailbox::MailboxRouter;
    use hyperactor::mailbox::PortReceiver;
    use hyperactor::mailbox::open_port;
    use hyperactor::reference::Index;
    use hyperactor::test_utils::tracing::set_tracing_env_filter;
    use maplit::btreemap;
    use maplit::hashmap;
    use ndslice::selection;
    use ndslice::selection::test_utils::collect_commactor_routing_tree;
    use test_utils::*;
    use timed_test::async_timed_test;
    use tokio::time::Duration;
    use tracing::Level;

    use super::*;
    use crate::comm::multicast::DestinationPort;
    use crate::comm::multicast::Uslice;

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
    pub(crate) fn collect_split_ports(original: &[PortId], split: &[PortId], deliver_here: bool) {
        let mutex = SPLIT_PORT_TREE.get_or_init(|| Mutex::new(vec![]));
        let mut tree = mutex.lock().unwrap();

        for (o, s) in original.iter().zip(split.iter()) {
            tree.deref_mut().push(Edge {
                from: o.clone(),
                to: s.clone(),
                is_leaf: deliver_here,
            });
        }
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
    fn get_ranks(
        paths: PathToLeaves<PortId>,
        client_reply: &PortId,
        dest_world: &WorldId,
    ) -> PathToLeaves<Index> {
        let ranks = paths
            .0
            .into_iter()
            .map(|(dst, mut path)| {
                let first = path.remove(0);
                // The first PortId is the client's reply port.
                assert_eq!(&first, client_reply);
                // Other ports's actor ID must be dest[?].comm[0], where ? is
                // the rank we want to extract here.
                assert_eq!(dst.actor_id().proc_id().world_id(), dest_world);
                assert_eq!(dst.actor_id().name(), "comm");
                let actor_path = path
                    .into_iter()
                    .map(|p| {
                        assert_eq!(p.actor_id().proc_id().world_id(), dest_world);
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
        dest_actors_and_caps: Vec<(ActorHandle<TestActor>, Mailbox)>,
        reply1_rx: PortReceiver<u64>,
        reply2_rx: PortReceiver<MyReply>,
        reply_tos: Vec<(PortRef<u64>, PortRef<MyReply>)>,
    }

    // Placeholder to make compiler happy.
    #[derive(Debug, Clone, Serialize, Deserialize, Named)]
    struct NonReducer;
    impl CommReducer for NonReducer {
        type Update = u64;

        fn reduce(&self, _left: Self::Update, _right: Self::Update) -> Self::Update {
            unimplemented!()
        }
    }

    struct NoneAccumulator;

    impl Accumulator for NoneAccumulator {
        type State = u64;
        type Update = u64;
        type Reducer = NonReducer;

        fn accumulate(&self, _state: &mut Self::State, _update: &Self::Update) {
            unimplemented!()
        }
    }

    async fn setup_mesh<A>(dest_world: &WorldId, accum: Option<A>) -> MeshSetup
    where
        A: Accumulator<Update = u64, State = u64> + Send + Sync + 'static,
    {
        tracing::info!("create a client proc and actor, which is used to send cast messages");
        let router = MailboxRouter::new();
        let (client_proc, client_actor) = spawn_comm_actors_with_router(id!(client), 1, &router)
            .await
            .unwrap()
            .pop()
            .expect("no monitor proc and actor is found");

        tracing::info!("create a mesh of destination mesh");
        let slice = Slice::new(0, vec![4, 4, 4], vec![16, 4, 1]).unwrap();

        tracing::debug!("start to spawn procs, comm actors and dest actors",);
        let procs_and_comm_actors =
            spawn_comm_actors_with_router(dest_world.clone(), slice.len(), &router)
                .await
                .unwrap();
        // Spawn a TestActor on each dest proc. These TestActors are the
        // destination actors.
        let dest_actor_name = "dest_actor";
        let mut queues = vec![];
        let mut dest_actors_and_caps = vec![];
        for (proc, _) in procs_and_comm_actors.iter() {
            let caps = proc.attach("dest_user").unwrap();
            let (tx, rx) = open_port(&caps);
            queues.push(rx);
            let test = proc
                .spawn::<TestActor>(
                    dest_actor_name,
                    TestActorParams {
                        forward_port: tx.bind(),
                    },
                )
                .await
                .unwrap();
            test.bind::<TestActor>();
            dest_actors_and_caps.push((test, caps));
        }
        tracing::info!("done with spawning procs, comm actors and dest actos.");

        let client = client_proc.attach("client_user").unwrap();
        let (reply_port_handle0, _) = open_port::<String>(&client);
        let reply_port_ref0 = reply_port_handle0.bind();
        let (reply_port_handle1, reply1_rx) = match accum {
            Some(a) => client.open_accum_port(a),
            None => open_port(&client),
        };
        let reply_port_ref1 = reply_port_handle1.bind();
        let (reply_port_handle2, reply2_rx) = open_port::<MyReply>(&client);
        let reply_port_ref2 = reply_port_handle2.bind();
        // Destination is every node in the world.
        let uslice = Uslice {
            slice,
            selection: selection::dsl::true_(),
        };
        client_actor
            .send(CastMessage {
                // Destination is every node in the world.
                dest: uslice.clone(),
                message: CastMessageEnvelope::new(
                    client_actor.actor_id().clone(),
                    DestinationPort::new::<TestActor, TestMessage>(GangId(
                        dest_world.clone(),
                        dest_actor_name.into(),
                    )),
                    TestMessage::CastAndReply {
                        arg: "abc".to_string(),
                        reply_to0: reply_port_ref0.clone(),
                        reply_to1: reply_port_ref1.clone(),
                        reply_to2: reply_port_ref2.clone(),
                    },
                    None,
                )
                .unwrap(),
            })
            .unwrap();

        tracing::info!("message was cast to dest actors");

        let mut reply_tos = vec![];
        // Verify dest actors received the message, and the reply ports were
        // split as expected.
        for queue in queues.iter_mut() {
            let msg = queue.recv().await.expect("missing");
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
                    assert_eq!(
                        reply_to1.port_id().actor_id().proc_id().world_id(),
                        dest_world
                    );
                    assert_eq!(reply_to1.port_id().actor_id().name(), "comm");
                    assert_ne!(reply_to2, reply_port_ref2);
                    assert_eq!(
                        reply_to2.port_id().actor_id().proc_id().world_id(),
                        dest_world
                    );
                    assert_eq!(reply_to2.port_id().actor_id().name(), "comm");
                    reply_tos.push((reply_to1, reply_to2));
                }
                _ => {
                    panic!("unexpected message: {:?}", msg);
                }
            }
        }
        tracing::info!("message was received by all dest actors");

        // Verify the split port paths are the same as the casting paths.
        {
            // Get the paths used in casting
            let sel_paths = PathToLeaves(
                collect_commactor_routing_tree(&uslice.selection, &uslice.slice)
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
                    get_ranks(PathToLeaves(reply1), reply_port_ref1.port_id(), dest_world),
                    get_ranks(PathToLeaves(reply2), reply_port_ref2.port_id(), dest_world),
                )
            };

            // split port paths should be the same as casting paths
            assert_eq!(sel_paths, reply1_paths);
            assert_eq!(sel_paths, reply2_paths);
        }

        MeshSetup {
            dest_actors_and_caps,
            reply1_rx,
            reply2_rx,
            reply_tos,
        }
    }

    #[async_timed_test(timeout_secs = 30)]
    async fn test_cast_and_reply() {
        set_tracing_env_filter(Level::INFO);
        let dest_world = id!(dest);
        let MeshSetup {
            dest_actors_and_caps,
            mut reply1_rx,
            mut reply2_rx,
            reply_tos,
            ..
        } = setup_mesh::<NoneAccumulator>(&dest_world, None).await;

        // Reply from each dest actor. The replies should be received by client.
        {
            for ((dest_actor, caps), (reply_to1, reply_to2)) in
                dest_actors_and_caps.iter().zip(reply_tos.iter())
            {
                let rank = dest_actor.actor_id().rank() as u64;
                reply_to1.send(caps, rank).unwrap();
                let my_reply = MyReply {
                    sender: dest_actor.actor_id().clone(),
                    value: rank,
                };
                reply_to2.send(caps, my_reply.clone()).unwrap();

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
            for ((dest_actor, caps), (_reply_to1, reply_to2)) in
                dest_actors_and_caps.iter().zip(reply_tos.iter())
            {
                let rank = dest_actor.actor_id().rank();
                let mut sent2 = vec![];
                for i in 0..n {
                    let value = (rank * 100 + i) as u64;
                    let my_reply = MyReply {
                        sender: dest_actor.actor_id().clone(),
                        value,
                    };
                    reply_to2.send(caps, my_reply.clone()).unwrap();
                    sent2.push(my_reply);
                }
                assert!(
                    expected2.insert(rank, sent2).is_none(),
                    "duplicate rank {rank} in map"
                );
            }

            let mut received2: HashMap<usize, Vec<MyReply>> = hashmap! {};

            for _ in 0..(n * dest_actors_and_caps.len()) {
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
        tokio::time::timeout(dur, async {
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
        set_tracing_env_filter(Level::INFO);
        let dest_world = id!(dest);
        let MeshSetup {
            dest_actors_and_caps,
            mut reply1_rx,
            reply_tos,
            ..
        } = setup_mesh(&dest_world, Some(accum::sum::<u64>())).await;

        // Now send multiple replies from the dest actors. They should all be
        // received by client. Replies sent from the same dest actor should
        // be received in the same order as they were sent out.
        {
            let mut sum = 0;
            let n = 100;
            for ((dest_actor, caps), (reply_to1, _reply_to2)) in
                dest_actors_and_caps.iter().zip(reply_tos.iter())
            {
                let rank = dest_actor.actor_id().rank();
                for i in 0..n {
                    let value = (rank + i) as u64;
                    reply_to1.send(caps, value).unwrap();
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
