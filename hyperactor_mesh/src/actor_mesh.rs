/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(dead_code)] // until used publically

use std::ops::Deref;
use std::sync::Arc;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorRef;
use hyperactor::Message;
use hyperactor::Named;
use hyperactor::PortHandle;
use hyperactor::RemoteHandles;
use hyperactor::RemoteMessage;
use hyperactor::actor::RemoteActor;
use hyperactor::mailbox::MailboxSenderError;
use hyperactor::mailbox::PortReceiver;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::Castable;
use hyperactor::message::IndexedErasedUnbound;
use hyperactor::message::Unbind;
use ndslice::Range;
use ndslice::Selection;
use ndslice::Shape;
use ndslice::ShapeError;
use ndslice::Slice;
use ndslice::dsl;
use ndslice::selection::ReifyView;
use serde::Deserialize;
use serde::Serialize;

use crate::Mesh;
use crate::comm::multicast::CastMessage;
use crate::comm::multicast::CastMessageEnvelope;
use crate::comm::multicast::CastRank;
use crate::comm::multicast::DestinationPort;
use crate::comm::multicast::Uslice;
use crate::metrics;
use crate::proc_mesh::ProcMesh;

/// A mesh of actors, all of which reside on the same [`ProcMesh`].
pub trait ActorMesh: Mesh {
    /// The type of actor in the mesh.
    type Actor: RemoteActor;

    /// Cast an [`M`]-typed message to the ranks selected by `sel`
    /// in this ActorMesh.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `CastError`.
    fn cast<M: Castable + Clone>(&self, selection: Selection, message: M) -> Result<(), CastError>
    where
        Self::Actor: RemoteHandles<Cast<M>> + RemoteHandles<IndexedErasedUnbound<Cast<M>>>,
    {
        let _ = metrics::ACTOR_MESH_CAST_DURATION.start(hyperactor::kv_pairs!(
            "message_type" => M::typename(),
            "message_variant" => message.arm().unwrap_or_default(),
        ));

        let message = Cast {
            rank: CastRank(usize::MAX),
            shape: self.shape().clone(),
            message,
        };
        let message = CastMessageEnvelope::new(
            self.proc_mesh().client().actor_id().clone(),
            DestinationPort::new::<Self::Actor, Cast<M>>(self.name().to_string()),
            message,
            None, // TODO: reducer typehash
        )?;

        // Sub-set the selection to the selection that represents the mesh's view
        // of the root mesh. We need to do this because the comm actor uses the
        // slice as the stream key; thus different sub-slices will result in potentially
        // out of order delivery.
        //
        // TODO: We should repair this by introducing an explicit stream key, associated
        // with the root mesh.
        let selection_of_slice = self
            .shape()
            .slice()
            .reify_view(self.shape().slice())
            .expect("invalid slice");
        let selection = dsl::intersection(selection, selection_of_slice);

        self.proc_mesh().comm_actor().send(
            self.proc_mesh().client(),
            CastMessage {
                dest: Uslice {
                    // TODO: currently this slice is being used as the stream key
                    // in comm actor. We should change it to an explicit id, maintained
                    // by the root proc mesh.
                    slice: self.proc_mesh().shape().slice().clone(),
                    selection,
                },
                message,
            },
        )?;

        Ok(())
    }

    /// The ProcMesh on top of which this actor mesh is spawned.
    fn proc_mesh(&self) -> &ProcMesh;

    /// The name global name of actors in this mesh.
    fn name(&self) -> &str;
}

/// Abstracts over shared and borrowed references to a [`ProcMesh`].
/// Given a shared ProcMesh, we can obtain a [`ActorMesh<'static, _>`]
/// for it, useful when lifetime must be managed dynamically.
enum ProcMeshRef<'a> {
    /// The reference is shared with an [`Arc`].
    Shared(Arc<ProcMesh>),
    /// The reference is borrowed with a parameterized
    /// lifetime.
    Borrowed(&'a ProcMesh),
}

impl Deref for ProcMeshRef<'_> {
    type Target = ProcMesh;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Shared(p) => p,
            Self::Borrowed(p) => p, // p: &ProcMesh
        }
    }
}

/// A mesh of actor instances. ActorMeshes are obtained by spawning an
/// actor on a [`ProcMesh`].
pub struct RootActorMesh<'a, A: RemoteActor> {
    proc_mesh: ProcMeshRef<'a>,
    name: String,
    pub(crate) ranks: Vec<ActorRef<A>>, // temporary until we remove `ArcActorMesh`.
}

impl<'a, A: RemoteActor> RootActorMesh<'a, A> {
    pub(crate) fn new(proc_mesh: &'a ProcMesh, name: String, ranks: Vec<ActorRef<A>>) -> Self {
        Self {
            proc_mesh: ProcMeshRef::Borrowed(proc_mesh),
            name,
            ranks,
        }
    }

    pub(crate) fn new_shared(
        proc_mesh: Arc<ProcMesh>,
        name: String,
        ranks: Vec<ActorRef<A>>,
    ) -> Self {
        Self {
            proc_mesh: ProcMeshRef::Shared(proc_mesh),
            name,
            ranks,
        }
    }

    /// Open a port on this ActorMesh.
    pub(crate) fn open_port<M: Message>(&self) -> (PortHandle<M>, PortReceiver<M>) {
        self.proc_mesh.client().open_port()
    }

    /// Until the selection logic is more powerful, we need a way to
    /// replicate the send patterns that the worker actor mesh actually does.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `CastError`.
    pub fn cast_slices<M: RemoteMessage + Clone>(
        &self,
        sel: Vec<Slice>,
        message: M,
    ) -> Result<(), CastError>
    where
        A: RemoteHandles<Cast<M>> + RemoteHandles<IndexedErasedUnbound<Cast<M>>>,
    {
        let _ = metrics::ACTOR_MESH_CAST_DURATION.start(hyperactor::kv_pairs!(
            "message_type" => M::typename(),
            "message_variant" => message.arm().unwrap_or_default(),
        ));
        for ref slice in sel {
            for rank in slice.iter() {
                let cast = Cast {
                    rank: CastRank(rank),
                    shape: self.shape().clone(),
                    message: message.clone(),
                };
                self.ranks[rank]
                    .send(self.proc_mesh.client(), cast)
                    .map_err(|err| CastError::MailboxSenderError(rank, err))?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl<'a, A: RemoteActor> Mesh for RootActorMesh<'a, A> {
    type Node = ActorRef<A>;
    type Sliced<'b>
        = SlicedActorMesh<'b, A>
    where
        'a: 'b;

    fn shape(&self) -> &Shape {
        self.proc_mesh.shape()
    }

    fn select<R: Into<Range>>(
        &self,
        label: &str,
        range: R,
    ) -> Result<Self::Sliced<'_>, ShapeError> {
        Ok(SlicedActorMesh(self, self.shape().select(label, range)?))
    }

    fn get(&self, rank: usize) -> Option<ActorRef<A>> {
        self.ranks.get(rank).cloned()
    }
}

impl<A: RemoteActor> ActorMesh for RootActorMesh<'_, A> {
    type Actor = A;

    fn proc_mesh(&self) -> &ProcMesh {
        &self.proc_mesh
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub struct SlicedActorMesh<'a, A: RemoteActor>(&'a RootActorMesh<'a, A>, Shape);

impl<'a, A: RemoteActor> SlicedActorMesh<'a, A> {
    pub fn new(actor_mesh: &'a RootActorMesh<'a, A>, shape: Shape) -> Self {
        Self(actor_mesh, shape)
    }

    pub fn shape(&self) -> &Shape {
        &self.1
    }
}

#[async_trait]
impl<A: RemoteActor> Mesh for SlicedActorMesh<'_, A> {
    type Node = ActorRef<A>;
    type Sliced<'b>
        = SlicedActorMesh<'b, A>
    where
        Self: 'b;

    fn shape(&self) -> &Shape {
        &self.1
    }

    fn select<R: Into<Range>>(
        &self,
        label: &str,
        range: R,
    ) -> Result<Self::Sliced<'_>, ShapeError> {
        Ok(Self(self.0, self.1.select(label, range)?))
    }

    fn get(&self, _index: usize) -> Option<ActorRef<A>> {
        unimplemented!()
    }
}

impl<A: RemoteActor> ActorMesh for SlicedActorMesh<'_, A> {
    type Actor = A;

    fn proc_mesh(&self) -> &ProcMesh {
        &self.0.proc_mesh
    }

    fn name(&self) -> &str {
        &self.0.name
    }
}

/// A message wrapper used to deliver an `M`-typed payload to a single
/// destination within an [`ActorMesh`].
///
/// `Cast<M>` is the per-recipient form of a broadcast or multicast
/// issued via [`ActorMesh::cast`]. It carries the message payload
/// along with the destination rank and its mesh coordinates.
///
/// `Cast<M>` implements [`Bind`] and [`Unbind`] generically, allowing
/// bindings to propagate through both the payload and the routing
/// metadata.
///
/// Actors that wish to receive routed `M`-typed messages should
/// implement handlers for `Cast<M>`.
#[derive(Debug, Serialize, Deserialize)]
pub struct Cast<M> {
    /// The rank of the receiving actor.
    pub rank: CastRank,
    /// The coordinates of the receiving actor in the actor mesh.
    pub shape: Shape,
    /// The message itself.
    pub message: M,
}

impl<M: Unbind> Unbind for Cast<M> {
    fn bindings(&self) -> anyhow::Result<Bindings> {
        let mut bindings = self.message.bindings()?;
        bindings.push(&self.rank)?;
        Ok(bindings)
    }
}

impl<M: Bind> Bind for Cast<M> {
    fn bind(mut self, bindings: &Bindings) -> anyhow::Result<Self> {
        self.message = self.message.bind(bindings)?;
        bindings.rebind([&mut self.rank].into_iter())?;
        Ok(self)
    }
}

impl<M: Named> Named for Cast<M> {
    fn typename() -> &'static str {
        hyperactor::intern_typename!(Self, "hyperactor_mesh::actor_mesh::Cast<{}>", M)
    }
}

/// The type of error of casting operations.
#[derive(Debug, thiserror::Error)]
pub enum CastError {
    #[error("invalid selection {0}: {1}")]
    InvalidSelection(Selection, ShapeError),

    #[error("send on rank {0}: {1}")]
    MailboxSenderError(usize, MailboxSenderError),

    #[error(transparent)]
    RootMailboxSenderError(#[from] MailboxSenderError),

    #[error(transparent)]
    ShapeError(#[from] ShapeError),

    #[error(transparent)]
    SerializationError(#[from] bincode::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// This has to be compiled outside of test mode because the bootstrap binary
// is not built in test mode, and requires access to TestActor.
pub(crate) mod test_util {
    use std::collections::VecDeque;

    use anyhow::ensure;
    use hyperactor::Handler;
    use hyperactor::Instance;
    use hyperactor::PortRef;
    use hyperactor::message::Bind;
    use hyperactor::message::Unbind;

    use super::*;

    // This can't be defined under a `#[cfg(test)]` because there needs to
    // be an entry in the spawnable actor registry in the executable
    // 'hyperactor_mesh_test_bootstrap' for the `tests::process` actor
    // mesh test suite.
    #[derive(Debug)]
    #[hyperactor::export_spawn(
        Cast<Echo>,
        Cast<GetRank>,
        Cast<Error>,
        GetRank,
        Relay,
        IndexedErasedUnbound<Cast<Echo>>,
        IndexedErasedUnbound<Cast<GetRank>>,
        IndexedErasedUnbound<Cast<Error>>,
    )]
    pub struct TestActor;

    #[async_trait]
    impl Actor for TestActor {
        type Params = ();

        async fn new(_params: Self::Params) -> Result<Self, anyhow::Error> {
            Ok(Self)
        }
    }

    /// Request message to retrieve the actor's rank.
    ///
    /// The `bool` in the tuple controls the outcome of the handler:
    /// - If `true`, the handler will send the rank and return
    ///   `Ok(())`.
    /// - If `false`, the handler will still send the rank, but return
    ///   an error (`Err(...)`).
    ///
    /// This is useful for testing both successful and failing
    /// responses from a single message type.
    #[derive(Debug, Serialize, Deserialize, Named, Clone)]
    pub struct GetRank(pub bool, pub PortRef<usize>);

    // TODO(pzhang) replace the boilerplate Bind/Unbind impls with a macro.
    impl Bind for GetRank {
        fn bind(mut self, bindings: &Bindings) -> anyhow::Result<Self> {
            let mut_ports = [self.1.port_id_mut()];
            bindings.rebind(mut_ports.into_iter())?;
            Ok(self)
        }
    }

    impl Unbind for GetRank {
        fn bindings(&self) -> anyhow::Result<Bindings> {
            let mut bindings = Bindings::default();
            let ports = [self.1.port_id()];
            bindings.insert(ports)?;
            Ok(bindings)
        }
    }

    #[async_trait]
    impl Handler<GetRank> for TestActor {
        async fn handle(
            &mut self,
            this: &Instance<Self>,
            GetRank(ok, reply): GetRank,
        ) -> Result<(), anyhow::Error> {
            let rank = this.self_id().rank();
            reply.send(this, rank)?;
            anyhow::ensure!(ok, "intentional error!"); // If `!ok` exit with `Err()`.
            Ok(())
        }
    }

    #[async_trait]
    impl Handler<Cast<GetRank>> for TestActor {
        async fn handle(
            &mut self,
            this: &Instance<Self>,
            Cast {
                rank,
                message: GetRank(ok, reply),
                ..
            }: Cast<GetRank>,
        ) -> Result<(), anyhow::Error> {
            reply.send(this, *rank)?;
            anyhow::ensure!(ok, "intentional error!"); // If `!ok` exit with `Err()`.
            Ok(())
        }
    }

    #[derive(Debug, Serialize, Deserialize, Named, Clone)]
    pub struct Echo(pub String, pub PortRef<String>);

    // TODO(pzhang) replace the boilerplate Bind/Unbind impls with a macro.
    impl Bind for Echo {
        fn bind(mut self, bindings: &Bindings) -> anyhow::Result<Self> {
            let mut_ports = [self.1.port_id_mut()];
            bindings.rebind(mut_ports.into_iter())?;
            Ok(self)
        }
    }

    impl Unbind for Echo {
        fn bindings(&self) -> anyhow::Result<Bindings> {
            let mut bindings = Bindings::default();
            let ports = [self.1.port_id()];
            bindings.insert(ports)?;
            Ok(bindings)
        }
    }

    #[async_trait]
    impl Handler<Cast<Echo>> for TestActor {
        async fn handle(
            &mut self,
            this: &Instance<Self>,
            Cast { message, .. }: Cast<Echo>,
        ) -> Result<(), anyhow::Error> {
            let Echo(message, reply_port) = message;
            reply_port.send(this, message)?;
            Ok(())
        }
    }

    #[derive(Debug, Serialize, Deserialize, Named, Clone)]
    pub struct Error(pub String);

    // TODO(pzhang) replace the boilerplate Bind/Unbind impls with a macro.
    impl Bind for Error {
        fn bind(self, _bindings: &Bindings) -> anyhow::Result<Self> {
            Ok(self)
        }
    }

    impl Unbind for Error {
        fn bindings(&self) -> anyhow::Result<Bindings> {
            Ok(Bindings::default())
        }
    }

    #[async_trait]
    impl Handler<Cast<Error>> for TestActor {
        async fn handle(
            &mut self,
            _this: &Instance<Self>,
            Cast {
                message: Error(error),
                ..
            }: Cast<Error>,
        ) -> Result<(), anyhow::Error> {
            Err(anyhow::anyhow!("{}", error))
        }
    }

    #[derive(Debug, Serialize, Deserialize, Named, Clone)]
    pub struct Relay(pub usize, pub VecDeque<PortRef<Relay>>);

    #[async_trait]
    impl Handler<Relay> for TestActor {
        async fn handle(
            &mut self,
            this: &Instance<Self>,
            Relay(count, mut hops): Relay,
        ) -> Result<(), anyhow::Error> {
            ensure!(!hops.is_empty(), "relay must have at least one hop");
            let next = hops.pop_front().unwrap();
            next.send(this, Relay(count + 1, hops))?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {

    use hyperactor::ActorId;
    use hyperactor::PortRef;
    use hyperactor::ProcId;
    use hyperactor::WorldId;
    use hyperactor::id;
    use hyperactor::mailbox::MessageEnvelope;
    use hyperactor::mailbox::Undeliverable;
    use hyperactor::message::Bind;
    use hyperactor::message::Unbind;
    use ndslice::shape;

    use super::*;

    // These tests are parametric over allocators.
    #[macro_export]
    macro_rules! actor_mesh_test_suite {
        ($allocator:expr_2021) => {
            use std::assert_matches::assert_matches;

            use ndslice::shape;
            use $crate::alloc::AllocSpec;
            use $crate::alloc::Allocator;
            use $crate::assign::Ranks;
            use $crate::sel_from_shape;
            use $crate::sel;
            use $crate::proc_mesh::SharedSpawnable;
            use std::collections::VecDeque;

            use super::*;
            use super::test_util::*;

            #[tokio::test]
            async fn test_basic() {
                let alloc = $allocator
                    .allocate(AllocSpec {
                        shape: shape! { replica = 4 },
                        constraints: Default::default(),
                    })
                    .await
                    .unwrap();

                let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();
                let actor_mesh: RootActorMesh<TestActor> = proc_mesh.spawn("echo", &()).await.unwrap();
                let (reply_handle, mut reply_receiver) = actor_mesh.open_port();
                actor_mesh
                    .cast(sel!(*), Echo("Hello".to_string(), reply_handle.bind()))
                    .unwrap();
                for _ in 0..4 {
                    assert_eq!(&reply_receiver.recv().await.unwrap(), "Hello");
                }
            }

            #[tokio::test]
            async fn test_ping_pong() {
                use hyperactor::test_utils::pingpong::PingPongActor;
                use hyperactor::test_utils::pingpong::PingPongMessage;
                use hyperactor::test_utils::pingpong::PingPongActorParams;

                let alloc = $allocator
                    .allocate(AllocSpec {
                        shape: shape! { replica = 2  },
                        constraints: Default::default(),
                    })
                    .await
                    .unwrap();
                let mesh = ProcMesh::allocate(alloc).await.unwrap();

                let (undeliverable_msg_tx, _) = mesh.client().open_port();
                let ping_pong_actor_params = PingPongActorParams::new(undeliverable_msg_tx.bind(), None);
                let actor_mesh: RootActorMesh<PingPongActor> = mesh
                    .spawn::<PingPongActor>("ping-pong", &ping_pong_actor_params)
                    .await
                    .unwrap();

                let ping: ActorRef<PingPongActor> = actor_mesh.get(0).unwrap();
                let pong: ActorRef<PingPongActor> = actor_mesh.get(1).unwrap();
                let (done_tx, done_rx) = mesh.client().open_once_port();
                ping.send(mesh.client(), PingPongMessage(4, pong.clone(), done_tx.bind())).unwrap();

                assert!(done_rx.recv().await.unwrap());
            }

            #[tokio::test]
            async fn test_pingpong_full_mesh() {
                use hyperactor::test_utils::pingpong::PingPongActor;
                use hyperactor::test_utils::pingpong::PingPongActorParams;
                use hyperactor::test_utils::pingpong::PingPongMessage;

                use futures::future::join_all;

                const X: usize = 3;
                const Y: usize = 3;
                const Z: usize = 3;
                let alloc = $allocator
                    .allocate(AllocSpec {
                        shape: shape! { x = X, y = Y, z = Z },
                        constraints: Default::default(),
                    })
                    .await
                    .unwrap();

                let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();
                let (undeliverable_tx, _undeliverable_rx) = proc_mesh.client().open_port();
                let params = PingPongActorParams::new(undeliverable_tx.bind(), None);
                let actor_mesh: RootActorMesh<PingPongActor> =
                    proc_mesh.spawn::<PingPongActor>("pingpong", &params).await.unwrap();
                let shape = actor_mesh.shape();
                let slice = shape.slice();

                // 6-neighbor offsets.
                const NEIGHBORS: &[[isize; 3]] = &[
                    [-1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                    [0, 0, 1],
                ];

                // Collect futures for all OncePorts.
                let mut futures = Vec::new();

                for rank in slice.iter() {
                    let coords = slice.coordinates(rank).unwrap();
                    let actor = actor_mesh.get(rank).unwrap();

                    for offset in NEIGHBORS {
                        let ndim = coords.len();
                        let mut neighbor_coords = Vec::with_capacity(ndim);
                        let mut in_bounds = true;

                        for dim in 0..ndim {
                            let c = coords[dim];
                            let o = offset[dim];
                            let size = slice.sizes()[dim];

                            let val = c as isize + o;
                            if val < 0 || val >= size as isize {
                                in_bounds = false;
                                break;
                            }
                            neighbor_coords.push(val as usize);
                        }

                        if !in_bounds {
                            continue;
                        }

                        if let Ok(neighbor_rank) = shape.slice().location(&neighbor_coords) {
                            let neighbor = actor_mesh.get(neighbor_rank).unwrap();
                            let (done_tx, done_rx) = proc_mesh.client().open_once_port();
                            actor
                                .send(
                                    proc_mesh.client(),
                                    PingPongMessage(4, neighbor.clone(), done_tx.bind()),
                                )
                                .unwrap();
                            futures.push(done_rx.recv());
                        }
                    }
                }

                let results = join_all(futures).await;
                for result in results {
                    assert_eq!(result.unwrap(), true);
                }
            }

            #[tokio::test]
            async fn test_cast() {
                let alloc = $allocator
                    .allocate(AllocSpec {
                        shape: shape! { replica = 2, host = 2, gpu = 8 },
                        constraints: Default::default(),
                    })
                    .await
                    .unwrap();

                let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();
                let actor_mesh: RootActorMesh<TestActor> = proc_mesh.spawn("echo", &()).await.unwrap();
                let dont_simulate_error = true;
                let (reply_handle, mut reply_receiver) = actor_mesh.open_port();
                actor_mesh
                    .cast(sel!(*), GetRank(dont_simulate_error, reply_handle.bind()))
                    .unwrap();
                let mut ranks = Ranks::new(actor_mesh.shape().slice().len());
                while !ranks.is_full() {
                    let rank = reply_receiver.recv().await.unwrap();
                    assert!(ranks.insert(rank, rank).is_none(), "duplicate rank {rank}");
                }
                // Retrieve all GPUs on replica=0, host=0
                let (reply_handle, mut reply_receiver) = actor_mesh.open_port();
                actor_mesh
                    .cast(
                        sel_from_shape!(actor_mesh.shape(), replica = 0, host = 0),
                        GetRank(dont_simulate_error, reply_handle.bind()),
                    )
                    .unwrap();
                let mut ranks = Ranks::new(8);
                while !ranks.is_full() {
                    let rank = reply_receiver.recv().await.unwrap();
                    assert!(ranks.insert(rank, rank).is_none(), "duplicate rank {rank}");
                }
            }

            #[tokio::test]
            async fn test_inter_actor_comms() {
                let alloc = $allocator
                    .allocate(AllocSpec {
                        // Sizes intentionally small to keep the time
                        // required for this test in the process case
                        // reasonable (< 60s).
                        shape: shape! { replica = 2, host = 2, gpu = 8 },
                        constraints: Default::default(),
                    })
                    .await
                    .unwrap();

                let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();
                let actor_mesh: RootActorMesh<TestActor> = proc_mesh.spawn("echo", &()).await.unwrap();

                // Bounce the message through all actors and return it to the sender (us).
                let mut hops: VecDeque<_> = actor_mesh.iter().map(|actor| actor.port()).collect();
                let (handle, mut rx) = proc_mesh.client().open_port();
                hops.push_back(handle.bind());
                hops.pop_front()
                    .unwrap()
                    .send(proc_mesh.client(), Relay(0, hops))
                    .unwrap();
                assert_matches!(
                    rx.recv().await.unwrap(),
                    Relay(count, hops)
                        if count == actor_mesh.shape().slice().len()
                        && hops.is_empty());
            }

            #[tokio::test]
            async fn test_inter_proc_mesh_comms() {
                let mut meshes = Vec::new();
                for _ in 0..2 {
                    let alloc = $allocator
                        .allocate(AllocSpec {
                            shape: shape! { replica = 1 },
                            constraints: Default::default(),
                        })
                        .await
                        .unwrap();

                    let proc_mesh = Arc::new(ProcMesh::allocate(alloc).await.unwrap());
                    let proc_mesh_clone = Arc::clone(&proc_mesh);
                    let actor_mesh : RootActorMesh<TestActor> = proc_mesh_clone.spawn("echo", &()).await.unwrap();
                    meshes.push((proc_mesh, actor_mesh));
                }

                let mut hops: VecDeque<_> = meshes
                    .iter()
                    .flat_map(|(_proc_mesh, actor_mesh)| actor_mesh.iter())
                    .map(|actor| actor.port())
                    .collect();
                let num_hops = hops.len();

                let client = meshes[0].0.client();
                let (handle, mut rx) = client.open_port();
                hops.push_back(handle.bind());
                hops.pop_front()
                    .unwrap()
                    .send(client, Relay(0, hops))
                    .unwrap();
                assert_matches!(
                    rx.recv().await.unwrap(),
                    Relay(count, hops)
                        if count == num_hops
                        && hops.is_empty());
            }

            #[timed_test::async_timed_test(timeout_secs = 60)]
            async fn test_actor_mesh_cast() {
                // Verify a full broadcast in the mesh. Send a message
                // to every actor and check each actor receives it.

                use $crate::sel;
                use $crate::comm::test_utils::TestActor as CastTestActor;
                use $crate::comm::test_utils::TestActorParams as CastTestActorParams;
                use $crate::comm::test_utils::TestMessage as CastTestMessage;

                let shape = shape! {replica = 4, host = 4, gpu = 4 };
                let num_actors = shape.slice().len();
                let alloc = $allocator
                    .allocate(AllocSpec {
                        shape,
                        constraints: Default::default(),
                    })
                    .await
                    .unwrap();

                let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();

                let (tx, mut rx) = hyperactor::mailbox::open_port(proc_mesh.client());
                let params = CastTestActorParams{ forward_port: tx.bind() };
                let actor_mesh: RootActorMesh<CastTestActor> = proc_mesh.spawn("actor", &params).await.unwrap();

                actor_mesh.cast(sel!(*), CastTestMessage::Forward("abc".to_string())).unwrap();

                for _ in 0..num_actors {
                    assert_eq!(rx.recv().await.unwrap(), CastTestMessage::Forward("abc".to_string()));
                }
            }

            #[tokio::test]
            async fn test_delivery_failure() {
                let alloc = $allocator
                    .allocate(AllocSpec {
                        shape: shape! { replica = 1  },
                        constraints: Default::default(),
                    })
                    .await
                    .unwrap();

                let name = alloc.name().to_string();
                let mesh = ProcMesh::allocate(alloc).await.unwrap();
                let unmonitored_reply_to = mesh.client().open_port::<usize>().0.bind();
                let (undeliverable_messages, mut undeliverable_rx) = mesh.client().open_port::<Undeliverable<MessageEnvelope>>();
                undeliverable_messages.bind_to(Undeliverable::<MessageEnvelope>::port());

                // Send a message to a non-existent actor (the proc however exists).
                let bad_actor = ActorRef::<TestActor>::attest(ActorId(ProcId(WorldId(name.clone()), 0), "foo".into(), 0));
                bad_actor.send(mesh.client(), GetRank(true, unmonitored_reply_to)).unwrap();

                // The message will be returned!
                let Undeliverable(msg) = undeliverable_rx.recv().await.unwrap();
                assert_eq!(mesh.client().actor_id(), msg.sender());
                assert_eq!(&bad_actor.actor_id().port_id(GetRank::port()), msg.dest());

                // TODO: Stop the proc.
            }
        }
    }

    mod local {
        use crate::alloc::local::LocalAllocator;

        actor_mesh_test_suite!(LocalAllocator);

        #[tokio::test]
        async fn test_send_failure() {
            use hyperactor::test_utils::pingpong::PingPongActor;
            use hyperactor::test_utils::pingpong::PingPongActorParams;
            use hyperactor::test_utils::pingpong::PingPongMessage;

            use crate::alloc::ProcStopReason;
            use crate::proc_mesh::ProcEvent;

            let config = hyperactor::config::global::lock();
            let _guard = config.override_key(
                hyperactor::config::MESSAGE_DELIVERY_TIMEOUT,
                tokio::time::Duration::from_secs(1),
            );

            let alloc = LocalAllocator
                .allocate(AllocSpec {
                    shape: shape! { replica = 2  },
                    constraints: Default::default(),
                })
                .await
                .unwrap();
            let monkey = alloc.chaos_monkey();
            let mut mesh = ProcMesh::allocate(alloc).await.unwrap();
            let mut events = mesh.events().unwrap();

            let (undeliverable_msg_tx, mut undeliverable_msg_rx) = mesh.client().open_port();
            let ping_pong_actor_params =
                PingPongActorParams::new(undeliverable_msg_tx.bind(), None);
            let actor_mesh: RootActorMesh<PingPongActor> = mesh
                .spawn::<PingPongActor>("ping-pong", &ping_pong_actor_params)
                .await
                .unwrap();

            let ping: ActorRef<PingPongActor> = actor_mesh.get(0).unwrap();
            let pong: ActorRef<PingPongActor> = actor_mesh.get(1).unwrap();

            // Kill ping.
            monkey(0, ProcStopReason::Killed(0, false));
            assert_matches!(
                events.next().await.unwrap(),
                ProcEvent::Stopped(0, ProcStopReason::Killed(0, false))
            );

            // Get 'pong' to send 'ping' a message. Since 'ping's
            // mailbox is stopped, the send will timeout and fail.
            let (unmonitored_done_tx, _) = mesh.client().open_once_port();
            pong.send(
                mesh.client(),
                PingPongMessage(1, ping.clone(), unmonitored_done_tx.bind()),
            )
            .unwrap();

            // The message will be returned!
            let Undeliverable(msg) = undeliverable_msg_rx.recv().await.unwrap();
            assert_eq!(msg.sender(), pong.actor_id());
            assert_eq!(
                msg.dest(),
                &ping.actor_id().port_id(PingPongMessage::port())
            );
        }

        // The intent is to emulate the behaviors of the Python
        // interaction of T225230867 "process hangs when i send
        // messages to a dead actor".
        #[tracing_test::traced_test]
        #[tokio::test]
        async fn test_behaviors_on_actor_error() {
            use crate::alloc::ProcStopReason;
            use crate::proc_mesh::ProcEvent;
            use crate::sel;

            let alloc = LocalAllocator
                .allocate(AllocSpec {
                    shape: shape! { replica = 1  },
                    constraints: Default::default(),
                })
                .await
                .unwrap();

            let stop = alloc.stopper();
            let mut mesh = ProcMesh::allocate(alloc).await.unwrap();
            let mut events = mesh.events().unwrap();

            let actor_mesh = mesh
                .spawn::<TestActor>("reply-then-fail", &())
                .await
                .unwrap();

            // `GetRank` with `false` means exit with error after
            // replying with rank.
            let (reply_handle, mut reply_receiver) = actor_mesh.open_port();
            actor_mesh
                .cast(sel!(*), GetRank(false, reply_handle.bind()))
                .unwrap();
            let rank = reply_receiver.recv().await.unwrap();
            assert_eq!(rank, 0);

            // The above is expected to trigger a proc crash.
            assert_matches!(
                events.next().await.unwrap(),
                ProcEvent::Crashed(0, reason) if reason.contains("intentional error!")
            );

            // Uncomment this to cause an infinite hang.
            /*
            let (reply_handle, mut reply_receiver) = actor_mesh.open_port();
                actor_mesh
                    .cast(sel!(*), GetRank(false, reply_handle.bind()))
                    .unwrap();
            let rank = reply_receiver.recv().await.unwrap();
            */

            // Stop the mesh.
            stop();
            assert_matches!(
                events.next().await.unwrap(),
                ProcEvent::Stopped(0, ProcStopReason::Stopped),
            );
            assert!(events.next().await.is_none());
        }
    } // mod local

    mod process {
        use tokio::process::Command;

        use crate::alloc::process::ProcessAllocator;

        #[cfg(fbcode_build)] // we use an external binary, produced by buck
        actor_mesh_test_suite!(ProcessAllocator::new(Command::new(
            buck_resources::get("monarch/hyperactor_mesh/bootstrap").unwrap()
        )));
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Named)]
    struct MyNamedStruct {
        field0: u64,
        field1: String,
        field2: PortRef<String>,
        field3: bool,
        field4: hyperactor::PortRef<u64>,
    }

    // TODO(pzhang) replace the boilerplate Bind/Unbind impls with a macro.
    impl Bind for MyNamedStruct {
        fn bind(mut self, bindings: &Bindings) -> anyhow::Result<Self> {
            let mut_ports = [self.field2.port_id_mut(), self.field4.port_id_mut()];
            bindings.rebind(mut_ports.into_iter())?;
            Ok(self)
        }
    }

    impl Unbind for MyNamedStruct {
        fn bindings(&self) -> anyhow::Result<Bindings> {
            let mut bindings = Bindings::default();
            let ports = [self.field2.port_id(), self.field4.port_id()];
            bindings.insert(ports)?;
            Ok(bindings)
        }
    }

    #[test]
    fn test_cast_bind_unbind() {
        let port_id2 = id!(world[0].client[0][2]);
        let port_id4 = id!(world[1].client[0][4]);
        let message = MyNamedStruct {
            field0: 0,
            field1: "hello".to_string(),
            field2: PortRef::attest(port_id2.clone()),
            field3: true,
            field4: PortRef::attest(port_id4.clone()),
        };

        let rank = CastRank(3);
        let cast = Cast {
            rank: rank.clone(),
            shape: shape! { replica = 2, host = 4, gpu = 8 },
            message: message.clone(),
        };

        // Verify Unbind is implemented correctly.
        let bindings = cast.bindings().unwrap();
        let mut expected = Bindings::default();
        expected
            .insert(&[port_id2.clone(), port_id4.clone()])
            .unwrap();
        expected.push(&cast.rank).unwrap();
        assert_eq!(bindings, expected);

        // Verify Bind is implemented correctly.
        let new_rank = CastRank(11);
        assert_ne!(rank.0, new_rank.0);
        let new_port_id2 = id!(world[0].comm[0][213]);
        assert_ne!(port_id2, new_port_id2);
        let new_port_id4 = id!(world[1].comm[0][423]);
        assert_ne!(port_id4, new_port_id4);
        assert_ne!(new_port_id2, new_port_id4);
        let mut new_bindings = Bindings::default();
        new_bindings
            .insert(&[new_port_id2.clone(), new_port_id4.clone()])
            .unwrap();
        new_bindings.push(&new_rank).unwrap();
        let new_cast = cast.bind(&new_bindings).unwrap();
        assert_eq!(
            new_cast.message,
            MyNamedStruct {
                field0: 0,
                field1: "hello".to_string(),
                field2: PortRef::attest(new_port_id2.clone()),
                field3: true,
                field4: PortRef::attest(new_port_id4.clone()),
            },
        );
        assert_eq!(new_cast.rank.0, new_rank.0);
    }
}
