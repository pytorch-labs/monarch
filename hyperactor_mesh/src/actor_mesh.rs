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
use hyperactor::message::IndexedErasedUnbound;
use hyperactor::message::Unbind;
use hyperactor::message::Unbound;
use ndslice::Range;
use ndslice::Selection;
use ndslice::Shape;
use ndslice::ShapeError;
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
    fn cast<M: RemoteMessage + Clone>(
        &self,
        selection: Selection,
        message: M,
    ) -> Result<(), CastError>
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

        self.proc_mesh().comm_actor().send(
            self.proc_mesh().client(),
            CastMessage {
                dest: Uslice {
                    slice: self.shape().slice().clone(),
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

/// A message that was cast in an [`ActorMesh`]. Actors that wish to
/// receive casted M-typed messages should implement handlers for `Cast<M>`.
#[derive(Debug, Serialize, Deserialize)]
pub struct Cast<M> {
    /// The rank of the receiving actor.
    pub rank: CastRank,
    /// The coordinates of the receiving actor in the actor mesh.
    pub shape: Shape,
    /// The message itself.
    pub message: M,
}

impl<M> Unbind for Cast<M> {
    fn unbind(self) -> anyhow::Result<Unbound<Self>> {
        let mut bindings = Bindings::default();
        bindings.insert([&self.rank])?;
        Ok(Unbound::new(self, bindings))
    }
}

impl<M> Bind for Cast<M> {
    fn bind(mut self, bindings: &Bindings) -> anyhow::Result<Self> {
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

    use super::*;

    // This can't be defined under a `#[cfg(test)]` because there needs to
    // be an entry in the spawnable actor registry in the executable
    // 'hyperactor_mesh_test_bootstrap' for the `tests::process` actor
    // mesh test suite.
    #[derive(Debug)]
    #[hyperactor::export_spawn(
        Cast<(String, PortRef<String>)>,
        Cast<GetRank>,
        Cast<Error>,
        GetRank,
        Relay,
        IndexedErasedUnbound<Cast<(String, PortRef<String>)>>,
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
    impl Handler<Cast<(String, PortRef<String>)>> for TestActor {
        async fn handle(
            &mut self,
            this: &Instance<Self>,
            Cast { message, .. }: Cast<(String, PortRef<String>)>,
        ) -> Result<(), anyhow::Error> {
            let (message, reply_port) = message;
            reply_port.send(this, message)?;
            Ok(())
        }
    }

    #[derive(Debug, Serialize, Deserialize, Named, Clone)]
    pub struct Error(pub String);

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
                hyperactor::test_utils::tracing::set_tracing_env_filter(tracing::Level::DEBUG);
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
                    .cast(sel!(*), ("Hello".to_string(), reply_handle.bind()))
                    .unwrap();
                for _ in 0..4 {
                    assert_eq!(&reply_receiver.recv().await.unwrap(), "Hello");
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

            #[tracing_test::traced_test]
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
        }
    }

    mod local {
        use crate::alloc::local::LocalAllocator;

        actor_mesh_test_suite!(LocalAllocator);

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

        #[tracing_test::traced_test]
        #[tokio::test]
        async fn test_behaviors_on_failed_send() {
            use hyperactor::ActorId;
            use hyperactor::ProcId;
            use hyperactor::WorldId;

            let alloc = LocalAllocator
                .allocate(AllocSpec {
                    shape: shape! { replica = 1  },
                    constraints: Default::default(),
                })
                .await
                .unwrap();

            let stop = alloc.stopper();
            let name = alloc.name().to_string();
            let mesh = ProcMesh::allocate(alloc).await.unwrap();

            let actor_mesh = mesh.spawn::<TestActor>("foo", &()).await.unwrap();

            let (reply_handle, mut reply_receiver) = actor_mesh.open_port();

            // Send a message to an actor that exists.
            let foo: ActorRef<TestActor> =
                ActorRef::attest(ActorId(ProcId(WorldId(name.clone()), 0), "foo".into(), 0));
            foo.send(mesh.client(), GetRank(true, reply_handle.bind()))
                .unwrap();
            // This is ok.
            let rank = reply_receiver.recv().await.unwrap();
            assert_eq!(rank, 0);

            // Send a message to an actor that doesn't exist but the proc does.
            let _bar: ActorRef<TestActor> =
                ActorRef::attest(ActorId(ProcId(WorldId(name.clone()), 0), "foo".into(), 1));
            // Uncomment for infinite hang.
            /*
            // Message gets logged.
            // delivery error: address not routable: no mailbox for actor _1ku59VQkwPSM[0].foo[1] registered in muxer
            bar.send(mesh.client(), GetRank(true, reply_handle.bind())).unwrap();
            */

            // Send a message to an actor on a proc that doesn't exist.
            let _baz: ActorRef<TestActor> =
                ActorRef::attest(ActorId(ProcId(WorldId(name.clone()), 1), "foo".into(), 0));
            // Uncomment for stack overflow.
            /*
            baz.send(mesh.client(), GetRank(true, reply_handle.bind())).unwrap();
            */

            // Stop the mesh.
            stop();
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
}
