#![allow(dead_code)] // until used publically

use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::Arc;

use anyhow::ensure;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorRef;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Message;
use hyperactor::Named;
use hyperactor::PortHandle;
use hyperactor::PortRef;
use hyperactor::RemoteHandles;
use hyperactor::RemoteMessage;
use hyperactor::actor::RemoteActor;
use hyperactor::cap;
use hyperactor::mailbox::MailboxSenderError;
use hyperactor::mailbox::PortReceiver;
use ndslice::Range;
use ndslice::Selection;
use ndslice::Shape;
use ndslice::ShapeError;
use ndslice::selection::EvalOpts;
use serde::Deserialize;
use serde::Serialize;

use crate::Mesh;
use crate::metrics;
use crate::proc_mesh::ProcMesh;

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
            Self::Borrowed(p) => p.deref(),
        }
    }
}

/// A mesh of actor instances. ActorMeshes are obtained by spawning an
/// actor on a [`ProcMesh`].
pub struct ActorMesh<'a, A: RemoteActor> {
    proc_mesh: ProcMeshRef<'a>,
    pub(crate) ranks: Vec<ActorRef<A>>, // temporary until we remove `ArcActorMesh`.
}

impl<'a, A: RemoteActor> ActorMesh<'a, A> {
    pub(crate) fn new(proc_mesh: &'a ProcMesh, ranks: Vec<ActorRef<A>>) -> Self {
        Self {
            proc_mesh: ProcMeshRef::Borrowed(proc_mesh),
            ranks,
        }
    }

    pub(crate) fn new_shared(proc_mesh: Arc<ProcMesh>, ranks: Vec<ActorRef<A>>) -> Self {
        Self {
            proc_mesh: ProcMeshRef::Shared(proc_mesh),
            ranks,
        }
    }

    /// Open a port on this ActorMesh.
    pub(crate) fn open_port<M: Message>(&self) -> (PortHandle<M>, PortReceiver<M>) {
        self.proc_mesh.client().open_port()
    }

    /// Cast an [`M`]-typed message to the ranks selected by `sel`
    /// in this ActorMesh.
    pub fn cast<M: RemoteMessage + Clone>(
        &self,
        sel: Selection,
        message: M,
    ) -> Result<(), CastError>
    where
        A: RemoteHandles<Cast<M>>,
    {
        let _ = metrics::ACTOR_MESH_CAST_DURATION.start(hyperactor::kv_pairs!(
            "message_type" => M::typename(),
            "message_variant" => message.arm().unwrap_or_default(),
        ));
        let ranks = sel
            .eval(&EvalOpts::strict(), self.shape().slice())
            .map_err(|err| CastError::InvalidSelection(sel, err))?;
        for rank in ranks {
            let cast = Cast {
                rank,
                shape: self.shape().clone(),
                message: message.clone(),
            };
            self.ranks[rank]
                .send(self.proc_mesh.client(), cast)
                .map_err(|err| CastError::MailboxSenderError(rank, err))?;
        }
        Ok(())
    }

    /// Cast an [`M`]-typed message to the ranks selected by `sel`
    pub fn cast_with_sender<M>(
        cap: &impl cap::CanSend,
        shape: &Shape,
        ranks: Vec<ActorRef<A>>,
        sel: Selection,
        message: M,
    ) -> Result<(), CastError>
    where
        M: RemoteMessage + Clone,
        A: RemoteHandles<Cast<M>>,
    {
        let _ = metrics::ACTOR_MESH_CAST_DURATION.start(hyperactor::kv_pairs!(
            "message_type" => M::typename(),
            "message_variant" => message.arm().unwrap_or_default(),
        ));
        let sel_iter = sel
            .eval(&EvalOpts::strict(), shape.slice())
            .map_err(|err| CastError::InvalidSelection(sel, err))?;
        for rank in sel_iter {
            let cast = Cast {
                rank,
                shape: shape.clone(),
                message: message.clone(),
            };
            ranks[rank]
                .send(cap, cast)
                .map_err(|err| CastError::MailboxSenderError(rank, err))?;
        }
        Ok(())
    }
}

#[async_trait]
impl<'a, A: RemoteActor> Mesh for ActorMesh<'a, A> {
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

pub struct SlicedActorMesh<'a, A: RemoteActor>(&'a ActorMesh<'a, A>, Shape);

impl<'a, A: RemoteActor> SlicedActorMesh<'a, A> {
    pub fn new(actor_mesh: &'a ActorMesh<'a, A>, shape: Shape) -> Self {
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

/// A message that was cast in an [`ActorMesh`]. Actors that wish to
/// receive casted M-typed messages should implement handlers for `Cast<M>`.
#[derive(Debug, Serialize, Deserialize)]
pub struct Cast<M> {
    /// The rank of the receiving actor.
    pub rank: usize,
    /// The coordinates of the receiving actor in the actor mesh.
    pub shape: Shape,
    /// The message itself.
    pub message: M,
}

impl<M: RemoteMessage> Named for Cast<M> {
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
    ShapeError(#[from] ShapeError),
}

mod test_util {
    use super::*;

    // This can't be defined under a `#[cfg(test)]` because there needs to
    // be an entry in the spawnable actor registry in the executable
    // 'hyperactor_mesh_test_bootstrap' for the `tests::process` actor
    // mesh test suite.
    #[derive(Debug)]
    #[hyperactor::export_spawn(Cast<(String, PortRef<String>)>, Cast<GetRank>, Relay)]
    pub struct TestActor;

    #[async_trait]
    impl Actor for TestActor {
        type Params = ();

        async fn new(_params: Self::Params) -> Result<Self, anyhow::Error> {
            Ok(Self)
        }
    }

    #[derive(Debug, Serialize, Deserialize, Named, Clone)]
    pub struct GetRank(pub PortRef<usize>);

    #[async_trait]
    impl Handler<Cast<GetRank>> for TestActor {
        async fn handle(
            &mut self,
            this: &Instance<Self>,
            Cast {
                rank,
                message: GetRank(reply),
                ..
            }: Cast<GetRank>,
        ) -> Result<(), anyhow::Error> {
            reply.send(this, rank)?;
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

    // Prototype. Replicated from hyperactor_mesh_core for the moment.
    #[macro_export] // ok since this is only enabled in tests
    macro_rules! select_ {
        ($shape:expr, $label:ident = $range:expr) => {
            ndslice::selection::selection_from_one($shape, stringify!($label), $range).unwrap()
        };

        ($shape:expr, $($label:ident = $val:literal),* $(,)?) => {
            ndslice::selection::selection_from($shape,
                           &[
                               $((stringify!($label), $val..$val+1)),*
                           ]).unwrap()
        };

        ($shape:expr, $($label:ident = $range:expr),* $(,)?) => {
            ndslice::selection::selection_from($shape, &[
                $((stringify!($label), $range)),*

            ]).unwrap()
        };
    }

    // These tests are parametric over allocators.
    #[macro_export]
    macro_rules! actor_mesh_test_suite {
        ($allocator:expr) => {
            use std::assert_matches::assert_matches;

            use ndslice::shape;
            use $crate::alloc::AllocSpec;
            use $crate::alloc::AllocConstraints;
            use $crate::alloc::Allocator;
            use $crate::assign::Ranks;
            use ndslice::selection::dsl::*;
            use $crate::proc_mesh::SharedSpawnable;

            use super::*;
            use super::test_util::*;

            #[tokio::test]
            async fn test_basic() {
                let alloc = $allocator
                    .allocate(AllocSpec {
                        shape: shape! { replica = 4 },
                        constraints: AllocConstraints::none(),
                    })
                    .await
                    .unwrap();

                let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();
                let actor_mesh: ActorMesh<TestActor> = proc_mesh.spawn("echo", &()).await.unwrap();
                let (reply_handle, mut reply_receiver) = actor_mesh.open_port();
                actor_mesh
                    .cast(all(true_()), ("Hello".to_string(), reply_handle.bind()))
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
                        constraints: AllocConstraints::none(),
                    })
                    .await
                    .unwrap();

                let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();
                let actor_mesh: ActorMesh<TestActor> = proc_mesh.spawn("echo", &()).await.unwrap();
                let (reply_handle, mut reply_receiver) = actor_mesh.open_port();
                actor_mesh
                    .cast(all(true_()), GetRank(reply_handle.bind()))
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
                        select_!(actor_mesh.shape(), replica = 0, host = 0),
                        GetRank(reply_handle.bind()),
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
                        constraints: AllocConstraints::none(),
                    })
                    .await
                    .unwrap();

                let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();
                let actor_mesh: ActorMesh<TestActor> = proc_mesh.spawn("echo", &()).await.unwrap();

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
                            constraints: AllocConstraints::none(),
                        })
                        .await
                        .unwrap();

                    let proc_mesh = Arc::new(ProcMesh::allocate(alloc).await.unwrap());
                    let proc_mesh_clone = Arc::clone(&proc_mesh);
                    let actor_mesh : ActorMesh<TestActor> = proc_mesh_clone.spawn("echo", &()).await.unwrap();
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
        }
    }

    mod local {
        use crate::alloc::local::LocalAllocator;

        actor_mesh_test_suite!(LocalAllocator);
    }

    mod process {
        // TODO: Fix T221991302.

        use tokio::process::Command;

        use crate::alloc::process::ProcessAllocator;

        #[cfg(fbcode_build)] // we use an external binary, produced by buck
        actor_mesh_test_suite!(ProcessAllocator::new(Command::new(
            buck_resources::get("monarch/hyperactor_mesh/bootstrap").unwrap()
        )));
    }
}
