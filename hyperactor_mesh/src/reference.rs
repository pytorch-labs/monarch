use std::cmp::Ord;
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use std::hash::Hash;
use std::hash::Hasher;

use hyperactor::ActorRef;
use hyperactor::Named;
use hyperactor::RemoteHandles;
use hyperactor::RemoteMessage;
use hyperactor::actor::RemoteActor;
use hyperactor::cap;
use ndslice::Selection;
use ndslice::Shape;
use serde::Deserialize;
use serde::Serialize;

use crate::ActorMesh;
use crate::Mesh;
use crate::actor_mesh::Cast;
use crate::actor_mesh::CastError;

#[macro_export]
macro_rules! mesh_id {
    ($proc_mesh:ident) => {
        $crate::reference::ProcMeshId(stringify!($proc_mesh).to_string(), "0".into())
    };
    ($proc_mesh:ident . $actor_mesh:ident) => {
        $crate::reference::ActorMeshId(
            $crate::reference::ProcMeshId(stringify!($proc_mesh).to_string()),
            stringify!($proc_mesh).to_string(),
            "0".into(),
        )
    };
    ($proc_mesh:ident . $actor_mesh:ident [$slice_index:ident]) => {
        $crate::reference::ActorMeshId(
            $crate::reference::ProcMeshId(stringify!($proc_mesh).to_string()),
            stringify!($actor_mesh),
            $slice_index.into(),
        )
    };
}

#[derive(
    Debug,
    Serialize,
    Deserialize,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Hash,
    Ord,
    Named
)]
pub struct ProcMeshId(String);

/// Actor Mesh ID.  This is a tuple of the ProcMesh ID, Actor Mesh ID, and Slice id, empty if not a slice.
#[derive(
    Debug,
    Serialize,
    Deserialize,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Hash,
    Ord,
    Named
)]
pub struct ActorMeshId(ProcMeshId, String, String);

/// Types references to Actor Meshes.
#[derive(Debug, Serialize, Deserialize)]
pub struct ActorMeshRef<A: RemoteActor> {
    pub(crate) mesh_id: ActorMeshId,
    shape: Shape,
    ranks: Vec<ActorRef<A>>,
}

impl<A: RemoteActor> ActorMeshRef<A> {
    pub fn from_mesh(mesh_id: ActorMeshId, mesh: &ActorMesh<A>) -> Self {
        Self {
            mesh_id: mesh_id.clone(),
            shape: mesh.shape().clone(),
            ranks: mesh.ranks.clone(),
        }
    }

    /// The caller guarantees that the provided mesh ID is also a valid,
    /// typed reference.  This is usually invoked to provide a guarantee
    /// that an externally-provided mesh ID (e.g., through a command
    /// line argument) is a valid reference.
    pub fn attest(mesh_id: ActorMeshId, ranks: Vec<ActorRef<A>>, shape: Shape) -> Self {
        Self {
            mesh_id,
            shape,
            ranks,
        }
    }

    /// The Actor Mesh ID corresponding with this reference.
    pub fn mesh_id(&self) -> &ActorMeshId {
        &self.mesh_id
    }

    /// Convert this actor mesh reference into its corresponding actor mesh ID.
    pub fn into_mesh_id(self) -> ActorMeshId {
        self.mesh_id
    }

    /// Shape of the Actor Mesh.
    pub fn shape(self) -> Shape {
        self.shape
    }

    pub fn cast<M>(
        self,
        cap: &impl cap::CanSend,
        sel: Selection,
        message: M,
    ) -> Result<(), CastError>
    where
        M: RemoteMessage + Clone,
        A: RemoteHandles<Cast<M>>,
    {
        ActorMesh::cast_with_sender(cap, &self.shape, self.ranks, sel, message)
    }
}

impl<A: RemoteActor> Clone for ActorMeshRef<A> {
    fn clone(&self) -> Self {
        Self {
            mesh_id: self.mesh_id.clone(),
            shape: self.shape.clone(),
            ranks: self.ranks.clone(),
        }
    }
}

impl<A: RemoteActor> PartialEq for ActorMeshRef<A> {
    fn eq(&self, other: &Self) -> bool {
        self.mesh_id == other.mesh_id
    }
}

impl<A: RemoteActor> Eq for ActorMeshRef<A> {}

impl<A: RemoteActor> PartialOrd for ActorMeshRef<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: RemoteActor> Ord for ActorMeshRef<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.mesh_id.cmp(&other.mesh_id)
    }
}

impl<A: RemoteActor> Hash for ActorMeshRef<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.mesh_id.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::id;
    use ndslice::shape;

    use super::*;
    use crate::ActorMesh;
    use crate::ProcMesh;
    use crate::alloc::AllocConstraints;
    use crate::alloc::AllocSpec;
    use crate::alloc::Allocator;
    use crate::alloc::local::LocalAllocator;
    use crate::test_utils::EmptyActor;
    use crate::test_utils::EmptyMessage;

    fn shape() -> Shape {
        shape! { replica = 4 }
    }

    async fn build_proc_mesh() -> ProcMesh {
        let alloc = LocalAllocator
            .allocate(AllocSpec {
                shape: shape(),
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();
        ProcMesh::allocate(alloc).await.unwrap()
    }

    #[tokio::test]
    async fn test_mesh_correct_id() {
        let mesh_id = mesh_id!(proc_mesh.actor_mesh);
        let actor_ref = ActorRef::<EmptyActor>::attest(id!(world[0].actor[1]));
        let mesh_ref =
            ActorMeshRef::<EmptyActor>::attest(mesh_id.clone(), vec![actor_ref], shape());

        assert_eq!(mesh_ref.mesh_id().clone(), mesh_id);
        assert_eq!(mesh_ref.shape().clone(), shape());
    }

    #[tokio::test]
    async fn test_actor_mesh_cast() {
        let proc_mesh = build_proc_mesh().await;
        let actor_mesh: ActorMesh<EmptyActor> = proc_mesh.spawn("test", &()).await.unwrap();
        let mesh_id = mesh_id!(proc_mesh.actor_mesh);
        let mesh_ref = ActorMeshRef::<EmptyActor>::from_mesh(mesh_id.clone(), &actor_mesh);

        assert!(
            mesh_ref
                .cast(proc_mesh.client(), Selection::True, EmptyMessage())
                .is_ok()
        );
    }
}
