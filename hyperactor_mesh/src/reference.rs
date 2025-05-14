use std::cmp::Ord;
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;

use hyperactor::Named;
use hyperactor::actor::RemoteActor;
use ndslice::Selection;
use ndslice::Shape;
use ndslice::selection::structurally_equal;
use serde::Deserialize;
use serde::Serialize;

#[macro_export]
macro_rules! mesh_id {
    ($proc_mesh:ident) => {
        $crate::reference::ProcMeshId(stringify!($proc_mesh).to_string(), "0".into())
    };
    ($proc_mesh:ident . $actor_mesh:ident) => {
        $crate::reference::ActorMeshId(
            $crate::reference::ProcMeshId(stringify!($proc_mesh).to_string()),
            stringify!($proc_mesh).to_string(),
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
pub struct ProcMeshId(pub String);

/// Actor Mesh ID.  Tuple of the ProcMesh ID and Actor Mesh ID.
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
pub struct ActorMeshId(pub ProcMeshId, pub String);

/// Types references to Actor Meshes.
#[derive(Debug, Serialize, Deserialize)]
pub struct ActorMeshRef<A: RemoteActor> {
    pub(crate) mesh_id: ActorMeshId,
    shape: Shape,
    selection: Selection,
    phantom: PhantomData<A>,
}

impl<A: RemoteActor> ActorMeshRef<A> {
    /// The caller guarantees that the provided mesh ID is also a valid,
    /// typed reference.  This is usually invoked to provide a guarantee
    /// that an externally-provided mesh ID (e.g., through a command
    /// line argument) is a valid reference.
    pub fn attest(mesh_id: ActorMeshId, shape: Shape) -> Self {
        Self {
            mesh_id,
            shape,
            selection: Selection::True,
            phantom: PhantomData,
        }
    }

    /// The caller guarantees that the provided mesh ID is also a valid,
    /// typed reference.  This is usually invoked to provide a guarantee
    /// that an externally-provided mesh ID (e.g., through a command
    /// line argument) is a valid reference.
    pub fn attest_with_selection(mesh_id: ActorMeshId, shape: Shape, selection: Selection) -> Self {
        Self {
            mesh_id,
            shape,
            selection,
            phantom: PhantomData,
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

    /// Selection of the Actor Mesh.
    pub fn selection(self) -> Selection {
        self.selection
    }
}

impl<A: RemoteActor> Clone for ActorMeshRef<A> {
    fn clone(&self) -> Self {
        Self {
            mesh_id: self.mesh_id.clone(),
            shape: self.shape.clone(),
            selection: self.selection.clone(),
            phantom: PhantomData,
        }
    }
}

impl<A: RemoteActor> PartialEq for ActorMeshRef<A> {
    fn eq(&self, other: &Self) -> bool {
        self.mesh_id == other.mesh_id
            && self.shape == other.shape
            && structurally_equal(&self.selection, &other.selection)
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
    use ndslice::shape;

    use super::*;
    use crate::test_utils::EmptyActor;

    fn shape() -> Shape {
        shape! { replica = 4 }
    }

    #[tokio::test]
    async fn test_mesh_correct_id() {
        let mesh_id = mesh_id!(proc_mesh.actor_mesh);
        let mesh_ref = ActorMeshRef::<EmptyActor>::attest(mesh_id.clone(), shape());

        assert_eq!(mesh_ref.mesh_id().clone(), mesh_id);
        assert_eq!(mesh_ref.shape().clone(), shape());

        // With Selection
        let selection = Selection::Range(0.into(), Box::new(Selection::True));
        let mesh_ref = ActorMeshRef::<EmptyActor>::attest_with_selection(
            mesh_id.clone(),
            shape(),
            selection.clone(),
        );
        assert!(structurally_equal(
            &mesh_ref.clone().selection(),
            &selection.clone()
        ));
    }
}
