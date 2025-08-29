/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ord;
use std::cmp::PartialOrd;
use std::hash::Hash;
use std::marker::PhantomData;

use hyperactor::ActorRef;
use hyperactor::Named;
use hyperactor::RemoteHandles;
use hyperactor::RemoteMessage;
use hyperactor::actor::RemoteActor;
use hyperactor::cap;
use hyperactor::message::Castable;
use hyperactor::message::IndexedErasedUnbound;
use ndslice::Range;
use ndslice::Selection;
use ndslice::Shape;
use ndslice::ShapeError;
use ndslice::selection::ReifySlice;
use serde::Deserialize;
use serde::Serialize;

use crate::CommActor;
use crate::actor_mesh::CastError;
use crate::actor_mesh::actor_mesh_cast;
use crate::actor_mesh::cast_to_sliced_mesh;

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

/// Actor Mesh ID.  Tuple of the ProcMesh ID and actor name.
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
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ActorMeshRef<A: RemoteActor> {
    pub(crate) mesh_id: ActorMeshId,
    /// The shape of the root mesh.
    root: Shape,
    /// If some, it mean this mesh ref points to a sliced mesh, and this field
    /// is this sliced mesh's shape. If None, it means this mesh ref points to
    /// the root mesh.
    sliced: Option<Shape>,
    /// The reference to the comm actor of the underlying Proc Mesh.
    comm_actor_ref: ActorRef<CommActor>,
    phantom: PhantomData<A>,
}

impl<A: RemoteActor> ActorMeshRef<A> {
    /// The caller guarantees that the provided mesh ID is also a valid,
    /// typed reference.  This is usually invoked to provide a guarantee
    /// that an externally-provided mesh ID (e.g., through a command
    /// line argument) is a valid reference.
    pub fn attest(mesh_id: ActorMeshId, root: Shape, comm_actor_ref: ActorRef<CommActor>) -> Self {
        Self {
            mesh_id,
            root,
            sliced: None,
            comm_actor_ref,
            phantom: PhantomData,
        }
    }

    /// The Actor Mesh ID corresponding with this reference.
    pub fn mesh_id(&self) -> &ActorMeshId {
        &self.mesh_id
    }

    /// Shape of the Actor Mesh.
    pub fn shape(&self) -> &Shape {
        match &self.sliced {
            Some(s) => s,
            None => &self.root,
        }
    }

    /// Cast an [`M`]-typed message to the ranks selected by `sel`
    /// in this ActorMesh.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `CastError`.
    pub fn cast<M>(
        &self,
        caps: &(impl cap::CanSend + cap::CanOpenPort),
        selection: Selection,
        message: M,
    ) -> Result<(), CastError>
    where
        A: RemoteHandles<M> + RemoteHandles<IndexedErasedUnbound<M>>,
        M: Castable + RemoteMessage,
    {
        match &self.sliced {
            Some(sliced_shape) => cast_to_sliced_mesh::<A, M>(
                caps,
                self.mesh_id.clone(),
                &self.comm_actor_ref,
                &selection,
                message,
                sliced_shape,
                &self.root,
            ),
            None => actor_mesh_cast::<A, M>(
                caps,
                self.mesh_id.clone(),
                &self.comm_actor_ref,
                selection,
                &self.root,
                &self.root,
                message,
            ),
        }
    }

    pub fn select<R: Into<Range>>(&self, label: &str, range: R) -> Result<Self, ShapeError> {
        let sliced = self.shape().select(label, range)?;
        Ok(Self {
            mesh_id: self.mesh_id.clone(),
            root: self.root.clone(),
            sliced: Some(sliced),
            comm_actor_ref: self.comm_actor_ref.clone(),
            phantom: PhantomData,
        })
    }

    pub fn new_with_shape(&self, new_shape: Shape) -> anyhow::Result<Self> {
        let base_slice = self.shape().slice();
        base_slice.reify_slice(new_shape.slice()).map_err(|e| {
            anyhow::anyhow!(
                "failed to reify the new shape into the base shape; this \
                normally means the new shape is not a valid slice of the base \
                error is: {e:?}"
            )
        })?;

        Ok(Self {
            mesh_id: self.mesh_id.clone(),
            root: self.root.clone(),
            sliced: Some(new_shape),
            comm_actor_ref: self.comm_actor_ref.clone(),
            phantom: PhantomData,
        })
    }
}

impl<A: RemoteActor> Clone for ActorMeshRef<A> {
    fn clone(&self) -> Self {
        Self {
            mesh_id: self.mesh_id.clone(),
            root: self.root.clone(),
            sliced: self.sliced.clone(),
            comm_actor_ref: self.comm_actor_ref.clone(),
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use hyperactor::Actor;
    use hyperactor::Bind;
    use hyperactor::Context;
    use hyperactor::Handler;
    use hyperactor::PortRef;
    use hyperactor::Unbind;
    use hyperactor_mesh_macros::sel;
    use ndslice::Extent;
    use ndslice::extent;

    use super::*;
    use crate::Mesh;
    use crate::ProcMesh;
    use crate::RootActorMesh;
    use crate::actor_mesh::ActorMesh;
    use crate::alloc::AllocSpec;
    use crate::alloc::Allocator;
    use crate::alloc::LocalAllocator;

    fn extent() -> Extent {
        extent!(replica = 4)
    }

    #[derive(Debug, Serialize, Deserialize, Named, Clone, Bind, Unbind)]
    struct MeshPingPongMessage(
        /*ttl:*/ u64,
        ActorMeshRef<MeshPingPongActor>,
        /*completed port:*/ #[binding(include)] PortRef<bool>,
    );

    #[derive(Debug, Clone)]
    #[hyperactor::export(
        spawn = true,
        handlers = [MeshPingPongMessage { cast = true }],
    )]
    struct MeshPingPongActor {
        mesh_ref: ActorMeshRef<MeshPingPongActor>,
    }

    #[derive(Debug, Serialize, Deserialize, Named, Clone)]
    struct MeshPingPongActorParams {
        mesh_id: ActorMeshId,
        shape: Shape,
        comm_actor_ref: ActorRef<CommActor>,
    }

    #[async_trait]
    impl Actor for MeshPingPongActor {
        type Params = MeshPingPongActorParams;

        async fn new(params: Self::Params) -> Result<Self, anyhow::Error> {
            Ok(Self {
                mesh_ref: ActorMeshRef::attest(params.mesh_id, params.shape, params.comm_actor_ref),
            })
        }
    }

    #[async_trait]
    impl Handler<MeshPingPongMessage> for MeshPingPongActor {
        async fn handle(
            &mut self,
            cx: &Context<Self>,
            MeshPingPongMessage(ttl, sender_mesh, done_tx): MeshPingPongMessage,
        ) -> Result<(), anyhow::Error> {
            if ttl == 0 {
                done_tx.send(cx, true)?;
                return Ok(());
            }
            let msg = MeshPingPongMessage(ttl - 1, self.mesh_ref.clone(), done_tx);
            sender_mesh.cast(cx, sel!(?), msg)?;
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_inter_mesh_ping_pong() {
        let alloc_ping = LocalAllocator
            .allocate(AllocSpec {
                extent: extent(),
                constraints: Default::default(),
                proc_name: None,
            })
            .await
            .unwrap();
        let alloc_pong = LocalAllocator
            .allocate(AllocSpec {
                extent: extent(),
                constraints: Default::default(),
                proc_name: None,
            })
            .await
            .unwrap();
        let ping_proc_mesh = ProcMesh::allocate(alloc_ping).await.unwrap();
        let ping_mesh: RootActorMesh<MeshPingPongActor> = ping_proc_mesh
            .spawn(
                "ping",
                &MeshPingPongActorParams {
                    mesh_id: ActorMeshId(
                        ProcMeshId(ping_proc_mesh.world_id().to_string()),
                        "ping".to_string(),
                    ),
                    shape: ping_proc_mesh.shape().clone(),
                    comm_actor_ref: ping_proc_mesh.comm_actor().clone(),
                },
            )
            .await
            .unwrap();
        assert_eq!(ping_proc_mesh.shape(), ping_mesh.shape());

        let pong_proc_mesh = ProcMesh::allocate(alloc_pong).await.unwrap();
        let pong_mesh: RootActorMesh<MeshPingPongActor> = pong_proc_mesh
            .spawn(
                "pong",
                &MeshPingPongActorParams {
                    mesh_id: ActorMeshId(
                        ProcMeshId(pong_proc_mesh.world_id().to_string()),
                        "pong".to_string(),
                    ),
                    shape: pong_proc_mesh.shape().clone(),
                    comm_actor_ref: pong_proc_mesh.comm_actor().clone(),
                },
            )
            .await
            .unwrap();

        let ping_mesh_ref: ActorMeshRef<MeshPingPongActor> = ping_mesh.bind();
        let pong_mesh_ref: ActorMeshRef<MeshPingPongActor> = pong_mesh.bind();

        let (done_tx, mut done_rx) = ping_proc_mesh.client().open_port::<bool>();
        ping_mesh_ref
            .cast(
                ping_proc_mesh.client(),
                sel!(?),
                MeshPingPongMessage(10, pong_mesh_ref, done_tx.bind()),
            )
            .unwrap();

        assert!(done_rx.recv().await.unwrap());
    }
}
