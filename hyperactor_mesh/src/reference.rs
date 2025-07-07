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
use hyperactor::actor::RemoteActor;
use hyperactor::cap;
use hyperactor::message::Castable;
use hyperactor::message::IndexedErasedUnbound;
use ndslice::Selection;
use ndslice::Shape;
use serde::Deserialize;
use serde::Serialize;

use crate::CommActor;
use crate::actor_mesh::CastError;
use crate::actor_mesh::actor_mesh_cast;

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
    /// The shape of the underlying Proc Mesh.
    proc_mesh_shape: Shape,
    /// The reference to the comm actor of the underlying Proc Mesh.
    comm_actor_ref: ActorRef<CommActor>,
    phantom: PhantomData<A>,
}

impl<A: RemoteActor> ActorMeshRef<A> {
    /// The caller guarantees that the provided mesh ID is also a valid,
    /// typed reference.  This is usually invoked to provide a guarantee
    /// that an externally-provided mesh ID (e.g., through a command
    /// line argument) is a valid reference.
    pub(crate) fn attest(
        mesh_id: ActorMeshId,
        shape: Shape,
        proc_mesh_shape: Shape,
        comm_actor_ref: ActorRef<CommActor>,
    ) -> Self {
        Self {
            mesh_id,
            shape,
            proc_mesh_shape,
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
        &self.shape
    }

    /// Shape of the underlying Proc Mesh.
    fn proc_mesh_shape(&self) -> &Shape {
        &self.proc_mesh_shape
    }

    fn name(&self) -> &str {
        &self.mesh_id.1
    }

    /// Cast an [`M`]-typed message to the ranks selected by `sel`
    /// in this ActorMesh.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `CastError`.
    pub fn cast<M: Castable + Clone>(
        &self,
        caps: &(impl cap::CanSend + cap::CanOpenPort),
        selection: Selection,
        message: M,
    ) -> Result<(), CastError>
    where
        A: RemoteHandles<M> + RemoteHandles<IndexedErasedUnbound<M>>,
    {
        actor_mesh_cast::<M, A>(
            caps,
            self.shape(),
            self.proc_mesh_shape(),
            self.name(),
            caps.mailbox().actor_id(),
            &self.comm_actor_ref,
            selection,
            message,
        )
    }
}

impl<A: RemoteActor> Clone for ActorMeshRef<A> {
    fn clone(&self) -> Self {
        Self {
            mesh_id: self.mesh_id.clone(),
            shape: self.shape.clone(),
            proc_mesh_shape: self.proc_mesh_shape.clone(),
            comm_actor_ref: self.comm_actor_ref.clone(),
            phantom: PhantomData,
        }
    }
}

impl<A: RemoteActor> PartialEq for ActorMeshRef<A> {
    fn eq(&self, other: &Self) -> bool {
        self.mesh_id == other.mesh_id && self.shape == other.shape
    }
}

impl<A: RemoteActor> Eq for ActorMeshRef<A> {}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use hyperactor::Actor;
    use hyperactor::Context;
    use hyperactor::Handler;
    use hyperactor::PortRef;
    use hyperactor::message::Bind;
    use hyperactor::message::Bindings;
    use hyperactor::message::Unbind;
    use hyperactor_mesh_macros::sel;
    use ndslice::shape;

    use super::*;
    use crate::Mesh;
    use crate::ProcMesh;
    use crate::RootActorMesh;
    use crate::actor_mesh::ActorMesh;
    use crate::alloc::AllocSpec;
    use crate::alloc::Allocator;
    use crate::alloc::LocalAllocator;

    fn shape() -> Shape {
        shape! { replica = 4 }
    }

    #[derive(Debug, Serialize, Deserialize, Named, Clone)]
    struct MeshPingPongMessage(
        /*ttl:*/ u64,
        ActorMeshRef<MeshPingPongActor>,
        /*completed port:*/ PortRef<bool>,
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
        proc_mesh_shape: Shape,
        comm_actor_ref: ActorRef<CommActor>,
    }

    #[async_trait]
    impl Actor for MeshPingPongActor {
        type Params = MeshPingPongActorParams;

        async fn new(params: Self::Params) -> Result<Self, anyhow::Error> {
            Ok(Self {
                mesh_ref: ActorMeshRef::attest(
                    params.mesh_id,
                    params.shape,
                    params.proc_mesh_shape,
                    params.comm_actor_ref,
                ),
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

    impl Unbind for MeshPingPongMessage {
        fn unbind(&self, bindings: &mut Bindings) -> anyhow::Result<()> {
            self.2.unbind(bindings)
        }
    }

    impl Bind for MeshPingPongMessage {
        fn bind(&mut self, bindings: &mut Bindings) -> anyhow::Result<()> {
            self.2.bind(bindings)
        }
    }

    #[tokio::test]
    async fn test_inter_mesh_ping_pong() {
        let alloc_ping = LocalAllocator
            .allocate(AllocSpec {
                shape: shape(),
                constraints: Default::default(),
            })
            .await
            .unwrap();
        let alloc_pong = LocalAllocator
            .allocate(AllocSpec {
                shape: shape(),
                constraints: Default::default(),
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
                    proc_mesh_shape: ping_proc_mesh.shape().clone(),
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
                    proc_mesh_shape: pong_proc_mesh.shape().clone(),
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
