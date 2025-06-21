/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # RDMA Manager Actor
//!
//! Manages RDMA connections and operations using `hyperactor` for asynchronous messaging.
//!
//! ## Architecture
//!
//! `RdmaManagerActor` is a per-host entity that:
//! - Manages connections to multiple remote RdmaManagerActors (i.e. across the hosts in a Monarch cluster)
//! - Handles memory registration, connection setup, and data transfer
//! - Manages all RdmaBuffers in its associated host
//!
//! ## Core Operations
//!
//! - Connection establishment with partner actors
//! - RDMA operations (put/write, get/read)
//! - Completion polling
//! - Memory region management
//!
//! ## Usage
//!
//! See test examples: `test_rdma_write_loopback` and `test_rdma_read_loopback`.
use std::collections::HashMap;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::OncePortRef;
use hyperactor::RefClient;
use hyperactor::supervision::ActorSupervisionEvent;
use serde::Deserialize;
use serde::Serialize;

use crate::ibverbs_primitives::IbverbsConfig;
use crate::ibverbs_primitives::RdmaQpInfo;
use crate::rdma_components::RdmaBuffer;
use crate::rdma_components::RdmaDomain;
use crate::rdma_components::RdmaQueuePair;

/// Represents a reference to a remote RDMA buffer that can be accessed via RDMA operations.
/// This struct encapsulates all the information needed to identify and access a memory region
/// on a remote host using RDMA.
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub enum RdmaManagerMessage {
    RequestBuffer {
        addr: usize,
        size: usize,
        #[reply]
        /// `reply` - Reply channel to return the RDMA buffer handle
        reply: OncePortRef<RdmaBuffer>,
    },
    RequestQueuePair {
        remote: ActorRef<RdmaManagerActor>,
        #[reply]
        /// `reply` - Reply channel to return the queue pair for communication
        reply: OncePortRef<RdmaQueuePair>,
    },
    IsConnected {
        /// `other` - The ActorId of the actor to check connection with
        other: ActorRef<RdmaManagerActor>,
        #[reply]
        /// `reply` - Reply channel to return whether the actors have connected
        reply: OncePortRef<bool>,
    },
    Connect {
        /// `other` - The ActorId of the actor to connect to
        other: ActorRef<RdmaManagerActor>,
        /// `endpoint` - Connection information needed to establish the RDMA connection
        endpoint: RdmaQpInfo,
    },
    InitializeQP {
        remote: ActorRef<RdmaManagerActor>,
        #[reply]
        /// `reply` - Reply channel to return the queue pair for communication
        reply: OncePortRef<bool>,
    },
    ConnectionInfo {
        /// `other` - The ActorId to get connection info for
        other: ActorRef<RdmaManagerActor>,
        #[reply]
        /// `reply` - Reply channel to return connection information needed for the RDMA connection
        reply: OncePortRef<RdmaQpInfo>,
    },
}

#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [
        RdmaManagerMessage,
    ],
)]
pub struct RdmaManagerActor {
    // Map between ActorIds and their corresponding RdmaQueuePair
    qp_map: HashMap<ActorId, RdmaQueuePair>,

    // The RDMA domain associated with this actor.
    //
    // The domain is responsible for managing the RDMA resources and configurations
    // specific to this actor. It encapsulates the context and protection domain
    // necessary for RDMA operations, ensuring that all RDMA activities are
    // performed within a consistent and isolated environment.
    //
    // This domain is initialized during the creation of the `RdmaManagerActor`
    // and is used throughout the actor's lifecycle to manage RDMA connections
    // and operations.
    domain: RdmaDomain,
    config: IbverbsConfig,
}

#[async_trait]
impl Actor for RdmaManagerActor {
    type Params = IbverbsConfig;

    async fn new(_params: Self::Params) -> Result<Self, anyhow::Error> {
        let config = _params;
        let domain = RdmaDomain::new(config.device.clone())
            .map_err(|e| anyhow::anyhow!("rdmaManagerActor could not create domain: {}", e))?;
        Ok(Self {
            qp_map: HashMap::new(),
            domain,
            config,
        })
    }

    async fn handle_supervision_event(
        &mut self,
        _this: &Instance<Self>,
        _event: &ActorSupervisionEvent,
    ) -> Result<bool, anyhow::Error> {
        tracing::error!("rdmaManagerActor supervision event: {:?}", _event);
        tracing::error!("rdmaManagerActor error occurred, stop the worker process, exit code: 1");
        std::process::exit(1);
    }
}

#[async_trait]
#[hyperactor::forward(RdmaManagerMessage)]
impl RdmaManagerMessageHandler for RdmaManagerActor {
    /// Requests a buffer to be registered with the RDMA domain.
    ///
    /// This function registers a memory region with the RDMA domain and returns an `RdmaBuffer`
    /// that encapsulates the necessary information for RDMA operations.
    ///
    /// # Arguments
    ///
    /// * `this` - The instance of the actor requesting the buffer.
    /// * `addr` - The starting address of the memory region to be registered.
    /// * `size` - The size of the memory region to be registered.
    ///
    /// # Returns
    ///
    /// * `Result<RdmaBuffer, anyhow::Error>` - On success, returns an `RdmaBuffer` containing
    ///   the registered memory region's details. On failure, returns an error.
    async fn request_buffer(
        &mut self,
        this: &Instance<Self>,
        addr: usize,
        size: usize,
    ) -> Result<RdmaBuffer, anyhow::Error> {
        let mr = self.domain.register_buffer(addr, size)?;
        Ok(RdmaBuffer {
            owner: this.bind().clone(),
            addr: mr.addr,
            size: mr.size,
            rkey: mr.rkey,
            lkey: mr.lkey,
        })
    }

    /// Requests a queue pair for communication with a remote RDMA manager actor.
    ///
    /// This function checks if a connection already exists with the specified remote actor.
    /// If not, it initializes a new queue pair and establishes a connection with the remote actor.
    /// It then retrieves the queue pair associated with the remote actor for communication.
    ///
    /// # Arguments
    ///
    /// * `this` - The instance of the actor requesting the queue pair.
    /// * `remote` - The ActorRef of the remote RDMA manager actor to communicate with.
    ///
    /// # Returns
    ///
    /// * `Result<RdmaQueuePair, anyhow::Error>` - On success, returns the queue pair for communication.
    ///   On failure, returns an error.
    async fn request_queue_pair(
        &mut self,
        this: &Instance<Self>,
        remote: ActorRef<RdmaManagerActor>,
    ) -> Result<RdmaQueuePair, anyhow::Error> {
        if !self.is_connected(this, remote.clone()).await? {
            self.initialize_qp(this, remote.clone()).await?;
            remote.initialize_qp(this, this.bind().clone()).await?;

            let remote_endpoint = remote.connection_info(this, this.bind().clone()).await?;
            self.connect(this, remote.clone(), remote_endpoint).await?;

            let local_endpoint = self.connection_info(this, remote.clone()).await?;
            remote
                .connect(this, this.bind().clone(), local_endpoint)
                .await?;
        }
        let qp = self
            .qp_map
            .get_mut(&remote.actor_id().clone())
            .ok_or_else(|| anyhow::anyhow!("on get, no connection found for actor {}", remote))?;
        Ok(qp.clone())
    }

    /// Convenience utility to create a new RdmaQueuePair.
    ///
    /// This initializes a new RDMA connection with another actor if one doesn't already exist.
    /// It creates a new RdmaQueuePair associated with the specified actor ID and adds it to the
    /// connection map.
    ///
    /// # Arguments
    ///
    /// * `other` - The ActorRef of the remote actor to connect with
    async fn initialize_qp(
        &mut self,
        _this: &Instance<Self>,
        other: ActorRef<RdmaManagerActor>,
    ) -> Result<bool, anyhow::Error> {
        let key = other.actor_id().clone();

        if let std::collections::hash_map::Entry::Vacant(e) = self.qp_map.entry(key) {
            let qp = RdmaQueuePair::new(self.domain.context, self.domain.pd, self.config.clone())
                .map_err(|e| anyhow::anyhow!("could not create RdmaQueuePair: {}", e))?;
            e.insert(qp);
            tracing::debug!("successfully created a connection with {:?}", other);
        }
        Ok(true)
    }

    /// Checks if a connection exists with another actor
    ///
    /// # Arguments
    /// * `other` - The ActorRef of the actor to check connection with
    ///
    /// # Returns
    /// * `bool` - True if connected, false otherwise
    async fn is_connected(
        &mut self,
        _this: &Instance<Self>,
        other: ActorRef<RdmaManagerActor>,
    ) -> Result<bool, anyhow::Error> {
        tracing::debug!("checking if connected with {:?}", other);
        if !self.qp_map.contains_key(&other.actor_id().clone()) {
            return Ok(false);
        }
        let qp_state = self
            .qp_map
            .get_mut(&other.actor_id().clone())
            .unwrap()
            .state()?;
        Ok(qp_state == ffi::ibv_qp_state::IBV_QPS_RTS)
    }

    /// Establishes a connection with another actor
    ///
    /// # Arguments
    /// * `other` - The ActorRef of the actor to connect to
    /// * `endpoint` - Connection information needed to establish the RDMA connection
    async fn connect(
        &mut self,
        _this: &Instance<Self>,
        other: ActorRef<RdmaManagerActor>,
        endpoint: RdmaQpInfo,
    ) -> Result<(), anyhow::Error> {
        tracing::debug!("connecting with {:?}", other);
        let qp = self
            .qp_map
            .get_mut(&other.actor_id().clone())
            .ok_or_else(|| {
                anyhow::anyhow!("on connect, no connection found for actor {}", other)
            })?;
        qp.connect(&endpoint)
            .map_err(|e| anyhow::anyhow!("could not connect to RDMA endpoint: {}", e))?;
        Ok(())
    }

    /// Gets connection information for establishing an RDMA connection
    ///
    /// # Arguments
    /// * `other` - The ActorRef to get connection info for
    ///
    /// # Returns
    /// * `RdmaQpInfo` - Connection information needed for the RDMA connection
    async fn connection_info(
        &mut self,
        _this: &Instance<Self>,
        other: ActorRef<RdmaManagerActor>,
    ) -> Result<RdmaQpInfo, anyhow::Error> {
        tracing::debug!("getting connection info with {:?}", other);

        let connection_info = self
            .qp_map
            .get_mut(&other.actor_id().clone())
            .ok_or_else(|| anyhow::anyhow!("no connection found for actor {}", other))?
            .get_qp_info()?;
        Ok(connection_info)
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::Mailbox;
    use hyperactor_mesh::Mesh;
    use hyperactor_mesh::ProcMesh;
    use hyperactor_mesh::RootActorMesh;
    use hyperactor_mesh::alloc::AllocSpec;
    use hyperactor_mesh::alloc::Allocator;
    use hyperactor_mesh::alloc::LocalAllocator;
    use ndslice::shape;

    use super::*;
    use crate::ibverbs_primitives::get_all_devices;
    use crate::test_utils::wait_for_completion;

    struct RdmaManagerTestEnv<'a> {
        buffer1: Box<[u8]>,
        buffer2: Box<[u8]>,
        client_1: &'a Mailbox,
        client_2: &'a Mailbox,
        actor_1: ActorRef<RdmaManagerActor>,
        actor_2: ActorRef<RdmaManagerActor>,
        rdma_handle_1: RdmaBuffer,
        rdma_handle_2: RdmaBuffer,
    }

    impl RdmaManagerTestEnv<'_> {
        /// Sets up the RDMA test environment.
        ///
        /// This function initializes the RDMA test environment by setting up two actor meshes
        /// with their respective RDMA configurations. It also prepares two buffers for testing
        /// RDMA operations and fills the first buffer with test data.
        ///
        /// # Arguments
        ///
        /// * `buffer_size` - The size of the buffers to be used in the test.
        /// * `devices` - Optional tuple specifying the indices of RDMA devices to use. If not provided, then
        ///   both RDMAManagerActors will default to the first indexed RDMA device.
        async fn setup(
            buffer_size: usize,
            devices: Option<(usize, usize)>,
        ) -> Result<Self, anyhow::Error> {
            let (config1, config2) = if let Some((dev1_idx, dev2_idx)) = devices {
                let all_devices = get_all_devices();
                if all_devices.len() < 5 {
                    return Err(anyhow::anyhow!(
                        "need at least 5 RDMA devices for this test"
                    ));
                }
                (
                    IbverbsConfig {
                        device: all_devices.clone().into_iter().nth(dev1_idx).unwrap(),
                        ..Default::default()
                    },
                    IbverbsConfig {
                        device: all_devices.clone().into_iter().nth(dev2_idx).unwrap(),
                        ..Default::default()
                    },
                )
            } else {
                (IbverbsConfig::default(), IbverbsConfig::default())
            };

            let alloc_1 = LocalAllocator
                .allocate(AllocSpec {
                    shape: shape! { proc = 1 },
                    constraints: Default::default(),
                })
                .await
                .unwrap();

            let proc_mesh_1 = Box::leak(Box::new(ProcMesh::allocate(alloc_1).await.unwrap()));
            let actor_mesh_1: RootActorMesh<'_, RdmaManagerActor> =
                proc_mesh_1.spawn("rdma_manager", &config1).await.unwrap();

            let alloc_2 = LocalAllocator
                .allocate(AllocSpec {
                    shape: shape! { proc = 1 },
                    constraints: Default::default(),
                })
                .await
                .unwrap();

            let proc_mesh_2 = Box::leak(Box::new(ProcMesh::allocate(alloc_2).await.unwrap()));
            let actor_mesh_2: RootActorMesh<'_, RdmaManagerActor> =
                proc_mesh_2.spawn("rdma_manager", &config2).await.unwrap();

            let mut buffer1 = vec![0u8; buffer_size].into_boxed_slice();
            let buffer2 = vec![0u8; buffer_size].into_boxed_slice();

            // Fill buffer1 with test data
            for (i, val) in buffer1.iter_mut().enumerate() {
                *val = (i % 256) as u8;
            }

            let actor_1 = actor_mesh_1.get(0).unwrap();
            let actor_2 = actor_mesh_2.get(0).unwrap();

            let rdma_handle_1 = actor_1
                .request_buffer(
                    proc_mesh_1.client(),
                    buffer1.as_ptr() as usize,
                    buffer1.len(),
                )
                .await?;
            let rdma_handle_2 = actor_2
                .request_buffer(
                    proc_mesh_2.client(),
                    buffer2.as_ptr() as usize,
                    buffer2.len(),
                )
                .await?;
            // Get keys from both actors.

            Ok(Self {
                buffer1,
                buffer2,
                client_1: proc_mesh_1.client(),
                client_2: proc_mesh_2.client(),
                actor_1,
                actor_2,
                rdma_handle_1,
                rdma_handle_2,
            })
        }

        async fn verify_buffers(&self, size: usize) -> Result<(), anyhow::Error> {
            for i in 0..size {
                assert_eq!(
                    self.buffer1[i], self.buffer2[i],
                    "data mismatch at position {}: {} != {}",
                    i, self.buffer1[i], self.buffer2[i]
                );
            }
            Ok(())
        }
    }

    // Test that RDMA write can be performed between two actors on the same device.
    #[timed_test::async_timed_test(timeout_secs = 20)]
    async fn test_rdma_read_loopback() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let env = RdmaManagerTestEnv::setup(BSIZE, Some((0, 0))).await?;
        let mut qp_1 = env
            .actor_1
            .request_queue_pair(&env.client_1.clone(), env.actor_2.clone())
            .await?;
        qp_1.put(env.rdma_handle_1.clone(), env.rdma_handle_2.clone())?;

        // Poll for completion
        wait_for_completion(&qp_1, 2).await?;

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    #[timed_test::async_timed_test(timeout_secs = 20)]
    async fn test_rdma_write_loopback() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let env = RdmaManagerTestEnv::setup(BSIZE, Some((0, 0))).await?;
        let mut qp_1 = env
            .actor_1
            .request_queue_pair(&env.client_1.clone(), env.actor_2.clone())
            .await?;
        qp_1.get(env.rdma_handle_1.clone(), env.rdma_handle_2.clone())?;

        // Poll for completion
        wait_for_completion(&qp_1, 2).await?;

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    // Test that RDMA read can be performed between two actors on separate devices.
    #[timed_test::async_timed_test(timeout_secs = 20)]
    async fn test_rdma_read_separate_devices() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let devices = get_all_devices();
        if devices.len() < 4 {
            println!(
                "skipping this test as it is only configured on H100 nodes with backend network"
            );
            return Ok(());
        }
        let env = RdmaManagerTestEnv::setup(BSIZE, Some((0, 4))).await?;
        let mut qp_1 = env
            .actor_1
            .request_queue_pair(&env.client_1.clone(), env.actor_2.clone())
            .await?;
        qp_1.put(env.rdma_handle_1.clone(), env.rdma_handle_2.clone())?;

        // Poll for completion
        wait_for_completion(&qp_1, 2).await?;

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    // Test that RDMA write can be performed between two actors on separate devices.
    #[timed_test::async_timed_test(timeout_secs = 20)]
    async fn test_rdma_write_separate_devices() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let devices = get_all_devices();
        if devices.len() < 5 {
            println!(
                "skipping this test as it is only configured on H100 nodes with backend network"
            );
            return Ok(());
        }
        let mut env = RdmaManagerTestEnv::setup(BSIZE, Some((0, 4))).await?;
        let mut qp_1 = env
            .actor_1
            .request_queue_pair(&env.client_1.clone(), env.actor_2.clone())
            .await?;
        qp_1.get(env.rdma_handle_1.clone(), env.rdma_handle_2.clone())?;

        // Poll for completion
        wait_for_completion(&qp_1, 2).await?;

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    // Tests RdmaBufer's `read_into` API
    #[timed_test::async_timed_test(timeout_secs = 20)]
    async fn test_rdma_read_into() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let devices = get_all_devices();
        if devices.len() < 5 {
            println!(
                "skipping this test as it is only configured on H100 nodes with backend network"
            );
            return Ok(());
        }
        let env = RdmaManagerTestEnv::setup(BSIZE, Some((0, 4))).await?;
        let mut rdma_handle_1 = env.rdma_handle_1.clone();
        rdma_handle_1
            .read_into(&env.client_1.clone(), env.rdma_handle_2.clone(), 2)
            .await?;

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    // Tests RdmaBufer's `write_from` API
    #[timed_test::async_timed_test(timeout_secs = 20)]
    async fn test_rdma_write_from() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let devices = get_all_devices();
        if devices.len() < 5 {
            println!(
                "skipping this test as it is only configured on H100 nodes with backend network"
            );
            return Ok(());
        }
        let env = RdmaManagerTestEnv::setup(BSIZE, Some((0, 4))).await?;
        let mut rdma_handle_1 = env.rdma_handle_1.clone();
        rdma_handle_1
            .write_from(&env.client_1.clone(), env.rdma_handle_2.clone(), 2)
            .await?;

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }
}
