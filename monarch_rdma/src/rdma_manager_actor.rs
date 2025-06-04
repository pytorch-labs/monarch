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

use crate::ibverbs_primitives::IbvWc;
use crate::ibverbs_primitives::IbverbsConfig;
use crate::ibverbs_primitives::RdmaMemoryRegionView;
use crate::ibverbs_primitives::RdmaOperation;
use crate::ibverbs_primitives::RdmaQpInfo;
use crate::rdma_components::RdmaDomain;
use crate::rdma_components::RdmaQueuePair;

/// Represents a reference to a remote RDMA buffer that can be accessed via RDMA operations.
/// This struct encapsulates all the information needed to identify and access a memory region
/// on a remote host using RDMA.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Serialize, Deserialize)]
pub struct RemoteBufferRef {
    /// The RDMA manager actor that owns this buffer
    pub manager: ActorRef<RdmaManagerActor>,
    /// Memory region view containing address and length information
    mr: RdmaMemoryRegionView,
    /// Remote key needed to access this memory region via RDMA
    rkey: u32,
}

impl RemoteBufferRef {
    /// Creates a new `RemoteBufferRef` instance.
    ///
    /// # Arguments
    ///
    /// * `manager` - The `ActorRef` to the `RdmaManagerActor` that owns this buffer.
    /// * `mr` - The `RdmaMemoryRegionView` containing address and length information of the memory region.
    /// * `rkey` - The remote key needed to access this memory region via RDMA.
    ///
    /// # Returns
    ///
    /// A new instance of `RemoteBufferRef` initialized with the provided manager, memory region view, and remote key.
    pub fn new(
        manager: ActorRef<RdmaManagerActor>,
        mr: RdmaMemoryRegionView,
        rkey: u32,
    ) -> RemoteBufferRef {
        RemoteBufferRef { manager, mr, rkey }
    }
}

#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub enum RdmaManagerMessage {
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

    Fetch {
        /// `local_memory_region` - The local memory region where data will be placed
        local_memory_region: RdmaMemoryRegionView,
        /// `remote_buffer` - The remote buffer provided by the caller, representing
        /// the memory location on the caller's host that contains the data
        remote_buffer: RemoteBufferRef,
        #[reply]
        /// `reply` - Reply channel to return the work completion ID
        reply: OncePortRef<u64>,
    },

    Put {
        /// `local_memory_region` - The local memory region containing data to be written
        local_memory_region: RdmaMemoryRegionView,
        /// `remote_buffer` - The remote buffer provided by the caller, representing
        /// the memory location where data will be written
        remote_buffer: RemoteBufferRef,
        #[reply]
        /// `reply` - Reply channel to return the work completion ID
        reply: OncePortRef<u64>,
    },

    PollCompletion {
        /// `other` - The ActorId of the actor associated with the operation
        other: ActorRef<RdmaManagerActor>,
        #[reply]
        /// `reply` - Reply channel to return work completion details if completed, None if not completed
        reply: OncePortRef<Option<IbvWc>>,
    },
    Release {
        /// `other` - The ActorId associated with the memory region
        other: ActorRef<RdmaManagerActor>,
        /// `region` - The memory region to release
        region: RdmaMemoryRegionView,
    },
    ConnectionInfo {
        /// `other` - The ActorId to get connection info for
        other: ActorRef<RdmaManagerActor>,
        #[reply]
        /// `reply` - Reply channel to return connection information needed for the RDMA connection
        reply: OncePortRef<RdmaQpInfo>,
    },
    GetKeys {
        #[reply]
        /// `reply` - Reply channel to return a tuple containing (lkey, rkey) where lkey is the local memory key
        /// and rkey is the remote memory key
        reply: OncePortRef<(u32, u32)>,
    },
}

#[derive(Debug)]
#[hyperactor::export_spawn(RdmaManagerMessage)]
pub struct RdmaManagerActor {
    // Map between ActorIds and their corresponding RdmaQueuePair
    qp_map: HashMap<ActorId, RdmaQueuePair>,

    // Map between a RemoteBufferRef and its latest work completion ID.
    work_id_map: HashMap<RemoteBufferRef, u64>,

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
}

impl RdmaManagerActor {
    /// Number of work completion IDs to reserve per actor ID / memory region
    ///
    /// This constant defines how many work completion IDs are reserved for each
    /// unique combination of actor ID and memory region. It helps avoid ID conflicts
    /// between different RDMA operations by allocating a dedicated range of IDs
    /// for each memory region.
    ///
    /// It's okay for work IDs to be re-used, so we expect to pass this limit
    /// and loop back to the beginning of the reservation block.
    const WORK_ID_RESERVATION_SIZE: u64 = 100;

    /// Convenience utility to create a new RdmaConnection.
    ///
    /// This initializes a new RDMA connection with another actor if one doesn't already exist.
    /// It creates a new RdmaQueuePair associated with the specified actor ID and adds it to the
    /// connection map.
    ///
    /// # Arguments
    ///
    /// * `other` - The ActorRef of the remote actor to connect with
    pub async fn initialize_qp(
        &mut self,
        other: ActorRef<RdmaManagerActor>,
    ) -> Result<(), anyhow::Error> {
        let key = other.actor_id().clone();
        if let std::collections::hash_map::Entry::Vacant(e) = self.qp_map.entry(key) {
            tracing::debug!("initializing connection with {:?}", other);
            let qp = RdmaQueuePair::new(&self.domain)
                .map_err(|e| anyhow::anyhow!("could not create RdmaQueuePair: {}", e))?;
            e.insert(qp);
            tracing::debug!("successfully created a connection with {:?}", other);
        }
        Ok(())
    }

    /// Generates a new work completion ID for a given memory region
    ///
    /// This generates a unique work completion ID for RDMA operations
    /// associated with a specific memory region. It manages a reservation system
    /// where each buffer gets a dedicated block of IDs to avoid
    /// conflicts between different operations.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The RemoteBufferRef containing the memory region
    ///
    /// # Returns
    ///
    /// A unique work completion ID for the RDMA operation
    async fn next_work_id(&mut self, buffer: &RemoteBufferRef) -> u64 {
        let base_id_reservation_size = Self::WORK_ID_RESERVATION_SIZE;
        let reservation_idx = self.work_id_map.len() as u64;

        let entry = self.work_id_map.entry(buffer.clone());
        let work_id = match entry {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let current = *e.get();
                let next = current + 1;

                // Check if we've reached the end of our reservation block
                let reservation_base =
                    (current / base_id_reservation_size) * base_id_reservation_size;
                let reservation_end = reservation_base + base_id_reservation_size - 1;

                if current >= reservation_end {
                    // Loop back to the start of this key's reservation block
                    e.insert(reservation_base);
                    reservation_base
                } else {
                    // Move to next ID in the reservation block
                    e.insert(next);
                    next
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                // First operation for this key, calculate its reservation block
                let base = reservation_idx * base_id_reservation_size;
                e.insert(base);
                base
            }
        };

        tracing::debug!(
            "calculated next work id for buffer {:?}: {}",
            buffer,
            work_id
        );

        work_id
    }
}

#[async_trait]
impl Actor for RdmaManagerActor {
    type Params = IbverbsConfig;

    async fn new(_params: Self::Params) -> Result<Self, anyhow::Error> {
        let config = _params;
        tracing::debug!("creating RdmaManagerActor with config {}", config);
        let domain = RdmaDomain::new(config)
            .map_err(|e| anyhow::anyhow!("rdmaManagerActor could not create domain: {}", e))?;
        Ok(Self {
            qp_map: HashMap::new(),
            work_id_map: HashMap::new(),
            domain,
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
        let connected = self.qp_map.contains_key(&other.actor_id().clone());
        Ok(connected)
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
        if !self.qp_map.contains_key(&other.actor_id().clone()) {
            self.initialize_qp(other.clone()).await?;
        }
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

    /// Performs an RDMA read operation (fetch data from remote memory)
    ///
    /// # Arguments
    /// * `local_memory_region` - The local memory region where data will be placed
    /// * `remote_buffer` - The remote buffer to read from
    ///
    /// # Returns
    /// * `u64` - Work completion ID for tracking the operation
    async fn fetch(
        &mut self,
        _this: &Instance<Self>,
        local_memory_region: RdmaMemoryRegionView,
        remote_buffer: RemoteBufferRef,
    ) -> Result<u64, anyhow::Error> {
        let work_id = self.next_work_id(&remote_buffer).await;

        let qp = self
            .qp_map
            .get_mut(&remote_buffer.manager.actor_id().clone())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "on fetch, no connection found for actor {}",
                    remote_buffer.manager.actor_id()
                )
            })?;
        tracing::debug!(
            "fetch for ({:?}) (qp: {:?}) with id {}",
            remote_buffer,
            qp,
            work_id
        );

        qp.post_send(
            local_memory_region.addr(),
            local_memory_region.len(),
            work_id,
            true,
            RdmaOperation::Read,
            remote_buffer.mr.addr(),
            remote_buffer.rkey,
        )
        .map_err(|e| anyhow::anyhow!("could not post RDMA read: {}", e))?;

        Ok(work_id)
    }

    /// Performs an RDMA write operation (put data to remote memory)
    ///
    /// # Arguments
    /// * `local_memory_region` - The local memory region containing data to be written
    /// * `remote_buffer` - The remote buffer to write to
    ///
    /// # Returns
    /// * `u64` - Work completion ID for tracking the operation
    async fn put(
        &mut self,
        _this: &Instance<Self>,
        local_memory_region: RdmaMemoryRegionView,
        remote_buffer: RemoteBufferRef,
    ) -> Result<u64, anyhow::Error> {
        let work_id = self.next_work_id(&remote_buffer).await;

        let qp = self
            .qp_map
            .get_mut(&remote_buffer.manager.actor_id().clone())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "on put, no connection found for actor {}",
                    remote_buffer.manager
                )
            })?;
        tracing::debug!(
            "put for ({:?}) (qp: {:?}) with id {}",
            remote_buffer,
            qp,
            work_id
        );
        qp.post_send(
            local_memory_region.addr(),
            local_memory_region.len(),
            work_id,
            true,
            RdmaOperation::Write,
            remote_buffer.mr.addr(),
            remote_buffer.rkey,
        )
        .map_err(|e| anyhow::anyhow!("could not post RDMA write: {}", e))?;

        Ok(work_id)
    }

    /// Polls for completion of an RDMA operation
    ///
    /// # Arguments
    /// * `other` - The ActorRef of the actor associated with the operation
    ///
    /// # Returns
    /// * `Option<IbvWc>` - Work completion details if completed, None if not completed
    async fn poll_completion(
        &mut self,
        _this: &Instance<Self>,
        other: ActorRef<RdmaManagerActor>,
    ) -> Result<Option<IbvWc>, anyhow::Error> {
        let qp = self
            .qp_map
            .get_mut(&other.actor_id().clone())
            .ok_or_else(|| anyhow::anyhow!("on poll, no connection found for actor {}", other))?;

        let wc = qp
            .poll_completion()
            .map_err(|e| anyhow::anyhow!("could not poll completion: {}", e))?;
        Ok(wc)
    }

    /// Releases a registered memory region
    ///
    /// # Arguments
    /// * `other` - The ActorRef associated with the memory region
    /// * `region` - The memory region to release
    async fn release(
        &mut self,
        _this: &Instance<Self>,
        other: ActorRef<RdmaManagerActor>,
        region: RdmaMemoryRegionView,
    ) -> Result<(), anyhow::Error> {
        let buffer = RemoteBufferRef {
            manager: other,
            mr: region,
            rkey: 0, // We don't need the actual rkey for removal
        };
        tracing::debug!("releasing {:?}", buffer);
        self.work_id_map.remove(&buffer);
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
        if !self.qp_map.contains_key(&other.actor_id().clone()) {
            self.initialize_qp(other.clone()).await?;
        }

        let connection_info = self
            .qp_map
            .get_mut(&other.actor_id().clone())
            .ok_or_else(|| anyhow::anyhow!("no connection found for actor {}", other))?
            .get_qp_info()?;
        Ok(connection_info)
    }

    /// Gets the local and remote memory keys from the RDMA domain
    ///
    /// # Returns
    /// * `(u32, u32)` - A tuple containing (lkey, rkey) where lkey is the local memory key
    ///   and rkey is the remote memory key
    async fn get_keys(&mut self, _this: &Instance<Self>) -> Result<(u32, u32), anyhow::Error> {
        let lkey = self.domain.lkey();
        let rkey = self.domain.rkey();
        tracing::debug!("lkey: {} rkey: {}", lkey, rkey);
        Ok((lkey, rkey))
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::Mailbox;
    use hyperactor::clock::Clock;
    use hyperactor::clock::RealClock;
    use hyperactor_mesh::Mesh;
    use hyperactor_mesh::ProcMesh;
    use hyperactor_mesh::RootActorMesh;
    use hyperactor_mesh::alloc::AllocConstraints;
    use hyperactor_mesh::alloc::AllocSpec;
    use hyperactor_mesh::alloc::Allocator;
    use hyperactor_mesh::alloc::LocalAllocator;
    use ndslice::shape;
    use tokio::time::Duration;

    use super::*;
    use crate::ibverbs_primitives::get_all_devices;

    struct RdmaManagerTestEnv {
        buffer1: Box<[u8]>,
        buffer2: Box<[u8]>,
        // Note: Using a static `proc_mesh` here is suboptimal because it leaks memory,
        // but it is necessary for the lifetime of `actor_mesh` in this test setup.
        // This should be fine for the sake of testing though.
        proc_mesh_1: &'static ProcMesh,
        proc_mesh_2: &'static ProcMesh,
        actor_1: ActorRef<RdmaManagerActor>,
        actor_2: ActorRef<RdmaManagerActor>,
        rkey1: u32,
        rkey2: u32,
    }

    impl RdmaManagerTestEnv {
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

            // Get keys from both actors.
            let actor_1 = actor_mesh_1.get(0).unwrap();
            let actor_2 = actor_mesh_2.get(0).unwrap();
            let (_, rkey1) = actor_1.get_keys(proc_mesh_1.client()).await?;
            let (_, rkey2) = actor_2.get_keys(proc_mesh_2.client()).await?;
            Ok(Self {
                buffer1,
                buffer2,
                proc_mesh_1,
                proc_mesh_2,
                actor_1,
                actor_2,
                rkey1,
                rkey2,
            })
        }

        // Initializes the RDMA connections between two actor meshes.
        //
        // This function sets up the RDMA connection by exchanging connection information
        // between two actors and establishing the connection using the provided endpoints.
        // It ensures that both actors are aware of each other's memory regions and can
        // perform RDMA operations.
        //
        // The function first retrieves the connection information for each actor's memory
        // region and then uses this information to establish the RDMA connection.
        async fn initialize(&mut self) -> Result<(), anyhow::Error> {
            // Get the endpoints
            let endpoint1 = self
                .actor_1
                .connection_info(self.proc_mesh_1.client(), self.actor_2.clone())
                .await?;
            let endpoint2 = self
                .actor_2
                .connection_info(self.proc_mesh_2.client(), self.actor_1.clone())
                .await?;

            // Connect to endpoints
            self.actor_1
                .connect(self.proc_mesh_1.client(), self.actor_2.clone(), endpoint2)
                .await?;

            self.actor_2
                .connect(self.proc_mesh_2.client(), self.actor_1.clone(), endpoint1)
                .await?;

            Ok(())
        }

        // Waits for the completion of an RDMA operation.
        //
        // This function polls for the completion of an RDMA operation by repeatedly
        // sending a `PollCompletion` message to the specified actor mesh and checking
        // the returned work completion status. It continues polling until the operation
        // completes or the specified timeout is reached.
        async fn wait_for_completion(
            &self,
            client: &Mailbox,
            waiting_actor: ActorRef<RdmaManagerActor>,
            calling_actor: ActorRef<RdmaManagerActor>,
            wr_id: u64,
            timeout_secs: u64,
        ) -> Result<bool, anyhow::Error> {
            let timeout = Duration::from_secs(timeout_secs);
            let start_time = std::time::Instant::now();
            while start_time.elapsed() < timeout {
                let wc = waiting_actor
                    .poll_completion(client, calling_actor.clone())
                    .await
                    .unwrap();
                match wc {
                    Some(wc) => {
                        if wc.wr_id() == wr_id {
                            return Ok(true);
                        }
                    }
                    None => {
                        RealClock.sleep(Duration::from_millis(1)).await;
                    }
                }
            }

            Ok(false)
        }

        // Verifies that both buffers contain the same data.
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
    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_rdma_write_loopback() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;

        let mut env = RdmaManagerTestEnv::setup(BSIZE, None).await?;
        env.initialize().await?;

        let client = env.proc_mesh_1.client();
        // Get work ID for the operation
        let remote_buffer = RemoteBufferRef {
            manager: env.actor_2.clone(),
            mr: RdmaMemoryRegionView::from_boxed_slice(&env.buffer2),
            rkey: env.rkey2,
        };
        let work_id = env
            .actor_1
            .put(
                client,
                RdmaMemoryRegionView::from_boxed_slice(&env.buffer1),
                remote_buffer,
            )
            .await?;
        let completed = env
            .wait_for_completion(client, env.actor_1.clone(), env.actor_2.clone(), work_id, 5)
            .await?;
        assert!(completed, "rdma write operation did not complete");

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    // Test that RDMA read can be performed between two actors on the same device.
    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_rdma_read_loopback() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;

        let mut env = RdmaManagerTestEnv::setup(BSIZE, None).await?;
        env.initialize().await?;

        let client = env.proc_mesh_2.client();
        // Get work ID for the operation
        let remote_buffer = RemoteBufferRef {
            manager: env.actor_1.clone(),
            mr: RdmaMemoryRegionView::from_boxed_slice(&env.buffer1),
            rkey: env.rkey1,
        };
        let work_id = env
            .actor_2
            .fetch(
                client,
                RdmaMemoryRegionView::from_boxed_slice(&env.buffer2),
                remote_buffer,
            )
            .await?;
        let completed = env
            .wait_for_completion(client, env.actor_2.clone(), env.actor_1.clone(), work_id, 5)
            .await?;
        assert!(completed, "rdma read operation did not complete");

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    // Test that RDMA read can be performed between two actors on separate devices.
    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_rdma_read_separate_devices() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let devices = get_all_devices();
        if devices.len() != 12 {
            println!(
                "skipping this test as it is only configured on H100 nodes with backend network"
            );
            return Ok(());
        }

        let mut env = RdmaManagerTestEnv::setup(BSIZE, Some((0, 4))).await?;
        env.initialize().await?;

        let client = env.proc_mesh_2.client();
        // Get work ID for the operation
        let remote_buffer = RemoteBufferRef {
            manager: env.actor_1.clone(),
            mr: RdmaMemoryRegionView::from_boxed_slice(&env.buffer1),
            rkey: env.rkey1,
        };
        let work_id = env
            .actor_2
            .fetch(
                client,
                RdmaMemoryRegionView::from_boxed_slice(&env.buffer2),
                remote_buffer,
            )
            .await?;
        let completed = env
            .wait_for_completion(client, env.actor_2.clone(), env.actor_1.clone(), work_id, 5)
            .await?;
        assert!(completed, "rdma read operation did not complete");

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    // Test that RDMA write can be performed between two actors on separate devices.
    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_rdma_write_separate_devices() -> Result<(), anyhow::Error> {
        const BSIZE: usize = 32;
        let devices = get_all_devices();
        if devices.len() != 12 {
            println!(
                "skipping this test as it is only configured on H100 nodes with backend network"
            );
            return Ok(());
        }
        let mut env = RdmaManagerTestEnv::setup(BSIZE, Some((0, 4))).await?;
        env.initialize().await?;

        let client = env.proc_mesh_1.client();
        // Get work ID for the operation
        let remote_buffer = RemoteBufferRef {
            manager: env.actor_2.clone(),
            mr: RdmaMemoryRegionView::from_boxed_slice(&env.buffer2),
            rkey: env.rkey2,
        };
        let work_id = env
            .actor_1
            .put(
                client,
                RdmaMemoryRegionView::from_boxed_slice(&env.buffer1),
                remote_buffer,
            )
            .await?;
        let completed = env
            .wait_for_completion(client, env.actor_1.clone(), env.actor_2.clone(), work_id, 5)
            .await?;
        assert!(completed, "rdma write operation did not complete");

        env.verify_buffers(BSIZE).await?;
        Ok(())
    }

    // Tests that keys can be retrieved from the RDMA manager.
    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_get_keys() -> Result<(), anyhow::Error> {
        let mut env = RdmaManagerTestEnv::setup(32, None).await?;
        env.initialize().await?;

        let client = env.proc_mesh_1.client();
        let (lkey, rkey) = env.actor_1.get_keys(client).await?;
        assert!(lkey > 0, "lkey should be greater than 0");
        assert!(rkey > 0, "rkey should be greater than 0");

        Ok(())
    }

    // Tests that connected() works correctly.
    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_connected() -> Result<(), anyhow::Error> {
        let mut env = RdmaManagerTestEnv::setup(32, None).await?;

        let client1 = env.proc_mesh_1.client();
        let client2 = env.proc_mesh_2.client();

        // Check that initially there's no connection
        let connected = env
            .actor_1
            .is_connected(client1, env.actor_2.clone())
            .await?;
        assert!(!connected, "connection should not be initialized yet");

        // Initialize the connection properly
        env.initialize().await?;

        // Check connection from actor_1 to actor_2
        let connected = env
            .actor_1
            .is_connected(client1, env.actor_2.clone())
            .await?;
        assert!(
            connected,
            "connection from actor_1 to actor_2 should be initialized"
        );

        // Check connection from actor_2 to actor_1
        let connected = env
            .actor_2
            .is_connected(client2, env.actor_1.clone())
            .await?;
        assert!(
            connected,
            "connection from actor_2 to actor_1 should be initialized"
        );

        Ok(())
    }
}
