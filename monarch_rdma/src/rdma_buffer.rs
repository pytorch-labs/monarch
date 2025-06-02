/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # RDMA Buffer
//!
//! This module provides an abstraction for Remote Direct Memory Access (RDMA) buffers.
//! The `RdmaBuffer` struct represents a handle or view into memory that can be accessed
//! remotely via RDMA operations.
//!
//! ## Architecture
//!
//! - `RdmaBuffer` is a lightweight handle to memory regions registered with RDMA
//! - Each buffer is associated with an `RdmaManagerActor` that handles the underlying RDMA operations
//! - The architecture follows a client-server model where actors can access each other's memory
//! - Connection management is handled automatically between actors when operations are performed
//! - Memory regions are registered with the RDMA subsystem for zero-copy access
//!
//! ## Usage Pattern
//!
//! - An actor creates an `RdmaBuffer` for its local memory and can share this handle with other actors
//! - Creating an `RdmaBuffer` requires an existing `RdmaManagerActor` on the same Proc. Multiple `RdmaBuffer` instances
//!   can be associated with the same `RdmaManagerActor`
//! - `RdmaBuffer` operations can only be performed in the context of actors, as they require access to a mailbox
//!   for communication
//! - Actors can interact with an `RdmaBuffer` using their own mailbox to send messages to the buffer's
//!   associated `RdmaManagerActor`
//!
//! ## Key Operations
//!
//! - Reading data from the buffer into a local destination (`read_into`)
//! - Writing data from a local source into the buffer (`write_from`)
//! - Releasing resources when they're no longer needed (`release`)
//! - Establishing connections between actors (`maybe_connect`, called automatically)
//!
//! RDMA operations can be performed synchronously (waiting for completion) or
//! asynchronously by setting the `wait_for_completion` parameter to `true` or `false` respectively.
//!
//! ## Safety
//!
//! RdmaBuffer exposes raw one-sided RDMA operations (reads, writes) that are inherently asynchronous
/// and unsynchronized. It **does not enforce locking or memory consistency guarantees** between
/// local and remote peers. The caller is responsible for ensuring proper synchronization to avoid
/// data races, stale reads, or inconsistent state.
///
/// This design choice is intentional: adding locking at this layer would impose assumptions and
/// constraints that may not align with higher-level usage patterns. Users who require safety
/// guarantees should implement coordination mechanisms at a higher abstraction layer.
use std::time::Duration;

use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::Mailbox;
use hyperactor::Named;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use serde::Deserialize;
use serde::Serialize;

use crate::RdmaManagerActor;
use crate::RdmaMemoryRegionView;
use crate::RemoteBufferRef;
use crate::rdma_manager_actor::RdmaManagerMessageClient;

/// A handle to memory that can be accessed remotely via RDMA operations.
///
/// This struct represents a view into memory that is registered with an `RdmaManagerActor`,
/// allowing other actors to read from or write to this memory region using RDMA operations.
/// Each buffer is associated with a specific `RdmaManagerActor` that handles the underlying
/// RDMA communication.
#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct RdmaBuffer {
    id: String,
    owner_id: ActorId,
    owner_ref: ActorRef<RdmaManagerActor>,
    lkey: u32,
    rkey: u32,
    memory_region: RdmaMemoryRegionView,
}

impl RdmaBuffer {
    /// Creates a new RDMA buffer handle associated with the specified memory region.
    ///
    /// This function establishes a connection with the `RdmaManagerActor` and registers
    /// the provided memory region for RDMA operations.
    ///
    /// TODO - a potential follow up here is to replace `owner_ref` with an `ActorHandle`
    /// and then bind it to produce the buffer. This approach is more accurate, as we expect
    /// that RdmaBuffers are created with local access to its `RdmaManagerActor`.
    /// This would require ActorHandles to be accessible from ActorMeshes.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this buffer
    /// * `owner_ref` - Reference to the owner's RdmaManagerActor
    /// * `client` - Mailbox used for communication
    /// * `memory_region` - View of the memory region to be registered for RDMA operations
    ///
    /// # Returns
    /// A new `RdmaBuffer` instance on success, or an error if registration fails
    pub async fn new(
        id: String,
        owner_ref: ActorRef<RdmaManagerActor>,
        client: &Mailbox,
        memory_region: RdmaMemoryRegionView,
    ) -> Result<Self, anyhow::Error> {
        let owner_id = owner_ref.actor_id();
        tracing::debug!(
            "creating RdmaBuffer with id {}, owner_id: {}, client: {:?}",
            id,
            owner_ref.actor_id(),
            client
        );
        let (lkey, rkey) = owner_ref.get_keys(client).await.unwrap();
        owner_ref
            .register_local_memory(client, memory_region.clone())
            .await
            .unwrap();
        tracing::debug!(
            "[buffer_{}] on rdma_buffer creation, lkey {}, rkey {}",
            id,
            lkey,
            rkey
        );
        Ok(Self {
            id,
            owner_id: owner_id.clone(),
            owner_ref,
            lkey,
            rkey,
            memory_region,
        })
    }

    /// Initializes the connection between the caller and the buffer's manager,
    /// if not already completed.
    ///
    /// This method ensures that the RDMA connection between the caller actor and the
    /// buffer's manager actor is established. If the connection already exists,
    /// it returns immediately. Otherwise, it exchanges connection information and
    /// establishes a new RDMA connection between the two actors.
    ///
    /// # Arguments
    /// * `caller_ref` - Reference to the caller's RdmaManagerActor
    /// * `client` - Mailbox used for communication
    ///
    /// # Returns
    /// `Ok(())` if the connection is established or already exists,
    /// or an error if the connection fails
    async fn maybe_connect(
        &self,
        caller_ref: &ActorRef<RdmaManagerActor>,
        client: &Mailbox,
    ) -> Result<(), anyhow::Error> {
        // First check if both RdmaManagerActors have connected to each other
        let caller_connected = caller_ref
            .is_connected(client, self.owner_ref.clone())
            .await?;
        let source_connected = self
            .owner_ref
            .is_connected(client, caller_ref.clone())
            .await?;

        // The two actors have already connected, go ahead and return.
        if caller_connected && (caller_connected == source_connected) {
            tracing::debug!(
                "[buffer_{}] actors {} (source) and {} (caller) have already connected, returning",
                self.id,
                self.owner_id,
                caller_ref.actor_id()
            );
            return Ok(());
        }
        tracing::debug!(
            "[buffer_{}] actors {} (source) and {} (caller) haven't connected, connecting now",
            self.id,
            self.owner_id,
            caller_ref.actor_id()
        );

        if caller_connected != source_connected {
            if caller_connected {
                return Err(anyhow::anyhow!(
                    "[buffer_{}] caller {} reports already having connected to source {}, but not vice versa, which is inconsistent.",
                    self.id,
                    caller_ref.actor_id(),
                    self.owner_id,
                ));
            } else {
                return Err(anyhow::anyhow!(
                    "[buffer_{}] source {} already connected to caller {}, but not vice versa, which is inconsistent.",
                    self.id,
                    self.owner_id,
                    caller_ref.actor_id(),
                ));
            }
        }

        // If the caller is itself, then only call connect once.
        if *caller_ref.actor_id() == self.owner_id {
            tracing::info!(
                "[buffer_{}] the caller is the same as the owner, only calling connect once.",
                self.id
            );
            let endpoint = caller_ref
                .connection_info(client, self.owner_ref.clone())
                .await?;
            caller_ref
                .connect(client, self.owner_ref.clone(), endpoint)
                .await?;
            return Ok(());
        }

        // Start the connection process
        let caller_endpoint = caller_ref
            .connection_info(client, self.owner_ref.clone())
            .await?;
        tracing::debug!(
            "[buffer_{}] caller endpoint: {:?}",
            self.id,
            caller_endpoint
        );

        let source_endpoint = self
            .owner_ref
            .connection_info(client, caller_ref.clone())
            .await?;
        tracing::debug!(
            "[buffer_{}] source endpoint: {:?}",
            self.id,
            source_endpoint
        );

        // Call connect on both actors
        caller_ref
            .connect(client, self.owner_ref.clone(), source_endpoint)
            .await?;
        self.owner_ref
            .connect(client, caller_ref.clone(), caller_endpoint)
            .await?;

        tracing::debug!(
            "[buffer_{}] rdma connection established between caller {} and buffer {}",
            self.id,
            caller_ref.actor_id(),
            self.owner_id
        );

        Ok(())
    }

    /// Waits for the completion of an RDMA operation.
    ///
    /// This method polls the completion queue until the specified work request completes
    /// or until the timeout is reached.
    ///
    /// # Arguments
    /// * `work_id` - The ID of the work request to wait for
    /// * `client` - Mailbox used for communication
    /// * `caller_ref` - Reference to the caller's RdmaManagerActor
    /// * `timeout_in_s` - Timeout in seconds
    ///
    /// # Returns
    /// `Ok(Some(work_id))` if the operation completes successfully within the timeout,
    /// or an error if the timeout is reached
    async fn wait_for_completion(
        &self,
        work_id: u64,
        client: &Mailbox,
        caller_ref: &ActorRef<RdmaManagerActor>,
        timeout_in_s: u64,
    ) -> Result<Option<u64>, anyhow::Error> {
        let timeout = Duration::from_secs(timeout_in_s);
        let start_time = std::time::Instant::now();

        while start_time.elapsed() < timeout {
            if let Some(wc) = self
                .owner_ref
                .poll_completion(client, caller_ref.clone())
                .await?
            {
                if wc.wr_id() == work_id {
                    tracing::debug!("work for {} completed", work_id);
                    return Ok(Some(work_id));
                }
            }

            RealClock.sleep(Duration::from_millis(1)).await;
        }

        println!("timed out!");
        Err(anyhow::anyhow!(
            "[buffer_{}] rdma operation did not complete in time",
            self.id
        ))
    }

    /// Read from the RdmaBuffer into the provided memory.
    ///
    /// This method transfers data from the buffer into the local memory region provided over RDMA.
    /// This involves calling a `Put` operation on the RdmaBuffer's actor side.
    ///
    /// # Arguments
    /// * `caller_mr` - View of the caller's memory region where data will be read into
    /// * `client` - Mailbox used for communication
    /// * `caller_ref` - Reference to the caller's RdmaManagerActor
    /// * `timeout_in_s` - Timeout in seconds for the RDMA operation to complete. If not set, the work_id is returned
    ///   immediately, allowing the application to handle its own polling behavior.
    ///
    /// # Returns
    /// `Ok(Some(work_id))` if the operation is initiated successfully (and completed if `timeout_in_s` is set),
    /// or an error if the operation fails
    pub async fn read_into(
        &self,
        caller_mr: RdmaMemoryRegionView,
        client: &Mailbox,
        caller_ref: &ActorRef<RdmaManagerActor>,
        timeout_in_s: Option<u32>,
    ) -> Result<Option<u64>, anyhow::Error> {
        tracing::debug!(
            "[buffer_{}] reading from {:?} into caller ({}) at {:?}",
            self.id,
            self.memory_region,
            caller_ref.actor_id(),
            caller_mr,
        );
        self.maybe_connect(caller_ref, client).await?;

        let (_, rkey) = caller_ref.get_keys(client).await?;
        let remote_buffer = RemoteBufferRef::new(caller_ref.clone(), caller_mr, rkey);

        let work_id = self
            .owner_ref
            .put(client, self.memory_region.clone(), remote_buffer)
            .await?;

        if let Some(timeout) = timeout_in_s {
            self.wait_for_completion(work_id, client, caller_ref, timeout as u64)
                .await
        } else {
            Ok(Some(work_id))
        }
    }

    /// Write from the provided memory into the RdmaBuffer.
    ///
    /// This method performs an RDMA write operation, transferring data from the caller's
    /// memory region to this buffer.
    /// This involves calling a `Fetch` operation on the RdmaBuffer's actor side.
    ///
    /// # Arguments
    /// * `caller_mr` - View of the caller's memory region containing the data to write
    /// * `client` - Mailbox used for communication
    /// * `caller_ref` - Reference to the caller's RdmaManagerActor
    /// * `timeout_in_s` - Timeout in seconds for the RDMA operation to complete. If not set, the work_id is returned
    ///   immediately, allowing the application to handle its own polling behavior.
    ///
    /// # Returns
    /// `Ok(Some(work_id))` if the operation is initiated successfully (and completed if `timeout_in_s` is set),
    /// or an error if the operation fails
    pub async fn write_from(
        &self,
        caller_mr: RdmaMemoryRegionView,
        client: &Mailbox,
        caller_ref: &ActorRef<RdmaManagerActor>,
        timeout_in_s: Option<u32>,
    ) -> Result<Option<u64>, anyhow::Error> {
        tracing::debug!(
            "[buffer_{}] writing {:?} from caller {} into buffer at {:?}",
            self.id,
            caller_mr,
            caller_ref.actor_id(),
            self.memory_region,
        );
        self.maybe_connect(caller_ref, client).await?;

        let (_, rkey) = caller_ref.get_keys(client).await?;
        let remote_buffer = RemoteBufferRef::new(caller_ref.clone(), caller_mr, rkey);
        let work_id = self
            .owner_ref
            .fetch(client, self.memory_region.clone(), remote_buffer)
            .await?;
        if let Some(timeout) = timeout_in_s {
            self.wait_for_completion(work_id, client, caller_ref, timeout as u64)
                .await
        } else {
            Ok(Some(work_id))
        }
    }

    /// Releases resources associated with this RDMA buffer.
    ///
    /// This method notifies the RdmaManagerActor to release any resources associated
    /// with the specified memory region associated with the caller.
    ///
    /// # Arguments
    /// * `client` - Mailbox used for communication
    /// * `caller_mr` - View of the caller's memory region to be released
    /// * `caller_ref` - Reference to the caller's RdmaManagerActor
    ///
    /// # Returns
    /// `Ok(())` if the resources are successfully released, or an error if the operation fails
    pub async fn release(
        &self,
        client: &Mailbox,
        caller_mr: RdmaMemoryRegionView,
        caller_ref: &ActorRef<RdmaManagerActor>,
    ) -> Result<(), anyhow::Error> {
        tracing::debug!("[buffer_{}] releasing buffer {:?}", self.id, caller_mr);
        self.owner_ref
            .release(client, caller_ref.clone(), caller_mr)
            .await?;
        Ok(())
    }

    /// Drops the buffer, notifying the owner to release its registered memory regions.
    ///
    /// Note that this does NOT drop underlying queue pairs.
    pub async fn drop_buffer(&self, client: &Mailbox) -> Result<(), anyhow::Error> {
        tracing::debug!("[buffer_{}] dropping buffer", self.id);
        self.owner_ref
            .drop_memory_region(client, self.memory_region.clone())
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use hyperactor_mesh::ActorMesh;
    use hyperactor_mesh::Mesh;
    use hyperactor_mesh::ProcMesh;
    use hyperactor_mesh::alloc::AllocConstraints;
    use hyperactor_mesh::alloc::AllocSpec;
    use hyperactor_mesh::alloc::Allocator;
    use hyperactor_mesh::alloc::LocalAllocator;
    use ibverbs_primitives::get_all_devices;
    use ndslice::shape;

    use super::*;
    use crate::IbverbsConfig;
    use crate::ibverbs_primitives;

    // Helper function to fill a buffer with pseudo-random values
    fn generate_random_data(buffer: &mut [u8], multiplier: u64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        for (i, val) in buffer.iter_mut().enumerate() {
            // Mix timestamp with index and optional multiplier to create different values
            *val = ((timestamp.wrapping_add(i as u64).wrapping_mul(multiplier)) % 256) as u8;
        }
    }

    // Helper function to create test data with pseudo-random values
    fn create_test_data(size: usize) -> Box<[u8]> {
        let mut data = vec![0u8; size].into_boxed_slice();
        generate_random_data(&mut data, 1);
        data
    }

    // Helper function to verify buffer contents match
    fn check_buffers_equal(buffer1: &[u8], buffer2: &[u8]) -> bool {
        if buffer1.len() != buffer2.len() {
            return false;
        }

        for i in 0..buffer1.len() {
            if buffer1[i] != buffer2[i] {
                return false;
            }
        }

        true
    }

    // Verifies that RDMA write operations work correctly
    // It reads data from a local buffer into the destination buffer
    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_rdma_write() -> Result<(), anyhow::Error> {
        const BUFFER_SIZE: usize = 64;

        let devices = get_all_devices();

        let config1: IbverbsConfig;
        let config2: IbverbsConfig;

        if devices.len() == 12 {
            // On H100 machines with 12 devices, use specific devices
            config1 = IbverbsConfig {
                device: devices.clone().into_iter().next().unwrap(),
                ..Default::default()
            };
            // The second device used is the 3rd. Main reason is because 0 and 3 are both backend
            // devices on gtn H100 devices.
            config2 = IbverbsConfig {
                device: devices.clone().into_iter().nth(3).unwrap(),
                ..Default::default()
            };
        } else {
            // For other configurations, use default settings (read/writes will occur on the same ibverbs device)
            println!(
                "using default IbverbsConfig as {} devices were found (expected 12 for H100)",
                devices.len()
            );
            config1 = IbverbsConfig::default();
            config2 = IbverbsConfig::default();
        }

        let buffer1_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();
        let alloc1 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: Default::default(),
            })
            .await?;
        let proc_mesh_1 = ProcMesh::allocate(alloc1).await?;
        let actor_mesh_1: ActorMesh<'_, RdmaManagerActor> =
            proc_mesh_1.spawn("rdma_manager_1", &config1).await.unwrap();
        let buffer2_data = create_test_data(BUFFER_SIZE);
        let alloc2 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: Default::default(),
            })
            .await?;
        let proc_mesh_2 = ProcMesh::allocate(alloc2).await?;
        let actor_mesh_2: ActorMesh<'_, RdmaManagerActor> =
            proc_mesh_2.spawn("rdma_manager_2", &config2).await.unwrap();

        let actor_ref = actor_mesh_1.get(0).unwrap();

        let mut rdma_buffer = RdmaBuffer::new(
            "buffer".to_string(),
            actor_ref,
            proc_mesh_1.client(),
            RdmaMemoryRegionView::from_boxed_slice(&buffer1_data),
        )
        .await?;

        let caller_ref = actor_mesh_2.get(0).unwrap();
        let client = proc_mesh_2.client();
        let caller_mr = RdmaMemoryRegionView::from_boxed_slice(&buffer2_data);

        rdma_buffer
            .write_from(caller_mr.clone(), client, &caller_ref, Some(3))
            .await
            .unwrap();

        assert!(check_buffers_equal(&buffer1_data, &buffer2_data));
        Ok(())
    }

    // Verifies that RDMA read operations work correctly
    // It reads data from the source buffer into the local buffer.
    #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_rdma_read() -> Result<(), anyhow::Error> {
        const BUFFER_SIZE: usize = 64;
        let devices = get_all_devices();

        let config1: IbverbsConfig;
        let config2: IbverbsConfig;

        if devices.len() == 12 {
            // On H100 machines with 12 devices, use specific devices
            config1 = IbverbsConfig {
                device: devices.clone().into_iter().next().unwrap(),
                ..Default::default()
            };
            // The second device used is the 3rd. Main reason is because 0 and 3 are both backend
            // devices on gtn H100 devices.
            config2 = IbverbsConfig {
                device: devices.clone().into_iter().nth(3).unwrap(),
                ..Default::default()
            };
        } else {
            // For other configurations, use default settings (read/writes will occur on the same ibverbs device)
            println!(
                "using default IbverbsConfig as {} devices were found (expected 12 for H100)",
                devices.len()
            );
            config1 = IbverbsConfig::default();
            config2 = IbverbsConfig::default();
        }

        let buffer1_data = create_test_data(BUFFER_SIZE);
        let alloc1 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: Default::default(),
            })
            .await?;
        let proc_mesh_1 = ProcMesh::allocate(alloc1).await?;
        let actor_mesh_1: ActorMesh<'_, RdmaManagerActor> =
            proc_mesh_1.spawn("rdma_manager_1", &config1).await.unwrap();
        let buffer2_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();
        let alloc2 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: Default::default(),
            })
            .await?;
        let proc_mesh_2 = ProcMesh::allocate(alloc2).await?;
        let actor_mesh_2: ActorMesh<'_, RdmaManagerActor> =
            proc_mesh_2.spawn("rdma_manager_2", &config2).await.unwrap();

        let actor_ref = actor_mesh_1.get(0).unwrap();

        let mut rdma_buffer = RdmaBuffer::new(
            "buffer".to_string(),
            actor_ref,
            proc_mesh_1.client(),
            RdmaMemoryRegionView::from_boxed_slice(&buffer1_data),
        )
        .await?;

        let caller_ref = actor_mesh_2.get(0).unwrap();
        let client = proc_mesh_2.client();
        let caller_mr = RdmaMemoryRegionView::from_boxed_slice(&buffer2_data);

        rdma_buffer
            .read_into(caller_mr.clone(), client, &caller_ref, Some(3))
            .await
            .unwrap();

        assert!(check_buffers_equal(&buffer1_data, &buffer2_data));
        Ok(())
    }

    // TODO - this currently fails, as handling Actor level errors hasn't been implemented
    // #[timed_test::async_timed_test(timeout_secs = 60)]
    async fn test_buffer_drop() -> Result<(), anyhow::Error> {
        const BUFFER_SIZE: usize = 64;
        let devices = get_all_devices();

        let config1: IbverbsConfig;
        let config2: IbverbsConfig;

        if devices.len() == 12 {
            // On H100 machines with 12 devices, use specific devices
            config1 = IbverbsConfig {
                device: devices.clone().into_iter().next().unwrap(),
                ..Default::default()
            };
            // The second device used is the 3rd. Main reason is because 0 and 3 are both backend
            // devices on gtn H100 devices.
            config2 = IbverbsConfig {
                device: devices.clone().into_iter().nth(3).unwrap(),
                ..Default::default()
            };
        } else {
            // For other configurations, use default settings (read/writes will occur on the same ibverbs device)
            println!(
                "using default IbverbsConfig as {} devices were found (expected 12 for H100)",
                devices.len()
            );
            config1 = IbverbsConfig::default();
            config2 = IbverbsConfig::default();
        }

        let buffer1_data = create_test_data(BUFFER_SIZE);
        let alloc1 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: Default::default(),
            })
            .await?;
        let proc_mesh_1 = ProcMesh::allocate(alloc1).await?;
        let actor_mesh_1: ActorMesh<'_, RdmaManagerActor> =
            proc_mesh_1.spawn("rdma_manager_1", &config1).await.unwrap();
        let buffer2_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();
        let alloc2 = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! {replica=1, host=1, gpu=1},
                constraints: Default::default(),
            })
            .await?;
        let proc_mesh_2 = ProcMesh::allocate(alloc2).await?;
        let actor_mesh_2: ActorMesh<'_, RdmaManagerActor> =
            proc_mesh_2.spawn("rdma_manager_2", &config2).await.unwrap();

        let actor_ref = actor_mesh_1.get(0).unwrap();

        let mut rdma_buffer = RdmaBuffer::new(
            "buffer".to_string(),
            actor_ref.clone(),
            proc_mesh_1.client(),
            RdmaMemoryRegionView::from_boxed_slice(&buffer1_data),
        )
        .await?;

        let caller_ref = actor_mesh_2.get(0).unwrap();
        let client = proc_mesh_2.client();
        let caller_mr = RdmaMemoryRegionView::from_boxed_slice(&buffer2_data);

        rdma_buffer
            .read_into(caller_mr.clone(), client, &caller_ref, Some(3))
            .await
            .unwrap();

        assert!(check_buffers_equal(&buffer1_data, &buffer2_data));
        tracing::info!("buffers are equal. dropping");

        // Drop the buffer
        rdma_buffer.drop_buffer(client).await?;

        // Try to use the buffer after dropping - should fail
        let result = rdma_buffer
            .read_into(caller_mr, client, &caller_ref, Some(3))
            .await;

        assert!(
            result.is_err(),
            "Operation should fail after buffer is dropped"
        );

        Ok(())
    }
}
