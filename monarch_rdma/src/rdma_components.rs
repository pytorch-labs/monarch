/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # RDMA Components
//!
//! This module provides the core RDMA building blocks for establishing and managing RDMA connections.
//!
//! ## Core Components
//!
//! * `RdmaDomain` - Manages RDMA resources including context, protection domain, and memory region
//! * `RdmaQueuePair` - Handles communication between endpoints via queue pairs and completion queues
//!
//! ## RDMA Overview
//!
//! Remote Direct Memory Access (RDMA) allows direct memory access from the memory of one computer
//! into the memory of another without involving either computer's operating system. This permits
//! high-throughput, low-latency networking with minimal CPU overhead.
//!
//! ## Connection Architecture
//!
//! The module manages the following ibverbs primitives:
//!
//! 1. **Queue Pairs (QP)**: Each connection has a send queue and a receive queue
//! 2. **Completion Queues (CQ)**: Events are reported when operations complete
//! 3. **Memory Regions (MR)**: Memory must be registered with the RDMA device before use
//! 4. **Protection Domains (PD)**: Provide isolation between different connections
//!
//! ## Connection Lifecycle
//!
//! 1. Create an `RdmaDomain` with `new()`
//! 2. Create an `RdmaQueuePair` from the domain
//! 3. Exchange connection info with remote peer (application must handle this)
//! 4. Connect to remote endpoint with `connect()`
//! 5. Perform RDMA operations (read/write)
//! 6. Poll for completions
//! 7. Resources are cleaned up when dropped

use std::collections::HashMap;
use std::ffi::CStr;
use std::io::Error;
use std::result::Result;
use std::time::Duration;

use hyperactor::ActorRef;
use hyperactor::Mailbox;
use hyperactor::Named;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
/// Direct access to low-level libibverbs rdmacore_sys.
use rdmacore_sys::ibv_qp_type;
use serde::Deserialize;
use serde::Serialize;

use crate::RdmaDevice;
use crate::RdmaManagerActor;
use crate::RdmaManagerMessageClient;
use crate::ibverbs_primitives::Gid;
use crate::ibverbs_primitives::IbvWc;
use crate::ibverbs_primitives::IbverbsConfig;
use crate::ibverbs_primitives::RdmaMemoryRegionView;
use crate::ibverbs_primitives::RdmaOperation;
use crate::ibverbs_primitives::RdmaQpInfo;

#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct RdmaBuffer {
    pub owner: ActorRef<RdmaManagerActor>,
    pub mr_id: u32,
    pub lkey: u32,
    pub rkey: u32,
    pub addr: usize,
    pub size: usize,
}

impl RdmaBuffer {
    /// Read from the RdmaBuffer into the provided memory.
    ///
    /// This method transfers data from the buffer into the local memory region provided over RDMA.
    /// This involves calling a `Put` operation on the RdmaBuffer's actor side.
    ///
    /// # Arguments
    /// * `client` - Mailbox used for communication
    /// * `remote` - RdmaBuffer representing the remote memory region
    /// * `timeout` - Timeout in seconds for the RDMA operation to complete.
    ///
    /// # Returns
    /// `Ok(bool)` indicating if the operation completed successfully.
    pub async fn read_into(
        &self,
        client: &Mailbox,
        remote: RdmaBuffer,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        tracing::debug!(
            "[buffer] reading from {:?} into remote ({:?}) at {:?}",
            self,
            remote.owner.actor_id(),
            remote,
        );
        let mut qp = self
            .owner
            .request_queue_pair(client, remote.owner.clone())
            .await?;

        qp.put(self.clone(), remote)?;
        self.wait_for_completion(qp, timeout).await
    }

    /// Write from the provided memory into the RdmaBuffer.
    ///
    /// This method performs an RDMA write operation, transferring data from the caller's
    /// memory region to this buffer.
    /// This involves calling a `Fetch` operation on the RdmaBuffer's actor side.
    ///
    /// # Arguments
    /// * `client` - Mailbox used for communication
    /// * `remote` - RdmaBuffer representing the remote memory region
    /// * `timeout` - Timeout in seconds for the RDMA operation to complete.
    ///
    /// # Returns
    /// `Ok(bool)` indicating if the operation completed successfully.
    pub async fn write_from(
        &self,
        client: &Mailbox,
        remote: RdmaBuffer,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        tracing::debug!(
            "[buffer] writing into {:?} from remote ({:?}) at {:?}",
            self,
            remote.owner.actor_id(),
            remote,
        );
        let mut qp = self
            .owner
            .request_queue_pair(client, remote.owner.clone())
            .await?;
        qp.get(self.clone(), remote)?;
        self.wait_for_completion(qp, timeout).await
    }
    /// Waits for the completion of an RDMA operation.
    ///
    /// This method polls the completion queue until the specified work request completes
    /// or until the timeout is reached.
    ///
    /// # Arguments
    /// * `qp` - The RDMA Queue Pair to poll for completion
    /// * `timeout` - Timeout in seconds for the RDMA operation to complete.
    ///
    /// # Returns
    /// `Ok(true)` if the operation completes successfully within the timeout,
    /// or an error if the timeout is reached
    async fn wait_for_completion(
        &self,
        qp: RdmaQueuePair,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        let timeout = Duration::from_secs(timeout);
        let start_time = std::time::Instant::now();

        while start_time.elapsed() < timeout {
            match qp.poll_completion() {
                Ok(Some(wc)) => {
                    if wc.wr_id() == 0 {
                        tracing::debug!("work completed");
                        return Ok(true);
                    }
                }
                Ok(None) => {
                    RealClock.sleep(Duration::from_millis(1)).await;
                }
                Err(e) => {
                    tracing::error!("polling completion failed: {}", e);
                    return Err(anyhow::anyhow!(e));
                }
            }
        }
        tracing::error!("timed out while waiting on request completion");
        Err(anyhow::anyhow!(
            "[buffer({:?})] rdma operation did not complete in time",
            self
        ))
    }
}

/// Represents a domain for RDMA operations, encapsulating the necessary resources
/// for establishing and managing RDMA connections.
///
/// An `RdmaDomain` manages the context, protection domain (PD), and memory region (MR)
/// required for RDMA operations. It provides the foundation for creating queue pairs
/// and establishing connections between RDMA devices.
///
/// # Fields
///
/// * `context`: A pointer to the RDMA device context, representing the connection to the RDMA device.
/// * `pd`: A pointer to the protection domain, which provides isolation between different connections.
/// * `mr_map`: A map of memory region IDs to pointers, representing registered memory regions.
/// * `counter`: A counter for generating unique memory region IDs.
pub struct RdmaDomain {
    pub context: *mut rdmacore_sys::ibv_context,
    pub pd: *mut rdmacore_sys::ibv_pd,
    mr_map: HashMap<u32, *mut rdmacore_sys::ibv_mr>,
    counter: u32,
}

impl std::fmt::Debug for RdmaDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RdmaDomain")
            .field("context", &format!("{:p}", self.context))
            .field("pd", &format!("{:p}", self.pd))
            .field("mr", &format!("{:?}", self.mr_map))
            .field("counter", &self.counter)
            .finish()
    }
}

// SAFETY:
// This function contains code marked unsafe as it interacts with the Rdma device through rdmacore_sys calls.
// RdmaDomain is `Send` because the raw pointers to ibverbs structs can be
// accessed from any thread, and it is safe to drop `RdmaDomain` (and run the
// ibverbs destructors) from any thread.
unsafe impl Send for RdmaDomain {}

// SAFETY:
// This function contains code marked unsafe as it interacts with the Rdma device through rdmacore_sys calls.
// RdmaDomain is `Sync` because the underlying ibverbs APIs are thread-safe.
unsafe impl Sync for RdmaDomain {}

impl Drop for RdmaDomain {
    fn drop(&mut self) {
        unsafe {
            rdmacore_sys::ibv_dealloc_pd(self.pd);
        }
    }
}

impl RdmaDomain {
    /// Creates a new RdmaDomain.
    ///
    /// This function initializes the RDMA device context, creates a protection domain,
    /// and registers a memory region with appropriate access permissions.
    ///
    /// SAFETY:
    /// Our memory region (MR) registration uses implicit ODP for RDMA access, which maps large virtual
    /// address ranges without explicit pinning. This is convenient, but it broadens the memory footprint
    /// exposed to the NIC and introduces a security liability.
    ///
    /// We currently assume a trusted, single-environment and are not enforcing finer-grained memory isolation
    /// at this layer. We plan to investigate mitigations - such as memory windows or tighter registration
    /// boundaries in future follow-ups.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration settings for the RDMA operations
    ///
    /// # Errors
    ///
    /// This function may return errors if:
    /// * No RDMA devices are found
    /// * The specified device cannot be found
    /// * Device context creation fails
    /// * Protection domain allocation fails
    /// * Memory region registration fails
    pub fn new(device: RdmaDevice) -> Result<Self, anyhow::Error> {
        tracing::debug!("creating RdmaDomain for device {}", device.name());
        // SAFETY:
        // This code uses unsafe rdmacore_sys calls to interact with the RDMA device, but is safe because:
        // - All pointers are properly initialized and checked for null before use
        // - Memory registration follows the ibverbs API contract with proper access flags
        // - Resources are properly cleaned up in error cases to prevent leaks
        // - The operations follow the documented RDMA protocol for device initialization
        unsafe {
            // Get the device based on the provided RdmaDevice
            let device_name = device.name();
            let mut num_devices = 0i32;
            let devices = rdmacore_sys::ibv_get_device_list(&mut num_devices as *mut _);

            if devices.is_null() || num_devices == 0 {
                return Err(anyhow::anyhow!("no RDMA devices found"));
            }

            // Find the device with the matching name
            let mut device_ptr = std::ptr::null_mut();
            for i in 0..num_devices {
                let dev = *devices.offset(i as isize);
                let dev_name =
                    CStr::from_ptr(rdmacore_sys::ibv_get_device_name(dev)).to_string_lossy();

                if dev_name == *device_name {
                    device_ptr = dev;
                    break;
                }
            }

            // If we didn't find the device, return an error
            if device_ptr.is_null() {
                rdmacore_sys::ibv_free_device_list(devices);
                return Err(anyhow::anyhow!("device '{}' not found", device_name));
            }
            tracing::info!("using RDMA device: {}", device_name);

            // Open device
            let context = rdmacore_sys::ibv_open_device(device_ptr);
            if context.is_null() {
                rdmacore_sys::ibv_free_device_list(devices);
                let os_error = Error::last_os_error();
                return Err(anyhow::anyhow!("failed to create context: {}", os_error));
            }

            // Create protection domain
            let pd = rdmacore_sys::ibv_alloc_pd(context);
            if pd.is_null() {
                rdmacore_sys::ibv_close_device(context);
                rdmacore_sys::ibv_free_device_list(devices);
                let os_error = Error::last_os_error();
                return Err(anyhow::anyhow!(
                    "failed to create protection domain (PD): {}",
                    os_error
                ));
            }

            // Avoids memory leaks
            rdmacore_sys::ibv_free_device_list(devices);

            Ok(RdmaDomain {
                context,
                pd,
                mr_map: HashMap::new(),
                counter: 0,
            })
        }
    }

    fn register_mr(
        &mut self,
        addr: usize,
        size: usize,
    ) -> Result<RdmaMemoryRegionView, anyhow::Error> {
        unsafe {
            let mut mem_type: i32 = 0;
            let ptr = addr as cuda_sys::CUdeviceptr;
            let err = cuda_sys::cuPointerGetAttribute(
                &mut mem_type as *mut _ as *mut std::ffi::c_void,
                cuda_sys::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                ptr,
            );
            let is_cuda = err == cuda_sys::CUresult::CUDA_SUCCESS;

            let access = rdmacore_sys::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
                | rdmacore_sys::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
                | rdmacore_sys::ibv_access_flags::IBV_ACCESS_REMOTE_READ
                | rdmacore_sys::ibv_access_flags::IBV_ACCESS_REMOTE_ATOMIC;

            let mr;
            if is_cuda {
                let mut fd: i32 = -1;
                cuda_sys::cuMemGetHandleForAddressRange(
                    &mut fd as *mut i32 as *mut std::ffi::c_void,
                    addr as cuda_sys::CUdeviceptr,
                    size,
                    cuda_sys::CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                    0,
                );
                mr = rdmacore_sys::ibv_reg_dmabuf_mr(self.pd, 0, size, 0, fd, access.0 as i32);
            } else {
                mr = rdmacore_sys::ibv_reg_mr(
                    self.pd,
                    addr as *mut std::ffi::c_void,
                    size,
                    access.0 as i32,
                );
            }

            if mr.is_null() {
                return Err(anyhow::anyhow!("failed to register memory region (MR)"));
            }
            let id = self.counter;
            self.mr_map.insert(id, mr);
            self.counter += 1;

            Ok(RdmaMemoryRegionView {
                id,
                addr: (*mr).addr as usize,
                size: (*mr).length,
                lkey: (*mr).lkey,
                rkey: (*mr).rkey,
            })
        }
    }

    fn deregister_mr(&mut self, id: u32) -> Result<(), anyhow::Error> {
        let mr = self.mr_map.remove(&id);
        if mr.is_some() {
            unsafe {
                rdmacore_sys::ibv_dereg_mr(mr.expect("mr is required"));
            }
        }
        Ok(())
    }

    pub fn register_buffer(
        &mut self,
        addr: usize,
        size: usize,
    ) -> Result<RdmaMemoryRegionView, anyhow::Error> {
        let region_view = self.register_mr(addr, size)?;
        Ok(region_view)
    }

    // Removes a specific address from memory region.   Currently we only support single address,
    // but in future we can expand/contract effective memory region.
    pub fn deregister_buffer(&mut self, buffer: RdmaBuffer) -> Result<(), anyhow::Error> {
        self.deregister_mr(buffer.mr_id)?;
        Ok(())
    }
}

/// Represents an RDMA Queue Pair (QP) that enables communication between two endpoints.
///
/// An `RdmaQueuePair` encapsulates the send and receive queues, completion queue,
/// and other resources needed for RDMA communication. It provides methods for
/// establishing connections and performing RDMA operations like read and write.
///
/// # Fields
///
/// * `cq` - Completion Queue pointer for tracking operation completions
/// * `qp` - Queue Pair pointer that manages send and receive operations
/// * `context` - RDMA device context pointer
/// * `config` - Configuration settings for the queue pair
/// * `lkey` - Local key for memory region access
/// * `rkey` - Remote key for memory region access
///
/// # Connection Lifecycle
///
/// 1. Create with `new()` from an `RdmaDomain`
/// 2. Get connection info with `get_qp_info()`
/// 3. Exchange connection info with remote peer (application must handle this)
/// 4. Connect to remote endpoint with `connect()`
/// 5. Perform RDMA operations with `post_send()`
/// 6. Poll for completions with `poll_completion()`

#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct RdmaQueuePair {
    cq: usize,      // *mut rdmacore_sys::ibv_cq,
    qp: usize,      // *mut rdmacore_sys::ibv_qp,
    context: usize, // *mut rdmacore_sys::ibv_context,
    config: IbverbsConfig,
}

impl RdmaQueuePair {
    /// Creates a new RdmaQueuePair from a given RdmaDomain.
    ///
    /// This function initializes a new Queue Pair (QP) and associated Completion Queue (CQ)
    /// using the resources from the provided RdmaDomain. The QP is created in the RESET state
    /// and must be transitioned to other states via the `connect()` method before use.
    ///
    /// # Arguments
    ///
    /// * `domain` - Reference to an RdmaDomain that provides the context, protection domain,
    ///   and memory region for this queue pair
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - A new RdmaQueuePair instance or an error if creation fails
    ///
    /// # Errors
    ///
    /// This function may return errors if:
    /// * Completion queue (CQ) creation fails
    /// * Queue pair (QP) creation fails
    pub fn new(
        context: *mut rdmacore_sys::ibv_context,
        pd: *mut rdmacore_sys::ibv_pd,
        config: IbverbsConfig,
    ) -> Result<Self, anyhow::Error> {
        tracing::debug!("creating an RdmaQueuePair from config {}", config);
        // SAFETY:
        // This code uses unsafe rdmacore_sys calls to interact with the RDMA device, but is safe because:
        // - All pointers are properly initialized and checked for null before use
        // - Resources (CQ, QP) are created following the ibverbs API contract
        // - Error handling properly cleans up resources in failure cases
        // - The operations follow the documented RDMA protocol for queue pair initialization
        unsafe {
            let cq = rdmacore_sys::ibv_create_cq(
                context,
                config.cq_entries,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            if cq.is_null() {
                let os_error = Error::last_os_error();
                return Err(anyhow::anyhow!(
                    "failed to create completion queue (CQ): {}",
                    os_error
                ));
            }

            // Create queue pair - note we currently share a CQ for both send and receive for simplicity.
            let mut qp_init_attr = rdmacore_sys::ibv_qp_init_attr {
                qp_context: std::ptr::null::<std::os::raw::c_void>() as *mut _,
                send_cq: cq,
                recv_cq: cq,
                srq: std::ptr::null::<rdmacore_sys::ibv_srq>() as *mut _,
                cap: rdmacore_sys::ibv_qp_cap {
                    max_send_wr: config.max_send_wr,
                    max_recv_wr: config.max_recv_wr,
                    max_send_sge: config.max_send_sge,
                    max_recv_sge: config.max_recv_sge,
                    max_inline_data: 0,
                },
                qp_type: ibv_qp_type::IBV_QPT_RC,
                sq_sig_all: 0,
            };

            let qp = rdmacore_sys::ibv_create_qp(pd, &mut qp_init_attr);
            if qp.is_null() {
                rdmacore_sys::ibv_destroy_cq(cq);
                let os_error = Error::last_os_error();
                return Err(anyhow::anyhow!(
                    "failed to create queue pair (QP): {}",
                    os_error
                ));
            }
            Ok(RdmaQueuePair {
                cq: cq as usize,
                qp: qp as usize,
                context: context as usize,
                config,
            })
        }
    }

    /// Returns the information required for a remote peer to connect to this queue pair.
    ///
    /// This method retrieves the local queue pair attributes and port information needed by
    /// a remote peer to establish an RDMA connection. The returned `RdmaQpInfo` contains
    /// the queue pair number, LID, GID, and other necessary connection parameters.
    ///
    /// # Returns
    ///
    /// * `Result<RdmaQpInfo>` - Connection information for the remote peer or an error
    ///
    /// # Errors
    ///
    /// This function may return errors if:
    /// * Port attribute query fails
    /// * GID query fails
    pub fn get_qp_info(&mut self) -> Result<RdmaQpInfo, anyhow::Error> {
        // SAFETY:
        // This code uses unsafe rdmacore_sys calls to query RDMA device information, but is safe because:
        // - All pointers are properly initialized before use
        // - Port and GID queries follow the documented ibverbs API contract
        // - Error handling properly checks return codes from ibverbs functions
        // - The memory address provided is only stored, not dereferenced in this function
        unsafe {
            let context = self.context as *mut rdmacore_sys::ibv_context;
            let qp = self.qp as *mut rdmacore_sys::ibv_qp;
            let mut port_attr = rdmacore_sys::ibv_port_attr::default();
            let errno = rdmacore_sys::ibv_query_port(
                context,
                self.config.port_num,
                &mut port_attr as *mut rdmacore_sys::ibv_port_attr as *mut _,
            );
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(anyhow::anyhow!(
                    "Failed to query port attributes: {}",
                    os_error
                ));
            }

            let mut gid = Gid::default();
            let ret = rdmacore_sys::ibv_query_gid(
                context,
                self.config.port_num,
                i32::from(self.config.gid_index),
                gid.as_mut(),
            );
            if ret != 0 {
                return Err(anyhow::anyhow!("Failed to query GID"));
            }

            Ok(RdmaQpInfo {
                qp_num: (*qp).qp_num,
                lid: port_attr.lid,
                gid: Some(gid),
                psn: self.config.psn,
            })
        }
    }

    pub fn state(&mut self) -> Result<u32, anyhow::Error> {
        // SAFETY: This block interacts with the RDMA device through rdmacore_sys calls.
        unsafe {
            let qp = self.qp as *mut rdmacore_sys::ibv_qp;
            let mut qp_attr = rdmacore_sys::ibv_qp_attr {
                ..Default::default()
            };
            let mut qp_init_attr = rdmacore_sys::ibv_qp_init_attr {
                ..Default::default()
            };
            let mask = rdmacore_sys::ibv_qp_attr_mask::IBV_QP_STATE;
            let errno =
                rdmacore_sys::ibv_query_qp(qp, &mut qp_attr, mask.0 as i32, &mut qp_init_attr);
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(anyhow::anyhow!("failed to query QP state: {}", os_error));
            }
            Ok(qp_attr.qp_state)
        }
    }
    /// Connect to a remote Rdma connection point.
    ///
    /// This performs the necessary QP state transitions (INIT->RTR->RTS) to establish a connection.
    ///
    /// # Arguments
    ///
    /// * `connection_info` - The remote connection info to connect to
    pub fn connect(&mut self, connection_info: &RdmaQpInfo) -> Result<(), anyhow::Error> {
        // SAFETY:
        // This unsafe block is necessary because we're interacting with the RDMA device through rdmacore_sys calls.
        // The operations are safe because:
        // 1. We're following the documented ibverbs API contract
        // 2. All pointers used are properly initialized and owned by this struct
        // 3. The QP state transitions (INIT->RTR->RTS) follow the required RDMA connection protocol
        // 4. Memory access is properly bounded by the registered memory regions
        unsafe {
            // Transition to INIT
            let qp = self.qp as *mut rdmacore_sys::ibv_qp;

            let qp_access_flags = rdmacore_sys::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
                | rdmacore_sys::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
                | rdmacore_sys::ibv_access_flags::IBV_ACCESS_REMOTE_READ;

            let mut qp_attr = rdmacore_sys::ibv_qp_attr {
                qp_state: rdmacore_sys::ibv_qp_state::IBV_QPS_INIT,
                qp_access_flags: qp_access_flags.0,
                pkey_index: self.config.pkey_index,
                port_num: self.config.port_num,
                ..Default::default()
            };

            let mask = rdmacore_sys::ibv_qp_attr_mask::IBV_QP_STATE
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_PORT
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;

            let errno = rdmacore_sys::ibv_modify_qp(qp, &mut qp_attr, mask.0 as i32);
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(anyhow::anyhow!(
                    "failed to transition QP to INIT: {}",
                    os_error
                ));
            }

            // Transition to RTR (Ready to Receive)
            let mut qp_attr = rdmacore_sys::ibv_qp_attr {
                qp_state: rdmacore_sys::ibv_qp_state::IBV_QPS_RTR,
                path_mtu: self.config.path_mtu,
                dest_qp_num: connection_info.qp_num,
                rq_psn: connection_info.psn,
                max_dest_rd_atomic: self.config.max_dest_rd_atomic,
                min_rnr_timer: self.config.min_rnr_timer,
                ah_attr: rdmacore_sys::ibv_ah_attr {
                    dlid: connection_info.lid,
                    sl: 0,
                    src_path_bits: 0,
                    port_num: self.config.port_num,
                    grh: Default::default(),
                    ..Default::default()
                },
                ..Default::default()
            };

            // If the remote connection info contains a Gid, the routing will be global.
            // Otherwise, it will be local, i.e. using LID.
            if let Some(gid) = connection_info.gid {
                qp_attr.ah_attr.is_global = 1;
                qp_attr.ah_attr.grh.dgid = gid.into();
                qp_attr.ah_attr.grh.hop_limit = 0xff;
                qp_attr.ah_attr.grh.sgid_index = self.config.gid_index;
            } else {
                // Use LID-based routing, e.g. for Infiniband/RoCEv1
                qp_attr.ah_attr.is_global = 0;
            }

            let mask = rdmacore_sys::ibv_qp_attr_mask::IBV_QP_STATE
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_AV
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_PATH_MTU
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_DEST_QPN
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_RQ_PSN
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;

            let errno = rdmacore_sys::ibv_modify_qp(qp, &mut qp_attr, mask.0 as i32);
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(anyhow::anyhow!(
                    "failed to transition QP to RTR: {}",
                    os_error
                ));
            }

            // Transition to RTS (Ready to Send)
            let mut qp_attr = rdmacore_sys::ibv_qp_attr {
                qp_state: rdmacore_sys::ibv_qp_state::IBV_QPS_RTS,
                sq_psn: self.config.psn,
                max_rd_atomic: self.config.max_rd_atomic,
                retry_cnt: self.config.retry_cnt,
                rnr_retry: self.config.rnr_retry,
                timeout: self.config.qp_timeout,
                ..Default::default()
            };

            let mask = rdmacore_sys::ibv_qp_attr_mask::IBV_QP_STATE
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_TIMEOUT
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_RETRY_CNT
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_SQ_PSN
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_RNR_RETRY
                | rdmacore_sys::ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;

            let errno = rdmacore_sys::ibv_modify_qp(qp, &mut qp_attr, mask.0 as i32);
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(anyhow::anyhow!(
                    "failed to transition QP to RTS: {}",
                    os_error
                ));
            }
            tracing::debug!(
                "connection sequence has successfully completed (qp: {:?})",
                qp
            );

            Ok(())
        }
    }

    pub fn put(&mut self, lhandle: RdmaBuffer, rhandle: RdmaBuffer) -> Result<(), anyhow::Error> {
        self.post_op(
            lhandle.addr,
            lhandle.lkey,
            lhandle.size,
            0,
            true,
            RdmaOperation::Write,
            rhandle.addr,
            rhandle.rkey,
        )
        .unwrap();
        Ok(())
    }

    pub fn get(&mut self, lhandle: RdmaBuffer, rhandle: RdmaBuffer) -> Result<(), anyhow::Error> {
        self.post_op(
            lhandle.addr,
            lhandle.lkey,
            lhandle.size,
            0,
            true,
            RdmaOperation::Read,
            rhandle.addr,
            rhandle.rkey,
        )
        .unwrap();
        Ok(())
    }

    /// Posts a request to the queue pair.
    ///
    /// # Arguments
    ///
    /// * `local_addr` - The local address containing data to send
    /// * `length` - Length of the data to send
    /// * `wr_id` - Work request ID for completion identification
    /// * `signaled` - Whether to generate a completion event
    /// * `op_type` - Optional operation type
    /// * `raddr` - the remote address, representing the memory location on the remote peer
    /// * `rkey` - the remote key, representing the key required to access the remote memory region
    fn post_op(
        &mut self,
        laddr: usize,
        lkey: u32,
        length: usize,
        wr_id: u64,
        signaled: bool,
        op_type: RdmaOperation,
        raddr: usize,
        rkey: u32,
    ) -> Result<(), anyhow::Error> {
        // SAFETY:
        // This code uses unsafe rdmacore_sys calls to post work requests to the RDMA device, but is safe because:
        // - All pointers (send_sge, send_wr) are properly initialized on the stack before use
        // - The memory address in `local_addr` is not dereferenced, only passed to the device
        // - The remote connection info is verified to exist before accessing
        // - The ibverbs post_send operation follows the documented API contract
        // - Error codes from the device are properly checked and propagated
        unsafe {
            let qp = self.qp as *mut rdmacore_sys::ibv_qp;
            let context = self.context as *mut rdmacore_sys::ibv_context;
            let mut send_sge = rdmacore_sys::ibv_sge {
                addr: laddr as u64,
                length: length as u32,
                lkey,
            };

            let send_flags = if signaled {
                rdmacore_sys::ibv_send_flags::IBV_SEND_SIGNALED.0
            } else {
                0
            };

            let mut send_wr = rdmacore_sys::ibv_send_wr {
                wr_id,
                next: std::ptr::null_mut(),
                sg_list: &mut send_sge as *mut _,
                num_sge: 1,
                opcode: op_type.into(),
                send_flags,
                wr: Default::default(),
                qp_type: Default::default(),
                __bindgen_anon_1: Default::default(),
                __bindgen_anon_2: Default::default(),
            };

            // Set remote address and rkey for RDMA operations
            send_wr.wr.rdma.remote_addr = raddr as u64;
            send_wr.wr.rdma.rkey = rkey;
            let mut bad_send_wr: *mut rdmacore_sys::ibv_send_wr = std::ptr::null_mut();
            let ops = &mut (*context).ops;
            let errno =
                ops.post_send.as_mut().unwrap()(qp, &mut send_wr as *mut _, &mut bad_send_wr);

            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(anyhow::anyhow!("Failed to post send request: {}", os_error));
            }
            tracing::debug!(
                "completed sending {:?} request (lkey: {}, addr: 0x{:x}, length {}) to (raddr 0x{:x}, rkey {})",
                op_type,
                lkey,
                laddr,
                length,
                raddr,
                rkey,
            );

            Ok(())
        }
    }

    /// Polls the completion queue for a completion event.
    ///
    /// This function performs a single poll of the completion queue and returns the result.
    /// It does not perform any timing or retry logic - the application is responsible for
    /// implementing any polling strategy (timeouts, retries, etc.).
    ///
    /// Note - while this method does not mutate the Rust struct (e.g. RdmaQueuePair),
    /// it does consume work completions from the underlying ibverbs completion queue (CQ)
    /// as a side effect. This is thread-safe, but may affect concurrent polls on
    /// the same completion queue.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(wc))` - A completion was found
    /// * `Ok(None)` - No completion was found
    /// * `Err(e)` - An error occurred
    pub fn poll_completion(&self) -> Result<Option<IbvWc>, anyhow::Error> {
        // SAFETY:
        // This code uses unsafe rdmacore_sys calls to poll the completion queue, but is safe because:
        // - The completion queue pointer is properly initialized and owned by this struct
        // - The work completion structure is properly zeroed before use
        // - We only access the completion queue through the documented ibverbs API
        // - Error codes from polling operations are properly checked and propagated
        // - The work completion validity is verified before returning it to the caller
        unsafe {
            let context = self.context as *mut rdmacore_sys::ibv_context;
            let cq = self.cq as *mut rdmacore_sys::ibv_cq;
            let mut wc = std::mem::MaybeUninit::<rdmacore_sys::ibv_wc>::zeroed().assume_init();
            let ops = &mut (*context).ops;

            let ret = ops.poll_cq.as_mut().unwrap()(cq, 1, &mut wc);

            if ret < 0 {
                return Err(anyhow::anyhow!(
                    "Failed to poll CQ: {}",
                    Error::last_os_error()
                ));
            }

            if ret > 0 {
                if !wc.is_valid() {
                    if let Some((status, vendor_err)) = wc.error() {
                        return Err(anyhow::anyhow!(
                            "Work completion failed with status: {:?}, vendor error: {}",
                            status,
                            vendor_err
                        ));
                    }
                }
                return Ok(Some(IbvWc::from(wc)));
            }

            // No completion found
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_connection() {
        let config = IbverbsConfig::default();
        let domain = RdmaDomain::new(config.device.clone());
        assert!(domain.is_ok());

        let domain = domain.unwrap();
        let queue_pair = RdmaQueuePair::new(domain.context, domain.pd, config.clone());
        assert!(queue_pair.is_ok());
    }

    #[test]
    fn test_loopback_connection() {
        let server_config = IbverbsConfig::default();
        let client_config = IbverbsConfig::default();

        let server_domain = RdmaDomain::new(server_config.device.clone()).unwrap();
        let client_domain = RdmaDomain::new(client_config.device.clone()).unwrap();

        let mut server_qp = RdmaQueuePair::new(
            server_domain.context,
            server_domain.pd,
            server_config.clone(),
        )
        .unwrap();
        let mut client_qp = RdmaQueuePair::new(
            client_domain.context,
            client_domain.pd,
            client_config.clone(),
        )
        .unwrap();

        let server_connection_info = server_qp.get_qp_info().unwrap();
        let client_connection_info = client_qp.get_qp_info().unwrap();

        assert!(server_qp.connect(&client_connection_info).is_ok());
        assert!(client_qp.connect(&server_connection_info).is_ok());
    }
}
