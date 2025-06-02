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
use std::ffi::CStr;
use std::io::Error;
use std::io::Result;

/// Direct access to low-level libibverbs FFI.
use ffi::ibv_qp_type;
use ibverbs::Gid;

use crate::ibverbs_primitives::IbvWc;
use crate::ibverbs_primitives::IbverbsConfig;
use crate::ibverbs_primitives::RdmaOperation;
use crate::ibverbs_primitives::RdmaQpInfo;

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
/// * `mr`: A pointer to the memory region, which must be registered with the RDMA device before use.
/// * `config`: Configuration settings for the RDMA operations.
/// * `lkey`: Local key for the memory region, used in local RDMA operations.
/// * `rkey`: Remote key for the memory region, used when remote peers access this memory region.
pub struct RdmaDomain {
    context: *mut ffi::ibv_context,
    pd: *mut ffi::ibv_pd,
    mr: *mut ffi::ibv_mr,
    config: IbverbsConfig,
    lkey: u32,
    rkey: u32,
}

impl std::fmt::Debug for RdmaDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RdmaDomain")
            .field("context", &format!("{:p}", self.context))
            .field("pd", &format!("{:p}", self.pd))
            .field("mr", &format!("{:p}", self.mr))
            .field("config", &self.config)
            .field("lkey", &self.lkey)
            .field("rkey", &self.rkey)
            .finish()
    }
}

// SAFETY:
// This function contains code marked unsafe as it interacts with the Rdma device through FFI calls.
// RdmaDomain is `Send` because the raw pointers to ibverbs structs can be
// accessed from any thread, and it is safe to drop `RdmaDomain` (and run the
// ibverbs destructors) from any thread.
unsafe impl Send for RdmaDomain {}

// SAFETY:
// This function contains code marked unsafe as it interacts with the Rdma device through FFI calls.
// RdmaDomain is `Sync` because the underlying ibverbs APIs are thread-safe.
unsafe impl Sync for RdmaDomain {}

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
    pub fn new(config: IbverbsConfig) -> Result<Self> {
        tracing::debug!("creating RdmaDomain for device {}", config.device.name());
        // SAFETY:
        // This code uses unsafe FFI calls to interact with the RDMA device, but is safe because:
        // - All pointers are properly initialized and checked for null before use
        // - Memory registration follows the ibverbs API contract with proper access flags
        // - Resources are properly cleaned up in error cases to prevent leaks
        // - The operations follow the documented RDMA protocol for device initialization
        unsafe {
            // Get the device based on the provided RdmaDevice
            let device_name = config.device.name();
            let mut num_devices = 0i32;
            let devices = ffi::ibv_get_device_list(&mut num_devices as *mut _);

            if devices.is_null() || num_devices == 0 {
                return Err(Error::new(
                    std::io::ErrorKind::NotFound,
                    "no RDMA devices found".to_string(),
                ));
            }

            // Find the device with the matching name
            let mut device_ptr = std::ptr::null_mut();
            for i in 0..num_devices {
                let dev = *devices.offset(i as isize);
                let dev_name = CStr::from_ptr(ffi::ibv_get_device_name(dev)).to_string_lossy();

                if dev_name == *device_name {
                    device_ptr = dev;
                    break;
                }
            }

            // If we didn't find the device, return an error
            if device_ptr.is_null() {
                ffi::ibv_free_device_list(devices);
                return Err(Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("device '{}' not found", device_name),
                ));
            }
            tracing::info!("using RDMA device: {}", device_name);

            // Open device
            let context = ffi::ibv_open_device(device_ptr);
            if context.is_null() {
                ffi::ibv_free_device_list(devices);
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to create context: {}", os_error),
                ));
            }

            // Create protection domain
            let pd = ffi::ibv_alloc_pd(context);
            if pd.is_null() {
                ffi::ibv_close_device(context);
                ffi::ibv_free_device_list(devices);
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to create protection domain (PD): {}", os_error),
                ));
            }

            // Register memory region
            // Note - we enable implicit ODP here by:
            // 1) setting access flag IBV_ACCESS_ON_DEMAND
            let access = ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ
                | ffi::ibv_access_flags::IBV_ACCESS_ON_DEMAND
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_ATOMIC;

            // 2) setting the address space to null, MAX_SIZE
            let mr = ffi::ibv_reg_mr(pd, std::ptr::null_mut(), usize::MAX, access.0 as i32);

            if mr.is_null() {
                ffi::ibv_dealloc_pd(pd);
                ffi::ibv_close_device(context);
                ffi::ibv_free_device_list(devices);
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to register memory region (MR): {}", os_error),
                ));
            }

            Ok(RdmaDomain {
                context,
                pd,
                mr,
                config,
                lkey: (*mr).lkey,
                rkey: (*mr).rkey,
            })
        }
    }

    /// Returns the local key (lkey) for the memory region.
    ///
    /// The local key is used when performing local RDMA operations on the registered memory region.
    /// It must be provided in the scatter-gather elements of work requests that access local memory.
    pub fn lkey(&self) -> u32 {
        self.lkey
    }

    /// Returns the remote key (rkey) for the memory region.
    ///
    /// The remote key is used by remote RDMA peers when they need to access this memory region.
    /// It must be shared with remote peers as part of connection establishment to enable
    /// RDMA read, write, and atomic operations on this memory region.
    pub fn rkey(&self) -> u32 {
        self.rkey
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
pub struct RdmaQueuePair {
    cq: *mut ffi::ibv_cq,
    qp: *mut ffi::ibv_qp,
    context: *mut ffi::ibv_context,
    config: IbverbsConfig,
    lkey: u32,
    rkey: u32,
}

impl std::fmt::Debug for RdmaQueuePair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RdmaQueuePair")
            .field("cq", &format!("{:p}", self.cq))
            .field("qp", &format!("{:p}", self.qp))
            .field("context", &format!("{:p}", self.context))
            .field("config", &self.config)
            .field("lkey", &self.lkey)
            .field("rkey", &self.rkey)
            .finish()
    }
}

// SAFETY:
// This function contains code marked unsafe as it interacts with the Rdma device through FFI calls.
// RdmaQueuePair is `Send` because the raw pointers to ibverbs structs can be
// accessed from any thread, and it is safe to drop `RdmaQueuePair` (and run the
// ibverbs destructors) from any thread.
unsafe impl Send for RdmaQueuePair {}

// SAFETY:
// This function contains code marked unsafe as it interacts with the Rdma device through FFI calls.
// `RdmaQueuePair` is `Sync` because the underlying ibverbs APIs are thread-safe.
unsafe impl Sync for RdmaQueuePair {}

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
    pub fn new(domain: &RdmaDomain) -> Result<Self> {
        tracing::debug!("creating an RdmaQueuePair from config {}", domain.config);
        // SAFETY:
        // This code uses unsafe FFI calls to interact with the RDMA device, but is safe because:
        // - All pointers are properly initialized and checked for null before use
        // - Resources (CQ, QP) are created following the ibverbs API contract
        // - Error handling properly cleans up resources in failure cases
        // - The operations follow the documented RDMA protocol for queue pair initialization
        unsafe {
            let context = domain.context;
            let config = domain.config.clone();
            let pd = domain.pd;
            let mr = domain.mr;
            // Create completion queue
            let cq = ffi::ibv_create_cq(
                domain.context,
                config.cq_entries,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            if cq.is_null() {
                ffi::ibv_dealloc_pd(pd);
                ffi::ibv_close_device(context);
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to create completion queue (CQ): {}", os_error),
                ));
            }

            // Create queue pair - note we currently share a CQ for both send and receive for simplicity.
            let mut qp_init_attr = ffi::ibv_qp_init_attr {
                qp_context: std::ptr::null::<std::os::raw::c_void>() as *mut _,
                send_cq: cq,
                recv_cq: cq,
                srq: std::ptr::null::<ffi::ibv_srq>() as *mut _,
                cap: ffi::ibv_qp_cap {
                    max_send_wr: config.max_send_wr,
                    max_recv_wr: config.max_recv_wr,
                    max_send_sge: config.max_send_sge,
                    max_recv_sge: config.max_recv_sge,
                    max_inline_data: 0,
                },
                qp_type: ibv_qp_type::IBV_QPT_RC,
                sq_sig_all: 0,
            };

            let qp = ffi::ibv_create_qp(pd, &mut qp_init_attr);
            if qp.is_null() {
                ffi::ibv_destroy_cq(cq);
                ffi::ibv_dealloc_pd(pd);
                ffi::ibv_close_device(context);
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to create queue pair (QP): {}", os_error),
                ));
            }

            Ok(RdmaQueuePair {
                cq,
                qp,
                context,
                config,
                lkey: (*mr).lkey,
                rkey: (*mr).rkey,
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
    pub fn get_qp_info(&mut self) -> Result<RdmaQpInfo> {
        // SAFETY:
        // This code uses unsafe FFI calls to query RDMA device information, but is safe because:
        // - All pointers are properly initialized before use
        // - Port and GID queries follow the documented ibverbs API contract
        // - Error handling properly checks return codes from ibverbs functions
        // - The memory address provided is only stored, not dereferenced in this function
        unsafe {
            let mut port_attr = ffi::ibv_port_attr::default();
            let errno = ffi::ibv_query_port(
                self.context,
                self.config.port_num,
                &mut port_attr as *mut ffi::ibv_port_attr as *mut _,
            );
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to query port attributes: {}", os_error),
                ));
            }

            let mut gid = Gid::default();
            let ret = ffi::ibv_query_gid(
                self.context,
                self.config.port_num,
                i32::from(self.config.gid_index),
                gid.as_mut(),
            );
            if ret != 0 {
                return Err(Error::new(std::io::ErrorKind::Other, "Failed to query GID"));
            }

            Ok(RdmaQpInfo {
                qp_num: (*self.qp).qp_num,
                lid: port_attr.lid,
                gid: Some(gid),
                psn: self.config.psn,
            })
        }
    }

    /// Connect to a remote Rdma connection point.
    ///
    /// This performs the necessary QP state transitions (INIT->RTR->RTS) to establish a connection.
    ///
    /// # Arguments
    ///
    /// * `connection_info` - The remote connection info to connect to
    pub fn connect(&mut self, connection_info: &RdmaQpInfo) -> Result<()> {
        // SAFETY:
        // This unsafe block is necessary because we're interacting with the RDMA device through FFI calls.
        // The operations are safe because:
        // 1. We're following the documented ibverbs API contract
        // 2. All pointers used are properly initialized and owned by this struct
        // 3. The QP state transitions (INIT->RTR->RTS) follow the required RDMA connection protocol
        // 4. Memory access is properly bounded by the registered memory regions
        unsafe {
            // Transition to INIT
            let qp_access_flags = ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ;

            let mut qp_attr = ffi::ibv_qp_attr {
                qp_state: ffi::ibv_qp_state::IBV_QPS_INIT,
                qp_access_flags: qp_access_flags.0,
                pkey_index: self.config.pkey_index,
                port_num: self.config.port_num,
                ..Default::default()
            };

            let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE
                | ffi::ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
                | ffi::ibv_qp_attr_mask::IBV_QP_PORT
                | ffi::ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;

            let errno = ffi::ibv_modify_qp(self.qp, &mut qp_attr, mask.0 as i32);
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to transition QP to INIT: {}", os_error),
                ));
            }

            // Transition to RTR (Ready to Receive)
            let mut qp_attr = ffi::ibv_qp_attr {
                qp_state: ffi::ibv_qp_state::IBV_QPS_RTR,
                path_mtu: self.config.path_mtu,
                dest_qp_num: connection_info.qp_num,
                rq_psn: connection_info.psn,
                max_dest_rd_atomic: self.config.max_dest_rd_atomic,
                min_rnr_timer: self.config.min_rnr_timer,
                ah_attr: ffi::ibv_ah_attr {
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

            let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE
                | ffi::ibv_qp_attr_mask::IBV_QP_AV
                | ffi::ibv_qp_attr_mask::IBV_QP_PATH_MTU
                | ffi::ibv_qp_attr_mask::IBV_QP_DEST_QPN
                | ffi::ibv_qp_attr_mask::IBV_QP_RQ_PSN
                | ffi::ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC
                | ffi::ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;

            let errno = ffi::ibv_modify_qp(self.qp, &mut qp_attr, mask.0 as i32);
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to transition QP to RTR: {}", os_error),
                ));
            }

            // Transition to RTS (Ready to Send)
            let mut qp_attr = ffi::ibv_qp_attr {
                qp_state: ffi::ibv_qp_state::IBV_QPS_RTS,
                sq_psn: self.config.psn,
                max_rd_atomic: self.config.max_rd_atomic,
                retry_cnt: self.config.retry_cnt,
                rnr_retry: self.config.rnr_retry,
                timeout: self.config.qp_timeout,
                ..Default::default()
            };

            let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE
                | ffi::ibv_qp_attr_mask::IBV_QP_TIMEOUT
                | ffi::ibv_qp_attr_mask::IBV_QP_RETRY_CNT
                | ffi::ibv_qp_attr_mask::IBV_QP_SQ_PSN
                | ffi::ibv_qp_attr_mask::IBV_QP_RNR_RETRY
                | ffi::ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;

            let errno = ffi::ibv_modify_qp(self.qp, &mut qp_attr, mask.0 as i32);
            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to transition QP to RTS: {}", os_error),
                ));
            }
            tracing::debug!(
                "connection sequence has successfully completed (qp: {:?})",
                self.qp
            );

            Ok(())
        }
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
    pub fn post_send(
        &mut self,
        local_addr: usize,
        length: usize,
        wr_id: u64,
        signaled: bool,
        op_type: RdmaOperation,
        raddr: usize,
        rkey: u32,
    ) -> Result<()> {
        // SAFETY:
        // This code uses unsafe FFI calls to post work requests to the RDMA device, but is safe because:
        // - All pointers (send_sge, send_wr) are properly initialized on the stack before use
        // - The memory address in `local_addr` is not dereferenced, only passed to the device
        // - The remote connection info is verified to exist before accessing
        // - The ibverbs post_send operation follows the documented API contract
        // - Error codes from the device are properly checked and propagated
        unsafe {
            let mut send_sge = ffi::ibv_sge {
                addr: local_addr as u64,
                length: length as u32,
                lkey: self.lkey,
            };

            let send_flags = if signaled {
                ffi::ibv_send_flags::IBV_SEND_SIGNALED.0
            } else {
                0
            };

            let mut send_wr = ffi::ibv_send_wr {
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
            let mut bad_send_wr: *mut ffi::ibv_send_wr = std::ptr::null_mut();
            let ops = &mut (*self.context).ops;
            let errno =
                ops.post_send.as_mut().unwrap()(self.qp, &mut send_wr as *mut _, &mut bad_send_wr);

            if errno != 0 {
                let os_error = Error::last_os_error();
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to post send request: {}", os_error),
                ));
            }
            tracing::debug!(
                "completed sending {:?} request (lkey: {}, addr: 0x{:x}, length {}) to (raddr 0x{:x}, rkey {})",
                op_type,
                self.lkey,
                local_addr,
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
    pub fn poll_completion(&self) -> Result<Option<IbvWc>> {
        // SAFETY:
        // This code uses unsafe FFI calls to poll the completion queue, but is safe because:
        // - The completion queue pointer is properly initialized and owned by this struct
        // - The work completion structure is properly zeroed before use
        // - We only access the completion queue through the documented ibverbs API
        // - Error codes from polling operations are properly checked and propagated
        // - The work completion validity is verified before returning it to the caller
        unsafe {
            let mut wc = std::mem::MaybeUninit::<ffi::ibv_wc>::zeroed().assume_init();
            let ops = &mut (*self.context).ops;

            let ret = ops.poll_cq.as_mut().unwrap()(self.cq, 1, &mut wc);

            if ret < 0 {
                return Err(Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to poll CQ: {}", Error::last_os_error()),
                ));
            }

            if ret > 0 {
                if !wc.is_valid() {
                    if let Some((status, vendor_err)) = wc.error() {
                        return Err(Error::new(
                            std::io::ErrorKind::Other,
                            format!(
                                "Work completion failed with status: {:?}, vendor error: {}",
                                status, vendor_err
                            ),
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
    use std::thread;
    use std::time::Duration;
    use std::time::Instant;

    use super::*;
    use crate::ibverbs_primitives::RdmaOperation;
    use crate::ibverbs_primitives::get_all_devices;
    use crate::ibverbs_primitives::ibverbs_supported;

    #[test]
    fn test_create_connection() {
        let config = IbverbsConfig::default();
        let domain = RdmaDomain::new(config.clone());
        assert!(domain.is_ok());

        let domain = domain.unwrap();
        let queue_pair = RdmaQueuePair::new(&domain);
        assert!(queue_pair.is_ok());
    }

    #[test]
    fn test_get_endpoint() {
        let config = IbverbsConfig::default();
        let domain = RdmaDomain::new(config.clone()).unwrap();
        let mut queue_pair = RdmaQueuePair::new(&domain).unwrap();

        // Using 0 as a placeholder address since we're just testing endpoint creation
        let connection_info = queue_pair.get_qp_info();
        assert!(connection_info.is_ok());
    }

    #[test]
    fn test_loopback_connection() {
        let server_config = IbverbsConfig::default();
        let client_config = IbverbsConfig::default();

        let server_domain = RdmaDomain::new(server_config).unwrap();
        let client_domain = RdmaDomain::new(client_config).unwrap();

        let mut server_qp = RdmaQueuePair::new(&server_domain).unwrap();
        let mut client_qp = RdmaQueuePair::new(&client_domain).unwrap();

        // Using 0 as placeholder addresses
        let server_connection_info = server_qp.get_qp_info().unwrap();
        let client_connection_info = client_qp.get_qp_info().unwrap();

        assert!(server_qp.connect(&client_connection_info).is_ok());
        assert!(client_qp.connect(&server_connection_info).is_ok());
    }

    #[test]
    fn test_loopback_rdma_write() {
        // Create buffers for our RDMA operations
        const BSIZE: usize = 128;
        let server_buffer = Box::new([0u8; BSIZE]);
        let mut client_buffer = Box::new([0u8; BSIZE]);

        // Fill the client buffer with test data
        for (i, val) in client_buffer.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }

        // Create domains and queue pairs
        let server_config = IbverbsConfig::default();
        let client_config = IbverbsConfig::default();

        let server_domain = RdmaDomain::new(server_config).unwrap();
        let client_domain = RdmaDomain::new(client_config).unwrap();

        let mut server_qp = RdmaQueuePair::new(&server_domain).unwrap();
        let mut client_qp = RdmaQueuePair::new(&client_domain).unwrap();

        // Get connection info with buffer addresses
        let server_connection_info = server_qp.get_qp_info().unwrap();
        let client_connection_info = client_qp.get_qp_info().unwrap();

        // Connect both sides
        assert!(server_qp.connect(&client_connection_info).is_ok());
        assert!(client_qp.connect(&server_connection_info).is_ok());

        // Client performs RDMA write to server
        client_qp
            .post_send(
                client_buffer.as_ptr() as usize,
                BSIZE,
                1,
                true,
                RdmaOperation::Write,
                server_buffer.as_ptr() as usize,
                server_domain.rkey,
            )
            .unwrap();

        // Poll for completion
        let mut write_completed = false;
        let timeout = Duration::from_secs(2);
        let start_time = Instant::now();

        while !write_completed && start_time.elapsed() < timeout {
            match client_qp.poll_completion() {
                Ok(Some(wc)) => {
                    if wc.wr_id() == 1 {
                        write_completed = true;
                    }
                }
                Ok(None) => {
                    // No completion found, sleep a bit before polling again
                    #[allow(clippy::disallowed_methods)]
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    panic!("Error polling for completion: {}", e);
                }
            }
        }
        assert!(write_completed, "RDMA write operation did not complete");

        // Verify data was correctly transferred
        for i in 0..BSIZE {
            assert_eq!(
                client_buffer[i], server_buffer[i],
                "Data mismatch at position {}",
                i
            );
        }
    }

    #[test]
    fn test_loopback_rdma_read() {
        // Create buffers for our RDMA operations
        const BSIZE: usize = 128;
        let mut server_buffer = Box::new([0u8; BSIZE]);
        let client_buffer = Box::new([0u8; BSIZE]);

        // Fill the server buffer with test data
        for (i, val) in server_buffer.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }

        // Create domains and queue pairs
        let server_config = IbverbsConfig::default();
        let client_config = IbverbsConfig::default();

        let server_domain = RdmaDomain::new(server_config).unwrap();
        let client_domain = RdmaDomain::new(client_config).unwrap();

        let mut server_qp = RdmaQueuePair::new(&server_domain).unwrap();
        let mut client_qp = RdmaQueuePair::new(&client_domain).unwrap();

        // Get connection info with buffer addresses
        let server_connection_info = server_qp.get_qp_info().unwrap();
        let client_connection_info = client_qp.get_qp_info().unwrap();

        // Connect both sides
        assert!(server_qp.connect(&client_connection_info).is_ok());
        assert!(client_qp.connect(&server_connection_info).is_ok());

        // Client performs RDMA read to server (read data from server to client)
        client_qp
            .post_send(
                client_buffer.as_ptr() as usize,
                BSIZE,
                1,
                true,
                RdmaOperation::Read,
                server_buffer.as_ptr() as usize,
                server_domain.rkey,
            )
            .unwrap();

        // Poll for completion
        let mut read_completed = false;
        let timeout = Duration::from_secs(2);
        let start_time = Instant::now();

        while !read_completed && start_time.elapsed() < timeout {
            match client_qp.poll_completion() {
                Ok(Some(wc)) => {
                    if wc.wr_id() == 1 {
                        read_completed = true;
                    }
                }
                Ok(None) => {
                    // No completion found, sleep a bit before polling again
                    #[allow(clippy::disallowed_methods)]
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    panic!("Error polling for completion: {}", e);
                }
            }
        }

        assert!(read_completed, "RDMA read operation did not complete");

        // Verify data was correctly transferred
        for i in 0..BSIZE {
            assert_eq!(
                server_buffer[i], client_buffer[i],
                "Data mismatch at position {}",
                i
            );
        }
    }

    #[test]
    fn test_two_device_write() {
        let devices = get_all_devices();
        if devices.len() != 12 {
            println!(
                "skipping this test as it is only configured on H100 nodes with backend network"
            );
            return;
        }
        const BSIZE: usize = 128;
        let server_buffer = Box::new([0u8; BSIZE]);
        let mut client_buffer = Box::new([0u8; BSIZE]);

        // Fill the client buffer with test data
        for (i, val) in client_buffer.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }

        let device1 = devices.clone().into_iter().next().unwrap();
        let device2 = devices.clone().into_iter().nth(4).unwrap();

        let server_config = IbverbsConfig {
            device: device1,
            ..Default::default()
        };
        let client_config = IbverbsConfig {
            device: device2,
            ..Default::default()
        };

        let server_domain = RdmaDomain::new(server_config).unwrap();
        let client_domain = RdmaDomain::new(client_config).unwrap();

        let mut server_qp = RdmaQueuePair::new(&server_domain).unwrap();
        let mut client_qp = RdmaQueuePair::new(&client_domain).unwrap();

        // Get connection info with buffer addresses
        let server_connection_info = server_qp.get_qp_info().unwrap();
        let client_connection_info = client_qp.get_qp_info().unwrap();

        // Connect both sides
        assert!(server_qp.connect(&client_connection_info).is_ok());
        assert!(client_qp.connect(&server_connection_info).is_ok());

        // Client performs RDMA write to server
        client_qp
            .post_send(
                client_buffer.as_ptr() as usize,
                BSIZE,
                1,
                true,
                RdmaOperation::Write,
                server_buffer.as_ptr() as usize,
                server_domain.rkey,
            )
            .unwrap();

        // Poll for completion
        let mut write_completed = false;
        let timeout = Duration::from_secs(2);
        let start_time = Instant::now();

        while !write_completed && start_time.elapsed() < timeout {
            match client_qp.poll_completion() {
                Ok(Some(wc)) => {
                    if wc.wr_id() == 1 {
                        write_completed = true;
                    }
                }
                Ok(None) => {
                    // No completion found, sleep a bit before polling again
                    #[allow(clippy::disallowed_methods)]
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    panic!("Error polling for completion: {}", e);
                }
            }
        }

        assert!(write_completed, "RDMA write operation did not complete");

        // Verify data was correctly transferred
        for i in 0..BSIZE {
            assert_eq!(
                client_buffer[i], server_buffer[i],
                "Data mismatch at position {}",
                i
            );
        }
    }

    #[test]
    pub fn test_two_device_read() {
        let devices = get_all_devices();
        if devices.len() != 12 {
            println!(
                "skipping this test as it is only configured on H100 nodes with backend network"
            );
            return;
        }

        // Create buffers for our RDMA operations
        const BSIZE: usize = 128;
        let mut server_buffer = Box::new([0u8; BSIZE]);
        let client_buffer = Box::new([0u8; BSIZE]);

        // Fill the server buffer with test data
        for (i, val) in server_buffer.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }

        let device1 = devices.clone().into_iter().next().unwrap();
        let device2 = devices.clone().into_iter().nth(4).unwrap();

        let server_config = IbverbsConfig {
            device: device1,
            ..Default::default()
        };
        let client_config = IbverbsConfig {
            device: device2,
            ..Default::default()
        };

        let server_domain = RdmaDomain::new(server_config).unwrap();
        let client_domain = RdmaDomain::new(client_config).unwrap();

        let mut server_qp = RdmaQueuePair::new(&server_domain).unwrap();
        let mut client_qp = RdmaQueuePair::new(&client_domain).unwrap();

        // Get connection info with buffer addresses
        let server_connection_info = server_qp.get_qp_info().unwrap();
        let client_connection_info = client_qp.get_qp_info().unwrap();

        // Connect both sides
        assert!(server_qp.connect(&client_connection_info).is_ok());
        assert!(client_qp.connect(&server_connection_info).is_ok());

        // Client performs RDMA read from server
        client_qp
            .post_send(
                client_buffer.as_ptr() as usize,
                BSIZE,
                1,
                true,
                RdmaOperation::Read,
                server_buffer.as_ptr() as usize,
                server_domain.rkey,
            )
            .unwrap();

        // Poll for completion
        let mut read_completed = false;
        let timeout = Duration::from_secs(2);
        let start_time = Instant::now();

        while !read_completed && start_time.elapsed() < timeout {
            match client_qp.poll_completion() {
                Ok(Some(wc)) => {
                    if wc.wr_id() == 1 {
                        read_completed = true;
                    }
                }
                Ok(None) => {
                    // No completion found, sleep a bit before polling again
                    #[allow(clippy::disallowed_methods)]
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    panic!("Error polling for completion: {}", e);
                }
            }
        }

        assert!(read_completed, "RDMA read operation did not complete");

        // Verify data was correctly transferred
        for i in 0..BSIZE {
            assert_eq!(
                server_buffer[i], client_buffer[i],
                "Data mismatch at position {}",
                i
            );
        }
    }
}
