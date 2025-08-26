# RDMaxcel

RDMaxcel (RDMA Acceleration) is a library that provides a unified interface for RDMA operations from both CPU and GPU, enabling seamless RDMA communication regardless of whether the caller is host code or device hardware.

## Overview

RDMaxcel bridges the gap between traditional CPU-based RDMA programming and GPU-accelerated RDMA operations by providing a consistent API that works identically in both environments. This enables applications to leverage the power of GPUs for RDMA operations without having to maintain separate code paths.

## Key Features

- **Unified Host/Device API**: Same functions can be called from both CPU and GPU code
- **Zero-copy Operations**: Direct GPU access to RDMA resources without CPU involvement
- **Hardware-agnostic Interface**: Abstract away differences between CPU and GPU execution
- **Low-latency Communication**: Direct GPU-initiated RDMA operations

## How RDMaxcel Works

### Work Queue Elements (WQEs)

Work Queue Elements (WQEs) are the fundamental building blocks of RDMA operations. They represent commands to the RDMA hardware that describe data transfer operations.

#### WQE Structure

In RDMaxcel, WQEs are composed of multiple segments:

1. **Control Segment**: Contains operation type, QP number, and other control information
2. **Remote Address Segment**: Specifies the remote memory address and key for RDMA operations
3. **Data Segment**: Describes the local memory buffer, including address, length, and memory key

The `wqe_params_t` structure encapsulates all parameters needed to create and post a WQE:

```c
typedef struct {
  uintptr_t laddr;      // Local memory address
  uint32_t lkey;        // Local memory key
  size_t length;        // Length of data to transfer
  uint64_t wr_id;       // Work request ID
  bool signaled;        // Whether completion should be signaled
  uint32_t op_type;     // Operation type (e.g., MLX5_OPCODE_RDMA_WRITE)
  uintptr_t raddr;      // Remote memory address
  uint32_t rkey;        // Remote memory key
  uint32_t qp_num;      // Queue pair number
  uint8_t* buf;         // WQE buffer address
  uint32_t* dbrec;      // Doorbell record
  uint32_t wqe_cnt;     // WQE count
} wqe_params_t;
```

#### Posting WQEs

RDMaxcel provides unified functions for posting send and receive WQEs:

- `send_wqe()`: Posts a send WQE (RDMA write, read, send, etc.)
- `recv_wqe()`: Posts a receive WQE

These functions can be called directly from both host and device code, with identical behavior. This is achieved through CUDA's `__host__ __device__` function attributes.

### Doorbells

Doorbells are the mechanism used to notify the RDMA hardware that new work has been queued and is ready for processing.

#### How Doorbells Work

1. **WQE Creation**: The application creates a WQE in memory
2. **Doorbell Ring**: The application "rings the doorbell" by writing to a special memory-mapped register
3. **Hardware Notification**: This write operation notifies the RDMA hardware that new work is available
4. **Work Processing**: The hardware reads the WQE from memory and executes the requested operation

#### Doorbell Implementation

RDMaxcel implements doorbell operations through the `db_ring()` function:

```c
__host__ __device__ void db_ring(void* dst, void* src);
```

This function copies 64 bytes (8 64-bit values) from the source buffer to the destination doorbell register. The unified `__host__ __device__` implementation ensures that this operation works identically whether called from CPU or GPU code.

### Same Code Path for Device and Host

One of the key innovations in RDMaxcel is the use of the same code path for both device (GPU) and host (CPU) operations. This is achieved through several techniques:

#### Unified Function Implementations

Core functions are implemented with CUDA's `__host__ __device__` attributes, allowing them to be compiled for both CPU and GPU execution:

```c
__host__ __device__ void send_wqe(wqe_params_t params);
__host__ __device__ void recv_wqe(wqe_params_t params);
__host__ __device__ void db_ring(void* dst, void* src);
__host__ __device__ void cqe_poll(int32_t* result, cqe_poll_params_t params);
```

#### Memory Registration

For GPU access to RDMA resources, RDMaxcel registers the necessary memory regions with CUDA:

```c
cudaError_t register_cuda_memory(
    struct mlx5dv_qp* dv_qp,
    struct mlx5dv_cq* dv_recv_cq,
    struct mlx5dv_cq* dv_send_cq);
```

This function registers queue pair buffers, completion queue buffers, and doorbell registers with CUDA, making them accessible from GPU code.

#### Kernel Wrappers

For GPU execution, RDMaxcel provides kernel wrapper functions that launch the core functions on the GPU:

```c
__global__ void cu_send_wqe(wqe_params_t params);
__global__ void cu_recv_wqe(wqe_params_t params);
__global__ void cu_db_ring(void* dst, void* src);
__global__ void cu_cqe_poll(int32_t* result, cqe_poll_params_t params);
```

And corresponding launch functions:

```c
void launch_send_wqe(wqe_params_t params);
void launch_recv_wqe(wqe_params_t params);
void launch_db_ring(void* dst, void* src);
cqe_poll_result_t launch_cqe_poll(void* mlx5dv_cq, int32_t cqe_idx);
```

### Caller Agnosticism

The design of RDMaxcel makes it agnostic to whether the caller is CPU or GPU hardware:

1. **Identical Function Signatures**: The same parameters are used for both CPU and GPU calls
2. **Consistent Memory Layout**: WQEs and CQEs have the same memory layout in both environments
3. **Unified Endianness Handling**: Byte swapping functions work identically on both platforms
4. **Transparent Memory Access**: Memory registration ensures GPU can access all required resources

## Benefits

- **Code Reuse**: Write RDMA code once, run it on both CPU and GPU
- **Simplified Development**: No need to maintain separate code paths
- **Performance**: Direct GPU-initiated RDMA operations without CPU involvement
- **Flexibility**: Choose the best execution environment for each workload

## Usage Example

```c
// Create and initialize RDMA resources
struct ibv_qp* qp = create_qp(...);
struct mlx5dv_qp* dv_qp = create_mlx5dv_qp(qp);
struct mlx5dv_cq* dv_send_cq = create_mlx5dv_send_cq(qp);
struct mlx5dv_cq* dv_recv_cq = create_mlx5dv_recv_cq(qp);

// Register memory with CUDA
register_cuda_memory(dv_qp, dv_recv_cq, dv_send_cq);

// Create WQE parameters
wqe_params_t params = {
    .laddr = local_buffer_addr,
    .lkey = local_memory_key,
    .length = transfer_size,
    .wr_id = work_id,
    .signaled = true,
    .op_type = MLX5_OPCODE_RDMA_WRITE,
    .raddr = remote_buffer_addr,
    .rkey = remote_memory_key,
    .qp_num = qp->qp_num,
    .buf = dv_qp->sq.buf,
    .dbrec = dv_qp->dbrec,
    .wqe_cnt = dv_qp->sq.wqe_cnt
};

// CPU execution
send_wqe(params);

// Or GPU execution
launch_send_wqe(params);
```

## Reference Documentation

### Mellanox Programming Manual

For detailed information about the various data structures and protocols used by Mellanox/NVIDIA adapters, refer to the official Mellanox Ethernet Adapters Programming Manual. This document provides comprehensive documentation of:

- Low-level hardware interfaces and data structures
- Work Queue Element (WQE) formats and opcodes
- Completion Queue Element (CQE) structures
- Memory protection keys and registration
- Direct hardware access patterns
- Doorbell and completion notification mechanisms

The manual is essential for understanding the underlying hardware primitives that RDMaxcel abstracts and provides unified access to.

**URL**: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf

## Conclusion

RDMaxcel provides a powerful abstraction for RDMA operations that works seamlessly across CPU and GPU environments. By using the same code path for both device and host operations, it simplifies development and enables new possibilities for GPU-accelerated networking applications.
