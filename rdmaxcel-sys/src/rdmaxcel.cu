/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rdmaxcel.h"

//------------------------------------------------------------------------------
// Byte Swapping Utilities
//------------------------------------------------------------------------------

/**
 * @brief Swaps the byte order of a 16-bit value (converts between little and
 * big endian)
 *
 * This function is used for endianness conversion when communicating with
 * InfiniBand hardware, which uses big-endian byte ordering.
 *
 * @param val The 16-bit value to swap
 * @return The byte-swapped value
 */
__host__ __device__ static inline uint16_t byte_swap16(uint16_t val) {
  return ((val & 0xFF00) >> 8) | ((val & 0x00FF) << 8);
}

/**
 * @brief Swaps the byte order of a 32-bit value (converts between little and
 * big endian)
 *
 * This function is used for endianness conversion when communicating with
 * InfiniBand hardware, which uses big-endian byte ordering.
 *
 * @param val The 32-bit value to swap
 * @return The byte-swapped value
 */
__host__ __device__ static inline uint32_t byte_swap32(uint32_t val) {
  return ((val & 0xFF000000) >> 24) | ((val & 0x00FF0000) >> 8) |
      ((val & 0x0000FF00) << 8) | ((val & 0x000000FF) << 24);
}

/**
 * @brief Swaps the byte order of a 64-bit value (converts between little and
 * big endian)
 *
 * This function is used for endianness conversion when communicating with
 * InfiniBand hardware, which uses big-endian byte ordering.
 *
 * @param val The 64-bit value to swap
 * @return The byte-swapped value
 */
__host__ __device__ static inline uint64_t byte_swap64(uint64_t val) {
  return ((val & 0xFF00000000000000ULL) >> 56) |
      ((val & 0x00FF000000000000ULL) >> 40) |
      ((val & 0x0000FF0000000000ULL) >> 24) |
      ((val & 0x000000FF00000000ULL) >> 8) |
      ((val & 0x00000000FF000000ULL) << 8) |
      ((val & 0x0000000000FF0000ULL) << 24) |
      ((val & 0x000000000000FF00ULL) << 40) |
      ((val & 0x00000000000000FFULL) << 56);
}

//------------------------------------------------------------------------------
// Doorbell Operations
//------------------------------------------------------------------------------

/**
 * @brief Rings a doorbell by copying 8 64-bit values from source to destination
 *
 * This function is used to notify the HCA (Host Channel Adapter) that new work
 * has been queued. It copies 8 64-bit values (64 bytes total) from the source
 * to the destination, which is typically a memory-mapped doorbell register.
 *
 * @param dst Pointer to the destination (doorbell register)
 * @param src Pointer to the source data
 */
__host__ __device__ void db_ring(void* dst, void* src) {
  volatile uint64_t* dst_v = (uint64_t*)dst;
  volatile uint64_t* src_v = (uint64_t*)src;
  dst_v[0] = src_v[0];
  dst_v[1] = src_v[1];
  dst_v[2] = src_v[2];
  dst_v[3] = src_v[3];
  dst_v[4] = src_v[4];
  dst_v[5] = src_v[5];
  dst_v[6] = src_v[6];
  dst_v[7] = src_v[7];
}

/**
 * @brief CUDA kernel wrapper for db_ring function
 *
 * This kernel launches a single thread to execute the db_ring function on the
 * GPU. It includes memory fences to ensure proper ordering of memory
 * operations.
 *
 * @param dst Pointer to the destination (doorbell register)
 * @param src Pointer to the source data
 */
__global__ void cu_db_ring(void* dst, void* src) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    db_ring(dst, src);
  }
  __syncthreads();
  __threadfence_system();
}

/**
 * @brief Host function to launch the cu_db_ring kernel
 *
 * This function launches the cu_db_ring kernel with a single thread.
 *
 * @param dst Pointer to the destination (doorbell register)
 * @param src Pointer to the source data
 */
void launch_db_ring(void* dst, void* src) {
  cu_db_ring<<<1, 1>>>(dst, src);
}

//------------------------------------------------------------------------------
// Work Queue Element (WQE) Operations
//------------------------------------------------------------------------------

/**
 * @brief Creates and posts a receive WQE (Work Queue Element)
 *
 * This function creates a receive WQE with the specified parameters and posts
 * it to the receive queue. For MLX5 receive WQEs, it creates a data segment and
 * updates the doorbell record.
 *
 * @param params Structure containing all parameters needed for the receive WQE
 */
__host__ __device__ void recv_wqe(wqe_params_t params) {
  // For MLX5 receive WQEs, we need to create a proper structure with:
  // 1. A next segment (mlx5_wqe_srq_next_seg)
  // 2. A data segment (mlx5_wqe_data_seg)

  // Declare individual segments instead of using the combined struct
  struct mlx5_wqe_data_seg data_seg;

  // Initialize the data segment
  data_seg.byte_count = byte_swap32(params.length);
  data_seg.lkey = byte_swap32(params.lkey);
  data_seg.addr = byte_swap64(params.laddr);

  // Calculate pointers for segments
  uintptr_t data_seg_ptr = (uintptr_t)params.buf;

  // Copy segments to WQE buffer
  memcpy((void*)data_seg_ptr, &data_seg, sizeof(data_seg));

  volatile uint32_t* dbrec = params.dbrec; // Declare a volatile pointer
  dbrec[MLX5_RCV_DBR] = byte_swap32(params.wr_id + 1);
}

/**
 * @brief CUDA kernel wrapper for recv_wqe function
 *
 * This kernel launches a single thread to execute the recv_wqe function on the
 * GPU.
 *
 * @param params Structure containing all parameters needed for the receive WQE
 */
__global__ void cu_recv_wqe(wqe_params_t params) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    recv_wqe(params);
  }
}

/**
 * @brief Host function to launch the cu_recv_wqe kernel
 *
 * This function launches the cu_recv_wqe kernel with a single thread and
 * synchronizes the device to ensure completion.
 *
 * @param params Structure containing all parameters needed for the receive WQE
 */
void launch_recv_wqe(wqe_params_t params) {
  // Launch kernel
  cu_recv_wqe<<<1, 1>>>(params);

  // Wait for kernel to complete
  cudaDeviceSynchronize();
}

/**
 * @brief Creates and posts a send WQE (Work Queue Element)
 *
 * This function creates a send WQE with the specified parameters and posts it
 * to the send queue. It creates control, remote address, and data segments,
 * and updates the doorbell record.
 *
 * @param params Structure containing all parameters needed for the send WQE
 */
__host__ __device__ void send_wqe(wqe_params_t params) {
  struct mlx5_wqe_ctrl_seg ctrl_seg = {0};
  struct mlx5_wqe_data_seg data_seg = {0};
  struct mlx5_wqe_raddr_seg raddr_seg = {0};

  uint32_t idx = params.wr_id;
  uint32_t buffer_idx = idx & (params.wqe_cnt - 1);

  // Set control segment
  ctrl_seg.fm_ce_se =
      params.signaled ? MLX5_WQE_CTRL_CQ_UPDATE | MLX5_WQE_CTRL_SOLICITED : 0;

  // Set opcode based on operation type
  ctrl_seg.opmod_idx_opcode = ((idx << 8) | params.op_type);

  // Convert to big endian
  ctrl_seg.opmod_idx_opcode = byte_swap32(ctrl_seg.opmod_idx_opcode);

  // Set QP number and data size (48 bytes / 16 = 3 DS)
  ctrl_seg.qpn_ds = (params.qp_num << 8 | (48 / 16));
  ctrl_seg.qpn_ds = byte_swap32(ctrl_seg.qpn_ds);

  // Set remote address segment
  raddr_seg.raddr = byte_swap64(params.raddr);
  raddr_seg.rkey = byte_swap32(params.rkey);

  // Set data segment
  data_seg.addr = byte_swap64(params.laddr);
  data_seg.byte_count = byte_swap32(params.length);
  data_seg.lkey = byte_swap32(params.lkey);

  // Calculate pointers for segments
  uintptr_t ctrl_seg_ptr =
      (uintptr_t)(params.buf) + (buffer_idx << MLX5_SEND_WQE_SHIFT);
  uintptr_t raddr_seg_ptr = ctrl_seg_ptr + sizeof(ctrl_seg);
  uintptr_t data_seg_ptr = raddr_seg_ptr + sizeof(raddr_seg);

  // Copy segments to WQE buffer
  memcpy((void*)ctrl_seg_ptr, &ctrl_seg, sizeof(ctrl_seg));
  memcpy((void*)raddr_seg_ptr, &raddr_seg, sizeof(raddr_seg));
  memcpy((void*)data_seg_ptr, &data_seg, sizeof(data_seg));

  volatile uint32_t* dbrec = params.dbrec;
  dbrec[MLX5_SND_DBR] = byte_swap32((idx + 1) & 0xFFFFFF);
}

/**
 * @brief CUDA kernel wrapper for send_wqe function
 *
 * This kernel launches a single thread to execute the send_wqe function on the
 * GPU.
 *
 * @param params Structure containing all parameters needed for the send WQE
 */
__global__ void cu_send_wqe(wqe_params_t params) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    send_wqe(params);
  }
}

/**
 * @brief Host function to launch the cu_send_wqe kernel
 *
 * This function launches the cu_send_wqe kernel with a single thread and
 * synchronizes the device to ensure completion.
 *
 * @param params Structure containing all parameters needed for the send WQE
 */
void launch_send_wqe(wqe_params_t params) {
  // Launch kernel
  cu_send_wqe<<<1, 1>>>(params);

  // Wait for kernel to complete
  cudaDeviceSynchronize();
}

//------------------------------------------------------------------------------
// Completion Queue Element (CQE) Operations
//------------------------------------------------------------------------------

/**
 * @brief Polls a completion queue for a new completion
 *
 * This function checks if there is a new completion in the completion queue.
 * If a valid completion is found, it updates the byte_cnt parameter with the
 * number of bytes transferred and increments the consumer index.
 *
 * @param byte_cnt Pointer to store the number of bytes transferred (-1 if no
 * valid completion, error if -2, success if >= 0)
 * @param params Structure containing all parameters needed for polling the CQ
 */
__host__ __device__ void cqe_poll(int32_t* byte_cnt, cqe_poll_params_t params) {
  assert(*byte_cnt == -1); // byte_cnt should be initialized to -1

  // Calculate the index in the CQ buffer
  uint32_t idx = params.consumer_index;
  uint32_t buffer_idx = idx & (params.cqe_cnt - 1);

  // Get the CQE at that index
  uint8_t* cqe = params.cqe_buf + (buffer_idx * params.cqe_size);

  // The op_own byte is the last byte of the CQE
  uint8_t op_own = cqe[params.cqe_size - 1];

  // Extract the opcode (upper 4 bits)
  uint8_t actual_opcode = op_own >> 4;

  // to check if the CQE is owned by SW (but opcode at 0xF implies also not
  // owned!)
  bool is_sw_owned = ((op_own & 0x1) == ((idx / params.cqe_cnt) & 0x1));
  is_sw_owned = is_sw_owned && (actual_opcode != 0xF);

  // this only checks for valid opcode, in some case should generate error
  const uint8_t FIRST_TWO_BITS_MASK = 0xC; // Binary: 1100
  bool is_valid_opcode = (actual_opcode & FIRST_TWO_BITS_MASK) == 0;

  if (is_sw_owned && is_valid_opcode) {
    *byte_cnt = byte_swap32(*(uint32_t*)(cqe + 44));
    volatile uint32_t* dbrec = (uint32_t*)params.dbrec;
    *dbrec = byte_swap32((idx + 1) & 0xFFFFFF);
  } else if (is_sw_owned && !is_valid_opcode) {
    *byte_cnt = -2; // signal error
    volatile uint32_t* dbrec = (uint32_t*)params.dbrec;
    *dbrec = byte_swap32((idx + 1) & 0xFFFFFF);
  }
}

/**
 * @brief CUDA kernel wrapper for cqe_poll function
 *
 * This kernel launches a single thread to execute the cqe_poll function on the
 * GPU. It includes memory fences to ensure proper ordering of memory
 * operations.
 *
 * @param result Pointer to store the result of the poll operation
 * @param params Structure containing all parameters needed for polling the CQ
 */
__global__ void cu_cqe_poll(int32_t* result, cqe_poll_params_t params) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    cqe_poll(result, params);
  }
  __syncthreads();
  __threadfence_system();
}

/**
 * @brief Host function to launch the cu_cqe_poll kernel
 *
 * This function allocates memory for the result, launches the cu_cqe_poll
 * kernel, and returns the result of the poll operation.
 *
 * @param mlx5dv_cq_void Pointer to the mlx5dv_cq structure
 * @param consumer_index Current consumer index
 * @return CQE_POLL_TRUE if a valid completion was found, CQE_POLL_FALSE
 * otherwise, or CQE_POLL_ERROR if an error occurred
 */
cqe_poll_result_t launch_cqe_poll(void* mlx5dv_cq_void, int consumer_index) {
  // Cast to proper types on CPU side
  struct mlx5dv_cq* cq = (struct mlx5dv_cq*)mlx5dv_cq_void;

  // Allocate memory for result
  int32_t* byte_cnt = nullptr;
  cudaError_t err = cudaMallocManaged(&byte_cnt, sizeof(int32_t));
  if (err != cudaSuccess) {
    return CQE_POLL_ERROR;
  }
  *byte_cnt = -1; // Initialize to false

  // Create the parameters struct
  cqe_poll_params_t params = {
      .cqe_buf = (uint8_t*)cq->buf,
      .cqe_size = cq->cqe_size,
      .consumer_index = (uint32_t)consumer_index,
      .cqe_cnt = cq->cqe_cnt,
      .dbrec = (uint32_t*)cq->dbrec};

  // Launch the kernel with the parameters struct
  cu_cqe_poll<<<1, 1>>>(byte_cnt, params);

  // Synchronize and get result
  cudaDeviceSynchronize();

  // Check for errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(byte_cnt);
    return CQE_POLL_ERROR;
  }

  // Get the result
  cqe_poll_result_t ret_val = *byte_cnt >= 0 ? CQE_POLL_TRUE : CQE_POLL_FALSE;
  cudaFree(byte_cnt);
  return ret_val;
}

/**
 * @brief Function to poll send completion queue
 *
 * This is a wrapper around launch_cqe_poll specifically for send completions.
 *
 * @param mlx5dv_cq_void Pointer to the mlx5dv_cq structure for the send CQ
 * @param consumer_index Current consumer index
 * @return CQE_POLL_TRUE if a valid completion was found, CQE_POLL_FALSE
 * otherwise, or CQE_POLL_ERROR if an error occurred
 */
cqe_poll_result_t launch_send_cqe_poll(
    void* mlx5dv_cq_void,
    int consumer_index) {
  return launch_cqe_poll(mlx5dv_cq_void, consumer_index);
}

/**
 * @brief Function to poll receive completion queue
 *
 * This is a wrapper around launch_cqe_poll specifically for receive
 * completions.
 *
 * @param mlx5dv_cq_void Pointer to the mlx5dv_cq structure for the receive CQ
 * @param consumer_index Current consumer index
 * @return CQE_POLL_TRUE if a valid completion was found, CQE_POLL_FALSE
 * otherwise, or CQE_POLL_ERROR if an error occurred
 */
cqe_poll_result_t launch_recv_cqe_poll(
    void* mlx5dv_cq_void,
    int consumer_index) {
  return launch_cqe_poll(mlx5dv_cq_void, consumer_index);
}
