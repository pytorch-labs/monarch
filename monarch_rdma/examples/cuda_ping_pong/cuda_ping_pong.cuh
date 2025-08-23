/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CUDA_PING_PONG_H
#define CUDA_PING_PONG_H

#include <stddef.h>
#include <stdint.h>
#include "monarch/rdmaxcel-sys/src/rdmaxcel.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of cudaError_t to avoid including cuda_runtime.h in Rust
// FFI
#ifndef __CUDA_RUNTIME_H__
typedef enum cudaError cudaError_t;
#endif

// Structure for RDMA parameters
typedef struct {
  void* cu_ptr; // actual cuda pointer corresponding to rdma local buffer
  uintptr_t laddr; // Local buffer address
  size_t lsize; // Local buffer size
  uint32_t lkey; // Local memory key
  uintptr_t raddr; // Remote buffer address
  size_t rsize; // Remote buffer size
  uint32_t rkey; // Remote memory key
  uint32_t qp_num; // Queue pair number
  uint8_t* rq_buf; // Receive queue buffer
  uint32_t rq_cnt; // Receive queue count
  uint8_t* sq_buf; // Send queue buffer
  uint32_t sq_cnt; // Send queue count
  uint32_t* qp_dbrec; // Queue pair doorbell record
  void* send_cqe_buf; // Send completion queue entry buffer
  uint32_t send_cqe_size; // Send completion queue entry size
  uint32_t send_cqe_cnt; // Send completion queue entry count
  uint32_t* send_cqe_dbrec; // Send completion queue doorbell record
  void* recv_cqe_buf; // Receive completion queue entry buffer
  uint32_t recv_cqe_size; // Receive completion queue entry size
  uint32_t recv_cqe_cnt; // Receive completion queue entry count
  uint32_t* recv_cqe_dbrec; // Receive completion queue doorbell record
  void* qp_db; // Queue pair doorbell register
} rdma_params_t;

// We'll use the functions from rdmaxcel.h directly

// Function to perform ping-pong test with RDMA parameters
__global__ void rdmaPingPong(
    void* data,
    int32_t iterations,
    size_t intial_length,
    wqe_params_t send_wqe_params,
    wqe_params_t recv_wqe_params,
    void* db_dst,
    cqe_poll_params_t send_cqe_params,
    cqe_poll_params_t recv_cqe_params,
    int32_t* poll_result,
    bool is_leader);

// Function to launch the ping-pong test
cudaError_t launchPingPong(
    const rdma_params_t* params,
    int iterations,
    int initial_length,
    int device_id);

#ifdef __cplusplus
}
#endif

#endif // CUDA_PING_PONG_H
