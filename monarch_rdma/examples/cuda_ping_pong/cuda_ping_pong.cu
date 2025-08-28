/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include "cuda_ping_pong.cuh"

// Include the rdmaxcel header for device functions
#include <monarch/rdmaxcel-sys/src/rdmaxcel.h>

__global__ void rdmaPingPong(
    void* data,
    int32_t iterations,
    size_t initial_length,
    wqe_params_t send_wqe_params,
    wqe_params_t recv_wqe_params,
    void* db_dst,
    cqe_poll_params_t send_cqe_params,
    cqe_poll_params_t recv_cqe_params,
    int32_t* poll_result,
    bool is_leader) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0 && !is_leader) {
    recv_wqe(recv_wqe_params);
    recv_wqe_params.wr_id += 1;
  }

  for (int j = 0; j < iterations * 2; j++) {
    if (i == 0) {
      if (is_leader) {
        send_wqe_params.length += initial_length;
        send_wqe(send_wqe_params);
        db_ring(
            db_dst,
            send_wqe_params.buf +
                64 * (send_wqe_params.wr_id % send_wqe_params.wqe_cnt));
        send_wqe_params.wr_id += 1;
        send_wqe_params.laddr += send_wqe_params.length;
        while (*poll_result == -1) {
          cqe_poll(poll_result, send_cqe_params);
        }
        assert(*poll_result != -2);
        send_cqe_params.consumer_index += 1;

        recv_wqe(recv_wqe_params);
        recv_wqe_params.wr_id += 1;
      } else {
        while (*poll_result == -1) {
          cqe_poll(poll_result, recv_cqe_params);
        }
        assert(*poll_result != -2);
        recv_cqe_params.consumer_index += 1;
        send_wqe_params.laddr += *poll_result;
        send_wqe_params.raddr = send_wqe_params.laddr;
        printf("msg number: %d, msg size = %d\n", j, *poll_result);
      }
      *poll_result = -1;
      is_leader = !is_leader;
    }
    __syncthreads();
    __threadfence_system();
  }
}

// Function to perform ping-pong test with RDMA parameters
extern "C" cudaError_t launchPingPong(
    const rdma_params_t* params,
    int iterations,
    int initial_length,
    int device_id) {
  // Set up WQE parameters for RDMA operation
  wqe_params_t send_wqe_params = {
      .laddr = params->laddr,
      .lkey = params->lkey,
      .length = 0, // start at 0 and increment
      .wr_id = 0,
      .signaled = true,
      .op_type = MLX5_OPCODE_RDMA_WRITE_IMM,
      .raddr = params->raddr,
      .rkey = params->rkey,
      .qp_num = params->qp_num,
      .buf = params->sq_buf,
      .dbrec = params->qp_dbrec,
      .wqe_cnt = params->sq_cnt};

  wqe_params_t recv_wqe_params = {
      .laddr = params->laddr,
      .lkey = params->lkey,
      .length = 0, // not used for recv
      .wr_id = 0,
      .signaled = true,
      .op_type = IBV_WC_RECV,
      .raddr = params->raddr,
      .rkey = params->rkey,
      .qp_num = params->qp_num,
      .buf = params->rq_buf,
      .dbrec = params->qp_dbrec,
      .wqe_cnt = params->rq_cnt};

  // Set up doorbell parameters
  void* db_dst = params->qp_db;

  // Set up send CQE poll parameters
  cqe_poll_params_t send_cqe_params = {
      .cqe_buf = (uint8_t*)params->send_cqe_buf,
      .cqe_size = params->send_cqe_size,
      .consumer_index = 0,
      .cqe_cnt = params->send_cqe_cnt,
      .dbrec = params->send_cqe_dbrec};

  // Set up receive CQE poll parameters
  cqe_poll_params_t recv_cqe_params = {
      .cqe_buf = (uint8_t*)params->recv_cqe_buf,
      .cqe_size = params->recv_cqe_size,
      .consumer_index = 0,
      .cqe_cnt = params->recv_cqe_cnt,
      .dbrec = params->recv_cqe_dbrec};

  cudaError_t err;
  int32_t* poll_result = nullptr;
  err = cudaMallocManaged(&poll_result, sizeof(int32_t));
  if (err != cudaSuccess) {
    return err;
  }
  *poll_result = -1;
  void* data = params->cu_ptr;
  bool is_leader = device_id == 0;

  rdmaPingPong<<<1, 32>>>(
      data,
      iterations,
      (size_t)initial_length,
      send_wqe_params,
      recv_wqe_params,
      db_dst,
      send_cqe_params,
      recv_cqe_params,
      poll_result,
      is_leader);

  if (is_leader) {
    err = cudaDeviceSynchronize();
  }
  return err;
}
