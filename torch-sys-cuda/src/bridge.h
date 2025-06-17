/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/cuda/CUDAEvent.h> // @manual=//caffe2:torch-cpp
#include <nccl.h> // @manual
#include <torch/torch.h> // @manual=//caffe2:torch-cpp

namespace monarch {

std::unique_ptr<at::cuda::CUDAEvent>
create_cuda_event(bool enable_timing, bool blocking, bool interprocess);

std::shared_ptr<c10::cuda::CUDAStream> get_current_stream(
    c10::DeviceIndex device);

std::shared_ptr<c10::cuda::CUDAStream> create_stream(
    c10::DeviceIndex device,
    int32_t priority);

void set_current_stream(const c10::cuda::CUDAStream& stream);

/// This function exists because ncclConfig initialization requires the use of
/// a macro. We cannot reference the macro directly from Rust code, so we wrap
/// the macro use in a function and bind that to Rust instead.
inline ncclConfig_t make_nccl_config() {
  ncclConfig_t ret = NCCL_CONFIG_INITIALIZER;
  return ret;
}

} // namespace monarch
