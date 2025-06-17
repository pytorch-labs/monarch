/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[allow(dead_code)]
#[cxx::bridge(namespace = "monarch")]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("ATen/cuda/CUDAEvent.h");
        include!("monarch/torch-sys-cuda/src/bridge.h");

        // CUDA Event APIs
        #[namespace = "at::cuda"]
        type CUDAEvent;
        fn create_cuda_event(
            enable_timing: bool,
            blocking: bool,
            interprocess: bool,
        ) -> UniquePtr<CUDAEvent>;
        fn record(self: Pin<&mut CUDAEvent>, stream: &CUDAStream);
        fn block(self: Pin<&mut CUDAEvent>, stream: &CUDAStream);
        fn query(self: &CUDAEvent) -> bool;
        fn elapsed_time(self: &CUDAEvent, end_event: &CUDAEvent) -> f32;
        fn synchronize(self: &CUDAEvent);

        // CUDA Stream APIs
        #[namespace = "c10::cuda"]
        type CUDAStream;
        #[namespace = ""]
        type CUstream_st = nccl_sys::CUstream_st;
        fn get_current_stream(device: i8) -> SharedPtr<CUDAStream>;
        fn set_current_stream(stream: &CUDAStream);
        fn create_stream(device: i8, priority: i32) -> SharedPtr<CUDAStream>;
        fn query(self: &CUDAStream) -> bool;
        fn synchronize(self: &CUDAStream);
        fn device_index(self: &CUDAStream) -> i8;
        fn stream(self: &CUDAStream) -> *mut CUstream_st;

        // nccl helpers
        #[namespace = ""]
        type ncclConfig_t = nccl_sys::ncclConfig_t;
        fn make_nccl_config() -> ncclConfig_t;
    }
}

use std::fmt::Debug;
use std::fmt::Error;
use std::fmt::Formatter;

// SAFETY: CUDAStream is thread safe
unsafe impl Send for ffi::CUDAStream {}
// SAFETY: see above
unsafe impl Sync for ffi::CUDAStream {}

impl Debug for ffi::CUDAStream {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.debug_struct("CUDAStream")
            .field("device", &format!("{}", self.device_index()))
            .field("stream", &format!("{:p}", self))
            .finish()
    }
}

impl Debug for ffi::CUDAEvent {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.debug_struct("CUDAEvent")
            .field("ptr", &format!("{:p}", self))
            .finish()
    }
}

// SAFETY: CUDAEvent is thread safe. The comments on `c10::Event` say it isn't, but in
// Rust we would consider it Sync because shared references are fine to access
// across threads.
unsafe impl Send for ffi::CUDAEvent {}
// SAFETY: see above
unsafe impl Sync for ffi::CUDAEvent {}
