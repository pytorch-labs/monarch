// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <nccl.h> // @manual=fbsource//third-party/ncclx:nccl
#include <rust/cxx.h>

namespace monarch::nccl {
// This function uses C++ APIs that can't be made with bindgen, so we make
// custom bindings. In particular, ncclCommDump uses std::unordered_map which
// doesn't automatically bridge to Rust HashMap.
// Calls ncclCommDump from NCCLX and appends the results to map, where each
// pair is a key/value pair. The vector always has an even number of items
// appended to it.
ncclResult_t nccl_comm_dump(ncclComm_t comm, rust::Vec<rust::String>& map);

} // namespace monarch::nccl
