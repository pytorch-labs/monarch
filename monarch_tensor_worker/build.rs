/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

fn main() {
    // `torch-sys` will set this env var through Cargo `links` metadata.
    let lib_path = std::env::var("DEP_TORCH_LIB_PATH").expect("DEP_TORCH_LIB_PATH to be set");
    // Set the rpath so that the dynamic linker can find libtorch and friends.
    println!("cargo::rustc-link-arg=-Wl,-rpath,{lib_path}");
}
