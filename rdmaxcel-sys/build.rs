/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::path::Path;
use std::path::PathBuf;

use glob::glob;
use which::which;

// Translated from torch/utils/cpp_extension.py
fn find_cuda_home() -> Option<String> {
    // Guess #1
    let mut cuda_home = env::var("CUDA_HOME")
        .ok()
        .or_else(|| env::var("CUDA_PATH").ok());

    if cuda_home.is_none() {
        // Guess #2
        if let Ok(nvcc_path) = which("nvcc") {
            // Get parent directory twice
            if let Some(cuda_dir) = nvcc_path.parent().and_then(|p| p.parent()) {
                cuda_home = Some(cuda_dir.to_string_lossy().into_owned());
            }
        } else {
            // Guess #3
            if cfg!(windows) {
                // Windows code
                let pattern = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*.*";
                let cuda_homes: Vec<_> = glob(pattern).unwrap().filter_map(Result::ok).collect();
                if !cuda_homes.is_empty() {
                    cuda_home = Some(cuda_homes[0].to_string_lossy().into_owned());
                } else {
                    cuda_home = None;
                }
            } else {
                // Not Windows
                let cuda_candidate = "/usr/local/cuda";
                if Path::new(cuda_candidate).exists() {
                    cuda_home = Some(cuda_candidate.to_string());
                } else {
                    cuda_home = None;
                }
            }
        }
    }
    cuda_home
}

fn emit_cuda_link_directives(cuda_home: &str) {
    let stubs_path = format!("{}/lib64/stubs", cuda_home);
    if Path::new(&stubs_path).exists() {
        println!("cargo:rustc-link-search=native={}", stubs_path);
    } else {
        let lib64_path = format!("{}/lib64", cuda_home);
        if Path::new(&lib64_path).exists() {
            println!("cargo:rustc-link-search=native={}", lib64_path);
        }
    }

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
}

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=/usr/lib");
    println!("cargo:rustc-link-search=/usr/lib64");

    // Link against the ibverbs library
    println!("cargo:rustc-link-lib=ibverbs");

    // Link against the mlx5 library
    println!("cargo:rustc-link-lib=mlx5");

    // Link against cuda library library
    if let Some(cuda_home) = find_cuda_home() {
        emit_cuda_link_directives(&cuda_home);
    }

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=src/rdmaxcel.h");

    // Add cargo metadata
    println!("cargo:rustc-cfg=cargo");
    println!("cargo:rustc-check-cfg=cfg(cargo)");

    // The bindgen::Builder is the main entry point to bindgen
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for
        .header("src/rdmaxcel.h")
        // Allow the specified functions, types, and variables
        .allowlist_function("ibv_.*")
        .allowlist_function("mlx5dv_.*")
        .allowlist_function("mlx5_wqe_.*")
        .allowlist_type("ibv_.*")
        .allowlist_type("mlx5dv_.*")
        .allowlist_type("mlx5_wqe_.*")
        .allowlist_var("MLX5_.*")
        // Block specific types that are manually defined in lib.rs
        .blocklist_type("ibv_wc")
        .blocklist_type("mlx5_wqe_ctrl_seg")
        // Apply the same bindgen flags as in the BUCK file
        .bitfield_enum("ibv_access_flags")
        .bitfield_enum("ibv_qp_attr_mask")
        .bitfield_enum("ibv_wc_flags")
        .bitfield_enum("ibv_send_flags")
        .bitfield_enum("ibv_port_cap_flags")
        .constified_enum_module("ibv_qp_type")
        .constified_enum_module("ibv_qp_state")
        .constified_enum_module("ibv_port_state")
        .constified_enum_module("ibv_wc_opcode")
        .constified_enum_module("ibv_wr_opcode")
        .constified_enum_module("ibv_wc_status")
        .derive_default(true)
        .prepend_enum_name(false)
        // Finish the builder and generate the bindings
        .generate()
        // Unwrap the Result and panic on failure
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
