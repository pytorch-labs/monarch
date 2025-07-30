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

const PYTHON_PRINT_DIRS: &str = r"
import sysconfig
print('PYTHON_INCLUDE_DIR:', sysconfig.get_config_var('INCLUDEDIR'))
print('PYTHON_LIB_DIR:', sysconfig.get_config_var('LIBDIR'))
";

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
                // Walk through possible locations, starting with newest
                for candidate in &[
                    "/usr/local/cuda-12.8",
                    "/usr/local/cuda-12.6",
                    "/usr/local/cuda-12.4",
                    "/usr/local/cuda-12.2",
                    "/usr/local/cuda-12.1",
                    "/usr/local/cuda-12.0",
                    "/usr/local/cuda-11.8",
                    "/usr/local/cuda-11.7",
                    "/usr/local/cuda-11.6",
                    "/usr/local/cuda-11.5",
                ] {
                    if Path::new(candidate).exists() {
                        cuda_home = Some(candidate.to_string());
                        break;
                    }
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

fn python_env_dirs() -> (Option<String>, Option<String>) {
    let output = std::process::Command::new(PathBuf::from("python3"))
        .arg("-c")
        .arg(PYTHON_PRINT_DIRS)
        .output()
        .unwrap_or_else(|_| panic!("error running python"));

    let mut include_dir = None;
    let mut lib_dir = None;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some(path) = line.strip_prefix("PYTHON_INCLUDE_DIR: ") {
            include_dir = Some(path.to_string());
        }
        if let Some(path) = line.strip_prefix("PYTHON_LIB_DIR: ") {
            lib_dir = Some(path.to_string());
        }
    }
    (include_dir, lib_dir)
}

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=/usr/lib");
    println!("cargo:rustc-link-search=/usr/lib64");

    // Link against the ibverbs library
    println!("cargo:rustc-link-lib=ibverbs");

    // Link against the mlx5 library
    println!("cargo:rustc-link-lib=mlx5");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=src/rdmaxcel.h");

    // Get the directory of the current crate
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| {
        // For buck2 run, we know the package is in fbcode/monarch/rdmaxcel-sys
        // Get the fbsource directory from the current directory path
        let current_dir = std::env::current_dir().expect("Failed to get current directory");
        let current_path = current_dir.to_string_lossy();

        // Find the fbsource part of the path
        if let Some(fbsource_pos) = current_path.find("fbsource") {
            let fbsource_path = &current_path[..fbsource_pos + "fbsource".len()];
            format!("{}/fbcode/monarch/rdmaxcel-sys", fbsource_path)
        } else {
            // If we can't find fbsource in the path, just use the current directory
            format!("{}/src", current_dir.to_string_lossy())
        }
    });

    // Create the absolute path to the header file
    let header_path = format!("{}/src/rdmaxcel.h", manifest_dir);

    // Check if the header file exists
    if !Path::new(&header_path).exists() {
        panic!("Header file not found at {}", header_path);
    }

    // Start building the bindgen configuration
    let mut builder = bindgen::Builder::default()
        // The input header we would like to generate bindings for
        .header(&header_path)
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=gnu++20")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Allow the specified functions, types, and variables
        .allowlist_function("ibv_.*")
        .allowlist_function("mlx5dv_.*")
        .allowlist_function("mlx5_wqe_.*")
        .allowlist_function("create_qp")
        .allowlist_function("create_mlx5dv_.*")
        .allowlist_function("register_cuda_memory")
        .allowlist_function("db_ring")
        .allowlist_function("cqe_poll")
        .allowlist_function("send_wqe")
        .allowlist_function("recv_wqe")
        .allowlist_function("launch_db_ring")
        .allowlist_function("launch_cqe_poll")
        .allowlist_function("launch_send_wqe")
        .allowlist_function("launch_recv_wqe")
        .allowlist_type("ibv_.*")
        .allowlist_type("mlx5dv_.*")
        .allowlist_type("mlx5_wqe_.*")
        .allowlist_type("cqe_poll_result_t")
        .allowlist_type("wqe_params_t")
        .allowlist_type("cqe_poll_params_t")
        .allowlist_var("MLX5_.*")
        .allowlist_var("IBV_.*")
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
        .prepend_enum_name(false);

    // Add CUDA include path if available
    if let Some(cuda_home) = find_cuda_home() {
        let cuda_include_path = format!("{}/include", cuda_home);
        if Path::new(&cuda_include_path).exists() {
            builder = builder.clang_arg(format!("-I{}", cuda_include_path));
        } else {
            eprintln!(
                "Warning: CUDA include directory not found at {}",
                cuda_include_path
            );
        }
    } else {
        eprintln!("Warning: CUDA home directory not found. Continuing without CUDA include path.");
    }

    // Include headers and libs from the active environment.
    let (include_dir, lib_dir) = python_env_dirs();
    if let Some(include_dir) = include_dir {
        builder = builder.clang_arg(format!("-I{}", include_dir));
    }
    if let Some(lib_dir) = lib_dir {
        println!("cargo::rustc-link-search=native={}", lib_dir);
        // Set cargo metadata to inform dependent binaries about how to set their
        // RPATH (see controller/build.rs for an example).
        println!("cargo::metadata=LIB_PATH={}", lib_dir);
    }
    if let Some(cuda_home) = find_cuda_home() {
        emit_cuda_link_directives(&cuda_home);
    }

    // Generate bindings
    let bindings = builder.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    match env::var("OUT_DIR") {
        Ok(out_dir) => {
            let out_path = PathBuf::from(out_dir);
            match bindings.write_to_file(out_path.join("bindings.rs")) {
                Ok(_) => {
                    println!("cargo:rustc-cfg=cargo");
                    println!("cargo:rustc-check-cfg=cfg(cargo)");
                }
                Err(e) => eprintln!("Warning: Couldn't write bindings: {}", e),
            }
        }
        Err(_) => {
            println!("Note: OUT_DIR not set, skipping bindings file generation");
        }
    }
}
