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

fn python_env_dirs() -> (Option<String>, Option<String>) {
    let output = std::process::Command::new(PathBuf::from("python"))
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
    let mut builder = bindgen::Builder::default()
        // The input header we would like to generate bindings for
        .header("src/wrapper.h")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=gnu++20")
        .clang_arg(format!("-I{}/include", find_cuda_home().unwrap()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Allow the specified functions and types
        .allowlist_function("cu.*")
        .allowlist_function("CU.*")
        .allowlist_type("cu.*")
        .allowlist_type("CU.*")
        // Use newtype enum style
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        });

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

    // Write the bindings to the $OUT_DIR/bindings.rs file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    builder
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo::rustc-cfg=cargo");
    println!("cargo::rustc-check-cfg=cfg(cargo)");
}
