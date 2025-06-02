/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! This build script locates the PyTorch libraries and headers based on your
//! current Python environment and makes them available to the cxx bridge for
//! linking. It does the same for CUDA.
//!
//! This script is not very general atm. Functionality that we would probably want:
//! * Support for platforms other than linux.
//! * Support for CPU only builds, HIP builds, etc.

#![feature(exit_status_error)]

use std::path::PathBuf;
use std::process::Stdio;

use cxx_build::CFG;

// From: https://github.com/LaurentMazare/tch-rs/blob/main/torch-sys/build.rs
const PYTHON_PRINT_PYTORCH_DETAILS: &str = r"
import torch
from torch.utils import cpp_extension
print('LIBTORCH_CXX11:', torch._C._GLIBCXX_USE_CXX11_ABI)
for include_path in cpp_extension.include_paths():
    print('LIBTORCH_INCLUDE:', include_path)
for library_path in cpp_extension.library_paths():
    print('LIBTORCH_LIB:', library_path)
print('CUDA_HOME:', cpp_extension.CUDA_HOME)
";

const PYTHON_PRINT_INCLUDE_PATH: &str = r"
import sysconfig
print('PYTHON_INCLUDE:', sysconfig.get_path('include'))
print('PYTHON_INCLUDE_DIR:', sysconfig.get_config_var('INCLUDEDIR'))
print('PYTHON_LIB_DIR:', sysconfig.get_config_var('LIBDIR'))
";

fn get_env_var_with_rerun(name: &str) -> Result<String, std::env::VarError> {
    println!("cargo::rerun-if-env-changed={}", name);
    std::env::var(name)
}

fn main() {
    let mut libtorch_include_dirs: Vec<PathBuf> = vec![];
    let mut libtorch_lib_dir: Option<PathBuf> = None;
    let mut cuda_home: Option<PathBuf> = None;
    let mut cxx11_abi = None;
    let python_interpreter = PathBuf::from("python");

    let use_pytorch_apis =
        get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS").unwrap_or_else(|_| "1".to_owned());
    if use_pytorch_apis == "1" {
        // We use the user's python installation of PyTorch to get the proper
        // headers/libraries for libtorch
        let output = std::process::Command::new(&python_interpreter)
            .arg("-c")
            .arg(PYTHON_PRINT_PYTORCH_DETAILS)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap_or_else(|_| panic!("error spawning {python_interpreter:?}"))
            .wait_with_output()
            .unwrap_or_else(|_| panic!("error waiting for {python_interpreter:?}"));
        output
            .status
            .exit_ok()
            .unwrap_or_else(|_| panic!("error running {python_interpreter:?}"));

        for line in String::from_utf8_lossy(&output.stdout).lines() {
            match line.strip_prefix("LIBTORCH_CXX11: ") {
                Some("False") => cxx11_abi = Some("0".to_owned()),
                Some("True") => cxx11_abi = Some("1".to_owned()),
                _ => {}
            };
            if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
                libtorch_include_dirs.push(PathBuf::from(path))
            }
            if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                libtorch_lib_dir = Some(PathBuf::from(path))
            }
            if let Some(path) = line.strip_prefix("CUDA_HOME: ") {
                cuda_home = Some(PathBuf::from(path))
            }
        }
    } else {
        cxx11_abi = Some(get_env_var_with_rerun("_GLIBCXX_USE_CXX11_ABI").unwrap());
        libtorch_include_dirs.extend(
            get_env_var_with_rerun("LIBTORCH_INCLUDE")
                .unwrap()
                .split(':')
                .map(|s| s.into()),
        );
        libtorch_lib_dir = Some(get_env_var_with_rerun("LIBTORCH_LIB").unwrap().into());
        cuda_home = Some(get_env_var_with_rerun("CUDA_HOME").unwrap().into());
    }

    let mut python_include: Option<PathBuf> = None;
    let mut python_include_dir: Option<PathBuf> = None;
    // Include Python headers, and headers / libs from the active env.
    let output = std::process::Command::new(&python_interpreter)
        .arg("-c")
        .arg(PYTHON_PRINT_INCLUDE_PATH)
        .output()
        .unwrap_or_else(|_| panic!("error running {python_interpreter:?}"));
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some(path) = line.strip_prefix("PYTHON_INCLUDE: ") {
            python_include = Some(PathBuf::from(path));
        }
        if let Some(path) = line.strip_prefix("PYTHON_INCLUDE_DIR: ") {
            python_include_dir = Some(PathBuf::from(path));
        }
        if let Some(path) = line.strip_prefix("PYTHON_LIB_DIR: ") {
            println!("cargo::rustc-link-search=native={}", path);
        }
    }

    let bindings = bindgen::Builder::default()
        .header("src/torch.hpp")
        .clang_args(
            libtorch_include_dirs
                .iter()
                .map(|path| format!("-I{}", path.display())),
        )
        .clang_arg(format!(
            "-I{}",
            python_include_dir.clone().unwrap().display()
        ))
        .clang_arg("-std=c++20")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_type("c10::MemoryFormat")
        .allowlist_type("c10::ScalarType")
        .allowlist_type("c10::Layout")
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .enable_cxx_namespaces()
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("cargo::rerun-if-changed=src/torch.hpp");

    // Include CUDA toolkit
    libtorch_include_dirs.push(format!("{}/include", cuda_home.clone().unwrap().display()).into());

    // Prefix includes with `monarch` to maintain consistency with fbcode
    // folder structre
    CFG.include_prefix = "monarch/torch-sys";
    cxx_build::bridge("src/bridge.rs")
        .file("src/bridge.cpp")
        .std("c++20")
        .includes(&libtorch_include_dirs)
        .include(python_include.unwrap())
        .include(python_include_dir.unwrap())
        // Suppress warnings, otherwise we get massive spew from libtorch
        .flag_if_supported("-w")
        .flag(&format!(
            "-Wl,-rpath={}",
            libtorch_lib_dir.clone().unwrap().display()
        ))
        .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi.unwrap()))
        .compile("torch-sys");

    // Link against the various torch libs
    println!(
        "cargo::rustc-link-search=native={}",
        libtorch_lib_dir.clone().unwrap().display()
    );
    println!("cargo::rustc-link-lib=torch_cuda");
    println!("cargo::rustc-link-lib=torch_cpu");
    println!("cargo::rustc-link-lib=torch");
    println!("cargo::rustc-link-lib=torch_python");
    println!("cargo::rustc-link-lib=c10");
    println!("cargo::rustc-link-lib=c10_cuda");
    println!(
        "cargo::rustc-link-arg=-Wl,-rpath,{}",
        libtorch_lib_dir.clone().unwrap().display()
    );

    // Link against CUDA runtime libs
    println!(
        "cargo::rustc-link-search=native={}/lib64",
        cuda_home.unwrap().display()
    );
    println!("cargo::rustc-link-lib=cudart");

    // Set cargo metadata to inform dependent binaries about how to set their
    // RPATH (see monarch_tensor_worker/build.rs for an example).
    println!(
        "cargo::metadata=LIB_PATH={}",
        libtorch_lib_dir.clone().unwrap().display()
    );

    println!("cargo::rerun-if-changed=src/bridge.rs");
    println!("cargo::rerun-if-changed=src/bridge.cpp");
    println!("cargo::rerun-if-changed=src/bridge.h");
}
