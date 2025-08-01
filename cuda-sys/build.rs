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

fn get_cuda_lib_dir() -> String {
    // Check if user explicitly set CUDA_LIB_DIR
    if let Ok(cuda_lib_dir) = env::var("CUDA_LIB_DIR") {
        return cuda_lib_dir;
    }

    // Try to deduce from CUDA_HOME
    if let Some(cuda_home) = find_cuda_home() {
        let lib64_path = format!("{}/lib64", cuda_home);
        if Path::new(&lib64_path).exists() {
            return lib64_path;
        }
    }

    // If we can't find it, error out with helpful message
    eprintln!("Error: CUDA library directory not found!");
    eprintln!("Please set CUDA_LIB_DIR environment variable to your CUDA library directory.");
    eprintln!();
    eprintln!("Example: export CUDA_LIB_DIR=/usr/local/cuda-12.0/lib64");
    eprintln!("Or: export CUDA_LIB_DIR=/usr/lib64");
    std::process::exit(1);
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

fn validate_cuda_installation() -> String {
    // Check for CUDA availability
    let cuda_home = find_cuda_home();
    if cuda_home.is_none() {
        eprintln!("Error: CUDA installation not found!");
        eprintln!("Please ensure CUDA is installed and one of the following is true:");
        eprintln!("  1. Set CUDA_HOME environment variable to your CUDA installation directory");
        eprintln!("  2. Set CUDA_PATH environment variable to your CUDA installation directory");
        eprintln!("  3. Ensure 'nvcc' is in your PATH");
        eprintln!("  4. Install CUDA to the default location (/usr/local/cuda on Linux)");
        eprintln!();
        eprintln!("Example: export CUDA_HOME=/usr/local/cuda-12.0");
        std::process::exit(1);
    }

    let cuda_home = cuda_home.unwrap();

    // Verify CUDA include directory exists
    let cuda_include_path = format!("{}/include", cuda_home);
    if !Path::new(&cuda_include_path).exists() {
        eprintln!(
            "Error: CUDA include directory not found at {}",
            cuda_include_path
        );
        eprintln!("Please verify your CUDA installation is complete.");
        std::process::exit(1);
    }

    cuda_home
}

fn main() {
    // Validate CUDA installation and get CUDA home path
    let cuda_home = validate_cuda_installation();

    // Start building the bindgen configuration
    let mut builder = bindgen::Builder::default()
        // The input header we would like to generate bindings for
        .header("src/wrapper.h")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=gnu++20")
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

    // Add CUDA include path (we already validated it exists)
    let cuda_include_path = format!("{}/include", cuda_home);
    builder = builder.clang_arg(format!("-I{}", cuda_include_path));

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

    // Get CUDA library directory and emit link directives
    let cuda_lib_dir = get_cuda_lib_dir();
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    match env::var("OUT_DIR") {
        Ok(out_dir) => {
            let out_path = PathBuf::from(out_dir);
            match builder.generate() {
                Ok(bindings) => match bindings.write_to_file(out_path.join("bindings.rs")) {
                    Ok(_) => {
                        println!("cargo::rustc-cfg=cargo");
                        println!("cargo::rustc-check-cfg=cfg(cargo)");
                    }
                    Err(e) => eprintln!("Warning: Couldn't write bindings: {}", e),
                },
                Err(e) => eprintln!("Warning: Unable to generate bindings: {}", e),
            }
        }
        Err(_) => {
            println!("Note: OUT_DIR not set, skipping bindings file generation");
        }
    }
}
