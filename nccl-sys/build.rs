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
        .header("src/nccl.h")
        .clang_arg(format!("-I{}/include", find_cuda_home().unwrap()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Communicator creation and management
        .allowlist_function("ncclGetLastError")
        .allowlist_function("ncclGetErrorString")
        .allowlist_function("ncclGetVersion")
        .allowlist_function("ncclGetUniqueId")
        .allowlist_function("ncclCommInitRank")
        .allowlist_function("ncclCommInitAll")
        .allowlist_function("ncclCommInitRankConfig")
        .allowlist_function("ncclCommInitRankScalable")
        .allowlist_function("ncclCommSplit")
        .allowlist_function("ncclCommFinalize")
        .allowlist_function("ncclCommDestroy")
        .allowlist_function("ncclCommAbort")
        .allowlist_function("ncclCommGetAsyncError")
        .allowlist_function("ncclCommCount")
        .allowlist_function("ncclCommCuDevice")
        .allowlist_function("ncclCommUserRank")
        .allowlist_function("ncclCommRegister")
        .allowlist_function("ncclCommDeregister")
        .allowlist_function("ncclMemAlloc")
        .allowlist_function("ncclMemFree")
        // Collective communication
        .allowlist_function("ncclAllReduce")
        .allowlist_function("ncclBroadcast")
        .allowlist_function("ncclReduce")
        .allowlist_function("ncclAllGather")
        .allowlist_function("ncclReduceScatter")
        // Group calls
        .allowlist_function("ncclGroupStart")
        .allowlist_function("ncclGroupEnd")
        .allowlist_function("ncclGroupSimulateEnd")
        // Point to point communication
        .allowlist_function("ncclSend")
        .allowlist_function("ncclRecv")
        // User-defined reduction operators
        .allowlist_function("ncclRedOpCreatePreMulSum")
        .allowlist_function("ncclRedOpDestroy")
        // Random nccl stuff we want
        .allowlist_function("cudaStream.*")
        .allowlist_function("cudaSetDevice")
        .allowlist_type("ncclComm_t")
        .allowlist_type("ncclResult_t")
        .allowlist_type("ncclDataType_t")
        .allowlist_type("ncclRedOp_t")
        .allowlist_type("ncclScalarResidence_t")
        .allowlist_type("ncclConfig_t")
        .allowlist_type("ncclSimInfo_t")
        .allowlist_var("NCCL_SPLIT_NOCOLOR")
        .allowlist_var("NCCL_MAJOR")
        .allowlist_var("NCCL_MINOR")
        .allowlist_var("NCCL_PATCH")
        .blocklist_type("ncclUniqueId")
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        });

    // Include headers and libs from the active environment
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

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    builder
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("cargo::rustc-link-lib=nccl");
    println!("cargo::rustc-cfg=cargo");
    println!("cargo::rustc-check-cfg=cfg(cargo)");
}
