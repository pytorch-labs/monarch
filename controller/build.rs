// This is needed due to the controller being built with torch/nccl deps due to monarch_messages.

fn main() {
    // `torch-sys` will set this env var through Cargo `links` metadata.
    let lib_path = std::env::var("DEP_TORCH_LIB_PATH").expect("DEP_TORCH_LIB_PATH to be set");
    // Set the rpath so that the dynamic linker can find libtorch and friends.
    println!("cargo::rustc-link-arg=-Wl,-rpath,{lib_path}");

    if let Ok(path) = std::env::var("DEP_NCCL_LIB_PATH") {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{path}");
    }

    // Disable new dtags, as conda envs generally use `RPATH` over `RUNPATH`.
    println!("cargo::rustc-link-arg=-Wl,--disable-new-dtags");
}
