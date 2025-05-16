fn main() {
    println!("cargo::rustc-check-cfg=cfg(enable_hyperactor_message_logging)");
}
