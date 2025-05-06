//! Utilities for launching hyperactor processes.

use std::sync::LazyLock;
use std::sync::OnceLock;

use crate::panic_handler;

/// A global runtime used in binding async and sync code. Do not use for executing long running or
/// compute intensive tasks.
pub(crate) static RUNTIME: LazyLock<tokio::runtime::Runtime> =
    LazyLock::new(|| tokio::runtime::Runtime::new().expect("failed to create global runtime"));

/// Initialize the Hyperactor runtime. Specifically:
/// - Set up panic handling, so that we get consistent panic stack traces in Actors.
/// - Initialize logging defaults.
/// - On Linux, set up signal handlers to ensure that managed child processes are reliably
///   terminated when their parents die. This is indicated by the environment variable
///   `HYPERACTOR_MANAGED_SUBPROCESS`.
pub fn initialize() {
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| {
        panic_handler::set_panic_hook();
        hyperactor_telemetry::initialize_logging();
        #[cfg(target_os = "linux")]
        linux::initialize();
    });
}

#[cfg(target_os = "linux")]
mod linux {
    use std::env;
    use std::process;

    use libc::PR_SET_PDEATHSIG;
    use nix::sys::signal::SIGUSR1;
    use nix::unistd::getpid;
    use nix::unistd::getppid;
    use tokio::signal::unix::SignalKind;
    use tokio::signal::unix::signal;

    pub(crate) fn initialize() {
        if env::var("HYPERACTOR_MANAGED_SUBPROCESS").is_err() {
            return;
        }

        super::RUNTIME.spawn(async {
            match signal(SignalKind::user_defined1()) {
                Ok(mut sigusr1) => {
                    // SAFETY: required for signal handling
                    unsafe {
                        libc::prctl(PR_SET_PDEATHSIG, SIGUSR1);
                    }
                    sigusr1.recv().await;
                    eprintln!(
                        "hyperactor[{}]: parent process {} died; exiting",
                        getpid(),
                        getppid()
                    );
                    process::exit(1);
                }
                Err(err) => {
                    eprintln!("failed to set up SIGUSR1 signal handler: {:?}", err);
                }
            }
        });
    }
}
