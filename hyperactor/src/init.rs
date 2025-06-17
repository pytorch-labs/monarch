/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
    use std::backtrace::Backtrace;
    use std::process;

    use libc::PR_SET_PDEATHSIG;
    use nix::sys::signal::SIGUSR1;
    use nix::sys::signal::SigHandler;
    use nix::unistd::getpid;
    use nix::unistd::getppid;
    use tokio::signal::unix::SignalKind;
    use tokio::signal::unix::signal;

    pub(crate) fn initialize() {
        // Safety: Because I want to
        unsafe {
            extern "C" fn handle_fatal_signal(signo: libc::c_int) {
                let bt = Backtrace::force_capture();
                let signame = nix::sys::signal::Signal::try_from(signo).expect("unknown signal");
                tracing::error!("stacktrace"= %bt, "fatal signal {signo}:{signame} received");
                std::process::exit(1);
            }
            nix::sys::signal::signal(
                nix::sys::signal::SIGABRT,
                SigHandler::Handler(handle_fatal_signal),
            )
            .expect("unable to register signal handler");
            nix::sys::signal::signal(
                nix::sys::signal::SIGSEGV,
                SigHandler::Handler(handle_fatal_signal),
            )
            .expect("unable to register signal handler");
        }

        if !crate::config::global::get(crate::config::IS_MANAGED_SUBPROCESS) {
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
                    tracing::error!(
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
