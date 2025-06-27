/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Utilities for launching hyperactor processes.

use std::sync::OnceLock;

use crate::clock::ClockKind;
use crate::panic_handler;

/// Initialize the Hyperactor runtime. Specifically:
/// - Set up panic handling, so that we get consistent panic stack traces in Actors.
/// - Initialize logging defaults.
pub fn initialize() {
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| {
        panic_handler::set_panic_hook();
        hyperactor_telemetry::initialize_logging(ClockKind::default());
        #[cfg(target_os = "linux")]
        linux::initialize();
    });
}

#[cfg(target_os = "linux")]
mod linux {
    use std::backtrace::Backtrace;

    use nix::sys::signal::SigHandler;

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
    }
}
