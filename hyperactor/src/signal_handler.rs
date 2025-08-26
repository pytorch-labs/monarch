/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![cfg(unix)]

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::io;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use nix::libc;
use nix::sys::signal;
use tokio_stream::StreamExt;

/// This type describes how a signal is currently handled by the
/// process.
///
/// This is derived from the kernel's `sigaction` for a given signal,
/// normalized into three categories:
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDisposition {
    /// The signal is explicitly ignored (`SIG_IGN`).
    Ignored,
    /// The default action for the signal will occur (`SIG_DFL`).
    Default,
    /// A custom signal handler has been installed (either via
    /// `sa_handler` or `sa_sigaction`).
    Custom,
}

impl fmt::Display for SignalDisposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalDisposition::Ignored => write!(f, "ignored"),
            SignalDisposition::Default => write!(f, "default"),
            SignalDisposition::Custom => write!(f, "custom handler"),
        }
    }
}

/// Query the current disposition of a signal (`signum`).
///
/// This inspects the kernel's `sigaction` state for the given signal
/// without changing it (by passing `act = NULL`).
///
/// Returns:
/// - [`SignalDisposition::Ignored`] if the handler is `SIG_IGN`
/// - [`SignalDisposition::Default`] if the handler is `SIG_DFL`
/// - [`SignalDisposition::Custom`] if a user-installed handler is
///   present
///
/// # Errors
/// Returns an `io::Error` if the underlying `sigaction` call fails,
/// for example if `signum` is invalid.
pub fn query_signal_disposition(signum: libc::c_int) -> io::Result<SignalDisposition> {
    // SAFETY:
    // - We call `libc::sigaction` with `act = NULL` to query state
    //   only.
    // - `old` is a properly allocated `MaybeUninit<sigaction>`, large
    //    enough to hold the kernel response.
    // - `sigaction` will write to `old` before we read it.
    // - Interpreting the union field (`sa_sigaction`) as a function
    //   pointer is safe here because we only compare it against the
    //   constants `SIG_IGN` and `SIG_DFL`.
    // - No undefined behavior results because we never call the
    //   pointer, we only compare its value.
    unsafe {
        // Query-only: act = NULL, oldact = &old.
        let mut old = MaybeUninit::<libc::sigaction>::uninit();
        if libc::sigaction(signum, ptr::null(), old.as_mut_ptr()) != 0 {
            return Err(io::Error::last_os_error());
        }
        let old = old.assume_init();

        // If SA_SIGINFO is set, the union stores a 3-arg handler =>
        // custom handler.
        if (old.sa_flags & libc::SA_SIGINFO) != 0 {
            return Ok(SignalDisposition::Custom);
        }

        // Otherwise the union stores the 1-arg handler. `libc`
        // exposes it as `sa_sigaction` in Rust. Compare the
        // function-pointer value against `SIG_IGN`/`SIG_DFL`.
        let handler = old.sa_sigaction;
        let ignore = libc::SIG_IGN;
        let default = libc::SIG_DFL;

        match handler {
            h if h == ignore => Ok(SignalDisposition::Ignored),
            h if h == default => Ok(SignalDisposition::Default),
            _ => Ok(SignalDisposition::Custom),
        }
    }
}

/// Returns the current [`SignalDisposition`] of `SIGPIPE`.
///
/// This is a convenience wrapper around [`query_signal_disposition`]
/// that checks specifically for the `SIGPIPE` signal. By default,
/// Rust's runtime startup code installs `SIG_IGN` for `SIGPIPE` (see
/// <https://github.com/rust-lang/rust/issues/62569>), but this
/// function lets you confirm whether it is currently ignored, set to
/// the default action, or handled by a custom handler.
pub fn sigpipe_disposition() -> io::Result<SignalDisposition> {
    query_signal_disposition(libc::SIGPIPE)
}

type AsyncCleanupCallback = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Global signal manager that coordinates cleanup across all signal handlers
pub(crate) struct GlobalSignalManager {
    cleanup_callbacks: Arc<Mutex<HashMap<u64, AsyncCleanupCallback>>>,
    next_id: Arc<Mutex<u64>>,
    _listener: tokio::task::JoinHandle<()>,
}

impl GlobalSignalManager {
    fn new() -> Self {
        let listener = tokio::spawn(async move {
            if let Ok(mut signals) =
                signal_hook_tokio::Signals::new([signal::SIGINT as i32, signal::SIGTERM as i32])
            {
                if let Some(signal) = signals.next().await {
                    // If parent died, stdout/stderr are broken pipes
                    // that cause uninterruptible sleep on write.
                    // Detect and redirect to file to prevent hanging.
                    crate::stdio_redirect::handle_broken_pipes();

                    tracing::info!("received signal: {}", signal);

                    get_signal_manager().execute_all_cleanups().await;

                    match signal::Signal::try_from(signal) {
                        Ok(sig) => {
                            if let Err(err) =
                                // SAFETY: We're setting the handle to SigDfl (default system behaviour)
                                unsafe { signal::signal(sig, signal::SigHandler::SigDfl) }
                            {
                                tracing::error!(
                                    "failed to restore default signal handler for {}: {}",
                                    sig,
                                    err
                                );
                            }

                            // Re-raise the signal to trigger default behavior (process termination)
                            if let Err(err) = signal::raise(sig) {
                                tracing::error!("failed to re-raise signal {}: {}", sig, err);
                            }
                        }
                        Err(err) => {
                            tracing::error!("failed to convert signal {}: {}", signal, err);
                        }
                    }
                }
            }
        });
        Self {
            cleanup_callbacks: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(0)),
            _listener: listener,
        }
    }

    /// Register a cleanup callback and return a unique ID for later unregistration
    fn register_cleanup(&self, callback: AsyncCleanupCallback) -> u64 {
        let mut next_id = self.next_id.lock().unwrap_or_else(|e| e.into_inner());
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        let mut callbacks = self
            .cleanup_callbacks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        callbacks.insert(id, callback);
        tracing::info!(
            "process {} registered signal cleanup callback with ID: {}",
            std::process::id(),
            id
        );
        id
    }

    /// Unregister a cleanup callback by ID
    fn unregister_cleanup(&self, id: u64) {
        let mut callbacks = self
            .cleanup_callbacks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if callbacks.remove(&id).is_some() {
            tracing::info!("unregistered signal cleanup callback with ID: {}", id);
        } else {
            tracing::warn!(
                "attempted to unregister non-existent cleanup callback with ID: {}",
                id
            );
        }
    }

    /// Execute all registered cleanup callbacks asynchronously
    async fn execute_all_cleanups(&self) {
        let callbacks = {
            let mut callbacks = self
                .cleanup_callbacks
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *callbacks)
        };

        let futures = callbacks.into_iter().map(|(id, future)| async move {
            tracing::debug!("executing cleanup callback with ID: {}", id);
            future.await;
        });

        futures::future::join_all(futures).await;
    }
}

/// Global instance of the signal manager
static SIGNAL_MANAGER: OnceLock<GlobalSignalManager> = OnceLock::new();

/// Get the global signal manager instance
pub(crate) fn get_signal_manager() -> &'static GlobalSignalManager {
    SIGNAL_MANAGER.get_or_init(GlobalSignalManager::new)
}

/// RAII guard that automatically unregisters a signal cleanup callback when dropped
pub struct SignalCleanupGuard {
    id: u64,
}

impl SignalCleanupGuard {
    fn new(id: u64) -> Self {
        Self { id }
    }

    /// Get the ID of the registered cleanup callback
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for SignalCleanupGuard {
    fn drop(&mut self) {
        get_signal_manager().unregister_cleanup(self.id);
    }
}

/// Register a cleanup callback to be executed on SIGINT/SIGTERM
/// Returns a unique ID that can be used to unregister the callback
pub fn register_signal_cleanup(callback: AsyncCleanupCallback) -> u64 {
    get_signal_manager().register_cleanup(callback)
}

/// Register a scoped cleanup callback to be executed on SIGINT/SIGTERM
/// Returns a guard that automatically unregisters the callback when dropped
pub fn register_signal_cleanup_scoped(callback: AsyncCleanupCallback) -> SignalCleanupGuard {
    let id = get_signal_manager().register_cleanup(callback);
    SignalCleanupGuard::new(id)
}

/// Unregister a previously registered cleanup callback
pub fn unregister_signal_cleanup(id: u64) {
    get_signal_manager().unregister_cleanup(id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigpipe_is_ignored_by_default() {
        let disp = sigpipe_disposition().expect("query failed");
        assert_eq!(
            disp,
            SignalDisposition::Ignored,
            "expected SIGPIPE to be ignored by default"
        );
    }
}
