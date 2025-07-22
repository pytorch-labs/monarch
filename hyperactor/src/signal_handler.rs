/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use nix::sys::signal;
use tokio_stream::StreamExt;

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
        tracing::info!("registered signal cleanup callback with ID: {}", id);
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
