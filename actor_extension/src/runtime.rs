/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::Cell;
use std::future::Future;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

pub fn get_tokio_runtime() -> &'static tokio::runtime::Runtime {
    static INSTANCE: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    })
}

thread_local! {
    static IS_MAIN_THREAD: Cell<bool> = const { Cell::new(false) };
}

pub fn initialize(py: Python) -> Result<()> {
    pyo3_async_runtimes::tokio::init_with_runtime(get_tokio_runtime())
        .map_err(|_| anyhow!("failed to initialize py3 async runtime"))?;

    // Initialize thread local state to identify the main Python thread.
    let threading = Python::import(py, "threading")?;
    let main_thread = threading.call_method0("main_thread")?;
    let current_thread = threading.getattr("current_thread")?.call0()?;
    ensure!(
        current_thread.is(&main_thread),
        "initialize called not on the main Python thread"
    );
    IS_MAIN_THREAD.set(true);

    Ok(())
}

/// Block the current thread on a future, but make sure to check for signals
/// originating from the Python signal handler.
///
/// Python's signal handler just sets a flag that it expects the Python
/// interpreter to handle later via a call to `PyErr_CheckSignals`. When we
/// enter into potentially long-running native code, we need to make sure to be
/// checking for signals frequently, otherwise we will ignore them. This will
/// manifest as `ctrl-C` not doing anything.
///
/// One additional wrinkle is that `PyErr_CheckSignals` only works on the main
/// Python thread; if it's called on any other thread it silently does nothing.
/// So, we check a thread-local to ensure we are on the main thread.
pub fn signal_safe_block_on<F>(py: Python, future: F) -> PyResult<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let runtime = get_tokio_runtime();
    // Release the GIL, otherwise the work in `future` that tries to acquire the
    // GIL on another thread may deadlock.
    Python::allow_threads(py, || {
        if IS_MAIN_THREAD.get() {
            // Spawn the future onto the tokio runtime
            let handle = runtime.spawn(future);
            // Block the current thread on waiting for *either* the future to
            // complete or a signal.
            runtime.block_on(async {
                tokio::select! {
                    result = handle => result.map_err(|e| PyRuntimeError::new_err(format!("JoinErr: {:?}", e))),
                    signal = async {
                        let sleep_for = std::time::Duration::from_millis(100);
                        loop {
                            // Acquiring the GIL in a loop is sad, hopefully once
                            // every 100ms is fine.
                            Python::with_gil(|py| {py.check_signals()})?;
                            #[allow(clippy::disallowed_methods)]
                            tokio::time::sleep(sleep_for).await;
                        }
                    } => signal
                }
            })
        } else {
            // If we're not on the main thread, we can just block it. We've
            // released the GIL, so the Python main thread will continue on, and
            // `PyErr_CheckSignals` doesn't do anything anyway.
            Ok(runtime.block_on(future))
        }
    })
}

/// A test function that sleeps indefinitely in a loop.
/// This is used for testing signal handling in signal_safe_block_on.
/// The function will sleep forever until interrupted by a signal.
#[pyfunction]
pub fn sleep_indefinitely_for_unit_tests(py: Python) -> PyResult<()> {
    // Create a future that sleeps indefinitely
    let future = async {
        loop {
            tracing::info!("idef sleeping for 100ms");
            #[allow(clippy::disallowed_methods)]
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    };

    // Use signal_safe_block_on to run the future, which should make it
    // interruptible by signals like SIGINT
    signal_safe_block_on(py, future)
}

/// Initialize the runtime module and expose Python functions
pub fn register_python_bindings(runtime_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    let sleep_indefinitely_fn =
        wrap_pyfunction!(sleep_indefinitely_for_unit_tests, runtime_mod.py())?;
    sleep_indefinitely_fn.setattr("__module__", "monarch._src.actor._extension.runtime")?;
    runtime_mod.add_function(sleep_indefinitely_fn)?;
    Ok(())
}
