/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

/// Log a message with the given metadata
#[pyfunction]
pub fn forward_to_tracing(message: &str, file: &str, lineno: i64, level: i32) {
    // Map level number to level name
    match level {
        40 => tracing::error!(file = file, lineno = lineno, message),
        30 => tracing::warn!(file = file, lineno = lineno, message),
        20 => tracing::info!(file = file, lineno = lineno, message),
        10 => tracing::debug!(file = file, lineno = lineno, message),
        _ => tracing::info!(file = file, lineno = lineno, message),
    }
}

use pyo3::Bound;
use pyo3::types::PyModule;

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let f = wrap_pyfunction!(forward_to_tracing, module)?;
    f.setattr(
        "__module__",
        "monarch._rust_bindings.hyperactor_extension.telemetry",
    )?;
    module.add_function(f)?;
    Ok(())
}
