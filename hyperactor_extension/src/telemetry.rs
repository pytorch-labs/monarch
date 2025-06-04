/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unsafe_op_in_unsafe_fn)]

use std::cell::Cell;

use pyo3::prelude::*;
use tracing::span::EnteredSpan;
// Thread local to store the current span
thread_local! {
    static ACTIVE_ACTOR_SPAN: Cell<Option<EnteredSpan>> = const { Cell::new(None) };
}

/// Enter the span stored in the thread local
#[pyfunction]
pub fn enter_span(module_name: String, method_name: String, actor_id: String) -> PyResult<()> {
    let mut maybe_span = ACTIVE_ACTOR_SPAN.take();
    if maybe_span.is_none() {
        maybe_span = Some(
            tracing::info_span!(
                "py_actor_method",
                name = method_name,
                target = module_name,
                actor_id = actor_id
            )
            .entered(),
        );
    }
    ACTIVE_ACTOR_SPAN.set(maybe_span);
    Ok(())
}

/// Exit the span stored in the thread local
#[pyfunction]
pub fn exit_span() -> PyResult<()> {
    ACTIVE_ACTOR_SPAN.replace(None);
    Ok(())
}

/// Log a message with the given metaata
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

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register the forward_to_tracing function
    let f = wrap_pyfunction!(forward_to_tracing, module)?;
    f.setattr(
        "__module__",
        "monarch._rust_bindings.hyperactor_extension.telemetry",
    )?;
    module.add_function(f)?;

    // Register the span-related functions
    let enter_span_fn = wrap_pyfunction!(enter_span, module)?;
    enter_span_fn.setattr(
        "__module__",
        "monarch._rust_bindings.hyperactor_extension.telemetry",
    )?;
    module.add_function(enter_span_fn)?;

    let exit_span_fn = wrap_pyfunction!(exit_span, module)?;
    exit_span_fn.setattr(
        "__module__",
        "monarch._rust_bindings.hyperactor_extension.telemetry",
    )?;
    module.add_function(exit_span_fn)?;

    Ok(())
}
