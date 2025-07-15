/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unsafe_op_in_unsafe_fn)]

use hyperactor::clock::ClockKind;
use hyperactor::clock::RealClock;
use hyperactor::clock::SimClock;
use hyperactor_telemetry::swap_telemetry_clock;
use pyo3::prelude::*;
use pyo3::types::PyTraceback;

/// Get the current span ID from the active span
#[pyfunction]
pub fn get_current_span_id() -> PyResult<u64> {
    Ok(tracing::Span::current().id().map_or(0, |id| id.into_u64()))
}

/// Log a message with the given metaata
#[pyfunction]
pub fn forward_to_tracing(py: Python, record: PyObject) -> PyResult<()> {
    let message = record.call_method0(py, "getMessage")?;
    let message: &str = message.extract(py)?;
    let lineno: i64 = record.getattr(py, "lineno")?.extract(py)?;
    let file = record.getattr(py, "filename")?;
    let file: &str = file.extract(py)?;
    let level: i32 = record.getattr(py, "levelno")?.extract(py)?;

    // Map level number to level name
    match level {
        40 | 50 => {
            let exc = record.getattr(py, "exc_info").ok();
            let traceback = exc
                .and_then(|exc| {
                    if exc.is_none(py) {
                        return None;
                    }
                    exc.extract::<(PyObject, PyObject, Bound<'_, PyTraceback>)>(py)
                        .ok()
                })
                .map(|(_, _, tb)| tb.format().unwrap_or_default());
            match traceback {
                Some(traceback) => {
                    tracing::error!(
                        file = file,
                        lineno = lineno,
                        stacktrace = traceback,
                        message
                    );
                }
                None => {
                    tracing::error!(file = file, lineno = lineno, message);
                }
            }
        }
        30 => tracing::warn!(file = file, lineno = lineno, message),
        20 => tracing::info!(file = file, lineno = lineno, message),
        10 => tracing::debug!(file = file, lineno = lineno, message),
        _ => tracing::info!(file = file, lineno = lineno, message),
    }
    Ok(())
}
#[pyfunction]
pub fn use_real_clock() -> PyResult<()> {
    swap_telemetry_clock(ClockKind::Real(RealClock));
    Ok(())
}

#[pyfunction]
pub fn use_sim_clock() -> PyResult<()> {
    swap_telemetry_clock(ClockKind::Sim(SimClock));
    Ok(())
}

#[pyclass(
    unsendable,
    subclass,
    module = "monarch._rust_bindings.monarch_hyperactor.telemetry"
)]
struct PySpan {
    span: tracing::span::EnteredSpan,
}

#[pymethods]
impl PySpan {
    #[new]
    fn new(name: &str, actor_id: Option<&str>) -> Self {
        let span = if let Some(actor_id) = actor_id {
            tracing::span!(
                tracing::Level::DEBUG,
                "python.span",
                name = name,
                actor_id = actor_id
            )
        } else {
            tracing::span!(tracing::Level::DEBUG, "python.span", name = name)
        };
        let entered_span = span.entered();

        Self { span: entered_span }
    }

    fn exit(&mut self) {
        self.span = tracing::span::Span::none().entered();
    }
}

use pyo3::Bound;
use pyo3::types::PyModule;

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register the forward_to_tracing function
    let f = wrap_pyfunction!(forward_to_tracing, module)?;
    f.setattr(
        "__module__",
        "monarch._rust_bindings.monarch_hyperactor.telemetry",
    )?;
    module.add_function(f)?;

    // Register the span-related functions
    let get_current_span_id_fn = wrap_pyfunction!(get_current_span_id, module)?;
    get_current_span_id_fn.setattr(
        "__module__",
        "monarch._rust_bindings.monarch_hyperactor.telemetry",
    )?;
    module.add_function(get_current_span_id_fn)?;

    let use_real_clock_fn = wrap_pyfunction!(use_real_clock, module)?;
    use_real_clock_fn.setattr(
        "__module__",
        "monarch._rust_bindings.monarch_hyperactor.telemetry",
    )?;
    module.add_function(use_real_clock_fn)?;

    let use_sim_clock_fn = wrap_pyfunction!(use_sim_clock, module)?;
    use_sim_clock_fn.setattr(
        "__module__",
        "monarch._rust_bindings.monarch_hyperactor.telemetry",
    )?;
    module.add_function(use_sim_clock_fn)?;

    module.add_class::<PySpan>()?;
    Ok(())
}
