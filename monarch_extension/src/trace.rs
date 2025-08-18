/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyfunction]
fn get_or_create_trace_id() -> String {
    hyperactor_telemetry::trace::get_or_create_trace_id()
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let f = wrap_pyfunction!(get_or_create_trace_id, module)?;
    f.setattr(
        "__module__",
        "monarch._rust_bindings.monarch_extension.trace",
    )?;
    module.add_function(f)?;
    Ok(())
}
