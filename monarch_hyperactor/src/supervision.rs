/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

create_exception!(
    monarch._rust_bindings.monarch_hyperactor.supervision,
    SupervisionError,
    PyRuntimeError
);

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Get the Python interpreter instance from the module
    let py = module.py();
    // Add the exception to the module using its type object
    module.add("SupervisionError", py.get_type::<SupervisionError>())?;
    Ok(())
}
