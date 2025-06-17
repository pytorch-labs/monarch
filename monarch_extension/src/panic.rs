/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyo3::prelude::*;

/// A function that panics when called.
/// This is used for testing panic handling in the Python bindings.
#[pyfunction]
pub fn panicking_function() {
    panic!("This is a deliberate panic from panicking_function");
}

/// Register Python bindings for the panic module.
pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let f = wrap_pyfunction!(panicking_function, module)?;
    f.setattr(
        "__module__",
        "monarch._rust_bindings.monarch_extension.panic",
    )?;
    module.add_function(f)?;
    Ok(())
}
