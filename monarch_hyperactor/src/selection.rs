/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ndslice::selection::Selection;
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyclass(
    name = "Selection",
    module = "monarch._rust_bindings.monarch_hyperactor.selection",
    frozen
)]
pub struct PySelection {
    inner: Selection,
}

impl PySelection {
    pub(crate) fn inner(&self) -> &Selection {
        &self.inner
    }
}

impl From<Selection> for PySelection {
    fn from(inner: Selection) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySelection {
    #[getter]
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    #[classmethod]
    #[pyo3(name = "from_string")]
    pub fn parse(_cls: Bound<'_, PyType>, input: &str) -> PyResult<Self> {
        let selection = ndslice::selection::parse::parse(input).map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!("parse error: {err}"))
        })?;

        Ok(PySelection::from(selection))
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySelection>()?;
    Ok(())
}
