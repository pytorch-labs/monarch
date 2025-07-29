/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor_mesh::sel;
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

    /// Parses a selection expression from a string.
    ///
    /// This allows you to construct a `PySelection` using the
    /// selection algebra surface syntax, such as `"(*, 0:4, ?)"`.
    ///
    /// Raises:
    ///     ValueError: If the input string is not a valid selection
    ///     expression.
    ///
    /// Example:
    ///     PySelection.from_string("(*, 1:3, ?)") # subset of a mesh
    #[classmethod]
    #[pyo3(name = "from_string")]
    pub fn parse(_cls: Bound<'_, PyType>, input: &str) -> PyResult<Self> {
        let selection = ndslice::selection::parse::parse(input).map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!("parse error: {err}"))
        })?;

        Ok(PySelection::from(selection))
    }

    /// Selects all elements in the mesh — use this to mean "route to
    /// all nodes".
    ///
    /// The '*' expression is automatically expanded to match the
    /// dimensionality of the slice. For example, in a 3D slice, the
    /// selection becomes `*, *, *`.
    #[classmethod]
    pub fn all(_cls: Bound<'_, PyType>) -> Self {
        PySelection::from(sel!(*))
    }

    /// Selects one element nondeterministically — use this to mean
    /// "route to a single random node".
    ///
    /// The '?' expression is automatically expanded to match the
    /// dimensionality of the slice. For example, in a 3D slice, the
    /// selection becomes `?, ?, ?`.
    #[classmethod]
    pub fn any(_cls: Bound<'_, PyType>) -> Self {
        PySelection::from(sel!(?))
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySelection>()?;
    Ok(())
}
