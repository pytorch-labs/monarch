/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ndslice::Shape;
use ndslice::Slice;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;

use crate::ndslice::PySlice;

#[pyclass(
    name = "Shape",
    module = "monarch._rust_bindings.monarch_hyperactor.shape",
    frozen
)]
pub struct PyShape {
    pub(super) inner: Shape,
}
#[pymethods]
impl PyShape {
    #[new]
    fn new(labels: Vec<String>, slice: PySlice) -> PyResult<Self> {
        let shape = Shape::new(labels, Slice::from(slice))
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;
        Ok(PyShape { inner: shape })
    }

    #[getter]
    fn ndslice(&self) -> PySlice {
        self.inner.slice().clone().into()
    }
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner.labels().to_vec()
    }
    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }
    fn coordinates<'py>(
        &self,
        py: Python<'py>,
        rank: usize,
    ) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        self.inner
            .coordinates(rank)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))
            .and_then(|x| PyDict::from_sequence_bound(x.to_object(py).bind(py)))
    }

    #[pyo3(signature = (**kwargs))]
    fn index(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyShape> {
        if let Some(kwargs) = kwargs {
            let mut indices: Vec<(String, usize)> = Vec::new();
            // translate kwargs into indices
            for (key, value) in kwargs.iter() {
                let key_str = key.extract::<String>()?;
                let idx = value.extract::<usize>()?;
                indices.push((key_str, idx));
            }
            Ok(PyShape {
                inner: self
                    .inner
                    .index(indices)
                    .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?,
            })
        } else {
            Ok(PyShape {
                inner: self.inner.clone(),
            })
        }
    }

    #[staticmethod]
    fn from_bytes(bytes: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let shape: Shape = bincode::deserialize(bytes.as_bytes())
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;
        Ok(PyShape::from(shape))
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let bytes = bincode::serialize(&slf.borrow().inner)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;
        let py_bytes = PyBytes::new_bound(slf.py(), &bytes);
        Ok((slf.getattr("from_bytes")?, (py_bytes,)))
    }

    fn ranks(&self) -> Vec<usize> {
        self.inner.slice().iter().collect()
    }

    fn __len__(&self) -> usize {
        self.inner.slice().len()
    }

    #[staticmethod]
    fn unity() -> PyShape {
        Shape::unity().into()
    }
}

impl From<Shape> for PyShape {
    fn from(shape: Shape) -> Self {
        PyShape { inner: shape }
    }
}

#[pyclass(
    name = "Point",
    module = "monarch._rust_bindings.monarch_hyperactor.shape",
    subclass,
    frozen
)]

struct PyPoint {
    rank: usize,
    shape: Py<PyShape>,
}

#[pymethods]
impl PyPoint {
    #[new]
    fn new(rank: usize, shape: Py<PyShape>) -> Self {
        PyPoint { rank, shape }
    }
    fn __getitem__(&self, py: Python, label: &str) -> PyResult<usize> {
        let shape = self.shape.bind(py).get();
        let ranks = shape
            .inner
            .slice()
            .coordinates(self.rank)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;
        if let Some(index) = shape.inner.labels().iter().position(|l| l == label) {
            Ok(ranks[index])
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "Dimension '{}' not found",
                label
            )))
        }
    }
    fn __len__(&self, py: Python) -> usize {
        self.shape.bind(py).get().__len__()
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.shape
            .bind(py)
            .get()
            .labels()
            .to_object(py)
            .call_method0(py, "__iter__")
    }
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> (
        pyo3::Bound<'py, pyo3::types::PyType>,
        (usize, pyo3::Py<PyShape>),
    ) {
        (
            slf.get_type(),
            (slf.get().rank, slf.get().shape.clone_ref(slf.py())),
        )
    }
    #[getter]
    fn shape(&self, py: Python<'_>) -> Py<PyShape> {
        self.shape.clone_ref(py)
    }
    #[getter]
    fn rank(&self) -> usize {
        self.rank
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyShape>()?;
    module.add_class::<PySlice>()?;
    module.add_class::<PyPoint>()?;
    Ok(())
}
