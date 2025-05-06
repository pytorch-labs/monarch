use std::collections::HashMap;

use hyperactor_mesh::alloc::AllocConstraints;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::shape::Shape;
use ndslice::Slice;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Helper trait that allows us to abstract over the different kinds of PyAlloc.
pub trait TakeableAlloc<T> {
    fn take(&self) -> Option<T>;
}

#[pyclass(name = "AllocConstraints", module = "monarch._monarch.hyperactor")]
pub struct PyAllocConstraints {
    inner: AllocConstraints,
}

#[pymethods]
impl PyAllocConstraints {
    #[new]
    #[pyo3(signature = (match_labels=None))]
    fn new(match_labels: Option<HashMap<String, String>>) -> PyResult<Self> {
        let mut constraints = AllocConstraints::none();
        if let Some(match_lables) = match_labels {
            constraints.match_labels = match_lables;
        }

        Ok(Self { inner: constraints })
    }
}

#[pyclass(name = "AllocSpec", module = "monarch._monarch.hyperactor")]
pub struct PyAllocSpec {
    pub inner: AllocSpec,
}

#[pymethods]
impl PyAllocSpec {
    #[new]
    #[pyo3(signature = (constraints, **kwargs))]
    fn new(constraints: &PyAllocConstraints, kwargs: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(kwargs) = kwargs else {
            return Err(PyValueError::new_err(
                "Shape must have at least one dimension",
            ));
        };
        let shape_dict = kwargs.downcast::<PyDict>()?;

        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (key, value) in shape_dict {
            keys.push(key.clone());
            values.push(value.clone());
        }

        let shape = Shape::new(
            keys.into_iter()
                .map(|key| key.extract::<String>())
                .collect::<PyResult<Vec<String>>>()?,
            Slice::new_row_major(
                values
                    .into_iter()
                    .map(|key| key.extract::<usize>())
                    .collect::<PyResult<Vec<usize>>>()?,
            ),
        )
        .map_err(|e| PyValueError::new_err(format!("Invalid shape: {:?}", e)))?;

        Ok(Self {
            inner: AllocSpec {
                shape,
                constraints: constraints.inner.clone(),
            },
        })
    }
}
