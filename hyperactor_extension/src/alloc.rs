use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use hyperactor::WorldId;
use hyperactor::channel::ChannelTransport;
use hyperactor_mesh::alloc::Alloc;
use hyperactor_mesh::alloc::AllocConstraints;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::AllocatorError;
use hyperactor_mesh::alloc::ProcState;
use hyperactor_mesh::shape::Shape;
use ndslice::Slice;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A python class that wraps a Rust Alloc trait object. It represents what
/// is shown on the python side. Internals are not exposed.
/// It ensures that the Alloc is only used once (i.e. moved) in rust.
#[pyclass(name = "Alloc", module = "monarch._monarch.hyperactor")]
pub struct PyAlloc {
    pub inner: Arc<Mutex<Option<PyAllocWrapper>>>,
}

impl PyAlloc {
    /// Create a new PyAlloc with provided boxed trait.
    pub fn new(inner: Box<dyn Alloc + Sync + Send>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(PyAllocWrapper { inner }))),
        }
    }

    /// Take the internal Alloc object.
    pub fn take(&self) -> Option<PyAllocWrapper> {
        self.inner.lock().unwrap().take()
    }
}

/// Internal wrapper to translate from a dyn Alloc to an impl Alloc. Used
/// to support polymorphism in the Python bindings.
pub struct PyAllocWrapper {
    inner: Box<dyn Alloc + Sync + Send>,
}

#[async_trait]
impl Alloc for PyAllocWrapper {
    async fn next(&mut self) -> Option<ProcState> {
        self.inner.next().await
    }

    fn shape(&self) -> &Shape {
        self.inner.shape()
    }

    fn world_id(&self) -> &WorldId {
        self.inner.world_id()
    }

    fn transport(&self) -> ChannelTransport {
        self.inner.transport()
    }

    async fn stop(&mut self) -> Result<(), AllocatorError> {
        self.inner.stop().await
    }
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
