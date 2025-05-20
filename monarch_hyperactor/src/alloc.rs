use std::collections::HashMap;
use std::sync::Arc;

use hyperactor_extension::alloc::PyAlloc;
use hyperactor_extension::alloc::PyAllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::LocalAllocator;
use hyperactor_mesh::alloc::ProcessAllocator;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::process::Command;

use crate::runtime::signal_safe_block_on;

#[pyclass(
    name = "LocalAllocatorBase",
    module = "monarch._monarch.hyperactor",
    subclass
)]
pub struct PyLocalAllocator;

#[pymethods]
impl PyLocalAllocator {
    #[new]
    fn new() -> Self {
        PyLocalAllocator {}
    }

    fn allocate_nonblocking<'py>(
        &self,
        py: Python<'py>,
        spec: &PyAllocSpec,
    ) -> PyResult<Bound<'py, PyAny>> {
        // We could use Bound here, and acquire the GIL inside of `future_into_py`, but
        // it is rather awkward with the current APIs, and we can anyway support Arc/Mutex
        // pretty easily.
        let spec = spec.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalAllocator
                .allocate(spec)
                .await
                .map(|inner| PyAlloc::new(Box::new(inner)))
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })
    }

    fn allocate_blocking<'py>(&self, py: Python<'py>, spec: &PyAllocSpec) -> PyResult<PyAlloc> {
        // We could use Bound here, and acquire the GIL inside of
        // `signal_safe_block_on`, but it is rather awkward with the current
        // APIs, and we can anyway support Arc/Mutex pretty easily.
        let spec = spec.inner.clone();
        signal_safe_block_on(py, async move {
            LocalAllocator
                .allocate(spec)
                .await
                .map(|inner| PyAlloc::new(Box::new(inner)))
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })?
    }
}

#[pyclass(
    name = "ProcessAllocatorBase",
    module = "monarch._monarch.hyperactor",
    subclass
)]
pub struct PyProcessAllocator {
    inner: Arc<tokio::sync::Mutex<ProcessAllocator>>,
}

#[pymethods]
impl PyProcessAllocator {
    #[new]
    #[pyo3(signature = (cmd, args=None, env=None))]
    fn new(cmd: String, args: Option<Vec<String>>, env: Option<HashMap<String, String>>) -> Self {
        let mut cmd = Command::new(cmd);
        if let Some(args) = args {
            cmd.args(args);
        }
        if let Some(env) = env {
            cmd.envs(env);
        }
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(ProcessAllocator::new(cmd))),
        }
    }

    fn allocate_nonblocking<'py>(
        &self,
        py: Python<'py>,
        spec: &PyAllocSpec,
    ) -> PyResult<Bound<'py, PyAny>> {
        // We could use Bound here, and acquire the GIL inside of `future_into_py`, but
        // it is rather awkward with the current APIs, and we can anyway support Arc/Mutex
        // pretty easily.
        let instance = Arc::clone(&self.inner);
        let spec = spec.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            instance
                .lock()
                .await
                .allocate(spec)
                .await
                .map(|inner| PyAlloc::new(Box::new(inner)))
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })
    }

    fn allocate_blocking<'py>(&self, py: Python<'py>, spec: &PyAllocSpec) -> PyResult<PyAlloc> {
        // We could use Bound here, and acquire the GIL inside of
        // `signal_safe_block_on`, but it is rather awkward with the current
        // APIs, and we can anyway support Arc/Mutex pretty easily.
        let instance = Arc::clone(&self.inner);
        let spec = spec.inner.clone();
        signal_safe_block_on(py, async move {
            instance
                .lock()
                .await
                .allocate(spec)
                .await
                .map(|inner| PyAlloc::new(Box::new(inner)))
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })?
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyProcessAllocator>()?;
    hyperactor_mod.add_class::<PyLocalAllocator>()?;

    Ok(())
}
