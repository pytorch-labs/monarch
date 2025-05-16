use std::collections::HashMap;
use std::sync::Arc;

use hyperactor_mesh::alloc::LocalAllocator;
use hyperactor_mesh::alloc::ProcessAllocator;
use hyperactor_meta_lib::alloc_mock::MockMast;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::alloc::PyMastAllocator;
use crate::alloc::PyMastAllocatorConfig;

#[pyclass(
    name = "MockMast",
    module = "monarch_meta._monarch_meta.hyperactor_meta"
)]
pub struct PyMockMast {
    inner: Arc<Mutex<MockMast>>,
}

#[pymethods]
impl PyMockMast {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MockMast::new())),
        }
    }

    pub fn add_local_task_group<'py>(
        &mut self,
        py: Python<'py>,
        name: String,
        num_tasks: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let instance = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            instance
                .lock()
                .await
                .add_task_group(name, num_tasks, LocalAllocator {})
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        })
    }

    #[pyo3(signature = (name, num_tasks, cmd, args=None, env=None))]
    pub fn add_process_task_group<'py>(
        &mut self,
        py: Python<'py>,
        name: String,
        num_tasks: u64,
        cmd: String,
        args: Option<Vec<String>>,
        env: Option<HashMap<String, String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut cmd = Command::new(cmd);
        if let Some(args) = args {
            cmd.args(args);
        }
        if let Some(env) = env {
            cmd.envs(env);
        }
        let instance = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            instance
                .lock()
                .await
                .add_task_group(name, num_tasks, ProcessAllocator::new(cmd))
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        })
    }

    pub fn stop_task_group<'py>(
        &mut self,
        py: Python<'py>,
        name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let instance = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            instance
                .lock()
                .await
                .stop_task_group(name)
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        })
    }

    pub fn get_mast_allocator<'py>(
        &mut self,
        py: Python<'py>,
        config: PyMastAllocatorConfig,
    ) -> PyResult<Bound<'py, PyAny>> {
        let instance = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            instance
                .lock()
                .await
                .get_mast_allocator(config.inner)
                .map(|allocator| PyMastAllocator {
                    inner: Arc::new(tokio::sync::Mutex::new(allocator)),
                })
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        })
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMockMast>()?;
    Ok(())
}
