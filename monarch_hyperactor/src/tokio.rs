/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::future::Future;
use std::pin::Pin;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use tokio::sync::Mutex;

use crate::runtime::signal_safe_block_on;

/// Helper struct to make a Python future passable in an actor message.
///
/// Also so that we don't have to write this massive type signature everywhere
pub(crate) struct PythonTask {
    future: Mutex<Pin<Box<dyn Future<Output = PyResult<PyObject>> + Send + 'static>>>,
}

impl PythonTask {
    pub(crate) fn new(fut: impl Future<Output = PyResult<PyObject>> + Send + 'static) -> Self {
        Self {
            future: Mutex::new(Box::pin(fut)),
        }
    }

    pub(crate) fn take(self) -> Pin<Box<dyn Future<Output = PyResult<PyObject>> + Send + 'static>> {
        self.future.into_inner()
    }
}

impl std::fmt::Debug for PythonTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonTask")
            .field("future", &"<PythonFuture>")
            .finish()
    }
}

#[pyclass(
    name = "PythonTask",
    module = "monarch._rust_bindings.monarch_hyperactor.tokio"
)]
pub struct PyPythonTask {
    inner: Option<PythonTask>,
}

impl From<PythonTask> for PyPythonTask {
    fn from(task: PythonTask) -> Self {
        Self { inner: Some(task) }
    }
}

#[pyclass(
    name = "JustStopWithValueIterator",
    module = "monarch._rust_bindings.monarch_hyperactor.actor"
)]
struct JustStopWithValueIterator {
    value: Option<PyObject>,
}

#[pymethods]
impl JustStopWithValueIterator {
    fn __next__(&mut self) -> PyResult<PyObject> {
        Err(PyStopIteration::new_err(self.value.take().unwrap()))
    }
}

impl PyPythonTask {
    pub fn new<F, T>(fut: F) -> PyResult<Self>
    where
        F: Future<Output = PyResult<T>> + Send + 'static,
        T: for<'py> IntoPyObject<'py>,
    {
        Ok(PythonTask::new(async {
            fut.await
                .and_then(|t| Python::with_gil(|py| t.into_py_any(py)))
        })
        .into())
    }
}

#[pymethods]
impl PyPythonTask {
    // kill this, its python code
    fn into_future(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let task = self
            .inner
            .take()
            .map(|task| task.take())
            .expect("PythonTask already consumed");
        Ok(pyo3_async_runtimes::tokio::future_into_py(py, task)?.unbind())
    }
    // rename to block_in_place
    fn block_on(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let task = self
            .inner
            .take()
            .map(|task| task.take())
            .expect("PythonTask already consumed");
        signal_safe_block_on(py, task)?
    }

    /// Returns an iterator of just
    fn __await__(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let lp = py
            .import("asyncio.events")
            .unwrap()
            .call_method0("_get_running_loop")
            .unwrap();
        if lp.is_none() {
            let value = self.block_on(py)?;
            Ok(JustStopWithValueIterator { value: Some(value) }.into_py_any(py)?)
        } else {
            self.into_future(py)?.call_method0(py, "__await__")
        }
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyPythonTask>()?;
    Ok(())
}
