/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::error::Error;
use std::future::Future;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::ActorId;
use hyperactor::Context;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::OncePortHandle;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::Unbind;
use hyperactor_mesh::comm::multicast::CastInfo;
use monarch_types::PickledPyObject;
use monarch_types::SerializablePyErr;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyBaseException;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyStopIteration;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyType;
use serde::Deserialize;
use serde::Serialize;
use serde_bytes::ByteBuf;
use tokio::sync::Mutex;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;
use tracing::Instrument;

use crate::config::SHARED_ASYNCIO_RUNTIME;
use crate::local_state_broker::BrokerId;
use crate::local_state_broker::LocalStateBrokerMessage;
use crate::mailbox::EitherPortRef;
use crate::mailbox::PyMailbox;
use crate::proc::InstanceWrapper;
use crate::proc::PyActorId;
use crate::proc::PyProc;
use crate::proc::PySerialized;
use crate::runtime::signal_safe_block_on;
use crate::shape::PyShape;

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
    fn into_future(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let task = self
            .inner
            .take()
            .map(|task| task.take())
            .expect("PythonTask already consumed");
        Ok(pyo3_async_runtimes::tokio::future_into_py(py, task)?.unbind())
    }
    fn block_on(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let task = self
            .inner
            .take()
            .map(|task| task.take())
            .expect("PythonTask already consumed");
        signal_safe_block_on(py, task)?
    }

    /// In an async context this turns the tokio::Future into
    /// an asyncio Future and awaits it.
    /// In a synchronous context, this just blocks on the future and
    /// immediately returns the value without pausing caller coroutine.
    /// See [avoiding async code duplication] for justitifcation.
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
