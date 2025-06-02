/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use hyperactor::Mailbox;
use hyperactor::Named;
use hyperactor::OncePortHandle;
use hyperactor::PortHandle;
use hyperactor::PortId;
use hyperactor::data::Serialized;
use hyperactor::mailbox::MailboxSender;
use hyperactor::mailbox::MessageEnvelope;
use hyperactor::mailbox::OncePortReceiver;
use hyperactor::mailbox::PortReceiver;
use hyperactor::mailbox::monitored_return_handle;
use hyperactor_mesh::actor_mesh::Cast;
use hyperactor_mesh::comm::multicast::CastRank;
use pyo3::exceptions::PyEOFError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::actor::PythonMessage;
use crate::proc::PyActorId;
use crate::runtime::signal_safe_block_on;
use crate::shape::PyShape;
#[derive(Clone, Debug)]
#[pyclass(
    name = "Mailbox",
    module = "monarch._rust_bindings.monarch_hyperactor.mailbox"
)]
pub struct PyMailbox {
    pub inner: Mailbox,
}

#[pymethods]
impl PyMailbox {
    fn open_port<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let (handle, receiver) = self.inner.open_port();
        let handle = Py::new(py, PythonPortHandle { inner: handle })?;
        let receiver = Py::new(
            py,
            PythonPortReceiver {
                inner: Arc::new(tokio::sync::Mutex::new(receiver)),
            },
        )?;
        Ok(PyTuple::new_bound(
            py,
            vec![handle.into_any(), receiver.into_any()],
        ))
    }

    fn open_once_port<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let (handle, receiver) = self.inner.open_once_port();
        let handle = Py::new(
            py,
            PythonOncePortHandle {
                inner: Some(handle),
            },
        )?;
        let receiver = Py::new(
            py,
            PythonOncePortReceiver {
                inner: std::sync::Mutex::new(Some(receiver)),
            },
        )?;
        Ok(PyTuple::new_bound(
            py,
            vec![handle.into_any(), receiver.into_any()],
        ))
    }

    pub(super) fn post<'py>(
        &self,
        py: Python<'py>,
        dest: PyObject,
        message: &PythonMessage,
    ) -> PyResult<()> {
        let port_id = if let Ok(actor_id) = dest.extract::<PyActorId>(py) {
            // Messages to an actor gets sent to the message port for PythonMessage.
            actor_id.inner.port_id(PythonMessage::port())
        } else if let Ok(port_id) = dest.extract::<PyPortId>(py) {
            port_id.inner.clone()
        } else {
            return Err(PyErr::new::<PyValueError, _>(
                "dest must be either an actor id or a port id",
            ));
        };

        let message = Serialized::serialize(message).map_err(|err| {
            PyRuntimeError::new_err(format!(
                "failed to serialize message ({:?}) to Serialized: {}",
                message, err
            ))
        })?;
        let envelope = MessageEnvelope::new(self.inner.actor_id().clone(), port_id, message);
        self.inner.post(envelope, monitored_return_handle());
        Ok(())
    }

    pub(super) fn post_cast(
        &self,
        dest: PyActorId,
        rank: usize,
        shape: &PyShape,
        message: &PythonMessage,
    ) -> PyResult<()> {
        let port_id = dest.inner.port_id(Cast::<PythonMessage>::port());
        let message = Cast {
            rank: CastRank(rank),
            shape: shape.inner.clone(),
            message: message.clone(),
        };
        let message = Serialized::serialize(&message).map_err(|err| {
            PyRuntimeError::new_err(format!(
                "failed to serialize message ({:?}) to Serialized: {}",
                message, err
            ))
        })?;
        let envelope = MessageEnvelope::new(self.inner.actor_id().clone(), port_id, message);
        self.inner.post(envelope, monitored_return_handle());
        Ok(())
    }

    #[getter]
    fn actor_id(&self) -> PyActorId {
        PyActorId {
            inner: self.inner.actor_id().clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(
    frozen,
    name = "PortId",
    module = "monarch._rust_bindings.monarch_hyperactor.mailbox"
)]
#[derive(Clone)]
pub struct PyPortId {
    inner: PortId,
}

impl From<PortId> for PyPortId {
    fn from(port_id: PortId) -> Self {
        Self { inner: port_id }
    }
}

#[pymethods]
impl PyPortId {
    #[new]
    #[pyo3(signature = (*, actor_id, port))]
    fn new(actor_id: &PyActorId, port: u64) -> Self {
        Self {
            inner: PortId(actor_id.inner.clone(), port),
        }
    }

    #[staticmethod]
    fn from_string(port_id: &str) -> PyResult<Self> {
        Ok(Self {
            inner: port_id.parse().map_err(|e| {
                PyValueError::new_err(format!("Failed to parse port id '{}': {}", port_id, e))
            })?,
        })
    }

    #[getter]
    fn actor_id(&self) -> PyActorId {
        PyActorId {
            inner: self.inner.actor_id().clone(),
        }
    }

    #[getter]
    fn index(&self) -> u64 {
        self.inner.index()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.to_string().hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.extract::<PyPortId>() {
            Ok(self.inner == other.inner)
        } else {
            Ok(false)
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        Ok((slf.getattr("from_string")?, (slf.borrow().__str__(),)))
    }
}

impl std::fmt::Debug for PyPortId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

#[derive(Clone, Debug)]
#[pyclass(
    name = "PortHandle",
    module = "monarch._rust_bindings.monarch_hyperactor.mailbox"
)]
pub(super) struct PythonPortHandle {
    inner: PortHandle<PythonMessage>,
}

#[pymethods]
impl PythonPortHandle {
    fn send(&self, message: PythonMessage) -> PyResult<()> {
        self.inner
            .send(message)
            .map_err(|err| PyErr::new::<PyEOFError, _>(format!("Port closed: {}", err)))?;
        Ok(())
    }

    // TODO: We probably should have a specific "PythonMessagePortRef" type here instead.
    fn bind(&self) -> PyPortId {
        PyPortId {
            inner: self.inner.bind().into_port_id(),
        }
    }
}

#[derive(Debug)]
#[pyclass(
    name = "PortReceiver",
    module = "monarch._rust_bindings.monarch_hyperactor.mailbox"
)]
pub(super) struct PythonPortReceiver {
    inner: Arc<tokio::sync::Mutex<PortReceiver<PythonMessage>>>,
}

#[pymethods]
impl PythonPortReceiver {
    fn recv<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            receiver
                .lock()
                .await
                .recv()
                .await
                .map_err(|err| PyErr::new::<PyEOFError, _>(format!("Port closed: {}", err)))
        })
    }
    fn blocking_recv<'py>(&mut self, py: Python<'py>) -> PyResult<PythonMessage> {
        let receiver = self.inner.clone();
        signal_safe_block_on(py, async move { receiver.lock().await.recv().await })?
            .map_err(|err| PyErr::new::<PyEOFError, _>(format!("Port closed: {}", err)))
    }
}

#[derive(Debug)]
#[pyclass(
    name = "OncePortHandle",
    module = "monarch._rust_bindings.monarch_hyperactor.mailbox"
)]
pub(super) struct PythonOncePortHandle {
    inner: Option<OncePortHandle<PythonMessage>>,
}

#[pymethods]
impl PythonOncePortHandle {
    fn send(&mut self, message: PythonMessage) -> PyResult<()> {
        let Some(port) = self.inner.take() else {
            return Err(PyErr::new::<PyValueError, _>("OncePort is already used"));
        };

        port.send(message)
            .map_err(|err| PyErr::new::<PyEOFError, _>(format!("Port closed: {}", err)))?;
        Ok(())
    }

    // TODO: We probably should have a specific "PythonMessagePortRef" type here instead.
    fn bind(&mut self) -> PyResult<PyPortId> {
        let Some(port) = self.inner.take() else {
            return Err(PyErr::new::<PyValueError, _>("OncePort is already used"));
        };
        Ok(PyPortId {
            inner: port.bind().into_port_id(),
        })
    }
}

#[pyclass(
    name = "OncePortReceiver",
    module = "monarch._rust_bindings.monarch_hyperactor.mailbox"
)]
pub(super) struct PythonOncePortReceiver {
    inner: std::sync::Mutex<Option<OncePortReceiver<PythonMessage>>>,
}

#[pymethods]
impl PythonOncePortReceiver {
    fn recv<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(receiver) = self.inner.lock().unwrap().take() else {
            return Err(PyErr::new::<PyValueError, _>("OncePort is already used"));
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            receiver
                .recv()
                .await
                .map_err(|err| PyErr::new::<PyEOFError, _>(format!("Port closed: {}", err)))
        })
    }
    fn blocking_recv<'py>(&mut self, py: Python<'py>) -> PyResult<PythonMessage> {
        let Some(receiver) = self.inner.lock().unwrap().take() else {
            return Err(PyErr::new::<PyValueError, _>("OncePort is already used"));
        };
        signal_safe_block_on(py, async move { receiver.recv().await })?
            .map_err(|err| PyErr::new::<PyEOFError, _>(format!("Port closed: {}", err)))
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyMailbox>()?;
    hyperactor_mod.add_class::<PyPortId>()?;
    hyperactor_mod.add_class::<PythonPortHandle>()?;
    hyperactor_mod.add_class::<PythonPortReceiver>()?;
    hyperactor_mod.add_class::<PythonOncePortHandle>()?;
    hyperactor_mod.add_class::<PythonOncePortReceiver>()?;

    Ok(())
}
