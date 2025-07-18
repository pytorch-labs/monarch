/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use hyperactor::ActorRef;
use hyperactor::id;
use hyperactor::mailbox::OncePortReceiver;
use hyperactor::mailbox::PortReceiver;
use hyperactor::supervision::ActorSupervisionEvent;
use hyperactor_mesh::Mesh;
use hyperactor_mesh::RootActorMesh;
use hyperactor_mesh::actor_mesh::ActorMesh;
use hyperactor_mesh::actor_mesh::ActorSupervisionEvents;
use hyperactor_mesh::reference::ActorMeshRef;
use hyperactor_mesh::shared_cell::SharedCell;
use hyperactor_mesh::shared_cell::SharedCellRef;
use pyo3::exceptions::PyEOFError;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PySlice;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::Mutex;

use crate::actor::PythonActor;
use crate::actor::PythonMessage;
use crate::mailbox::PyMailbox;
use crate::mailbox::PythonOncePortReceiver;
use crate::mailbox::PythonPortReceiver;
use crate::proc::PyActorId;
use crate::proc_mesh::Keepalive;
use crate::runtime::signal_safe_block_on;
use crate::selection::PySelection;
use crate::shape::PyShape;
use crate::supervision::SupervisionError;

#[pyclass(
    name = "PythonActorMesh",
    module = "monarch._rust_bindings.monarch_hyperactor.actor_mesh"
)]
pub struct PythonActorMesh {
    inner: SharedCell<RootActorMesh<'static, PythonActor>>,
    client: PyMailbox,
    _keepalive: Keepalive,
    unhealthy_event: Arc<std::sync::Mutex<Option<Option<ActorSupervisionEvent>>>>,
    user_monitor_sender: tokio::sync::broadcast::Sender<Option<ActorSupervisionEvent>>,
    monitor: tokio::task::JoinHandle<()>,
}

impl PythonActorMesh {
    /// Create a new [`PythonActorMesh`] with a monitor that will observe supervision
    /// errors for this mesh, and update its state properly.
    pub(crate) fn monitored(
        inner: SharedCell<RootActorMesh<'static, PythonActor>>,
        client: PyMailbox,
        keepalive: Keepalive,
        events: ActorSupervisionEvents,
    ) -> Self {
        let (user_monitor_sender, _) =
            tokio::sync::broadcast::channel::<Option<ActorSupervisionEvent>>(1);
        let unhealthy_event = Arc::new(std::sync::Mutex::new(None));
        let monitor = tokio::spawn(Self::actor_mesh_monitor(
            events,
            user_monitor_sender.clone(),
            unhealthy_event.clone(),
        ));
        Self {
            inner,
            client,
            _keepalive: keepalive,
            unhealthy_event,
            user_monitor_sender,
            monitor,
        }
    }

    /// Monitor of the actor mesh. It processes supervision errors for the mesh, and keeps mesh
    /// health state up to date.
    async fn actor_mesh_monitor(
        mut events: ActorSupervisionEvents,
        user_sender: tokio::sync::broadcast::Sender<Option<ActorSupervisionEvent>>,
        unhealthy_event: Arc<std::sync::Mutex<Option<Option<ActorSupervisionEvent>>>>,
    ) {
        loop {
            let event = events.next().await;
            let mut inner_unhealthy_event = unhealthy_event.lock().unwrap();
            *inner_unhealthy_event = Some(event.clone());

            // Ignore the sender error when there is no receiver, which happens when there
            // is no active requests to this mesh.
            let _ = user_sender.send(event.clone());

            if event.is_none() {
                // The mesh is stopped, so we can stop the monitor.
                break;
            }
        }
    }

    fn try_inner(&self) -> PyResult<SharedCellRef<RootActorMesh<'static, PythonActor>>> {
        self.inner
            .borrow()
            .map_err(|_| PyRuntimeError::new_err("`PythonActorMesh` has already been stopped"))
    }

    fn pickling_err(&self) -> PyErr {
        PyErr::new::<PyNotImplementedError, _>(
            "PythonActorMesh cannot be pickled. If applicable, use bind() \
            to get a PythonActorMeshRef, and use that instead."
                .to_string(),
        )
    }
}

#[pymethods]
impl PythonActorMesh {
    fn cast(&self, selection: &PySelection, message: &PythonMessage) -> PyResult<()> {
        let unhealthy_event = self
            .unhealthy_event
            .lock()
            .expect("failed to acquire unhealthy_event lock");
        if let Some(ref event) = *unhealthy_event {
            return Err(PyRuntimeError::new_err(format!(
                "actor mesh is unhealthy with reason: {:?}",
                event
            )));
        }

        self.try_inner()?
            .cast(selection.inner().clone(), message.clone())
            .map_err(|err| PyException::new_err(err.to_string()))?;
        Ok(())
    }

    fn bind(&self) -> PyResult<PythonActorMeshRef> {
        let mesh = self.try_inner()?;
        Ok(PythonActorMeshRef { inner: mesh.bind() })
    }

    fn get_supervision_event(&self) -> PyResult<Option<PyActorSupervisionEvent>> {
        let unhealthy_event = self
            .unhealthy_event
            .lock()
            .expect("failed to acquire unhealthy_event lock");

        Ok(unhealthy_event.as_ref().map(|event| match event {
            None => PyActorSupervisionEvent {
                // Dummy actor as place holder to indicate the whole mesh is stopped
                // TODO(albertli): remove this when pushing all supervision logic to rust.
                actor_id: id!(default[0].actor[0]).into(),
                actor_status: "actor mesh is stopped due to proc mesh shutdown".to_string(),
            },
            Some(event) => PyActorSupervisionEvent::from(event.clone()),
        }))
    }

    // Consider defining a "PythonActorRef", which carries specifically
    // a reference to python message actors.
    fn get(&self, rank: usize) -> PyResult<Option<PyActorId>> {
        Ok(self
            .try_inner()?
            .get(rank)
            .map(ActorRef::into_actor_id)
            .map(PyActorId::from))
    }

    // Start monitoring the actor mesh by subscribing to its supervision events. For each supervision
    // event, it is consumed by PythonActorMesh first, then gets sent to the monitor for user to consume.
    fn monitor<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let receiver = self.user_monitor_sender.subscribe();
        let monitor_instance = PyActorMeshMonitor {
            receiver: SharedCell::from(Mutex::new(receiver)),
        };
        Ok(monitor_instance.into_py(py))
    }

    #[pyo3(signature = (**kwargs))]
    fn slice(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PythonActorMeshRef> {
        self.bind()?.slice(kwargs)
    }

    #[getter]
    pub fn client(&self) -> PyMailbox {
        self.client.clone()
    }

    #[getter]
    fn shape(&self) -> PyResult<PyShape> {
        Ok(PyShape::from(self.try_inner()?.shape().clone()))
    }

    // Override the pickling methods to provide a meaningful error message.
    fn __reduce__(&self) -> PyResult<()> {
        Err(self.pickling_err())
    }

    fn __reduce_ex__(&self, _proto: u8) -> PyResult<()> {
        Err(self.pickling_err())
    }
}

#[pyclass(
    frozen,
    name = "PythonActorMeshRef",
    module = "monarch._rust_bindings.monarch_hyperactor.actor_mesh"
)]
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct PythonActorMeshRef {
    inner: ActorMeshRef<PythonActor>,
}

#[pymethods]
impl PythonActorMeshRef {
    fn cast(
        &self,
        client: &PyMailbox,
        selection: &PySelection,
        message: &PythonMessage,
    ) -> PyResult<()> {
        self.inner
            .cast(&client.inner, selection.inner().clone(), message.clone())
            .map_err(|err| PyException::new_err(err.to_string()))?;
        Ok(())
    }

    #[pyo3(signature = (**kwargs))]
    fn slice(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        // When the input type is `int`, convert it into `ndslice::Range`.
        fn convert_int(index: isize) -> PyResult<ndslice::Range> {
            if index < 0 {
                return Err(PyException::new_err(format!(
                    "does not support negative index in selection: {}",
                    index
                )));
            }
            Ok(ndslice::Range::from(index as usize))
        }

        // When the input type is `slice`, convert it into `ndslice::Range`.
        fn convert_py_slice<'py>(s: &Bound<'py, PySlice>) -> PyResult<ndslice::Range> {
            fn get_attr<'py>(s: &Bound<'py, PySlice>, attr: &str) -> PyResult<Option<isize>> {
                let v = s.getattr(attr)?.extract::<Option<isize>>()?;
                if v.is_some() && v.unwrap() < 0 {
                    return Err(PyException::new_err(format!(
                        "does not support negative {} in slice: {}",
                        attr,
                        v.unwrap(),
                    )));
                }
                Ok(v)
            }

            let start = get_attr(s, "start")?.unwrap_or(0);
            let stop: Option<isize> = get_attr(s, "stop")?;
            let step = get_attr(s, "step")?.unwrap_or(1);
            Ok(ndslice::Range(
                start as usize,
                stop.map(|s| s as usize),
                step as usize,
            ))
        }

        if kwargs.is_none() || kwargs.unwrap().is_empty() {
            return Err(PyException::new_err("selection cannot be empty"));
        }

        let mut sliced = self.inner.clone();

        for entry in kwargs.unwrap().items() {
            let label = entry.get_item(0)?.str()?;
            let label_str = label.to_str()?;

            let value = entry.get_item(1)?;

            let range = if let Ok(index) = value.extract::<isize>() {
                convert_int(index)?
            } else if let Ok(s) = value.downcast::<PySlice>() {
                convert_py_slice(s)?
            } else {
                return Err(PyException::new_err(
                    "selection only supports type int or slice",
                ));
            };
            sliced = sliced.select(label_str, range).map_err(|err| {
                PyException::new_err(format!(
                    "failed to select label {}; error is: {}",
                    label_str, err
                ))
            })?;
        }

        Ok(Self { inner: sliced })
    }

    #[getter]
    fn shape(&self) -> PyShape {
        PyShape::from(self.inner.shape().clone())
    }

    #[staticmethod]
    fn from_bytes(bytes: &Bound<'_, PyBytes>) -> PyResult<Self> {
        bincode::deserialize(bytes.as_bytes())
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let bytes = bincode::serialize(&*slf.borrow())
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;
        let py_bytes = PyBytes::new(slf.py(), &bytes);
        Ok((slf.as_any().getattr("from_bytes")?, (py_bytes,)))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl Drop for PythonActorMesh {
    fn drop(&mut self) {
        tracing::info!(
            "Dropping PythonActorMesh: {}",
            self.inner.borrow().unwrap().name()
        );
        self.monitor.abort();
    }
}

#[pyclass(
    name = "ActorMeshMonitor",
    module = "monarch._rust_bindings.monarch_hyperactor.actor_mesh"
)]
pub struct PyActorMeshMonitor {
    receiver: SharedCell<Mutex<tokio::sync::broadcast::Receiver<Option<ActorSupervisionEvent>>>>,
}

#[pymethods]
impl PyActorMeshMonitor {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __anext__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let receiver = self.receiver.clone();
        Ok(pyo3_async_runtimes::tokio::future_into_py(py, get_next(receiver))?.into())
    }
}

impl PyActorMeshMonitor {
    pub async fn next(&self) -> PyResult<PyObject> {
        get_next(self.receiver.clone()).await
    }
}

impl Clone for PyActorMeshMonitor {
    fn clone(&self) -> Self {
        Self {
            receiver: self.receiver.clone(),
        }
    }
}

async fn get_next(
    receiver: SharedCell<Mutex<tokio::sync::broadcast::Receiver<Option<ActorSupervisionEvent>>>>,
) -> PyResult<PyObject> {
    let receiver = receiver.clone();

    let receiver = receiver
        .borrow()
        .expect("`Actor mesh receiver` is shutdown");
    let mut receiver = receiver.lock().await;
    let event = receiver.recv().await.unwrap();

    let supervision_event = match event {
        None => PyActorSupervisionEvent {
            // Dummy actor as place holder to indicate the whole mesh is stopped
            // TODO(albertli): remove this when pushing all supervision logic to rust.
            actor_id: id!(default[0].actor[0]).into(),
            actor_status: "actor mesh is stopped due to proc mesh shutdown".to_string(),
        },
        Some(event) => PyActorSupervisionEvent::from(event.clone()),
    };
    tracing::info!("recv supervision event: {supervision_event:?}");

    Ok(Python::with_gil(|py| supervision_event.into_py(py)))
}

// TODO(albertli): this is temporary remove this when pushing all supervision logic to rust.
#[pyclass(
    name = "MonitoredPortReceiver",
    module = "monarch._rust_bindings.monarch_hyperactor.actor_mesh"
)]
pub(super) struct MonitoredPythonPortReceiver {
    inner: Arc<tokio::sync::Mutex<PortReceiver<PythonMessage>>>,
    monitor: PyActorMeshMonitor,
}

#[pymethods]
impl MonitoredPythonPortReceiver {
    #[new]
    fn new(receiver: &PythonPortReceiver, monitor: &PyActorMeshMonitor) -> Self {
        let inner = receiver.inner();
        MonitoredPythonPortReceiver {
            inner,
            monitor: monitor.clone(),
        }
    }

    fn recv<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver = self.inner.clone();
        let monitor = self.monitor.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut receiver = receiver.lock().await;
            tokio::select! {
                result = receiver.recv() => {
                    result.map_err(|err| PyErr::new::<PyEOFError, _>(format!("port closed: {}", err)))
                }
                event = monitor.next() => {
                    Err(PyErr::new::<SupervisionError, _>(format!("supervision error: {:?}", event.unwrap())))
                }
            }
        })
    }

    fn blocking_recv<'py>(&mut self, py: Python<'py>) -> PyResult<PythonMessage> {
        let receiver = self.inner.clone();
        let monitor = self.monitor.clone();
        signal_safe_block_on(py, async move {
            let mut receiver = receiver.lock().await;
            tokio::select! {
                result = receiver.recv() => {
                   result.map_err(|err| PyErr::new::<PyEOFError, _>(format!("port closed: {}", err)))
                }
                event = monitor.next() => {
                    Err(PyErr::new::<SupervisionError, _>(format!("supervision error: {:?}", event.unwrap())))
                }
            }
        })?
    }
}

#[pyclass(
    name = "MonitoredOncePortReceiver",
    module = "monarch._rust_bindings.monarch_hyperactor.actor_mesh"
)]
pub(super) struct MonitoredPythonOncePortReceiver {
    inner: Arc<std::sync::Mutex<Option<OncePortReceiver<PythonMessage>>>>,
    monitor: PyActorMeshMonitor,
}

#[pymethods]
impl MonitoredPythonOncePortReceiver {
    #[new]
    fn new(receiver: &PythonOncePortReceiver, monitor: &PyActorMeshMonitor) -> Self {
        let inner = receiver.inner();
        MonitoredPythonOncePortReceiver {
            inner,
            monitor: monitor.clone(),
        }
    }

    fn recv<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(receiver) = self.inner.lock().unwrap().take() else {
            return Err(PyErr::new::<PyValueError, _>("OncePort is already used"));
        };
        let monitor = self.monitor.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::select! {
                result = receiver.recv() => {
                    result.map_err(|err| PyErr::new::<PyEOFError, _>(format!("port closed: {}", err)))
                }
                event = monitor.next() => {
                    Err(PyErr::new::<SupervisionError, _>(format!("supervision error: {:?}", event.unwrap())))
                }
            }
        })
    }

    fn blocking_recv<'py>(&mut self, py: Python<'py>) -> PyResult<PythonMessage> {
        let Some(receiver) = self.inner.lock().unwrap().take() else {
            return Err(PyErr::new::<PyValueError, _>("OncePort is already used"));
        };
        let monitor = self.monitor.clone();
        signal_safe_block_on(py, async move {
            tokio::select! {
                result = receiver.recv() => {
                   result.map_err(|err| PyErr::new::<PyEOFError, _>(format!("port closed: {}", err)))
                }
                event = monitor.next() => {
                    Err(PyErr::new::<SupervisionError, _>(format!("supervision error: {:?}", event.unwrap())))
                }
            }
        })?
    }
}

#[pyclass(
    name = "ActorSupervisionEvent",
    module = "monarch._rust_bindings.monarch_hyperactor.actor_mesh"
)]
#[derive(Debug)]
pub struct PyActorSupervisionEvent {
    /// Actor ID of the actor where supervision event originates from.
    #[pyo3(get)]
    actor_id: PyActorId,
    /// String representation of the actor status.
    /// TODO(T230628951): make it an enum or a struct for easier consumption.
    #[pyo3(get)]
    actor_status: String,
}

#[pymethods]
impl PyActorSupervisionEvent {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "<PyActorSupervisionEvent: actor_id: {:?}, status: {}>",
            self.actor_id, self.actor_status
        ))
    }
}

impl From<ActorSupervisionEvent> for PyActorSupervisionEvent {
    fn from(event: ActorSupervisionEvent) -> Self {
        PyActorSupervisionEvent {
            actor_id: event.actor_id().clone().into(),
            actor_status: event.actor_status().to_string(),
        }
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PythonActorMesh>()?;
    hyperactor_mod.add_class::<PythonActorMeshRef>()?;
    hyperactor_mod.add_class::<PyActorMeshMonitor>()?;
    hyperactor_mod.add_class::<MonitoredPythonPortReceiver>()?;
    hyperactor_mod.add_class::<MonitoredPythonOncePortReceiver>()?;
    hyperactor_mod.add_class::<PyActorSupervisionEvent>()?;
    Ok(())
}
