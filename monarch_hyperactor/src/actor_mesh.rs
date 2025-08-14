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
use hyperactor::supervision::ActorSupervisionEvent;
use hyperactor_mesh::Mesh;
use hyperactor_mesh::RootActorMesh;
use hyperactor_mesh::actor_mesh::ActorMesh;
use hyperactor_mesh::actor_mesh::ActorSupervisionEvents;
use hyperactor_mesh::proc_mesh::ActorEventRouter;
use hyperactor_mesh::reference::ActorMeshRef;
use hyperactor_mesh::shared_cell::SharedCell;
use hyperactor_mesh::shared_cell::SharedCellRef;
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

use crate::actor::PythonActor;
use crate::actor::PythonMessage;
use crate::mailbox::PyMailbox;
use crate::proc::PyActorId;
use crate::proc_mesh::Keepalive;
use crate::pytokio::PyPythonTask;
use crate::runtime::get_tokio_runtime;
use crate::selection::PySelection;
use crate::shape::PyShape;
use crate::supervision::SupervisionError;
use crate::supervision::Unhealthy;

#[pyclass(
    name = "PythonActorMesh",
    module = "monarch._rust_bindings.monarch_hyperactor.actor_mesh"
)]
pub struct PythonActorMesh {
    inner: SharedCell<RootActorMesh<'static, PythonActor>>,
    client: PyMailbox,
    _keepalive: Keepalive,
    unhealthy_event: Arc<std::sync::Mutex<Unhealthy<ActorSupervisionEvent>>>,
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
        mesh_shape: ndslice::Shape,
    ) -> Self {
        let (user_monitor_sender, _) =
            tokio::sync::broadcast::channel::<Option<ActorSupervisionEvent>>(1);
        let unhealthy_event = Arc::new(std::sync::Mutex::new(Unhealthy::SoFarSoGood));
        let monitor = tokio::spawn(Self::actor_mesh_monitor(
            events,
            user_monitor_sender.clone(),
            Arc::clone(&unhealthy_event),
            mesh_shape,
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
        unhealthy_event: Arc<std::sync::Mutex<Unhealthy<ActorSupervisionEvent>>>,
        mesh_shape: ndslice::Shape,
    ) {
        loop {
            let event = events.next().await;
            tracing::debug!("actor_mesh_monitor received supervision event: {event:?}");
            let mut inner_unhealthy_event = unhealthy_event.lock().unwrap();
            match &event {
                None => *inner_unhealthy_event = Unhealthy::StreamClosed,
                Some(event) => {
                    // Ignore if the crashed actor is not a part of the mesh.
                    if mesh_shape
                        .slice()
                        .iter()
                        .any(|index| index == event.actor_id.rank())
                    {
                        *inner_unhealthy_event = Unhealthy::Crashed(event.clone())
                    }
                }
            }

            // Ignore the sender error when there is no receiver,
            // which happens when there is no active requests to this
            // mesh.
            let ret = user_sender.send(event.clone());
            tracing::debug!("actor_mesh_monitor user_sender send: {ret:?}");

            if event.is_none() {
                // The mesh is stopped, so we can stop the monitor.
                break;
            }
        }
    }

    fn try_inner(&self) -> PyResult<SharedCellRef<RootActorMesh<'static, PythonActor>>> {
        self.inner.borrow().map_err(|_| {
            SupervisionError::new_err("`PythonActorMesh` has already been stopped".to_string())
        })
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
    fn cast(
        &self,
        mailbox: &PyMailbox,
        selection: &PySelection,
        message: &PythonMessage,
    ) -> PyResult<()> {
        let unhealthy_event = self
            .unhealthy_event
            .lock()
            .expect("failed to acquire unhealthy_event lock");

        match &*unhealthy_event {
            Unhealthy::SoFarSoGood => (),
            Unhealthy::Crashed(event) => {
                return Err(SupervisionError::new_err(format!(
                    "Actor {:?} is unhealthy with reason: {}",
                    event.actor_id, event.actor_status
                )));
            }
            Unhealthy::StreamClosed => {
                return Err(SupervisionError::new_err(
                    "actor mesh is stopped due to proc mesh shutdown".to_string(),
                ));
            }
        }

        self.try_inner()?
            .cast(&mailbox.inner, selection.inner().clone(), message.clone())
            .map_err(|err| PyException::new_err(err.to_string()))?;
        Ok(())
    }

    fn bind(&self) -> PyResult<PythonActorMeshRef> {
        let mesh = self.try_inner()?;
        let unhealthy_event = self.unhealthy_event.clone();
        let actor_event_router = mesh.actor_event_router().clone();
        let monitor = get_tokio_runtime().spawn(Self::actor_mesh_monitor(
            ActorSupervisionEvents::new(
                actor_event_router.bind(mesh.name().to_string()),
                mesh.id(),
            ),
            self.user_monitor_sender.clone(),
            unhealthy_event.clone(),
            mesh.shape().clone(),
        ));
        let mesh_monitor = Some(PythonActorMeshRefMonitor {
            user_monitor_sender: self.user_monitor_sender.clone(),
            monitor,
            unhealthy_event,
            actor_event_router,
        });
        Ok(PythonActorMeshRef {
            inner: mesh.bind(),
            mesh_monitor,
        })
    }

    fn get_supervision_event(&self) -> PyResult<Option<PyActorSupervisionEvent>> {
        let unhealthy_event = self
            .unhealthy_event
            .lock()
            .expect("failed to acquire unhealthy_event lock");

        match &*unhealthy_event {
            Unhealthy::SoFarSoGood => Ok(None),
            Unhealthy::StreamClosed => Ok(Some(PyActorSupervisionEvent {
                // Dummy actor as place holder to indicate the whole mesh is stopped
                // TODO(albertli): remove this when pushing all supervision logic to rust.
                actor_id: id!(default[0].actor[0]).into(),
                actor_status: "actor mesh is stopped due to proc mesh shutdown".to_string(),
            })),
            Unhealthy::Crashed(event) => Ok(Some(PyActorSupervisionEvent::from(event.clone()))),
        }
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
    fn supervision_event(&self) -> PyResult<PyPythonTask> {
        let mut receiver = self.user_monitor_sender.subscribe();
        PyPythonTask::new(async move {
            let event = receiver.recv().await;
            let event = match event {
                Ok(Some(event)) => PyActorSupervisionEvent::from(event.clone()),
                Ok(None) | Err(_) => PyActorSupervisionEvent {
                    // Dummy actor as placeholder to indicate the whole mesh is stopped
                    // TODO(albertli): remove this when pushing all supervision logic to rust.
                    actor_id: id!(default[0].actor[0]).into(),
                    actor_status: "actor mesh is stopped due to proc mesh shutdown".to_string(),
                },
            };
            Ok(PyErr::new::<SupervisionError, _>(format!(
                "Actor {:?} exited because of the following reason: {}",
                event.actor_id, event.actor_status
            )))
        })
    }

    #[pyo3(signature = (**kwargs))]
    fn slice(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PythonActorMeshRef> {
        self.bind()?.slice(kwargs)
    }

    fn new_with_shape(&self, shape: PyShape) -> PyResult<PythonActorMeshRef> {
        self.bind()?.new_with_shape(shape)
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

    fn stop<'py>(&self) -> PyResult<PyPythonTask> {
        let actor_mesh = self.inner.clone();
        PyPythonTask::new(async move {
            let actor_mesh = actor_mesh
                .take()
                .await
                .map_err(|_| PyRuntimeError::new_err("`ActorMesh` has already been stopped"))?;
            actor_mesh.stop().await.map_err(|err| {
                PyException::new_err(format!("Failed to stop actor mesh: {}", err))
            })?;
            Ok(())
        })
    }

    #[getter]
    fn stopped(&self) -> PyResult<bool> {
        Ok(self.inner.borrow().is_err())
    }
}

#[derive(Debug)]
struct PythonActorMeshRefMonitor {
    user_monitor_sender: tokio::sync::broadcast::Sender<Option<ActorSupervisionEvent>>,
    /// background task listening to stream of supervision events
    monitor: tokio::task::JoinHandle<()>,
    /// state updated by monitor
    unhealthy_event: Arc<std::sync::Mutex<Unhealthy<ActorSupervisionEvent>>>,
    actor_event_router: ActorEventRouter,
}

#[pyclass(
    frozen,
    name = "PythonActorMeshRef",
    module = "monarch._rust_bindings.monarch_hyperactor.actor_mesh"
)]
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct PythonActorMeshRef {
    inner: ActorMeshRef<PythonActor>,
    #[serde(skip)]
    /// Monitors the mesh ref if and only if the mesh ref was created locally
    /// We cannot monitor a mesh ref that we have received remotely
    mesh_monitor: Option<PythonActorMeshRefMonitor>,
}

#[pymethods]
impl PythonActorMeshRef {
    fn cast(
        &self,
        client: &PyMailbox,
        selection: &PySelection,
        message: &PythonMessage,
    ) -> PyResult<()> {
        if let Some(e) = self.get_supervision_event()? {
            return Err(SupervisionError::new_err(format!(
                "Actor {:?} is unhealthy with reason: {}",
                e.actor_id, e.actor_status
            )));
        }
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
        let mesh_monitor = match &self.mesh_monitor {
            Some(PythonActorMeshRefMonitor {
                user_monitor_sender,
                actor_event_router,
                ..
            }) => {
                let user_monitor_sender = user_monitor_sender.clone();
                let unhealthy_event = Arc::new(std::sync::Mutex::new(Unhealthy::SoFarSoGood));
                let rx = actor_event_router.bind(self.inner.mesh_id().1.clone());
                let monitor = tokio::spawn(PythonActorMesh::actor_mesh_monitor(
                    ActorSupervisionEvents::new(rx, self.inner.mesh_id().clone()),
                    user_monitor_sender.clone(),
                    unhealthy_event.clone(),
                    sliced.shape().clone(),
                ));
                Some(PythonActorMeshRefMonitor {
                    unhealthy_event,
                    monitor,
                    user_monitor_sender,
                    actor_event_router: actor_event_router.clone(),
                })
            }
            None => None,
        };

        Ok(Self {
            inner: sliced,
            mesh_monitor,
        })
    }

    fn get_supervision_event(&self) -> PyResult<Option<PyActorSupervisionEvent>> {
        match &self.mesh_monitor {
            Some(PythonActorMeshRefMonitor {
                unhealthy_event, ..
            }) => {
                let unhealthy_event = unhealthy_event
                    .lock()
                    .expect("failed to acquire unhealthy_event lock");

                match &*unhealthy_event {
                    Unhealthy::SoFarSoGood => Ok(None),
                    Unhealthy::StreamClosed => Ok(Some(PyActorSupervisionEvent {
                        // Dummy actor as place holder to indicate the whole mesh is stopped
                        // TODO(albertli): remove this when pushing all supervision logic to rust.
                        actor_id: id!(default[0].actor[0]).into(),
                        actor_status: "actor mesh is stopped due to proc mesh shutdown".to_string(),
                    })),
                    Unhealthy::Crashed(event) => {
                        Ok(Some(PyActorSupervisionEvent::from(event.clone())))
                    }
                }
            }
            None => Ok(None),
        }
    }

    fn supervision_event(&self) -> PyResult<Option<PyPythonTask>> {
        match &self.mesh_monitor {
            Some(PythonActorMeshRefMonitor {
                user_monitor_sender,
                ..
            }) => {
                let mut receiver = user_monitor_sender.subscribe();

                Ok(Some(PyPythonTask::new(async move {
                    let event = receiver.recv().await;
                    let event = match event {
                        Ok(Some(event)) => PyActorSupervisionEvent::from(event.clone()),
                        Ok(None) | Err(_) => PyActorSupervisionEvent {
                            // Dummy actor as placeholder to indicate the whole mesh is stopped
                            // TODO(albertli): remove this when pushing all supervision logic to rust.
                            actor_id: id!(default[0].actor[0]).into(),
                            actor_status: "actor mesh is stopped due to proc mesh shutdown"
                                .to_string(),
                        },
                    };
                    Ok(PyErr::new::<SupervisionError, _>(format!(
                        "supervision error: {:?}",
                        event
                    )))
                })?))
            }
            None => Ok(None),
        }
    }

    fn new_with_shape(&self, shape: PyShape) -> PyResult<PythonActorMeshRef> {
        let sliced = self
            .inner
            .new_with_shape(shape.get_inner().clone())
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;

        let mesh_monitor = match &self.mesh_monitor {
            Some(PythonActorMeshRefMonitor {
                user_monitor_sender,
                actor_event_router,
                unhealthy_event,
                ..
            }) => {
                let user_monitor_sender = user_monitor_sender.clone();
                let unhealthy_event = Arc::new(std::sync::Mutex::new(
                    match &*unhealthy_event.lock().unwrap_or_else(|e| e.into_inner()) {
                        Unhealthy::SoFarSoGood => Unhealthy::SoFarSoGood,
                        Unhealthy::Crashed(event) => {
                            if sliced
                                .shape()
                                .slice()
                                .iter()
                                .any(|index| index == event.actor_id.rank())
                            {
                                Unhealthy::Crashed(event.clone())
                            } else {
                                Unhealthy::SoFarSoGood
                            }
                        }
                        Unhealthy::StreamClosed => Unhealthy::StreamClosed,
                    },
                ));
                let monitor = get_tokio_runtime().spawn(PythonActorMesh::actor_mesh_monitor(
                    ActorSupervisionEvents::new(
                        actor_event_router.bind(self.inner.mesh_id().1.clone()),
                        self.inner.mesh_id().clone(),
                    ),
                    user_monitor_sender.clone(),
                    unhealthy_event.clone(),
                    sliced.shape().clone(),
                ));
                Some(PythonActorMeshRefMonitor {
                    unhealthy_event,
                    monitor,
                    user_monitor_sender,
                    actor_event_router: actor_event_router.clone(),
                })
            }
            None => None,
        };

        Ok(Self {
            inner: sliced,
            mesh_monitor,
        })
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
        format!(
            "PythonActorMeshRef {{ inner: {:?}, mesh_monitor: <skipped> }}",
            self.inner
        )
    }
}

impl Drop for PythonActorMeshRef {
    fn drop(&mut self) {
        match &self.mesh_monitor {
            Some(PythonActorMeshRefMonitor { monitor, .. }) => monitor.abort(),
            None => {}
        }
    }
}

impl Drop for PythonActorMesh {
    fn drop(&mut self) {
        if let Ok(mesh) = self.inner.borrow() {
            tracing::debug!("Dropping PythonActorMesh: {}", mesh.name());
        } else {
            tracing::debug!(
                "Dropping stopped PythonActorMesh. The underlying mesh is already stopped."
            );
        }
        self.monitor.abort();
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
            actor_id: event.actor_id.clone().into(),
            actor_status: event.actor_status.to_string(),
        }
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PythonActorMesh>()?;
    hyperactor_mod.add_class::<PythonActorMeshRef>()?;
    hyperactor_mod.add_class::<PyActorSupervisionEvent>()?;
    Ok(())
}
