/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use hyperactor::WorldId;
use hyperactor_extension::alloc::PyAlloc;
use hyperactor_mesh::alloc::Alloc;
use hyperactor_mesh::alloc::ProcStopReason;
use hyperactor_mesh::proc_mesh::ProcEvent;
use hyperactor_mesh::proc_mesh::ProcEvents;
use hyperactor_mesh::proc_mesh::ProcMesh;
use hyperactor_mesh::proc_mesh::SharedSpawnable;
use monarch_types::PickledPyObject;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::pycell::PyRef;
use pyo3::types::PyType;
use tokio::sync::Mutex;
use tokio::sync::mpsc;

use crate::actor_mesh::PythonActorMesh;
use crate::mailbox::PyMailbox;
use crate::runtime::signal_safe_block_on;
use crate::shape::PyShape;

#[pyclass(
    name = "ProcMesh",
    module = "monarch._rust_bindings.monarch_hyperactor.proc_mesh"
)]
pub struct PyProcMesh {
    pub inner: Arc<ProcMesh>,
    keepalive: Keepalive,
    proc_events: Arc<Mutex<ProcEvents>>,
    stop_monitor_sender: mpsc::Sender<bool>,
    user_monitor_registered: AtomicBool,
}

fn allocate_proc_mesh<'py>(py: Python<'py>, alloc: &PyAlloc) -> PyResult<Bound<'py, PyAny>> {
    let alloc = match alloc.take() {
        Some(alloc) => alloc,
        None => {
            return Err(PyException::new_err(
                "Alloc object already been used".to_string(),
            ));
        }
    };
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let world_id = alloc.world_id().clone();
        let mesh = ProcMesh::allocate(alloc)
            .await
            .map_err(|err| PyException::new_err(err.to_string()))?;
        Ok(PyProcMesh::monitored(mesh, world_id))
    })
}

fn allocate_proc_mesh_blocking<'py>(py: Python<'py>, alloc: &PyAlloc) -> PyResult<PyProcMesh> {
    let alloc = match alloc.take() {
        Some(alloc) => alloc,
        None => {
            return Err(PyException::new_err(
                "Alloc object already been used".to_string(),
            ));
        }
    };
    signal_safe_block_on(py, async move {
        let world_id = alloc.world_id().clone();
        let mesh = ProcMesh::allocate(alloc)
            .await
            .map_err(|err| PyException::new_err(err.to_string()))?;
        Ok(PyProcMesh::monitored(mesh, world_id))
    })?
}

impl PyProcMesh {
    /// Create a new [`PyProcMesh`] with a monitor that crashes the
    /// process on any proc failure.
    fn monitored(mut proc_mesh: ProcMesh, world_id: WorldId) -> Self {
        let (sender, abort_receiver) = mpsc::channel::<bool>(1);
        let proc_events = Arc::new(Mutex::new(proc_mesh.events().unwrap()));
        let monitor = tokio::spawn(Self::default_proc_mesh_monitor(
            proc_events.clone(),
            world_id,
            abort_receiver,
        ));
        Self {
            inner: Arc::new(proc_mesh),
            keepalive: Keepalive::new(monitor),
            proc_events,
            stop_monitor_sender: sender,
            user_monitor_registered: AtomicBool::new(false),
        }
    }

    /// The default monitor of the proc mesh for crashes. If a proc crashes, we print the reason
    /// to stderr and exit with code 1.
    async fn default_proc_mesh_monitor(
        events: Arc<Mutex<ProcEvents>>,
        world_id: WorldId,
        mut abort_receiver: mpsc::Receiver<bool>,
    ) {
        let mut proc_events = events.lock().await;
        loop {
            tokio::select! {
                event = proc_events.next() => {
                    if let Some(event) = event {
                        match event {
                            // A graceful stop should not be cause for alarm, but
                            // everything else should be considered a crash.
                            ProcEvent::Stopped(_, ProcStopReason::Stopped) => continue,
                            event => {
                                eprintln!("ProcMesh {}: {}", world_id, event);
                                std::process::exit(1)
                            }
                        }
                    }
                }
                _ = abort_receiver.recv() => {
                    // The default monitor is aborted, this happens when user takes over
                    // the monitoring responsibility.
                    eprintln!("stop default supervision monitor for ProcMesh {}", world_id);
                    break;
                }
            }
        }
    }
}

#[pymethods]
impl PyProcMesh {
    #[classmethod]
    fn allocate_nonblocking<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        alloc: &PyAlloc,
    ) -> PyResult<Bound<'py, PyAny>> {
        allocate_proc_mesh(py, alloc)
    }

    #[classmethod]
    fn allocate_blocking<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        alloc: &PyAlloc,
    ) -> PyResult<PyProcMesh> {
        allocate_proc_mesh_blocking(py, alloc)
    }

    fn spawn_nonblocking<'py>(
        &self,
        py: Python<'py>,
        name: String,
        actor: &Bound<'py, PyType>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pickled_type = PickledPyObject::pickle(actor.as_any())?;
        let proc_mesh = Arc::clone(&self.inner);
        let keepalive = self.keepalive.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let actor_mesh = proc_mesh.spawn(&name, &pickled_type).await?;
            let python_actor_mesh = PythonActorMesh {
                inner: Arc::new(actor_mesh),
                client: PyMailbox {
                    inner: proc_mesh.client().clone(),
                },
                _keepalive: keepalive,
            };
            Python::with_gil(|py| python_actor_mesh.into_py_any(py))
        })
    }

    fn spawn_blocking<'py>(
        &self,
        py: Python<'py>,
        name: String,
        actor: &Bound<'py, PyType>,
    ) -> PyResult<PyObject> {
        let pickled_type = PickledPyObject::pickle(actor.as_any())?;
        let proc_mesh = Arc::clone(&self.inner);
        let keepalive = self.keepalive.clone();
        signal_safe_block_on(py, async move {
            let actor_mesh = proc_mesh.spawn(&name, &pickled_type).await?;
            let python_actor_mesh = PythonActorMesh {
                inner: Arc::new(actor_mesh),
                client: PyMailbox {
                    inner: proc_mesh.client().clone(),
                },
                _keepalive: keepalive,
            };
            Python::with_gil(|py| python_actor_mesh.into_py_any(py))
        })?
    }

    // User can call this to monitor the proc mesh events. This will override
    // the default monitor that exits the client on process crash, so user can
    // handle the process crash in their own way.
    fn monitor<'py>(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        if self
            .user_monitor_registered
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return Err(PyException::new_err(
                "user already registered a monitor for this proc mesh".to_string(),
            ));
        }

        // Stop the default monitor
        let monitor_abort = self.stop_monitor_sender.clone();
        let proc_events = self.proc_events.clone();

        Ok(pyo3_async_runtimes::tokio::future_into_py(py, async move {
            monitor_abort.send(true).await.unwrap();

            // Create a new user monitor
            Ok(PyProcMeshMonitor { proc_events })
        })?
        .into())
    }

    #[getter]
    fn client(&self) -> PyMailbox {
        PyMailbox {
            inner: self.inner.client().clone(),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("<ProcMesh {}>", self.inner))
    }

    #[getter]
    fn shape(&self) -> PyShape {
        self.inner.shape().clone().into()
    }
}

/// A keepalive token that aborts a task only after the last clone
/// of the token is dropped.
#[derive(Clone, Debug)]
pub(crate) struct Keepalive {
    /// The function of this field is to maintain a reference to the
    /// state.
    _state: Arc<KeepaliveState>,
}

impl Keepalive {
    fn new(handle: tokio::task::JoinHandle<()>) -> Self {
        Self {
            _state: Arc::new(KeepaliveState(handle)),
        }
    }
}

#[derive(Debug)]
struct KeepaliveState(tokio::task::JoinHandle<()>);

impl Drop for KeepaliveState {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[pyclass(
    name = "ProcMeshMonitor",
    module = "monarch._rust_bindings.monarch_hyperactor.proc_mesh"
)]
pub struct PyProcMeshMonitor {
    proc_events: Arc<Mutex<ProcEvents>>,
}

#[pymethods]
impl PyProcMeshMonitor {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let events = self.proc_events.clone();
        Ok(pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut proc_events = events.lock().await;
            let event: Option<_> = proc_events.next().await;
            match event {
                Some(event) => Ok(PyProcEvent::from(event)),
                None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(
                    "stop iteration",
                )),
            }
        })?
        .into())
    }
}

#[pyclass(
    name = "ProcEvent",
    module = "monarch._rust_bindings.monarch_hyperactor.proc_mesh"
)]
pub enum PyProcEvent {
    /// The proc of the given rank was stopped with the provided reason.
    /// The arguments represent the rank id and stop reason.
    #[pyo3(name = "Stopped")]
    Stopped(usize, String),
    /// The proc crashed, with the provided "reason". This is reserved for
    /// unhandled supervision events.
    /// The arguments represent the rank id and crash reason.
    #[pyo3(name = "Crashed")]
    Crashed(usize, String),
}

impl From<ProcEvent> for PyProcEvent {
    fn from(event: ProcEvent) -> Self {
        match event {
            ProcEvent::Stopped(pid, reason) => PyProcEvent::Stopped(pid, reason.to_string()),
            ProcEvent::Crashed(pid, reason) => PyProcEvent::Crashed(pid, reason),
        }
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyProcMesh>()?;
    hyperactor_mod.add_class::<PyProcMeshMonitor>()?;
    hyperactor_mod.add_class::<PyProcEvent>()?;
    Ok(())
}
