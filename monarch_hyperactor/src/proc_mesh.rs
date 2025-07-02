/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use hyperactor::Actor;
use hyperactor::Mailbox;
use hyperactor::RemoteMessage;
use hyperactor::WorldId;
use hyperactor::actor::RemoteActor;
use hyperactor::proc::Proc;
use hyperactor_extension::alloc::PyAlloc;
use hyperactor_mesh::RootActorMesh;
use hyperactor_mesh::alloc::Alloc;
use hyperactor_mesh::alloc::ProcStopReason;
use hyperactor_mesh::proc_mesh::ProcEvent;
use hyperactor_mesh::proc_mesh::ProcEvents;
use hyperactor_mesh::proc_mesh::ProcMesh;
use hyperactor_mesh::proc_mesh::SharedSpawnable;
use hyperactor_mesh::shared_cell::SharedCell;
use hyperactor_mesh::shared_cell::SharedCellPool;
use hyperactor_mesh::shared_cell::SharedCellRef;
use monarch_types::PickledPyObject;
use ndslice::Shape;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::pycell::PyRef;
use pyo3::types::PyType;
use tokio::sync::Mutex;
use tokio::sync::mpsc;

use crate::actor_mesh::PythonActorMesh;
use crate::mailbox::PyMailbox;
use crate::runtime::signal_safe_block_on;
use crate::shape::PyShape;

// A wrapper around `ProcMesh` which keeps track of all `RootActorMesh`s that it spawns.
pub struct TrackedProcMesh {
    inner: SharedCellRef<ProcMesh>,
    cell: SharedCell<ProcMesh>,
    children: SharedCellPool,
}

impl Debug for TrackedProcMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&*self.inner, f)
    }
}

impl Display for TrackedProcMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&*self.inner, f)
    }
}

impl From<ProcMesh> for TrackedProcMesh {
    fn from(mesh: ProcMesh) -> Self {
        let cell = SharedCell::from(mesh);
        let inner = cell.borrow().unwrap();
        Self {
            inner,
            cell,
            children: SharedCellPool::new(),
        }
    }
}

impl TrackedProcMesh {
    pub async fn spawn<A: Actor + RemoteActor>(
        &self,
        actor_name: &str,
        params: &A::Params,
    ) -> Result<SharedCell<RootActorMesh<'static, A>>, anyhow::Error>
    where
        A::Params: RemoteMessage,
    {
        let mesh = self.cell.borrow()?;
        let actor = mesh.spawn(actor_name, params).await?;
        Ok(self.children.insert(actor))
    }

    pub fn client(&self) -> &Mailbox {
        self.inner.client()
    }

    pub fn shape(&self) -> &Shape {
        self.inner.shape()
    }

    pub fn client_proc(&self) -> &Proc {
        self.inner.client_proc()
    }

    pub fn into_inner(self) -> (SharedCell<ProcMesh>, SharedCellPool) {
        (self.cell, self.children)
    }
}

#[pyclass(
    name = "ProcMesh",
    module = "monarch._rust_bindings.monarch_hyperactor.proc_mesh"
)]
pub struct PyProcMesh {
    inner: SharedCell<TrackedProcMesh>,
    keepalive: Keepalive,
    proc_events: SharedCell<Mutex<ProcEvents>>,
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
        let proc_events = SharedCell::from(Mutex::new(proc_mesh.events().unwrap()));
        let monitor = tokio::spawn(Self::default_proc_mesh_monitor(
            proc_events
                .borrow()
                .expect("borrowing immediately after creation"),
            world_id,
            abort_receiver,
        ));
        Self {
            inner: SharedCell::from(TrackedProcMesh::from(proc_mesh)),
            keepalive: Keepalive::new(monitor),
            proc_events,
            stop_monitor_sender: sender,
            user_monitor_registered: AtomicBool::new(false),
        }
    }

    /// The default monitor of the proc mesh for crashes. If a proc crashes, we print the reason
    /// to stderr and exit with code 1.
    async fn default_proc_mesh_monitor(
        events: SharedCellRef<Mutex<ProcEvents>>,
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
                _ = async {
                    tokio::select! {
                        _ = events.preempted() => (),
                        _ = abort_receiver.recv() => (),
                    }
                 } => {
                    // The default monitor is aborted, this happens when user takes over
                    // the monitoring responsibility.
                    eprintln!("stop default supervision monitor for ProcMesh {}", world_id);
                    break;
                }
            }
        }
    }

    pub fn try_inner(&self) -> PyResult<SharedCellRef<TrackedProcMesh>> {
        self.inner
            .borrow()
            .map_err(|_| PyRuntimeError::new_err("`ProcMesh` has already been stopped"))
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
        let proc_mesh = self.try_inner()?;
        let keepalive = self.keepalive.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mailbox = proc_mesh.client().clone();
            let actor_mesh = proc_mesh.spawn(&name, &pickled_type).await?;
            let python_actor_mesh = PythonActorMesh {
                inner: actor_mesh,
                client: PyMailbox { inner: mailbox },
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
        let proc_mesh = self.try_inner()?;
        let keepalive = self.keepalive.clone();
        signal_safe_block_on(py, async move {
            let mailbox = proc_mesh.client().clone();
            let actor_mesh = proc_mesh.spawn(&name, &pickled_type).await?;
            let python_actor_mesh = PythonActorMesh {
                inner: actor_mesh,
                client: PyMailbox { inner: mailbox },
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
    fn client(&self) -> PyResult<PyMailbox> {
        Ok(PyMailbox {
            inner: self.try_inner()?.client().clone(),
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("<ProcMesh {}>", *self.try_inner()?))
    }

    #[getter]
    fn shape(&self) -> PyResult<PyShape> {
        Ok(self.try_inner()?.shape().clone().into())
    }

    fn stop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let tracked_proc_mesh = self.inner.clone();
        let proc_events = self.proc_events.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            async {
                // "Take" the proc mesh wrapper.  Once we do, it should be impossible for new
                // actor meshes to be spawned.
                let (proc_mesh, children) = tracked_proc_mesh
                    .take()
                    .await
                    .map_err(|_| PyRuntimeError::new_err("`ProcMesh` has already been stopped"))?
                    .into_inner();
                // Now we discard all in-flight actor meshes.  After this, the `ProcMesh` should be "unused".
                children.discard_all().await?;
                // Finally, take ownership of the inner proc mesh, which will allowing dropping it.
                let _proc_mesh = proc_mesh.take().await?;
                // Grab the alloc back from `ProcEvents` and use that to stop the mesh.
                let mut alloc = proc_events.take().await?.into_inner().into_alloc();
                alloc.stop_and_wait().await?;
                anyhow::Ok(())
            }
            .await?;
            PyResult::Ok(())
        })
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
    proc_events: SharedCell<Mutex<ProcEvents>>,
}

#[pymethods]
impl PyProcMeshMonitor {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let events = self.proc_events.clone();
        Ok(pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let events = events
                .borrow()
                .map_err(|_| PyRuntimeError::new_err("`ProcEvents` is shutdown"))?;
            let mut proc_events = events.lock().await;
            tokio::select! {
                () = events.preempted() => {
                    Err(PyRuntimeError::new_err("shutting down `ProcEvents`"))
                },
                event = proc_events.next() => {
                    match event {
                        Some(event) => Ok(PyProcEvent::from(event)),
                        None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(
                            "stop iteration",
                        )),
                    }
                }
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
