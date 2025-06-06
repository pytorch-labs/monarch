/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use hyperactor::WorldId;
use hyperactor_extension::alloc::PyAlloc;
use hyperactor_mesh::alloc::Alloc;
use hyperactor_mesh::alloc::ProcStopReason;
use hyperactor_mesh::proc_mesh::ProcEvent;
use hyperactor_mesh::proc_mesh::ProcEvents;
use hyperactor_mesh::proc_mesh::ProcMesh;
use hyperactor_mesh::proc_mesh::SharedSpawnable;
use monarch_types::PickledPyObject;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyType;

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
    monitor: tokio::task::JoinHandle<()>,
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
        let monitor = tokio::spawn(Self::monitor_proc_mesh(
            proc_mesh.events().unwrap(),
            world_id,
        ));
        Self {
            inner: Arc::new(proc_mesh),
            monitor,
        }
    }

    /// Monitor the proc mesh for crashes. If a proc crashes, we print the reason
    /// to stderr and exit with code 1.
    async fn monitor_proc_mesh(mut events: ProcEvents, world_id: WorldId) {
        while let Some(event) = events.next().await {
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
}

impl Drop for PyProcMesh {
    fn drop(&mut self) {
        self.monitor.abort();
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let actor_mesh = proc_mesh.spawn(&name, &pickled_type).await?;
            let python_actor_mesh = PythonActorMesh {
                inner: Arc::new(actor_mesh),
                client: PyMailbox {
                    inner: proc_mesh.client().clone(),
                },
            };
            Ok(Python::with_gil(|py| python_actor_mesh.into_py(py)))
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
        signal_safe_block_on(py, async move {
            let actor_mesh = proc_mesh.spawn(&name, &pickled_type).await?;
            let python_actor_mesh = PythonActorMesh {
                inner: Arc::new(actor_mesh),
                client: PyMailbox {
                    inner: proc_mesh.client().clone(),
                },
            };
            Ok(Python::with_gil(|py| python_actor_mesh.into_py(py)))
        })?
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

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyProcMesh>()?;
    Ok(())
}
