/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::future::Future;
use std::sync::Arc;

use hyperactor_extension::alloc::PyAlloc;
use hyperactor_extension::alloc::PyAllocWrapper;
use hyperactor_mesh::actor_mesh::ActorMesh;
use hyperactor_mesh::proc_mesh::ProcMesh;
use hyperactor_mesh::proc_mesh::SharedSpawnable;
use monarch_rdma::IbverbsConfig;
use monarch_rdma::RdmaManagerActor;
use monarch_types::PickledPyObject;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::actor_mesh::PythonActorMesh;
use crate::mailbox::PyMailbox;
use crate::runtime::signal_safe_block_on;

/// `PyProcMesh` is a Python wrapper for `ProcMesh`.
/// If the system supports RDMA (Remote Direct Memory Access), an RDMA manager
/// actor is spawned to enable the use of RDMABuffers (see monarch_rdma).
#[pyclass(
    name = "ProcMesh",
    module = "monarch._rust_bindings.monarch_hyperactor.proc_mesh"
)]
pub struct PyProcMesh {
    pub(super) inner: Arc<ProcMesh>,
    pub(super) rdma_manager: Option<Arc<ActorMesh<'static, RdmaManagerActor>>>,
}

/// Creates a new `PyProcMesh` instance asynchronously.
///
/// This function initializes a new process mesh with the provided allocator and
/// sets up RDMA (Remote Direct Memory Access) if supported by the system.
///
/// # Arguments
/// * `alloc` - A wrapper around the Python allocator to be used for the process mesh
///
/// # Returns
/// * A future that resolves to a `PyResult<PyProcMesh>` - the new process mesh or an error
///
/// # Errors
/// * Returns a `PyException` if allocation fails or if RDMA initialization fails
fn create_py_proc_mesh(alloc: PyAllocWrapper) -> impl Future<Output = PyResult<PyProcMesh>> {
    async move {
        let mesh = ProcMesh::allocate(alloc)
            .await
            .map_err(|err| PyException::new_err(err.to_string()))?;

        let inner = Arc::new(mesh);

        let rdma_manager = if monarch_rdma::ibverbs_supported() {
            // TODO - make this configurable
            let config = IbverbsConfig::default();
            tracing::debug!("rdma is enabled, using device {}", config.device);
            let actor_mesh = inner
                .spawn("rdma_manager", &config)
                .await
                .map_err(|err| PyException::new_err(err.to_string()))?;
            Some(Arc::new(actor_mesh))
        } else {
            tracing::info!("rdma is not enabled on this hardware");
            None
        };

        Ok(PyProcMesh {
            inner,
            rdma_manager,
        })
    }
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

    pyo3_async_runtimes::tokio::future_into_py(py, create_py_proc_mesh(alloc))
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
    signal_safe_block_on(py, create_py_proc_mesh(alloc))?
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
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyProcMesh>()?;
    Ok(())
}
