use std::sync::Arc;

use hyperactor_extension::alloc::PyAlloc;
use hyperactor_mesh::proc_mesh::ProcMesh;
use hyperactor_mesh::proc_mesh::SharedSpawnable;
use monarch_types::PickledPyObject;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::actor_mesh::PythonActorMesh;
use crate::mailbox::PyMailbox;

#[pyclass(name = "ProcMesh", module = "monarch._monarch.hyperactor")]
pub struct PyProcMesh {
    pub(super) inner: Arc<ProcMesh>,
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
        let mesh = ProcMesh::allocate(alloc)
            .await
            .map_err(|err| PyException::new_err(err.to_string()))?;
        Ok(PyProcMesh {
            inner: Arc::new(mesh),
        })
    })
}

#[pymethods]
impl PyProcMesh {
    #[classmethod]
    fn allocate<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        alloc: &PyAlloc,
    ) -> PyResult<Bound<'py, PyAny>> {
        allocate_proc_mesh(py, alloc)
    }

    fn spawn<'py>(
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

    #[getter]
    fn client(&self) -> PyMailbox {
        PyMailbox {
            inner: self.inner.client().clone(),
        }
    }
    #[getter]
    fn proc_id(&self) -> String {
        self.inner.proc_id().to_string()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("<ProcMesh {}>", self.inner))
    }
}
