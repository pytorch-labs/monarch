/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unsafe_op_in_unsafe_fn)]

use hyperactor::ActorHandle;
use hyperactor_mesh::RootActorMesh;
use hyperactor_mesh::actor_mesh::ActorMesh;
use hyperactor_mesh::logging::LogClientActor;
use hyperactor_mesh::logging::LogClientMessage;
use hyperactor_mesh::logging::LogForwardActor;
use hyperactor_mesh::logging::LogForwardMessage;
use hyperactor_mesh::selection::Selection;
use hyperactor_mesh::shared_cell::SharedCell;
use monarch_hyperactor::proc_mesh::PyProcMesh;
use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass(
    frozen,
    name = "LoggingMeshClient",
    module = "monarch._rust_bindings.monarch_extension.logging"
)]
pub struct LoggingMeshClient {
    actor_mesh: SharedCell<RootActorMesh<'static, LogForwardActor>>,
    client_actor: ActorHandle<LogClientActor>,
}

#[pymethods]
impl LoggingMeshClient {
    #[staticmethod]
    #[pyo3(signature = (*, proc_mesh))]
    fn spawn<'py>(py: Python<'py>, proc_mesh: &PyProcMesh) -> PyResult<Bound<'py, PyAny>> {
        let proc_mesh = proc_mesh.try_inner()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let client_actor = proc_mesh.client_proc().spawn("log_client", ()).await?;
            let client_actor_ref = client_actor.bind();
            let actor_mesh = proc_mesh.spawn("log_forwarder", &client_actor_ref).await?;
            Ok(Self {
                actor_mesh,
                client_actor,
            })
        })
    }

    fn set_mode<'py>(
        &self,
        _py: Python<'py>,
        stream_to_client: bool,
        aggregate_window_sec: Option<u64>,
    ) -> PyResult<()> {
        let inner_mesh = self.actor_mesh.borrow().map_err(anyhow::Error::msg)?;

        inner_mesh
            .cast(
                Selection::True,
                LogForwardMessage::SetMode { stream_to_client },
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        self.client_actor
            .send(LogClientMessage::SetAggregate {
                aggregate_window_sec,
            })
            .map_err(anyhow::Error::msg)?;
        if aggregate_window_sec.is_some() && !stream_to_client {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Cannot set aggregate window without streaming to client".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LoggingMeshClient>()?;
    Ok(())
}
