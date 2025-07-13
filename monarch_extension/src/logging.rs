/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unsafe_op_in_unsafe_fn)]

use hyperactor::ActorRef;
use hyperactor_mesh::RootActorMesh;
use hyperactor_mesh::actor_mesh::ActorMesh;
use hyperactor_mesh::logging::LogClientActor;
use hyperactor_mesh::logging::LogForwardActor;
use hyperactor_mesh::logging::LogForwardMessage;
use hyperactor_mesh::selection::Selection;
use hyperactor_mesh::shared_cell::SharedCell;
use monarch_hyperactor::proc_mesh::PyProcMesh;
use monarch_hyperactor::runtime::signal_safe_block_on;
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
}

#[pymethods]
impl LoggingMeshClient {
    #[staticmethod]
    #[pyo3(signature = (*, proc_mesh))]
    fn spawn_blocking(py: Python, proc_mesh: &PyProcMesh) -> PyResult<Self> {
        let proc_mesh = proc_mesh.try_inner()?;
        signal_safe_block_on(py, async move {
            let client_actor: ActorRef<LogClientActor> = proc_mesh
                .client_proc()
                .spawn("log_client", ())
                .await?
                .bind();
            let actor_mesh = proc_mesh.spawn("log_forwarder", &client_actor).await?;
            Ok(Self { actor_mesh })
        })?
    }

    fn set_mode<'py>(&self, py: Python<'py>, stream_to_client: bool) -> PyResult<()> {
        let inner_mesh = self.actor_mesh.borrow().map_err(anyhow::Error::msg)?;
        signal_safe_block_on(py, async move {
            inner_mesh.cast(
                Selection::True,
                LogForwardMessage::SetMode { stream_to_client },
            )
        })?
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LoggingMeshClient>()?;
    Ok(())
}
