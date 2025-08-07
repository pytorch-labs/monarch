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
use monarch_hyperactor::logging::LoggerRuntimeActor;
use monarch_hyperactor::logging::LoggerRuntimeMessage;
use monarch_hyperactor::proc_mesh::PyProcMesh;
use monarch_hyperactor::pytokio::PyPythonTask;
use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass(
    frozen,
    name = "LoggingMeshClient",
    module = "monarch._rust_bindings.monarch_extension.logging"
)]
pub struct LoggingMeshClient {
    // handles remote process log forwarding; no python runtime
    forwarder_mesh: SharedCell<RootActorMesh<'static, LogForwardActor>>,
    // handles python logger; has python runtime
    logger_mesh: SharedCell<RootActorMesh<'static, LoggerRuntimeActor>>,
    client_actor: ActorHandle<LogClientActor>,
}

#[pymethods]
impl LoggingMeshClient {
    #[staticmethod]
    fn spawn(proc_mesh: &PyProcMesh) -> PyResult<PyPythonTask> {
        let proc_mesh = proc_mesh.try_inner()?;
        PyPythonTask::new(async move {
            let client_actor = proc_mesh.client_proc().spawn("log_client", ()).await?;
            let client_actor_ref = client_actor.bind();
            let forwarder_mesh = proc_mesh.spawn("log_forwarder", &client_actor_ref).await?;
            let logger_mesh = proc_mesh.spawn("logger", &()).await?;
            Ok(Self {
                forwarder_mesh,
                logger_mesh,
                client_actor,
            })
        })
    }

    fn set_mode<'py>(
        &self,
        _py: Python<'py>,
        stream_to_client: bool,
        aggregate_window_sec: Option<u64>,
        level: u8,
    ) -> PyResult<()> {
        if aggregate_window_sec.is_some() && !stream_to_client {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Cannot set aggregate window without streaming to client".to_string(),
            ));
        }

        let forwarder_inner_mesh = self.forwarder_mesh.borrow().map_err(anyhow::Error::msg)?;
        forwarder_inner_mesh
            .cast(
                Selection::True,
                LogForwardMessage::SetMode { stream_to_client },
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let logger_inner_mesh = self.logger_mesh.borrow().map_err(anyhow::Error::msg)?;
        logger_inner_mesh
            .cast(Selection::True, LoggerRuntimeMessage::SetLogging { level })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        self.client_actor
            .send(LogClientMessage::SetAggregate {
                aggregate_window_sec,
            })
            .map_err(anyhow::Error::msg)?;

        Ok(())
    }
}

impl Drop for LoggingMeshClient {
    fn drop(&mut self) {
        let _ = self.client_actor.drain_and_stop().unwrap();
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LoggingMeshClient>()?;
    Ok(())
}
