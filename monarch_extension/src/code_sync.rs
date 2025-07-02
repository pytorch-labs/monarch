/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unsafe_op_in_unsafe_fn)]

use std::path::PathBuf;

use hyperactor_mesh::RootActorMesh;
use hyperactor_mesh::SlicedActorMesh;
use hyperactor_mesh::code_sync::WorkspaceLocation;
use hyperactor_mesh::code_sync::rsync;
use hyperactor_mesh::shape::Shape;
use hyperactor_mesh::shared_cell::SharedCell;
use monarch_hyperactor::proc_mesh::PyProcMesh;
use monarch_hyperactor::runtime::signal_safe_block_on;
use monarch_hyperactor::shape::PyShape;
use pyo3::Bound;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyModule;
use serde::Deserialize;
use serde::Serialize;

#[pyclass(
    frozen,
    name = "WorkspaceLocation",
    module = "monarch._rust_bindings.monarch_extension.code_sync"
)]
#[derive(Clone, Debug, Serialize, Deserialize)]
enum PyWorkspaceLocation {
    Constant(PathBuf),
    FromEnvVar(String),
}

impl From<PyWorkspaceLocation> for WorkspaceLocation {
    fn from(workspace: PyWorkspaceLocation) -> WorkspaceLocation {
        match workspace {
            PyWorkspaceLocation::Constant(v) => WorkspaceLocation::Constant(v),
            PyWorkspaceLocation::FromEnvVar(v) => WorkspaceLocation::FromEnvVar(v),
        }
    }
}

#[pymethods]
impl PyWorkspaceLocation {
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

    fn resolve(&self) -> PyResult<PathBuf> {
        let loc: WorkspaceLocation = self.clone().into();
        loc.resolve()
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }
}

#[pyclass(
    frozen,
    name = "RsyncMeshClient",
    module = "monarch._rust_bindings.monarch_extension.code_sync"
)]
pub struct RsyncMeshClient {
    actor_mesh: SharedCell<RootActorMesh<'static, rsync::RsyncActor>>,
    shape: Shape,
    workspace: PathBuf,
}

#[pymethods]
impl RsyncMeshClient {
    #[staticmethod]
    #[pyo3(signature = (*, proc_mesh, shape, local_workspace, remote_workspace))]
    fn spawn_blocking(
        py: Python,
        proc_mesh: &PyProcMesh,
        shape: &PyShape,
        local_workspace: PathBuf,
        remote_workspace: PyWorkspaceLocation,
    ) -> PyResult<Self> {
        let proc_mesh = proc_mesh.try_inner()?;
        let shape = shape.get_inner().clone();
        signal_safe_block_on(py, async move {
            let actor_mesh = proc_mesh
                .spawn(
                    "rsync",
                    &rsync::RsyncParams {
                        workspace: remote_workspace.into(),
                    },
                )
                .await?;
            Ok(Self {
                actor_mesh,
                shape,
                workspace: local_workspace,
            })
        })?
    }

    fn sync_workspace<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let workspace = self.workspace.clone();
        let inner_mesh = self.actor_mesh.borrow().map_err(anyhow::Error::msg)?;
        let shape = self.shape.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mesh = SlicedActorMesh::new(&inner_mesh, shape);
            Ok(rsync::rsync_mesh(mesh, workspace).await?)
        })
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyWorkspaceLocation>()?;
    module.add_class::<RsyncMeshClient>()?;
    Ok(())
}
