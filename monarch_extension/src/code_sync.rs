/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unsafe_op_in_unsafe_fn)]

use std::path::PathBuf;
use std::sync::Arc;

use hyperactor_mesh::RootActorMesh;
use hyperactor_mesh::SlicedActorMesh;
use hyperactor_mesh::code_sync::rsync;
use hyperactor_mesh::proc_mesh::SharedSpawnable;
use hyperactor_mesh::shape::Shape;
use monarch_hyperactor::proc_mesh::PyProcMesh;
use monarch_hyperactor::runtime::signal_safe_block_on;
use monarch_hyperactor::shape::PyShape;
use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass(
    frozen,
    name = "RsyncMeshClient",
    module = "monarch._rust_bindings.monarch_extension.code_sync"
)]
pub struct RsyncMeshClient {
    actor_mesh: Arc<RootActorMesh<'static, rsync::RsyncActor>>,
    shape: Shape,
    workspace: PathBuf,
}

#[pymethods]
impl RsyncMeshClient {
    #[staticmethod]
    #[pyo3(signature = (*, proc_mesh, shape, workspace))]
    fn spawn_blocking(
        py: Python,
        proc_mesh: &PyProcMesh,
        shape: &PyShape,
        workspace: PathBuf,
    ) -> PyResult<Self> {
        let proc_mesh = Arc::clone(&proc_mesh.inner);
        let shape = shape.get_inner().clone();
        signal_safe_block_on(py, async move {
            let actor_mesh = proc_mesh.spawn("rsync", &rsync::RsyncParams {}).await?;
            Ok(Self {
                actor_mesh: Arc::new(actor_mesh),
                shape,
                workspace,
            })
        })?
    }

    fn sync_workspace<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let workspace = self.workspace.clone();
        let inner_mesh = self.actor_mesh.clone();
        let shape = self.shape.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mesh = SlicedActorMesh::new(&inner_mesh, shape);
            Ok(rsync::rsync_mesh(mesh, workspace).await?)
        })
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RsyncMeshClient>()?;
    Ok(())
}
