/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor::ActorRef;
use hyperactor_mesh::Mesh;
use hyperactor_mesh::RootActorMesh;
use hyperactor_mesh::actor_mesh::ActorMesh;
use hyperactor_mesh::shared_cell::SharedCell;
use hyperactor_mesh::shared_cell::SharedCellRef;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::actor::PythonActor;
use crate::actor::PythonMessage;
use crate::mailbox::PyMailbox;
use crate::proc::PyActorId;
use crate::proc_mesh::Keepalive;
use crate::selection::PySelection;
use crate::shape::PyShape;

#[pyclass(
    name = "PythonActorMesh",
    module = "monarch._rust_bindings.monarch_hyperactor.actor_mesh"
)]
pub struct PythonActorMesh {
    pub inner: SharedCell<RootActorMesh<'static, PythonActor>>,
    pub client: PyMailbox,
    pub(super) _keepalive: Keepalive,
}

impl PythonActorMesh {
    fn try_inner(&self) -> PyResult<SharedCellRef<RootActorMesh<'static, PythonActor>>> {
        self.inner
            .borrow()
            .map_err(|_| PyRuntimeError::new_err("`PythonActorMesh` has already been stopped"))
    }
}

#[pymethods]
impl PythonActorMesh {
    fn cast(&self, selection: &PySelection, message: &PythonMessage) -> PyResult<()> {
        self.try_inner()?
            .cast(selection.inner().clone(), message.clone())
            .map_err(|err| PyException::new_err(err.to_string()))?;
        Ok(())
    }

    // Consider defining a "PythonActorRef", which carries specifically
    // a reference to python message actors.
    fn get(&self, rank: usize) -> PyResult<Option<PyActorId>> {
        Ok(self
            .try_inner()?
            .get(rank)
            .map(ActorRef::into_actor_id)
            .map(PyActorId::from))
    }

    #[getter]
    pub fn client(&self) -> PyMailbox {
        self.client.clone()
    }

    #[getter]
    fn shape(&self) -> PyResult<PyShape> {
        Ok(PyShape::from(self.try_inner()?.shape().clone()))
    }
}
pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PythonActorMesh>()?;
    Ok(())
}
