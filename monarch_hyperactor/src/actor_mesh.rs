/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor_mesh::ActorMesh;
use hyperactor_mesh::Mesh;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use crate::actor::PythonActor;
use crate::actor::PythonMessage;
use crate::mailbox::PyMailbox;
use crate::proc::PyActorId;
use crate::shape::PyShape;

#[pyclass(
    name = "PythonActorMesh",
    module = "monarch._rust_bindings.monarch_hyperactor.actor_mesh"
)]
pub struct PythonActorMesh {
    pub(super) inner: Arc<ActorMesh<'static, PythonActor>>,
    pub(super) client: PyMailbox,
}

#[pymethods]
impl PythonActorMesh {
    fn cast(&self, message: &PythonMessage) -> PyResult<()> {
        use ndslice::selection::dsl::*;
        self.inner
            .cast(all(true_()), message.clone())
            .map_err(|err| PyException::new_err(err.to_string()))?;
        Ok(())
    }

    // Consider defining a "PythonActorRef", which carries specifically
    // a reference to python message actors.
    fn get(&self, rank: usize) -> Option<PyActorId> {
        self.inner
            .get(rank)
            .map(ActorRef::into_actor_id)
            .map(PyActorId::from)
    }

    #[getter]
    pub fn client(&self) -> PyMailbox {
        self.client.clone()
    }

    #[getter]
    fn shape(&self) -> PyShape {
        PyShape::from(self.inner.shape().clone())
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PythonActorMesh>()?;
    Ok(())
}
