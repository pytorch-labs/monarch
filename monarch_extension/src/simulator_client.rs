/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![cfg(feature = "tensor_engine")]

use actor_extension_lib::runtime::signal_safe_block_on;
use anyhow::anyhow;
use hyperactor::PortId;
use hyperactor::attrs::Attrs;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::Tx;
use hyperactor::channel::dial;
use hyperactor::data::Serialized;
use hyperactor::id;
use hyperactor::mailbox::MessageEnvelope;
use hyperactor::simnet::OperationalMessage;
use hyperactor::simnet::ProxyMessage;
use hyperactor::simnet::SpawnMesh;
use hyperactor::simnet::TrainingScriptState;
use monarch_simulator_lib::bootstrap::bootstrap;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A wrapper around [ndslice::Slice] to expose it to python.
/// It is a compact representation of indices into the flat
/// representation of an n-dimensional array. Given an offset, sizes of
/// each dimension, and strides for each dimension, Slice can compute
/// indices into the flat array.
#[pyclass(
    name = "SimulatorClient",
    frozen,
    module = "monarch._rust_bindings.monarch_extension.simulator_client"
)]
#[derive(Clone)]
pub(crate) struct SimulatorClient {
    proxy_addr: ChannelAddr,
}

fn wrap_operational_message(operational_message: OperationalMessage) -> MessageEnvelope {
    let serialized_operational_message = Serialized::serialize(&operational_message).unwrap();
    let proxy_message = ProxyMessage::new(None, None, serialized_operational_message);
    let serialized_proxy_message = Serialized::serialize(&proxy_message).unwrap();
    let sender_id = id!(simulator_client[0].sender_actor);
    // a dummy port ID. We are delivering message with low level mailbox.
    // The port ID is not used.
    let port_id = PortId(id!(simulator[0].actor), 0);
    MessageEnvelope::new(sender_id, port_id, serialized_proxy_message, Attrs::new())
}

#[pyfunction]
fn bootstrap_simulator_backend(
    py: Python,
    system_addr: String,
    proxy_addr: String,
    world_size: i32,
) -> PyResult<()> {
    signal_safe_block_on(py, async move {
        match bootstrap(
            system_addr.parse().unwrap(),
            proxy_addr.parse().unwrap(),
            world_size as usize,
        )
        .await
        {
            Ok(_) => Ok(()),
            Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
        }
    })?
}

fn set_training_script_state(state: TrainingScriptState, proxy_addr: ChannelAddr) -> PyResult<()> {
    let operational_message = OperationalMessage::SetTrainingScriptState(state);
    let external_message = wrap_operational_message(operational_message);
    let tx = dial(proxy_addr).map_err(|err| anyhow!(err))?;
    tx.post(external_message);
    Ok(())
}

#[pymethods]
impl SimulatorClient {
    #[new]
    fn new(proxy_addr: &str) -> PyResult<Self> {
        Ok(Self {
            proxy_addr: proxy_addr
                .parse::<ChannelAddr>()
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        })
    }

    fn kill_world(&self, world_name: &str) -> PyResult<()> {
        let operational_message = OperationalMessage::KillWorld(world_name.to_string());
        let external_message = wrap_operational_message(operational_message);
        let tx = dial(self.proxy_addr.clone()).map_err(|err| anyhow!(err))?;
        tx.post(external_message);
        Ok(())
    }

    fn spawn_mesh(
        &self,
        system_addr: &str,
        controller_actor_id: &str,
        worker_world: &str,
    ) -> PyResult<()> {
        let spawn_mesh = SpawnMesh::new(
            system_addr.parse().unwrap(),
            controller_actor_id.parse().unwrap(),
            worker_world.parse().unwrap(),
        );
        let operational_message = OperationalMessage::SpawnMesh(spawn_mesh);
        let external_message = wrap_operational_message(operational_message);
        let tx = dial(self.proxy_addr.clone()).map_err(|err| anyhow!(err))?;
        tx.post(external_message);
        Ok(())
    }

    fn set_training_script_state_running(&self) -> PyResult<()> {
        set_training_script_state(TrainingScriptState::Running, self.proxy_addr.clone())
    }

    fn set_training_script_state_waiting(&self) -> PyResult<()> {
        set_training_script_state(TrainingScriptState::Waiting, self.proxy_addr.clone())
    }
}

pub(crate) fn register_python_bindings(simulator_client_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    simulator_client_mod.add_class::<SimulatorClient>()?;
    let f = wrap_pyfunction!(bootstrap_simulator_backend, simulator_client_mod)?;
    f.setattr(
        "__module__",
        "monarch._rust_bindings.monarch_extension.simulator_client",
    )?;
    simulator_client_mod.add_function(f)?;
    Ok(())
}
