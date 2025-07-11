/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![cfg(feature = "tensor_engine")]

use std::sync::Arc;

use anyhow::anyhow;
use hyperactor::WorldId;
use hyperactor::channel::ChannelAddr;
use hyperactor::simnet;
use hyperactor::simnet::TrainingScriptState;
use hyperactor::simnet::simnet_handle;
use monarch_hyperactor::runtime::signal_safe_block_on;
use monarch_simulator_lib::simulator::TensorEngineSimulator;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tokio::sync::Mutex;

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
    inner: Arc<Mutex<TensorEngineSimulator>>,
    world_size: usize,
}

fn set_training_script_state(state: TrainingScriptState) -> PyResult<()> {
    simnet_handle()
        .map_err(|e| anyhow!(e))?
        .set_training_script_state(state);
    Ok(())
}

#[pymethods]
impl SimulatorClient {
    #[new]
    fn new(py: Python, system_addr: String, world_size: i32) -> PyResult<Self> {
        signal_safe_block_on(py, async move {
            simnet::start();

            Ok(Self {
                inner: Arc::new(Mutex::new(
                    TensorEngineSimulator::new(
                        system_addr
                            .parse::<ChannelAddr>()
                            .map_err(|err| PyValueError::new_err(err.to_string()))?,
                    )
                    .await
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?,
                )),
                world_size: world_size as usize,
            })
        })?
    }

    fn kill_world(&self, py: Python, world_name: &str) -> PyResult<()> {
        let simulator = self.inner.clone();
        let world_name = world_name.to_string();

        signal_safe_block_on(py, async move {
            simulator
                .lock()
                .await
                .kill_world(&world_name)
                .map_err(|err| anyhow!(err))?;
            Ok(())
        })?
    }

    fn spawn_mesh(
        &self,
        py: Python,
        system_addr: &str,
        controller_actor_id: &str,
        worker_world: &str,
    ) -> PyResult<()> {
        let simulator = self.inner.clone();
        let world_size = self.world_size;
        let system_addr = system_addr.parse::<ChannelAddr>().unwrap();
        let worker_world = worker_world.parse::<WorldId>().unwrap();
        let controller_actor_id = controller_actor_id.parse().unwrap();

        signal_safe_block_on(py, async move {
            simulator
                .lock()
                .await
                .spawn_mesh(system_addr, controller_actor_id, worker_world, world_size)
                .await
                .map_err(|err| anyhow!(err))?;
            Ok(())
        })?
    }

    fn set_training_script_state_running(&self) -> PyResult<()> {
        set_training_script_state(TrainingScriptState::Running)
    }

    fn set_training_script_state_waiting(&self) -> PyResult<()> {
        set_training_script_state(TrainingScriptState::Waiting)
    }
}

pub(crate) fn register_python_bindings(simulator_client_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    simulator_client_mod.add_class::<SimulatorClient>()?;
    Ok(())
}
