/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use hyperactor::clock::Clock;
use hyperactor::clock::SimClock;
use hyperactor::simnet;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "start_event_loop")]
pub fn start_simnet_event_loop(py: Python) -> PyResult<Bound<'_, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        simnet::start(
            ChannelAddr::any(ChannelTransport::Unix),
            ChannelAddr::any(ChannelTransport::Unix),
            1000,
        )
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(())
    })
}

#[pyfunction]
#[pyo3(name="sleep",signature=(seconds))]
pub fn py_sim_sleep<'py>(py: Python<'py>, seconds: f64) -> PyResult<Bound<'py, PyAny>> {
    let millis = (seconds * 1000.0).ceil() as u64;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let duration = tokio::time::Duration::from_millis(millis);
        SimClock.sleep(duration).await;
        Ok(())
    })
}

pub(crate) fn register_python_bindings(simulation_tools_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    {
        let f = wrap_pyfunction!(py_sim_sleep, simulation_tools_mod)?;
        f.setattr(
            "__module__",
            "monarch._rust_bindings.monarch_extension.simulation_tools",
        )?;
        simulation_tools_mod.add_function(f)?;
    }
    {
        let f = wrap_pyfunction!(start_simnet_event_loop, simulation_tools_mod)?;
        f.setattr(
            "__module__",
            "monarch._rust_bindings.monarch_extension.simulation_tools",
        )?;
        simulation_tools_mod.add_function(f)?;
    }
    Ok(())
}
