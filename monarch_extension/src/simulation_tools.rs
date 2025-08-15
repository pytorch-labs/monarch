/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor::clock::Clock;
use hyperactor::clock::SimClock;
use hyperactor::simnet;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "start_event_loop")]
pub fn start_simnet_event_loop(py: Python) -> PyResult<Bound<'_, PyAny>> {
    monarch_hyperactor::runtime::future_into_py(py, async move {
        simnet::start();
        Ok(())
    })
}

#[pyfunction]
#[pyo3(name="sleep",signature=(seconds))]
pub fn py_sim_sleep<'py>(py: Python<'py>, seconds: f64) -> PyResult<Bound<'py, PyAny>> {
    let millis = (seconds * 1000.0).ceil() as u64;
    monarch_hyperactor::runtime::future_into_py(py, async move {
        let duration = tokio::time::Duration::from_millis(millis);
        SimClock.sleep(duration).await;
        Ok(())
    })
}

#[pyfunction]
#[pyo3(name = "is_simulator_active")]
pub fn is_simulator_active() -> PyResult<bool> {
    // Use the existing simnet_handle() function to check if SimNet is active
    Ok(hyperactor::simnet::simnet_handle().is_ok())
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
    {
        let f = wrap_pyfunction!(is_simulator_active, simulation_tools_mod)?;
        f.setattr(
            "__module__",
            "monarch._rust_bindings.monarch_extension.simulation_tools",
        )?;
        simulation_tools_mod.add_function(f)?;
    }
    Ok(())
}
