/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unsafe_op_in_unsafe_fn)]

mod client;
mod controller;
mod debugger;
mod simulator_client;
mod worker;

use pyo3::prelude::*;

fn get_or_add_new_module<'py>(
    module: &Bound<'py, PyModule>,
    module_name: &str,
) -> PyResult<Bound<'py, pyo3::types::PyModule>> {
    let mut current_module = module.clone();
    let mut parts = Vec::new();
    for part in module_name.split(".") {
        parts.push(part);
        let submodule = current_module.getattr(part).ok();
        if let Some(submodule) = submodule {
            current_module = submodule.extract()?;
        } else {
            let new_module = PyModule::new_bound(current_module.py(), part)?;
            current_module.add_submodule(&new_module)?;
            current_module
                .py()
                .import_bound("sys")?
                .getattr("modules")?
                .set_item(
                    format!("monarch._rust_bindings.{}", parts.join(".")),
                    new_module.clone(),
                )?;
            current_module = new_module;
        }
    }
    Ok(current_module)
}

#[pymodule]
#[pyo3(name = "_rust_bindings")]
pub fn mod_init(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ::hyperactor::initialize();
    monarch_hyperactor::runtime::initialize(module.py())?;

    monarch_hyperactor::shape::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.shape",
    )?)?;

    monarch_hyperactor::selection::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.selection",
    )?)?;

    client::register_python_bindings(&get_or_add_new_module(module, "monarch_extension.client")?)?;
    worker::register_python_bindings(&get_or_add_new_module(module, "monarch_extension.worker")?)?;

    controller::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_extension.controller",
    )?)?;

    monarch_hyperactor::bootstrap::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.bootstrap",
    )?)?;

    monarch_hyperactor::proc::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.proc",
    )?)?;

    monarch_hyperactor::actor::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.actor",
    )?)?;

    monarch_hyperactor::mailbox::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.mailbox",
    )?)?;

    monarch_hyperactor::alloc::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.alloc",
    )?)?;
    monarch_hyperactor::actor_mesh::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.actor_mesh",
    )?)?;
    monarch_hyperactor::proc_mesh::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.proc_mesh",
    )?)?;

    monarch_rdma_extension::register_python_bindings(&get_or_add_new_module(module, "rdma")?)?;

    monarch_hyperactor::runtime::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.runtime",
    )?)?;
    debugger::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_extension.debugger",
    )?)?;
    monarch_messages::debugger::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_messages.debugger",
    )?)?;
    simulator_client::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_extension.simulator_client",
    )?)?;
    hyperactor_extension::alloc::register_python_bindings(&get_or_add_new_module(
        module,
        "hyperactor_extension.alloc",
    )?)?;
    hyperactor_extension::telemetry::register_python_bindings(&get_or_add_new_module(
        module,
        "hyperactor_extension.telemetry",
    )?)?;
    ::controller::bootstrap::register_python_bindings(&get_or_add_new_module(
        module,
        "controller.bootstrap",
    )?)?;

    ::monarch_worker::bootstrap::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_worker.bootstrap",
    )?)?;

    Ok(())
}
