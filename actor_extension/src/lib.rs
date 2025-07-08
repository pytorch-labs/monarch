/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! This module is used to expose Rust bindings for code supporting the
//! `monarch.actor` module.
//!
//! It is imported by `monarch` as `monarch._src.actor._extension`.
use pyo3::prelude::*;

mod blocking;
mod code_sync;
mod panic;

#[cfg(fbcode_build)]
mod meta;

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
            let new_module = PyModule::new(current_module.py(), part)?;
            current_module.add_submodule(&new_module)?;
            current_module
                .py()
                .import("sys")?
                .getattr("modules")?
                .set_item(
                    format!("monarch._src.actor._extension.{}", parts.join(".")),
                    new_module.clone(),
                )?;
            current_module = new_module;
        }
    }
    Ok(current_module)
}

#[pymodule]
#[pyo3(name = "_extension")]
pub fn mod_init(module: &Bound<'_, PyModule>) -> PyResult<()> {
    monarch_hyperactor::runtime::initialize(module.py())?;
    let runtime = monarch_hyperactor::runtime::get_tokio_runtime();
    ::hyperactor::initialize(runtime.handle().clone());

    monarch_hyperactor::shape::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.shape",
    )?)?;

    monarch_hyperactor::selection::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.selection",
    )?)?;
    code_sync::register_python_bindings(&get_or_add_new_module(module, "code_sync")?)?;
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
    monarch_hyperactor::channel::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.channel",
    )?)?;
    monarch_hyperactor::actor_mesh::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.actor_mesh",
    )?)?;
    monarch_hyperactor::proc_mesh::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.proc_mesh",
    )?)?;

    monarch_hyperactor::runtime::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_hyperactor.runtime",
    )?)?;
    hyperactor_extension::alloc::register_python_bindings(&get_or_add_new_module(
        module,
        "hyperactor_extension.alloc",
    )?)?;
    hyperactor_extension::telemetry::register_python_bindings(&get_or_add_new_module(
        module,
        "hyperactor_extension.telemetry",
    )?)?;
    crate::panic::register_python_bindings(&get_or_add_new_module(module, "panic")?)?;

    crate::blocking::register_python_bindings(&get_or_add_new_module(module, "blocking")?)?;

    #[cfg(fbcode_build)]
    crate::meta::register_python_bindings(&get_or_add_new_module(module, "meta")?)?;
    Ok(())
}
