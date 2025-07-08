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

pub mod actor;
mod actor_mesh;
mod alloc;
mod blocking;
mod bootstrap;
mod channel;
mod code_sync;
mod config;
pub mod mailbox;
pub mod ndslice;
mod panic;
pub mod proc;
pub mod proc_mesh;
pub mod runtime;
mod selection;
pub mod shape;
mod telemetry;

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
    crate::runtime::initialize(module.py())?;
    let runtime = crate::runtime::get_tokio_runtime();

    ::hyperactor::initialize(runtime.handle().clone());

    crate::actor_mesh::register_python_bindings(&get_or_add_new_module(module, "actor_mesh")?)?;
    crate::actor::register_python_bindings(&get_or_add_new_module(module, "actor")?)?;
    crate::alloc::register_python_bindings(&get_or_add_new_module(module, "alloc")?)?;
    crate::blocking::register_python_bindings(&get_or_add_new_module(module, "blocking")?)?;
    crate::bootstrap::register_python_bindings(&get_or_add_new_module(module, "bootstrap")?)?;
    crate::channel::register_python_bindings(&get_or_add_new_module(module, "channel")?)?;
    crate::code_sync::register_python_bindings(&get_or_add_new_module(module, "code_sync")?)?;
    crate::mailbox::register_python_bindings(&get_or_add_new_module(module, "mailbox")?)?;
    crate::panic::register_python_bindings(&get_or_add_new_module(module, "panic")?)?;
    crate::proc_mesh::register_python_bindings(&get_or_add_new_module(module, "proc_mesh")?)?;
    crate::proc::register_python_bindings(&get_or_add_new_module(module, "proc")?)?;
    crate::runtime::register_python_bindings(&get_or_add_new_module(module, "runtime")?)?;
    crate::selection::register_python_bindings(&get_or_add_new_module(module, "selection")?)?;
    crate::shape::register_python_bindings(&get_or_add_new_module(module, "shape")?)?;
    crate::telemetry::register_python_bindings(&get_or_add_new_module(module, "telemetry")?)?;

    #[cfg(fbcode_build)]
    crate::meta::register_python_bindings(&get_or_add_new_module(module, "meta")?)?;
    Ok(())
}
