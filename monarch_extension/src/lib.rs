#![allow(unsafe_op_in_unsafe_fn)]

mod client;
mod controller;
mod debugger;
mod simulator_client;
mod worker;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_rust_bindings")]
pub fn mod_init(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ::hyperactor::initialize();
    monarch_hyperactor::runtime::initialize(module.py())?;

    monarch_hyperactor::shape::register_python_bindings(module)?;
    client::register_python_bindings(module)?;
    worker::register_python_bindings(module)?;
    controller::register_python_bindings(module)?;
    monarch_hyperactor::register_python_bindings(module)?;
    monarch_hyperactor::runtime::register_python_bindings(module)?;
    debugger::register_python_bindings(module)?;
    simulator_client::register_python_bindings(module)?;
    hyperactor_extension::register_python_bindings(module)?;

    Ok(())
}
