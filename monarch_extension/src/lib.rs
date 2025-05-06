#![allow(unsafe_op_in_unsafe_fn)]

mod client;
mod controller;
mod debugger;
mod simulator_client;
mod worker;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_lib")]
pub fn mod_init(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ::hyperactor::initialize();
    monarch_hyperactor::runtime::initialize(module.py())?;

    monarch_hyperactor::shape::init_pymodule(module)?;
    client::init_pymodule(module)?;
    worker::init_pymodule(module)?;
    controller::init_pymodule(module)?;
    monarch_hyperactor::init_pymodule(module)?;
    monarch_hyperactor::runtime::init_pymodule(module)?;
    debugger::init_pymodule(module)?;
    simulator_client::init_pymodule(module)?;

    Ok(())
}
