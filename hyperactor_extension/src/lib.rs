#![allow(unsafe_op_in_unsafe_fn)]

pub mod alloc;
pub mod python_registration;

use pyo3::Bound;
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::alloc::PyAlloc;
use crate::alloc::PyAllocConstraints;
use crate::alloc::PyAllocSpec;

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = crate::python_registration::get_or_add_new_module(module, "hyperactor")?;

    module.add_class::<PyAlloc>()?;
    module.add_class::<PyAllocConstraints>()?;
    module.add_class::<PyAllocSpec>()?;

    Ok(())
}
