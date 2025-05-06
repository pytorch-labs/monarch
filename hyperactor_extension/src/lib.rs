pub mod alloc;

use pyo3::Bound;
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::alloc::PyAllocSpec;

pub fn init_pymodule(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAllocSpec>()?;

    Ok(())
}
