#![allow(unsafe_op_in_unsafe_fn)]

pub mod alloc;
pub mod alloc_mock;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_lib_meta")]
pub fn mod_init(module: &Bound<'_, PyModule>) -> PyResult<()> {
    //Safety: This needs to be called here because we can't use fbinit::main
    unsafe {
        fbinit::perform_init();
    }

    ::hyperactor::initialize();

    let hyperactor_mod = PyModule::new_bound(module.py(), "hyperactor_meta")?;
    alloc::register_python_bindings(&hyperactor_mod)?;
    alloc_mock::register_python_bindings(&hyperactor_mod)?;

    module.add_submodule(&hyperactor_mod)?;
    Ok(())
}
