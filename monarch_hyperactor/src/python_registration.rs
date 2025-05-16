use pyo3::prelude::*;
use pyo3::types::PyModule;

pub fn add_new_module<'py>(
    module: &Bound<'py, PyModule>,
    module_name: &str,
) -> PyResult<Bound<'py, pyo3::types::PyModule>> {
    let new_module = PyModule::new_bound(module.py(), module_name)?;
    module.add_submodule(&new_module)?;

    // submodules are normally separate .so libraries. However we package them in one omnibus .so
    // to make sure they are still importable as if they were a package, we have to manually load them
    // into modules otherwise we get _rust is not a package errors.
    module
        .py()
        .import_bound("sys")?
        .getattr("modules")?
        .set_item(
            format!("monarch._rust_bindings.{}", module_name),
            new_module.clone(),
        )?;
    Ok(new_module)
}
