use pyo3::prelude::*;
use pyo3::types::PyModule;

pub fn get_or_add_new_module<'py>(
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
