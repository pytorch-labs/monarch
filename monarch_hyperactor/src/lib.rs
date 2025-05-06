#![allow(unsafe_op_in_unsafe_fn)]

pub mod actor;
pub mod actor_mesh;
pub mod alloc;
mod bootstrap;
pub mod mailbox;
pub mod ndslice;
pub mod proc;
pub mod proc_mesh;
pub mod runtime;
pub mod shape;

use pyo3::Bound;
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

pub fn init_pymodule(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let hyperactor_mod = PyModule::new_bound(module.py(), "hyperactor")?;

    hyperactor_mod.add_function(wrap_pyfunction!(proc::init_proc, &hyperactor_mod)?)?;
    hyperactor_mod.add_function(wrap_pyfunction!(actor::init_asyncio_loop, &hyperactor_mod)?)?;
    hyperactor_mod.add_function(wrap_pyfunction!(
        bootstrap::bootstrap_main,
        &hyperactor_mod
    )?)?;

    hyperactor_mod.add_class::<proc::PyProc>()?;
    hyperactor_mod.add_class::<proc::PyActorId>()?;
    hyperactor_mod.add_class::<proc::PySerialized>()?;

    hyperactor_mod.add_class::<actor::PickledMessage>()?;
    hyperactor_mod.add_class::<actor::PickledMessageClientActor>()?;
    hyperactor_mod.add_class::<actor::PythonActorHandle>()?;
    hyperactor_mod.add_class::<actor::PythonMessage>()?;
    hyperactor_mod.add_class::<actor::PythonActorHandle>()?;

    hyperactor_mod.add_class::<mailbox::PyMailbox>()?;
    hyperactor_mod.add_class::<mailbox::PyPortId>()?;
    hyperactor_mod.add_class::<mailbox::PythonPortHandle>()?;
    hyperactor_mod.add_class::<mailbox::PythonPortReceiver>()?;
    hyperactor_mod.add_class::<mailbox::PythonOncePortHandle>()?;
    hyperactor_mod.add_class::<mailbox::PythonOncePortReceiver>()?;

    hyperactor_mod.add_class::<alloc::PyProcessAllocator>()?;
    hyperactor_mod.add_class::<alloc::PyProcessAlloc>()?;
    hyperactor_mod.add_class::<alloc::PyLocalAllocator>()?;
    hyperactor_mod.add_class::<alloc::PyLocalAlloc>()?;

    hyperactor_mod.add_class::<proc_mesh::PyProcMesh>()?;
    hyperactor_mod.add_class::<actor_mesh::PythonActorMesh>()?;
    hyperactor_mod.add_class::<shape::PyShape>()?;

    // Register common types
    hyperactor_extension::init_pymodule(&hyperactor_mod)?;

    module.add_submodule(&hyperactor_mod)?;
    Ok(())
}
