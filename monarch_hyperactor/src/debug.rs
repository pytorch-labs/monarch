/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor_mesh::debug::EXTERNAL_DEBUG_ROUTER_ID;
use pyo3::prelude::*;

use crate::channel::PyChannelAddr;
use crate::mailbox::Instance;
use crate::mailbox::PyMailbox;
use crate::proc::PyActorId;
use crate::runtime::signal_safe_block_on;

#[pyfunction]
fn debug_cli_client(
    py: Python<'_>,
    server_addr: &PyChannelAddr,
    listen_addr: &PyChannelAddr,
) -> PyResult<Instance> {
    let server_addr = server_addr.inner.clone();
    let listen_addr = listen_addr.inner.clone();
    Ok(Instance::from(signal_safe_block_on(py, async {
        hyperactor_mesh::debug::debug_cli_client(server_addr, listen_addr).await
    })??))
}

#[pyfunction]
fn get_external_debug_router_id() -> PyResult<PyActorId> {
    Ok(PyActorId {
        inner: EXTERNAL_DEBUG_ROUTER_ID.clone(),
    })
}

#[pyfunction]
fn bind_debug_cli_actor(
    debug_cli_actor_id: PyActorId,
    response_addr: &PyChannelAddr,
) -> PyResult<()> {
    hyperactor_mesh::debug::bind_debug_cli_actor(
        debug_cli_actor_id.inner.clone(),
        response_addr.inner.clone(),
    )
    .map_err(PyErr::from)
}

#[pyfunction]
fn init_debug_server(
    py: Python<'_>,
    debug_controller_mailbox: PyMailbox,
    listen_addr: &PyChannelAddr,
) -> PyResult<()> {
    let listen_addr = listen_addr.inner.clone();
    Ok(signal_safe_block_on(py, async {
        hyperactor_mesh::debug::init_debug_server(debug_controller_mailbox.inner, listen_addr).await
    })??)
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    fn wrap_pyfunction(
        hy_mod: &Bound<'_, PyModule>,
        f: Bound<'_, pyo3::types::PyCFunction>,
    ) -> PyResult<()> {
        f.setattr(
            "__module__",
            "monarch._rust_bindings.monarch_hyperactor.debug",
        )?;
        hy_mod.add_function(f)?;
        Ok(())
    };

    wrap_pyfunction(
        hyperactor_mod,
        wrap_pyfunction!(debug_cli_client, hyperactor_mod)?,
    )?;
    wrap_pyfunction(
        hyperactor_mod,
        wrap_pyfunction!(get_external_debug_router_id, hyperactor_mod)?,
    )?;
    wrap_pyfunction(
        hyperactor_mod,
        wrap_pyfunction!(bind_debug_cli_actor, hyperactor_mod)?,
    )?;
    wrap_pyfunction(
        hyperactor_mod,
        wrap_pyfunction!(init_debug_server, hyperactor_mod)?,
    )?;
    Ok(())
}
