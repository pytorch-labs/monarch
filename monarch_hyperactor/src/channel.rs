/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use pyo3::prelude::*;

/// Python binding for [`hyperactor::channel::ChannelTransport`]
#[pyclass(
    name = "ChannelTransport",
    module = "monarch._rust_bindings.monarch_hyperactor.channel",
    eq
)]
#[derive(PartialEq, Clone, Copy)]
pub enum PyChannelTransport {
    Tcp,
    MetaTls,
    Local,
    Unix,
    // Sim(/*proxy address:*/ ChannelAddr), TODO kiuk@ add support
}

#[pyclass(
    name = "ChannelAddr",
    module = "monarch._rust_bindings.monarch_hyperactor.channel"
)]
pub struct PyChannelAddr;

#[pymethods]
impl PyChannelAddr {
    /// Returns an "any" address for the given transport type.
    /// Primarily used to bind servers
    #[staticmethod]
    fn any(transport: PyChannelTransport) -> PyResult<String> {
        Ok(ChannelAddr::any(transport.into()).to_string())
    }
}

impl From<PyChannelTransport> for ChannelTransport {
    fn from(val: PyChannelTransport) -> Self {
        match val {
            PyChannelTransport::Tcp => ChannelTransport::Tcp,
            PyChannelTransport::MetaTls => ChannelTransport::MetaTls,
            PyChannelTransport::Local => ChannelTransport::Local,
            PyChannelTransport::Unix => ChannelTransport::Unix,
        }
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyChannelTransport>()?;
    hyperactor_mod.add_class::<PyChannelAddr>()?;
    Ok(())
}
