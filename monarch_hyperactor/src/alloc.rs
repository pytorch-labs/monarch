/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use hyperactor::WorldId;
use hyperactor::channel::ChannelAddr;
use hyperactor_extension::alloc::PyAlloc;
use hyperactor_extension::alloc::PyAllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::LocalAllocator;
use hyperactor_mesh::alloc::ProcessAllocator;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAlloc;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAllocHost;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAllocInitializer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::process::Command;

use crate::runtime::signal_safe_block_on;

#[pyclass(
    name = "LocalAllocatorBase",
    module = "monarch._rust_bindings.monarch_hyperactor.alloc",
    subclass
)]
pub struct PyLocalAllocator;

#[pymethods]
impl PyLocalAllocator {
    #[new]
    fn new() -> Self {
        PyLocalAllocator {}
    }

    fn allocate_nonblocking<'py>(
        &self,
        py: Python<'py>,
        spec: &PyAllocSpec,
    ) -> PyResult<Bound<'py, PyAny>> {
        // We could use Bound here, and acquire the GIL inside of `future_into_py`, but
        // it is rather awkward with the current APIs, and we can anyway support Arc/Mutex
        // pretty easily.
        let spec = spec.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalAllocator
                .allocate(spec)
                .await
                .map(|inner| PyAlloc::new(Box::new(inner)))
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })
    }

    fn allocate_blocking<'py>(&self, py: Python<'py>, spec: &PyAllocSpec) -> PyResult<PyAlloc> {
        // We could use Bound here, and acquire the GIL inside of
        // `signal_safe_block_on`, but it is rather awkward with the current
        // APIs, and we can anyway support Arc/Mutex pretty easily.
        let spec = spec.inner.clone();
        signal_safe_block_on(py, async move {
            LocalAllocator
                .allocate(spec)
                .await
                .map(|inner| PyAlloc::new(Box::new(inner)))
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })?
    }
}

#[pyclass(
    name = "ProcessAllocatorBase",
    module = "monarch._rust_bindings.monarch_hyperactor.alloc",
    subclass
)]
pub struct PyProcessAllocator {
    inner: Arc<tokio::sync::Mutex<ProcessAllocator>>,
}

#[pymethods]
impl PyProcessAllocator {
    #[new]
    #[pyo3(signature = (cmd, args=None, env=None))]
    fn new(cmd: String, args: Option<Vec<String>>, env: Option<HashMap<String, String>>) -> Self {
        let mut cmd = Command::new(cmd);
        if let Some(args) = args {
            cmd.args(args);
        }
        if let Some(env) = env {
            cmd.envs(env);
        }
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(ProcessAllocator::new(cmd))),
        }
    }

    fn allocate_nonblocking<'py>(
        &self,
        py: Python<'py>,
        spec: &PyAllocSpec,
    ) -> PyResult<Bound<'py, PyAny>> {
        // We could use Bound here, and acquire the GIL inside of `future_into_py`, but
        // it is rather awkward with the current APIs, and we can anyway support Arc/Mutex
        // pretty easily.
        let instance = Arc::clone(&self.inner);
        let spec = spec.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            instance
                .lock()
                .await
                .allocate(spec)
                .await
                .map(|inner| PyAlloc::new(Box::new(inner)))
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })
    }

    fn allocate_blocking<'py>(&self, py: Python<'py>, spec: &PyAllocSpec) -> PyResult<PyAlloc> {
        // We could use Bound here, and acquire the GIL inside of
        // `signal_safe_block_on`, but it is rather awkward with the current
        // APIs, and we can anyway support Arc/Mutex pretty easily.
        let instance = Arc::clone(&self.inner);
        let spec = spec.inner.clone();
        signal_safe_block_on(py, async move {
            instance
                .lock()
                .await
                .allocate(spec)
                .await
                .map(|inner| PyAlloc::new(Box::new(inner)))
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })?
    }
}

struct PyRemoteProcessAllocInitializer {
    addrs: Vec<String>,
}

#[async_trait]
impl RemoteProcessAllocInitializer for PyRemoteProcessAllocInitializer {
    async fn initialize_alloc(&mut self) -> Result<Vec<RemoteProcessAllocHost>, anyhow::Error> {
        self.addrs
            .iter()
            .map(|channel_addr| {
                let addr = ChannelAddr::from_str(channel_addr)?;
                let remote_host = match addr {
                    ChannelAddr::Tcp(socket_addr) => RemoteProcessAllocHost {
                        id: socket_addr.ip().to_string(),
                        hostname: socket_addr.ip().to_string(),
                    },
                    ChannelAddr::MetaTls(hostname, _) => RemoteProcessAllocHost {
                        id: hostname.clone(),
                        hostname: hostname.clone(),
                    },
                    ChannelAddr::Unix(_) => RemoteProcessAllocHost {
                        id: addr.to_string(),
                        hostname: addr.to_string(),
                    },
                    _ => {
                        anyhow::bail!("Unsupported transport for channel address: `{addr:?}`")
                    }
                };
                Ok(remote_host)
            })
            .collect()
    }
}

#[pyclass(
    name = "RemoteAllocatorBase",
    module = "monarch._rust_bindings.monarch_hyperactor.alloc",
    subclass
)]
pub struct PyRemoteAllocator {
    world_id: String,
    addrs: Vec<String>,
    heartbeat_interval_millis: u64,
}

const DEFAULT_REMOTE_ALLOCATOR_PORT: u16 = 26600;
const DEFAULT_REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL_MILLIS: u64 = 5000;

#[pymethods]
impl PyRemoteAllocator {
    #[classattr]
    const DEFAULT_PORT: u16 = DEFAULT_REMOTE_ALLOCATOR_PORT;

    #[classattr]
    const DEFAULT_HEARTBEAT_INTERVAL_MILLIS: u64 =
        DEFAULT_REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL_MILLIS;

    #[new]
    #[pyo3(signature = (
        world_id,
        addrs,
        heartbeat_interval_millis = DEFAULT_REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL_MILLIS,
    ))]
    fn new(world_id: String, addrs: Vec<String>, heartbeat_interval_millis: u64) -> PyResult<Self> {
        Ok(Self {
            world_id,
            addrs,
            heartbeat_interval_millis,
        })
    }

    fn allocate_nonblocking<'py>(
        &self,
        py: Python<'py>,
        spec: &PyAllocSpec,
    ) -> PyResult<Bound<'py, PyAny>> {
        let addrs = self.addrs.clone();
        let world_id = self.world_id.clone();
        let spec_inner = spec.inner.clone();
        let heartbeat_interval_millis = self.heartbeat_interval_millis;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // all addrs expected to have the same transport; use the first one
            let first_addr = addrs.first().expect("addrs should not be empty");
            let first_addr = ChannelAddr::from_str(first_addr)?;
            let transport = first_addr.transport();
            let port = match first_addr {
                ChannelAddr::Tcp(socket_addr) => socket_addr.port(),
                ChannelAddr::MetaTls(_, port) => port,
                ChannelAddr::Unix(_) => 0,
                ChannelAddr::Local(_) => 0,
                ChannelAddr::Sim(_) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Unsupported channel_addr: {first_addr:?}"
                    )));
                }
            };

            let alloc = RemoteProcessAlloc::new(
                spec_inner,
                WorldId(world_id),
                transport,
                port,
                Duration::from_millis(heartbeat_interval_millis),
                PyRemoteProcessAllocInitializer { addrs },
            )
            .await?;

            Ok(PyAlloc::new(Box::new(alloc)))
        })
    }
    fn allocate_blocking<'py>(&self, py: Python<'py>, spec: &PyAllocSpec) -> PyResult<PyAlloc> {
        let addrs = self.addrs.clone();
        let world_id = self.world_id.clone();
        let spec_inner = spec.inner.clone();
        let heartbeat_interval_millis = self.heartbeat_interval_millis;

        signal_safe_block_on(py, async move {
            // all addrs expected to have the same transport; use the first one
            let first_addr = addrs.first().expect("addrs should not be empty");
            let first_addr = ChannelAddr::from_str(first_addr)?;
            let transport = first_addr.transport();
            let port = match first_addr {
                ChannelAddr::Tcp(socket_addr) => socket_addr.port(),
                ChannelAddr::MetaTls(_, port) => port,
                ChannelAddr::Unix(_) => 0,
                ChannelAddr::Local(_) => 0,
                ChannelAddr::Sim(_) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Unsupported channel_addr: {first_addr:?}"
                    )));
                }
            };

            let alloc = RemoteProcessAlloc::new(
                spec_inner,
                WorldId(world_id),
                transport,
                port,
                Duration::from_millis(heartbeat_interval_millis),
                PyRemoteProcessAllocInitializer { addrs },
            )
            .await?;

            Ok(PyAlloc::new(Box::new(alloc)))
        })?
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyProcessAllocator>()?;
    hyperactor_mod.add_class::<PyLocalAllocator>()?;
    hyperactor_mod.add_class::<PyRemoteAllocator>()?;

    Ok(())
}
