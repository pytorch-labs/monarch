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

use anyhow::anyhow;
use async_trait::async_trait;
use hyperactor::WorldId;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use hyperactor_extension::alloc::PyAlloc;
use hyperactor_extension::alloc::PyAllocSpec;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::AllocatorError;
use hyperactor_mesh::alloc::LocalAllocator;
use hyperactor_mesh::alloc::ProcessAllocator;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAlloc;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAllocHost;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAllocInitializer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::process::Command;

use crate::channel::PyChannelAddr;
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
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
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

/// A `[hyperactor_mesh::alloc::RemoteProcessAllocInitializer]` wrapper to enable subclassing from Python.
///
/// Basically follows https://pyo3.rs/v0.25.0/trait-bounds.html.
/// The Python subclass should implement `def initialize_alloc(self) -> list[str]`.
pub struct PyRemoteProcessAllocInitializer {
    // instance of a Python subclass of `monarch._rust_bindings.monarch_hyperactor.alloc.RemoteProcessAllocInitializer`.
    py_inner: Py<PyAny>,
}

impl Clone for PyRemoteProcessAllocInitializer {
    fn clone(&self) -> Self {
        Self {
            py_inner: Python::with_gil(|py| Py::clone_ref(&self.py_inner, py)),
        }
    }
}
impl PyRemoteProcessAllocInitializer {
    /// calls the initializer's `initialize_alloc()` as implemented in python
    ///
    /// NOTE: changes to python method calls must be made in sync with
    ///   the method signature of `RemoteAllocInitializer` in
    ///   `monarch/python/monarch/_rust_bindings/monarch_hyperactor/alloc.pyi`
    async fn py_initialize_alloc(&self) -> PyResult<Vec<String>> {
        // call the function as implemented in python
        let future = Python::with_gil(|py| -> PyResult<_> {
            let coroutine = self.py_inner.bind(py).call_method0("initialize_alloc")?;
            pyo3_async_runtimes::tokio::into_future(coroutine)
        })?;

        let addrs = future.await?;
        Python::with_gil(|py| -> PyResult<Vec<String>> { addrs.extract(py) })
    }

    async fn get_transport_and_port(&self) -> PyResult<(ChannelTransport, u16)> {
        // NOTE: the upstream RemoteAllocator APIs take (transport, port, hostnames)
        //   (e.g. assumes the same transport and port for all servers).
        //   Until that is fixed we have to assume the same here.
        //   Get the transport and port from the first address
        // TODO T227130269
        let addrs = self.py_initialize_alloc().await?;
        let addr = addrs
            .first()
            .ok_or_else(|| anyhow!("initializer must return non-empty list of addresses"))?;
        let channel_addr = PyChannelAddr::parse(addr)?;
        let port = channel_addr.get_port()?;
        let transport = channel_addr.get_transport()?;
        Ok((transport.into(), port))
    }
}

#[async_trait]
impl RemoteProcessAllocInitializer for PyRemoteProcessAllocInitializer {
    async fn initialize_alloc(&mut self) -> Result<Vec<RemoteProcessAllocHost>, anyhow::Error> {
        // call the function as implemented in python
        let addrs = self.py_initialize_alloc().await?;
        addrs
            .iter()
            .map(|channel_addr| {
                let addr = ChannelAddr::from_str(channel_addr)?;
                let (id, hostname) = match addr {
                    ChannelAddr::Tcp(socket) => (socket.ip().to_string(), socket.ip().to_string()),
                    ChannelAddr::MetaTls(hostname, _) => (hostname.clone(), hostname.clone()),
                    ChannelAddr::Unix(_) => (addr.to_string(), addr.to_string()),
                    _ => anyhow::bail!("unsupported transport for channel address: `{addr}`"),
                };
                Ok(RemoteProcessAllocHost { id, hostname })
            })
            .collect()
    }
}

#[pyclass(
    name = "RemoteAllocatorBase",
    module = "monarch._rust_bindings.monarch_hyperactor.alloc",
    subclass
)]
#[derive(Clone)]
pub struct PyRemoteAllocator {
    // IMPORTANT: other than the `initializer` this struct should not hold any non-trivially
    //   clonable data (e.g. such that the Clone derive-attribute would not work).
    //   This allows us to avoid having yet-another-wrapper for PyRemoteAllocator since
    //   PyRemoteProcessAllocInitializer is already a wrapper and its wrapped Py<PyAny> is
    //   shared by reference.
    world_id: String,
    initializer: PyRemoteProcessAllocInitializer,
    heartbeat_interval: Duration,
}

#[async_trait]
impl Allocator for PyRemoteAllocator {
    type Alloc = RemoteProcessAlloc;

    async fn allocate(&mut self, spec: AllocSpec) -> Result<Self::Alloc, AllocatorError> {
        let initializer = self.initializer.clone();
        let (transport, port) = initializer
            .get_transport_and_port()
            .await
            .map_err(|e| AllocatorError::Other(e.into()))?;

        let alloc = RemoteProcessAlloc::new(
            spec,
            WorldId(self.world_id.clone()),
            transport,
            port,
            self.heartbeat_interval,
            initializer,
        )
        .await?;
        Ok(alloc)
    }
}

#[pymethods]
impl PyRemoteAllocator {
    #[new]
    #[pyo3(signature = (
        world_id,
        initializer,
        heartbeat_interval = Duration::from_secs(5),
    ))]
    fn new(
        world_id: String,
        initializer: Py<PyAny>,
        heartbeat_interval: Duration,
    ) -> PyResult<Self> {
        Ok(Self {
            world_id,
            initializer: PyRemoteProcessAllocInitializer {
                py_inner: initializer,
            },
            heartbeat_interval,
        })
    }

    fn allocate_nonblocking<'py>(
        &self,
        py: Python<'py>,
        spec: &PyAllocSpec,
    ) -> PyResult<Bound<'py, PyAny>> {
        let spec = spec.inner.clone();
        let mut cloned = self.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            cloned
                .allocate(spec)
                .await
                .map(|alloc| PyAlloc::new(Box::new(alloc)))
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        })
    }
    fn allocate_blocking<'py>(&self, py: Python<'py>, spec: &PyAllocSpec) -> PyResult<PyAlloc> {
        let spec = spec.inner.clone();
        let mut cloned = self.clone();

        signal_safe_block_on(py, async move {
            cloned
                .allocate(spec)
                .await
                .map(|alloc| PyAlloc::new(Box::new(alloc)))
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })?
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyProcessAllocator>()?;
    hyperactor_mod.add_class::<PyLocalAllocator>()?;
    hyperactor_mod.add_class::<PyRemoteAllocator>()?;

    Ok(())
}
