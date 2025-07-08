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
use std::sync::Mutex;
use std::time::Duration;

use anyhow::anyhow;
use async_trait::async_trait;
use hyperactor::WorldId;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use hyperactor_mesh::alloc::Alloc;
use hyperactor_mesh::alloc::AllocConstraints;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::AllocatorError;
use hyperactor_mesh::alloc::LocalAllocator;
use hyperactor_mesh::alloc::ProcState;
use hyperactor_mesh::alloc::ProcessAllocator;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAlloc;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAllocHost;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAllocInitializer;
use ndslice::Shape;
use ndslice::Slice;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::process::Command;

use crate::channel::PyChannelAddr;
use crate::runtime::signal_safe_block_on;

#[pyclass(
    name = "LocalAllocatorBase",
    module = "monarch._src.actor._extension.monarch_hyperactor.alloc",
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
    module = "monarch._src.actor._extension.monarch_hyperactor.alloc",
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
    // instance of a Python subclass of `monarch._src.actor._extension.monarch_hyperactor.alloc.RemoteProcessAllocInitializer`.
    py_inner: Py<PyAny>,

    // allocation constraints passed onto the allocator's allocate call and passed along to python initializer.
    constraints: AllocConstraints,
}

impl PyRemoteProcessAllocInitializer {
    /// calls the initializer's `initialize_alloc()` as implemented in python
    ///
    /// NOTE: changes to python method calls must be made in sync with
    ///   the method signature of `RemoteAllocInitializer` in
    ///   `monarch/python/monarch/_rust_bindings/monarch_hyperactor/alloc.pyi`
    async fn py_initialize_alloc(&self) -> PyResult<Vec<String>> {
        // call the function as implemented in python
        let args = (&self.constraints.match_labels,);
        let future = Python::with_gil(|py| -> PyResult<_> {
            let coroutine = self
                .py_inner
                .bind(py)
                .call_method1("initialize_alloc", args)?;
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
                    ChannelAddr::Tcp(socket) => {
                        if socket.is_ipv6() {
                            // ipv6 addresses need to be wrapped in square-brackets [ipv6_addr]
                            // since the return value here gets concatenated with 'port' to make up a sockaddr
                            let ipv6_addr = format!("[{}]", socket.ip());
                            (ipv6_addr.clone(), ipv6_addr.clone())
                        } else {
                            let ipv4_addr = socket.ip().to_string();
                            (ipv4_addr.clone(), ipv4_addr.clone())
                        }
                    }
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
    module = "monarch._src.actor._extension.monarch_hyperactor.alloc",
    subclass
)]
pub struct PyRemoteAllocator {
    world_id: String,
    initializer: Py<PyAny>,
    heartbeat_interval: Duration,
}

impl Clone for PyRemoteAllocator {
    fn clone(&self) -> Self {
        Self {
            world_id: self.world_id.clone(),
            initializer: Python::with_gil(|py| Py::clone_ref(&self.initializer, py)),
            heartbeat_interval: self.heartbeat_interval.clone(),
        }
    }
}
#[async_trait]
impl Allocator for PyRemoteAllocator {
    type Alloc = RemoteProcessAlloc;

    async fn allocate(&mut self, spec: AllocSpec) -> Result<Self::Alloc, AllocatorError> {
        let py_inner = Python::with_gil(|py| Py::clone_ref(&self.initializer, py));
        let constraints = spec.constraints.clone();
        let initializer = PyRemoteProcessAllocInitializer {
            py_inner,
            constraints,
        };

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
            initializer,
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

/// A python class that wraps a Rust Alloc trait object. It represents what
/// is shown on the python side. Internals are not exposed.
/// It ensures that the Alloc is only used once (i.e. moved) in rust.
#[pyclass(
    name = "Alloc",
    module = "monarch._src.actor._extension.monarch_hyperactor.alloc"
)]
pub struct PyAlloc {
    pub inner: Arc<Mutex<Option<PyAllocWrapper>>>,
}

impl PyAlloc {
    /// Create a new PyAlloc with provided boxed trait.
    pub fn new(inner: Box<dyn Alloc + Sync + Send>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(PyAllocWrapper { inner }))),
        }
    }

    /// Take the internal Alloc object.
    pub fn take(&self) -> Option<PyAllocWrapper> {
        self.inner.lock().unwrap().take()
    }
}

#[pymethods]
impl PyAlloc {
    fn __repr__(&self) -> PyResult<String> {
        let data = self.inner.lock().unwrap();
        match &*data {
            None => Ok("Alloc(None)".to_string()),
            Some(wrapper) => Ok(format!("Alloc({})", wrapper.shape())),
        }
    }
}

/// Internal wrapper to translate from a dyn Alloc to an impl Alloc. Used
/// to support polymorphism in the Python bindings.
pub struct PyAllocWrapper {
    inner: Box<dyn Alloc + Sync + Send>,
}

#[async_trait]
impl Alloc for PyAllocWrapper {
    async fn next(&mut self) -> Option<ProcState> {
        self.inner.next().await
    }

    fn shape(&self) -> &Shape {
        self.inner.shape()
    }

    fn world_id(&self) -> &WorldId {
        self.inner.world_id()
    }

    fn transport(&self) -> ChannelTransport {
        self.inner.transport()
    }

    async fn stop(&mut self) -> Result<(), AllocatorError> {
        self.inner.stop().await
    }
}

#[pyclass(
    name = "AllocConstraints",
    module = "monarch._src.actor._extension.monarch_hyperactor.alloc"
)]
pub struct PyAllocConstraints {
    inner: AllocConstraints,
}

#[pymethods]
impl PyAllocConstraints {
    #[new]
    #[pyo3(signature = (match_labels=None))]
    fn new(match_labels: Option<HashMap<String, String>>) -> PyResult<Self> {
        let mut constraints = AllocConstraints::default();
        if let Some(match_lables) = match_labels {
            constraints.match_labels = match_lables;
        }

        Ok(Self { inner: constraints })
    }
}

#[pyclass(
    name = "AllocSpec",
    module = "monarch._src.actor._extension.monarch_hyperactor.alloc"
)]
pub struct PyAllocSpec {
    pub inner: AllocSpec,
}

#[pymethods]
impl PyAllocSpec {
    #[new]
    #[pyo3(signature = (constraints, **kwargs))]
    fn new(constraints: &PyAllocConstraints, kwargs: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(kwargs) = kwargs else {
            return Err(PyValueError::new_err(
                "Shape must have at least one dimension",
            ));
        };
        let shape_dict = kwargs.downcast::<PyDict>()?;

        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (key, value) in shape_dict {
            keys.push(key.clone());
            values.push(value.clone());
        }

        let shape = Shape::new(
            keys.into_iter()
                .map(|key| key.extract::<String>())
                .collect::<PyResult<Vec<String>>>()?,
            Slice::new_row_major(
                values
                    .into_iter()
                    .map(|key| key.extract::<usize>())
                    .collect::<PyResult<Vec<usize>>>()?,
            ),
        )
        .map_err(|e| PyValueError::new_err(format!("Invalid shape: {:?}", e)))?;

        Ok(Self {
            inner: AllocSpec {
                shape,
                constraints: constraints.inner.clone(),
            },
        })
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyProcessAllocator>()?;
    hyperactor_mod.add_class::<PyLocalAllocator>()?;
    hyperactor_mod.add_class::<PyRemoteAllocator>()?;
    hyperactor_mod.add_class::<PyAlloc>()?;
    hyperactor_mod.add_class::<PyAllocConstraints>()?;
    hyperactor_mod.add_class::<PyAllocSpec>()?;

    Ok(())
}
