use std::sync::Arc;
use std::time::Duration;

use hyperactor::channel::ChannelTransport;
use hyperactor_extension::alloc::PyAlloc;
use hyperactor_extension::alloc::PyAllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_meta_lib::alloc::ALLOC_LABEL_TASK_GROUP;
use hyperactor_meta_lib::alloc::DEFAULT_REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL;
use hyperactor_meta_lib::alloc::DEFAULT_REMOTE_ALLOCATOR_PORT;
use hyperactor_meta_lib::alloc::MastAllocator;
use hyperactor_meta_lib::alloc::MastAllocatorConfig;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass(
    name = "MastAllocatorConfig",
    module = "monarch_meta._monarch_meta.hyperactor_meta"
)]
#[derive(Clone)]
pub struct PyMastAllocatorConfig {
    pub(crate) inner: MastAllocatorConfig,
}

#[pymethods]
impl PyMastAllocatorConfig {
    #[classattr]
    const DEFAULT_REMOTE_ALLOCATOR_PORT: u16 = DEFAULT_REMOTE_ALLOCATOR_PORT;
    #[classattr]
    const DEFAULT_REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL: Duration =
        DEFAULT_REMOTE_ALLOCATOR_HEARTBEAT_INTERVAL;

    #[new]
    #[pyo3(signature = (job_name=None, transport=None, remote_allocator_port=None))]
    fn new(
        job_name: Option<String>,
        transport: Option<String>,
        remote_allocator_port: Option<u16>,
    ) -> PyResult<Self> {
        let mut config = MastAllocatorConfig {
            job_name,
            ..Default::default()
        };
        if let Some(transport) = transport {
            config.transport = match transport.to_lowercase().as_str() {
                "metatls" => ChannelTransport::MetaTls,
                "tcp" => ChannelTransport::Tcp,
                _ => Err(anyhow::anyhow!("unsupported transport: {}", transport))?,
            }
        }
        if let Some(remote_allocator_port) = remote_allocator_port {
            config.remote_allocator_port = remote_allocator_port;
        }

        Ok(Self { inner: config })
    }
}

#[pyclass(
    name = "MastAllocator",
    module = "monarch_meta._monarch_meta.hyperactor_meta"
)]
#[derive(Clone)]
pub struct PyMastAllocator {
    pub(crate) inner: Arc<tokio::sync::Mutex<MastAllocator>>,
}

#[pymethods]
impl PyMastAllocator {
    #[classattr]
    const ALLOC_LABEL_TASK_GROUP: &'static str = ALLOC_LABEL_TASK_GROUP;

    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyMastAllocatorConfig>) -> PyResult<Self> {
        let config = match config {
            Some(config) => config.inner,
            None => MastAllocatorConfig::default(),
        };
        MastAllocator::new(config)
            .map(|inner| Self {
                inner: Arc::new(tokio::sync::Mutex::new(inner)),
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }

    fn allocate<'py>(&self, py: Python<'py>, spec: &PyAllocSpec) -> PyResult<Bound<'py, PyAny>> {
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
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        })
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMastAllocatorConfig>()?;
    module.add_class::<PyMastAllocator>()?;
    Ok(())
}
