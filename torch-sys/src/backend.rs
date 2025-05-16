use std::sync::OnceLock;

use anyhow::Result;
use async_trait::async_trait;
use cxx::CxxVector;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::runtime::Handle;

use crate::Tensor;
use crate::bridge::ffi;
pub use crate::bridge::ffi::AllToAllOptions;
pub use crate::bridge::ffi::AllreduceOptions;
pub use crate::bridge::ffi::BarrierOptions;
pub use crate::bridge::ffi::BroadcastOptions;
pub use crate::bridge::ffi::GatherOptions;
pub use crate::bridge::ffi::ReduceOp;
pub use crate::bridge::ffi::ReduceOptions;
pub use crate::bridge::ffi::ReduceScatterOptions;
pub use crate::bridge::ffi::ScatterOptions;

static REGISTER: OnceLock<()> = OnceLock::new();
static INIT: OnceLock<()> = OnceLock::new();

#[async_trait]
pub trait Work: Sync + Send + 'static {
    type Error;
    async fn wait(&self) -> Result<(), Self::Error>;
    async fn is_completed(&self) -> Result<bool, Self::Error>;
}

/// A wrapper around the `Backend` trait to make it usable from the C++ bridge.
pub(crate) struct BoxedWork(pub Box<dyn Work<Error = anyhow::Error>>);

// TODO(agallagher): Support the `Work` return type for async -- currently we
// do everything sync and bridge into async code.
impl BoxedWork {
    pub fn wait(&self) -> Result<(), anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Handle::current().block_on(self.0.wait())
    }
    pub fn is_completed(&self) -> Result<bool, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Handle::current().block_on(self.0.is_completed())
    }
}

// A Rust translation of the `Backend` base class in torch, used to create custom
// process group backends:
// https://github.com/pytorch/pytorch/blob/6178be822dc3fb307e950c337876f05dd63582b2/torch/csrc/distributed/c10d/Backend.hpp#L20
#[async_trait]
pub trait Backend: Sync + Send + 'static {
    type Error;
    async fn allreduce(
        &self,
        tensors: &CxxVector<Tensor>,
        opts: AllreduceOptions,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn allgather(
        &self,
        output: &CxxVector<Tensor>,
        input: &Tensor,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn _allgather_base(
        &self,
        output: &Tensor,
        input: &Tensor,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn barrier(
        &self,
        opts: BarrierOptions,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn reduce(
        &self,
        input: &Tensor,
        opts: ReduceOptions,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn _reduce_scatter_base(
        &self,
        output: &Tensor,
        input: &Tensor,
        opts: ReduceScatterOptions,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn send(
        &self,
        tensors: &CxxVector<Tensor>,
        dst_rank: i32,
        tag: i32,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn recv(
        &self,
        tensors: &CxxVector<Tensor>,
        src_rank: i32,
        tag: i32,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn gather(
        &self,
        outputs: &CxxVector<Tensor>,
        input: &Tensor,
        opts: GatherOptions,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn scatter(
        &self,
        output: &Tensor,
        inputs: &CxxVector<Tensor>,
        opts: ScatterOptions,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn broadcast(
        &self,
        tensors: &CxxVector<Tensor>,
        opts: BroadcastOptions,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn alltoall_base(
        &self,
        output_buffer: &Tensor,
        input_buffer: &Tensor,
        opts: AllToAllOptions,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
    async fn alltoall(
        &self,
        output_tensors: &CxxVector<Tensor>,
        input_tensors: &CxxVector<Tensor>,
        opts: AllToAllOptions,
    ) -> Result<Box<dyn Work<Error = Self::Error>>, Self::Error>;
}

/// A wrapper around the `Backend` trait to make it usable from the C++ bridge.
pub(crate) struct BoxedBackend(pub Box<dyn Backend<Error = anyhow::Error>>);

impl BoxedBackend {
    pub fn allreduce(
        &self,
        tensors: &CxxVector<Tensor>,
        opts: AllreduceOptions,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(
            Handle::current().block_on(self.0.allreduce(tensors, opts))?,
        )))
    }

    pub fn allgather(
        &self,
        output: &CxxVector<Tensor>,
        input: &Tensor,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(
            Handle::current().block_on(self.0.allgather(output, input))?,
        )))
    }

    pub fn _allgather_base(
        &self,
        output: &Tensor,
        input: &Tensor,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(
            Handle::current().block_on(self.0._allgather_base(output, input))?,
        )))
    }

    pub fn barrier(&self, opts: BarrierOptions) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(
            Handle::current().block_on(self.0.barrier(opts))?,
        )))
    }

    pub fn reduce(
        &self,
        input: &Tensor,
        opts: ReduceOptions,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(
            Handle::current().block_on(self.0.reduce(input, opts))?,
        )))
    }

    pub fn _reduce_scatter_base(
        &self,
        output: &Tensor,
        input: &Tensor,
        opts: ReduceScatterOptions,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(Handle::current().block_on(
            self.0._reduce_scatter_base(output, input, opts),
        )?)))
    }

    pub fn send(
        &self,
        tensors: &CxxVector<Tensor>,
        dst_rank: i32,
        tag: i32,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(
            Handle::current().block_on(self.0.send(tensors, dst_rank, tag))?,
        )))
    }

    pub fn recv(
        &self,
        tensors: &CxxVector<Tensor>,
        src_rank: i32,
        tag: i32,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(
            Handle::current().block_on(self.0.recv(tensors, src_rank, tag))?,
        )))
    }

    pub fn gather(
        &self,
        outputs: &CxxVector<Tensor>,
        input: &Tensor,
        opts: GatherOptions,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(
            Handle::current().block_on(self.0.gather(outputs, input, opts))?,
        )))
    }

    pub fn scatter(
        &self,
        output: &Tensor,
        inputs: &CxxVector<Tensor>,
        opts: ScatterOptions,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        Ok(Box::new(BoxedWork(
            Handle::current().block_on(self.0.scatter(output, inputs, opts))?,
        )))
    }

    pub fn broadcast(
        &self,
        tensors: &CxxVector<Tensor>,
        opts: BroadcastOptions,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(
            Handle::current().block_on(self.0.broadcast(tensors, opts))?,
        )))
    }

    pub fn alltoall_base(
        &self,
        output_buffer: &Tensor,
        input_buffer: &Tensor,
        opts: AllToAllOptions,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(Handle::current().block_on(
            self.0.alltoall_base(output_buffer, input_buffer, opts),
        )?)))
    }

    pub fn alltoall(
        &self,
        output_tensors: &CxxVector<Tensor>,
        input_tensors: &CxxVector<Tensor>,
        opts: AllToAllOptions,
    ) -> Result<Box<BoxedWork>, anyhow::Error> {
        // Re-enter the parents runtime to run async code.
        Ok(Box::new(BoxedWork(Handle::current().block_on(
            self.0.alltoall(output_tensors, input_tensors, opts),
        )?)))
    }
}

fn register(py: Python<'_>) -> PyResult<()> {
    // Import torch.distributed module
    let module = py.import_bound("torch.distributed")?;

    // Get the register_backend attribute from Backend
    let backend = module.getattr("Backend")?;
    let register_backend = backend.getattr("register_backend")?;

    // Create a Python callable from our Rust function
    let create_backend = ffi::create_monarch_backend().into_py(py);

    // We use the extended API so that callers can pass in the inner, pre-
    // initialized backend via `pg_options`.
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("devices", vec!["cuda"])?;
    kwargs.set_item("extended_api", true)?;

    // Register the backend
    register_backend
        .call(("monarch", create_backend), Some(&kwargs))
        .inspect_err(|e| tracing::error!("failed init backend: {}", e))?;

    Ok(())
}

fn init_process_group(py: Python<'_>, world_size: usize, rank: usize) -> PyResult<()> {
    let torchd = py.import_bound("torch.distributed")?;

    // Get the register_backend attribute from Backend
    let backend = torchd.getattr("Backend")?;
    let register_backend = backend.getattr("register_backend")?;

    // Create a Python callable from our Rust function
    let create_backend = ffi::create_null_backend().into_py(py);

    // We use the extended API so that callers can pass in the inner, pre-
    // initialized backend via `pg_options`.
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("extended_api", true)?;

    // Register the backend
    register_backend
        .call(("null", create_backend), Some(&kwargs))
        .inspect_err(|e| tracing::error!("failed init backend: {}", e))?;

    // Init the process group.
    let kwargs = PyDict::new_bound(py);
    // Use a special noop backend that errors out if it's actually used.
    kwargs.set_item("backend", "null")?;
    kwargs.set_item("rank", rank)?;
    // Since the communicator we give it is pre-initialized, we don't acually
    // end up using the store, but the `init_process_group` requires that one
    // is passed in.
    kwargs.set_item("store", torchd.call_method1("FileStore", ("/dev/null",))?)?;
    kwargs.set_item("world_size", world_size)?;

    torchd.call_method("init_process_group", (), Some(&kwargs))?;

    Ok(())
}

pub fn ensure_init_process_group(py: Python<'_>, world_size: usize, rank: usize) -> PyResult<()> {
    py.allow_threads(move || {
        INIT.get_or_try_init(move || {
            Python::with_gil(|py| init_process_group(py, world_size, rank))
        })
    })?;
    Ok(())
}

pub fn new_group<'py, B: Backend<Error = anyhow::Error>>(
    py: Python<'py>,
    ranks: Vec<usize>,
    backend: B,
) -> PyResult<Bound<'py, PyAny>> {
    // Make sure we've registered the monarch backend.
    py.allow_threads(|| REGISTER.get_or_try_init(|| Python::with_gil(register)))?;

    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("backend", "monarch")?;
    kwargs.set_item("ranks", ranks)?;
    kwargs.set_item(
        "pg_options",
        Box::into_raw(Box::new(BoxedBackend(Box::new(backend)))) as u64,
    )?;

    let torchd = py.import_bound("torch.distributed")?;

    torchd.call_method("new_group", (), Some(&kwargs))
}
