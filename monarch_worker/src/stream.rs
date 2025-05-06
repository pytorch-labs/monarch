use std::cell::OnceCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::future::Future;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use anyhow::ensure;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::PortHandle;
use hyperactor::actor::ActorHandle;
use hyperactor::data::Serialized;
use hyperactor::forward;
use hyperactor::mailbox::Mailbox;
use hyperactor::mailbox::OncePortHandle;
use hyperactor::mailbox::PortReceiver;
use hyperactor::proc::Proc;
use monarch_messages::controller::ControllerMessageClient;
use monarch_messages::controller::Seq;
use monarch_messages::controller::WorkerError;
use monarch_messages::worker::CallFunctionError;
use monarch_messages::worker::CallFunctionParams;
use monarch_messages::worker::StreamRef;
use monarch_types::PyTree;
use monarch_types::SerializablePyErr;
use monarch_types::TryIntoPyObjectUnsafe;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use tokio::runtime::Handle;
use tokio::task::JoinHandle;
use torch_sys::BorrowType;
use torch_sys::CudaDevice;
use torch_sys::MultiBorrow;
use torch_sys::RValue;
use torch_sys::TensorCell;
use torch_sys::cuda::Event;
use torch_sys::cuda::Stream;
use torch_sys::deep_clone;
use torch_sys::factory_empty;
use torch_sys::factory_zeros;
use tracing_subscriber::fmt::Subscriber;

use crate::ControllerActor;
use crate::DeviceMesh;
use crate::Factory;
use crate::Reduction;
use crate::Ref;
use crate::ResolvableFunction;
use crate::StreamCreationMode;
use crate::WireValue;
use crate::comm::CommBackend;
use crate::comm::CommMessage;
use crate::comm::CommMessageClient;
use crate::comm::NcclCommActor;
use crate::pipe::PipeMessage;

pub type TensorCellResult = Result<TensorCell, Arc<CallFunctionError>>;

// These thread locals are accessed by the python runtime for debugging sessions.
thread_local! {
    pub static CONTROLLER_ACTOR_REF: OnceCell<ActorRef<ControllerActor>> = const { OnceCell::new() };
    pub static PROC: OnceCell<Proc> = const { OnceCell::new() };
    pub static ROOT_ACTOR_ID: OnceCell<ActorId> = const { OnceCell::new() };
}

#[derive(Debug)]
struct Recording {
    // TODO(slurye): Use this field in a future diff.
    #[allow(dead_code)]
    messages: Vec<StreamMessage>,
}

impl Recording {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }
}

#[derive(Debug, PartialEq)]
enum RecordingState {
    Defining {
        recording: Ref,
        // Set of borrow ids used to track proper borrow usage inside
        // a recording.
        defined_borrows: HashSet<u64>,
    },
    // TODO(slurye): Use this variant in a futrue diff.
    #[allow(dead_code)]
    Running,
}

/// Messages handled by the stream. Generally these are stream-local versions of
/// [`crate::WorkerMessage`].
#[derive(Handler, HandleClient, Debug, Named)]
#[named(dump = false)]
pub enum StreamMessage {
    CallFunction(
        CallFunctionParams,
        HashMap<Ref, DeviceMesh>,
        HashMap<Ref, (DeviceMesh, Vec<String>, Arc<ActorHandle<NcclCommActor>>)>,
    ),

    BorrowCreate {
        /// Id for the borrow.
        borrow: u64,
        /// Tensor to borrow.
        tensor: Ref,
        /// Port for sending the first use CUDA event + borrowed tensor to
        /// the borrower.
        first_use_sender: PortHandle<(Option<Event>, TensorCellResult)>,
    },

    BorrowFirstUse {
        /// Id for the borrow.
        borrow: u64,
        /// Ref for storing the borrowed tensor.
        result: Ref,
        /// Port for receiving the first use CUDA event + borrowed tensor from
        /// the provider stream.
        first_use_receiver: PortReceiver<(Option<Event>, TensorCellResult)>,
    },

    BorrowLastUse {
        /// Id for the borrow.
        borrow: u64,
        /// Ref for the borrowed tensor.
        result: Ref,
        /// Port for sending the last use CUDA event.
        last_use_sender: PortHandle<Option<Event>>,
    },

    BorrowDrop {
        borrow: u64,
        /// Port for receiving the last use CUDA event.
        last_use_receiver: PortReceiver<Option<Event>>,
    },

    DeleteRefs(Vec<Ref>),

    RequestStatus(#[reply] OncePortHandle<()>),

    InitComm(ActorHandle<NcclCommActor>),

    Reduce {
        comm: Arc<ActorHandle<NcclCommActor>>,
        dim_size: i64,
        result: Ref,
        local_tensor: Ref,
        factory: Factory,
        reduction: Reduction,
        scatter: bool,
        in_place: bool,
        out: Option<Ref>,
    },

    SendTensor {
        result: Ref,
        from_rank: Option<usize>,
        to_rank: Option<usize>,
        tensor: Ref,
        factory: Factory,
        comm: Arc<ActorHandle<NcclCommActor>>,
    },

    SendValue {
        seq: Seq,
        worker_actor_id: ActorId,
        mutates: Vec<Ref>,
        function: Option<ResolvableFunction>,
        args: Vec<WireValue>,
        kwargs: HashMap<String, WireValue>,
        device_meshes: HashMap<Ref, DeviceMesh>,
        pipe: Option<PortHandle<PipeMessage>>,
    },

    SetValue {
        results: Vec<Option<Ref>>,
        pipe: Result<PortHandle<PipeMessage>, CallFunctionError>,
    },

    DefineRecording {
        recording: Ref,
    },

    FinalizeRecording {
        recording: Ref,
    },

    CallRecording {
        seq: Seq,
        recording: Ref,
        results: Vec<Ref>,
        actuals: Vec<Ref>,
    },

    RecordingFormal {
        result: Ref,
        argument_index: usize,
    },

    RecordingResult {
        result: Ref,
        output_index: usize,
    },

    SetRefUnitTestsOnly(Ref, WireValue),

    SetTensorRefUnitTestsOnly(Ref, TensorCellResult),

    GetRefUnitTestsOnly(
        Ref, // value
        #[reply] OncePortHandle<Option<Result<WireValue, Arc<CallFunctionError>>>>,
    ),

    GetTensorRefUnitTestsOnly(Ref, #[reply] OncePortHandle<Option<TensorCellResult>>),
}

impl StreamMessage {
    fn clone_for_recording(&self) -> Self {
        match self {
            StreamMessage::RecordingFormal {
                result,
                argument_index,
            } => StreamMessage::RecordingFormal {
                result: *result,
                argument_index: *argument_index,
            },
            StreamMessage::RecordingResult {
                result,
                output_index,
            } => StreamMessage::RecordingResult {
                result: *result,
                output_index: *output_index,
            },
            StreamMessage::DeleteRefs(refs) => StreamMessage::DeleteRefs(refs.clone()),
            other => panic!(
                "StreamMessage variant not supported in recording: {:?}",
                other
            ),
        }
    }

    // Get the set of refs that this message defines.
    fn get_defined_refs(&self) -> HashSet<Ref> {
        match self {
            StreamMessage::RecordingFormal { result, .. } => HashSet::from([*result]),
            // TODO(slurye): Add remaining message types.
            _ => HashSet::new(),
        }
    }

    // Get the set of refs that this message mutates.
    fn get_mutated_refs(&self) -> HashSet<Ref> {
        // TODO(slurye): Add message types that mutate their inputs.
        HashSet::new()
    }
}

/// A stream represents a linear sequence of execution. Operations on different
/// streams can execute concurrently.
///
/// For CUDA operators, streams will invoke the corresponding stream management
/// APIs to perform synchronization.
///
/// For CPU operators, streams will just execute synchronously on their own OS
/// thread.
#[derive(Debug)]
pub struct StreamActor {
    world_size: usize,
    rank: usize,
    /// Mapping of refs in the controller environment to TensorIndex in this
    /// stream's local environment.
    // TODO(agallagher): Use `ValueError` as the error type.
    env: HashMap<Ref, Result<RValue, Arc<CallFunctionError>>>,
    /// How to create the stream.
    creation_mode: StreamCreationMode,
    /// CUDA stream that this actor will enqueue operations on. None if "device"
    /// is not a CUDA device.
    /// NOTE: We lazily create the stream, so that we do it from the dedicated
    /// Stream OS thread as, otherwise, we see deadlocks when done from
    /// unexpected threads.
    cuda_stream: OnceLock<Option<Stream>>,
    /// Device this stream should be scheduled on.
    device: Option<CudaDevice>,
    /// Communicator for this stream. Optional as we lazily initialize it.
    comm: Option<ActorHandle<NcclCommActor>>,
    /// Actor ref of the controller that created this stream.
    controller_actor: ActorRef<ControllerActor>,
    remote_process_groups: HashMap<Ref, PyObject>,
    recordings: HashMap<Ref, Recording>,
    active_recording: Option<RecordingState>,
}

/// Parameters for creating a [`Stream`].
#[derive(Debug, Clone)]
pub struct StreamParams {
    pub world_size: usize,
    pub rank: usize,
    /// Controls how the underlying CUDA stream is created.
    pub creation_mode: StreamCreationMode,
    /// Id of this stream in the worker actor's stream table.
    pub id: StreamRef,
    /// Device this stream should be scheduled on. If none, don't do stream
    /// synchronization.
    pub device: Option<CudaDevice>,
    /// Actor ref of the controller that created this stream.
    pub controller_actor: ActorRef<ControllerActor>,
}

#[async_trait]
impl Actor for StreamActor {
    type Params = StreamParams;
    async fn new(
        StreamParams {
            world_size,
            rank,
            id: _,
            device,
            controller_actor,
            creation_mode,
        }: Self::Params,
    ) -> Result<Self> {
        Ok(Self {
            world_size,
            rank,
            env: HashMap::new(),
            creation_mode,
            cuda_stream: OnceLock::new(),
            device,
            comm: None,
            controller_actor,
            remote_process_groups: HashMap::new(),
            recordings: HashMap::new(),
            active_recording: None,
        })
    }

    async fn init(&mut self, this: &Instance<Self>) -> Result<()> {
        // These thread locals are exposed via python functions, so we need to set them in the
        // same thread that python will run in. That means we need to initialize them here in
        // StreamActor::init instead of in StreamActor::new.
        CONTROLLER_ACTOR_REF.with(|controller_actor_ref| {
            controller_actor_ref.set(self.controller_actor.clone()).ok()
        });
        PROC.with(|proc| proc.set(this.proc().clone()).ok());
        ROOT_ACTOR_ID.with(|root_actor_id| {
            root_actor_id
                .set(ActorId::root(
                    this.self_id().proc_id().clone(),
                    this.self_id().name().to_string(),
                ))
                .ok()
        });
        // Set the current stream for this actor thread.
        if let Some(stream) = self.cuda_stream() {
            Stream::set_current_stream(stream);
        }
        Ok(())
    }

    /// Specialize spawn_server_task for StreamActor, because we want to run the stream on a
    /// dedicated OS thread. This is because:
    ///   - Streams do expensive blocking CPU operations (like calling CPU kernels).
    ///   - Torch/CUDA make use of thread-local state, so moving tasks across
    ///     threads is problematic.
    fn spawn_server_task<F>(future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let (join_tx, join_rx) = tokio::sync::oneshot::channel();
        // It is important that we spawn a standalone thread for the work here,
        // as opposed to using `spawn_blocking` to spawn a tokio-managed thread.
        // This is because the worker stream may call uninterruptible FFI code
        // that can deadlock (CUDA, NCCL).
        // If we use a tokio-managed blocking thread, then runtime teardown will
        // try to wait for tasks on that thread to reach an await point, and
        // hang forever.
        let builder = std::thread::Builder::new().name("worker-stream".to_string());
        let _thread_handle = builder.spawn(move || {
            // Spawn a new thread with a single-threaded tokio runtime to run the
            // actor loop.  We avoid the current-threaded runtime, so that we can
            // use `block_in_place` for nested async-to-sync-to-async flows.
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_io()
                .build()
                .unwrap();
            let result = rt.block_on(async {
                tokio::task::block_in_place(|| {
                    // Allow e.g. destructing py objects on this thread, which
                    // can happen at shutdown when the a stream actors env map
                    // for rvalues is dropped (e.g. P1673311499).
                    // https://github.com/PyO3/pyo3/discussions/3499
                    Python::with_gil(|py| {
                        py.allow_threads(|| {
                            let result = Handle::current().block_on(future);
                            if join_tx.send(result).is_err() {
                                panic!("could not send join result")
                            }
                        })
                    })
                })
            });
            rt.shutdown_timeout(Duration::from_weeks(1));
            result
        });

        // In order to bridge the synchronous join handle with the async world,
        // smuggle the result through a channel.
        tokio::spawn(async move { join_rx.await.unwrap() })
    }
}

/// The arguments we accept as inputs to Python function calls.
#[derive(Debug)]
enum PyArg<'a> {
    RValue(RValue),
    DeviceMesh(&'a DeviceMesh),
    PyObject(PyObject),
}

/// Serialize into a `PyObject`.
impl<'b> TryIntoPyObjectUnsafe<PyAny> for &PyArg<'b> {
    unsafe fn try_to_object_unsafe<'a>(self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        match self {
            // SAFETY: This inherits the unsafety of `rvalue_to_ivalue` (see comment
            // above).
            PyArg::RValue(rval) => unsafe { rval.try_to_object_unsafe(py) },
            PyArg::DeviceMesh(mesh) => Ok(Py::new(py, (*mesh).clone())?.into_bound(py).into_any()),
            PyArg::PyObject(obj) => Ok(obj.clone_ref(py).into_bound(py)),
        }
    }
}

impl StreamActor {
    fn cuda_stream(&self) -> Option<&Stream> {
        self.cuda_stream
            .get_or_init(|| {
                self.device.map(|device| match self.creation_mode {
                    StreamCreationMode::UseDefaultStream => {
                        Stream::get_current_stream_on_device(device)
                    }
                    StreamCreationMode::CreateNewStream => Stream::new_with_device(device),
                })
            })
            .as_ref()
    }

    fn ref_to_rvalue(&self, ref_: &Ref) -> Result<RValue, CallFunctionError> {
        let rvalue = self
            .env
            .get(ref_)
            .ok_or_else(|| CallFunctionError::RefNotFound(*ref_))?;
        match rvalue {
            Ok(val) => Ok(val.clone()),
            Err(err) => {
                let err = err.unwrap_dependent_error().unwrap_or_else(|| err.clone());
                Err(CallFunctionError::DependentError(err))
            }
        }
    }

    fn wire_to_rvalue(&self, value: WireValue) -> Result<RValue, CallFunctionError> {
        let ret = match value {
            WireValue::Ref(val) => self.ref_to_rvalue(&val)?,
            // TODO: We might want to support GenericList / GenericDict etc.
            WireValue::RefList(val) => {
                let mut ret = Vec::with_capacity(val.len());
                for v in val {
                    match self.ref_to_rvalue(&v) {
                        Ok(RValue::Tensor(t)) => ret.push(t),
                        Err(err) => {
                            return Err(err);
                        }
                        Ok(val) => {
                            return Err(CallFunctionError::UnsupportedArgType(
                                "wire_to_rvalue".into(),
                                format!("RefList([{:?}])", val),
                            ));
                        }
                    }
                }
                RValue::TensorList(ret)
            }
            WireValue::Int(val) => RValue::Int(val),
            WireValue::IntList(val) => RValue::IntList(val),
            WireValue::Double(val) => RValue::Double(val),
            WireValue::Bool(val) => RValue::Bool(val),
            WireValue::String(val) => RValue::String(val),
            WireValue::Device(val) => RValue::Device(val),
            WireValue::Layout(val) => RValue::Layout(val),
            WireValue::ScalarType(val) => RValue::ScalarType(val),
            WireValue::MemoryFormat(val) => RValue::MemoryFormat(val),
            WireValue::PyObject(val) => RValue::PyObject(val),
            WireValue::None(()) => RValue::None,
            WireValue::IValue(val) => RValue::Opaque(val.into()),
        };
        Ok(ret)
    }

    fn handle_results(
        &mut self,
        result_refs: &Vec<Option<Ref>>,
        actual_results: Result<Vec<RValue>, CallFunctionError>,
    ) -> Result<(), Arc<CallFunctionError>> {
        // Check if the expected number of returns is correct, otherwise convert
        // into an error.
        let op_results = actual_results.and_then(|actual_results| {
            if result_refs.len() == actual_results.len() {
                Ok(actual_results
                    .into_iter()
                    .zip(result_refs.iter())
                    .filter_map(|(result, ref_)| ref_.map(|ref_| (ref_, result)))
                    .collect::<Vec<(Ref, RValue)>>())
            } else {
                Err(CallFunctionError::UnexpectedNumberOfReturns {
                    expected: result_refs.len(),
                    actual: actual_results.len(),
                })
            }
        });

        // Propagate the results (either the actual values or an error) to the
        // right entries in the global env mapping.
        match op_results {
            Ok(op_results) => {
                for (ref_, rvalue) in op_results.into_iter() {
                    let prev = self.env.insert(ref_, Ok(rvalue));
                    assert!(prev.is_none(), "Duplicate write to reference: {:?}", ref_);
                }
                Ok(())
            }
            Err(err) => {
                let err = Arc::new(err);
                for ref_ in result_refs {
                    match ref_ {
                        Some(ref_) => {
                            let prev = self.env.insert(*ref_, Err(err.clone()));
                            assert!(prev.is_none(), "Duplicate write to reference: {:?}", ref_);
                        }
                        None => {}
                    }
                }
                Err(err)
            }
        }
    }

    fn call_torch_op(
        &self,
        op: String,
        overload: String,
        args: Vec<WireValue>,
        kwargs: HashMap<String, WireValue>,
    ) -> Result<Vec<RValue>, CallFunctionError> {
        let args = args
            .into_iter()
            .map(|arg| self.wire_to_rvalue(arg))
            .collect::<Result<Vec<_>, _>>()?;
        let kwargs = kwargs
            .into_iter()
            .map(|(k, v)| self.wire_to_rvalue(v).map(|rvalue| (k, rvalue)))
            .collect::<Result<HashMap<_, _>, CallFunctionError>>()?;

        let results = torch_sys::call_op::call_op(op, overload, &args, &kwargs, true)?;

        // Handle the case where the op returns nothing and convert it to a list of None.
        // This is to ensure handle results does not error out as the client will call
        // such a function with expected results of size 1.
        Ok(if results.is_empty() {
            vec![RValue::None]
        } else {
            results
        })
    }

    fn call_python_fn(
        &mut self,
        this: &Instance<Self>,
        function: ResolvableFunction,
        args: Vec<WireValue>,
        kwargs: HashMap<String, WireValue>,
        mutates: &[Ref],
        device_meshes: HashMap<Ref, DeviceMesh>,
        remote_process_groups: HashMap<
            Ref,
            (DeviceMesh, Vec<String>, Arc<ActorHandle<NcclCommActor>>),
        >,
    ) -> Result<PyTree<RValue>, CallFunctionError> {
        Python::with_gil(|py| {
            let function = function.resolve(py).map_err(|e| {
                CallFunctionError::InvalidRemoteFunction(format!(
                    "failed to resolve function {}: {}",
                    function, e
                ))
            })?;

            let remote_process_groups = remote_process_groups
                .into_iter()
                .map(|(gref, (mesh, dims, comm))| {
                    let group = match self.remote_process_groups.entry(gref) {
                        Entry::Occupied(ent) => ent.get().clone_ref(py),
                        Entry::Vacant(ent) => {
                            // We need to run `init_process_group` before any
                            // remote process groups can get created.
                            torch_sys::backend::ensure_init_process_group(
                                py,
                                self.world_size,
                                self.rank,
                            )?;

                            // Create a backend object to wrap the comm and use
                            // it to create a new torch group.
                            let ranks = mesh.get_ranks_for_dim_slice(&dims)?;
                            let group_size = ranks.len();
                            let backend = CommBackend::new(
                                comm,
                                Mailbox::new_detached(this.self_id().clone()),
                                self.rank,
                                group_size,
                                self.world_size,
                            );
                            ent.insert(torch_sys::backend::new_group(py, ranks, backend)?.unbind())
                                .clone_ref(py)
                        }
                    };
                    PyResult::Ok((gref, group))
                })
                .collect::<Result<HashMap<_, _>, _>>()
                .map_err(SerializablePyErr::from_fn(py))?;

            // SAFETY: We will be making an unchecked clone of each tensor to pass to to
            // C++, so we need to hold a borrow of each input tensor for the duration of
            // this function.
            let mut multiborrow = MultiBorrow::new();

            let resolve = |val: WireValue| {
                val.into_py_object()
                    .map_err(|e| {
                        CallFunctionError::UnsupportedArgType(
                            format!("{:?}", function),
                            format!("{:?}", e),
                        )
                    })?
                    .unpickle(py)
                    .map_err(SerializablePyErr::from_fn(py))?
                    .extract::<PyTree<PyObject>>()
                    .map_err(SerializablePyErr::from_fn(py))?
                    .try_into_map(|obj| {
                        Ok(if let Ok(ref_) = Ref::from_py_object(obj.bind(py)) {
                            if let Some(mesh) = device_meshes.get(&ref_) {
                                PyArg::DeviceMesh(mesh)
                            } else if let Some(pg) = remote_process_groups.get(&ref_) {
                                PyArg::PyObject(pg.clone_ref(py))
                            } else {
                                let rval = self.ref_to_rvalue(&ref_)?;
                                PyArg::RValue(rval)
                            }
                        } else {
                            PyArg::PyObject(obj)
                        })
                    })
            };

            // Resolve refs
            let py_args: Vec<PyTree<PyArg>> = args
                .into_iter()
                .map(resolve)
                .collect::<Result<_, CallFunctionError>>()?;
            let py_kwargs: HashMap<_, PyTree<PyArg>> = kwargs
                .into_iter()
                .map(|(k, object)| Ok((k, resolve(object)?)))
                .collect::<Result<_, CallFunctionError>>()?;

            // Add a shared-borrow for each rvalue reference.
            py_args
                .iter()
                .chain(py_kwargs.values())
                .flat_map(|o| o.iter())
                .for_each(|arg| {
                    if let PyArg::RValue(rval) = arg {
                        multiborrow.add(rval, BorrowType::Shared);
                    }
                });

            // Add mutable borrows for params we're mutating.
            let mutates: Vec<_> = mutates
                .iter()
                .map(|r| self.ref_to_rvalue(r))
                .collect::<Result<_, CallFunctionError>>()?;
            mutates
                .iter()
                .for_each(|rval| multiborrow.add(rval, BorrowType::Mutable));

            // Execute the borrow.
            let _borrow = multiborrow.borrow()?;

            tracing::debug!(
                "calling python function: {function:?} with args: {py_args:?} and kwargs: {py_kwargs:?}"
            );
            // Call function.
            // Use custom subscriber to route Worker messages to stdout.
            let scoped_subscriber = Subscriber::builder().with_writer(std::io::stdout).finish();
            let result: Bound<'_, PyAny> =
                tracing::subscriber::with_default(scoped_subscriber, || {
                    function
                        .call(
                            // SAFETY: The borrows above guard the unchecked clones done by
                            // `rvalue_to_ivalue`. This may result in multiple mutable
                            // references to tensor data, but the Python side is responsible
                            // for making sure that is safe
                            // TODO(agallagher): The args/kwargs conversion traits generate
                            // the appropriate types here, but they get casted to `PyAny`.
                            // It'd be nice to make `TryToPyObjectUnsafe` take a template
                            // arg for the converted py object to avoid this downcast.
                            unsafe { py_args.try_to_object_unsafe(py) }
                                .map_err(SerializablePyErr::from_fn(py))?,
                            Some(
                                // SAFETY: Same.
                                &unsafe { py_kwargs.try_to_object_unsafe(py) }
                                    .map_err(SerializablePyErr::from_fn(py))?,
                            ),
                        )
                        .map_err(SerializablePyErr::from_fn(py))
                })?;
            tracing::debug!("python function: {function:?} result: {result:?}");

            // Parse the python result as an `Object`, which should preserve the
            // original Python object structure, while providing access to the
            // leaves as `RValue`s.
            Ok(PyTree::<RValue>::extract_bound(&result).map_err(SerializablePyErr::from_fn(py))?)
        })
    }

    /// Retrieve `ref_` or create a fake value with the provided factory if it
    /// is an error. We use this for collective calls, where even if there was
    /// an upstream failure, we still have participate in the collective to
    /// avoid deadlocking the other ranks. It's okay to just put a nonsense
    /// value here of the correct shape; the controller will have been notified
    /// of the upstream failure and will know to ignore everything dependent on
    /// it.
    fn get_or_fake_on_err(&self, ref_: Ref, factory: &Factory) -> Result<TensorCell> {
        let rvalue = self
            .env
            .get(&ref_)
            .ok_or_else(|| anyhow!("tensor not found in stream: {ref_:#?}"))?;

        match rvalue {
            Ok(val) => Ok(val.clone().try_into().map_err(|e| anyhow!("{}", e))?),
            Err(_) => {
                let t = factory_zeros(&factory.size, factory.dtype, factory.layout, factory.device);
                Ok(TensorCell::new(t))
            }
        }
    }

    // TODO(slurye): Use this function in a future diff.
    #[allow(dead_code)]
    fn get_defining_recording(&mut self) -> Option<&mut Recording> {
        self.active_recording
            .as_mut()
            .and_then(|state| match state {
                RecordingState::Defining { recording, .. } => {
                    match self.recordings.get_mut(recording) {
                        Some(recording) => Some(recording),
                        // Panic, because this would be a logic error in the program.
                        None => panic!("recording not found: {:?}", recording),
                    }
                }
                RecordingState::Running => None,
            })
    }
}

#[async_trait]
#[forward(StreamMessage)]
impl StreamMessageHandler for StreamActor {
    async fn call_function(
        &mut self,
        this: &Instance<Self>,
        params: CallFunctionParams,
        device_meshes: HashMap<Ref, DeviceMesh>,
        remote_process_groups: HashMap<
            Ref,
            (DeviceMesh, Vec<String>, Arc<ActorHandle<NcclCommActor>>),
        >,
    ) -> Result<()> {
        params.function.panic_if_requested();

        let actual_results = match params.function.as_torch_op() {
            // Use block-in-place to allow nested callbacks to re-enter the runtime
            // to run async code.
            Some((op, overload)) => tokio::task::block_in_place(|| {
                self.call_torch_op(op, overload, params.args, params.kwargs)
            }),
            _ => {
                // Use block-in-place to allow nested callbacks to re-enter the
                // runtime to run async code.
                tokio::task::block_in_place(|| {
                    self.call_python_fn(
                        this,
                        params.function,
                        params.args,
                        params.kwargs,
                        &params.mutates,
                        device_meshes,
                        remote_process_groups,
                    )
                    // TODO: Currently, we throw away the actual result and just take
                    // the rvalues, but we probably want to fix this at some point.
                    .map(|results| results.into_leaves())
                })
            }
        };

        match self.handle_results(&params.results, actual_results) {
            Ok(()) => Ok(()),
            Err(err) => {
                let err_to_set = match &*err {
                    // Do not send a response message for dependent errors, as the
                    // original error should have already sent a message.
                    CallFunctionError::DependentError(e) => e,
                    // Any other kind of error should send a response message.
                    _ => {
                        let worker_error = WorkerError {
                            backtrace: format!("{err}"),
                            worker_actor_id: this.self_id().clone(),
                        };
                        tracing::error!("{worker_error}");
                        self.controller_actor
                            .remote_function_failed(this, params.seq, worker_error)
                            .await?;
                        &err
                    }
                };
                for ref_ in params.mutates {
                    self.env.insert(ref_, Err(err_to_set.clone()));
                }
                Ok(())
            }
        }
    }

    async fn borrow_create(
        &mut self,
        _this: &Instance<Self>,
        borrow: u64,
        tensor: Ref,
        first_use_sender: PortHandle<(Option<Event>, TensorCellResult)>,
    ) -> Result<()> {
        let rvalue_result = self
            .env
            .get(&tensor)
            .ok_or_else(|| anyhow!("invalid reference for borrow_create: {:#?}", tensor))?;

        let result = match rvalue_result {
            Ok(rvalue) => Ok(rvalue.clone().try_into().map_err(|e| anyhow!("{}", e))?),
            Err(e) => Err(e.clone()),
        };

        let event = self.cuda_stream().map(|stream| stream.record_event(None));
        first_use_sender.send((event, result)).map_err(|err| {
            anyhow!(
                "failed sending first use event for borrow {:?}: {:?}",
                borrow,
                err
            )
        })
    }

    async fn borrow_first_use(
        &mut self,
        _this: &Instance<Self>,
        borrow: u64,
        result: Ref,
        first_use_receiver: PortReceiver<(Option<Event>, TensorCellResult)>,
    ) -> Result<()> {
        let mut first_use_receiver = first_use_receiver;
        let (first_use_event, cell) = first_use_receiver.recv().await.map_err(|err| {
            anyhow!(
                "failed receiving first use event for borrow {:?}: {:?}",
                borrow,
                err
            )
        })?;

        if let Some(stream) = self.cuda_stream() {
            stream.wait_event(
                &mut first_use_event.expect("sent borrow to CUDA stream, expected a CUDA event"),
            );
        }
        match cell {
            Ok(cell) => {
                self.env.insert(result, Ok(cell.into()));
            }
            Err(err) => {
                self.env.insert(result, Err(err.clone()));
            }
        }
        Ok(())
    }

    async fn borrow_last_use(
        &mut self,
        _this: &Instance<Self>,
        borrow: u64,
        result: Ref,
        last_use_sender: PortHandle<Option<Event>>,
    ) -> Result<()> {
        let event = self.cuda_stream().map(|stream| stream.record_event(None));
        let _ = self.env.remove(&result).ok_or(anyhow!(
            "Invalid reference for borrow_last_use: {result:#?}"
        ))?;

        last_use_sender.send(event).map_err(|err| {
            anyhow!(
                "failed sending last use event for borrow {:?}: {:?}",
                borrow,
                err
            )
        })
    }

    async fn borrow_drop(
        &mut self,
        _this: &Instance<Self>,
        borrow: u64,
        last_use_receiver: PortReceiver<Option<Event>>,
    ) -> Result<()> {
        let mut last_use_receiver = last_use_receiver;
        let last_use_event = last_use_receiver.recv().await.map_err(|err| {
            anyhow!(
                "failed receiving last use event for borrow {:?}: {:?}",
                borrow,
                err
            )
        })?;

        if let Some(stream) = self.cuda_stream() {
            stream.wait_event(
                &mut last_use_event.expect("sent borrow to CUDA stream, expected a CUDA event"),
            );
        }
        // let the cell drop.
        Ok(())
    }

    async fn delete_refs(&mut self, _this: &Instance<Self>, refs: Vec<Ref>) -> Result<()> {
        if let Some(recording) = self.get_defining_recording() {
            recording.messages.push(StreamMessage::DeleteRefs(refs));
            return Ok(());
        }

        for ref_ in refs.iter() {
            self.env.remove(ref_);
        }
        Ok(())
    }

    async fn request_status(&mut self, _this: &Instance<Self>) -> Result<()> {
        if self.get_defining_recording().is_some() {
            bail!("request_status not allowed in recording");
        }

        Ok(())
    }

    async fn init_comm(
        &mut self,
        _this: &Instance<Self>,
        comm: ActorHandle<NcclCommActor>,
    ) -> Result<()> {
        if self.get_defining_recording().is_some() {
            bail!("init_comm not allowed in recording");
        }

        self.comm = Some(comm);
        Ok(())
    }

    async fn reduce(
        &mut self,
        this: &Instance<Self>,
        comm: Arc<ActorHandle<NcclCommActor>>,
        dim_size: i64,
        result: Ref,
        local_tensor: Ref,
        factory: Factory,
        reduction: Reduction,
        scatter: bool,
        in_place: bool,
        out: Option<Ref>,
    ) -> Result<()> {
        let stream = self
            .cuda_stream()
            .expect("reductions not yet supported for non-CUDA workers")
            .clone();
        let input_cell = self.get_or_fake_on_err(local_tensor, &factory)?;
        let out_cell = out
            .map(|out| self.get_or_fake_on_err(out, &factory))
            .transpose()?;
        let output_cell = match reduction {
            Reduction::Stack => {
                if scatter {
                    let output_cell = if in_place {
                        input_cell.clone()
                    } else {
                        out_cell.unwrap_or({
                            let borrow = input_cell.try_borrow().map_err(|e| anyhow!("{e:?}"))?;
                            let cloned = deep_clone(&borrow);
                            TensorCell::new(cloned)
                        })
                    };
                    comm.all_to_all_single(this, output_cell.clone(), input_cell, stream)
                        .await?;
                    output_cell
                } else {
                    ensure!(
                        !in_place,
                        "in-place, non-scatter not supported for stack reduce"
                    );

                    let output_cell = out_cell.unwrap_or({
                        // In Python, this would be [dim_size, *factory.sizes]
                        let sizes = [&[dim_size][..], &factory.size[..]].concat();
                        let output =
                            factory_empty(&sizes, factory.dtype, factory.layout, factory.device);
                        TensorCell::new(output)
                    });

                    comm.all_gather_into_tensor(this, output_cell.clone(), input_cell, stream)
                        .await?;
                    output_cell
                }
            }
            Reduction::ReduceOp(op) => {
                if scatter {
                    ensure!(!in_place, "in-place, scatter not supported for reduce");

                    let output_cell = out_cell.unwrap_or({
                        let output = factory_empty(
                            &factory.size[1..],
                            factory.dtype,
                            factory.layout,
                            factory.device,
                        );
                        TensorCell::new(output)
                    });
                    comm.reduce_scatter_tensor(this, output_cell.clone(), input_cell, op, stream)
                        .await?;
                    output_cell
                } else {
                    let output_cell = if in_place {
                        input_cell.clone()
                    } else {
                        out_cell.unwrap_or({
                            let borrow = input_cell.try_borrow().map_err(|e| anyhow!("{e:?}"))?;
                            let cloned = deep_clone(&borrow);
                            TensorCell::new(cloned)
                        })
                    };
                    comm.all_reduce(this, output_cell.clone(), op, stream)
                        .await?;
                    output_cell
                }
            }
        };

        // Populate result
        self.env.insert(result, Ok(output_cell.into()));
        Ok(())
    }

    async fn send_tensor(
        &mut self,
        this: &Instance<Self>,
        result: Ref,
        from_rank: Option<usize>,
        to_rank: Option<usize>,
        tensor: Ref,
        factory: Factory,
        comm: Arc<ActorHandle<NcclCommActor>>,
    ) -> Result<()> {
        if to_rank.is_none() && from_rank.is_none() {
            bail!("tried to send tensor without a to/from rank");
        }

        // Value is local, so we do not have to actually send it.
        if from_rank == to_rank {
            let input_cell = self
                .env
                .get(&tensor)
                .ok_or_else(|| anyhow!("tensor not found in stream: {tensor:#?}"))?;
            let output_cell = match input_cell {
                Ok(RValue::Tensor(input_cell)) => {
                    // We create a defensive copy here to prevent mutations on
                    // the input tensor from affecting output tensor.
                    // Should we copy if input ref == output ref?
                    // Should we support copy-on-write to avoid unnecessary copy?
                    let borrow = input_cell.try_borrow().map_err(|e| anyhow!("{e:?}"))?;
                    let cloned = deep_clone(&borrow);
                    Ok(RValue::Tensor(TensorCell::new(cloned)))
                }
                Ok(rval) => bail!("tensor ref is not a tensor: {:?}", rval),
                Err(err) => Err(err.clone()),
            };
            self.env.insert(result, output_cell);
            return Ok(());
        }

        let mut messages = Vec::new();

        if let Some(to_rank) = to_rank {
            let input_cell = self.get_or_fake_on_err(tensor, &factory)?;
            messages.push(CommMessage::Send(
                input_cell,
                to_rank.try_into().unwrap(),
                self.cuda_stream()
                    .expect("tried to send_tensor on non-cuda stream")
                    .clone(),
                this.open_once_port().0,
            ));
        }

        if let Some(from_rank) = from_rank {
            let output_cell = TensorCell::new(factory_empty(
                &factory.size,
                factory.dtype,
                factory.layout,
                factory.device,
            ));
            messages.push(CommMessage::Recv(
                output_cell.clone(),
                from_rank.try_into().unwrap(),
                self.cuda_stream()
                    .expect("tried to send_tensor on non-cuda stream")
                    .clone(),
                this.open_once_port().0,
            ));
            self.env.insert(result, Ok(output_cell.into()));
        }

        comm.group(
            this,
            messages,
            self.cuda_stream()
                .expect("tried to send_tensor on non-cuda stream")
                .clone(),
        )
        .await?;
        Ok(())
    }

    async fn send_value(
        &mut self,
        this: &Instance<Self>,
        seq: Seq,
        worker_actor_id: ActorId,
        mutates: Vec<Ref>,
        function: Option<ResolvableFunction>,
        args: Vec<WireValue>,
        kwargs: HashMap<String, WireValue>,
        device_meshes: HashMap<Ref, DeviceMesh>,
        pipe: Option<PortHandle<PipeMessage>>,
    ) -> Result<()> {
        let result = if let Some(function) = function {
            // If a function was provided, use that to resolve the value.
            match function.as_torch_op() {
                Some((op, overload)) => {
                    self.call_torch_op(op, overload, args, kwargs)
                        .map(|rvalues| {
                            if rvalues.len() == 1 {
                                Ok(rvalues[0].clone().into())
                            } else {
                                // TODO: Replace with native pytrees when possible
                                Python::with_gil(|py| {
                                    Ok((|| {
                                        let py_rvalues = rvalues
                                            .into_iter()
                                            // SAFETY: This inherits the unsafety of `try_to_object_unsafe`.
                                            .map(|rvalue| unsafe {
                                                rvalue.try_to_object_unsafe(py)
                                            })
                                            .collect::<Result<Vec<_>, _>>()?;
                                        PyTuple::new_bound(py, &py_rvalues)
                                            .extract::<PyTree<RValue>>()
                                    })()
                                    .map_err(SerializablePyErr::from_fn(py))?)
                                })
                            }
                        })?
                }
                // Use block-in-place to allow nested callbacks to re-enter the
                // runtime to run async code.
                _ => tokio::task::block_in_place(|| {
                    self.call_python_fn(
                        this,
                        function,
                        args,
                        kwargs,
                        &mutates,
                        device_meshes,
                        HashMap::new(),
                    )
                }),
            }
        } else {
            // If there's no function provided, there should be exactly one arg
            // and no kwargs.
            match (args.len(), kwargs.len()) {
                (1, 0) => Python::with_gil(|py| {
                    let arg = args[0]
                        .as_py_object()
                        .ok_or_else(|| {
                            CallFunctionError::UnsupportedArgType(
                                "send_value".to_string(),
                                "expected a PyObject as the first arg".to_string(),
                            )
                        })?
                        .unpickle(py)
                        .map_err(SerializablePyErr::from_fn(py))?;
                    arg.extract::<PyTree<PyObject>>()
                        .map_err(SerializablePyErr::from_fn(py))?
                        .try_into_map(|obj| {
                            let bound_obj = obj.bind(py);
                            if let Ok(ref_) = Ref::from_py_object(bound_obj) {
                                self.ref_to_rvalue(&ref_)
                            } else {
                                Ok(bound_obj
                                    .extract::<RValue>()
                                    .map_err(SerializablePyErr::from_fn(py))?)
                            }
                        })
                }),
                _ => Err(CallFunctionError::TooManyArgsForValue {
                    args: format!("{:?}", args),
                    kwargs: format!("{:?}", kwargs),
                }),
            }
        };

        let value = match result {
            Ok(rvalue) => {
                // When returning a tensor, we copy out to decouple from the GPU,
                // as the worker will either serialize and send this to the controller
                // or to a pipe and we see hangs if it tries to pull from the GPU
                // in its thread.
                Ok(rvalue.into_map(|rval| match rval {
                    RValue::Tensor(tensor) => RValue::Tensor(tensor.try_cpu().unwrap()),
                    RValue::TensorList(tensors) => RValue::TensorList(
                        tensors
                            .into_iter()
                            .map(|tensor| tensor.try_cpu().unwrap())
                            .collect(),
                    ),
                    rval => rval,
                }))
            }
            Err(err) => {
                let err = Arc::new(err);
                for ref_ in mutates {
                    self.env.insert(ref_, Err(err.clone()));
                }
                Err(err)
            }
        }
        .map_err(|err| {
            let err = err.unwrap_dependent_error().unwrap_or(err);
            WorkerError {
                backtrace: format!("{:?}", err),
                worker_actor_id,
            }
        });

        // Actually send the value.
        if let Some(pipe) = pipe {
            pipe.send(PipeMessage::SendValue(value))?;
        } else {
            let result = match value {
                Ok(value) => Ok(Serialized::serialize_anon(&value).map_err(anyhow::Error::from)?),
                Err(e) => Err(e),
            };
            self.controller_actor
                .fetch_result(this, seq, result)
                .await?;
        }

        Ok(())
    }

    async fn set_value(
        &mut self,
        this: &Instance<Self>,
        results: Vec<Option<Ref>>,
        pipe: Result<PortHandle<PipeMessage>, CallFunctionError>,
    ) -> Result<()> {
        let (tx, rx) = this.open_once_port();
        let value = async {
            pipe?
                .send(PipeMessage::RecvValue(tx))
                .map_err(anyhow::Error::from)
                .map_err(CallFunctionError::from)?;
            rx.recv()
                .await
                .map_err(anyhow::Error::from)
                .map_err(CallFunctionError::from)
        }
        .await;

        // Apply results to env.
        // `handle_results` will return the error passed in, which we treat as
        // as a value and not an actual error.
        let _ = self.handle_results(&results, value.map(|v| v.into_leaves()));
        Ok(())
    }

    async fn define_recording(&mut self, _this: &Instance<Self>, recording: Ref) -> Result<()> {
        if self.active_recording.is_some() {
            bail!("different recording already active");
        }
        match self.recordings.entry(recording) {
            Entry::Occupied(_) => bail!("recording {:?} already defined", recording),
            Entry::Vacant(entry) => entry.insert(Recording::new()),
        };
        self.active_recording = Some(RecordingState::Defining {
            recording,
            defined_borrows: HashSet::new(),
        });
        Ok(())
    }

    async fn finalize_recording(&mut self, _this: &Instance<Self>, recording: Ref) -> Result<()> {
        match self.active_recording {
            Some(RecordingState::Defining {
                recording: active_recording,
                ref defined_borrows,
            }) if active_recording == recording => {
                ensure!(
                    defined_borrows.is_empty(),
                    "all borrows created within recording must be dropped within recording"
                );
                self.active_recording = None;
            }
            _ => bail!("cannot finalize recording that isn't active"),
        }
        Ok(())
    }

    async fn recording_formal(
        &mut self,
        _this: &Instance<Self>,
        result: Ref,
        argument_index: usize,
    ) -> Result<()> {
        match self.get_defining_recording() {
            Some(recording) => {
                recording.messages.push(StreamMessage::RecordingFormal {
                    result,
                    argument_index,
                });
            }
            None => bail!("recording_formal called outside of recording"),
        };
        Ok(())
    }

    async fn recording_result(
        &mut self,
        _this: &Instance<Self>,
        result: Ref,
        output_index: usize,
    ) -> Result<()> {
        match self.get_defining_recording() {
            Some(recording) => {
                recording.messages.push(StreamMessage::RecordingResult {
                    result,
                    output_index,
                });
            }
            None => bail!("recording_result called outside of recording"),
        };
        Ok(())
    }

    async fn call_recording(
        &mut self,
        this: &Instance<Self>,
        seq: Seq,
        recording: Ref,
        results: Vec<Ref>,
        actuals: Vec<Ref>,
    ) -> Result<()> {
        if self.active_recording.is_some() {
            bail!("cannot call recording while another recording is active");
        }

        let messages = match self.recordings.get(&recording) {
            Some(recording) => recording
                .messages
                .iter()
                .map(|message| message.clone_for_recording())
                .collect::<Vec<_>>(),
            None => bail!("recording {:?} not found", recording),
        };

        self.active_recording = Some(RecordingState::Running);

        // Global error for all messages in the recording. The first time a message
        // fails in the recording, we set the error. We then need to propagate this
        // error to all of the refs mutated by the entire recording, as well as the
        // result refs.
        let mut error: Option<Arc<CallFunctionError>> = None;
        // The set of all refs defined by this recording (excluding "results"),
        // which we need to ensure are deleted when the recording is done executing.
        let mut all_defined_refs = HashSet::new();
        // The set of all refs mutated by this recording. If there is an error with
        // any message, all of these refs need to have the correct error set.
        let mut all_mutated_refs = HashSet::new();
        // Map from the result ref of a RecordingFormal message to the associated
        // actual ref from "actuals". We need to track this in order to properly
        // handle recordings that mutate refs contained in "actuals" -- every
        // message in the recording that interacts with the recording inputs will
        // interact with the formal ref rather than the actual ref.
        let mut formal_to_actual_refs = HashMap::new();
        for message in messages.into_iter() {
            let defined_refs = message.get_defined_refs();
            let mutated_refs_with_formals = message.get_mutated_refs();
            let mutated_refs_with_actuals = mutated_refs_with_formals
                .iter()
                .map(|ref_| match formal_to_actual_refs.get(ref_) {
                    Some(actual_ref) => *actual_ref,
                    None => *ref_,
                })
                .collect::<HashSet<_>>();

            all_defined_refs.extend(defined_refs.clone());
            all_mutated_refs.extend(
                mutated_refs_with_actuals
                    .iter()
                    .filter(|ref_| !all_defined_refs.contains(*ref_)),
            );

            match message {
                StreamMessage::RecordingFormal {
                    result: formal_ref,
                    argument_index,
                } => match actuals.get(argument_index) {
                    None => bail!("recording_formal called with too few arguments"),
                    Some(actual_ref) => {
                        formal_to_actual_refs.insert(formal_ref, *actual_ref);
                        self.env
                            .insert(formal_ref, self.ref_to_rvalue(actual_ref).map_err(Arc::new));
                    }
                },
                StreamMessage::RecordingResult {
                    result: result_ref,
                    output_index,
                } => match results.get(output_index) {
                    None => bail!("recording_result called with too few results"),
                    Some(actual_result_ref) => {
                        self.env.insert(
                            *actual_result_ref,
                            self.ref_to_rvalue(&result_ref).map_err(Arc::new),
                        );
                    }
                },
                StreamMessage::DeleteRefs(refs) => {
                    for ref_ in &refs {
                        all_defined_refs.remove(ref_);
                    }
                    StreamMessageHandler::handle(self, this, StreamMessage::DeleteRefs(refs))
                        .await?;
                }
                _ => unimplemented!(),
            };

            // It's not entirely trivial to determine whether a message "failed" or not.
            // For example, the CallFunction message can return Ok(..) if there is an error
            // in the underlying function call. But in that case, we would still want to
            // consider the recording call as "failed". Unlike in python, where we can just
            // wrap everything in try-except, in rust, we need to track the defined/mutated refs
            // for each individual message. After processing the message, if any of the associated
            // defined/mutated refs contain an error, then we know the recording has failed.
            if error.is_none() {
                // The stream message would have operated on the formal ref rather than the actual
                // ref, so we check mutated_refs_with_formals rather than mutated_refs_with_actuals.
                // Later, if there is an error, the error will be propagated to the actuals.
                for ref_ in defined_refs.iter().chain(mutated_refs_with_formals.iter()) {
                    if let Some(Err(err)) = self.env.get(ref_) {
                        error = Some(err.clone());
                        match err.as_ref() {
                            // A DependentError should already have been reported to the controller,
                            // so we don't need to do anything.
                            CallFunctionError::DependentError(_) => (),
                            err => {
                                // Report failure to the controller.
                                self.controller_actor
                                    .remote_function_failed(
                                        this,
                                        seq,
                                        WorkerError {
                                            backtrace: format!("{err}"),
                                            worker_actor_id: this.self_id().clone(),
                                        },
                                    )
                                    .await?
                            }
                        };
                        break;
                    }
                }

                // Continue processing the remaining stream messages regardless of error.
                // We need to do this partially for error propagation, but also because
                // certain messages (like borrows and reductions) need to run regardless
                // in order to prevent deadlocks.
            }
        }

        // Sanity check. The only refs remaining in all_defined_refs should be the
        // formal refs, since the controller should have generated DeleteRefs messages
        // for all other refs defined by the recording.
        assert_eq!(all_defined_refs.len(), formal_to_actual_refs.len());

        // Delete the formal refs.
        StreamMessageHandler::handle(
            self,
            this,
            StreamMessage::DeleteRefs(all_defined_refs.into_iter().collect()),
        )
        .await?;

        // Any refs mutated by the recording and all results should have the same error
        // (the original error that caused the recording to fail).
        if error.is_some() {
            for ref_ in results.iter().chain(all_mutated_refs.iter()) {
                self.env.insert(*ref_, Err(error.clone().unwrap()));
            }
        }

        self.active_recording = None;
        Ok(())
    }

    async fn set_ref_unit_tests_only(
        &mut self,
        _this: &Instance<Self>,
        reference: Ref,
        value: WireValue,
    ) -> Result<()> {
        self.env
            .insert(reference, Ok(self.wire_to_rvalue(value).unwrap()));
        Ok(())
    }

    async fn set_tensor_ref_unit_tests_only(
        &mut self,
        _this: &Instance<Self>,
        reference: Ref,
        tensor_result: TensorCellResult,
    ) -> Result<()> {
        match tensor_result {
            Ok(tensor_cell) => {
                self.env.insert(reference, Ok(RValue::Tensor(tensor_cell)));
            }
            Err(err) => {
                self.env.insert(reference, Err(err));
            }
        }
        Ok(())
    }

    async fn get_ref_unit_tests_only(
        &mut self,
        _this: &Instance<Self>,
        reference: Ref,
    ) -> Result<Option<Result<WireValue, Arc<CallFunctionError>>>> {
        /// For testing only, doesn't support Tensor or TensorList.
        fn rvalue_to_wire(
            value: Result<RValue, Arc<CallFunctionError>>,
        ) -> Result<WireValue, Arc<CallFunctionError>> {
            Ok(match value? {
                RValue::Int(val) => WireValue::Int(val),
                RValue::IntList(val) => WireValue::IntList(val),
                RValue::Double(val) => WireValue::Double(val),
                RValue::Bool(val) => WireValue::Bool(val),
                RValue::String(val) => WireValue::String(val),
                RValue::Layout(val) => WireValue::Layout(val),
                RValue::Device(val) => WireValue::Device(val),
                RValue::ScalarType(val) => WireValue::ScalarType(val),
                RValue::MemoryFormat(val) => WireValue::MemoryFormat(val),
                RValue::None => WireValue::None(()),
                other => WireValue::String(format!("unsupported rvalue type: {:?}", other)),
            })
        }
        Ok(self
            .env
            .get(&reference)
            .map(|rvalue| rvalue_to_wire(rvalue.clone())))
    }

    async fn get_tensor_ref_unit_tests_only(
        &mut self,
        _this: &Instance<Self>,
        reference: Ref,
    ) -> Result<Option<TensorCellResult>> {
        match self.env.get(&reference) {
            Some(Ok(rvalue)) => match rvalue {
                RValue::Tensor(tensor) => Ok(Some(Ok(tensor.clone()))),
                other => bail!("expected tensor, got {:?}", other),
            },
            Some(Err(err)) => Ok(Some(Err(err.clone()))),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::actor::ActorStatus;
    use hyperactor::id;
    use hyperactor::supervision::ActorSupervisionEvent;
    use monarch_messages::controller::ControllerMessage;
    use monarch_messages::worker::StreamCreationMode;
    use monarch_types::PickledPyObject;
    use timed_test::async_timed_test;
    use torch_sys::factory_float_tensor;
    use torch_sys::nccl::UniqueId;
    use torch_sys::testing::allclose;

    use super::*;
    use crate::comm::CommParams;
    use crate::test_util;

    struct TestSetup {
        proc: Proc,
        stream_actor: ActorHandle<StreamActor>,
        client: Mailbox,
        // Unused, but necessary, because proc needs a supervision
        // port -- otherwise an actor failure will cause a crash.
        #[allow(dead_code)]
        supervision_rx: PortReceiver<ActorSupervisionEvent>,
        controller_rx: PortReceiver<ControllerMessage>,
    }

    impl TestSetup {
        async fn new() -> Result<Self> {
            test_util::test_setup()?;

            let proc = Proc::local();
            let (_, controller_actor, controller_rx) =
                proc.attach_actor::<ControllerActor, ControllerMessage>("controller")?;
            let client = proc.attach("client")?;
            let (supervision_tx, supervision_rx) = client.open_port();
            proc.set_supervision_coordinator(supervision_tx)?;
            let stream_actor = proc
                .spawn::<StreamActor>(
                    "stream",
                    StreamParams {
                        world_size: 1,
                        rank: 0,
                        creation_mode: StreamCreationMode::UseDefaultStream,
                        id: 0.into(),
                        device: None,
                        controller_actor: controller_actor.clone(),
                    },
                )
                .await?;

            Ok(Self {
                proc,
                stream_actor,
                client,
                supervision_rx,
                controller_rx,
            })
        }
    }

    async fn assert_actor_failed_with_msg(proc: &Proc, actor_id: &ActorId, expected_msg: String) {
        loop {
            let status = proc
                .ledger_snapshot()
                .roots
                .get(actor_id)
                .unwrap()
                .status
                .clone();
            if let ActorStatus::Failed(msg) = status {
                assert!(msg.contains(&expected_msg));
                break;
            } else {
                tokio::task::yield_now().await;
            }
        }
    }

    async fn assert_refs_do_not_exist(test_setup: &TestSetup, refs: &[Ref]) {
        for ref_ in refs {
            assert!(
                test_setup
                    .stream_actor
                    .get_tensor_ref_unit_tests_only(&test_setup.client, *ref_)
                    .await
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_handle_results() {
        let controller_ref = ActorRef::attest(id!(test[0].actor[0]));

        let param = StreamParams {
            world_size: 1,
            rank: 0,
            creation_mode: StreamCreationMode::UseDefaultStream,
            id: 0.into(),
            device: None,
            controller_actor: controller_ref,
        };
        let mut actor = StreamActor::new(param).await.unwrap();

        actor
            .handle_results(
                &vec![Some(0.into()), Some(Ref { id: 1 })],
                Ok(vec![RValue::Int(1), RValue::Int(2)]),
            )
            .unwrap();

        actor
            .handle_results(
                &vec![Some(Ref { id: 4 }), None],
                Ok(vec![RValue::Int(1), RValue::Int(2)]),
            )
            .unwrap();

        assert!(
            actor
                .handle_results(
                    &vec![Some(Ref { id: 2 }), Some(Ref { id: 3 })],
                    Ok(vec![RValue::Int(1), RValue::Int(2), RValue::Int(3)]),
                )
                .is_err()
        );

        assert!(
            actor
                .handle_results(
                    &vec![Some(Ref { id: 6 }), Some(Ref { id: 7 }), None],
                    Ok(vec![RValue::Int(1), RValue::Int(2)]),
                )
                .is_err()
        );
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_define_recording_other_recording_active() -> Result<()> {
        let test_setup = TestSetup::new().await?;
        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 0.into())
            .await?;
        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 1.into())
            .await?;
        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "different recording already active".into(),
        )
        .await;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_define_recording_already_defined() -> Result<()> {
        let test_setup = TestSetup::new().await?;
        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 0.into())
            .await?;
        test_setup
            .stream_actor
            .finalize_recording(&test_setup.client, 0.into())
            .await?;
        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 0.into())
            .await?;
        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "already defined".into(),
        )
        .await;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_finalize_recording_other_recording_active() -> Result<()> {
        let test_setup = TestSetup::new().await?;
        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 0.into())
            .await?;
        test_setup
            .stream_actor
            .finalize_recording(&test_setup.client, 1.into())
            .await?;
        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "cannot finalize recording that isn't active".into(),
        )
        .await;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_recording_formal_outside_recording() -> Result<()> {
        let test_setup = TestSetup::new().await?;
        test_setup
            .stream_actor
            .recording_formal(&test_setup.client, 0.into(), 0)
            .await?;
        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "recording_formal called outside of recording".into(),
        )
        .await;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_recording_result_outside_recording() -> Result<()> {
        let test_setup = TestSetup::new().await?;
        test_setup
            .stream_actor
            .recording_result(&test_setup.client, 0.into(), 0)
            .await?;
        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "recording_result called outside of recording".into(),
        )
        .await;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_call_recording_other_recording_active() -> Result<()> {
        let test_setup = TestSetup::new().await?;
        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 0.into())
            .await?;
        test_setup
            .stream_actor
            .call_recording(&test_setup.client, 0.into(), 0.into(), vec![], vec![])
            .await?;
        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "cannot call recording while another recording is active".into(),
        )
        .await;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_call_recording_not_found() -> Result<()> {
        let test_setup = TestSetup::new().await?;
        test_setup
            .stream_actor
            .call_recording(&test_setup.client, 0.into(), 0.into(), vec![], vec![])
            .await?;
        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "not found".into(),
        )
        .await;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_recording_formal_too_few_arguments() -> Result<()> {
        let test_setup = TestSetup::new().await?;

        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 0.into())
            .await?;

        test_setup
            .stream_actor
            .recording_formal(&test_setup.client, 1.into(), 0)
            .await?;

        test_setup
            .stream_actor
            .finalize_recording(&test_setup.client, 0.into())
            .await?;

        test_setup
            .stream_actor
            .call_recording(&test_setup.client, 0.into(), 0.into(), vec![], vec![])
            .await?;

        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "recording_formal called with too few arguments".into(),
        )
        .await;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_recording_result_too_few_results() -> Result<()> {
        let test_setup = TestSetup::new().await?;

        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 0.into())
            .await?;

        test_setup
            .stream_actor
            .recording_result(&test_setup.client, 1.into(), 0)
            .await?;

        test_setup
            .stream_actor
            .finalize_recording(&test_setup.client, 0.into())
            .await?;

        test_setup
            .stream_actor
            .call_recording(&test_setup.client, 0.into(), 0.into(), vec![], vec![])
            .await?;

        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "recording_result called with too few results".into(),
        )
        .await;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_basic_call_recording() -> Result<()> {
        let mut test_setup = TestSetup::new().await?;

        // Define a recording equivalent to:
        // def f(x, y):
        //   return y, x
        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 0.into())
            .await?;

        let formal0_ref = 1.into();
        let formal0_index = 1;
        test_setup
            .stream_actor
            .recording_formal(&test_setup.client, formal0_ref, formal0_index)
            .await?;

        let formal1_ref = 2.into();
        let formal1_index = 0;
        test_setup
            .stream_actor
            .recording_formal(&test_setup.client, formal1_ref, formal1_index)
            .await?;

        let result0_ref = formal0_ref;
        let result0_index = 0;
        test_setup
            .stream_actor
            .recording_result(&test_setup.client, result0_ref, result0_index)
            .await?;

        let result1_ref = formal1_ref;
        let result1_index = 1;
        test_setup
            .stream_actor
            .recording_result(&test_setup.client, result1_ref, result1_index)
            .await?;

        test_setup
            .stream_actor
            .finalize_recording(&test_setup.client, 0.into())
            .await?;

        let actual0_ref = 3.into();
        let actual0_tensor = TensorCell::new(factory_float_tensor(
            &[1.0, 2.0, 3.0],
            "cuda".try_into().unwrap(),
        ));

        let actual1_ref = 4.into();
        let actual1_tensor = TensorCell::new(factory_float_tensor(
            &[4.0, 5.0],
            "cuda".try_into().unwrap(),
        ));

        test_setup
            .stream_actor
            .set_tensor_ref_unit_tests_only(
                &test_setup.client,
                actual0_ref,
                Ok(actual0_tensor.clone()),
            )
            .await?;

        test_setup
            .stream_actor
            .set_tensor_ref_unit_tests_only(
                &test_setup.client,
                actual1_ref,
                Ok(actual1_tensor.clone()),
            )
            .await?;

        // Call the recording with valid tensors for the actual inputs,
        // and store the results in refs 5 and 6.
        let actual_result0_ref = 5.into();
        let actual_result1_ref = 6.into();
        test_setup
            .stream_actor
            .call_recording(
                &test_setup.client,
                0.into(),
                0.into(),
                vec![actual_result0_ref, actual_result1_ref],
                vec![actual0_ref, actual1_ref],
            )
            .await?;

        // Ensure the results are correct.
        let result0_tensor = test_setup
            .stream_actor
            .get_tensor_ref_unit_tests_only(&test_setup.client, actual_result0_ref)
            .await?;
        let result1_tensor = test_setup
            .stream_actor
            .get_tensor_ref_unit_tests_only(&test_setup.client, actual_result1_ref)
            .await?;
        assert!(
            allclose(
                &result0_tensor.unwrap().unwrap().borrow(),
                &actual1_tensor.borrow()
            )
            .unwrap()
        );
        assert!(
            allclose(
                &result1_tensor.unwrap().unwrap().borrow(),
                &actual0_tensor.borrow()
            )
            .unwrap()
        );

        // Ensure the temporary refs associated with the formals/results have
        // been deleted.
        assert_refs_do_not_exist(&test_setup, &[formal0_ref, formal1_ref]).await;

        // Pass an invalid ref as an input to the recording. Both result refs should contain
        // a RefNotFound error after running the recording, and the error should be reported to
        // the controller because it's not a dependent error.
        let nonexistent_ref = 105.into();
        test_setup
            .stream_actor
            .call_recording(
                &test_setup.client,
                1.into(),
                0.into(),
                vec![actual_result0_ref, actual_result1_ref],
                vec![nonexistent_ref, actual1_ref],
            )
            .await?;

        for ref_ in [actual_result0_ref, actual_result1_ref] {
            let result = test_setup
                .stream_actor
                .get_tensor_ref_unit_tests_only(&test_setup.client, ref_)
                .await
                .unwrap();
            assert!(
                matches!(result.unwrap().unwrap_err().as_ref(), CallFunctionError::RefNotFound(err_ref) if *err_ref == nonexistent_ref)
            );
        }

        assert_refs_do_not_exist(&test_setup, &[formal0_ref, formal1_ref]).await;

        let controller_msg = test_setup.controller_rx.recv().await?;
        assert!(matches!(
            controller_msg,
            ControllerMessage::RemoteFunctionFailed {
                seq,
                error
            } if seq == 1.into() && error.backtrace.contains("ref not found")
        ));

        // Call the recording where one of the input tensors "failed" on a previous
        // invocation. Both result refs should have a DependentError that wraps this error,
        // and the error should not be reported to the controller.
        let error = Arc::new(CallFunctionError::Anyhow(anyhow!("bad tensor")));
        test_setup
            .stream_actor
            .set_tensor_ref_unit_tests_only(&test_setup.client, actual1_ref, Err(error.clone()))
            .await?;

        test_setup
            .stream_actor
            .call_recording(
                &test_setup.client,
                2.into(),
                0.into(),
                vec![actual_result0_ref, actual_result1_ref],
                vec![actual0_ref, actual1_ref],
            )
            .await?;

        for ref_ in [actual_result0_ref, actual_result1_ref] {
            let result = test_setup
                .stream_actor
                .get_tensor_ref_unit_tests_only(&test_setup.client, ref_)
                .await
                .unwrap();
            assert!(
                matches!(result.unwrap().unwrap_err().as_ref(), CallFunctionError::DependentError(dep_err) if Arc::ptr_eq(dep_err, &error))
            );
        }

        let ref_to_send = Python::with_gil(|py| {
            PickledPyObject::pickle(actual_result0_ref.into_py(py).bind(py)).unwrap()
        });

        test_setup
            .stream_actor
            .send_value(
                &test_setup.client,
                3.into(),
                test_setup.stream_actor.actor_id().clone(),
                Vec::new(),
                None,
                vec![WireValue::PyObject(ref_to_send)],
                HashMap::new(),
                HashMap::new(),
                None,
            )
            .await?;

        // This tests that the DependentError was never reported to the controller.
        // If it were reported to the controller, the next message would match
        // RemoteFunctionFailed instead of FetchResult.
        let controller_msg = test_setup.controller_rx.recv().await?;
        assert!(matches!(
            controller_msg,
            ControllerMessage::FetchResult {
                seq,
                value: Err(error)
            } if seq == 3.into() && error.backtrace.contains("bad tensor")
        ));

        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_request_status_in_recording() -> Result<()> {
        let test_setup = TestSetup::new().await?;
        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 0.into())
            .await?;
        test_setup
            .stream_actor
            .request_status(&test_setup.client)
            .await
            .expect_err("request_status should have failed");
        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "request_status not allowed in recording".into(),
        )
        .await;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_init_comm_in_recording() -> Result<()> {
        let test_setup = TestSetup::new().await?;
        test_setup
            .stream_actor
            .define_recording(&test_setup.client, 0.into())
            .await?;

        let dummy_comm = test_setup
            .proc
            .spawn::<NcclCommActor>(
                "comm",
                CommParams::New {
                    device: CudaDevice::new(0.into()),
                    unique_id: UniqueId::new()?,
                    world_size: 1,
                    rank: 0,
                },
            )
            .await?;

        test_setup
            .stream_actor
            .init_comm(&test_setup.client, dummy_comm)
            .await?;
        assert_actor_failed_with_msg(
            &test_setup.proc,
            test_setup.stream_actor.actor_id(),
            "init_comm not allowed in recording".into(),
        )
        .await;
        Ok(())
    }
}
