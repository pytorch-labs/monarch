/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::future::Future;
use std::future::pending;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::ActorId;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor::forward;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::Unbind;
use hyperactor_mesh::comm::multicast::CastInfo;
use monarch_types::PickledPyObject;
use monarch_types::SerializablePyErr;
use ndslice::Shape;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::PyBaseException;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyType;
use serde::Deserialize;
use serde::Serialize;
use serde_bytes::ByteBuf;
use tokio::sync::Mutex;
use tokio::sync::oneshot;

use crate::config::SHARED_ASYNCIO_RUNTIME;
use crate::mailbox::EitherPortRef;
use crate::mailbox::PyMailbox;
use crate::proc::InstanceWrapper;
use crate::proc::PyActorId;
use crate::proc::PyProc;
use crate::proc::PySerialized;
use crate::runtime::signal_safe_block_on;
use crate::shape::PyShape;

#[pyclass(frozen, module = "monarch._rust_bindings.monarch_hyperactor.actor")]
#[derive(Serialize, Deserialize, Named)]
pub struct PickledMessage {
    sender_actor_id: ActorId,
    message: ByteBuf,
}

impl std::fmt::Debug for PickledMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PickledMessage(sender_actor_id: {:?} message: {})",
            self.sender_actor_id,
            hyperactor::data::HexFmt(self.message.as_slice()),
        )
    }
}

#[pymethods]
impl PickledMessage {
    #[new]
    #[pyo3(signature = (*, sender_actor_id, message))]
    fn new(sender_actor_id: &PyActorId, message: Vec<u8>) -> Self {
        Self {
            sender_actor_id: sender_actor_id.into(),
            message: ByteBuf::from(message),
        }
    }

    #[getter]
    fn sender_actor_id(&self) -> PyActorId {
        self.sender_actor_id.clone().into()
    }

    #[getter]
    fn message<'a>(&self, py: Python<'a>) -> Bound<'a, PyBytes> {
        PyBytes::new(py, self.message.as_ref())
    }

    fn serialize(&self) -> PyResult<PySerialized> {
        PySerialized::new(self)
    }
}

#[pyclass(module = "monarch._rust_bindings.monarch_hyperactor.actor")]
pub struct PickledMessageClientActor {
    instance: Arc<Mutex<InstanceWrapper<PickledMessage>>>,
}

#[pymethods]
impl PickledMessageClientActor {
    #[new]
    fn new(proc: &PyProc, actor_name: &str) -> PyResult<Self> {
        Ok(Self {
            instance: Arc::new(Mutex::new(InstanceWrapper::new(proc, actor_name)?)),
        })
    }

    /// Send a message to any actor that can receive the corresponding serialized message.
    fn send(&self, actor_id: &PyActorId, message: &PySerialized) -> PyResult<()> {
        let instance = self.instance.blocking_lock();
        instance.send(actor_id, message)
    }

    /// Get the next message from the queue. It will block until a message is received
    /// or the timeout is reached in which case it will return None
    /// If the actor has been stopped, this returns an error.
    #[pyo3(signature = (*, timeout_msec = None))]
    fn get_next_message<'py>(
        &mut self,
        py: Python<'py>,
        timeout_msec: Option<u64>,
    ) -> PyResult<PyObject> {
        let instance = self.instance.clone();
        let result = signal_safe_block_on(py, async move {
            instance.lock().await.next_message(timeout_msec).await
        })?;
        Python::with_gil(|py| {
            result
                .map(|res| res.into_py_any(py))?
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
    }

    /// Stop the background task and return any messages that were received.
    /// TODO: This is currently just aborting the task, we should have a better way to stop it.
    fn drain_and_stop<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut instance = self.instance.blocking_lock();
        let messages = instance
            .drain_and_stop()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_iter()
            .map(|message| message.into_py_any(py))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, messages)
    }

    fn world_status<'py>(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        let instance = self.instance.clone();

        let worlds = signal_safe_block_on(py, async move {
            instance.lock().await.world_status(Default::default()).await
        })??;
        Python::with_gil(|py| {
            let py_dict = PyDict::new(py);
            for (world, status) in worlds {
                py_dict.set_item(world.to_string(), status.to_string())?;
            }
            Ok(py_dict.into())
        })
    }

    #[getter]
    fn actor_id(&self) -> PyResult<PyActorId> {
        let instance = self.instance.blocking_lock();
        Ok(PyActorId::from(instance.actor_id().clone()))
    }
}

#[pyclass(frozen, module = "monarch._rust_bindings.monarch_hyperactor.actor")]
#[derive(Default, Clone, Serialize, Deserialize, Named, PartialEq)]
pub struct PythonMessage {
    pub(crate) method: String,
    pub(crate) message: ByteBuf,
    response_port: Option<EitherPortRef>,
    rank: Option<usize>,
}

impl PythonMessage {
    pub fn with_rank(self, rank: usize) -> PythonMessage {
        PythonMessage {
            rank: Some(rank),
            ..self
        }
    }
    pub fn new_from_buf(
        method: String,
        message: Vec<u8>,
        response_port: Option<EitherPortRef>,
        rank: Option<usize>,
    ) -> Self {
        Self {
            method,
            message: message.into(),
            response_port,
            rank,
        }
    }
}

impl std::fmt::Debug for PythonMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonMessage")
            .field("method", &self.method)
            .field(
                "message",
                &hyperactor::data::HexFmt(self.message.as_slice()).to_string(),
            )
            .finish()
    }
}

impl Unbind for PythonMessage {
    fn unbind(&self, bindings: &mut Bindings) -> anyhow::Result<()> {
        self.response_port.unbind(bindings)
    }
}

impl Bind for PythonMessage {
    fn bind(&mut self, bindings: &mut Bindings) -> anyhow::Result<()> {
        self.response_port.bind(bindings)
    }
}

#[pymethods]
impl PythonMessage {
    #[new]
    #[pyo3(signature = (method, message, response_port, rank))]
    pub fn new(
        method: String,
        message: &[u8],
        response_port: Option<EitherPortRef>,
        rank: Option<usize>,
    ) -> Self {
        Self::new_from_buf(method, message.into(), response_port, rank)
    }

    #[getter]
    fn method(&self) -> &String {
        &self.method
    }

    #[getter]
    fn message<'a>(&self, py: Python<'a>) -> Bound<'a, PyBytes> {
        PyBytes::new(py, self.message.as_ref())
    }

    #[getter]
    fn response_port(&self) -> Option<EitherPortRef> {
        self.response_port.clone()
    }

    #[getter]
    fn rank(&self) -> Option<usize> {
        self.rank
    }
}

#[pyclass(module = "monarch._rust_bindings.monarch_hyperactor.actor")]
pub(super) struct PythonActorHandle {
    pub(super) inner: ActorHandle<PythonActor>,
}

#[pymethods]
impl PythonActorHandle {
    // TODO: do the pickling in rust
    fn send(&self, message: &PythonMessage) -> PyResult<()> {
        self.inner
            .send(message.clone())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(())
    }

    fn bind(&self) -> PyActorId {
        self.inner.bind::<PythonActor>().into_actor_id().into()
    }
}

/// An actor for which message handlers are implemented in Python.
#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [
        PythonMessage { cast = true },
    ],
)]
pub(super) struct PythonActor {
    /// The Python object that we delegate message handling to. An instance of
    /// `monarch.actor_mesh._Actor`.
    pub(super) actor: PyObject,

    /// Stores a reference to the Python event loop to run Python coroutines on.
    /// This is None when using single runtime mode, Some when using per-actor mode.
    task_locals: Option<pyo3_async_runtimes::TaskLocals>,
}

impl PythonActor {
    /// Get the TaskLocals to use for this actor.
    /// Returns either the shared TaskLocals or this actor's own TaskLocals based on configuration.
    fn get_task_locals(&self, py: Python) -> &pyo3_async_runtimes::TaskLocals {
        self.task_locals.as_ref().unwrap_or_else(|| {
            // Use shared TaskLocals
            static SHARED_TASK_LOCALS: OnceLock<pyo3_async_runtimes::TaskLocals> = OnceLock::new();
            Python::allow_threads(py, || SHARED_TASK_LOCALS.get_or_init(create_task_locals))
        })
    }
}

#[async_trait]
impl Actor for PythonActor {
    type Params = PickledPyObject;

    async fn new(actor_type: PickledPyObject) -> Result<Self, anyhow::Error> {
        Ok(Python::with_gil(|py| -> Result<Self, SerializablePyErr> {
            let unpickled = actor_type.unpickle(py)?;
            let class_type: &Bound<'_, PyType> = unpickled.downcast()?;
            let actor: PyObject = class_type.call0()?.into_py_any(py)?;

            // Only create per-actor TaskLocals if not using shared runtime
            let task_locals = (!hyperactor::config::global::get(SHARED_ASYNCIO_RUNTIME))
                .then(|| Python::allow_threads(py, create_task_locals));

            Ok(Self { actor, task_locals })
        })?)
    }
}

/// Create a new TaskLocals with its own asyncio event loop in a dedicated thread.
fn create_task_locals() -> pyo3_async_runtimes::TaskLocals {
    let (tx, rx) = std::sync::mpsc::channel();
    let _ = std::thread::spawn(move || {
        Python::with_gil(|py| {
            let asyncio = Python::import(py, "asyncio").unwrap();
            let event_loop = asyncio.call_method0("new_event_loop").unwrap();
            asyncio
                .call_method1("set_event_loop", (event_loop.clone(),))
                .unwrap();

            let task_locals = pyo3_async_runtimes::TaskLocals::new(event_loop.clone())
                .copy_context(py)
                .unwrap();
            tx.send(task_locals).unwrap();
            event_loop.call_method0("run_forever").unwrap();
        });
    });
    rx.recv().unwrap()
}

// [Panics in async endpoints]
// This class exists to solve a deadlock when an async endpoint calls into some
// Rust code that panics.
//
// When an async endpoint is invoked and calls into Rust, the following sequence happens:
//
// hyperactor message -> PythonActor::handle() -> call _Actor.handle() in Python
//   -> convert the resulting coroutine into a Rust future, but scheduled on
//      the Python asyncio event loop (`into_future_with_locals`)
//   -> set a callback on Python asyncio loop to ping a channel that fulfills
//      the Rust future when the Python coroutine has finished. ('PyTaskCompleter`)
//
// This works fine for normal results and Python exceptions: we will take the
// result of the callback and send it through the channel, where it will be
// returned to the `await`er of the Rust future.
//
// This DOESN'T work for panics. The behavior of a panic in pyo3-bound code is
// that it will get caught by pyo3 and re-thrown to Python as a PanicException.
// And if that PanicException ever makes it back to Rust, it will get unwound
// instead of passed around as a normal PyErr type.
//
// So:
//   - Endpoint panics.
//   - This panic is captured as a PanicException in Python and
//     stored as the result of the Python asyncio task.
//   - When the callback in `PyTaskCompleter` queries the status of the task to
//     pass it back to the Rust awaiter, instead of getting a Result type, it
//     just starts resumes unwinding the PanicException
//   - This triggers a deadlock, because the whole task dies without ever
//     pinging the response channel, and the Rust awaiter will never complete.
//
// We work around this by passing a side-channel to our Python task so that it,
// in Python, can catch the PanicException and notify the Rust awaiter manually.
// In this way we can guarantee that the awaiter will complete even if the
// `PyTaskCompleter` callback explodes.
#[pyclass(module = "monarch._rust_bindings.monarch_hyperactor.actor")]
struct PanicFlag {
    sender: Option<tokio::sync::oneshot::Sender<PyObject>>,
}

#[pymethods]
impl PanicFlag {
    fn signal_panic(&mut self, ex: PyObject) {
        self.sender.take().unwrap().send(ex).unwrap();
    }
}

#[async_trait]
impl Handler<PythonMessage> for PythonActor {
    async fn handle(&mut self, cx: &Context<Self>, message: PythonMessage) -> anyhow::Result<()> {
        let mailbox = PyMailbox {
            inner: cx.mailbox_for_py().clone(),
        };
        // Create a channel for signaling panics in async endpoints.
        // See [Panics in async endpoints].
        let (sender, receiver) = oneshot::channel();

        let future = Python::with_gil(|py| -> Result<_, SerializablePyErr> {
            let (rank, shape) = cx.cast_info();
            let awaitable = self.actor.call_method(
                py,
                "handle",
                (
                    mailbox,
                    rank,
                    PyShape::from(shape),
                    message,
                    PanicFlag {
                        sender: Some(sender),
                    },
                ),
                None,
            )?;

            pyo3_async_runtimes::into_future_with_locals(
                self.get_task_locals(py),
                awaitable.into_bound(py),
            )
            .map_err(|err| err.into())
        })?;

        // Spawn a child actor to await the Python handler method.
        let handler = AsyncEndpointTask::spawn(cx, ()).await?;
        handler.run(cx, PythonTask::new(future), receiver).await?;
        Ok(())
    }
}

/// Helper struct to make a Python future passable in an actor message.
///
/// Also so that we don't have to write this massive type signature everywhere
struct PythonTask {
    future: Mutex<Pin<Box<dyn Future<Output = PyResult<PyObject>> + Send + 'static>>>,
}

impl PythonTask {
    fn new(fut: impl Future<Output = PyResult<PyObject>> + Send + 'static) -> Self {
        Self {
            future: Mutex::new(Box::pin(fut)),
        }
    }

    async fn take(self) -> Pin<Box<dyn Future<Output = PyResult<PyObject>> + Send + 'static>> {
        self.future.into_inner()
    }
}

impl fmt::Debug for PythonTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonTask")
            .field("future", &"<PythonFuture>")
            .finish()
    }
}

/// An ['Actor'] used to monitor the result of an async endpoint. We use an
/// actor so that:
/// - Actually waiting on the async endpoint can happen concurrently with other endpoints.
/// - Any uncaught errors in the async endpoint will get propagated as a supervision event.
#[derive(Debug)]
struct AsyncEndpointTask {}

/// An invocation of an async endpoint on a [`PythonActor`].
#[derive(Handler, HandleClient, Debug)]
enum AsyncEndpointInvocation {
    Run(PythonTask, oneshot::Receiver<PyObject>),
}

#[async_trait]
impl Actor for AsyncEndpointTask {
    type Params = ();

    async fn new(_params: Self::Params) -> anyhow::Result<Self> {
        Ok(Self {})
    }
}

#[async_trait]
#[forward(AsyncEndpointInvocation)]
impl AsyncEndpointInvocationHandler for AsyncEndpointTask {
    async fn run(
        &mut self,
        cx: &Context<Self>,
        task: PythonTask,
        side_channel: oneshot::Receiver<PyObject>,
    ) -> anyhow::Result<()> {
        // Drive our PythonTask to completion, but listen on the side channel
        // and raise an error if we hear anything there.

        let err_or_never = async {
            // The side channel will resolve with a value if a panic occured during
            // processing of the async endpoint, see [Panics in async endpoints].
            match side_channel.await {
                Ok(value) => Python::with_gil(|py| -> Result<(), SerializablePyErr> {
                    let err: PyErr = value
                        .downcast_bound::<PyBaseException>(py)
                        .unwrap()
                        .clone()
                        .into();
                    Err(SerializablePyErr::from(py, &err))
                }),
                // An Err means that the sender has been dropped without sending.
                // That's okay, it just means that the Python task has completed.
                // In that case, just never resolve this future. We expect the other
                // branch of the select to finish eventually.
                Err(_) => pending().await,
            }
        };
        let future = task.take().await;
        let result: Result<(), SerializablePyErr> = tokio::select! {
            result = future => {
                match result {
                    Ok(_) => Ok(()),
                    Err(e) => Err(e.into()),
                }
            },
            result = err_or_never => {
                result
            }
        };
        result?;

        // Stop this actor now that its job is done.
        cx.stop()?;
        Ok(())
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PickledMessage>()?;
    hyperactor_mod.add_class::<PickledMessageClientActor>()?;
    hyperactor_mod.add_class::<PythonActorHandle>()?;
    hyperactor_mod.add_class::<PythonMessage>()?;
    hyperactor_mod.add_class::<PanicFlag>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use hyperactor::PortRef;
    use hyperactor::accum::ReducerSpec;
    use hyperactor::data::Serialized;
    use hyperactor::id;
    use hyperactor::message::ErasedUnbound;
    use hyperactor::message::Unbound;
    use hyperactor::reference::UnboundPort;

    use super::*;

    #[test]
    fn test_python_message_bind_unbind() {
        let reducer_spec = ReducerSpec {
            typehash: 123,
            builder_params: Some(Serialized::serialize(&"abcdefg12345".to_string()).unwrap()),
        };
        let port_ref = PortRef::<PythonMessage>::attest_reducible(
            id!(world[0].client[0][123]),
            Some(reducer_spec),
        );
        let message = PythonMessage {
            method: "test".to_string(),
            message: ByteBuf::from(vec![1, 2, 3]),
            response_port: Some(EitherPortRef::Unbounded(port_ref.clone().into())),
            rank: None,
        };
        {
            let mut erased = ErasedUnbound::try_from_message(message.clone()).unwrap();
            let mut bindings = vec![];
            erased
                .visit_mut::<UnboundPort>(|b| {
                    bindings.push(b.clone());
                    Ok(())
                })
                .unwrap();
            assert_eq!(bindings, vec![UnboundPort::from(&port_ref)]);
            let unbound = Unbound::try_from_message(message.clone()).unwrap();
            assert_eq!(message, unbound.bind().unwrap());
        }

        let no_port_message = PythonMessage {
            response_port: None,
            ..message
        };
        {
            let mut erased = ErasedUnbound::try_from_message(no_port_message.clone()).unwrap();
            let mut bindings = vec![];
            erased
                .visit_mut::<UnboundPort>(|b| {
                    bindings.push(b.clone());
                    Ok(())
                })
                .unwrap();
            assert_eq!(bindings.len(), 0);
            let unbound = Unbound::try_from_message(no_port_message.clone()).unwrap();
            assert_eq!(no_port_message, unbound.bind().unwrap());
        }
    }
}
