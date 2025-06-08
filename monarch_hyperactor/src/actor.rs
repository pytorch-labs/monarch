/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::future::Future;
use std::future::pending;
use std::sync::Arc;
use std::sync::OnceLock;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::ActorId;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::PortId;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::IndexedErasedUnbound;
use hyperactor::message::Unbind;
use hyperactor_mesh::actor_mesh::Cast;
use monarch_types::PickledPyObject;
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
        PyBytes::new_bound(py, self.message.as_ref())
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
                .map(|res| res.into_py(py))
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
            .map(|message| message.into_py(py))
            .collect::<Vec<PyObject>>();
        Ok(PyList::new_bound(py, messages))
    }

    fn world_status<'py>(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        let instance = self.instance.clone();

        let worlds = signal_safe_block_on(py, async move {
            instance.lock().await.world_status(Default::default()).await
        })??;
        Python::with_gil(|py| {
            let py_dict = PyDict::new_bound(py);
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
#[derive(Clone, Serialize, Deserialize, Named, PartialEq)]
pub struct PythonMessage {
    method: String,
    message: ByteBuf,
    response_port: Option<PortId>,
    rank_in_response: bool,
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
    fn bindings(&self) -> anyhow::Result<Bindings> {
        let mut bindings = Bindings::default();
        if let Some(response_port) = &self.response_port {
            bindings.push(response_port)?;
        }
        Ok(bindings)
    }
}

impl Bind for PythonMessage {
    fn bind(mut self, bindings: &Bindings) -> anyhow::Result<Self> {
        if let Some(response_port) = &mut self.response_port {
            bindings.rebind::<PortId>([response_port].into_iter())?;
        }
        Ok(self)
    }
}

#[pymethods]
impl PythonMessage {
    #[new]
    #[pyo3(signature = (method, message, response_port = None, rank_in_response = false))]
    fn new(
        method: String,
        message: Vec<u8>,
        response_port: Option<crate::mailbox::PyPortId>,
        rank_in_response: bool,
    ) -> Self {
        Self {
            method,
            message: ByteBuf::from(message),
            response_port: response_port.map(Into::into),
            rank_in_response,
        }
    }

    #[getter]
    fn method(&self) -> &String {
        &self.method
    }

    #[getter]
    fn message<'a>(&self, py: Python<'a>) -> Bound<'a, PyBytes> {
        PyBytes::new_bound(py, self.message.as_ref())
    }

    #[getter]
    fn response_port(&self) -> Option<crate::mailbox::PyPortId> {
        self.response_port.clone().map(Into::into)
    }

    #[getter]
    fn rank_in_response(&self) -> bool {
        self.rank_in_response
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

#[derive(Debug)]
#[hyperactor::export_spawn(PythonMessage, Cast<PythonMessage>, IndexedErasedUnbound<Cast<PythonMessage>>)]
pub(super) struct PythonActor {
    pub(super) actor: PyObject,
}

#[async_trait]
impl Actor for PythonActor {
    type Params = PickledPyObject;

    async fn new(actor_type: PickledPyObject) -> Result<Self, anyhow::Error> {
        Ok(Python::with_gil(|py| -> PyResult<Self> {
            let unpickled = actor_type.unpickle(py)?;
            let class_type: &Bound<'_, PyType> = unpickled.downcast()?;
            let actor: PyObject = class_type.call0()?.to_object(py);

            Ok(Self { actor })
        })?)
    }
}

/// Get the event loop state to run PythonActor handlers in. We construct a
/// fresh event loop in its own thread for us to schedule this work onto, to
/// avoid disturbing any event loops that the user might be running.
fn get_task_locals(py: Python) -> &'static pyo3_async_runtimes::TaskLocals {
    static TASK_LOCALS: OnceLock<pyo3_async_runtimes::TaskLocals> = OnceLock::new();

    // Temporarily release the GIL (as the thread we are about to create needs it).
    Python::allow_threads(py, || {
        TASK_LOCALS.get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::channel();
            let _ = std::thread::spawn(move || {
                Python::with_gil(|py| {
                    let asyncio = Python::import_bound(py, "asyncio").unwrap();
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
        })
    })
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

// Drive `future` to completion, but listen to `side_channel` and error out if
// there's we hear anything from it.
async fn drive_future(
    future: impl Future<Output = PyResult<PyObject>> + Send,
    side_channel: oneshot::Receiver<PyObject>,
) -> anyhow::Result<()> {
    let err_or_never = async {
        match side_channel.await {
            Ok(value) => Python::with_gil(|py| -> anyhow::Result<()> {
                let err: PyErr = value
                    .downcast_bound::<PyBaseException>(py)
                    .unwrap()
                    .clone()
                    .into();
                match err.traceback_bound(py) {
                    None => Err(anyhow::anyhow!("{} <no traceback available>", err)),
                    Some(traceback) => Err(anyhow::anyhow!("{}: {}", err, traceback.format()?)),
                }
            }),
            // An Err means that the sender has been dropped without sending.
            // That's okay, it just means that the Python task has completed.
            // In that case, just never resolve this future. We expect the other
            // branch of the select to finish eventually.
            Err(_) => pending().await,
        }
    };
    tokio::select! {
        result = future => {
            result?;
        },
        result = err_or_never => {
            result?;
        }
    }
    Ok(())
}

#[async_trait]
impl Handler<PythonMessage> for PythonActor {
    async fn handle(
        &mut self,
        this: &Instance<Self>,
        message: PythonMessage,
    ) -> anyhow::Result<()> {
        // Create a channel for signaling panics in async endpoints.
        // See [Panics in async endpoints].
        let (sender, receiver) = oneshot::channel();

        let future = Python::with_gil(|py| -> PyResult<_> {
            let mailbox = PyMailbox {
                inner: this.mailbox_for_py().clone(),
            };
            let awaitable = tokio::task::block_in_place(|| {
                self.actor.call_method_bound(
                    py,
                    "handle",
                    (
                        mailbox,
                        message,
                        PanicFlag {
                            sender: Some(sender),
                        },
                    ),
                    None,
                )
            })?;

            if awaitable.is_none(py) {
                return Ok(None);
            }
            pyo3_async_runtimes::into_future_with_locals(
                get_task_locals(py),
                awaitable.into_bound(py),
            )
            .map(Some)
        })?;

        if let Some(future) = future {
            drive_future(future, receiver).await?;
        }
        Ok(())
    }
}

#[async_trait]
impl Handler<Cast<PythonMessage>> for PythonActor {
    async fn handle(
        &mut self,
        this: &Instance<Self>,
        Cast {
            message,
            rank,
            shape,
        }: Cast<PythonMessage>,
    ) -> anyhow::Result<()> {
        // Create a channel for signaling panics in async endpoints.
        // See [Panics in async endpoints].
        let (sender, receiver) = oneshot::channel();

        let future = Python::with_gil(|py| -> PyResult<_> {
            let mailbox = PyMailbox {
                inner: this.mailbox_for_py().clone(),
            };

            let awaitable = tokio::task::block_in_place(|| {
                self.actor.call_method_bound(
                    py,
                    "handle_cast",
                    (
                        mailbox,
                        rank.0,
                        PyShape::from(shape),
                        message,
                        PanicFlag {
                            sender: Some(sender),
                        },
                    ),
                    None,
                )
            })?;

            if awaitable.is_none(py) {
                return Ok(None);
            }
            pyo3_async_runtimes::into_future_with_locals(
                get_task_locals(py),
                awaitable.into_bound(py),
            )
            .map(Some)
        })?;

        if let Some(future) = future {
            drive_future(future, receiver).await?;
        }
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
    use hyperactor::id;

    use super::*;

    #[test]
    fn test_python_message_bind_unbind() {
        let message = PythonMessage {
            method: "test".to_string(),
            message: ByteBuf::from(vec![1, 2, 3]),
            response_port: Some(id!(world[0].client[0][123])),
            rank_in_response: false,
        };
        {
            let unbound = message.clone().unbind().unwrap();
            assert_eq!(message, unbound.bind().unwrap());
        }

        let no_port_message = PythonMessage {
            response_port: None,
            ..message
        };
        {
            let unbound = no_port_message.clone().unbind().unwrap();
            assert_eq!(no_port_message, unbound.bind().unwrap());
        }
    }
}
