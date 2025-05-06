use std::sync::Arc;
use std::sync::OnceLock;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::ActorId;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor_mesh::actor_mesh::Cast;
use monarch_types::PickledPyObject;
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

use crate::mailbox::PyMailbox;
use crate::proc::InstanceWrapper;
use crate::proc::PyActorId;
use crate::proc::PyProc;
use crate::proc::PySerialized;
use crate::runtime::signal_safe_block_on;
use crate::shape::PyShape;

#[pyclass(frozen, module = "monarch._monarch.hyperactor")]
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

#[pyclass(module = "monarch._monarch.hyperactor")]
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

#[pyclass(frozen, module = "monarch._monarch.hyperactor")]
#[derive(Clone, Serialize, Deserialize, Named)]
pub struct PythonMessage {
    method: String,
    message: ByteBuf,
}

impl std::fmt::Debug for PythonMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonMessage")
            .field(
                "message",
                &hyperactor::data::HexFmt(self.message.as_slice()).to_string(),
            )
            .finish()
    }
}

#[pymethods]
impl PythonMessage {
    #[new]
    #[pyo3(signature = (method, message))]
    fn new(method: String, message: Vec<u8>) -> Self {
        Self {
            method,
            message: ByteBuf::from(message),
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
}

#[pyclass(module = "monarch._monarch.hyperactor")]
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
#[hyperactor::export_spawn(PythonMessage, Cast<PythonMessage>)]
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

#[async_trait]
impl Handler<PythonMessage> for PythonActor {
    async fn handle(
        &mut self,
        this: &Instance<Self>,
        message: PythonMessage,
    ) -> anyhow::Result<()> {
        let future = Python::with_gil(|py| -> PyResult<_> {
            let mailbox = PyMailbox {
                inner: this.mailbox_for_py().clone(),
            };
            let awaitable = self
                .actor
                .call_method_bound(py, "handle", (mailbox, message), None)?;
            let task_locals = TASK_LOCALS
                .get()
                .ok_or_else(|| PyRuntimeError::new_err("init_asyncio_loop not called"))?;
            pyo3_async_runtimes::into_future_with_locals(task_locals, awaitable.into_bound(py))
        })?;
        future.await?;
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
        let future = Python::with_gil(|py| -> PyResult<_> {
            let mailbox = PyMailbox {
                inner: this.mailbox_for_py().clone(),
            };

            let awaitable = self.actor.call_method_bound(
                py,
                "handle_cast",
                (mailbox, rank, PyShape::from(shape), message),
                None,
            )?;
            let task_locals = TASK_LOCALS
                .get()
                .ok_or_else(|| PyRuntimeError::new_err("init_asyncio_loop not called"))?;
            pyo3_async_runtimes::into_future_with_locals(task_locals, awaitable.into_bound(py))
        })?;
        future.await?;
        Ok(())
    }
}

static TASK_LOCALS: OnceLock<pyo3_async_runtimes::TaskLocals> = OnceLock::new();

#[pyfunction]
#[pyo3(signature = ())]
pub fn init_asyncio_loop(py: Python<'_>) -> PyResult<()> {
    let _ =
        TASK_LOCALS.set(pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)?);
    Ok(())
}
