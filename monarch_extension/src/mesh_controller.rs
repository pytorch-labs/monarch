/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::VecDeque;
use std::iter::repeat_n;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use controller::history;
use hyperactor::ActorRef;
use hyperactor_mesh::actor_mesh::ActorMesh;
use hyperactor_mesh::actor_mesh::RootActorMesh;
use hyperactor_mesh::proc_mesh::SharedSpawnable;
use monarch_hyperactor::ndslice::PySlice;
use monarch_hyperactor::proc::InstanceWrapper;
use monarch_hyperactor::proc::PyActorId;
use monarch_hyperactor::proc::PyProc;
use monarch_hyperactor::proc_mesh::PyProcMesh;
use monarch_hyperactor::runtime::signal_safe_block_on;
use monarch_messages::client::Exception;
use monarch_messages::controller::ControllerActor;
use monarch_messages::controller::ControllerMessage;
use monarch_messages::debugger::DebuggerAction;
use monarch_messages::debugger::DebuggerActor;
use monarch_messages::debugger::DebuggerMessage;
use monarch_messages::worker::Ref;
use monarch_messages::worker::WorkerMessage;
use monarch_messages::worker::WorkerParams;
use monarch_tensor_worker::AssignRankMessage;
use monarch_tensor_worker::WorkerActor;
use ndslice::Slice;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tokio::sync::Mutex;

use crate::convert::convert;

#[pyclass(
    subclass,
    module = "monarch._rust_bindings.monarch_extension.mesh_controller"
)]
struct _Controller {
    controller_instance: Arc<Mutex<InstanceWrapper<ControllerMessage>>>,
    workers: RootActorMesh<'static, WorkerActor>,
    pending_messages: VecDeque<PyObject>,
    history: history::History,
}

impl _Controller {
    fn add_responses(
        &mut self,
        py: Python<'_>,
        responses: Vec<(
            monarch_messages::controller::Seq,
            Option<Result<hyperactor::data::Serialized, monarch_messages::client::Exception>>,
        )>,
    ) -> PyResult<()> {
        for (seq, response) in responses {
            let message = crate::client::WorkerResponse::new(seq, response);
            self.pending_messages.push_back(message.into_py(py));
        }
        Ok(())
    }
    fn fill_messages<'py>(&mut self, py: Python<'py>, timeout_msec: Option<u64>) -> PyResult<()> {
        let instance = self.controller_instance.clone();
        let result = signal_safe_block_on(py, async move {
            instance.lock().await.next_message(timeout_msec).await
        })??;
        result.map(|m| self.add_message(m)).transpose()?;
        Ok(())
    }

    fn add_message(&mut self, message: ControllerMessage) -> PyResult<()> {
        Python::with_gil(|py| -> PyResult<()> {
            match message {
                ControllerMessage::DebuggerMessage {
                    debugger_actor_id,
                    action,
                } => {
                    let dm = crate::client::DebuggerMessage::new(debugger_actor_id.into(), action)?
                        .into_py(py);
                    self.pending_messages.push_back(dm);
                }
                ControllerMessage::Status {
                    seq,
                    worker_actor_id,
                    controller: false,
                } => {
                    let rank = worker_actor_id.rank();
                    let responses = self.history.rank_completed(rank, seq);
                    self.add_responses(py, responses)?;
                }
                ControllerMessage::RemoteFunctionFailed { seq, error } => {
                    self.history
                        .propagate_exception(seq, Exception::Error(seq, seq, error));
                }
                ControllerMessage::FetchResult { seq, value } => {
                    self.history.set_result(seq, value);
                }
                message => {
                    panic!("unexpected message: {:?}", message);
                }
            };
            Ok(())
        })
    }
    fn send_slice(&mut self, slice: Slice, message: WorkerMessage) -> PyResult<()> {
        self.workers
            .cast_slices(vec![slice], message)
            .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))
        // let shape = Shape::new(
        //     (0..slice.sizes().len()).map(|i| format!("d{i}")).collect(),
        //     slice,
        // )
        // .unwrap();
        // println!("SENDING TO {:?} {:?}", &shape, &message);
        // let worker_slice = SlicedActorMesh::new(&self.workers, shape);
        // worker_slice
        //     .cast(ndslice::Selection::True, message)
        //     .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))
    }
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

#[pymethods]
impl _Controller {
    #[new]
    fn new(py: Python, py_proc_mesh: &PyProcMesh) -> PyResult<Self> {
        let proc_mesh = py_proc_mesh.inner.as_ref();
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let controller_instance: InstanceWrapper<ControllerMessage> = InstanceWrapper::new(
            &PyProc::new_from_proc(proc_mesh.client_proc().clone()),
            &format!("tensor_engine_controller_{}", id),
        )?;

        let controller_actor_ref =
            ActorRef::<ControllerActor>::attest(controller_instance.actor_id().clone());

        let slice = proc_mesh.shape().slice();
        if !slice.is_contiguous() || slice.offset() != 0 {
            return Err(PyValueError::new_err(
                "NYI: proc mesh for workers must be contiguous and start at offset 0",
            ));
        }
        let world_size = slice.len();
        let param = WorkerParams {
            world_size,
            // Rank assignment is consistent with proc indices.
            rank: 0,
            device_index: Some(0),
            controller_actor: controller_actor_ref,
        };

        let py_proc_mesh = Arc::clone(&py_proc_mesh.inner);
        let workers: anyhow::Result<RootActorMesh<'_, WorkerActor>> =
            signal_safe_block_on(py, async move {
                let workers = py_proc_mesh
                    .spawn(&format!("tensor_engine_workers_{}", id), &param)
                    .await?;
                //workers.cast(ndslice::Selection::True, )?;
                workers.cast_slices(
                    vec![py_proc_mesh.shape().slice().clone()],
                    AssignRankMessage::AssignRank(),
                )?;
                Ok(workers)
            })?;
        Ok(Self {
            workers: workers?,
            controller_instance: Arc::new(Mutex::new(controller_instance)),
            pending_messages: VecDeque::new(),
            history: history::History::new(world_size),
        })
    }

    fn node<'py>(
        &mut self,
        seq: u64,
        defs: Bound<'py, PyAny>,
        uses: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let failures = self.history.add_invocation(
            seq.into(),
            uses.try_iter()?
                .map(|x| Ref::from_py_object(&x?))
                .collect::<PyResult<Vec<Ref>>>()?,
            defs.try_iter()?
                .map(|x| Ref::from_py_object(&x?))
                .collect::<PyResult<Vec<Ref>>>()?,
        );
        self.add_responses(defs.py(), failures)?;
        Ok(())
    }

    fn drop_refs(&mut self, refs: Vec<Ref>) -> Result<(), anyhow::Error> {
        self.history.delete_invocations_for_refs(refs);
        Ok(())
    }

    fn send<'py>(&mut self, ranks: Bound<'py, PyAny>, message: Bound<'py, PyAny>) -> PyResult<()> {
        let message: WorkerMessage = convert(message)?;
        if let Ok(slice) = ranks.extract::<PySlice>() {
            self.send_slice(slice.into(), message)?;
        } else {
            let slices = ranks.extract::<Vec<PySlice>>()?;
            for (slice, message) in slices.iter().zip(repeat_n(message, slices.len())) {
                self.send_slice(slice.into(), message)?;
            }
        };
        Ok(())
    }

    #[pyo3(signature = (*, timeout_msec = None))]
    fn _get_next_message<'py>(
        &mut self,
        py: Python<'py>,
        timeout_msec: Option<u64>,
    ) -> PyResult<Option<PyObject>> {
        if self.pending_messages.is_empty() {
            self.fill_messages(py, timeout_msec)?;
        }
        Ok(self.pending_messages.pop_front())
    }

    fn _debugger_attach(&mut self, pdb_actor: PyActorId) -> PyResult<()> {
        let pdb_actor: ActorRef<DebuggerActor> = ActorRef::attest(pdb_actor.into());
        pdb_actor
            .send(
                self.controller_instance.blocking_lock().mailbox(),
                DebuggerMessage::Action {
                    action: DebuggerAction::Attach(),
                },
            )
            .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
        Ok(())
    }

    fn _debugger_write(&mut self, pdb_actor: PyActorId, bytes: Vec<u8>) -> PyResult<()> {
        let pdb_actor: ActorRef<DebuggerActor> = ActorRef::attest(pdb_actor.into());
        pdb_actor
            .send(
                self.controller_instance.blocking_lock().mailbox(),
                DebuggerMessage::Action {
                    action: DebuggerAction::Write { bytes },
                },
            )
            .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
        Ok(())
    }
    fn _drain_and_stop(&mut self, py: Python<'_>) -> PyResult<()> {
        self.send_slice(
            self.workers.proc_mesh().shape().slice().clone(),
            WorkerMessage::Exit { error: None },
        )?;
        let instance = self.controller_instance.clone();
        let _ = signal_safe_block_on(py, async move { instance.lock().await.drain_and_stop() })??;
        Ok(())
    }
}

pub(crate) fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<_Controller>()?;
    Ok(())
}
