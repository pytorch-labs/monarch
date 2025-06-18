/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::iter::repeat_n;
use std::sync;
use std::sync::Arc;
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;

use hyperactor::ActorRef;
use hyperactor::data::Serialized;
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
use monarch_messages::controller::Seq;
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
    history: History,
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
                    let responses = self
                        .history
                        .propagate_exception(seq, Exception::Error(seq, seq, error));
                    self.add_responses(py, responses)?;
                }
                ControllerMessage::FetchResult {
                    seq,
                    value: Ok(value),
                } => {
                    self.history.set_result(seq, value);
                }
                ControllerMessage::FetchResult {
                    seq,
                    value: Err(error),
                } => {
                    let responses = self
                        .history
                        .propagate_exception(seq, Exception::Error(seq, seq, error));
                    self.add_responses(py, responses)?;
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
        let id = NEXT_ID.fetch_add(1, atomic::Ordering::Relaxed);
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
            history: History::new(world_size),
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

    fn drop_refs(&mut self, refs: Vec<Ref>) {
        self.history.drop_refs(refs);
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

/// An invocation tracks a discrete node in the graph of operations executed by
/// the worker based on instructions from the client.
/// It is useful for tracking the dependencies of an operation and propagating
/// failures. In the future this will be used with more data dependency tracking
/// to support better failure handling.

#[derive(Debug)]
enum Status {
    Errored(Exception),
    Complete(),

    /// When incomplete this holds this list of users of this invocation,
    /// so a future error can be propagated to them.,
    Incomplete(HashMap<Seq, Arc<sync::Mutex<Invocation>>>),
}
#[derive(Debug)]
struct Invocation {
    /// The sequence number of the invocation. This should be unique and increasing across all
    /// invocations.
    seq: Seq,
    status: Status,
    /// Result reported to a future if this invocation was a fetch
    /// Not all Invocations will be fetched so sometimes a Invocation will complete with
    /// both result and error == None
    result: Option<Serialized>,
}

impl Invocation {
    fn new(seq: Seq) -> Self {
        Self {
            seq,
            status: Status::Incomplete(HashMap::new()),
            result: None,
        }
    }

    fn add_user(&mut self, user: Arc<sync::Mutex<Invocation>>) {
        match &mut self.status {
            Status::Complete() => {}
            Status::Incomplete(users) => {
                let seq = user.lock().unwrap().seq;
                users.insert(seq, user);
            }
            Status::Errored(err) => {
                user.lock().unwrap().set_exception(err.clone());
            }
        }
    }

    /// Invocation results can only go from valid to failed, or be
    /// set if the invocation result is empty.
    fn set_result(&mut self, result: Serialized) {
        if self.result.is_none() {
            self.result = Some(result);
        }
    }

    fn succeed(&mut self) {
        match self.status {
            Status::Incomplete(_) => self.status = Status::Complete(),
            _ => {}
        }
    }

    fn set_exception(&mut self, exception: Exception) -> Vec<Arc<sync::Mutex<Invocation>>> {
        match exception {
            Exception::Error(_, caused_by_new, error) => {
                let err = Status::Errored(Exception::Error(self.seq, caused_by_new, error));
                match &self.status {
                    Status::Errored(Exception::Error(_, caused_by_current, _))
                        if caused_by_new < *caused_by_current =>
                    {
                        self.status = err;
                    }
                    Status::Incomplete(users) => {
                        let users = users.values().cloned().collect();
                        self.status = err;
                        return users;
                    }
                    Status::Complete() => {
                        panic!("Complete invocation getting an exception set")
                    }
                    _ => {}
                }
            }
            Exception::Failure(_) => {
                tracing::error!(
                    "system failures {:?} can never be assigned for an invocation",
                    exception
                );
            }
        }
        return vec![];
    }

    fn msg_result(&self) -> Option<Result<Serialized, Exception>> {
        match &self.status {
            Status::Complete() => self.result.clone().map(|x| Ok(x)),
            Status::Errored(err) => Some(Err(err.clone())),
            Status::Incomplete(_) => {
                panic!("Incomplete invocation doesn't have a result yet")
            }
        }
    }
}

/// The history of invocations sent by the client to be executed on the workers.
/// This is used to track dependencies between invocations and to propagate exceptions.
/// It purges history for completed invocations to avoid memory bloat.
/// TODO: Revisit this setup around purging refs automatically once we start doing
/// more complex data dependency tracking. We will want to be more aware of things like
/// borrows, drops etc. directly.
#[derive(Debug)]
struct History {
    /// The first incomplete Seq for each rank. This is used to determine which
    /// Seqs are no longer relevant and can be purged from the history.
    first_incomplete_seqs: MinVector<Seq>,
    /// The minimum incomplete Seq across all ranks.
    min_incomplete_seq: Seq,
    /// A map of seq to the invocation that it represents for all seq >= min_incomplete_seq
    inflight_invocations: HashMap<Seq, Arc<sync::Mutex<Invocation>>>,
    /// A map of reference to the seq for the invocation that defines it. This is used to
    /// compute dependencies between invocations.
    invocation_for_ref: HashMap<Ref, Arc<sync::Mutex<Invocation>>>,
    // no new sequence numbers should be below this bound. use for
    // sanity checking.
    seq_lower_bound: Seq,
}

/// A vector that keeps track of the minimum value.
#[derive(Debug)]
struct MinVector<T> {
    data: Vec<T>,
    value_counts: BTreeMap<T, usize>,
}

impl<T> MinVector<T>
where
    T: Ord + Copy,
{
    fn new(data: Vec<T>) -> Self {
        let mut value_counts = BTreeMap::new();
        for &value in &data {
            *value_counts.entry(value).or_insert(0) += 1;
        }
        MinVector { data, value_counts }
    }

    fn set(&mut self, index: usize, value: T) {
        // Decrease the count of the old value
        let old_value = self.data[index];
        if let Some(count) = self.value_counts.get_mut(&old_value) {
            *count -= 1;
            if *count == 0 {
                self.value_counts.remove(&old_value);
            }
        }
        // Update the value in the vector
        self.data[index] = value;

        // Increase the count of the new value
        *self.value_counts.entry(value).or_insert(0) += 1;
    }

    fn min(&self) -> T {
        *self.value_counts.keys().next().unwrap()
    }
}

impl History {
    pub fn new(world_size: usize) -> Self {
        Self {
            first_incomplete_seqs: MinVector::new(vec![Seq::default(); world_size]),
            min_incomplete_seq: Seq::default(),
            invocation_for_ref: HashMap::new(),
            inflight_invocations: HashMap::new(),
            seq_lower_bound: 0.into(),
        }
    }

    #[cfg(test)]
    pub fn first_incomplete_seqs(&self) -> &[Seq] {
        self.first_incomplete_seqs.vec()
    }

    pub fn drop_refs(&mut self, refs: Vec<Ref>) {
        for r in refs {
            self.invocation_for_ref.remove(&r);
        }
    }

    /// Add an invocation to the history.
    pub fn add_invocation(
        &mut self,
        seq: Seq,
        uses: Vec<Ref>,
        defs: Vec<Ref>,
    ) -> Vec<(Seq, Option<Result<Serialized, Exception>>)> {
        assert!(
            seq >= self.seq_lower_bound,
            "nonmonotonic seq: {:?}; current lower bound: {:?}",
            seq,
            self.seq_lower_bound,
        );
        self.seq_lower_bound = seq;
        let invocation = Arc::new(sync::Mutex::new(Invocation::new(seq)));
        self.inflight_invocations.insert(seq, invocation.clone());
        for ref use_ in uses {
            let producer = self.invocation_for_ref.get(use_).unwrap();
            producer.lock().unwrap().add_user(invocation.clone());
        }

        for def in defs {
            self.invocation_for_ref.insert(def, invocation.clone());
        }
        let invocation = invocation.lock().unwrap();
        if matches!(invocation.status, Status::Errored(_)) {
            vec![(seq, invocation.msg_result())]
        } else {
            vec![]
        }
    }

    /// Propagate worker error to the invocation with the given Seq. This will also propagate
    /// to all seqs that depend on this seq directly or indirectly.
    pub fn propagate_exception(
        &mut self,
        seq: Seq,
        exception: Exception,
    ) -> Vec<(Seq, Option<Result<Serialized, Exception>>)> {
        let mut results = Vec::new();
        let invocation = self.inflight_invocations.get(&seq).unwrap().clone();

        let mut queue: Vec<Arc<sync::Mutex<Invocation>>> = vec![invocation];
        let mut visited = HashSet::new();

        while let Some(invocation) = queue.pop() {
            let mut invocation = invocation.lock().unwrap();
            if !visited.insert(invocation.seq) {
                continue;
            };
            queue.extend(invocation.set_exception(exception.clone()));
            results.push((seq, invocation.msg_result()));
        }
        results
    }

    /// Mark the given rank as completed up to but excluding the given Seq. This will also purge history for
    /// any Seqs that are no longer relevant (completed on all ranks).
    pub fn rank_completed(
        &mut self,
        rank: usize,
        seq: Seq,
    ) -> Vec<(Seq, Option<Result<Serialized, Exception>>)> {
        self.first_incomplete_seqs.set(rank, seq);
        let prev = self.min_incomplete_seq;
        self.min_incomplete_seq = self.first_incomplete_seqs.min();

        let mut results: Vec<(Seq, Option<Result<Serialized, Exception>>)> = Vec::new();
        for i in Seq::iter_between(prev, self.min_incomplete_seq) {
            let invocation = self.inflight_invocations.remove(&i).unwrap();
            let mut invocation = invocation.lock().unwrap();

            if matches!(invocation.status, Status::Errored(_)) {
                // we already reported output early when it errored
                continue;
            }
            invocation.succeed();
            results.push((i, invocation.msg_result()));
        }
        results
    }

    pub fn set_result(&mut self, seq: Seq, result: Serialized) {
        let invocation = self.inflight_invocations.get(&seq).unwrap();
        invocation.lock().unwrap().set_result(result);
    }
}
