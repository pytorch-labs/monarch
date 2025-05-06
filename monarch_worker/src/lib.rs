#![feature(assert_matches)]
#![feature(duration_constructors)]
#![feature(exit_status_error)]
// NOTE: Until https://github.com/PyO3/pyo3/pull/4674, `pyo3::pymethods` trigger
// and unsafe-op-in-unsafe-fn warnings.
#![allow(unsafe_op_in_unsafe_fn)]

//! A `hyperactor`-based implementation of a PyTorch worker actor.
//!
//! The worker is responsible for executing PyTorch operations on a local
//! device. It assumes it has exclusive access to device resources, and manages
//! concurrency internally via device-specific constructs (CUDA stream, threads,
//! etc.).
//!
//! This is a port of `monarch/python/controller/worker.py` but does have gaps due
//! to drift that needs to be reconciled.
//! This mainly includes:
//! - Support for record and replay
//! - debugger support
//! - general drift in exisitng messages

pub mod bootstrap;
mod borrow;
mod comm;
pub mod device_mesh;
pub mod pipe;
pub mod py_pipe;
pub mod stream;
pub mod test_util;

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use anyhow::ensure;
use async_trait::async_trait;
use borrow::Borrow;
use comm::CommMessageClient;
use comm::CommParams;
use comm::NcclCommActor;
use derive_more::TryInto;
use device_mesh::DeviceMesh;
use futures::future::try_join_all;
use hyperactor::Actor;
use hyperactor::ActorRef;
use hyperactor::Instance;
use hyperactor::actor::ActorHandle;
use hyperactor::forward;
use hyperactor::reference::ActorId;
use hyperactor::supervision::ActorSupervisionEvent;
use itertools::Itertools;
use monarch_messages::controller::ControllerActor;
use monarch_messages::controller::ControllerMessageClient;
use monarch_messages::controller::Seq;
use monarch_messages::wire_value::WireValue;
use monarch_messages::worker::CallFunctionError;
use monarch_messages::worker::*;
use monarch_types::PyTree;
use ndslice::Slice;
use pipe::PipeActor;
use pipe::PipeParams;
use sorted_vec::SortedVec;
use stream::StreamActor;
use stream::StreamMessageClient;
use stream::StreamParams;
use torch_sys::CudaDevice;
use torch_sys::DeviceIndex;
use torch_sys::Layout;
use torch_sys::RValue;
use torch_sys::ScalarType;
use torch_sys::TensorCell;
use torch_sys::factory_zeros;
use torch_sys::nccl::NcclConfig;
use torch_sys::nccl::ReduceOp;
use torch_sys::nccl::UniqueId;

#[derive(Debug)]
struct RemoteProcessGroupState {
    device_mesh_ref: Ref,
    dims: SortedVec<String>,
    comms: HashMap<StreamRef, Arc<ActorHandle<NcclCommActor>>>,
}

impl RemoteProcessGroupState {
    fn new(device_mesh_ref: Ref, dims: SortedVec<String>) -> Self {
        Self {
            device_mesh_ref,
            dims,
            comms: HashMap::new(),
        }
    }
}

/// A PyTorch runtime instance, operating on a single accelerator device,
/// controlled via hyperactor messaging.
///
/// Generally this is a thin multiplexer over a set of [`Stream`]s that do the
/// real work.
///
/// See [`WorkerMessage`] for what it can do!
#[derive(Debug)]
#[hyperactor::export_spawn(WorkerMessage)]
pub struct WorkerActor {
    device: Option<CudaDevice>,
    streams: HashMap<StreamRef, Arc<ActorHandle<StreamActor>>>,
    /// Maps streams to the device mesh and a map of dim names to the concrete
    /// communicator actor that represents the dimension for that stream.
    device_meshes: HashMap<
        Ref,
        (
            DeviceMesh,
            // A map for comms for this mesh for a given pair of stream and dims.
            HashMap<(StreamRef, SortedVec<String>), (usize, Arc<ActorHandle<NcclCommActor>>)>,
        ),
    >,
    world_size: usize,
    rank: usize,
    borrows: HashMap<u64, Borrow>,
    comm: Option<ActorHandle<NcclCommActor>>,
    controller_actor: ActorRef<ControllerActor>,
    /// Pipes created for the worker.
    pipes: HashMap<Ref, Result<ActorHandle<PipeActor>, Arc<CallFunctionError>>>,
    /// Remember the process groups "created" via `CreateRemoteProcessGroup` for
    /// subsequent `CallFunction` calls, as this is where the actual allocation
    /// will happen.
    remote_process_groups: HashMap<Ref, RemoteProcessGroupState>,
    /// The comm actor for each pair of streams that need to send/recv tensors.
    send_recv_comms: HashMap<(StreamRef, StreamRef), Arc<ActorHandle<NcclCommActor>>>,
}

#[async_trait]
impl Actor for WorkerActor {
    type Params = WorkerParams;

    async fn new(
        WorkerParams {
            world_size,
            rank,
            device_index,
            controller_actor,
        }: Self::Params,
    ) -> Result<Self> {
        Ok(Self {
            device: device_index.map(|i| CudaDevice::new(DeviceIndex(i))),
            streams: HashMap::new(),
            device_meshes: HashMap::new(),
            world_size,
            rank,
            borrows: HashMap::new(),
            comm: None,
            controller_actor,
            pipes: HashMap::new(),
            remote_process_groups: HashMap::new(),
            send_recv_comms: HashMap::new(),
        })
    }

    async fn handle_supervision_event(
        &mut self,
        _this: &Instance<Self>,
        _event: &ActorSupervisionEvent,
    ) -> Result<bool, anyhow::Error> {
        // Exit the worker directly on any worker actor errors, with error exit code.
        tracing::info!("worker error happened, stop the worker process, exit code: 1");
        std::process::exit(1);
    }
}

#[async_trait]
#[forward(WorkerMessage)]
impl WorkerMessageHandler for WorkerActor {
    async fn backend_network_init(
        &mut self,
        this: &Instance<Self>,
        unique_id: UniqueId,
    ) -> Result<()> {
        let device = self
            .device
            .expect("tried to init backend network on a non-CUDA worker");
        let comm = NcclCommActor::spawn(
            this,
            CommParams::New {
                device,
                unique_id,
                world_size: self.world_size.try_into().unwrap(),
                rank: self.rank.try_into().unwrap(),
            },
        )
        .await?;

        let tensor = factory_zeros(&[1], ScalarType::Float, Layout::Strided, device.into());
        let cell = TensorCell::new(tensor);

        comm.all_reduce(
            this,
            cell,
            ReduceOp::Sum,
            torch_sys::cuda::Stream::get_current_stream(),
        )
        .await?;

        // TODO: this blocks forward progress of the the actor loop while we
        // wait for the streams to catch up. Once we have a way of spawning
        // tasks that the actor system can monitor in a non-blocking way, we
        // should remove this.

        // We need to be careful to initialize the streams in a consistent order
        // across all workers to avoid NCCL deadlocks. Use the refs to provide
        // this order, as a stream's ref will be the same across all workers.
        let sorted_streams = self
            .streams
            .iter()
            .sorted_by_key(|(k, _)| *k)
            .map(|(_, v)| v.as_ref());

        let mut splits = Vec::new();
        for _ in 0..sorted_streams.len() {
            // Do the split in this event loop, to provide a deterministic
            // order.
            splits.push(comm.split_all(this, None).await?);
        }
        let _: Vec<()> = try_join_all(
            sorted_streams
                .into_iter()
                .zip(splits.into_iter())
                .map(|(stream, split)| stream.init_comm(this, split)),
        )
        .await?;

        self.comm = Some(comm);

        Ok(())
    }

    async fn backend_network_point_to_point_init(
        &mut self,
        this: &Instance<Self>,
        from_stream: StreamRef,
        to_stream: StreamRef,
    ) -> Result<()> {
        if !self.streams.contains_key(&from_stream) {
            bail!("invalid from_stream id: {:#?}", from_stream);
        }
        if !self.streams.contains_key(&to_stream) {
            bail!("invalid to_stream id: {:#?}", to_stream);
        }
        let global_comm = self
            .comm
            .as_ref()
            .context("tried to call Reduce before BackendNetworkInit")?;
        let comm = global_comm.split_all(this, None).await?;
        self.send_recv_comms
            .insert((from_stream, to_stream), Arc::new(comm));
        Ok(())
    }

    async fn call_function(
        &mut self,
        this: &Instance<Self>,
        params: CallFunctionParams,
    ) -> Result<()> {
        let stream = self
            .streams
            .get(&params.stream)
            .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", params.stream))?
            .clone();

        let device_meshes = if params.function.as_torch_op().is_some() {
            HashMap::new()
        } else {
            self.device_meshes
                .iter()
                .map(|(k, v)| (k.clone(), v.0.clone()))
                .collect()
        };

        let mut remote_process_groups = HashMap::new();
        for remote_process_group_ref in &params.remote_process_groups {
            if let Some(state) = self.remote_process_groups.get(remote_process_group_ref) {
                let dims_vec = state.dims.iter().cloned().collect();
                let (device_mesh, _) = self
                    .device_meshes
                    .get(&state.device_mesh_ref)
                    .ok_or_else(|| {
                        anyhow::anyhow!("invalid device mesh id: {:#?}", state.device_mesh_ref)
                    })?
                    .clone();
                let comm = state.comms
                    .get(&params.stream)
                    .ok_or_else(|| {
                        anyhow::anyhow!("no comm found for remote process group {remote_process_group_ref:#?} stream {stream:#?}")
                    })?
                    .clone();
                remote_process_groups.insert(
                    remote_process_group_ref.clone(),
                    (device_mesh, dims_vec, comm),
                );
            }
        }

        stream
            .call_function(this, params, device_meshes, remote_process_groups)
            .await?;

        Ok(())
    }

    async fn command_group(
        &mut self,
        this: &Instance<Self>,
        params: Vec<WorkerMessage>,
    ) -> Result<()> {
        for msg in params {
            WorkerMessageHandler::handle(self, this, msg).await?;
        }
        Ok(())
    }

    async fn create_stream(
        &mut self,
        this: &Instance<Self>,
        result: StreamRef,
        creation_mode: StreamCreationMode,
    ) -> Result<()> {
        let handle: ActorHandle<StreamActor> = StreamActor::spawn(
            this,
            StreamParams {
                world_size: self.world_size,
                rank: self.rank,
                creation_mode,
                id: result,
                device: self.device,
                controller_actor: self.controller_actor.clone(),
            },
        )
        .await?;
        self.streams.insert(result, Arc::new(handle));
        Ok(())
    }

    async fn create_device_mesh(
        &mut self,
        _this: &Instance<Self>,
        result: Ref,
        names: Vec<String>,
        ranks: Slice,
    ) -> Result<()> {
        self.device_meshes.insert(
            result,
            (DeviceMesh::new(names, ranks, self.rank)?, HashMap::new()),
        );
        Ok(())
    }

    async fn create_remote_process_group(
        &mut self,
        _this: &Instance<Self>,
        result: Ref,
        device_mesh: Ref,
        dims: Vec<String>,
    ) -> Result<()> {
        self.device_meshes
            .get(&device_mesh)
            .with_context(|| format!("invalid device mesh id: {:#?}", device_mesh))?;
        match self.remote_process_groups.entry(result) {
            Entry::Vacant(ent) => ent.insert(RemoteProcessGroupState::new(
                device_mesh,
                SortedVec::from_unsorted(dims),
            )),
            Entry::Occupied(ent) => bail!("remote process group {:?} already create", ent.key()),
        };
        Ok(())
    }

    async fn borrow_create(
        &mut self,
        this: &Instance<Self>,
        result: Ref,
        borrow_id: u64,
        tensor_ref: Ref,
        from_stream: StreamRef,
        to_stream: StreamRef,
    ) -> Result<()> {
        let from_stream = self
            .streams
            .get(&from_stream)
            .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", &from_stream))?
            .clone();
        let to_stream = self
            .streams
            .get(&to_stream)
            .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", &to_stream))?
            .clone();

        let borrow =
            Borrow::create(this, borrow_id, tensor_ref, result, from_stream, to_stream).await?;
        self.borrows.insert(borrow_id, borrow);
        Ok(())
    }

    async fn borrow_first_use(&mut self, this: &Instance<Self>, borrow: u64) -> Result<()> {
        let borrow = self
            .borrows
            .get_mut(&borrow)
            .ok_or_else(|| anyhow!("invalid borrow id: {:#?}", borrow))?;

        borrow.first_use(this).await?;
        Ok(())
    }

    async fn borrow_last_use(&mut self, this: &Instance<Self>, borrow: u64) -> Result<()> {
        let borrow = self
            .borrows
            .get_mut(&borrow)
            .ok_or_else(|| anyhow::anyhow!("invalid borrow id: {:#?}", borrow))?;

        borrow.last_use(this).await?;
        Ok(())
    }

    async fn borrow_drop(&mut self, this: &Instance<Self>, borrow_id: u64) -> Result<()> {
        let borrow = self
            .borrows
            .get_mut(&borrow_id)
            .ok_or_else(|| anyhow::anyhow!("invalid borrow id: {:#?}", borrow_id))?;

        borrow.drop(this).await?;
        self.borrows.remove(&borrow_id);
        Ok(())
    }

    async fn delete_refs(&mut self, this: &Instance<Self>, refs: Vec<Ref>) -> Result<()> {
        // Fan the delete message to all streams.
        // Check for errors.
        // TODO: this blocks forward progress of the the actor loop while we
        // wait for the streams to catch up. Once we have a way of spawning
        // tasks that the actor system can monitor in a non-blocking way, we
        // should remove this.
        let _: Vec<()> = try_join_all(
            self.streams
                .values()
                .map(|s| s.delete_refs(this, refs.clone())),
        )
        .await?;
        Ok(())
    }

    async fn request_status(
        &mut self,
        this: &Instance<Self>,
        seq: Seq,
        controller: bool,
    ) -> Result<()> {
        // TODO: this blocks forward progress of the the actor loop while we
        // wait for the streams to catch up. Once we have a way of spawning
        // tasks that the actor system can monitor in a non-blocking way, we
        // should remove this.
        let _: Vec<()> = try_join_all(
            self.streams
                .values()
                .map(|stream| stream.request_status(this)),
        )
        .await?;

        ControllerMessageClient::status(
            &self.controller_actor,
            this,
            seq.next(),
            this.self_id().clone(),
            controller,
        )
        .await?;
        Ok(())
    }

    async fn reduce(
        &mut self,
        this: &Instance<Self>,
        result: Ref,
        local_tensor: Ref,
        factory: Factory,
        source_mesh: Ref,
        stream_ref: StreamRef,
        dims: Vec<String>,
        reduction: Reduction,
        scatter: bool,
        in_place: bool,
        out: Option<Ref>,
    ) -> Result<()> {
        // Sort for stable indexing.
        let dims = SortedVec::from_unsorted(dims);
        let stream = self
            .streams
            .get(&stream_ref)
            .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", stream_ref))?
            .clone();

        let (_, comm_map) = self
            .device_meshes
            .get_mut(&source_mesh)
            .ok_or_else(|| anyhow::anyhow!("invalid device mesh id: {:#?}", source_mesh))?;

        let (size, comm) = comm_map
            .get(&(stream_ref, dims.clone()))
            .ok_or_else(|| anyhow::anyhow!("no comm found for stream {stream:#?}, dims {dims:#?}"))?
            .clone();

        stream
            .reduce(
                this,
                comm,
                size.try_into()?,
                result,
                local_tensor,
                factory,
                reduction,
                scatter,
                in_place,
                out,
            )
            .await?;

        Ok(())
    }

    async fn create_pipe(
        &mut self,
        this: &Instance<Self>,
        result: Ref,
        // TODO(agallagher): This is used in the python impl to name the socket
        // path to use for comms, but we don't currently use a named socket.
        _key: String,
        function: ResolvableFunction,
        max_messages: i64,
        device_mesh: Ref,
        args: Vec<WireValue>,
        kwargs: HashMap<String, WireValue>,
    ) -> Result<()> {
        self.pipes.insert(
            result,
            // We never explicitly fail.  In the event we see an error creating
            // the pipe, store the error instead.
            async {
                let args: Vec<PyTree<RValue>> = args
                    .into_iter()
                    .map(|object| RValue::PyObject(object.into_py_object().unwrap()).into())
                    .collect();
                let kwargs: HashMap<_, PyTree<RValue>> = kwargs
                    .into_iter()
                    .map(|(k, object)| {
                        (k, RValue::PyObject(object.into_py_object().unwrap()).into())
                    })
                    .collect();
                let device_mesh = self
                    .device_meshes
                    .get(&device_mesh)
                    .ok_or_else(|| CallFunctionError::RefNotFound(device_mesh))?;
                let pipe = PipeActor::spawn(
                    this,
                    PipeParams {
                        function,
                        max_messages,
                        ranks: device_mesh.0.ranks(),
                        sizes: device_mesh.0.sizes(),
                        args,
                        kwargs,
                    },
                )
                .await?;
                Ok(pipe)
            }
            .await
            .map_err(Arc::new),
        );
        Ok(())
    }

    async fn send_tensor(
        &mut self,
        this: &Instance<Self>,
        result: Ref,
        from_ranks: Slice,
        to_ranks: Slice,
        tensor: Ref,
        factory: Factory,
        from_stream: StreamRef,
        to_stream: StreamRef,
    ) -> Result<()> {
        let comm = self
            .send_recv_comms
            .get(&(from_stream, to_stream))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "could not find stream to stream comm for: {:#?}",
                    (from_stream, to_stream)
                )
            })?
            .clone();

        let to_rank = from_ranks
            .index(self.rank)
            .map(|index| to_ranks.get(index).ok())
            .ok()
            .flatten();
        let from_rank = to_ranks
            .index(self.rank)
            .map(|index| from_ranks.get(index).ok())
            .ok()
            .flatten();

        let stream = if to_rank.is_none() {
            self.streams
                .get(&to_stream)
                .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", to_stream))?
                .clone()
        } else if from_rank.is_none() {
            self.streams
                .get(&from_stream)
                .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", from_stream))?
                .clone()
        } else if from_stream == to_stream {
            self.streams
                .get(&from_stream)
                .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", from_stream))?
                .clone()
        } else {
            unimplemented!(
                "We haven't implemented to_mesh between streams if a rank participates as both a sender and receiver. \
                It is possible, but would require the recv stream to send the output buffer tensor to the send stream and sync. \
                Then the send stream would do the nccl op, and then sync with sending stream again."
            );
        };

        stream
            .send_tensor(this, result, from_rank, to_rank, tensor, factory, comm)
            .await?;

        Ok(())
    }

    async fn exit(
        &mut self,
        this: &Instance<Self>,
        error: Option<(Option<ActorId>, String)>,
    ) -> Result<()> {
        for (_, stream) in self.streams.drain() {
            stream.drain_and_stop()?;
            Arc::into_inner(stream)
                .expect("there should be no owners of this stream handle except the worker stream table")
                .await;
        }

        let self_error_exit_code = std::env::var("MONARCH_WORKER_SELF_ERROR_EXIT_CODE")
            .ok()
            .and_then(|val| val.parse::<i32>().ok())
            .unwrap_or(1);
        let peer_error_exit_code = std::env::var("MONARCH_WORKER_PEER_ERROR_EXIT_CODE")
            .ok()
            .and_then(|val| val.parse::<i32>().ok())
            .unwrap_or(1);

        // Exit the worker process if there is an error.
        let exit_code = match error {
            Some((Some(actor_id), reason)) => {
                tracing::error!(
                    "stopping the worker, actor {} failed with error: {}",
                    actor_id,
                    reason
                );
                if *this.self_id() == actor_id {
                    self_error_exit_code
                } else {
                    peer_error_exit_code
                }
            }
            Some((None, reason)) => {
                tracing::error!("stopping the worker, reason: {}", reason);
                1
            }
            None => 0,
        };

        if exit_code != 0 {
            tracing::info!("stopping the worker process, exit code: {}", exit_code);
            std::process::exit(exit_code);
        }
        Ok(())
    }

    async fn send_value(
        &mut self,
        this: &Instance<Self>,
        seq: Seq,
        destination: Option<Ref>,
        mutates: Vec<Ref>,
        function: Option<ResolvableFunction>,
        args: Vec<WireValue>,
        kwargs: HashMap<String, WireValue>,
        stream: StreamRef,
    ) -> Result<()> {
        // Resolve the stream.
        let stream = self
            .streams
            .get(&stream)
            .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", stream))?;

        let device_meshes = if function.as_ref().is_none_or(|f| f.as_torch_op().is_some()) {
            HashMap::new()
        } else {
            self.device_meshes
                .iter()
                .map(|(k, v)| (k.clone(), v.0.clone()))
                .collect()
        };

        // Resolve the pipe, if a ref to it is provided.
        let pipe = destination
            .map(|pipe| {
                self.pipes
                    .get(&pipe)
                    .ok_or_else(|| anyhow::anyhow!("invalid pipe id: {:#?}", pipe))
            })
            // TODO(agallagher): Fix error prop.
            .transpose()?
            .map(|r| r.as_ref().map_err(|e| e.clone()))
            .transpose()?
            .map(|pipe| pipe.port());

        // Resolve the value on the stream, then send the value to the pipe if provided,
        // or back to the controller if not.
        stream
            .send_value(
                this,
                seq,
                this.self_id().clone(),
                mutates,
                function,
                args,
                kwargs,
                device_meshes,
                pipe,
            )
            .await
    }

    async fn split_comm(
        &mut self,
        this: &Instance<Self>,
        dims: Vec<String>,
        device_mesh: Ref,
        stream_ref: StreamRef,
        config: Option<NcclConfig>,
    ) -> Result<()> {
        let global_comm = self
            .comm
            .as_ref()
            .context("tried to call SplitComm before BackendNetworkInit")?;
        match self.device_meshes.get_mut(&device_mesh) {
            Some((device_mesh, comm_map)) => {
                // This rank is in the group to be split off. Split a new
                // communicator for it off from the global communicator.
                let stream = self
                    .streams
                    .get(&stream_ref)
                    .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", stream_ref))?;

                let dims = SortedVec::from_unsorted(dims);

                anyhow::ensure!(
                    !comm_map.contains_key(&(stream_ref, dims.clone())),
                    "comm already exists for stream {stream:#?}, dims {dims:#?}"
                );
                let ranks_for_group = device_mesh.get_ranks_for_dim_slice(&dims)?;
                let size = ranks_for_group.len();
                let split_comm = global_comm
                    .split_from(
                        this,
                        ranks_for_group
                            .into_iter()
                            .map(|v| v.clone().try_into())
                            .collect::<Result<Vec<_>, _>>()?,
                        config,
                    )
                    .await?
                    .context("split comm should include self rank")?;
                comm_map.insert((stream_ref, dims), (size, Arc::new(split_comm)));
            }
            None => {
                // This rank is not in the group to be split off. We still need to
                // participate in the commSplit call, however.
                global_comm.split_from(this, vec![], config).await?;
            }
        }
        Ok(())
    }

    async fn split_comm_for_process_group(
        &mut self,
        this: &Instance<Self>,
        remote_process_group_ref: Ref,
        stream_ref: StreamRef,
        config: Option<NcclConfig>,
    ) -> Result<()> {
        ensure!(
            self.streams.contains_key(&stream_ref),
            "invalid stream id: {:#?}",
            stream_ref
        );
        let global_comm = self
            .comm
            .as_ref()
            .context("tried to call SplitComm before BackendNetworkInit")?;
        let state = self
            .remote_process_groups
            .get_mut(&remote_process_group_ref)
            .with_context(|| format!("invalid remote process group id: {:#?}", stream_ref))?;
        match self.device_meshes.get_mut(&state.device_mesh_ref) {
            Some((device_mesh, _)) => {
                // This rank is in the group to be split off. Split a new
                // communicator for it off from the global communicator.
                let entry = match state.comms.entry(stream_ref) {
                    Entry::Vacant(entry) => entry,
                    Entry::Occupied(_) => bail!(
                        "comm already exists for remote process group {:#?} on stream {:#?}",
                        remote_process_group_ref,
                        stream_ref,
                    ),
                };
                let ranks_for_group = device_mesh.get_ranks_for_dim_slice(&state.dims)?;
                let split_comm = global_comm
                    .split_from(
                        this,
                        ranks_for_group
                            .into_iter()
                            .map(|v| v.clone().try_into())
                            .collect::<Result<Vec<_>, _>>()?,
                        config,
                    )
                    .await?
                    .context("split comm should include self rank")?;
                entry.insert(Arc::new(split_comm));
            }
            None => {
                // This rank is not in the group to be split off. We still need to
                // participate in the commSplit call, however.
                global_comm.split_from(this, vec![], config).await?;
            }
        }
        Ok(())
    }

    async fn pipe_recv(
        &mut self,
        this: &Instance<Self>,
        _seq: Seq,
        results: Vec<Option<Ref>>,
        pipe: Ref,
        stream: StreamRef,
    ) -> Result<()> {
        // Get a port for the pipe
        let pipe = match self.pipes.get(&pipe) {
            None => Err(CallFunctionError::RefNotFound(pipe)),
            Some(pipe) => match pipe.as_ref() {
                Ok(pipe) => Ok(pipe.port()),
                Err(e) => Err(CallFunctionError::DependentError(e.clone())),
            },
        };

        // Resolve the stream.
        let stream = self
            .streams
            .get(&stream)
            .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", stream))?;

        // Push result into the stream.
        stream.set_value(this, results, pipe).await
    }

    async fn set_ref_unit_tests_only(
        &mut self,
        this: &Instance<Self>,
        reference: Ref,
        value: WireValue,
        stream: StreamRef,
    ) -> Result<()> {
        let stream = self
            .streams
            .get(&stream)
            .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", &stream))?;

        stream.set_ref_unit_tests_only(this, reference, value).await
    }

    async fn get_ref_unit_tests_only(
        &mut self,
        this: &Instance<Self>,
        ref_id: Ref,
        stream: StreamRef,
    ) -> Result<Option<Result<WireValue, ValueError>>> {
        let stream = self
            .streams
            .get(&stream)
            .ok_or_else(|| anyhow::anyhow!("invalid stream id: {:#?}", &stream))?;
        Ok(stream
            .get_ref_unit_tests_only(this, ref_id.clone())
            .await?
            .map(|o| Ok(o?)))
    }

    async fn define_recording(
        &mut self,
        _this: &Instance<Self>,
        _result: Ref,
        _nresults: usize,
        _nformals: usize,
        _commands: Vec<WorkerMessage>,
    ) -> Result<()> {
        unimplemented!()
    }

    async fn recording_formal(
        &mut self,
        _this: &Instance<Self>,
        _result: Ref,
        _argument_index: usize,
        _stream: StreamRef,
    ) -> Result<()> {
        unimplemented!()
    }

    async fn recording_result(
        &mut self,
        _this: &Instance<Self>,
        _result: Ref,
        _output_index: usize,
        _stream: StreamRef,
    ) -> Result<()> {
        unimplemented!()
    }

    async fn call_recording(
        &mut self,
        _this: &Instance<Self>,
        _seq: Seq,
        _recording: Ref,
        _results: Vec<Ref>,
        _actuals: Vec<Ref>,
    ) -> Result<()> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;
    use std::process::Stdio;

    use anyhow::Result;
    use hyperactor::Mailbox;
    use hyperactor::Named;
    use hyperactor::WorldId;
    use hyperactor::actor::ActorStatus;
    use hyperactor::channel::ChannelAddr;
    use hyperactor::id;
    use hyperactor::mailbox::open_port;
    use hyperactor::proc::Proc;
    use hyperactor_multiprocess::System;
    use hyperactor_multiprocess::proc_actor::Environment;
    use hyperactor_multiprocess::proc_actor::ProcActor;
    use hyperactor_multiprocess::proc_actor::ProcMessageClient;
    use hyperactor_multiprocess::system_actor::SYSTEM_ACTOR_REF;
    use hyperactor_multiprocess::system_actor::Shape;
    use hyperactor_multiprocess::system_actor::SystemActorParams;
    use hyperactor_multiprocess::system_actor::SystemMessageClient;
    use hyperactor_multiprocess::system_actor::SystemSnapshotFilter;
    use hyperactor_multiprocess::system_actor::WorldStatus;
    use monarch_messages::controller::ControllerMessage;
    use monarch_messages::controller::WorkerError;
    use monarch_types::PickledPyObject;
    use monarch_types::PyTree;
    use pyo3::IntoPy;
    use pyo3::Python;
    use pyo3::prelude::*;
    use pyo3::types::PyList;
    use pyo3::types::PyString;
    use rand::Rng;
    use rand::distributions::Alphanumeric;
    use timed_test::async_timed_test;
    use tokio::io::BufReader;
    use tokio::process::Command;
    use tokio_retry::Retry;
    use tokio_retry::strategy::FixedInterval;
    use torch_sys::Device;
    use torch_sys::DeviceIndex;
    use torch_sys::MemoryFormat;

    use super::*;
    use crate::test_util::test_setup;

    #[async_timed_test(timeout_secs = 60)]
    async fn basic_worker() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, mut controller_rx) = proc.attach_actor("controller").unwrap();

        let worker_handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();
        worker_handle
            .command_group(
                &client,
                vec![
                    WorkerMessage::CreateStream {
                        id: 1.into(),
                        stream_creation: StreamCreationMode::UseDefaultStream,
                    },
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 0.into(),
                        results: vec![Some(0.into())],
                        mutates: vec![],
                        function: "torch.ops.aten.ones.default".into(),
                        args: vec![WireValue::IntList(vec![2, 3])],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 2.into(),
                        results: vec![Some(Ref { id: 2 })],
                        mutates: vec![0.into()],
                        function: "torch.ops.aten.sub_.Scalar".into(),
                        args: vec![WireValue::Ref(0.into()), WireValue::Int(1)],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 3.into(),
                        results: vec![Some(Ref { id: 3 })],
                        mutates: vec![],
                        function: "torch.ops.aten.zeros.default".into(),
                        args: vec![WireValue::IntList(vec![2, 3])],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 4.into(),
                        results: vec![Some(Ref { id: 4 })],
                        mutates: vec![],
                        function: "torch.ops.aten.allclose.default".into(),
                        args: vec![WireValue::Ref(0.into()), WireValue::Ref(Ref { id: 3 })],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                ],
            )
            .await
            .unwrap();

        let result: bool = worker_handle
            .get_ref_unit_tests_only(&client, Ref { id: 4 }, 1.into())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .try_into()
            .unwrap();
        worker_handle.drain_and_stop().unwrap();
        worker_handle.await;
        let error_responses = controller_rx.drain();
        assert!(
            error_responses.is_empty(),
            "Expected no error responses, got: {:#?}",
            error_responses
        );
        assert!(result);

        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn error_sends_response() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, mut controller_rx) = proc.attach_actor("controller").unwrap();

        let worker_handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();
        worker_handle
            .command_group(
                &client,
                vec![
                    WorkerMessage::CreateStream {
                        id: 1.into(),
                        stream_creation: StreamCreationMode::UseDefaultStream,
                    },
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 0.into(),
                        results: vec![Some(0.into())],
                        mutates: vec![],
                        function: "torch.ops.aten.rand.default".into(),
                        args: vec![],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::Exit { error: None },
                ],
            )
            .await
            .unwrap();

        worker_handle.drain_and_stop().unwrap();
        worker_handle.await;
        let response_message = controller_rx.recv().await.unwrap();
        match response_message {
            ControllerMessage::RemoteFunctionFailed {
                seq,
                error: WorkerError { backtrace: msg, .. },
            } => {
                assert_eq!(seq, 0.into());
                assert!(msg.contains("aten::rand() is missing value for argument 'size'"))
            }
            _ => panic!("unexpected response {:#?}", response_message),
        }

        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn mutated_refs_are_updated_with_error() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, mut controller_rx) = proc.attach_actor("controller").unwrap();

        let worker_handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();
        worker_handle
            .command_group(
                &client,
                vec![
                    WorkerMessage::CreateStream {
                        id: 1.into(),
                        stream_creation: StreamCreationMode::UseDefaultStream,
                    },
                    WorkerMessage::SetRefUnitTestsOnly {
                        reference: 0.into(),
                        value: WireValue::Int(1),
                        stream: 1.into(),
                    },
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 0.into(),
                        results: vec![Some(Ref { id: 2 })],
                        mutates: vec![0.into()],
                        function: "i.dont.exist".into(),
                        args: vec![],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                ],
            )
            .await
            .unwrap();

        let result = worker_handle
            .get_ref_unit_tests_only(&client, 0.into(), 1.into())
            .await?;

        // Stop/drain worker before asserts to avoid hangs.
        worker_handle.drain_and_stop().unwrap();
        worker_handle.await;

        let mutated_ref = result
            .context("no such ref")?
            .err()
            .context("expected error")?
            .into_call_function_error()?;
        assert!(mutated_ref.contains("InvalidRemoteFunction"));

        let responses = controller_rx.drain();
        assert_eq!(
            responses.len(),
            1,
            "Expected one response, got: {:#?}",
            responses
        );
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn accessing_errored_dependency() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, mut controller_rx) = proc.attach_actor("controller").unwrap();

        let worker_handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();
        worker_handle
            .command_group(
                &client,
                vec![
                    WorkerMessage::CreateStream {
                        id: 1.into(),
                        stream_creation: StreamCreationMode::UseDefaultStream,
                    },
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 0.into(),
                        results: vec![Some(0.into())],
                        mutates: vec![],
                        function: "i.dont.exist".into(),
                        args: vec![],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 1.into(),
                        results: vec![Some(1.into())],
                        mutates: vec![],
                        function: "torch.ops.aten.sub_.Scalar".into(),
                        args: vec![WireValue::Ref(0.into())],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::Exit { error: None },
                ],
            )
            .await
            .unwrap();

        worker_handle.drain_and_stop().unwrap();
        worker_handle.await;

        let responses = controller_rx.drain();
        assert_eq!(
            responses.len(),
            1,
            "Expected one response, got: {:#?}",
            responses
        );

        match &responses[0] {
            ControllerMessage::RemoteFunctionFailed { seq, .. } => {
                assert_eq!(seq, &0.into())
            }
            _ => panic!("unexpected response {:#?}", responses[0]),
        };
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn py_remote_function_calls() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, mut controller_rx) = proc.attach_actor("controller").unwrap();

        let worker_handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();
        let (split_arg, sort_list, mesh_ref, dim, layout, none, scalar, device, memory_format) =
            Python::with_gil(|py| {
                let split_arg: PickledPyObject = PyString::new_bound(py, "/fbs/fbc/foo/bar")
                    .into_any()
                    .try_into()?;
                let sort_list: PickledPyObject = PyList::new_bound(py, [65, 34, 79, 1, 5])
                    .into_any()
                    .try_into()?;
                let mesh_ref: PickledPyObject =
                    Ref { id: 5 }.into_py(py).into_bound(py).try_into()?;
                let dim: PickledPyObject = PyString::new_bound(py, "x").into_any().try_into()?;
                let layout: PickledPyObject =
                    py.import_bound("torch")?.getattr("strided")?.try_into()?;
                let none: PickledPyObject = py.None().into_any().into_bound(py).try_into()?;
                let scalar: PickledPyObject =
                    py.import_bound("torch")?.getattr("float32")?.try_into()?;
                let device: PickledPyObject = py
                    .import_bound("torch")?
                    .getattr("device")?
                    .call1(("cuda:1",))?
                    .try_into()?;
                let memory_format: PickledPyObject = py
                    .import_bound("torch")?
                    .getattr("contiguous_format")?
                    .try_into()?;
                PyResult::Ok((
                    split_arg,
                    sort_list,
                    mesh_ref,
                    dim,
                    layout,
                    none,
                    scalar,
                    device,
                    memory_format,
                ))
            })?;

        worker_handle
            .command_group(
                &client,
                vec![
                    WorkerMessage::CreateStream {
                        id: 1.into(),
                        stream_creation: StreamCreationMode::UseDefaultStream,
                    },
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 0.into(),
                        results: vec![Some(0.into()), Some(Ref { id: 2 })],
                        mutates: vec![],
                        function: "os.path.split".into(),
                        args: vec![split_arg.into()],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 2.into(),
                        results: vec![Some(4.into()), None, None, None, None],
                        mutates: vec![],
                        function: "builtins.sorted".into(),
                        args: vec![sort_list.into()],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CreateDeviceMesh {
                        result: 5.into(),
                        names: vec!["x".into()],
                        ranks: Slice::new(0, vec![2], vec![1]).unwrap(),
                    },
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 2.into(),
                        results: vec![Some(6.into())],
                        mutates: vec![],
                        function: "monarch.monarch_worker.test_utils.mesh_rank".into(),
                        args: vec![mesh_ref.into(), dim.into()],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 4.into(),
                        results: vec![Some(7.into())],
                        mutates: vec![],
                        function: "monarch.monarch_worker.test_utils.test_scalar_type".into(),
                        args: vec![scalar.into()],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 5.into(),
                        results: vec![Some(8.into())],
                        mutates: vec![],
                        function: "monarch.monarch_worker.test_utils.test_layout".into(),
                        args: vec![layout.into()],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 6.into(),
                        results: vec![Some(9.into())],
                        mutates: vec![],
                        function: "monarch.monarch_worker.test_utils.test_none".into(),
                        args: vec![none.into()],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    // Verify that a function that returns `None` matches up with an
                    // empty result list.
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 7.into(),
                        results: vec![None],
                        mutates: vec![],
                        function: "monarch.monarch_worker.test_utils.none".into(),
                        args: vec![],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 8.into(),
                        results: vec![Some(10.into())],
                        mutates: vec![],
                        function: "monarch.monarch_worker.test_utils.test_device".into(),
                        args: vec![device.into()],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 9.into(),
                        results: vec![Some(11.into())],
                        mutates: vec![],
                        function: "monarch.monarch_worker.test_utils.test_memory_format".into(),
                        args: vec![memory_format.into()],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    // Test that list of tests can be passes correctly
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 10.into(),
                        results: vec![Some(12.into())],
                        mutates: vec![],
                        function: "torch.ops.aten.ones.default".into(),
                        args: vec![WireValue::IntList(vec![2, 3])],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 11.into(),
                        results: vec![Some(13.into())],
                        mutates: vec![],
                        function: "torch.ops.aten.stack.default".into(),
                        args: vec![WireValue::RefList(vec![12.into(), 12.into()])],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                ],
            )
            .await
            .unwrap();

        let result1: String = worker_handle
            .get_ref_unit_tests_only(&client, 0.into(), 1.into())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .try_into()
            .unwrap();
        let result2: String = worker_handle
            .get_ref_unit_tests_only(&client, 2.into(), 1.into())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .try_into()
            .unwrap();
        let result3: i64 = worker_handle
            .get_ref_unit_tests_only(&client, 4.into(), 1.into())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .try_into()
            .unwrap();
        let result4: i64 = worker_handle
            .get_ref_unit_tests_only(&client, 6.into(), 1.into())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(
            ScalarType::Float,
            worker_handle
                .get_ref_unit_tests_only(&client, 7.into(), 1.into())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .try_into()
                .unwrap()
        );
        assert_eq!(
            Layout::Strided,
            worker_handle
                .get_ref_unit_tests_only(&client, 8.into(), 1.into())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .try_into()
                .unwrap()
        );
        assert_matches!(
            worker_handle
                .get_ref_unit_tests_only(&client, 9.into(), 1.into())
                .await
                .unwrap()
                .unwrap()
                .unwrap(),
            WireValue::None(()),
        );
        let device: Device = CudaDevice::new(DeviceIndex(1)).into();
        assert_eq!(
            device,
            worker_handle
                .get_ref_unit_tests_only(&client, 10.into(), 1.into())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .try_into()
                .unwrap()
        );
        assert_matches!(
            worker_handle
                .get_ref_unit_tests_only(&client, 11.into(), 1.into())
                .await
                .unwrap()
                .unwrap()
                .unwrap(),
            WireValue::MemoryFormat(MemoryFormat::Contiguous),
        );

        worker_handle.drain_and_stop().unwrap();
        worker_handle.await;
        let error_responses = controller_rx.drain();
        assert!(
            error_responses.is_empty(),
            "Expected no error responses, got: {:#?}",
            error_responses
        );

        assert_eq!(result1, "/fbs/fbc/foo");
        assert_eq!(result2, "bar");
        assert_eq!(result3, 1);
        assert_eq!(result4, 0);

        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn delete_refs() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, _) = proc.attach_actor("controller").unwrap();

        let worker_handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();
        worker_handle
            .command_group(
                &client,
                vec![
                    WorkerMessage::CreateStream {
                        id: 0.into(),
                        stream_creation: StreamCreationMode::CreateNewStream,
                    },
                    WorkerMessage::CreateStream {
                        id: 1.into(),
                        stream_creation: StreamCreationMode::CreateNewStream,
                    },
                    WorkerMessage::SetRefUnitTestsOnly {
                        reference: Ref { id: 2 },
                        value: WireValue::Bool(false),
                        stream: 0.into(),
                    },
                    WorkerMessage::SetRefUnitTestsOnly {
                        reference: Ref { id: 3 },
                        value: WireValue::Bool(true),
                        stream: 0.into(),
                    },
                    WorkerMessage::SetRefUnitTestsOnly {
                        reference: Ref { id: 4 },
                        value: WireValue::Int(0),
                        stream: 1.into(),
                    },
                    WorkerMessage::DeleteRefs(vec![Ref { id: 2 }, Ref { id: 4 }]),
                ],
            )
            .await
            .unwrap();

        let result: bool = worker_handle
            .get_ref_unit_tests_only(&client, Ref { id: 3 }, 0.into())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .try_into()
            .unwrap();
        let fail_result = worker_handle
            .get_ref_unit_tests_only(&client, Ref { id: 4 }, 1.into())
            .await
            .unwrap();

        worker_handle.drain_and_stop().unwrap();
        worker_handle.await;

        assert!(result, "should be able to get a non-deleted ref");
        assert!(fail_result.is_none(), "should fail to get a deleted ref");

        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn request_status() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, mut controller_rx) = proc.attach_actor("controller").unwrap();

        let worker_handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();
        worker_handle
            .command_group(
                &client,
                vec![
                    WorkerMessage::CreateStream {
                        id: 0.into(),
                        stream_creation: StreamCreationMode::CreateNewStream,
                    },
                    WorkerMessage::CreateStream {
                        id: 1.into(),
                        stream_creation: StreamCreationMode::CreateNewStream,
                    },
                ],
            )
            .await
            .unwrap();

        for i in 0..100 {
            // call alternating functions on this stream.
            worker_handle
                .call_function(
                    &client,
                    CallFunctionParams {
                        seq: i.into(),
                        results: vec![Some(Ref { id: i + 2 })],
                        mutates: vec![],
                        function: "torch.ops.aten.ones.default".into(),
                        args: vec![WireValue::IntList(vec![2, 3])],
                        kwargs: HashMap::new(),
                        stream: (i % 2).into(),
                        remote_process_groups: vec![],
                    },
                )
                .await
                .unwrap();
        }

        worker_handle
            .request_status(&client, 100.into(), false)
            .await
            .unwrap();

        worker_handle.drain_and_stop().unwrap();
        worker_handle.await;

        let mut responses = controller_rx.drain();
        assert_eq!(
            responses.len(),
            1,
            "Expected one response, got: {:#?}",
            responses
        );

        let response = responses.pop().unwrap();
        match response {
            ControllerMessage::Status { seq, .. } => {
                assert_eq!(seq, 101.into())
            }
            _ => panic!("unexpected response {:#?}", response),
        };

        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn backend_network_init() {
        let proc = Proc::local();
        let (client, controller_ref, _) = proc.attach_actor("controller").unwrap();

        let worker_handle1 = proc
            .spawn::<WorkerActor>(
                "worker0",
                WorkerParams {
                    world_size: 2,
                    rank: 0,
                    device_index: Some(0),
                    controller_actor: controller_ref.clone(),
                },
            )
            .await
            .unwrap();
        let worker_handle2 = proc
            .spawn::<WorkerActor>(
                "worker1",
                WorkerParams {
                    world_size: 2,
                    rank: 1,
                    device_index: Some(1),
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();

        let unique_id = UniqueId::new().unwrap();
        worker_handle1
            .backend_network_init(&client, unique_id.clone())
            .await
            .unwrap();
        worker_handle2
            .backend_network_init(&client, unique_id)
            .await
            .unwrap();

        worker_handle1.drain_and_stop().unwrap();
        worker_handle1.await;
        worker_handle2.drain_and_stop().unwrap();
        worker_handle2.await;
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn send_value() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, mut controller_rx) = proc.attach_actor("controller").unwrap();

        let worker_handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();
        worker_handle
            .command_group(
                &client,
                vec![
                    WorkerMessage::CreateStream {
                        id: 1.into(),
                        stream_creation: StreamCreationMode::UseDefaultStream,
                    },
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 0.into(),
                        results: vec![Some(0.into())],
                        mutates: vec![],
                        function: "torch.ops.aten.ones.default".into(),
                        args: vec![WireValue::IntList(vec![2, 3])],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::SendValue {
                        seq: 1.into(),
                        destination: None,
                        mutates: vec![],
                        function: None,
                        args: vec![WireValue::Ref(0.into())],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                    },
                    WorkerMessage::SendValue {
                        seq: 2.into(),
                        destination: None,
                        mutates: vec![],
                        function: Some("torch.ops.aten.var_mean.default".into()),
                        args: vec![WireValue::Ref(0.into())],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                    },
                    WorkerMessage::Exit { error: None },
                ],
            )
            .await
            .unwrap();

        worker_handle.drain_and_stop()?;
        assert_matches!(worker_handle.await, ActorStatus::Stopped);

        let mut responses = controller_rx.drain();
        assert_eq!(
            responses.len(),
            2,
            "Expected one response, got: {:#?}",
            responses
        );

        match responses.pop().unwrap() {
            ControllerMessage::FetchResult { seq, value } => {
                assert_eq!(seq, 2.into());
                let value = value.unwrap().deserialized::<PyTree<RValue>>().unwrap();
                assert_eq!(value.leaves().len(), 2);
            }
            resp => panic!("unexpected response {:#?}", resp),
        };
        match responses.pop().unwrap() {
            ControllerMessage::FetchResult { seq, .. } => {
                assert_eq!(seq, 1.into())
            }
            resp => panic!("unexpected response {:#?}", resp),
        };
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn send_value_err_result() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, mut controller_rx) = proc.attach_actor("controller").unwrap();

        let worker_handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();

        let ref_arg: PickledPyObject =
            Python::with_gil(|py| Ref { id: 2 }.into_py(py).into_bound(py).try_into())?;

        worker_handle
            .command_group(
                &client,
                vec![
                    WorkerMessage::CreateStream {
                        id: 1.into(),
                        stream_creation: StreamCreationMode::UseDefaultStream,
                    },
                    WorkerMessage::SetRefUnitTestsOnly {
                        reference: Ref { id: 2 },
                        value: WireValue::Bool(false),
                        stream: 1.into(),
                    },
                    WorkerMessage::SendValue {
                        seq: 1.into(),
                        destination: None,
                        mutates: vec![Ref { id: 2 }],
                        function: Some("non.existent.function".into()),
                        args: vec![],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                    },
                    WorkerMessage::SendValue {
                        seq: 2.into(),
                        destination: None,
                        mutates: vec![],
                        function: None,
                        args: vec![ref_arg.into()],
                        kwargs: HashMap::new(),
                        stream: 1.into(),
                    },
                    WorkerMessage::Exit { error: None },
                ],
            )
            .await
            .unwrap();

        worker_handle.drain_and_stop()?;
        assert_matches!(worker_handle.await, ActorStatus::Stopped);

        let mut responses = controller_rx.drain();
        assert_eq!(
            responses.len(),
            2,
            "Expected one response, got: {:#?}",
            responses
        );

        match responses.pop() {
            Some(ControllerMessage::FetchResult { seq, value }) => {
                assert_eq!(seq, 2.into());
                assert!(value.is_err());
                assert!(
                    value
                        .unwrap_err()
                        .backtrace
                        .contains("InvalidRemoteFunction")
                );
            }
            _ => panic!("unexpected response {:#?}", responses),
        }
        match responses.pop() {
            Some(ControllerMessage::FetchResult { seq, value }) => {
                assert_eq!(seq, 1.into());
                assert!(value.is_err());
                assert!(
                    value
                        .unwrap_err()
                        .backtrace
                        .contains("InvalidRemoteFunction")
                );
            }
            _ => panic!("unexpected response {:#?}", responses),
        }
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn pipe_send_recv() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, mut controller_rx) = proc.attach_actor("controller").unwrap();

        let handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();
        let (resolve_value_arg, torch_eq_arg1, torch_eq_arg2): (
            PickledPyObject,
            PickledPyObject,
            PickledPyObject,
        ) = Python::with_gil(|py| {
            PyResult::Ok((
                PyList::new_bound(py, [2, 3]).into_any().try_into()?,
                Ref { id: 2 }.into_py(py).into_bound(py).try_into()?,
                Ref { id: 4 }.into_py(py).into_bound(py).try_into()?,
            ))
        })?;

        handle
            .command_group(
                &client,
                vec![
                    WorkerMessage::CreateStream {
                        id: 0.into(),
                        stream_creation: StreamCreationMode::UseDefaultStream,
                    },
                    WorkerMessage::CreateDeviceMesh {
                        result: 1.into(),
                        names: vec!["x".into()],
                        ranks: Slice::new(0, vec![2], vec![1]).unwrap(),
                    },
                    // Create a tensor value which we'll send through the pipe.
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 0.into(),
                        results: vec![Some(2.into())],
                        mutates: vec![],
                        function: "torch.ops.aten.ones.default".into(),
                        args: vec![WireValue::IntList(vec![2, 3])],
                        kwargs: HashMap::new(),
                        stream: 0.into(),
                        remote_process_groups: vec![],
                    }),
                    WorkerMessage::CreatePipe {
                        result: 3.into(),
                        key: "unused".into(),
                        function: "monarch.monarch_worker.test_utils.handler".into(),
                        max_messages: 1,
                        mesh: 1.into(),
                        args: vec![],
                        kwargs: HashMap::new(),
                    },
                    WorkerMessage::SendValue {
                        seq: 1.into(),
                        destination: Some(3.into()),
                        mutates: vec![],
                        function: Some("monarch.monarch_worker.test_utils.resolve_value".into()),
                        args: vec![resolve_value_arg.into()],
                        kwargs: HashMap::new(),
                        stream: 0.into(),
                    },
                    WorkerMessage::PipeRecv {
                        seq: 2.into(),
                        results: vec![Some(4.into())],
                        pipe: 3.into(),
                        stream: 0.into(),
                    },
                    WorkerMessage::CallFunction(CallFunctionParams {
                        seq: 0.into(),
                        results: vec![Some(5.into())],
                        mutates: vec![],
                        function: "torch.equal".into(),
                        args: vec![torch_eq_arg1.into(), torch_eq_arg2.into()],
                        kwargs: HashMap::new(),
                        stream: 0.into(),
                        remote_process_groups: vec![],
                    }),
                ],
            )
            .await
            .unwrap();

        let matches: bool = handle
            .get_ref_unit_tests_only(&client, 5.into(), 0.into())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .try_into()
            .unwrap();
        assert!(matches);

        handle.drain_and_stop()?;
        assert_matches!(handle.await, ActorStatus::Stopped);

        let responses = controller_rx.drain();
        assert_eq!(
            responses.len(),
            0,
            "Expected one response, got: {:#?}",
            responses
        );

        Ok(())
    }

    fn get_random_channel_addr() -> ChannelAddr {
        let random_string = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(24)
            .map(char::from)
            .collect::<String>();
        format!("unix!@{random_string}").parse().unwrap()
    }

    async fn ensure_world_ready(client: Mailbox, world: WorldId) -> Result<()> {
        tracing::info!("checking whether world {world} is ready");
        let retry_strategy = FixedInterval::from_millis(1000).take(100);
        Retry::spawn(retry_strategy, async || {
            let snapshot = SYSTEM_ACTOR_REF
                .snapshot(&client, SystemSnapshotFilter::default())
                .await?;
            let world_snapshot = snapshot.worlds.get(&world).ok_or(anyhow!("no world"))?;
            tracing::info!("world status: {:?}", world_snapshot.status);
            match world_snapshot.status {
                WorldStatus::Live => Ok(()),
                _ => Err(anyhow!("world is not live")),
            }
        })
        .await?;
        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn remote_process_group() -> Result<()> {
        test_setup()?;

        // Spin up a system to manage the test setup.
        let timeout: Duration = Duration::from_secs(10);
        let params = SystemActorParams::new(timeout, timeout);
        let system_addr = get_random_channel_addr();
        let _system_handle = System::serve(system_addr.clone(), params).await?;

        // Create a fake controller for the workers to talk to.
        let client = System::new(system_addr.clone()).attach().await?;
        let (handle, mut controller_rx) = client.open_port::<ControllerMessage>();
        handle.bind_to(ControllerMessage::port());
        let controller_ref: ActorRef<ControllerActor> = ActorRef::attest(client.actor_id().clone());

        // Create the worker world
        let world_size = 2;
        SYSTEM_ACTOR_REF
            .upsert_world(
                &client,
                id!(world),
                Shape::Definite(vec![world_size]),
                4,
                Environment::Local,
                HashMap::new(),
            )
            .await?;

        // Bootstrap a proc for each worker
        let mut worker_process_handles = vec![];
        let mut worker_procs: Vec<ActorRef<ProcActor>> = vec![];
        for rank in 0..world_size {
            let world_id = "world".to_string();
            let proc_id = format!("{world_id}[{rank}]");
            worker_procs.push(ActorRef::attest(format!("world[{rank}].proc[0]").parse()?));

            let mut handle = Command::new(
                std::env::var("MONARCH_WORKER_EXE")
                    .map_err(|e| anyhow::anyhow!("could not get var MONARCH_WORKER_EXE: {}", e))?,
            )
            .arg("worker")
            .arg(format!("--bootstrap-addr={system_addr}"))
            .arg(format!("--world-id={world_id}"))
            .arg(format!("--proc-id={proc_id}"))
            .env("HYPERACTOR_MANAGED_SUBPROCESS", "1")
            .stdout(Stdio::piped())
            .stdin(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

            let out = handle.stdout.take().unwrap();
            tokio::spawn(async move {
                let mut reader = BufReader::new(out);
                tokio::io::copy(&mut reader, &mut tokio::io::stderr())
                    .await
                    .unwrap();
            });
            worker_process_handles.push(handle);
        }

        // Wait for procs to initialize
        ensure_world_ready(client.clone(), id!(world)).await?;

        // Spawn workers on each proc
        let (spawned_port, mut spawned_receiver) = open_port(&client);
        for (rank, worker_proc) in worker_procs.iter().enumerate() {
            let params = WorkerParams {
                world_size,
                rank,
                device_index: Some(rank.try_into().unwrap()),
                controller_actor: controller_ref.clone(),
            };
            worker_proc
                .spawn(
                    &client,
                    "monarch_worker::WorkerActor".to_owned(),
                    "worker".to_owned(),
                    bincode::serialize(&params)?,
                    spawned_port.bind(),
                )
                .await?;
        }
        let mut spawned = vec![];
        while spawned.len() < world_size {
            spawned.push(spawned_receiver.recv().await?);
        }
        tracing::info!("spawned {} worker actors", world_size);
        let workers: Vec<ActorRef<WorkerActor>> = (0..world_size)
            .map(|rank| format!("world[{rank}].worker[0]"))
            .map(|name| ActorRef::attest(name.parse().unwrap()))
            .collect();

        let remote_proc_grp_ref: PickledPyObject =
            Python::with_gil(|py| Ref { id: 2 }.into_py(py).into_bound(py).try_into())?;

        let unique_id = UniqueId::new()?;
        let messages = vec![
            WorkerMessage::CreateStream {
                id: 0.into(),
                stream_creation: StreamCreationMode::UseDefaultStream,
            },
            WorkerMessage::BackendNetworkInit(unique_id.clone()),
            WorkerMessage::CreateDeviceMesh {
                result: 1.into(),
                names: vec!["x".into()],
                ranks: Slice::new(0, vec![2], vec![1]).unwrap(),
            },
            WorkerMessage::CreateRemoteProcessGroup {
                result: 2.into(),
                device_mesh: 1.into(),
                dims: vec!["x".into()],
            },
            WorkerMessage::SplitCommForProcessGroup {
                remote_process_group: 2.into(),
                stream: 0.into(),
                config: None,
            },
            WorkerMessage::CallFunction(CallFunctionParams {
                seq: 0.into(),
                results: vec![Some(3.into())],
                mutates: vec![],
                function: "monarch.monarch_worker.test_utils.test_remote_process_group".into(),
                args: vec![remote_proc_grp_ref.into()],
                kwargs: HashMap::new(),
                stream: 0.into(),
                remote_process_groups: vec![2.into()],
            }),
        ];

        workers[0].command_group(&client, messages.clone()).await?;
        workers[1].command_group(&client, messages).await?;

        let _ = workers[0]
            .get_ref_unit_tests_only(&client, 3.into(), 0.into())
            .await?
            .unwrap()
            .unwrap();

        let error_responses = controller_rx.drain();
        assert!(
            error_responses.is_empty(),
            "Expected no error responses, got: {:#?}",
            error_responses
        );

        Ok(())
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn propagate_pipe_create_failure() -> Result<()> {
        test_setup()?;

        let proc = Proc::local();
        let (client, controller_ref, mut controller_rx) = proc.attach_actor("controller").unwrap();

        let handle = proc
            .spawn::<WorkerActor>(
                "worker",
                WorkerParams {
                    world_size: 1,
                    rank: 0,
                    device_index: None,
                    controller_actor: controller_ref,
                },
            )
            .await
            .unwrap();
        handle
            .send(WorkerMessage::CommandGroup(vec![
                WorkerMessage::CreateStream {
                    id: 0.into(),
                    stream_creation: StreamCreationMode::UseDefaultStream,
                },
                WorkerMessage::CreatePipe {
                    result: 2.into(),
                    key: "unused".into(),
                    mesh: 1.into(),
                    function: "monarch.monarch_worker.pipe_test_utils.handler".into(),
                    max_messages: 1,
                    args: vec![],
                    kwargs: HashMap::new(),
                },
                WorkerMessage::PipeRecv {
                    seq: 1.into(),
                    results: vec![Some(3.into())],
                    pipe: 2.into(),
                    stream: 0.into(),
                },
            ]))
            .unwrap();

        let value: Result<_, ValueError> = handle
            .get_ref_unit_tests_only(&client, 3.into(), 0.into())
            .await
            .unwrap()
            .unwrap();
        assert!(value.is_err());

        handle.drain_and_stop()?;
        //handle.await;
        assert_matches!(handle.await, ActorStatus::Stopped);

        let responses = controller_rx.drain();
        assert_eq!(
            responses.len(),
            0,
            "Expected zero responses, got: {:#?}",
            responses
        );

        Ok(())
    }
}
