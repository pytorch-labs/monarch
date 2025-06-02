/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use futures::FutureExt;
use futures::future::select_all;
use hyperactor::Named;
use hyperactor::ProcId;
use hyperactor::WorldId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelRx;
use hyperactor::channel::ChannelTransport;
use hyperactor::channel::ChannelTx;
use hyperactor::channel::Rx;
use hyperactor::channel::Tx;
use hyperactor::channel::TxStatus;
use hyperactor::clock;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use hyperactor::mailbox::DialMailboxRouter;
use hyperactor::mailbox::MailboxServer;
use hyperactor::mailbox::monitored_return_handle;
use hyperactor::reference::Reference;
use hyperactor::serde_json;
use mockall::automock;
use ndslice::Shape;
use serde::Deserialize;
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::unbounded_channel;
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::WatchStream;
use tokio_util::sync::CancellationToken;

use crate::alloc::Alloc;
use crate::alloc::AllocSpec;
use crate::alloc::Allocator;
use crate::alloc::AllocatorError;
use crate::alloc::ProcState;
use crate::alloc::ProcStopReason;
use crate::alloc::ProcessAllocator;

/// Control messages sent from remote process allocator to local allocator.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub enum RemoteProcessAllocatorMessage {
    /// Create allocation with given spec and send updates to bootstrap_addr.
    Allocate {
        /// Allocation spec. Shape is a slice of the original shape with
        /// correct rank offset.
        spec: AllocSpec,
        /// Bootstrap address to be used for sending updates.
        bootstrap_addr: ChannelAddr,
        /// Ordered list of hosts in this allocation. Can be used to
        /// pre-populate the any local configurations such as torch.dist.
        hosts: Vec<String>,
        /// How often to send heartbeat messages to check if client is alive.
        heartbeat_interval: Duration,
    },
    /// Stop allocation.
    Stop,
    /// Heartbeat message to check if remote process allocator and its
    /// host are alive.
    HeartBeat,
}

/// Control message sent from local allocator to remote allocator
/// relaying process state updates.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub enum RemoteProcessProcStateMessage {
    /// Allocation successful and Update, Done messages will follow.
    Allocated { world_id: WorldId, shape: Shape },
    /// ProcState updates.
    Update(ProcState),
    /// Underlying Alloc is done.
    Done(WorldId),
    /// Heartbeat message to check if client is alive.
    HeartBeat,
}

/// Allocator with a service frontend that wraps ProcessAllocator.
pub struct RemoteProcessAllocator {
    cancel_token: CancellationToken,
}

impl RemoteProcessAllocator {
    /// Create a new allocator. It will not start until start() is called.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            cancel_token: CancellationToken::new(),
        })
    }

    /// Stop the allocator. This will stop any ongoing allocations.
    pub fn terminate(&self) {
        self.cancel_token.cancel();
    }

    /// Start a remote process allocator with given cmd listening for
    /// RemoteProcessAllocatorMessage on serve_addr. Call will block until cancelled.
    /// The implementation is simple such that it can only handle one Alloc at
    /// a time. Generally that's the most common use-case.
    /// Flow works as follows:
    /// 1. Client sends Allocate message to serve_addr.
    /// 2. Allocator connects to bootstrap_addr, creates Alloc and sends Allocated message.
    /// 3. Allocator streams one or more Update messages to bootstrap_addr as Alloc progresses
    ///    making the following changes:
    ///    * Remap mesh_agent listen address to our own forwarder actor address.
    /// 4. Allocator sends Done message to bootstrap_addr when Alloc is done.
    ///
    /// At any point, client can send Stop message to serve_addr to stop the allocator.
    pub async fn start(&self, cmd: Command, serve_addr: ChannelAddr) -> Result<(), anyhow::Error> {
        let process_allocator = ProcessAllocator::new(cmd);
        self.start_with_allocator(serve_addr, process_allocator)
            .await
    }

    /// Start a remote process allocator with given allocator listening for
    /// RemoteProcessAllocatorMessage on serve_addr.
    /// Used for testing.
    pub async fn start_with_allocator<A: Allocator + Send + Sync + 'static>(
        &self,
        serve_addr: ChannelAddr,
        mut process_allocator: A,
    ) -> Result<(), anyhow::Error>
    where
        <A as Allocator>::Alloc: Send,
        <A as Allocator>::Alloc: Sync,
    {
        tracing::info!("starting remote allocator on: {}", serve_addr);
        let (_, mut rx) = channel::serve(serve_addr)
            .await
            .map_err(anyhow::Error::from)?;

        struct ActiveAllocation {
            handle: JoinHandle<()>,
            cancel_token: CancellationToken,
        }
        async fn ensure_previous_alloc_stopped(active_allocation: &mut Option<ActiveAllocation>) {
            if let Some(active_allocation) = active_allocation.take() {
                tracing::info!("previous alloc found, stopping");
                active_allocation.cancel_token.cancel();
                // should be ok to wait even if original caller has gone since heartbeat
                // will eventually timeout and exit the loop.
                if let Err(e) = active_allocation.handle.await {
                    tracing::error!("allocation handler failed: {}", e);
                }
            }
        }

        let mut active_allocation: Option<ActiveAllocation> = None;
        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Ok(RemoteProcessAllocatorMessage::Allocate {
                            spec,
                            bootstrap_addr,
                            hosts,
                            heartbeat_interval,
                        }) => {
                            tracing::info!("received allocation request: {:?}", spec);

                            ensure_previous_alloc_stopped(&mut active_allocation).await;

                            match process_allocator.allocate(spec.clone()).await {
                                Ok(alloc) => {
                                    let cancel_token = CancellationToken::new();
                                    active_allocation = Some(ActiveAllocation {
                                        cancel_token: cancel_token.clone(),
                                        handle: tokio::spawn(Self::handle_allocation_request(
                                        Box::new(alloc) as Box<dyn Alloc + Send + Sync>,
                                        bootstrap_addr,
                                        hosts,
                                        heartbeat_interval,
                                        cancel_token,
                                    )),
                                })
                                }
                                Err(e) => {
                                    tracing::error!("allocation for {:?} failed: {}", spec, e);
                                    continue;
                                }
                            }
                        }
                        Ok(RemoteProcessAllocatorMessage::Stop) => {
                            tracing::info!("received stop request");

                            ensure_previous_alloc_stopped(&mut active_allocation).await;
                        }
                        Ok(RemoteProcessAllocatorMessage::HeartBeat) => {}
                        Err(e) => {
                            tracing::error!("upstream channel error: {}", e);
                            continue;
                        }
                    }
                }
                _ = self.cancel_token.cancelled() => {
                    tracing::info!("main loop cancelled");

                    ensure_previous_alloc_stopped(&mut active_allocation).await;

                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_allocation_request(
        alloc: Box<dyn Alloc + Send + Sync>,
        bootstrap_addr: ChannelAddr,
        hosts: Vec<String>,
        heartbeat_interval: Duration,
        cancel_token: CancellationToken,
    ) {
        // start proc message forwarder
        let (forwarder_addr, forwarder_rx) =
            match channel::serve(ChannelAddr::any(bootstrap_addr.transport())).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!("failed to to bootstrap forwarder actor: {}", e);
                    return;
                }
            };
        let router = DialMailboxRouter::new();
        let mailbox_handle = router
            .clone()
            .serve(forwarder_rx, monitored_return_handle());
        tracing::info!("started forwarder on: {}", forwarder_addr);

        // Check if we need to write TORCH_ELASTIC_CUSTOM_HOSTNAMES_LIST_FILE
        // See: https://github.com/fairinternal/xlformers/blob/llama4_monarch/tools/launching/torchx/entrypoint/generate_ranks.py
        if let Ok(hosts_file) = std::env::var("TORCH_ELASTIC_CUSTOM_HOSTNAMES_LIST_FILE") {
            tracing::info!("writing hosts to {}", hosts_file);
            #[derive(Serialize)]
            struct Hosts {
                hostnames: Vec<String>,
            }
            match serde_json::to_string(&Hosts { hostnames: hosts }) {
                Ok(json) => match tokio::fs::File::create(&hosts_file).await {
                    Ok(mut file) => {
                        if file.write_all(json.as_bytes()).await.is_err() {
                            tracing::error!("failed to write hosts to {}", hosts_file);
                            return;
                        }
                    }
                    Err(e) => {
                        tracing::error!("failed to open hosts file {}: {}", hosts_file, e);
                        return;
                    }
                },
                Err(e) => {
                    tracing::error!("failed to serialize hosts: {}", e);
                    return;
                }
            }
        }

        Self::handle_allocation_loop(
            alloc,
            bootstrap_addr,
            router,
            forwarder_addr,
            heartbeat_interval,
            cancel_token,
        )
        .await;

        mailbox_handle.stop();
        if let Err(e) = mailbox_handle.await {
            tracing::error!("failed to join forwarder: {}", e);
        }
    }

    async fn handle_allocation_loop(
        mut alloc: Box<dyn Alloc + Send + Sync>,
        bootstrap_addr: ChannelAddr,
        router: DialMailboxRouter,
        forward_addr: ChannelAddr,
        heartbeat_interval: Duration,
        cancel_token: CancellationToken,
    ) {
        let tx = match channel::dial(bootstrap_addr) {
            Ok(tx) => tx,
            Err(err) => {
                tracing::error!("failed to dial bootstrap address: {}", err);
                return;
            }
        };
        if let Err(e) = tx
            .send(RemoteProcessProcStateMessage::Allocated {
                world_id: alloc.world_id().clone(),
                shape: alloc.shape().clone(),
            })
            .await
        {
            tracing::error!("failed to send Allocated message: {}", e);
            return;
        }

        let mut mesh_agents_by_proc_id = HashMap::new();
        let mut running = true;
        let tx_status = tx.status().clone();
        let mut tx_watcher = WatchStream::new(tx_status);
        loop {
            tokio::select! {
                _ = cancel_token.cancelled(), if running => {
                    tracing::info!("cancelled, stopping allocation");
                    running = false;
                    if let Err(e) = alloc.stop().await {
                        tracing::error!("stop failed: {}", e);
                        break;
                    }
                }
                status = tx_watcher.next(), if running => {
                    match status  {
                        Some(TxStatus::Closed) => {
                            tracing::error!("upstream channel state closed");
                            break;
                        },
                        _ => {
                            tracing::debug!("got channel event: {:?}", status.unwrap());
                            continue;
                        }
                    }
                }
                e = alloc.next() => {
                    match e {
                        Some(event) => {
                            tracing::debug!("got event: {:?}", event);
                            let event = match event {
                                ProcState::Created { .. } => event,
                                ProcState::Running { proc_id, mesh_agent, addr } => {
                                    tracing::debug!("remapping mesh_agent {}: addr {} -> {}", mesh_agent, addr, forward_addr);
                                    mesh_agents_by_proc_id.insert(proc_id.clone(), mesh_agent.clone());
                                    router.bind(mesh_agent.actor_id().proc_id().clone().into(), addr);
                                    ProcState::Running { proc_id, mesh_agent, addr: forward_addr.clone() }
                                },
                                  ProcState::Stopped { proc_id, reason } => {
                                    match mesh_agents_by_proc_id.remove(&proc_id) {
                                        Some(mesh_agent) => {
                                            tracing::debug!("unmapping mesh_agent {}", mesh_agent);
                                            let agent_ref: Reference = mesh_agent.actor_id().proc_id().clone().into();
                                            router.unbind(&agent_ref);
                                        },
                                        None => {
                                            tracing::warn!("mesh_agent not found for proc_id: {}", proc_id);
                                        }
                                    }
                                    ProcState::Stopped { proc_id, reason }
                                },
                            };
                            tracing::debug!("sending event: {:?}", event);
                            tx.post(RemoteProcessProcStateMessage::Update(event));
                        }
                        None => {
                            tracing::debug!("sending done");
                            tx.post(RemoteProcessProcStateMessage::Done(alloc.world_id().clone()));
                            running = false;
                            break;
                        }
                    }
                }
                _ = RealClock.sleep(heartbeat_interval) => {
                    tracing::trace!("sending heartbeat");
                    tx.post(RemoteProcessProcStateMessage::HeartBeat);
                }
            }
        }
        tracing::debug!("allocation handler loop exited");
        if running {
            tracing::info!("stopping processes");
            if let Err(e) = alloc.stop_and_wait().await {
                tracing::error!("stop failed: {}", e);
                return;
            }
            tracing::info!("stop finished");
        }
    }
}

/// HostId is a string-based identifier of the host. It may be the same
/// as the hostname but in some situations a separate ID may be used.
type HostId = String;

/// A host entry passed down from the initializer.
#[derive(Clone)]
pub struct RemoteProcessAllocHost {
    /// A unique identifier of that host. Hostname may be used by ID can
    /// be different in some situations.
    pub id: HostId,
    /// The FQDN of the host.
    pub hostname: String,
}

struct RemoteProcessAllocHostState {
    tx: ChannelTx<RemoteProcessAllocatorMessage>,
    active_procs: HashSet<ProcId>,
    /// Slice offset for this host.
    offset: usize,
    /// World ID for this host as indicated from Allocated message.
    world_id: Option<WorldId>,
}

#[automock]
#[async_trait]
/// Interface to provide the set of hosts to be used by RemoteProcessAlloc.
pub trait RemoteProcessAllocInitializer {
    /// Initializes and returns a list of hosts to be used by this RemoteProcessAlloc.
    async fn initialize_alloc(&mut self) -> Result<Vec<RemoteProcessAllocHost>, anyhow::Error>;
}

/// A generalized implementation of an Alloc using one or more hosts running
/// RemoteProcessAlloc for process allocation.
pub struct RemoteProcessAlloc {
    // The initializer to be called at the first next() invocation to obtain
    // allocated hosts.
    initializer: Box<dyn RemoteProcessAllocInitializer + Send + Sync>,
    spec: AllocSpec,
    remote_allocator_port: u16,
    remote_allocator_heartbeat_interval: Duration,
    transport: ChannelTransport,
    world_id: WorldId,
    ordered_hosts: Vec<RemoteProcessAllocHost>,
    // Indicates that the initial remote allocation requests have been sent.
    started: bool,
    // Indicates that this Alloc is active (we have at least one remote process running).
    running: bool,
    hosts_by_offset: HashMap<usize, HostId>,
    host_states: HashMap<HostId, RemoteProcessAllocHostState>,
    world_shapes: HashMap<WorldId, Shape>,
    event_queue: VecDeque<ProcState>,
    comm_watcher_tx: UnboundedSender<HostId>,
    comm_watcher_rx: UnboundedReceiver<HostId>,

    bootstrap_addr: ChannelAddr,
    rx: ChannelRx<RemoteProcessProcStateMessage>,
}

impl RemoteProcessAlloc {
    /// Create a new Alloc. initializer will be called on the first invocation of next()
    /// to obtain a list of allocate hosts. Then Allocate message will be sent to all
    /// RemoteProcessAllocator on all hosts. Heartbeats will be used to maintain health
    /// status of remote hosts.
    pub async fn new(
        spec: AllocSpec,
        world_id: WorldId,
        transport: ChannelTransport,
        remote_allocator_port: u16,
        remote_allocator_heartbeat_interval: Duration,
        initializer: impl RemoteProcessAllocInitializer + Send + Sync + 'static,
    ) -> Result<Self, anyhow::Error> {
        let (bootstrap_addr, rx) = channel::serve(ChannelAddr::any(transport.clone()))
            .await
            .map_err(anyhow::Error::from)?;

        tracing::info!(
            "starting alloc for {} on: {}",
            world_id,
            bootstrap_addr.clone()
        );

        let (comm_watcher_tx, comm_watcher_rx) = unbounded_channel();

        Ok(Self {
            spec,
            world_id,
            transport,
            remote_allocator_port,
            remote_allocator_heartbeat_interval,
            initializer: Box::new(initializer),
            world_shapes: HashMap::new(),
            ordered_hosts: Vec::new(),
            hosts_by_offset: HashMap::new(),
            host_states: HashMap::new(),
            bootstrap_addr,
            event_queue: VecDeque::new(),
            comm_watcher_tx,
            comm_watcher_rx,
            rx,
            started: false,
            running: true,
        })
    }

    /// Start a RemoteProcessAllocator tx watcher. It spawns a task that monitor the underlying
    /// TxStatus and repot to the main next() loop via comm_watcher_tx. It does not however
    /// send the actual heartbteats. Just watches the status.
    /// The task will terminate once comm_watcher_rx is closed.
    ///
    /// A task approach was used here as opposed to select! to avoid nesting complexities with
    /// select!() and select_all().
    async fn start_comm_watcher(&self) {
        let mut tx_watchers = Vec::new();
        for host in &self.ordered_hosts {
            let tx_status = self.host_states.get(&host.id).unwrap().tx.status().clone();
            let watcher = WatchStream::new(tx_status);
            tx_watchers.push((watcher, host.id.clone()));
        }
        let tx = self.comm_watcher_tx.clone();
        tokio::spawn(async move {
            loop {
                let mut tx_status_futures = Vec::new();
                for (watcher, _) in &mut tx_watchers {
                    let fut = watcher.next().boxed();
                    tx_status_futures.push(fut);
                }
                let (tx_status, index, _) = select_all(tx_status_futures).await;
                let host_id = match tx_watchers.get(index) {
                    Some((_, host_id)) => host_id.clone(),
                    None => {
                        // Should never happen
                        tracing::error!(
                            "got selected index {} with no matching host in {}",
                            index,
                            tx_watchers.len()
                        );
                        continue;
                    }
                };
                if let Some(tx_status) = tx_status {
                    tracing::debug!("host {} channel event: {:?}", host_id, tx_status);
                    if tx_status == TxStatus::Closed {
                        if tx.send(host_id.clone()).is_err() {
                            // other side closed
                            break;
                        }
                        tx_watchers.remove(index);
                    }
                }
            }
        });
    }

    /// Ensure that we have the list of allocated hosts and that we send Allocate
    /// request to all o fthem. Once done, start_comm_watcher() is called to start
    /// the channel watcher.
    /// Function is idempotent.
    async fn ensure_started(&mut self) -> Result<(), anyhow::Error> {
        if self.started {
            return Ok(());
        }

        self.started = true;
        let hosts = self
            .initializer
            .initialize_alloc()
            .await
            .context("alloc initializer error")?;
        // prepare a list of host names in this allocation to be sent
        // to remote allocators.
        let hostnames: Vec<_> = hosts.iter().map(|e| e.hostname.clone()).collect();
        tracing::info!("obtained {} hosts for this allocation", hostnames.len());

        // disrtibuted procs based on most minor dimension
        for (host_index, host_shape) in (self
            .spec
            .shape
            .select_iter(self.spec.shape.labels().len() - 1)
            .context(format!(
                "failed to do select iterator for shape {}",
                self.spec.shape
            ))?)
        .enumerate()
        {
            let host = &hosts[host_index];
            tracing::debug!("allocating: {} for host: {}", host_shape, host.id);

            let remote_addr = match self.transport {
                ChannelTransport::MetaTls => {
                    format!("metatls!{}:{}", host.hostname, self.remote_allocator_port)
                }
                ChannelTransport::Tcp => {
                    format!("tcp!{}:{}", host.hostname, self.remote_allocator_port)
                }
                // Used only for testing.
                ChannelTransport::Unix => host.hostname.clone(),
                _ => {
                    anyhow::bail!(
                        "unsupported transport for host {}: {:?}",
                        host.id,
                        self.transport
                    );
                }
            };

            tracing::debug!("dialing remote: {} for host {}", remote_addr, host.id);
            let tx = channel::dial(remote_addr.parse()?)
                .map_err(anyhow::Error::from)
                .context(format!(
                    "failed to dial remote {} for host {}",
                    remote_addr, host.id
                ))?;
            tx.post(RemoteProcessAllocatorMessage::Allocate {
                bootstrap_addr: self.bootstrap_addr.clone(),
                spec: AllocSpec {
                    shape: host_shape.clone(),
                    constraints: self.spec.constraints.clone(),
                },
                hosts: hostnames.clone(),
                heartbeat_interval: self.remote_allocator_heartbeat_interval,
            });

            let offset = host_shape.slice().offset();
            self.hosts_by_offset.insert(offset, host.id.clone());
            self.host_states.insert(
                host.id.clone(),
                RemoteProcessAllocHostState {
                    tx,
                    active_procs: HashSet::new(),
                    offset,
                    world_id: None,
                },
            );
        }

        self.ordered_hosts = hosts;

        self.start_comm_watcher().await;

        self.started = true;

        Ok(())
    }

    fn host_id_for_world_id(&self, world_id: &WorldId) -> Option<HostId> {
        if let Some(shape) = self.world_shapes.get(world_id) {
            if let Some(host_id) = self.hosts_by_offset.get(&shape.slice().offset()) {
                return Some(host_id.clone());
            }
        }

        None
    }

    // Given a proc id, return the host id that it is running on.
    fn host_id_for_proc_id(&self, proc_id: &ProcId) -> Option<HostId> {
        self.host_id_for_world_id(proc_id.world_id())
    }

    // Given a proc_id, obtain the interal HostState structure.
    fn host_state_for_proc_id(
        &mut self,
        proc_id: &ProcId,
    ) -> Result<&mut RemoteProcessAllocHostState, anyhow::Error> {
        if let Some(host_id) = self.host_id_for_proc_id(proc_id) {
            if let Some(task_state) = self.host_states.get_mut(&host_id) {
                Ok(task_state)
            } else {
                // Should never happen
                anyhow::bail!(
                    "task state not found for proc id: {}, host id: {}",
                    proc_id,
                    host_id
                );
            }
        } else {
            // Should never happen
            anyhow::bail!("task not found for proc id: {}", proc_id);
        }
    }

    fn add_proc_id_to_host_state(&mut self, proc_id: &ProcId) -> Result<(), anyhow::Error> {
        let task_state = self.host_state_for_proc_id(proc_id)?;
        if !task_state.active_procs.insert(proc_id.clone()) {
            // Should not happen but we can ignore
            tracing::error!("proc id already in host state: {}", proc_id);
        }
        Ok(())
    }

    fn remove_proc_from_host_state(&mut self, proc_id: &ProcId) -> Result<(), anyhow::Error> {
        let task_state = self.host_state_for_proc_id(proc_id)?;
        if !task_state.active_procs.remove(proc_id) {
            // Should not happen but we can ignore
            tracing::error!("proc id already in host state: {}", proc_id);
        }
        Ok(())
    }

    // Reproject proc world coords to global shape coords.
    fn project_proc_into_global_shape(
        &self,
        proc_id: &ProcId,
        coords: &[usize],
    ) -> Result<Vec<usize>, anyhow::Error> {
        let world_id = proc_id.world_id();
        match self.world_shapes.get(world_id) {
            Some(shape) => {
                let world_location = match shape.slice().location(coords) {
                    Ok(world_location) => world_location,
                    Err(e) => anyhow::bail!(
                        "failed to get world location for coords: {:?}, world shape: {}: {}",
                        coords,
                        shape,
                        e
                    ),
                };

                match self.spec.shape.slice().coordinates(world_location) {
                    Ok(coords) => Ok(coords),
                    Err(e) => anyhow::bail!(
                        "failed to get coordinates for location: {}, shape: {}: {}",
                        world_location,
                        self.spec.shape,
                        e
                    ),
                }
            }
            None => anyhow::bail!("failed to find shape for world id: {}", world_id),
        }
    }

    // Cleanup a comm-failed host information by its ID.
    fn cleanup_host_channel_closed(
        &mut self,
        host_id: HostId,
    ) -> Result<Vec<ProcId>, anyhow::Error> {
        let state = match self.host_states.remove(&host_id) {
            Some(state) => state,
            None => {
                // this should never happen.
                anyhow::bail!(
                    "got channel closed event for host {} which has no known state",
                    host_id
                );
            }
        };
        self.ordered_hosts.retain(|host| host.id != host_id);
        self.hosts_by_offset.remove(&state.offset);
        if let Some(world_id) = state.world_id {
            self.world_shapes.remove(&world_id);
        }
        let proc_ids = state.active_procs.iter().cloned().collect();

        Ok(proc_ids)
    }
}

#[async_trait]
impl Alloc for RemoteProcessAlloc {
    async fn next(&mut self) -> Option<ProcState> {
        loop {
            if let state @ Some(_) = self.event_queue.pop_front() {
                break state;
            }

            if !self.running {
                break None;
            }

            if let Err(e) = self.ensure_started().await {
                tracing::error!("failed to ensure started: {}", e);
                break None;
            }

            let mut heartbeat_time =
                hyperactor::clock::RealClock.now() + self.remote_allocator_heartbeat_interval;
            // rerun outer loop in case we pushed new items to the event queue
            let mut reloop = false;
            let update = loop {
                tokio::select! {
                    msg = self.rx.recv() => {
                        tracing::trace!("received message: {:?}", msg);
                        match msg {
                            Ok(RemoteProcessProcStateMessage::Allocated{world_id, shape}) => {
                                tracing::info!("received allocated world id: {}", world_id);
                                match self.hosts_by_offset.get(&shape.slice().offset()) {
                                    Some(host_id) => {
                                        // update state
                                        match self.host_states.get_mut(host_id) {
                                            Some(state) => {
                                                state.world_id = Some(world_id.clone());
                                                if let Some(old_entry) = self.world_shapes.insert(world_id.clone(), shape) {
                                                    // should never happen
                                                    tracing::warn!("got allocated for duplicate world id: {} with known shape: {}", world_id, old_entry);
                                                }
                                            }
                                            None => {
                                                // should never happen
                                                tracing::error!("got allocated for host ID: {} with no known state", host_id);
                                            }
                                        }
                                    }
                                    None => {
                                        // should never happen
                                        tracing::error!("got allocated for unknown world: {}, shape: {}", world_id, shape);
                                    }
                                }
                            }
                            Ok(RemoteProcessProcStateMessage::Update(proc_state)) => {
                                match proc_state {
                                    ProcState::Created { ref proc_id, .. } => {
                                        if let Err(e) = self.add_proc_id_to_host_state(proc_id) {
                                            tracing::error!("failed to add proc id to host state: {}", e);
                                        }
                                    }
                                    ProcState::Stopped{ ref proc_id, ..} => {
                                        if let Err(e) = self.remove_proc_from_host_state(proc_id) {
                                            tracing::error!("failed to remove proc id from host state: {}", e);
                                        }
                                    }
                                    _ => {}
                                }
                                break Some(proc_state);
                            }
                            Ok(RemoteProcessProcStateMessage::Done(world_id)) => {
                                tracing::info!("allocator world_id: {} is done", world_id);
                                if let Some(host_id) = self.host_id_for_world_id(&world_id) {
                                    if let Some(state) = self.host_states.get(&host_id) {
                                        if !state.active_procs.is_empty() {
                                            tracing::error!("received done for world id: {} with active procs: {:?}", world_id, state.active_procs);
                                        }
                                    } else {
                                        tracing::error!("received done for unknown state world id: {}", world_id);
                                    }
                                } else {
                                    tracing::error!("received done for unknown world id: {}", world_id);
                                }
                                if self.world_shapes.remove(&world_id).is_none() {
                                    tracing::error!("received done for unknown world id: {}", world_id);
                                } else if self.world_shapes.is_empty() {
                                    self.running = false;
                                    break None;
                                }
                            }
                            Ok(RemoteProcessProcStateMessage::HeartBeat) => {}
                            Err(e) => {
                                tracing::error!("error receiving events: {}", e);
                                // We've lost our main listening channel. No fixing. Block and let
                                // caller timeout and recycle us.
                                hyperactor::clock::RealClock.sleep(std::time::Duration::from_secs(1)).await;
                            }
                        }
                    }

                    _ = clock::RealClock.sleep_until(heartbeat_time) => {
                        self.host_states.iter().for_each(|(_, host_state)| host_state.tx.post(RemoteProcessAllocatorMessage::HeartBeat));
                        heartbeat_time = hyperactor::clock::RealClock.now() + self.remote_allocator_heartbeat_interval;
                    }

                    closed_host_id = self.comm_watcher_rx.recv() => {
                        if let Some(closed_host_id) = closed_host_id {
                            tracing::debug!("host {} channel closed, cleaning up", closed_host_id);
                            let proc_ids = match self.cleanup_host_channel_closed(closed_host_id) {
                                Ok(proc_ids) => proc_ids,
                                Err(err) => {
                                    tracing::error!("failed to cleanup disconnected host: {}", err);
                                    continue;
                                }
                            };
                            for proc_id in proc_ids {
                                tracing::debug!("queuing Stopped state for {}", proc_id);
                                self.event_queue.push_back(ProcState::Stopped{proc_id, reason: ProcStopReason::HostWatchdog});
                            }
                            // Check if there are any hosts left
                            if self.host_states.is_empty() {
                                tracing::info!("no more hosts left, stopping the alloc");
                                self.running = false;
                            }
                            // Kick back to the outer loop to pop off the queue if necessary
                            reloop = true;
                            break None;
                        } else {
                            // other side closed
                            tracing::warn!("unexpected comm watcher channel close");
                            break None;
                        }
                    }
                }
            };

            if reloop {
                // Kick back to the outer loop to pop off the queue.
                // Should not have produced an update.
                assert!(update.is_none());
                continue;
            }

            break match update {
                Some(ProcState::Created { proc_id, coords }) => {
                    match self.project_proc_into_global_shape(&proc_id, &coords) {
                        Ok(global_coords) => {
                            tracing::debug!(
                                "reprojected coords: {:?} -> {:?}",
                                coords,
                                global_coords
                            );
                            Some(ProcState::Created {
                                proc_id,
                                coords: global_coords,
                            })
                        }
                        Err(e) => {
                            tracing::error!(
                                "failed to project coords for proc: {}: {}",
                                proc_id,
                                e
                            );
                            None
                        }
                    }
                }

                _ => update,
            };
        }
    }

    fn shape(&self) -> &Shape {
        &self.spec.shape
    }

    fn world_id(&self) -> &WorldId {
        &self.world_id
    }

    fn transport(&self) -> ChannelTransport {
        self.transport.clone()
    }

    async fn stop(&mut self) -> Result<(), AllocatorError> {
        tracing::info!("stopping alloc");

        for (host_id, task_state) in self.host_states.iter_mut() {
            tracing::debug!("stopping alloc at host {}", host_id);
            task_state.tx.post(RemoteProcessAllocatorMessage::Stop);
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::assert_matches::assert_matches;

    use hyperactor::ActorRef;
    use hyperactor::channel::ChannelRx;
    use hyperactor::id;
    use ndslice::shape;
    use tokio::sync::oneshot;

    use super::*;
    use crate::alloc::ChannelTransport;
    use crate::alloc::MockAlloc;
    use crate::alloc::MockAllocWrapper;
    use crate::alloc::MockAllocator;
    use crate::alloc::ProcStopReason;
    use crate::proc_mesh::mesh_agent::MeshAgent;

    async fn read_all_created(rx: &mut ChannelRx<RemoteProcessProcStateMessage>, alloc_len: usize) {
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Created { .. }) => i += 1,
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
    }

    async fn read_all_running(rx: &mut ChannelRx<RemoteProcessProcStateMessage>, alloc_len: usize) {
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Running { .. }) => i += 1,
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
    }

    async fn read_all_stopped(rx: &mut ChannelRx<RemoteProcessProcStateMessage>, alloc_len: usize) {
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Stopped { .. }) => i += 1,
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
    }

    fn set_procstate_execptations(alloc: &mut MockAlloc, shape: Shape) {
        let alloc_len = shape.slice().len();
        alloc.expect_shape().return_const(shape.clone());
        for i in 0..alloc_len {
            let proc_id = format!("test[{}]", i).parse().unwrap();
            let coords = shape.slice().coordinates(i).unwrap();
            alloc
                .expect_next()
                .times(1)
                .return_once(|| Some(ProcState::Created { proc_id, coords }));
        }
        for i in 0..alloc_len {
            let proc_id = format!("test[{}]", i).parse().unwrap();
            let mesh_agent = ActorRef::<MeshAgent>::attest(
                format!("test[{}].mesh_agent[{}]", i, i).parse().unwrap(),
            );
            alloc.expect_next().times(1).return_once(|| {
                Some(ProcState::Running {
                    proc_id,
                    addr: ChannelAddr::Unix("/proc0".parse().unwrap()),
                    mesh_agent,
                })
            });
        }
        for i in 0..alloc_len {
            let proc_id = format!("test[{}]", i).parse().unwrap();
            alloc.expect_next().times(1).return_once(|| {
                Some(ProcState::Stopped {
                    proc_id,
                    reason: ProcStopReason::Unknown,
                })
            });
        }
    }

    #[timed_test::async_timed_test(timeout_secs = 5)]
    async fn test_simple() {
        hyperactor_telemetry::initialize_logging();
        let serve_addr = ChannelAddr::any(ChannelTransport::Unix);
        let bootstrap_addr = ChannelAddr::any(ChannelTransport::Unix);
        let (_, mut rx) = channel::serve(bootstrap_addr.clone()).await.unwrap();

        let spec = AllocSpec {
            shape: shape!(host = 1, gpu = 2),
            constraints: Default::default(),
        };
        let tx = channel::dial(serve_addr.clone()).unwrap();

        let alloc_len = spec.shape.slice().len();

        let world_id: WorldId = id!(test_world_id);
        let mut alloc = MockAlloc::new();
        alloc.expect_world_id().return_const(world_id.clone());
        alloc.expect_shape().return_const(spec.shape.clone());

        set_procstate_execptations(&mut alloc, spec.shape.clone());

        // final none
        alloc.expect_next().return_const(None);

        let mut allocator = MockAllocator::new();
        let total_messages = alloc_len * 3 + 1;
        let mock_wrapper = MockAllocWrapper::new_block_next(
            alloc,
            // block after create, running, stopped and done.
            total_messages,
        );
        allocator
            .expect_allocate()
            .times(1)
            .return_once(move |_| Ok(mock_wrapper));

        let remote_allocator = RemoteProcessAllocator::new();
        let handle = tokio::spawn({
            let remote_allocator = remote_allocator.clone();
            async move {
                remote_allocator
                    .start_with_allocator(serve_addr, allocator)
                    .await
            }
        });

        tx.send(RemoteProcessAllocatorMessage::Allocate {
            spec: spec.clone(),
            bootstrap_addr,
            hosts: vec![],
            heartbeat_interval: Duration::from_secs(1),
        })
        .await
        .unwrap();

        // Allocated
        let m = rx.recv().await.unwrap();
        assert!(
            matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape)
        );

        // All Created events
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Created { proc_id, coords }) => {
                    let expected_proc_id = format!("test[{}]", i).parse().unwrap();
                    let expected_coords = spec.shape.slice().coordinates(i).unwrap();
                    assert_eq!(proc_id, expected_proc_id);
                    assert_eq!(coords, expected_coords);
                    i += 1;
                }
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
        // All Running events
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Running {
                    proc_id,
                    mesh_agent,
                    addr: _,
                }) => {
                    let expected_proc_id = format!("test[{}]", i).parse().unwrap();
                    let expected_mesh_agent = ActorRef::<MeshAgent>::attest(
                        format!("test[{}].mesh_agent[{}]", i, i).parse().unwrap(),
                    );
                    assert_eq!(proc_id, expected_proc_id);
                    assert_eq!(mesh_agent, expected_mesh_agent);
                    i += 1;
                }
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
        // All Stopped events
        let mut i: usize = 0;
        while i < alloc_len {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Update(ProcState::Stopped {
                    proc_id,
                    reason: ProcStopReason::Unknown,
                }) => {
                    let expected_proc_id = format!("test[{}]", i).parse().unwrap();
                    assert_eq!(proc_id, expected_proc_id);
                    i += 1;
                }
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }
        // Done
        loop {
            let m = rx.recv().await.unwrap();
            match m {
                RemoteProcessProcStateMessage::Done(id) => {
                    assert_eq!(id, world_id);
                    break;
                }
                RemoteProcessProcStateMessage::HeartBeat => {}
                _ => panic!("unexpected message: {:?}", m),
            }
        }

        remote_allocator.terminate();
        handle.await.unwrap().unwrap();
    }

    #[timed_test::async_timed_test(timeout_secs = 15)]
    async fn test_normal_stop() {
        hyperactor_telemetry::initialize_logging();
        let serve_addr = ChannelAddr::any(ChannelTransport::Unix);
        let bootstrap_addr = ChannelAddr::any(ChannelTransport::Unix);
        let (_, mut rx) = channel::serve(bootstrap_addr.clone()).await.unwrap();

        let spec = AllocSpec {
            shape: shape!(host = 1, gpu = 2),
            constraints: Default::default(),
        };
        let tx = channel::dial(serve_addr.clone()).unwrap();

        let alloc_len = spec.shape.slice().len();

        let world_id: WorldId = id!(test_world_id);
        let mut alloc = MockAllocWrapper::new_block_next(
            MockAlloc::new(),
            // block after all created, all running
            alloc_len * 2,
        );
        let next_tx = alloc.notify_tx();
        alloc.alloc.expect_world_id().return_const(world_id.clone());
        alloc.alloc.expect_shape().return_const(spec.shape.clone());

        set_procstate_execptations(&mut alloc.alloc, spec.shape.clone());

        alloc.alloc.expect_next().return_const(None);
        alloc.alloc.expect_stop().times(1).return_once(|| Ok(()));

        let mut allocator = MockAllocator::new();
        allocator
            .expect_allocate()
            .times(1)
            .return_once(|_| Ok(alloc));

        let remote_allocator = RemoteProcessAllocator::new();
        let handle = tokio::spawn({
            let remote_allocator = remote_allocator.clone();
            async move {
                remote_allocator
                    .start_with_allocator(serve_addr, allocator)
                    .await
            }
        });

        tx.send(RemoteProcessAllocatorMessage::Allocate {
            spec: spec.clone(),
            bootstrap_addr,
            hosts: vec![],
            heartbeat_interval: Duration::from_millis(200),
        })
        .await
        .unwrap();

        // Allocated
        let m = rx.recv().await.unwrap();
        assert_matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape);

        read_all_created(&mut rx, alloc_len).await;
        read_all_running(&mut rx, alloc_len).await;

        // allocation finished. now we stop it.
        tracing::info!("stopping allocation");
        tx.send(RemoteProcessAllocatorMessage::Stop).await.unwrap();
        // receive all stops
        next_tx.send(()).unwrap();

        read_all_stopped(&mut rx, alloc_len).await;

        remote_allocator.terminate();
        handle.await.unwrap().unwrap();
    }

    #[timed_test::async_timed_test(timeout_secs = 15)]
    async fn test_realloc() {
        hyperactor_telemetry::initialize_logging();
        let serve_addr = ChannelAddr::any(ChannelTransport::Unix);
        let bootstrap_addr = ChannelAddr::any(ChannelTransport::Unix);
        let (_, mut rx) = channel::serve(bootstrap_addr.clone()).await.unwrap();

        let spec = AllocSpec {
            shape: shape!(host = 1, gpu = 2),
            constraints: Default::default(),
        };
        let tx = channel::dial(serve_addr.clone()).unwrap();

        let alloc_len = spec.shape.slice().len();

        let world_id: WorldId = id!(test_world_id);
        let mut alloc1 = MockAllocWrapper::new_block_next(
            MockAlloc::new(),
            // block after all created, all running
            alloc_len * 2,
        );
        let next_tx1 = alloc1.notify_tx();
        alloc1
            .alloc
            .expect_world_id()
            .return_const(world_id.clone());
        alloc1.alloc.expect_shape().return_const(spec.shape.clone());

        set_procstate_execptations(&mut alloc1.alloc, spec.shape.clone());
        alloc1.alloc.expect_next().return_const(None);
        alloc1.alloc.expect_stop().times(1).return_once(|| Ok(()));
        // second allocation
        let mut alloc2 = MockAllocWrapper::new_block_next(
            MockAlloc::new(),
            // block after all created, all running
            alloc_len * 2,
        );
        let next_tx2 = alloc2.notify_tx();
        alloc2
            .alloc
            .expect_world_id()
            .return_const(world_id.clone());
        alloc2.alloc.expect_shape().return_const(spec.shape.clone());
        set_procstate_execptations(&mut alloc2.alloc, spec.shape.clone());
        alloc2.alloc.expect_next().return_const(None);
        alloc2.alloc.expect_stop().times(1).return_once(|| Ok(()));

        let mut allocator = MockAllocator::new();
        allocator
            .expect_allocate()
            .times(1)
            .return_once(|_| Ok(alloc1));
        // second alloc
        allocator
            .expect_allocate()
            .times(1)
            .return_once(|_| Ok(alloc2));

        let remote_allocator = RemoteProcessAllocator::new();
        let handle = tokio::spawn({
            let remote_allocator = remote_allocator.clone();
            async move {
                remote_allocator
                    .start_with_allocator(serve_addr, allocator)
                    .await
            }
        });

        tx.send(RemoteProcessAllocatorMessage::Allocate {
            spec: spec.clone(),
            bootstrap_addr: bootstrap_addr.clone(),
            hosts: vec![],
            heartbeat_interval: Duration::from_millis(200),
        })
        .await
        .unwrap();

        // Allocated
        let m = rx.recv().await.unwrap();
        assert_matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape);

        read_all_created(&mut rx, alloc_len).await;
        read_all_running(&mut rx, alloc_len).await;

        // allocation finished now we request a new one
        tx.send(RemoteProcessAllocatorMessage::Allocate {
            spec: spec.clone(),
            bootstrap_addr,
            hosts: vec![],
            heartbeat_interval: Duration::from_millis(200),
        })
        .await
        .unwrap();
        // unblock next for the first allocation
        next_tx1.send(()).unwrap();
        // we expect a stop(), then Stopped proc states, then a new Allocated
        read_all_stopped(&mut rx, alloc_len).await;
        let m = rx.recv().await.unwrap();
        assert_matches!(m, RemoteProcessProcStateMessage::Done(_));
        let m = rx.recv().await.unwrap();
        assert_matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape);
        // ProcStates for the new allocation
        read_all_created(&mut rx, alloc_len).await;
        read_all_running(&mut rx, alloc_len).await;
        // finally stop
        tracing::info!("stopping allocation");
        tx.send(RemoteProcessAllocatorMessage::Stop).await.unwrap();
        // receive all stops
        next_tx2.send(()).unwrap();

        read_all_stopped(&mut rx, alloc_len).await;

        remote_allocator.terminate();
        handle.await.unwrap().unwrap();
    }

    #[timed_test::async_timed_test(timeout_secs = 15)]
    async fn test_upstream_closed() {
        // Use temporary config for this test
        let _guard = hyperactor::config::global::set_temp_config(hyperactor::config::Config {
            message_delivery_timeout: Duration::from_secs(1),
            ..Default::default()
        });

        hyperactor_telemetry::initialize_logging();
        let serve_addr = ChannelAddr::any(ChannelTransport::Unix);
        let bootstrap_addr = ChannelAddr::any(ChannelTransport::Unix);
        let (_, mut rx) = channel::serve(bootstrap_addr.clone()).await.unwrap();

        let spec = AllocSpec {
            shape: shape!(host = 1, gpu = 2),
            constraints: Default::default(),
        };
        let tx = channel::dial(serve_addr.clone()).unwrap();

        let alloc_len = spec.shape.slice().len();

        let world_id: WorldId = id!(test_world_id);
        let mut alloc = MockAllocWrapper::new_block_next(
            MockAlloc::new(),
            // block after all created, all running
            alloc_len * 2,
        );
        let next_tx = alloc.notify_tx();
        alloc.alloc.expect_world_id().return_const(world_id.clone());
        alloc.alloc.expect_shape().return_const(spec.shape.clone());

        set_procstate_execptations(&mut alloc.alloc, spec.shape.clone());

        alloc.alloc.expect_next().return_const(None);
        // we expect a stop due to the failure
        // synchronize test with the stop
        let (stop_tx, stop_rx) = oneshot::channel();
        alloc.alloc.expect_stop().times(1).return_once(|| {
            stop_tx.send(()).unwrap();
            Ok(())
        });

        let mut allocator = MockAllocator::new();
        allocator
            .expect_allocate()
            .times(1)
            .return_once(|_| Ok(alloc));

        let remote_allocator = RemoteProcessAllocator::new();
        let handle = tokio::spawn({
            let remote_allocator = remote_allocator.clone();
            async move {
                remote_allocator
                    .start_with_allocator(serve_addr, allocator)
                    .await
            }
        });

        tx.send(RemoteProcessAllocatorMessage::Allocate {
            spec: spec.clone(),
            bootstrap_addr,
            hosts: vec![],
            heartbeat_interval: Duration::from_millis(200),
        })
        .await
        .unwrap();

        // Allocated
        let m = rx.recv().await.unwrap();
        assert_matches!(m, RemoteProcessProcStateMessage::Allocated {world_id, shape} if world_id == world_id && shape == spec.shape);

        read_all_created(&mut rx, alloc_len).await;
        read_all_running(&mut rx, alloc_len).await;

        // allocation finished. terminate connection.
        tracing::info!("closing upstream");
        drop(rx);
        // wait for the heartbeat to expire
        #[allow(clippy::disallowed_methods)]
        tokio::time::sleep(Duration::from_secs(2)).await;
        // wait for the stop to be called
        stop_rx.await.unwrap();
        // unblock next
        next_tx.send(()).unwrap();
        remote_allocator.terminate();
        handle.await.unwrap().unwrap();
    }
}

#[cfg(test)]
mod test_alloc {
    use ndslice::shape;
    use timed_test::async_timed_test;

    use super::*;

    #[async_timed_test(timeout_secs = 60)]
    async fn test_alloc_simple() {
        // Use temporary config for this test
        let _guard = hyperactor::config::global::set_temp_config(hyperactor::config::Config {
            message_delivery_timeout: Duration::from_secs(1),
            ..Default::default()
        });
        hyperactor_telemetry::initialize_logging();

        let spec = AllocSpec {
            shape: shape!(host = 2, gpu = 2),
            constraints: Default::default(),
        };
        let world_id = WorldId("test_world_id".to_string());
        let transport = ChannelTransport::Unix;
        let heartbeat = Duration::from_millis(100);

        let task1_allocator = RemoteProcessAllocator::new();
        let task1_addr = ChannelAddr::any(ChannelTransport::Unix);
        let task1_addr_string = task1_addr.to_string();
        let task1_cmd =
            Command::new(buck_resources::get("monarch/hyperactor_mesh/bootstrap").unwrap());
        let task2_allocator = RemoteProcessAllocator::new();
        let task2_addr = ChannelAddr::any(ChannelTransport::Unix);
        let task2_addr_string = task2_addr.to_string();
        let task2_cmd =
            Command::new(buck_resources::get("monarch/hyperactor_mesh/bootstrap").unwrap());
        let task1_allocator_copy = task1_allocator.clone();
        let task1_allocator_handle = tokio::spawn(async move {
            tracing::info!("spawning task1");
            task1_allocator_copy
                .start(task1_cmd, task1_addr)
                .await
                .unwrap();
        });
        let task2_allocator_copy = task2_allocator.clone();
        let task2_allocator_handle = tokio::spawn(async move {
            task2_allocator_copy
                .start(task2_cmd, task2_addr)
                .await
                .unwrap();
        });

        let mut initializer = MockRemoteProcessAllocInitializer::new();
        initializer.expect_initialize_alloc().return_once(move || {
            Ok(vec![
                RemoteProcessAllocHost {
                    hostname: task1_addr_string,
                    id: "task1".to_string(),
                },
                RemoteProcessAllocHost {
                    hostname: task2_addr_string,
                    id: "task2".to_string(),
                },
            ])
        });
        let mut alloc =
            RemoteProcessAlloc::new(spec.clone(), world_id, transport, 0, heartbeat, initializer)
                .await
                .unwrap();
        let mut procs = HashSet::new();
        let mut started_procs = HashSet::new();
        let mut proc_coords = HashSet::new();
        let alloc_len = spec.shape.slice().len();
        for _ in 0..alloc_len * 2 {
            let proc_state = alloc.next().await.unwrap();
            tracing::debug!("test got message: {:?}", proc_state);
            match proc_state {
                ProcState::Created { proc_id, coords } => {
                    procs.insert(proc_id);
                    proc_coords.insert(coords);
                }
                ProcState::Running { proc_id, .. } => {
                    assert!(procs.contains(&proc_id));
                    started_procs.insert(proc_id);
                }
                _ => panic!("expected Created or Running"),
            }
        }
        assert_eq!(procs, started_procs);
        // ensure coords coverage
        for rank in 0..spec.shape.slice().len() {
            let coords = spec.shape.slice().coordinates(rank).unwrap();
            assert!(proc_coords.contains(&coords));
        }

        // ensure no more pending items
        let timeout = hyperactor::clock::RealClock.now() + std::time::Duration::from_millis(1000);
        tokio::select! {
            _ = hyperactor::clock::RealClock
            .sleep_until(timeout) => {},
            _ = alloc.next() => panic!("expected no more items"),
        }

        // stop the allocation
        alloc.stop().await.unwrap();
        for _ in 0..spec.shape.slice().len() {
            let proc_state = alloc.next().await.unwrap();
            tracing::info!("test received next proc_state: {:?}", proc_state);
            match proc_state {
                ProcState::Stopped { proc_id, reason } => {
                    assert!(started_procs.remove(&proc_id));
                    assert_eq!(reason, ProcStopReason::Stopped);
                }
                _ => panic!("expected stopped"),
            }
        }
        // Exactly one None
        let proc_state = alloc.next().await;
        assert!(proc_state.is_none());
        // Anything afterwards is None
        let proc_state = alloc.next().await;
        assert!(proc_state.is_none());

        task1_allocator.terminate();
        task1_allocator_handle.await.unwrap();
        task2_allocator.terminate();
        task2_allocator_handle.await.unwrap();
    }

    #[async_timed_test(timeout_secs = 60)]
    async fn test_alloc_host_failure() {
        // Use temporary config for this test
        let _guard = hyperactor::config::global::set_temp_config(hyperactor::config::Config {
            message_delivery_timeout: Duration::from_secs(1),
            ..Default::default()
        });
        hyperactor_telemetry::initialize_logging();

        let spec = AllocSpec {
            shape: shape!(host = 2, gpu = 2),
            constraints: Default::default(),
        };
        let world_id = WorldId("test_world_id".to_string());
        let transport = ChannelTransport::Unix;
        let heartbeat = Duration::from_millis(100);

        let task1_allocator = RemoteProcessAllocator::new();
        let task1_addr = ChannelAddr::any(ChannelTransport::Unix);
        let task1_addr_string = task1_addr.to_string();
        let task1_cmd =
            Command::new(buck_resources::get("monarch/hyperactor_mesh/bootstrap").unwrap());
        let task2_allocator = RemoteProcessAllocator::new();
        let task2_addr = ChannelAddr::any(ChannelTransport::Unix);
        let task2_addr_string = task2_addr.to_string();
        let task2_cmd =
            Command::new(buck_resources::get("monarch/hyperactor_mesh/bootstrap").unwrap());
        let task1_allocator_copy = task1_allocator.clone();
        let task1_allocator_handle = tokio::spawn(async move {
            tracing::info!("spawning task1");
            task1_allocator_copy
                .start(task1_cmd, task1_addr)
                .await
                .unwrap();
            tracing::info!("task1 terminated");
        });
        let task2_allocator_copy = task2_allocator.clone();
        let task2_allocator_handle = tokio::spawn(async move {
            task2_allocator_copy
                .start(task2_cmd, task2_addr)
                .await
                .unwrap();
            tracing::info!("task2 terminated");
        });

        let mut initializer = MockRemoteProcessAllocInitializer::new();
        initializer.expect_initialize_alloc().return_once(move || {
            Ok(vec![
                RemoteProcessAllocHost {
                    hostname: task1_addr_string,
                    id: "task1".to_string(),
                },
                RemoteProcessAllocHost {
                    hostname: task2_addr_string,
                    id: "task2".to_string(),
                },
            ])
        });
        let mut alloc =
            RemoteProcessAlloc::new(spec.clone(), world_id, transport, 0, heartbeat, initializer)
                .await
                .unwrap();
        let alloc_len = spec.shape.slice().len();
        for _ in 0..alloc_len * 2 {
            match alloc.next().await {
                Some(ProcState::Created { .. }) | Some(ProcState::Running { .. }) => {}
                _ => panic!("expected Created or Running"),
            }
        }

        // ensure no more pending items
        let timeout = hyperactor::clock::RealClock.now() + std::time::Duration::from_millis(1000);
        tokio::select! {
            _ = hyperactor::clock::RealClock
            .sleep_until(timeout) => {},
            _ = alloc.next() => panic!("expected no more items"),
        }

        // now we kill task1 and wait for timeout
        tracing::info!("aborting task1 allocator");
        task1_allocator_handle.abort();
        RealClock.sleep(heartbeat * 2).await;
        for _ in 0..spec.shape.slice().len() / 2 {
            let proc_state = alloc.next().await.unwrap();
            tracing::info!("test received next proc_state: {:?}", proc_state);
            match proc_state {
                ProcState::Stopped { proc_id: _, reason } => {
                    assert_eq!(reason, ProcStopReason::HostWatchdog);
                }
                _ => panic!("expected stopped"),
            }
        }
        // no more events
        let timeout = hyperactor::clock::RealClock.now() + std::time::Duration::from_millis(1000);
        tokio::select! {
            _ = hyperactor::clock::RealClock
            .sleep_until(timeout) => {},
            _ = alloc.next() => panic!("expected no more items"),
        }

        // abort the second host
        tracing::info!("aborting task2 allocator");
        task2_allocator_handle.abort();
        RealClock.sleep(heartbeat * 2).await;
        for _ in 0..spec.shape.slice().len() / 2 {
            let proc_state = alloc.next().await.unwrap();
            tracing::info!("test received next proc_state: {:?}", proc_state);
            match proc_state {
                ProcState::Stopped { proc_id: _, reason } => {
                    assert_eq!(reason, ProcStopReason::HostWatchdog);
                }
                _ => panic!("expected stopped"),
            }
        }
        // now the alloc has stopped we always get None
        let proc_state = alloc.next().await;
        assert!(proc_state.is_none());
        // Anything afterwards is None
        let proc_state = alloc.next().await;
        assert!(proc_state.is_none());
    }
}
