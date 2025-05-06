use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use const_format::concatcp;
use hpcscheduler::HpcTaskExecutionAttempt;
use hpcscheduler::HpcTaskGroupExecutionAttempt;
use hpcscheduler::HpcTaskGroupState;
use hpcscheduler_srclients;
use hpcscheduler_srclients::HpcSchedulerReadOnlyService;
use hyperactor::ProcId;
use hyperactor::WorldId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelRx;
use hyperactor::channel::ChannelTransport;
use hyperactor::channel::ChannelTx;
use hyperactor::channel::Rx;
use hyperactor::channel::Tx;
use hyperactor::clock;
use hyperactor::clock::Clock;
use hyperactor_mesh::alloc::Alloc;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::AllocatorError;
use hyperactor_mesh::alloc::ProcState;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAllocatorMessage;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessProcStateMessage;
use mockall::automock;
use ndslice::Shape;
use ndslice::Slice;
use regex::Regex;
use serde::Deserialize;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::sync::Mutex;

/// Domain name for all monarch reserved labels.
/// TODO: Consolidate.
pub static MONARCH_LABEL_PREFIX: &str = "monarch.meta.com/";
/// Force allocator to use provided task group by name.
pub static ALLOC_LABEL_TASK_GROUP: &str = concatcp!("mast.", MONARCH_LABEL_PREFIX, "taskGroup");

/// TODO: Remove once we have a way to extract it from the task group API.
static DEFAULT_REMOTE_ALLOCATOR_PORT: u16 = 26600;
static DEFAULT_TASK_GROUP_REFRESH_INTERVAL: Duration = Duration::from_secs(10);

/// MAST/TW task ID.
type TaskId = String;

/// Trait to abstract MAST task group task ranking.
#[async_trait]
pub trait TaskRanker {
    async fn rank(
        &self,
        tasks: Vec<(TaskId, HpcTaskExecutionAttempt)>,
    ) -> Result<Vec<(TaskId, HpcTaskExecutionAttempt)>, anyhow::Error>;
}

struct TaskRemoteState {
    tx: ChannelTx<RemoteProcessAllocatorMessage>,
    attempt_index: i32,
    active_procs: HashSet<ProcId>,
}

/// Implementation of a MAST task group based Alloc. For now it _implicitly_
/// assumes that there is only one Alloc per task group.
pub struct TaskGroupAlloc {
    task_group_getter: Arc<Mutex<dyn TaskGroupGetter + Send + Sync>>,
    task_ranker: Box<dyn TaskRanker + Send + Sync>,
    ordered_tasks: Vec<(TaskId, HpcTaskExecutionAttempt)>,
    spec: AllocSpec,
    remote_allocator_port: u16,
    transport: ChannelTransport,
    world_id: WorldId,
    task_states: HashMap<TaskId, TaskRemoteState>,
    task_by_offset: HashMap<usize, TaskId>,
    rx: ChannelRx<RemoteProcessProcStateMessage>,
    // Indicates that the initial remote allocation requests have been sent.
    started: bool,
    // Indicates that this Alloc is active (we have at least one remote process running).
    running: bool,
    world_shapes: HashMap<WorldId, Shape>,
    event_queue: VecDeque<ProcState>,

    bootstrap_addr: ChannelAddr,
    task_group_name: String,
    job_version: i32,
    job_attempt_index: i32,
    task_group_attempt_index: i32,
    task_group_state: HpcTaskGroupState,
}

impl TaskGroupAlloc {
    /// Create a new Alloc that fetches status using task_group_getter and ranks tasks
    /// using task_ranker with spec. It will be associated with task_group_name.
    /// Note that it implicitly assumes that no other Alloc has been created for the same
    /// task group.
    async fn new(
        task_group_getter: Arc<Mutex<dyn TaskGroupGetter + Send + Sync>>,
        task_ranker: Box<dyn TaskRanker + Send + Sync>,
        task_group_name: String,
        spec: AllocSpec,
        remote_allocator_port: u16,
        transport: ChannelTransport,
    ) -> Result<Self, anyhow::Error> {
        let (bootstrap_addr, rx) = channel::serve(ChannelAddr::any(transport.clone()))
            .await
            .map_err(anyhow::Error::from)?;

        tracing::info!(
            "starting alloc for task group: {}, on: {}",
            task_group_name,
            bootstrap_addr.clone()
        );

        let alloc = Self {
            task_group_getter,
            task_ranker,
            task_group_name: task_group_name.clone(),
            ordered_tasks: Vec::new(),
            spec,
            remote_allocator_port,
            transport,
            world_id: Self::get_world_id(task_group_name)?,
            task_states: HashMap::new(),
            rx,
            started: false,
            running: true,
            world_shapes: HashMap::new(),
            task_by_offset: HashMap::new(),
            event_queue: VecDeque::new(),
            bootstrap_addr,
            job_version: 0,
            job_attempt_index: 0,
            task_group_attempt_index: 0,
            task_group_state: HpcTaskGroupState::UNKNOWN,
        };

        Ok(alloc)
    }

    /// Ensure that the remote allocation requests have been sent.
    async fn ensure_started(&mut self) -> Result<(), anyhow::Error> {
        if self.started {
            return Ok(());
        }

        // prepare a list of host names in this allocation to be sent
        // to remote allocators.
        let hosts: Vec<_> = self
            .ordered_tasks
            .iter()
            // unwrap is safe because we checked for hostnames before.
            .map(|(_, attempt)| attempt.hostname.as_ref().unwrap().clone())
            .collect();
        // disrtibuted procs based on most minor dimension
        for (task_index, task_shape) in (self
            .spec
            .shape
            .select_iter(self.spec.shape.labels().len() - 1)
            .context(format!(
                "failed to do select iterator for shape {}",
                self.spec.shape
            ))?)
        .enumerate()
        {
            let (task_id, task) = &self.ordered_tasks[task_index];
            tracing::debug!("allocating: {} for task: {}", task_shape, task_id);

            let hostname = match task.hostname {
                Some(ref hostname) => hostname.clone(),
                None => anyhow::bail!("expected to find hostname for task {}", task_id),
            };
            let remote_addr = match self.transport {
                ChannelTransport::MetaTls => {
                    format!("metatls!:{}:{}", hostname, self.remote_allocator_port)
                }
                ChannelTransport::Tcp => format!("tcp!{}:{}", hostname, self.remote_allocator_port),
                // Used only for testing.
                ChannelTransport::Unix => hostname,
                _ => {
                    anyhow::bail!(
                        "unsupported transport for task {}: {:?}",
                        task_id,
                        self.transport
                    );
                }
            };
            let tx = channel::dial(remote_addr.parse()?)
                .map_err(anyhow::Error::from)
                .context(format!(
                    "failed to dial remote {} for task {}",
                    remote_addr, task_id
                ))?;
            tx.send(RemoteProcessAllocatorMessage::Allocate {
                bootstrap_addr: self.bootstrap_addr.clone(),
                spec: AllocSpec {
                    shape: task_shape.clone(),
                    constraints: self.spec.constraints.clone(),
                },
                hosts: hosts.clone(),
            })
            .await
            .map_err(anyhow::Error::from)
            .context(format!(
                "failed to send allocate message to {} for task {}",
                remote_addr, task_id,
            ))?;

            self.task_by_offset
                .insert(task_shape.slice().offset(), task_id.clone());
            self.task_states.insert(
                task_id.clone(),
                TaskRemoteState {
                    tx,
                    attempt_index: task.attemptIndex,
                    active_procs: HashSet::new(),
                },
            );
        }

        self.started = true;

        Ok(())
    }

    /// Block until successful retrieval of current status.
    async fn get_task_group_status(&self) -> Result<(i32, i32, HpcTaskGroupExecutionAttempt)> {
        loop {
            match self
                .task_group_getter
                .lock()
                .await
                .get_task_group_status()
                .await
            {
                Ok((job_version, job_attempt_index, task_group_attempt)) => {
                    if task_group_attempt.state != HpcTaskGroupState::RUNNING {
                        tracing::warn!("task group is not RUNNING: {}", task_group_attempt.state);
                    } else {
                        break Ok((job_version, job_attempt_index, task_group_attempt));
                    }
                }
                Err(e) => {
                    tracing::warn!("unable to get task group status: {}", e);
                }
            }
            hyperactor::clock::RealClock
                .sleep(std::time::Duration::from_secs(1))
                .await;
        }
    }

    /// Check the health status of the task group to:
    /// 1. Check task group if:
    ///    a. Job has been updated to a newer version.
    ///    b. Job has been restarted by checking its attempts.
    ///    c. Task group has been restarted by checking its attemptIndex.
    /// 2. Check each task if:
    ///    a. Attempt index has changed indicating a restart.
    ///    b. Task is no longer found in the task group.
    ///
    /// If any of the above was found, the corresponding task IDs are returned.
    async fn check_task_group_health(&mut self) -> Result<Vec<TaskId>, anyhow::Error> {
        if self.task_group_state != HpcTaskGroupState::RUNNING {
            anyhow::bail!(
                "check_task_group_health should only be called when task group is running"
            );
        }
        let (job_version, job_attempt_index, task_group_attempt) =
            self.get_task_group_status().await?;

        // If already running validate task group wasn't restarted.
        if self.job_version != job_version
            || self.job_attempt_index != job_attempt_index
            || self.task_group_attempt_index != task_group_attempt.attemptIndex
        {
            tracing::error!(
                "task group {} was restarted: job version: {}->{}, job attempt: {}->{}, task attempt: {}->{}",
                self.task_group_name,
                self.job_version,
                job_version,
                self.job_attempt_index,
                job_attempt_index,
                self.task_group_attempt_index,
                task_group_attempt.attemptIndex
            );
            return Ok(self.task_states.keys().cloned().collect());
        }

        // check all tasks are ok
        let mut failed_tasks = Vec::new();
        for (task_id, task_state) in self.task_states.iter() {
            let attempts = match task_group_attempt.taskExecutionAttempts.get(task_id) {
                Some(attempts) => attempts,
                None => {
                    tracing::error!("task {} is no longer found in task group", task_id);
                    failed_tasks.push(task_id.clone());
                    continue;
                }
            };
            let latest_attempt = match attempts.last() {
                Some(attempt) => attempt,
                None => {
                    tracing::error!(
                        "task {} is no attempts, used to be on attempt {}",
                        task_id,
                        task_state.attempt_index
                    );
                    failed_tasks.push(task_id.clone());
                    continue;
                }
            };
            if latest_attempt.attemptIndex != task_state.attempt_index {
                tracing::error!(
                    "task {} was restarted, attempt index {}->{}",
                    task_id,
                    task_state.attempt_index,
                    latest_attempt.attemptIndex
                );
                failed_tasks.push(task_id.clone());
            }
        }
        Ok(failed_tasks)
    }

    /// Ensure that the task group is in RUNNING state and all tasks have hostnames.
    /// It will block until the conditions have been met.
    async fn ensure_task_group_ready(&mut self) -> Result<(), anyhow::Error> {
        if self.task_group_state == HpcTaskGroupState::RUNNING {
            return Ok(());
        }

        loop {
            let (job_version, job_attempt_index, task_group_attempt) =
                self.get_task_group_status().await?;

            // Ensure all tasks are attempted and have hostnames
            // This should always be the case when task group is in running.
            let mut tasks = Vec::new();
            let mut not_ready = false;
            for (task_id, attempts) in task_group_attempt.taskExecutionAttempts.iter() {
                if let Some(attempt) = attempts.last() {
                    if attempt.hostname.is_none() {
                        tracing::warn!("task {} has no hostname", task_id);
                        not_ready = true;
                        break;
                    }
                    tasks.push((task_id.clone(), attempt.clone()));
                } else {
                    tracing::warn!("task {} has no attempts", task_id);
                    not_ready = true;
                    break;
                }
            }
            if not_ready {
                hyperactor::clock::RealClock
                    .sleep(std::time::Duration::from_secs(1))
                    .await;
                continue;
            }

            // Rank tasks
            self.ordered_tasks = self
                .task_ranker
                .rank(tasks)
                .await
                .context("failed to rank tasks")?;
            self.job_version = job_version;
            self.job_attempt_index = job_attempt_index;
            self.task_group_attempt_index = task_group_attempt.attemptIndex;
            self.task_group_state = task_group_attempt.state;
            break;
        }
        Ok(())
    }

    // Given a proc id, return the task id that it is running on.
    fn task_id_for_proc_id(&self, proc_id: &ProcId) -> Option<TaskId> {
        // First get shape of the world it is in
        if let Some(shape) = self.world_shapes.get(proc_id.world_id()) {
            // Then find the task using the offset of the shape
            if let Some(task_id) = self.task_by_offset.get(&shape.slice().offset()) {
                // Then get the task id for the offset
                return Some(task_id.clone());
            }
        }

        None
    }

    fn task_state_for_proc_id(
        &mut self,
        proc_id: &ProcId,
    ) -> Result<&mut TaskRemoteState, anyhow::Error> {
        if let Some(task_id) = self.task_id_for_proc_id(proc_id) {
            if let Some(task_state) = self.task_states.get_mut(&task_id) {
                Ok(task_state)
            } else {
                // Should never happen
                anyhow::bail!(
                    "task state not found for proc id: {}, task id: {}",
                    proc_id,
                    task_id
                );
            }
        } else {
            // Should never happen
            anyhow::bail!("task not found for proc id: {}", proc_id);
        }
    }

    fn add_proc_id_to_task_state(&mut self, proc_id: &ProcId) -> Result<(), anyhow::Error> {
        let task_state = self.task_state_for_proc_id(proc_id)?;
        if !task_state.active_procs.insert(proc_id.clone()) {
            // Should not happen but we can ignore
            tracing::error!("proc id already in task state: {}", proc_id);
        }
        Ok(())
    }

    fn remove_proc_from_to_task_state(&mut self, proc_id: &ProcId) -> Result<(), anyhow::Error> {
        let task_state = self.task_state_for_proc_id(proc_id)?;
        if !task_state.active_procs.remove(proc_id) {
            // Should not happen but we can ignore
            tracing::error!("proc id already in task state: {}", proc_id);
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

    fn get_world_id(task_group_name: String) -> Result<WorldId, anyhow::Error> {
        let re = Regex::new(r"[\[\]\.]").context("failed to build world_id regex")?;
        let safe_name = re.replace_all(&task_group_name, "_");
        safe_name.to_string().parse().context(format!(
            "failed to create world ID from safe name {}",
            safe_name
        ))
    }
}

#[async_trait]
impl Alloc for TaskGroupAlloc {
    async fn next(&mut self) -> Option<ProcState> {
        // outer poll loop
        loop {
            if let state @ Some(_) = self.event_queue.pop_front() {
                break state;
            }

            if !self.running {
                break None;
            }

            if let Err(e) = self.ensure_task_group_ready().await {
                tracing::error!("failed to ensure task group ready: {}", e);
                break None;
            }
            if let Err(e) = self.ensure_started().await {
                tracing::error!("failed to ensure started: {}", e);
                break None;
            }
            match self.check_task_group_health().await {
                Ok(ref failed_tasks) => {
                    if !failed_tasks.is_empty() {
                        // Task group failed. All active procs are stopped now.
                        for task_id in failed_tasks {
                            let task_state = match self.task_states.get_mut(task_id) {
                                Some(task_state) => task_state,
                                None => {
                                    // Should never happen
                                    tracing::error!("Unable to find failed task {} state", task_id);
                                    continue;
                                }
                            };
                            tracing::warn!("task {} failed, stopping all its procs", task_id);
                            for proc_id in task_state.active_procs.iter() {
                                self.event_queue
                                    .push_back(ProcState::Stopped(proc_id.clone()));
                            }
                            task_state.active_procs.clear();
                        }
                        // Remove the failed tasks
                        self.task_states
                            .retain(|task_id, _| !failed_tasks.contains(task_id));
                        // If no more healthy tasks, stop the alloc
                        if self.task_states.is_empty() {
                            tracing::error!("no more healthy tasks, stopping alloc");
                            self.running = false;
                        }
                        continue;
                    }
                }
                Err(e) => {
                    tracing::error!("failed check task group health: {}", e);
                    break None;
                }
            }

            let update = loop {
                tokio::select! {
                    msg = self.rx.recv() => {
                        tracing::debug!("received message: {:?}", msg);
                        match msg {
                            Ok(RemoteProcessProcStateMessage::Allocated{world_id, shape}) => {
                                tracing::info!("received allocated world id: {}", world_id);
                                self.world_shapes.insert(world_id, shape);
                            }
                            Ok(RemoteProcessProcStateMessage::Update(proc_state)) => {
                                match proc_state {
                                    ProcState::Created { ref proc_id, .. } => {
                                        if let Err(e) = self.add_proc_id_to_task_state(proc_id) {
                                            tracing::error!("failed to add proc id to task state: {}", e);
                                        }
                                    }
                                    ProcState::Stopped(ref proc_id) => {
                                        if let Err(e) = self.remove_proc_from_to_task_state(proc_id) {
                                            tracing::error!("failed to remove proc id from task state: {}", e);
                                        }
                                    }
                                    _ => {}
                                }
                                break Some(proc_state);
                            }
                            Ok(RemoteProcessProcStateMessage::Done(world_id)) => {
                                tracing::info!("allocator world_id: {} is done", world_id);
                                if self.world_shapes.remove(&world_id).is_none() {
                                    tracing::error!("received done for unknown world id: {}", world_id);
                                } else if self.world_shapes.is_empty() {
                                    self.running = false;
                                    break None;
                                }
                            }
                            Err(e) => {
                                tracing::error!("error receiving events: {}", e);
                                // We've lost our main listening channel. No fixing. Block and let
                                // caller timeout and recycle us.
                                hyperactor::clock::RealClock.sleep(std::time::Duration::from_secs(1)).await;
                            }
                        }
                    }
                    _ = clock::RealClock.sleep(self.task_group_getter.lock().await.refresh_interval()) => {
                        // poll now
                        break None;
                    }
                }
            };
            let update = match update {
                Some(update) => update,
                None => continue,
            };

            if let ProcState::Created { proc_id, coords } = update {
                match self.project_proc_into_global_shape(&proc_id, &coords) {
                    Ok(global_coords) => {
                        tracing::info!("reprojected coords: {:?} -> {:?}", coords, global_coords);
                        break Some(ProcState::Created {
                            proc_id,
                            coords: global_coords,
                        });
                    }
                    Err(e) => {
                        tracing::error!("failed to project coords for proc: {}: {}", proc_id, e);
                        break None;
                    }
                }
            } else {
                break Some(update);
            }
        }
    }

    fn shape(&self) -> &Shape {
        &self.spec.shape
    }

    fn world_id(&self) -> &WorldId {
        &self.world_id
    }

    fn transport(&self) -> ChannelTransport {
        ChannelTransport::MetaTls
    }

    async fn stop(&mut self) -> Result<(), AllocatorError> {
        tracing::info!("stopping alloc");

        for (task_id, task_state) in self.task_states.iter_mut() {
            tracing::debug!("stopping alloc at task {}", task_id);
            if let Err(e) = task_state
                .tx
                .send(RemoteProcessAllocatorMessage::Stop)
                .await
            {
                tracing::error!("failed to send stop message to {}: {}", task_id, e);
            }
        }

        Ok(())
    }
}

/// Configurations for the MAST allocator.
pub struct MastAllocatorConfig {
    /// MAST job name. If not specified it will be obtained from environment variables.
    pub job_name: Option<String>,
    /// Transport to communicate to workers. Defaults to `MetaTls``.
    pub transport: ChannelTransport,
    /// Port to use for remote allocator. Defaults to `DEFAULT_REMOTE_ALLOCATOR_PORT`.
    pub remote_allocator_port: u16,
}

impl Default for MastAllocatorConfig {
    fn default() -> Self {
        Self {
            job_name: None,
            transport: ChannelTransport::MetaTls,
            remote_allocator_port: DEFAULT_REMOTE_ALLOCATOR_PORT,
        }
    }
}

/// An Allocator implementation to be used inside MAST jobs.
pub struct MastAllocator {
    client: Arc<dyn HpcSchedulerReadOnlyService + Send + Sync>,
    job_name: String,
    config: MastAllocatorConfig,
}

impl MastAllocator {
    /// Creates a new Allocator for the current MAST job.
    pub async fn new(config: MastAllocatorConfig) -> Result<Self, anyhow::Error> {
        let client = hpcscheduler_srclients::make_HpcSchedulerService_srclient!(
            fbinit::expect_init(),
            tiername = "mast.api.read"
        )?;
        Self::new_with_client(client, config).await
    }

    /// Creates a new Allocator with a specific MAST API client. Used for testing.
    pub async fn new_with_client(
        client: Arc<dyn HpcSchedulerReadOnlyService + Send + Sync>,
        config: MastAllocatorConfig,
    ) -> Result<Self, anyhow::Error> {
        let job_name = match config.job_name {
            Some(ref job_name) => job_name.clone(),
            None => std::env::var("MAST_HPC_JOB_NAME")
                .context("failed to read job name from environment variable MAST_HPC_JOB_NAME")?,
        };
        Ok(Self {
            client,
            job_name,
            config,
        })
    }
}

#[async_trait]
impl Allocator for MastAllocator {
    type Alloc = TaskGroupAlloc;

    /// Creates a new Alloc for the given spec. The spec must contain the following labels:
    /// * ALLOC_LABEL_TASK_GROUP: Name of job task group to use for the allocation.
    ///
    /// Tasks in the allocation will be ranked according to the following:
    /// * If TORCH_ELASTIC_CUSTOM_HOSTNAMES_LIST_FILE is set, use TorchElasticTaskRanker
    ///   which uses xlformer compatible ranking.
    /// * Otherwise, use TWTaskRanker which uses TW task ID to rank tasks.
    async fn allocate(&mut self, spec: AllocSpec) -> Result<Self::Alloc, AllocatorError> {
        let task_group_name = match spec.constraints.match_labels.get(ALLOC_LABEL_TASK_GROUP) {
            Some(task_group_name) => task_group_name.clone(),
            None => {
                return Err(AllocatorError::Other(anyhow::anyhow!(
                    "expected to find task group name label {} in constraints",
                    ALLOC_LABEL_TASK_GROUP
                )));
            }
        };

        tracing::info!(
            "request to allocate task group: {}, shape: {}",
            task_group_name,
            spec.shape
        );

        // Some sanity checks
        let num_dims = spec.shape.slice().num_dim();
        if num_dims <= 1 {
            return Err(AllocatorError::Other(anyhow::anyhow!(
                "expected at least 2 labels in shape: {}",
                spec.shape.labels().join(", ")
            )));
        }
        // Number of hosts is the product of all dimensions except the last one.
        let num_hosts = spec.shape.slice().sizes()[0..num_dims - 1]
            .iter()
            .cloned()
            .reduce(|acc, e| acc * e)
            .unwrap();
        let procs_per_host = spec.shape.slice().sizes()[num_dims - 1];

        let status_getter: Arc<Mutex<dyn TaskGroupGetter + Sync + Send>> =
            Arc::new(Mutex::new(TaskGroupGetterImpl::new(
                self.client.clone(),
                self.job_name.clone(),
                task_group_name.clone(),
            )));

        let (_, _, task_group_attempt) = status_getter.lock().await.get_task_group_status().await?;
        if num_hosts > task_group_attempt.numTasks as usize {
            // Represent available capacity back to Shape form to compare with requested shape.
            let capacity_dims = spec
                .shape
                .coordinates(task_group_attempt.numTasks as usize * procs_per_host)
                .unwrap()
                .iter()
                .cloned()
                .map(|(_, dim)| dim)
                .collect::<Vec<_>>();
            let capacity_shape = Shape::new(
                Vec::from(spec.shape.labels()),
                Slice::new_row_major(capacity_dims),
            )
            .unwrap();
            return Err(AllocatorError::NotEnoughResources {
                requested: spec.shape.clone(),
                available: capacity_shape,
            });
        }

        // TODO: Replicate rank assignment logic here to make sure that we match what torch.dist
        //       is expecting when reading TORCH_ELASTIC_CUSTOM_HOSTNAMES_LIST_FILE, which is
        //       done on the worker side.
        //       In the future, we should move away from this and virtualize the ranks.
        let ranker: Box<dyn TaskRanker + Sync + Send> =
            if std::env::var("TORCH_ELASTIC_CUSTOM_HOSTNAMES_LIST_FILE").is_ok() {
                let config_dir = std::env::var("STATIC_CONFIG_DIR").context(
                    "unable to read environment variable STATIC_CONFIG_DIR for static host configs",
                )?;
                Box::new(
                    TorchElasticTaskRanker::new(config_dir)
                        .await
                        .context("failed to create torchx ranker")?,
                )
            } else {
                Box::new(TWTaskRanker {})
            };

        TaskGroupAlloc::new(
            status_getter,
            ranker,
            task_group_name.clone(),
            spec,
            self.config.remote_allocator_port,
            self.config.transport.clone(),
        )
        .await
        .map_err(AllocatorError::Other)
    }
}

#[automock]
#[async_trait]
trait TaskGroupGetter {
    async fn get_task_group_status(
        &mut self,
    ) -> Result<(i32, i32, hpcscheduler::HpcTaskGroupExecutionAttempt), anyhow::Error>;
    fn refresh_interval(&self) -> Duration;
}

struct TaskGroupGetterImpl {
    client: Arc<dyn HpcSchedulerReadOnlyService + Send + Sync>,
    job_name: String,
    task_group_name: String,
    job_version: i32,
    job_attempt_index: i32,
    task_group_attempt: Option<HpcTaskGroupExecutionAttempt>,
    last_update_time: std::time::Instant,
    refresh_interval: std::time::Duration,
}

impl TaskGroupGetterImpl {
    fn new(
        client: Arc<dyn HpcSchedulerReadOnlyService + Send + Sync>,
        job_name: String,
        task_group_name: String,
    ) -> Self {
        Self::new_with_options(
            client,
            job_name,
            task_group_name,
            DEFAULT_TASK_GROUP_REFRESH_INTERVAL,
        )
    }

    fn new_with_options(
        client: Arc<dyn HpcSchedulerReadOnlyService + Send + Sync>,
        job_name: String,
        task_group_name: String,
        refresh_interval: std::time::Duration,
    ) -> Self {
        Self {
            client,
            job_name,
            task_group_name,
            job_version: 0,
            job_attempt_index: 0,
            task_group_attempt: None,
            last_update_time: std::time::Instant::now()
                .checked_sub(refresh_interval)
                .unwrap(),
            refresh_interval,
        }
    }

    async fn refresh_task_group_status(&mut self) -> Result<(), anyhow::Error> {
        let request = hpcscheduler::GetHpcJobStatusRequest {
            hpcJobName: self.job_name.clone(),
            ..Default::default()
        };
        let response = self.client.getHpcJobStatus(&request).await?;
        self.job_version = response.version;
        self.job_attempt_index = response.latestAttempt.attemptIndex;

        let execution_attempts = match response
            .latestAttempt
            .taskGroupExecutionAttempts
            .get(self.task_group_name.as_str())
        {
            Some(attempts) => attempts,
            None => {
                self.task_group_attempt = None;
                anyhow::bail!(
                    "task group {} not found in job {}",
                    self.task_group_name,
                    self.job_name
                );
            }
        };
        match execution_attempts.last() {
            Some(attempt) => {
                self.task_group_attempt = Some(attempt.clone());
            }
            None => {
                self.task_group_attempt = None;
                anyhow::bail!("no task group attempts found for {}", self.task_group_name);
            }
        }

        self.last_update_time = std::time::Instant::now();

        Ok(())
    }
}

#[async_trait]
impl TaskGroupGetter for TaskGroupGetterImpl {
    async fn get_task_group_status(
        &mut self,
    ) -> Result<(i32, i32, hpcscheduler::HpcTaskGroupExecutionAttempt), anyhow::Error> {
        if self.last_update_time.elapsed() > self.refresh_interval {
            self.refresh_task_group_status().await?;
        }

        if let Some(attempt) = self.task_group_attempt.as_ref() {
            return Ok((self.job_version, self.job_attempt_index, attempt.clone()));
        } else {
            anyhow::bail!(
                "task group {} status not found in job {}",
                self.task_group_name,
                self.job_name
            );
        }
    }

    fn refresh_interval(&self) -> Duration {
        self.refresh_interval
    }
}

struct TWTaskRanker {}

#[async_trait]
impl TaskRanker for TWTaskRanker {
    async fn rank(
        &self,
        tasks: Vec<(TaskId, HpcTaskExecutionAttempt)>,
    ) -> Result<Vec<(TaskId, HpcTaskExecutionAttempt)>, anyhow::Error> {
        let mut ordered_tasks = tasks.clone();
        ordered_tasks.sort_by_key(|(task_id, _)| task_id.clone());
        Ok(tasks)
    }
}

struct TorchElasticTaskRanker {
    entries: HashMap<String, Vec<String>>,
}

impl TorchElasticTaskRanker {
    pub async fn new(static_config_dir: String) -> Result<Self, anyhow::Error> {
        let hostname = hostname::get()
            .ok()
            .and_then(|hostname| hostname.into_string().ok())
            .context("failed to retrieve hostname")?;
        Self::new_with_hostname(static_config_dir, hostname).await
    }

    /// TODO: Using hostname to obtain region is not the right way and will break for
    ///       multi-region task groups.
    pub async fn new_with_hostname(
        static_config_dir: String,
        hostname: String,
    ) -> Result<Self, anyhow::Error> {
        // see: https://github.com/fairinternal/xlformers/blob/llama4_monarch/tools/launching/torchx/entrypoint/rank_assignment.sh
        let pat = Regex::new(r"\.([a-z]{3})[0-9]").context("failed to create region regex")?;
        let region = match pat.captures(&hostname) {
            Some(captures) => {
                if let Some(region) = captures.get(1) {
                    region.as_str().to_string()
                } else {
                    anyhow::bail!("failed to find region in hostname {}", hostname);
                }
            }
            None => anyhow::bail!("failed to find region in hostname {}", hostname),
        };

        let input_hosts_file = format!("{}/{}.jsonl", static_config_dir, region);
        tracing::info!(
            "using region: {}, from hostname: {}, input file: {} for task ranking",
            region,
            hostname,
            input_hosts_file.clone()
        );

        let entries = Self::load(input_hosts_file.clone()).await?;
        Ok(Self { entries })
    }

    async fn load(input_hosts_file: String) -> Result<HashMap<String, Vec<String>>, anyhow::Error> {
        #[derive(Debug, Deserialize)]
        struct HostEntry {
            name: Option<String>,
            datacenter: Option<String>,
            backend_topology: Option<String>,
        }

        let mut entries = HashMap::new();
        let file = tokio::fs::File::open(input_hosts_file.clone())
            .await
            .context(format!(
                "failed to open static hosts file {}",
                input_hosts_file
            ))?;
        let mut reader = BufReader::new(file).lines();
        while let Some(line) = reader.next_line().await? {
            let entry: HostEntry = serde_json::from_str(&line)?;
            let name = match entry.name {
                Some(name) => name,
                None => continue,
            };
            let name = name.trim_end_matches(".facebook.com").to_string();
            if let Some(topology) = entry.backend_topology {
                let topology = topology.split("/").map(|s| s.to_string()).collect();
                entries.insert(name, topology);
            } else if let Some(datacenter) = entry.datacenter {
                entries.insert(
                    name,
                    vec![datacenter, "".to_string(), "".to_string(), "".to_string()],
                );
            }
        }

        Ok(entries)
    }
}

#[async_trait]
impl TaskRanker for TorchElasticTaskRanker {
    async fn rank(
        &self,
        tasks: Vec<(TaskId, HpcTaskExecutionAttempt)>,
    ) -> Result<Vec<(TaskId, HpcTaskExecutionAttempt)>, anyhow::Error> {
        let mut tasks_with_keys = Vec::new();
        for (task_id, task) in tasks {
            let hostname = match task.hostname {
                Some(ref hostname) => hostname.clone(),
                None => anyhow::bail!("expected to find hostname for task {}", task_id),
            };
            let hostname = hostname.trim_end_matches(".facebook.com").to_string();
            let key = match self.entries.get(&hostname) {
                Some(key) => (
                    key[0].to_string(),
                    key[1].to_string(),
                    key[2].to_string(),
                    key[3].to_string(),
                    hostname,
                ),
                None => anyhow::bail!(
                    "expected to find key for task: {}, hostname: {}",
                    task_id,
                    hostname
                ),
            };
            tasks_with_keys.push((task_id, task, key));
        }

        tasks_with_keys.sort_by_key(|(_, _, key)| key.clone());

        Ok(tasks_with_keys
            .into_iter()
            .map(|(task_id, task, _)| (task_id, task))
            .collect())
    }
}

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;
    use std::collections::HashSet;
    use std::sync::Arc;

    use hyperactor_mesh::alloc::AllocConstraints;
    use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAllocator;
    use ndslice::shape;
    use timed_test::async_timed_test;
    use tokio::process::Command;

    use super::*;
    use crate::alloc::MockTaskGroupGetter;

    #[tokio::test]
    async fn test_get_task_group_status() {
        let job_name = "test_job".to_string();
        let task_group_name = "test_task_group".to_string();
        let refresh_interval = std::time::Duration::from_millis(100);
        let client = Arc::new(hpcscheduler_srclients::make_HpcSchedulerReadOnlyService_mock());

        let mut getter = TaskGroupGetterImpl::new_with_options(
            client.clone(),
            job_name.clone(),
            task_group_name.clone(),
            refresh_interval,
        );

        let mut response = hpcscheduler::GetHpcJobStatusResponse {
            version: 1,
            latestAttempt: hpcscheduler::HpcJobExecutionAttempt {
                attemptIndex: 2,
                taskGroupExecutionAttempts: BTreeMap::from([(
                    task_group_name.clone(),
                    vec![HpcTaskGroupExecutionAttempt {
                        attemptIndex: 3,
                        state: hpcscheduler::HpcTaskGroupState::RUNNING,
                        ..Default::default()
                    }],
                )]),
                ..Default::default()
            },
            ..Default::default()
        };
        client.getHpcJobStatus.ret(response.clone());
        let (job_version, job_attempt_index, task_group_attempt) =
            getter.get_task_group_status().await.unwrap();
        assert_eq!(job_version, 1);
        assert_eq!(job_attempt_index, 2);
        assert_eq!(task_group_attempt.attemptIndex, 3);

        // Cached. Should no update.
        response.version = 2;
        client.getHpcJobStatus.ret(response.clone());
        let (job_version, _, _) = getter.get_task_group_status().await.unwrap();
        assert_eq!(job_version, 1);

        // Sleep to trigger update.
        hyperactor::clock::RealClock.sleep(refresh_interval).await;
        let (job_version, _, _) = getter.get_task_group_status().await.unwrap();
        assert_eq!(job_version, 2);

        // Failures should reset the state. Simulate with empty taskGroupExecutionAttempts.
        hyperactor::clock::RealClock.sleep(refresh_interval).await;
        response.latestAttempt.taskGroupExecutionAttempts = BTreeMap::new();
        client.getHpcJobStatus.ret(response.clone());
        let r = getter.get_task_group_status().await;
        assert!(r.is_err());
    }

    #[async_timed_test(timeout_secs = 30)]
    async fn test_alloc() {
        // Temporary hack until we fix T222132226.
        std::env::set_var("MONARCH_MESSAGE_DELIVERY_TIMEOUT_SECS", "1");

        let remote_port = 26600;
        let task_group_name = "test_task_group".to_string();
        let task_ranker = Box::new(TWTaskRanker {});
        let task_group_getter = Arc::new(Mutex::new(MockTaskGroupGetter::new()));
        let spec = AllocSpec {
            shape: shape!(host = 2, gpu = 4),
            constraints: AllocConstraints::none(),
        };

        let task1_allocator = RemoteProcessAllocator::new();
        let task1_addr = ChannelAddr::any(ChannelTransport::Unix);
        let task1_cmd =
            Command::new(buck_resources::get("monarch/hyperactor_meta/bootstrap").unwrap());
        let task2_allocator = RemoteProcessAllocator::new();
        let task2_addr = ChannelAddr::any(ChannelTransport::Unix);
        let task2_cmd =
            Command::new(buck_resources::get("monarch/hyperactor_meta/bootstrap").unwrap());

        let task_group_attempt = HpcTaskGroupExecutionAttempt {
            attemptIndex: 3,
            state: HpcTaskGroupState::RUNNING,
            taskExecutionAttempts: BTreeMap::from([
                (
                    "task2".to_string(),
                    vec![HpcTaskExecutionAttempt {
                        hostname: Some(task2_addr.to_string()),
                        ..Default::default()
                    }],
                ),
                (
                    "task1".to_string(),
                    vec![HpcTaskExecutionAttempt {
                        hostname: Some(task1_addr.to_string()),
                        ..Default::default()
                    }],
                ),
            ]),
            ..Default::default()
        };

        tokio::spawn(async move {
            task1_allocator.start(task1_cmd, task1_addr).await.unwrap();
        });
        tokio::spawn(async move {
            task2_allocator.start(task2_cmd, task2_addr).await.unwrap();
        });

        task_group_getter
            .lock()
            .await
            .expect_get_task_group_status()
            .returning(move || Ok((1, 2, task_group_attempt.clone())));
        task_group_getter
            .lock()
            .await
            .expect_refresh_interval()
            .return_const(Duration::from_millis(100));

        let mut alloc = TaskGroupAlloc::new(
            task_group_getter,
            task_ranker,
            task_group_name,
            spec.clone(),
            remote_port,
            ChannelTransport::Unix,
        )
        .await
        .unwrap();

        // Created + Running
        let mut procs = HashSet::new();
        let mut started_procs = HashSet::new();
        let mut proc_coords = HashSet::new();
        for _ in 0..spec.shape.slice().len() * 2 {
            let proc_state = alloc.next().await.unwrap();
            tracing::info!("test received next proc_state: {:?}", proc_state);
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
        tokio::select! {
            _ = hyperactor::clock::RealClock
            .sleep(std::time::Duration::from_millis(500)) => {},
            _ = alloc.next() => panic!("expected no more items"),
        }

        // stop the allocation
        alloc.stop().await.unwrap();
        for _ in 0..spec.shape.slice().len() {
            let proc_state = alloc.next().await.unwrap();
            tracing::info!("test received next proc_state: {:?}", proc_state);
            match proc_state {
                ProcState::Stopped(proc_id) => {
                    assert!(started_procs.remove(&proc_id));
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
    }

    #[async_timed_test(timeout_secs = 30)]
    async fn test_alloc_single_task_group_failure() {
        // Temporary hack until we fix T222132226.
        std::env::set_var("MONARCH_MESSAGE_DELIVERY_TIMEOUT_SECS", "1");

        let remote_port = 26600;
        let task_group_name = "test_task_group".to_string();
        let task_ranker = Box::new(TWTaskRanker {});
        let task_group_getter = Arc::new(Mutex::new(MockTaskGroupGetter::new()));
        task_group_getter
            .lock()
            .await
            .expect_refresh_interval()
            .return_const(Duration::from_millis(100));
        let spec = AllocSpec {
            shape: shape!(host = 2, gpu = 4),
            constraints: AllocConstraints::none(),
        };

        let task1_allocator = RemoteProcessAllocator::new();
        let task1_addr = ChannelAddr::any(ChannelTransport::Unix);
        let task1_cmd =
            Command::new(buck_resources::get("monarch/hyperactor_meta/bootstrap").unwrap());
        let task2_allocator = RemoteProcessAllocator::new();
        let task2_addr = ChannelAddr::any(ChannelTransport::Unix);
        let task2_cmd =
            Command::new(buck_resources::get("monarch/hyperactor_meta/bootstrap").unwrap());

        let task_group_attempt = HpcTaskGroupExecutionAttempt {
            attemptIndex: 3,
            state: HpcTaskGroupState::RUNNING,
            taskExecutionAttempts: BTreeMap::from([
                (
                    "task2".to_string(),
                    vec![HpcTaskExecutionAttempt {
                        hostname: Some(task2_addr.to_string()),
                        ..Default::default()
                    }],
                ),
                (
                    "task1".to_string(),
                    vec![HpcTaskExecutionAttempt {
                        hostname: Some(task1_addr.to_string()),
                        ..Default::default()
                    }],
                ),
            ]),
            ..Default::default()
        };
        let task_group_attempt3 = task_group_attempt.clone();

        tokio::spawn(async move {
            task1_allocator.start(task1_cmd, task1_addr).await.unwrap();
        });
        tokio::spawn(async move {
            task2_allocator.start(task2_cmd, task2_addr).await.unwrap();
        });

        task_group_getter
            .lock()
            .await
            .expect_get_task_group_status()
            .returning(move || Ok((1, 2, task_group_attempt3.clone())));

        let mut alloc = TaskGroupAlloc::new(
            task_group_getter.clone(),
            task_ranker,
            task_group_name,
            spec.clone(),
            remote_port,
            ChannelTransport::Unix,
        )
        .await
        .unwrap();

        // Created + Running
        let mut started_procs = HashSet::new();
        for _ in 0..spec.shape.slice().len() * 2 {
            let proc_state = alloc.next().await.unwrap();
            tracing::info!("test received next proc_state: {:?}", proc_state);
            match proc_state {
                ProcState::Created { .. } => {}
                ProcState::Running { proc_id, .. } => {
                    started_procs.insert(proc_id);
                }
                _ => panic!("expected Created or Running"),
            }
        }

        tracing::info!("test simulating group restart");
        task_group_getter.lock().await.checkpoint();
        // Simulate task group restart
        let mut task_group_attempt_4 = task_group_attempt.clone();
        task_group_attempt_4.attemptIndex = 4;
        task_group_getter
            .lock()
            .await
            .expect_get_task_group_status()
            .returning(move || Ok((1, 2, task_group_attempt_4.clone())));

        // Should get stop event for all procs
        for _ in 0..spec.shape.slice().len() {
            let proc_state = alloc.next().await.unwrap();
            tracing::info!("test received next proc_state: {:?}", proc_state);
            match proc_state {
                ProcState::Stopped(proc_id) => {
                    assert!(started_procs.remove(&proc_id));
                }
                _ => panic!("expected Created or Running"),
            }
        }
        assert!(started_procs.is_empty());
        // one last none
        assert!(alloc.next().await.is_none());
    }

    #[async_timed_test(timeout_secs = 30)]
    async fn test_alloc_single_task_failure() {
        // Temporary hack until we fix T222132226.
        std::env::set_var("MONARCH_MESSAGE_DELIVERY_TIMEOUT_SECS", "1");

        let remote_port = 26600;
        let task_group_name = "test_task_group".to_string();
        let task_ranker = Box::new(TWTaskRanker {});
        let task_group_getter = Arc::new(Mutex::new(MockTaskGroupGetter::new()));
        task_group_getter
            .lock()
            .await
            .expect_refresh_interval()
            .return_const(Duration::from_millis(100));
        let spec = AllocSpec {
            shape: shape!(host = 2, gpu = 4),
            constraints: AllocConstraints::none(),
        };

        let task1_allocator = RemoteProcessAllocator::new();
        let task1_addr = ChannelAddr::any(ChannelTransport::Unix);
        let task1_cmd =
            Command::new(buck_resources::get("monarch/hyperactor_meta/bootstrap").unwrap());
        let task2_allocator = RemoteProcessAllocator::new();
        let task2_addr = ChannelAddr::any(ChannelTransport::Unix);
        let task2_cmd =
            Command::new(buck_resources::get("monarch/hyperactor_meta/bootstrap").unwrap());

        let task_group_attempt = HpcTaskGroupExecutionAttempt {
            attemptIndex: 3,
            state: HpcTaskGroupState::RUNNING,
            taskExecutionAttempts: BTreeMap::from([
                (
                    "task2".to_string(),
                    vec![HpcTaskExecutionAttempt {
                        hostname: Some(task2_addr.to_string()),
                        ..Default::default()
                    }],
                ),
                (
                    "task1".to_string(),
                    vec![HpcTaskExecutionAttempt {
                        hostname: Some(task1_addr.to_string()),
                        ..Default::default()
                    }],
                ),
            ]),
            ..Default::default()
        };
        let task_group_attempt3 = task_group_attempt.clone();

        tokio::spawn(async move {
            task1_allocator.start(task1_cmd, task1_addr).await.unwrap();
        });
        tokio::spawn(async move {
            task2_allocator.start(task2_cmd, task2_addr).await.unwrap();
        });

        task_group_getter
            .lock()
            .await
            .expect_get_task_group_status()
            .returning(move || Ok((1, 2, task_group_attempt3.clone())));

        let mut alloc = TaskGroupAlloc::new(
            task_group_getter.clone(),
            task_ranker,
            task_group_name,
            spec.clone(),
            remote_port,
            ChannelTransport::Unix,
        )
        .await
        .unwrap();

        // Created + Running
        let mut started_procs = HashSet::new();
        let mut proc_coords = HashMap::new();
        for _ in 0..spec.shape.slice().len() * 2 {
            let proc_state = alloc.next().await.unwrap();
            tracing::info!("test received next proc_state: {:?}", proc_state);
            match proc_state {
                ProcState::Created { proc_id, coords } => {
                    proc_coords.insert(proc_id, coords);
                }
                ProcState::Running { proc_id, .. } => {
                    started_procs.insert(proc_id);
                }
                _ => panic!("expected Created or Running"),
            }
        }

        tracing::info!("test simulating task restart");
        task_group_getter.lock().await.checkpoint();
        task_group_getter
            .lock()
            .await
            .expect_refresh_interval()
            .return_const(Duration::from_millis(100));
        // Simulate task group restart
        let mut task_group_attempt_4 = task_group_attempt.clone();
        task_group_attempt_4
            .taskExecutionAttempts
            .get_mut("task2")
            .unwrap()
            .get_mut(0)
            .unwrap()
            .attemptIndex = 4;
        task_group_getter
            .lock()
            .await
            .expect_get_task_group_status()
            .returning(move || Ok((1, 2, task_group_attempt_4.clone())));

        // Should get stop event procs in task2
        for _ in 0..spec.shape.slice().len() / 2 {
            let proc_state = alloc.next().await.unwrap();
            tracing::info!("test received next proc_state: {:?}", proc_state);
            match proc_state {
                ProcState::Stopped(proc_id) => {
                    assert!(started_procs.remove(&proc_id));
                    assert!(proc_coords.remove(&proc_id).is_some());
                }
                _ => panic!("expected Created or Running"),
            }
        }
        assert!(started_procs.len() == spec.shape.slice().len() / 2);
        // remaining procs should all be in task1
        for (_, coords) in proc_coords {
            assert!(coords[0] == 0)
        }
        // ensure no more pending items
        tokio::select! {
            _ = hyperactor::clock::RealClock
            .sleep(std::time::Duration::from_millis(500)) => {},
            _ = alloc.next() => panic!("expected no more items"),
        }
    }

    #[tokio::test]
    async fn test_allocator_errors() {
        let client = Arc::new(hpcscheduler_srclients::make_HpcSchedulerReadOnlyService_mock());
        let response = hpcscheduler::GetHpcJobStatusResponse {
            version: 1,
            latestAttempt: hpcscheduler::HpcJobExecutionAttempt {
                attemptIndex: 2,
                taskGroupExecutionAttempts: BTreeMap::from([(
                    "test_group".to_string(),
                    vec![HpcTaskGroupExecutionAttempt {
                        attemptIndex: 1,
                        numTasks: 4,
                        ..Default::default()
                    }],
                )]),
                ..Default::default()
            },
            ..Default::default()
        };
        client.getHpcJobStatus.ret(response.clone());

        let config = MastAllocatorConfig {
            job_name: Some("test".to_string()),
            ..Default::default()
        };
        let mut allocator = MastAllocator::new_with_client(client, config)
            .await
            .unwrap();
        // Test just capacity
        allocator
            .allocate(AllocSpec {
                shape: shape!(host = 4, gpu = 8),
                constraints: AllocConstraints {
                    match_labels: HashMap::from([(
                        ALLOC_LABEL_TASK_GROUP.to_string(),
                        "test_group".to_string(),
                    )]),
                },
            })
            .await
            .unwrap();
        // Test over capacity
        assert!(
            allocator
                .allocate(AllocSpec {
                    shape: shape!(host = 5, gpu = 8),
                    constraints: AllocConstraints {
                        match_labels: HashMap::from([(
                            ALLOC_LABEL_TASK_GROUP.to_string(),
                            "test_group".to_string(),
                        )]),
                    },
                })
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_torchx_ranker() {
        let region = "gtn1";

        let temp_dir = tempfile::tempdir().unwrap();
        let region_file = format!(
            "{}/{}.jsonl",
            temp_dir.path().to_str().unwrap(),
            &region[0..3]
        );
        tokio::fs::write(region_file, _TEST_STATIC_HOSTS_DATA)
            .await
            .unwrap();
        let tasks = vec![
            (
                "task4".to_string(),
                HpcTaskExecutionAttempt {
                    hostname: Some("twshared9322.03.gtn1.facebook.com".to_string()),
                    ..Default::default()
                },
            ),
            (
                "task3".to_string(),
                HpcTaskExecutionAttempt {
                    hostname: Some("twshared9303.03.gtn1.facebook.com".to_string()),
                    ..Default::default()
                },
            ),
            (
                "task2".to_string(),
                HpcTaskExecutionAttempt {
                    hostname: Some("twshared12034.03.gtn1.facebook.com".to_string()),
                    ..Default::default()
                },
            ),
            (
                "task1".to_string(),
                HpcTaskExecutionAttempt {
                    hostname: Some("twshared8678.03.gtn1.facebook.com".to_string()),
                    ..Default::default()
                },
            ),
        ];

        let ranker = TorchElasticTaskRanker::new_with_hostname(
            temp_dir.path().to_str().unwrap().to_string(),
            format!("devgpu004.{}", region),
        )
        .await
        .unwrap();
        let ranked = ranker.rank(tasks).await.unwrap();
        assert_eq!(ranked[0].0, "task1");
        assert_eq!(ranked[1].0, "task2");
        assert_eq!(ranked[2].0, "task3");
        assert_eq!(ranked[3].0, "task4");
    }

    static _TEST_STATIC_HOSTS_DATA: &str = r#"{"id": 322274647, "name": "twshared8701.03.gtn1.facebook.com", "datacenter": "gtn1", "backend_topology": "/gtn1.1C//rtsw044.c083.f00.gtn1", "nics": [{"type": "ETH0", "ipv6_addr": "2401:db00:161c:3904:face:0000:0013:0000"}, {"type": "ETH1", "ipv6_addr": "2401:db00:161c:3904:face:0000:005d:0000"}, {"type": "ETH2", "ipv6_addr": "2401:db00:161c:3904:face:0000:0082:0000"}, {"type": "ETH3", "ipv6_addr": "2401:db00:161c:3904:face:0000:00a7:0000"}, {"type": "SVC0", "ipv6_addr": "2803:6082:60e4:1013:0000:0000:0000:0001"}, {"type": "SVC0_1", "ipv6_addr": "2803:6082:60e4:105d:0000:0000:0000:0001"}, {"type": "SVC0_2", "ipv6_addr": "2803:6082:60e4:1082:0000:0000:0000:0001"}, {"type": "SVC0_3", "ipv6_addr": "2803:6082:60e4:10a7:0000:0000:0000:0001"}, {"type": "OOB", "ipv6_addr": "2401:db00:161c:3904:face:0000:01f4:0000"}, {"type": "BETH0", "ipv6_addr": "2401:db00:161b:8564:bace:0000:00cc:0000"}, {"type": "BETH1", "ipv6_addr": "2401:db00:161b:8567:bace:0000:00f1:0000"}, {"type": "BETH2", "ipv6_addr": "2401:db00:161b:8560:bace:0000:0116:0000"}, {"type": "BETH3", "ipv6_addr": "2401:db00:161b:8566:bace:0000:013b:0000"}, {"type": "BETH4", "ipv6_addr": "2401:db00:161b:8565:bace:0000:0160:0000"}, {"type": "BETH5", "ipv6_addr": "2401:db00:161b:8563:bace:0000:0185:0000"}, {"type": "BETH6", "ipv6_addr": "2401:db00:161b:8562:bace:0000:01aa:0000"}, {"type": "BETH7", "ipv6_addr": "2401:db00:161b:8561:bace:0000:01cf:0000"}]}
{"id": 322274655, "name": "twshared12034.03.gtn1.facebook.com", "datacenter": "gtn1", "backend_topology": "/gtn1.1C//rtsw044.c083.f00.gtn1", "nics": [{"type": "ETH0", "ipv6_addr": "2401:db00:161c:3904:face:0000:01e9:0000"}, {"type": "ETH1", "ipv6_addr": "2401:db00:161c:3904:face:0000:0233:0000"}, {"type": "ETH2", "ipv6_addr": "2401:db00:161c:3904:face:0000:027d:0000"}, {"type": "ETH3", "ipv6_addr": "2401:db00:161c:3904:face:0000:02a2:0000"}, {"type": "SVC0", "ipv6_addr": "2803:6082:60e4:11e9:0000:0000:0000:0001"}, {"type": "SVC0_1", "ipv6_addr": "2803:6082:60e4:1233:0000:0000:0000:0001"}, {"type": "SVC0_2", "ipv6_addr": "2803:6082:60e4:127d:0000:0000:0000:0001"}, {"type": "SVC0_3", "ipv6_addr": "2803:6082:60e4:12a2:0000:0000:0000:0001"}, {"type": "OOB", "ipv6_addr": "2401:db00:161c:3904:face:0000:03ef:0000"}, {"type": "BETH0", "ipv6_addr": "2401:db00:161b:856a:bace:0000:02c7:0000"}, {"type": "BETH1", "ipv6_addr": "2401:db00:161b:8569:bace:0000:02ec:0000"}, {"type": "BETH2", "ipv6_addr": "2401:db00:161b:856e:bace:0000:0311:0000"}, {"type": "BETH3", "ipv6_addr": "2401:db00:161b:8568:bace:0000:0336:0000"}, {"type": "BETH4", "ipv6_addr": "2401:db00:161b:856b:bace:0000:035b:0000"}, {"type": "BETH5", "ipv6_addr": "2401:db00:161b:856d:bace:0000:0380:0000"}, {"type": "BETH6", "ipv6_addr": "2401:db00:161b:856c:bace:0000:03a5:0000"}, {"type": "BETH7", "ipv6_addr": "2401:db00:161b:856f:bace:0000:03ca:0000"}]}
{"id": 322277418, "name": "twshared8689.03.gtn1.facebook.com", "datacenter": "gtn1", "backend_topology": "/gtn1.1C//rtsw043.c083.f00.gtn1", "nics": [{"type": "ETH0", "ipv6_addr": "2401:db00:161c:3902:face:0000:00f1:0000"}, {"type": "ETH1", "ipv6_addr": "2401:db00:161c:3902:face:0000:0116:0000"}, {"type": "ETH2", "ipv6_addr": "2401:db00:161c:3902:face:0000:013b:0000"}, {"type": "ETH3", "ipv6_addr": "2401:db00:161c:3902:face:0000:0160:0000"}, {"type": "SVC0", "ipv6_addr": "2803:6082:60e4:08f1:0000:0000:0000:0001"}, {"type": "SVC0_1", "ipv6_addr": "2803:6082:60e4:0916:0000:0000:0000:0001"}, {"type": "SVC0_2", "ipv6_addr": "2803:6082:60e4:093b:0000:0000:0000:0001"}, {"type": "SVC0_3", "ipv6_addr": "2803:6082:60e4:0960:0000:0000:0000:0001"}, {"type": "OOB", "ipv6_addr": "2401:db00:161c:3902:face:0000:02d2:0000"}, {"type": "BETH0", "ipv6_addr": "2401:db00:161b:8544:bace:0000:0185:0000"}, {"type": "BETH1", "ipv6_addr": "2401:db00:161b:8547:bace:0000:01aa:0000"}, {"type": "BETH2", "ipv6_addr": "2401:db00:161b:8540:bace:0000:01cf:0000"}, {"type": "BETH3", "ipv6_addr": "2401:db00:161b:8546:bace:0000:01f4:0000"}, {"type": "BETH4", "ipv6_addr": "2401:db00:161b:8545:bace:0000:0219:0000"}, {"type": "BETH5", "ipv6_addr": "2401:db00:161b:8543:bace:0000:023e:0000"}, {"type": "BETH6", "ipv6_addr": "2401:db00:161b:8542:bace:0000:0263:0000"}, {"type": "BETH7", "ipv6_addr": "2401:db00:161b:8541:bace:0000:0288:0000"}]}
{"id": 322277671, "name": "twshared8678.03.gtn1.facebook.com", "datacenter": "gtn1", "backend_topology": "/gtn1.1C//rtsw043.c083.f00.gtn1", "nics": [{"type": "ETH0", "ipv6_addr": "2401:db00:161c:3902:face:0000:0283:0000"}, {"type": "ETH1", "ipv6_addr": "2401:db00:161c:3902:face:0000:02a8:0000"}, {"type": "ETH2", "ipv6_addr": "2401:db00:161c:3902:face:0000:02cd:0000"}, {"type": "ETH3", "ipv6_addr": "2401:db00:161c:3902:face:0000:02f2:0000"}, {"type": "SVC0", "ipv6_addr": "2803:6082:60e4:0a83:0000:0000:0000:0001"}, {"type": "SVC0_1", "ipv6_addr": "2803:6082:60e4:0aa8:0000:0000:0000:0001"}, {"type": "SVC0_2", "ipv6_addr": "2803:6082:60e4:0acd:0000:0000:0000:0001"}, {"type": "SVC0_3", "ipv6_addr": "2803:6082:60e4:0af2:0000:0000:0000:0001"}, {"type": "OOB", "ipv6_addr": "2401:db00:161c:3902:face:0000:0040:0000"}, {"type": "BETH0", "ipv6_addr": "2401:db00:161b:854a:bace:0000:0317:0000"}, {"type": "BETH1", "ipv6_addr": "2401:db00:161b:8549:bace:0000:033c:0000"}, {"type": "BETH2", "ipv6_addr": "2401:db00:161b:854e:bace:0000:0361:0000"}, {"type": "BETH3", "ipv6_addr": "2401:db00:161b:8548:bace:0000:0386:0000"}, {"type": "BETH4", "ipv6_addr": "2401:db00:161b:854b:bace:0000:03ab:0000"}, {"type": "BETH5", "ipv6_addr": "2401:db00:161b:854d:bace:0000:03d0:0000"}, {"type": "BETH6", "ipv6_addr": "2401:db00:161b:854c:bace:0000:03f5:0000"}, {"type": "BETH7", "ipv6_addr": "2401:db00:161b:854f:bace:0000:001b:0000"}]}
{"id": 322324190, "name": "twshared9322.03.gtn1.facebook.com", "datacenter": "gtn1", "backend_topology": "/gtn1.1C//rtsw045.c083.f00.gtn1", "nics": [{"type": "ETH0", "ipv6_addr": "2401:db00:161c:3b00:face:0000:032b:0000"}, {"type": "ETH1", "ipv6_addr": "2401:db00:161c:3b00:face:0000:0350:0000"}, {"type": "ETH2", "ipv6_addr": "2401:db00:161c:3b00:face:0000:0375:0000"}, {"type": "ETH3", "ipv6_addr": "2401:db00:161c:3b00:face:0000:039a:0000"}, {"type": "SVC0", "ipv6_addr": "2803:6082:60ec:032b:0000:0000:0000:0001"}, {"type": "SVC0_1", "ipv6_addr": "2803:6082:60ec:0350:0000:0000:0000:0001"}, {"type": "SVC0_2", "ipv6_addr": "2803:6082:60ec:0375:0000:0000:0000:0001"}, {"type": "SVC0_3", "ipv6_addr": "2803:6082:60ec:039a:0000:0000:0000:0001"}, {"type": "OOB", "ipv6_addr": "2401:db00:161c:3b00:face:0000:00e8:0000"}, {"type": "BETH0", "ipv6_addr": "2401:db00:161b:8584:bace:0000:03bf:0000"}, {"type": "BETH1", "ipv6_addr": "2401:db00:161b:8587:bace:0000:03e4:0000"}, {"type": "BETH2", "ipv6_addr": "2401:db00:161b:8580:bace:0000:000a:0000"}, {"type": "BETH3", "ipv6_addr": "2401:db00:161b:8586:bace:0000:002f:0000"}, {"type": "BETH4", "ipv6_addr": "2401:db00:161b:8585:bace:0000:0054:0000"}, {"type": "BETH5", "ipv6_addr": "2401:db00:161b:8583:bace:0000:0079:0000"}, {"type": "BETH6", "ipv6_addr": "2401:db00:161b:8582:bace:0000:009e:0000"}, {"type": "BETH7", "ipv6_addr": "2401:db00:161b:8581:bace:0000:00c3:0000"}]}
{"id": 322324437, "name": "twshared9303.03.gtn1.facebook.com", "datacenter": "gtn1", "backend_topology": "/gtn1.1C//rtsw045.c083.f00.gtn1", "nics": [{"type": "ETH0", "ipv6_addr": "2401:db00:161c:3b00:face:0000:008f:0000"}, {"type": "ETH1", "ipv6_addr": "2401:db00:161c:3b00:face:0000:00b4:0000"}, {"type": "ETH2", "ipv6_addr": "2401:db00:161c:3b00:face:0000:00d9:0000"}, {"type": "ETH3", "ipv6_addr": "2401:db00:161c:3b00:face:0000:00fe:0000"}, {"type": "SVC0", "ipv6_addr": "2803:6082:60ec:008f:0000:0000:0000:0001"}, {"type": "SVC0_1", "ipv6_addr": "2803:6082:60ec:00b4:0000:0000:0000:0001"}, {"type": "SVC0_2", "ipv6_addr": "2803:6082:60ec:00d9:0000:0000:0000:0001"}, {"type": "SVC0_3", "ipv6_addr": "2803:6082:60ec:00fe:0000:0000:0000:0001"}, {"type": "OOB", "ipv6_addr": "2401:db00:161c:3b00:face:0000:034e:0000"}, {"type": "BETH0", "ipv6_addr": "2401:db00:161b:858a:bace:0000:0123:0000"}, {"type": "BETH1", "ipv6_addr": "2401:db00:161b:8589:bace:0000:016d:0000"}, {"type": "BETH2", "ipv6_addr": "2401:db00:161b:858e:bace:0000:0192:0000"}, {"type": "BETH3", "ipv6_addr": "2401:db00:161b:8588:bace:0000:01dc:0000"}, {"type": "BETH4", "ipv6_addr": "2401:db00:161b:858b:bace:0000:0226:0000"}, {"type": "BETH5", "ipv6_addr": "2401:db00:161b:858d:bace:0000:0270:0000"}, {"type": "BETH6", "ipv6_addr": "2401:db00:161b:858c:bace:0000:02ba:0000"}, {"type": "BETH7", "ipv6_addr": "2401:db00:161b:858f:bace:0000:0304:0000"}]}"#;
}
