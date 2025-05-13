use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use hpcscheduler_srclients::thrift::HpcSchedulerErrorCode;
use hpcscheduler_srclients::thrift::HpcTaskGroupState;
use hpcscheduler_srclients::thrift::HpcTaskState;
use hpcscheduler_srclients::thrift::errors::hpc_scheduler_read_only_service;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::AllocatorError;
use hyperactor_mesh::alloc::remoteprocess::RemoteProcessAllocator;
use tokio::sync::Mutex;

use crate::alloc::MastAllocator;
use crate::alloc::MastAllocatorConfig;

struct Task {
    remote_allocator: Arc<RemoteProcessAllocator>,
    remote_allocator_addr: ChannelAddr,
}

struct TaskGroup {
    tasks: HashMap<String, Task>,
}

/// Simple wrapper to expose multiple allocators to multiple task
/// remote allocators.
struct AllocatorWrapper<A: Allocator + Sync + Send> {
    allocator: Arc<Mutex<A>>,
}

#[async_trait]
impl<A: Allocator + Sync + Send> Allocator for AllocatorWrapper<A> {
    type Alloc = A::Alloc;

    async fn allocate(&mut self, spec: AllocSpec) -> Result<Self::Alloc, AllocatorError> {
        self.allocator.lock().await.allocate(spec).await
    }
}

/// Mock MAST implementation for testing purposes. It is able to run
/// all MAST components locally.
pub struct MockMast {
    task_groups: HashMap<String, TaskGroup>,
}

impl MockMast {
    /// Create a new mock MAST instance. Next you need to call `add_task_group()` to
    /// allocate task groups. Finally use `get_mast_allocator()` to obtain an Allocator
    /// to use with Mesh.
    pub fn new() -> Self {
        Self {
            task_groups: HashMap::new(),
        }
    }

    /// Add a new task group to the mock MAST instance. The task group will be
    /// will use given allocator to allocate procs for each of its tasks.
    pub async fn add_task_group<A: Allocator + Sync + Send + 'static>(
        &mut self,
        name: String,
        num_tasks: u64,
        allocator: A,
    ) -> Result<(), anyhow::Error>
    where
        <A as Allocator>::Alloc: Send + Sync,
    {
        let mut tasks = HashMap::new();
        let allocator = Arc::new(Mutex::new(allocator));
        for i in 0..num_tasks {
            let task_id = format!("{name}_{i}");
            let remote_allocator = RemoteProcessAllocator::new();
            let serve_addr = ChannelAddr::any(ChannelTransport::Unix);
            let task = Task {
                remote_allocator: remote_allocator.clone(),
                remote_allocator_addr: serve_addr.clone(),
            };
            let cloned_task_id = task_id.clone();
            let task_allocator = AllocatorWrapper {
                allocator: allocator.clone(),
            };
            tokio::spawn(async move {
                remote_allocator
                    .start_with_allocator(serve_addr, task_allocator)
                    .await
                    .expect("failed to start remote allocator");
                tracing::info!("mock allocator for task {} exited", cloned_task_id);
            });
            tasks.insert(task_id, task);
        }

        self.task_groups.insert(name, TaskGroup { tasks });

        Ok(())
    }

    /// Obtain a MastAllocator that mocks the given MAST job and its task groups.
    pub fn get_mast_allocator(
        &mut self,
        config: MastAllocatorConfig,
    ) -> Result<MastAllocator, anyhow::Error> {
        let job_name = match config.job_name {
            Some(ref job_name) => job_name.clone(),
            None => anyhow::bail!("job name is required"),
        };
        let attempts = self
            .task_groups
            .iter()
            .map(|(task_group_name, task_group)| {
                (
                    task_group_name.clone(),
                    vec![hpcscheduler::HpcTaskGroupExecutionAttempt {
                        state: HpcTaskGroupState::RUNNING,
                        numTasks: task_group.tasks.len() as i32,
                        taskExecutionAttempts: BTreeMap::from_iter(task_group.tasks.iter().map(
                            |(task_id, task)| {
                                (
                                    task_id.clone(),
                                    vec![hpcscheduler::HpcTaskExecutionAttempt {
                                        state: HpcTaskState::RUNNING,
                                        hostname: Some(task.remote_allocator_addr.to_string()),
                                        ..Default::default()
                                    }],
                                )
                            },
                        )),
                        ..Default::default()
                    }],
                )
            })
            .collect::<Vec<_>>();
        let latest_attempt = hpcscheduler::HpcJobExecutionAttempt {
            attemptIndex: 1,
            taskGroupExecutionAttempts: BTreeMap::from_iter(attempts),
            ..Default::default()
        };
        let mock_client = Arc::new(hpcscheduler_srclients::make_HpcSchedulerReadOnlyService_mock());
        mock_client.getHpcJobStatus.mock_result(move |request| {
            if request.hpcJobName != job_name {
                return Err(hpc_scheduler_read_only_service::GetHpcJobStatusError::e(
                    hpcscheduler::HpcSchedulerServiceException {
                        message: format!(
                            "job {} not found, expecting: {}",
                            request.hpcJobName, job_name
                        ),
                        errorCode: HpcSchedulerErrorCode::JOB_NOT_FOUND,
                        ..Default::default()
                    },
                ));
            }
            Ok(hpcscheduler::GetHpcJobStatusResponse {
                hpcJobName: job_name.clone(),
                state: hpcscheduler::HpcJobState::RUNNING,
                latestAttempt: latest_attempt.clone(),
                ..Default::default()
            })
        });

        MastAllocator::new_with_client(
            mock_client,
            MastAllocatorConfig {
                transport: ChannelTransport::Unix,
                ..config
            },
        )
    }

    /// Stop the given task group and destroying the underlying procs.
    pub fn stop_task_group(&mut self, task_group_name: String) -> Result<(), anyhow::Error> {
        let task_group = match self.task_groups.get(&task_group_name) {
            Some(task_group) => task_group,
            None => anyhow::bail!("unknown task group {}", task_group_name),
        };

        for task in task_group.tasks.values() {
            task.remote_allocator.terminate();
        }

        self.task_groups.remove(&task_group_name);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use hyperactor_mesh::ActorMesh;
    use hyperactor_mesh::Mesh;
    use hyperactor_mesh::ProcMesh;
    use hyperactor_mesh::alloc::AllocConstraints;
    use hyperactor_mesh::alloc::Allocator;
    use hyperactor_mesh::alloc::LocalAllocator;
    use hyperactor_mesh::mesh_id;
    use hyperactor_mesh::reference::ActorMeshRef;
    use hyperactor_mesh::selection::Selection;
    use hyperactor_mesh::test_utils::EmptyActor;
    use hyperactor_mesh::test_utils::EmptyMessage;
    use ndslice::shape;

    use super::*;
    use crate::alloc::ALLOC_LABEL_TASK_GROUP;

    #[tokio::test]
    async fn test_mast_allocator() {
        hyperactor_telemetry::initialize_logging();

        let mut mock_mast = MockMast::new();
        mock_mast
            .add_task_group("task_group_1".to_string(), 2, LocalAllocator {})
            .await
            .unwrap();

        let mut allocator = mock_mast
            .get_mast_allocator(MastAllocatorConfig {
                job_name: Some("job_name".to_string()),
                transport: ChannelTransport::Unix,
                ..Default::default()
            })
            .unwrap();

        let alloc = allocator
            .allocate(AllocSpec {
                constraints: AllocConstraints {
                    match_labels: HashMap::from([(
                        ALLOC_LABEL_TASK_GROUP.to_string(),
                        "task_group_1".to_string(),
                    )]),
                },
                shape: shape!(hosts = 2, gpu = 4),
            })
            .await
            .unwrap();
        let proc_mesh = ProcMesh::allocate(alloc).await.unwrap();
        let actor_mesh: ActorMesh<EmptyActor> = proc_mesh.spawn("test", &()).await.unwrap();

        assert!(actor_mesh.cast(Selection::True, EmptyMessage()).is_ok());
    }
}
