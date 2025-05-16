use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use hyperactor::WorldId;
use lazy_static::lazy_static;
use tokio::sync::Mutex;
use tokio::sync::oneshot;
use tokio::sync::oneshot::Receiver;
use tokio::sync::oneshot::Sender;

use crate::SimulatorError;

lazy_static! {
    /// A handle for SimNet through which you can send and schedule events in the
    /// network.
    static ref COLLECTIVE_COORDINATOR: CollectiveCoorindator = CollectiveCoorindator::new();
}

#[derive(Debug, PartialEq)]
enum MeshState {
    Healthy,
    Unhealthy,
}

#[derive(Debug, PartialEq)]
pub enum CollectiveResult {
    /// Collective has completed successfully
    Done,
    /// One or more peers are unavailable
    PeerUnavailable,
}

#[derive(Debug)]
struct CollectiveCoorindator {
    // TODO(lky): revisit this in the future to support multiple workers in a mesh
    /// A map from worker world id to mesh state
    meshes: Arc<Mutex<HashMap<WorldId, MeshState>>>,
    collective_counter: AtomicUsize,
    /// A flag to synchronize participants to the same phase.
    step: Arc<Mutex<String>>,
    result_senders: Arc<Mutex<Vec<Sender<CollectiveResult>>>>,
}

impl CollectiveCoorindator {
    fn new() -> Self {
        Self {
            meshes: Arc::new(Mutex::new(HashMap::new())),
            collective_counter: AtomicUsize::new(0),
            result_senders: Arc::new(Mutex::new(vec![])),
            step: Arc::new(Mutex::new("".to_string())),
        }
    }

    async fn deactivate_mesh(&self, world_id: WorldId) -> Result<(), SimulatorError> {
        let mut meshes = self.meshes.lock().await;
        let mesh_mut = meshes
            .get_mut(&world_id)
            .ok_or(SimulatorError::MeshNotFound(world_id.to_string()))?;
        *mesh_mut = MeshState::Unhealthy;
        Ok(())
    }

    async fn activate_mesh(&self, world_id: WorldId, step: &str) {
        let mut current_step = self.step.lock().await;
        if *current_step != step {
            *current_step = step.to_string();
            self.meshes.lock().await.clear();
            self.collective_counter.store(0, Ordering::SeqCst);
            self.result_senders.lock().await.clear();
        }
        self.meshes
            .lock()
            .await
            .insert(world_id, MeshState::Healthy);
    }

    /// Run the collective. Once the collective is complete, the result will be sent to the result_tx channel.
    async fn collect(&self) -> Receiver<CollectiveResult> {
        let (result_tx, result_rx) = oneshot::channel::<CollectiveResult>();
        self.collective_counter.fetch_add(1, Ordering::SeqCst);
        self.result_senders.lock().await.push(result_tx);
        // If any of the mesh is unhealthy, we should fail the collective.
        if self
            .meshes
            .lock()
            .await
            .values()
            .any(|mesh| mesh == &MeshState::Unhealthy)
        {
            for result_tx in self.result_senders.lock().await.drain(..) {
                // Fail to send result back should not be reported back to the caller.
                // Since the caller that triggers the send is the last caller.
                if let Err(e) = result_tx.send(CollectiveResult::PeerUnavailable) {
                    tracing::error!("failed to send result back to caller: {:?}", e);
                }
            }
        }
        if self.collective_counter.load(Ordering::SeqCst) == self.meshes.lock().await.len() {
            self.collective_counter.store(0, Ordering::SeqCst);
            for result_tx in self.result_senders.lock().await.drain(..) {
                // Fail to send result back should not be reported back to the caller.
                // Since the caller that triggers the send is the last caller.
                if let Err(e) = result_tx.send(CollectiveResult::Done) {
                    tracing::error!("failed to send result back to caller: {:?}", e);
                }
            }
        }
        result_rx
    }

    async fn is_active(&self, world_id: WorldId) -> bool {
        self.meshes
            .lock()
            .await
            .get(&world_id)
            .unwrap_or(&MeshState::Unhealthy)
            == &MeshState::Healthy
    }
}

pub async fn activate_mesh(world_id: WorldId, step: &str) {
    COLLECTIVE_COORDINATOR.activate_mesh(world_id, step).await;
}

pub async fn collect() -> Receiver<CollectiveResult> {
    COLLECTIVE_COORDINATOR.collect().await
}

pub async fn deactivate_mesh(world_id: WorldId) -> Result<(), SimulatorError> {
    COLLECTIVE_COORDINATOR.deactivate_mesh(world_id).await
}

pub async fn is_active(world_id: WorldId) -> bool {
    COLLECTIVE_COORDINATOR.is_active(world_id).await
}

#[cfg(test)]
mod tests {

    use std::time::Duration;

    use hyperactor::id;
    use tokio::time::timeout;

    use super::*;

    #[tokio::test]
    async fn test_collective_coordinator_success() {
        let world_0 = id!(world_0);
        let world_1 = id!(world_1);

        let collective_coordinator = CollectiveCoorindator::new();
        collective_coordinator.activate_mesh(world_0, "1").await;
        collective_coordinator.activate_mesh(world_1, "1").await;

        let mut result_rx_0 = collective_coordinator.collect().await;

        // Assert that the collective will timeout after 1 second, since world_1 did not call for collect yet.
        assert!(
            timeout(Duration::from_secs(1), &mut result_rx_0)
                .await
                .is_err()
        );

        let result_rx_1 = collective_coordinator.collect().await;

        assert_eq!(result_rx_0.await.unwrap(), CollectiveResult::Done);
        assert_eq!(result_rx_1.await.unwrap(), CollectiveResult::Done);
    }

    #[tokio::test]
    async fn test_collective_coordinator_failure() {
        let world_0 = id!(world_0);
        let world_1 = id!(world_1);

        let collective_coordinator = CollectiveCoorindator::new();
        collective_coordinator.activate_mesh(world_0, "1").await;
        collective_coordinator
            .activate_mesh(world_1.clone(), "1")
            .await;
        collective_coordinator
            .deactivate_mesh(world_1)
            .await
            .unwrap();

        let result_rx_0 = collective_coordinator.collect().await;

        assert_eq!(
            result_rx_0.await.unwrap(),
            CollectiveResult::PeerUnavailable
        );
    }
}
