//! Messages used in supervision.

use std::fmt::Debug;

use serde::Deserialize;
use serde::Serialize;

use crate as hyperactor; // for macros
use crate::Named;
use crate::actor::ActorStatus;
use crate::reference::ActorId;

/// This is the local actor supervision event. Child actor will propagate this event to its parent.
#[derive(Clone, Debug, Serialize, Deserialize, Named)]
pub struct ActorSupervisionEvent {
    /// The actor id of the child actor where the event is triggered.
    actor_id: ActorId,
    /// Status of the child actor.
    actor_status: ActorStatus,
}

impl ActorSupervisionEvent {
    /// Create a new actor supervision event.
    pub fn new(actor_id: ActorId, actor_status: ActorStatus) -> Self {
        Self {
            actor_id,
            actor_status,
        }
    }
    /// Get the actor id of the supervision event.
    pub fn actor_id(&self) -> &ActorId {
        &self.actor_id
    }
    /// Get the actor status of the supervision event.
    pub fn actor_status(&self) -> &ActorStatus {
        &self.actor_status
    }

    /// Consume this event to a tuple.
    pub fn into_inner(self) -> (ActorId, ActorStatus) {
        (self.actor_id, self.actor_status)
    }
}
