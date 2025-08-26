/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Messages used in supervision.

use std::fmt;
use std::fmt::Debug;

use derivative::Derivative;
use serde::Deserialize;
use serde::Serialize;

use crate as hyperactor; // for macros
use crate::Named;
use crate::actor::ActorStatus;
use crate::attrs::Attrs;
use crate::reference::ActorId;

/// This is the local actor supervision event. Child actor will propagate this event to its parent.
#[derive(Clone, Debug, Derivative, Serialize, Deserialize, Named)]
#[derivative(PartialEq, Eq)]
pub struct ActorSupervisionEvent {
    /// The actor id of the child actor where the event is triggered.
    pub actor_id: ActorId,
    /// Status of the child actor.
    pub actor_status: ActorStatus,
    /// If this event is associated with a message, the message headers.
    #[derivative(PartialEq = "ignore")]
    pub message_headers: Option<Attrs>,
    /// Optional supervision event that caused this event, for recursive propagation.
    pub caused_by: Option<Box<ActorSupervisionEvent>>,
}

impl ActorSupervisionEvent {
    /// Compute an actor status from this event, ensuring that "caused-by"
    /// events are included in failure states. This should be used as the
    /// actor status when reporting events to users.
    pub fn status(&self) -> ActorStatus {
        match &self.actor_status {
            ActorStatus::Failed(msg) => ActorStatus::Failed(format!("{}: {}", self, msg)),
            status => status.clone(),
        }
    }
}

impl std::error::Error for ActorSupervisionEvent {}

impl fmt::Display for ActorSupervisionEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.actor_id, self.actor_status)?;
        if let Some(message_headers) = &self.message_headers {
            let headers = serde_json::to_string(&message_headers)
                .expect("could not serialize message headers");
            write!(f, " (headers: {})", headers)?;
        }
        if let Some(caused_by) = &self.caused_by {
            write!(f, ": caused by: {})", caused_by)?;
        }
        Ok(())
    }
}
