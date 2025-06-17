/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use enum_as_inner::EnumAsInner;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor::OncePortRef;
use hyperactor::RefClient;
use hyperactor::channel::ChannelAddr;
use hyperactor::reference::ActorId;
use hyperactor::reference::Index;
use hyperactor::reference::ProcId;
use hyperactor::reference::WorldId;
use serde::Deserialize;
use serde::Serialize;

/// Supervision message used to collect supervision state of a world.
#[derive(
    Handler,
    HandleClient,
    RefClient,
    Serialize,
    Deserialize,
    Debug,
    Clone,
    PartialEq,
    EnumAsInner,
    Named
)]
pub enum WorldSupervisionMessage {
    /// Request supervision state of a world. The reply will be sent back via
    /// the once port ref in the message. None result indicates the world isn't
    /// managed by the system.
    State(WorldId, #[reply] OncePortRef<Option<WorldSupervisionState>>),
}

/// The supervision state of a world. It contains the supervision state of
/// all procs in the world.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Named)]
pub struct WorldSupervisionState {
    /// A map from proc id to proc supervision state.
    pub procs: HashMap<Index, ProcSupervisionState>,
}

impl WorldSupervisionState {
    /// Return whether this world is healthy, world is healthy if all its procs are healthy.
    pub fn is_healthy(&self) -> bool {
        self.procs.values().all(ProcSupervisionState::is_healthy)
    }
}

/// Message to communicate proc supervision state.
#[derive(
    Handler,
    HandleClient,
    RefClient,
    Serialize,
    Deserialize,
    Debug,
    Clone,
    PartialEq,
    EnumAsInner,
    Named
)]
pub enum ProcSupervisionMessage {
    /// Update proc supervision state. The reply will be sent back via the once
    /// port ref in the message to indicate whether the message receiver is
    /// healthy or not.
    Update(ProcSupervisionState, #[reply] OncePortRef<()>),
}

/// The health of a proc.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Named)]
pub enum ProcStatus {
    /// No known issues.
    Alive,

    /// The proc hasn't provided any supervision updates in a
    /// reasonable time.
    Expired,

    /// A failure to obtain a TCP/IP connection to the proc was
    /// encountered.
    ConnectionFailure,
}

impl std::fmt::Display for ProcStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let description = match self {
            ProcStatus::Alive => "Alive",
            ProcStatus::Expired => "Expired",
            ProcStatus::ConnectionFailure => "Connection failure",
        };
        write!(f, "{}", description)
    }
}

/// The supervision state of a proc. It contains the supervision state of
/// actors in the proc. This message is used for both supervision update and
/// supervision state query.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Named)]
pub struct ProcSupervisionState {
    /// The world to which this proc belongs.
    pub world_id: WorldId,
    /// The proc id.
    pub proc_id: ProcId,
    /// Address of the proc.
    pub proc_addr: ChannelAddr,
    /// The proc health.
    pub proc_health: ProcStatus,
    /// Contains the supervision state of (failed) actors in the proc.
    /// Actors can appear more than once here if they have multiple failures
    pub failed_actors: Vec<(ActorId, hyperactor::actor::ActorStatus)>,
}

impl ProcSupervisionState {
    /// Returns whether this proc has any failed actors.
    pub fn has_failed_actor(&self) -> bool {
        !self.failed_actors.is_empty()
    }

    /// Return whether this proc is healthy, proc is alive and there is not failed actor.
    pub fn is_healthy(&self) -> bool {
        matches!(self.proc_health, ProcStatus::Alive) && !self.has_failed_actor()
    }
}

hyperactor::alias!(ProcSupervisor, ProcSupervisionMessage); // For proc supervisor to implement (e.g. system actor)
hyperactor::alias!(WorldSupervisor, WorldSupervisionMessage); // For world supervisor to implement (e.g. system actor)
hyperactor::alias!(SupervisionClient, WorldSupervisionState); // For the end receiver of supervision events to implement (e.g. client)
