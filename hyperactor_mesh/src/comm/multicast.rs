//! The comm actor that provides message casting and result accumulation.

use hyperactor::Named;
use hyperactor::data::Serialized;
use hyperactor::reference::ActorId;
use hyperactor::reference::GangId;
use hyperactor::reference::Index;
use hyperactor::reference::PortId;
use hyperactor::reference::ProcId;
use ndslice::Slice;
use ndslice::selection::Selection;
use ndslice::selection::routing::RoutingFrame;
use serde::Deserialize;
use serde::Serialize;

/// A union of slices that can be used to represent arbitrary subset of
/// ranks in a gang. It is represented by a Slice together with a Selection.
/// This is used to define the destination of a cast message or the source of
/// accumulation request.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Uslice {
    /// A slice representing a whole gang.
    pub slice: Slice,
    /// A selection used to represent any subset of the gang.
    pub selection: Selection,
}

/// An envelope that carries a message destined to a group of actors.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Named)]
pub struct CastMessageEnvelope {
    /// The sender of this message.
    pub sender: ActorId,
    /// The destination port of the message. It could match multiple actors with
    /// rank wildcard.
    pub dest_port: DestinationPort,
    /// The serialized message.
    pub data: Serialized,
}

/// Destination port id of a message. It is a `PortId` with the rank masked out.
/// The rank is resolved by the destination Selection of the message. We can use
/// `DestinationPort::port_id(rank)` to get the actual `PortId` of the message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Named)]
pub struct DestinationPort {
    /// Destination gang id, consisting of world id and actor name.
    pub gang_id: GangId,
    /// Index of destination actors in their proc.
    pub actor_idx: Index,
    /// The port index of the destination actors, it is derived from the
    /// message type.
    pub port: u64,
}

impl DestinationPort {
    /// Get the actual port id of an actor for a rank.
    pub fn port_id(&self, rank: usize) -> PortId {
        PortId(
            ActorId(
                ProcId(self.gang_id.world_id().clone(), rank),
                self.gang_id.name().to_string(),
                self.actor_idx.clone(),
            ),
            self.port.clone(),
        )
    }
}

/// The is used to start casting a message to a group of actors.
#[derive(Serialize, Deserialize, Debug, Clone, Named)]
pub struct CastMessage {
    /// The cast destination.
    pub dest: Uslice,
    /// The message to cast.
    pub message: CastMessageEnvelope,
}

/// Forward a message to procs of next hops. This is used by comm actor to
/// forward a message to other comm actors following the selection topology.
/// This message is not visible to the clients.
#[derive(Serialize, Deserialize, Debug, Clone, Named)]
pub(crate) struct ForwardMessage {
    /// The destination of the message.
    pub(crate) dest: RoutingFrame,
    /// The message to distribute.
    pub(crate) message: CastMessageEnvelope,
}
