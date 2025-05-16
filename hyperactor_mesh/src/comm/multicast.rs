//! The comm actor that provides message casting and result accumulation.

use hyperactor::Named;
use hyperactor::RemoteHandles;
use hyperactor::RemoteMessage;
use hyperactor::actor::RemoteActor;
use hyperactor::data::Serialized;
use hyperactor::message::Castable;
use hyperactor::message::ErasedUnbound;
use hyperactor::message::IndexedErasedUnbound;
use hyperactor::reference::ActorId;
use hyperactor::reference::GangId;
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
    sender: ActorId,
    /// The destination port of the message. It could match multiple actors with
    /// rank wildcard.
    dest_port: DestinationPort,
    /// The serialized message.
    data: ErasedUnbound,
}

impl CastMessageEnvelope {
    /// Create a new CastMessageEnvelope.
    pub fn new<T: Castable + Serialize + Named>(
        sender: ActorId,
        dest_port: DestinationPort,
        message: T,
    ) -> anyhow::Result<Self> {
        let data = ErasedUnbound::try_from_message(message)?;
        Ok(Self {
            sender,
            dest_port,
            data,
        })
    }

    /// Create a new CastMessageEnvelope from serialized data. Only use this
    /// when the message do not contain reply ports. Or it does but you are okay
    /// with the destination actors reply to the client actor directly.
    pub fn from_serialized(sender: ActorId, dest_port: DestinationPort, data: Serialized) -> Self {
        Self {
            sender,
            dest_port,
            data: ErasedUnbound::new(data),
        }
    }

    pub(crate) fn dest_port(&self) -> &DestinationPort {
        &self.dest_port
    }

    pub(crate) fn data(&self) -> &ErasedUnbound {
        &self.data
    }
}

/// Destination port id of a message. It is a `PortId` with the rank masked out,
/// and the messege is always sent to the root actor because only root actor
/// can be accessed externally. The rank is resolved by the destination Selection
/// of the message. We can use `DestinationPort::port_id(rank)` to get the actual
/// `PortId` of the message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Named)]
pub struct DestinationPort {
    /// Destination gang id, consisting of world id and actor name.
    gang_id: GangId,
    /// The port index of the destination actors, it is derived from the
    /// message type and cached here.
    port: u64,
}

impl DestinationPort {
    /// Create a new DestinationPort for Actor type A and message type M.
    pub fn new<A, M>(gang_id: GangId) -> Self
    where
        A: RemoteActor + RemoteHandles<IndexedErasedUnbound<M>>,
        M: Castable + RemoteMessage,
    {
        Self {
            gang_id,
            port: IndexedErasedUnbound::<M>::port(),
        }
    }

    /// Get the actual port id of an actor for a rank.
    pub fn port_id(&self, rank: usize) -> PortId {
        PortId(
            ActorId(
                ProcId(self.gang_id.world_id().clone(), rank),
                self.gang_id.name().to_string(),
                // Only root actor can be accessed externally.
                0,
            ),
            self.port,
        )
    }

    /// Get the gang id of the destination actors.
    pub fn gang_id(&self) -> &GangId {
        &self.gang_id
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
    /// The comm actor who originally casted the message.
    pub(crate) sender: ActorId,
    /// The destination of the message.
    pub(crate) dests: Vec<RoutingFrame>,
    /// The sequence number of this message.
    pub(crate) seq: usize,
    /// The sequence number of the previous message receieved.
    pub(crate) last_seq: usize,
    /// The message to distribute.
    pub(crate) message: CastMessageEnvelope,
}
