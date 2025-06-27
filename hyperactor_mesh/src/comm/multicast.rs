/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The comm actor that provides message casting and result accumulation.

use hyperactor::Named;
use hyperactor::RemoteHandles;
use hyperactor::RemoteMessage;
use hyperactor::actor::RemoteActor;
use hyperactor::attrs::Attrs;
use hyperactor::data::Serialized;
use hyperactor::declare_attrs;
use hyperactor::message::Castable;
use hyperactor::message::ErasedUnbound;
use hyperactor::message::IndexedErasedUnbound;
use hyperactor::reference::ActorId;
use ndslice::Shape;
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
    /// typehash of the reducer used to accumulate the message in split ports.
    pub reducer_typehash: Option<u64>,
    /// The shape of the cast.
    shape: Shape,
}

impl CastMessageEnvelope {
    /// Create a new CastMessageEnvelope.
    pub fn new<T: Castable + Serialize + Named>(
        sender: ActorId,
        dest_port: DestinationPort,
        shape: Shape,
        message: T,
        reducer_typehash: Option<u64>,
    ) -> Result<Self, anyhow::Error> {
        let data = ErasedUnbound::try_from_message(message)?;
        Ok(Self {
            sender,
            dest_port,
            data,
            reducer_typehash,
            shape,
        })
    }

    /// Create a new CastMessageEnvelope from serialized data. Only use this
    /// when the message do not contain reply ports. Or it does but you are okay
    /// with the destination actors reply to the client actor directly.
    pub fn from_serialized(
        sender: ActorId,
        dest_port: DestinationPort,
        shape: Shape,
        data: Serialized,
    ) -> Self {
        Self {
            sender,
            dest_port,
            data: ErasedUnbound::new(data),
            reducer_typehash: None,
            shape,
        }
    }

    pub(crate) fn dest_port(&self) -> &DestinationPort {
        &self.dest_port
    }

    pub(crate) fn data(&self) -> &ErasedUnbound {
        &self.data
    }

    pub(crate) fn data_mut(&mut self) -> &mut ErasedUnbound {
        &mut self.data
    }

    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }
}

/// Destination port id of a message. It is a `PortId` with the rank masked out,
/// and the messege is always sent to the root actor because only root actor
/// can be accessed externally. The rank is resolved by the destination Selection
/// of the message. We can use `DestinationPort::port_id(rank)` to get the actual
/// `PortId` of the message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Named)]
pub struct DestinationPort {
    /// The actor name to which the message should be delivered.
    actor_name: String,
    /// The port index of the destination actors, it is derived from the
    /// message type and cached here.
    port: u64,
}

impl DestinationPort {
    /// Create a new DestinationPort for a global actor name and message type.
    pub fn new<A, M>(actor_name: String) -> Self
    where
        A: RemoteActor + RemoteHandles<IndexedErasedUnbound<M>>,
        M: Castable + RemoteMessage,
    {
        Self {
            actor_name,
            port: IndexedErasedUnbound::<M>::port(),
        }
    }

    /// The port id of the destination.
    pub fn port(&self) -> u64 {
        self.port
    }

    /// Get the actor name of the destination.
    pub fn actor_name(&self) -> &str {
        &self.actor_name
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

declare_attrs! {
    /// Used inside headers for cast messages to store
    /// the rank of the receiver.
    attr CAST_RANK: usize;
    /// Used inside headers to store the shape of the
    /// actor mesh that a message was cast to.
    attr CAST_SHAPE: Shape;
}

pub fn set_cast_info_on_headers(headers: &mut Attrs, rank: usize, shape: Shape) {
    headers.set(CAST_RANK, rank);
    headers.set(CAST_SHAPE, shape);
}

pub fn get_cast_info_from_headers(headers: &Attrs) -> Option<(usize, Shape)> {
    headers
        .get(CAST_RANK)
        .map(|rank| headers.get(CAST_SHAPE).map(|shape| (*rank, shape.clone())))?
}

pub fn get_cast_info_from_headers_or_err(headers: &Attrs) -> anyhow::Result<(usize, Shape)> {
    let rank = headers
        .get(CAST_RANK)
        .ok_or_else(|| anyhow::anyhow!("{} not found in headers", CAST_RANK.name()))?;
    let shape = headers
        .get(CAST_SHAPE)
        .ok_or_else(|| anyhow::anyhow!("{} not found in headers", CAST_SHAPE.name()))?
        .clone();
    Ok((*rank, shape))
}
