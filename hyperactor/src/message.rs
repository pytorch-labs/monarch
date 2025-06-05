/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! This module provides a framework for mutating serialized messages without
//! the need to deserialize them. This capability is useful when sending messages
//! to a remote destination throughout intermeidate nodes, where the intermediate
//! nodes do not contain the message's type information.
//!
//! Briefly, it works by following these steps:
//!
//! 1. On the sender side, mutable information is extracted from the typed
//!    message through [Unbind], and stored in a [Bindings] object. This object
//!    is bundled with the serialized message in an [ErasedUnbound] object, which
//!    is sent over the wire.
//! 2. On intermediate nodes, the [ErasedUnbound] object is relayed. The
//!    muation is applied on its bindings field, if needed.
//! 3. One the receiver side, the [ErasedUnbound] object is received as
//!    [IndexedErasedUnbound], where the type information is restored. Mutated
//!    information contained in its bindings field is applied to the message
//!    through [Bind], which results in the final typed message.
//!
//! One main use case of this framework is to mutate the reply ports of a
//! muticast message, so the replies can be relayed through intermediate nodes,
//! rather than directly sent to the original sender. Therefore, a [Castable]
//! trait is defined, which collects requirements for message types using
//! multicast.

use std::collections::HashMap;
use std::marker::PhantomData;

use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate as hyperactor;
use crate::ActorRef;
use crate::Mailbox;
use crate::Named;
use crate::RemoteHandles;
use crate::RemoteMessage;
use crate::actor::RemoteActor;
use crate::data::Serialized;
use crate::intern_typename; // for macros

/// A message `M` that is [`Unbind`] can be converted into an [`Unbound<M>`]
/// containing the message along with a set of extracted parameters that can
/// be independently manipulated, and then later reconstituted (rebound) into
/// an `M`-typed message again.
pub trait Unbind: Sized {
    /// Unbinds the message into an envelope [`Unbound<M>`] containing
    /// the message along with extracted parameters that can are
    /// independently accessible.
    fn unbind(self) -> anyhow::Result<Unbound<Self>> {
        let bindings = self.bindings()?;
        Ok(Unbound {
            message: self,
            bindings,
        })
    }

    /// Get the bindings of this message.
    fn bindings(&self) -> anyhow::Result<Bindings>;
}

/// A message `M` that is [`Bind`] can bind a set of externally provided
/// parameters into the message. It is intended to be used in conjunction
/// with [`Unbind`] to extract portions of a message, manipulate these
/// independently, and then reconstitute the message.
pub trait Bind: Sized {
    /// Update itself with information contained in bindings, and return the
    /// result.
    fn bind(self, bindings: &Bindings) -> anyhow::Result<Self>;
}

/// This trait collects the necessary requirements for messages that are can be
/// cast.
pub trait Castable: RemoteMessage + Bind + Unbind {}
impl<T: RemoteMessage + Bind + Unbind> Castable for T {}

/// Information extracted from a message through [Unbind], which can be merged
/// back to the message through [Bind].
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bindings(HashMap<u64, Vec<Serialized>>);

impl Bindings {
    /// Inserts values into the binding.
    /// If the binding did not have this type present, None is returned.
    /// If the binding already had this type, replace the old values with new
    /// values, and the old value is returned.
    pub fn insert<T: Serialize + Named>(
        &mut self,
        values: impl IntoIterator<Item = &T>,
    ) -> anyhow::Result<Option<Vec<Serialized>>> {
        let ser = values
            .into_iter()
            .map(|v| Serialized::serialize(v))
            .collect::<Result<Vec<Serialized>, _>>()?;
        Ok(self.0.insert(T::typehash(), ser))
    }

    /// Appends an element to the back of its type's corresponding vector in the
    /// binding.
    pub fn push<T: Serialize + Named>(&mut self, value: &T) -> anyhow::Result<()> {
        let ser = Serialized::serialize(value)?;
        self.0.entry(T::typehash()).or_default().push(ser);
        Ok(())
    }

    /// Get this type's values from the binding.
    /// If the binding did not have this type present, empty Vec is returned.
    pub fn get<T: DeserializeOwned + Named>(&self) -> anyhow::Result<Vec<T>> {
        match self.0.get(&T::typehash()) {
            None => Ok(vec![]),
            Some(ser) => {
                let deser = ser
                    .iter()
                    .map(|v| v.deserialized::<T>())
                    .collect::<Result<Vec<T>, _>>()?;
                Ok(deser)
            }
        }
    }

    /// Rebind all values of type `T`. The input iterator must exactly match the
    /// number of `T`-typed values in the binding.
    pub fn rebind<T: DeserializeOwned + Named>(
        &self,
        mut_refs: impl ExactSizeIterator<Item = &mut T>,
    ) -> anyhow::Result<()> {
        let bound_values = self.get::<T>()?;
        anyhow::ensure!(
            bound_values.len() == mut_refs.len(),
            "the length of type {} in binding is {}, which is different from the length of \
            references it binds to {}.",
            T::typename(),
            bound_values.len(),
            mut_refs.len(),
        );

        for (p_ref, p) in mut_refs.zip(bound_values.into_iter()) {
            *p_ref = p;
        }
        Ok(())
    }

    /// Returns true if the binding contains no values of type `T`.
    pub fn is_empty<T: Named>(&self) -> bool {
        self.0.get(&T::typehash()).is_none_or(Vec::is_empty)
    }

    /// Returns the number of values of type `T` in the binding.
    pub fn len<T: Named>(&self) -> usize {
        self.0.get(&T::typehash()).map_or(0, Vec::len)
    }
}

/// An object contains a message, and its bindings extracted through [Unbind].
#[derive(Debug, PartialEq)]
pub struct Unbound<M> {
    message: M,
    bindings: Bindings,
}

impl<M> Unbound<M> {
    /// Build a new object.
    pub fn new(message: M, bindings: Bindings) -> Self {
        Self { message, bindings }
    }
}

impl<M: Bind> Unbound<M> {
    /// Bind its bindings to its message through [Bind], and return the result.
    pub fn bind(self) -> anyhow::Result<M> {
        self.message.bind(&self.bindings)
    }
}

/// Unbound, with its message type M erased through serialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Named)]
pub struct ErasedUnbound {
    message: Serialized,
    bindings: Bindings,
}

impl ErasedUnbound {
    /// Create an object directly from Serialized without binding.
    pub fn new(message: Serialized) -> Self {
        Self {
            message,
            bindings: Bindings::default(),
        }
    }

    /// Create an object from a typed message.
    // Note: cannot implement TryFrom<T> due to conflict with core crate's blanket impl.
    // More can be found in this issue: https://github.com/rust-lang/rust/issues/50133
    pub fn try_from_message<T: Unbind + Serialize + Named>(msg: T) -> Result<Self, anyhow::Error> {
        let unbound = msg.unbind()?;
        let serialized = Serialized::serialize(&unbound.message)?;
        Ok(Self {
            message: serialized,
            bindings: unbound.bindings,
        })
    }

    /// Get ports inside bindings.
    pub fn get<T: DeserializeOwned + Named>(&self) -> anyhow::Result<Vec<T>> {
        self.bindings.get()
    }

    /// Update ports inside bindings.
    pub fn replace<T: Serialize + Named>(
        &mut self,
        new_values: impl ExactSizeIterator<Item = &T>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(self.bindings.len::<T>() == new_values.len());
        self.bindings.insert(new_values)?;
        Ok(())
    }

    /// Replace all `T`-typed values in the binding with `new_value`.
    pub fn maybe_replace<T: Serialize + Named>(&mut self, new_value: &T) -> anyhow::Result<bool> {
        let n = self.bindings.len::<T>();
        if n == 0 {
            return Ok(false);
        }

        self.bindings.insert(vec![new_value; n])?;
        Ok(true)
    }

    fn downcast<M: DeserializeOwned>(self) -> anyhow::Result<Unbound<M>> {
        let message: M = self.message.deserialized()?;
        Ok(Unbound {
            message,
            bindings: self.bindings,
        })
    }
}

/// Type used for indexing an erased unbound.
/// Has the same serialized representation as `ErasedUnbound`.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "ErasedUnbound")]
pub struct IndexedErasedUnbound<M>(ErasedUnbound, PhantomData<M>);

impl<M: DeserializeOwned> IndexedErasedUnbound<M> {
    pub(crate) fn downcast(self) -> anyhow::Result<Unbound<M>> {
        self.0.downcast()
    }
}

impl<M: Bind> IndexedErasedUnbound<M> {
    /// Used in unit tests to bind CastBlobT<M> to the given actor. Do not use in
    /// production.
    pub fn bind_for_test_only<A>(actor_ref: ActorRef<A>, mailbox: &Mailbox) -> anyhow::Result<()>
    where
        A: RemoteActor + RemoteHandles<M> + RemoteHandles<IndexedErasedUnbound<M>>,
        M: RemoteMessage,
    {
        let mailbox_clone = mailbox.clone();
        let port_handle = mailbox.open_enqueue_port::<IndexedErasedUnbound<M>>(move |m| {
            let bound_m = m.downcast()?.bind()?;
            actor_ref.send(&mailbox_clone, bound_m)?;
            Ok(())
        });
        port_handle.bind_to(IndexedErasedUnbound::<M>::port());
        Ok(())
    }
}

impl<M> From<ErasedUnbound> for IndexedErasedUnbound<M> {
    fn from(erased: ErasedUnbound) -> Self {
        Self(erased, PhantomData)
    }
}

impl<M: Named + 'static> Named for IndexedErasedUnbound<M> {
    fn typename() -> &'static str {
        intern_typename!(Self, "hyperactor::message::IndexedErasedUnbound<{}>", M)
    }
}

macro_rules! impl_bind_unbind_basic {
    ($t:ty) => {
        impl Bind for $t {
            fn bind(self, bindings: &Bindings) -> anyhow::Result<Self> {
                anyhow::ensure!(
                    bindings.0.is_empty(),
                    "bindings for {} should be empty, but got {:?}",
                    stringify!($t),
                    bindings
                );
                Ok(self)
            }
        }

        impl Unbind for $t {
            fn bindings(&self) -> anyhow::Result<Bindings> {
                Ok(Bindings::default())
            }
        }
    };
}

impl_bind_unbind_basic!(());
impl_bind_unbind_basic!(bool);
impl_bind_unbind_basic!(i8);
impl_bind_unbind_basic!(u8);
impl_bind_unbind_basic!(i16);
impl_bind_unbind_basic!(u16);
impl_bind_unbind_basic!(i32);
impl_bind_unbind_basic!(u32);
impl_bind_unbind_basic!(i64);
impl_bind_unbind_basic!(u64);
impl_bind_unbind_basic!(i128);
impl_bind_unbind_basic!(u128);
impl_bind_unbind_basic!(isize);
impl_bind_unbind_basic!(usize);
impl_bind_unbind_basic!(String);

#[cfg(test)]
mod tests {
    use hyperactor::PortRef;
    use hyperactor::id;
    use maplit::hashmap;

    use super::*;
    use crate::PortId;

    // Used to demonstrate a user defined reply type.
    #[derive(Debug, PartialEq, Serialize, Deserialize, Named)]
    struct MyReply(String);

    // Used to demonstrate a two-way message type.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Named)]
    struct MyMessage {
        arg0: bool,
        arg1: u32,
        reply0: PortRef<String>,
        reply1: PortRef<MyReply>,
    }

    // TODO(pzhang) add macro to auto-gen this implementation.
    impl Unbind for MyMessage {
        fn bindings(&self) -> anyhow::Result<Bindings> {
            let mut bindings = Bindings::default();
            let ports = [self.reply0.port_id(), self.reply1.port_id()];
            bindings.insert::<PortId>(ports)?;
            Ok(bindings)
        }
    }

    // TODO(pzhang) add macro to auto-gen this implementation.
    impl Bind for MyMessage {
        fn bind(mut self, bindings: &Bindings) -> anyhow::Result<Self> {
            let mut_ports = [self.reply0.port_id_mut(), self.reply1.port_id_mut()];
            bindings.rebind::<PortId>(mut_ports.into_iter())?;
            Ok(self)
        }
    }

    #[test]
    fn test_castable() {
        let original_port0 = PortRef::attest(id!(world[0].actor[0][123]));
        let original_port1 = PortRef::attest(id!(world[1].actor1[0][456]));
        let my_message = MyMessage {
            arg0: true,
            arg1: 42,
            reply0: original_port0.clone(),
            reply1: original_port1.clone(),
        };

        let serialized_my_message = Serialized::serialize(&my_message).unwrap();

        // convert to ErasedUnbound
        let mut erased = ErasedUnbound::try_from_message(my_message.clone()).unwrap();
        assert_eq!(
            erased,
            ErasedUnbound {
                message: serialized_my_message.clone(),
                bindings: Bindings(hashmap! {
                    PortId::typehash() => vec![
                        Serialized::serialize(original_port0.port_id()).unwrap(),
                        Serialized::serialize(original_port1.port_id()).unwrap(),
                    ],
                }),
            }
        );

        // Modify the port in the erased
        let new_port_id0 = id!(world[0].comm[0][680]);
        assert_ne!(&new_port_id0, original_port0.port_id());
        let new_port_id1 = id!(world[1].comm[0][257]);
        assert_ne!(&new_port_id1, original_port1.port_id());

        erased
            .replace::<PortId>(vec![&new_port_id0, &new_port_id1].into_iter())
            .unwrap();
        let new_bindings = Bindings(hashmap! {
            PortId::typehash() => vec![
                Serialized::serialize(&new_port_id0).unwrap(),
                Serialized::serialize(&new_port_id1).unwrap(),
            ],
        });
        assert_eq!(
            erased,
            ErasedUnbound {
                message: serialized_my_message.clone(),
                bindings: new_bindings.clone(),
            }
        );

        // convert back to MyMessage
        let unbound = erased.downcast::<MyMessage>().unwrap();
        assert_eq!(
            unbound,
            Unbound {
                message: my_message,
                bindings: new_bindings,
            }
        );
        let new_my_message = unbound.bind().unwrap();
        assert_eq!(
            new_my_message,
            MyMessage {
                arg0: true,
                arg1: 42,
                reply0: PortRef::attest(new_port_id0),
                reply1: PortRef::attest(new_port_id1),
            }
        );
    }
}
