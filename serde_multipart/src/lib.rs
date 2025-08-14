/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Serde codec for multipart messages.
//!
//! Using [`serialize`] / [`deserialize`], fields typed [`Part`] are extracted
//! from the main payload and appended to a list of `parts`. Each part is backed by
//! [`bytes::Bytes`] for cheap, zero-copy sharing.
//!
//! On decode, the body and its parts are reassembled into the original value
//! without copying.
//!
//! The on-the-wire form is a [`Message`] (body + parts). Your transport sends
//! and receives [`Message`]s; the codec reconstructs the value, enabling
//! efficient network I/O without compacting data into a single buffer.
//!
//! Implementation note: this crate uses Rust's min_specialization feature to enable
//! the use of [`Part`]s with any Serde serializer or deserializer. This feature
//! is fairly restrictive, and thus the API offered by [`serialize`] / [`deserialize`]
//! is not customizable. If customization is needed, you need to add specialization
//! implementations for these codecs. See [`part::PartSerializer`] and [`part::PartDeserializer`]
//! for details.

#![feature(min_specialization)]
#![feature(assert_matches)]

use std::cell::UnsafeCell;
use std::ptr::NonNull;

use bincode::Options;
use bytes::Buf;
use bytes::BufMut;
use bytes::Bytes;
use bytes::buf::UninitSlice;

mod de;
mod part;
mod ser;
use bytes::BytesMut;
use part::Part;

/// A multi-part message, comprising a message body and a list of parts.
pub struct Message {
    body: Bytes,
    parts: Vec<Part>,
}

impl Message {
    /// Returns a new message with the given body and parts.
    pub fn from_body_and_parts(body: Bytes, parts: Vec<Part>) -> Self {
        Self { body, parts }
    }

    /// The body of the message.
    pub fn body(&self) -> &Bytes {
        &self.body
    }

    /// The list of parts of the message.
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// Returns the total number of parts (body + number of parts) in the message.
    pub fn len(&self) -> usize {
        1 + self.parts.len()
    }

    /// Returns whether the message is empty. It is always false, since the body
    /// is always defined.
    pub fn is_empty(&self) -> bool {
        false // there is always a body
    }

    /// Convert this message into its constituent components.
    pub fn into_inner(self) -> (Bytes, Vec<Part>) {
        (self.body, self.parts)
    }
}

/// An unsafe cell of a [`BytesMut`]. This is used to implement an io::Writer
/// for the serializer without exposing lifetime parameters (which cannot be)
/// specialized.
struct UnsafeBufCell {
    buf: UnsafeCell<BytesMut>,
}

impl UnsafeBufCell {
    /// Create a new cell from a [`BytesMut`].
    fn from_bytes_mut(bytes: BytesMut) -> Self {
        Self {
            buf: UnsafeCell::new(bytes),
        }
    }

    /// Convert this cell into its underlying [`BytesMut`].
    fn into_inner(self) -> BytesMut {
        self.buf.into_inner()
    }

    /// Borrow the cell, without lifetime checks. The caller must guarantee that
    /// the returned cell cannot be used after the cell is dropped (usually through
    /// [`UnsafeBufCell::into_inner`]).
    unsafe fn borrow_unchecked(&self) -> UnsafeBufCellRef {
        let ptr =
            // SAFETY: the user is providing the necessary invariants
            unsafe { NonNull::new_unchecked(self.buf.get()) };
        UnsafeBufCellRef { ptr }
    }
}

/// A borrowed reference to an [`UnsafeBufCell`].
struct UnsafeBufCellRef {
    ptr: NonNull<BytesMut>,
}

/// SAFETY: we're extending the implementation of the underlying [`BytesMut`];
/// adding an additional layer of danger by disregarding lifetimes.
unsafe impl BufMut for UnsafeBufCellRef {
    fn remaining_mut(&self) -> usize {
        // SAFETY: extending the implementation of the underlying [`BytesMut`]
        unsafe { self.ptr.as_ref().remaining_mut() }
    }

    unsafe fn advance_mut(&mut self, cnt: usize) {
        // SAFETY: extending the implementation of the underlying [`BytesMut`]
        unsafe { self.ptr.as_mut().advance_mut(cnt) }
    }

    fn chunk_mut(&mut self) -> &mut UninitSlice {
        // SAFETY: extending the implementation of the underlying [`BytesMut`]
        unsafe { self.ptr.as_mut().chunk_mut() }
    }
}

/// Serialize the provided value into a multipart message. The value is encoded using an
/// extended version of [`bincode`] that skips serializing [`Part`]s, which are instead
/// held directly by the returned message.
///
/// Serialize uses the same codec options as [`bincode::serialize`] / [`bincode::deserialize`].
/// These are currently not customizable unless an explicit specialization is also provided.
pub fn serialize<S: ?Sized + serde::Serialize>(value: &S) -> Result<Message, bincode::Error> {
    let buffer = UnsafeBufCell::from_bytes_mut(BytesMut::new());
    // SAFETY: we know here that, once the below "value.serialize()" is done, there are no more
    // extant references to this buffer; we are thus safe to reclaim the buffer into the message
    let buffer_writer = unsafe { buffer.borrow_unchecked() };
    let serializer = bincode::Serializer::new(buffer_writer.writer(), options());
    let mut serializer: part::BincodeSerializer = ser::bincode::Serializer::new(serializer);
    value.serialize(&mut serializer)?;
    Ok(Message {
        body: buffer.into_inner().freeze(),
        parts: serializer.into_parts(),
    })
}

/// Deserialize a message serialized by `[serialize]`, stitching together the original
/// message without copying the underlying buffers.
pub fn deserialize<'a, T>(message: Message) -> Result<T, bincode::Error>
where
    T: serde::Deserialize<'a>,
{
    let (body, parts) = message.into_inner();
    let bincode_deserializer = bincode::Deserializer::with_reader(body.reader(), options());
    let mut deserializer = part::BincodeDeserializer::new(bincode_deserializer, parts.into());
    let value = T::deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(value)
}

/// Construct the set of options used by the specialized serializer and deserializer.
fn options() -> part::BincodeOptionsType {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use serde::Deserialize;
    use serde::Serialize;
    use serde::de::DeserializeOwned;

    use super::*;

    fn test_roundtrip<T>(value: T, expected_parts: usize)
    where
        T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let message = serialize(&value).unwrap();
        assert_eq!(message.len(), expected_parts);
        let deserialized_value = deserialize(message).unwrap();
        assert_eq!(value, deserialized_value);

        // Test normal bincode passthrough:
        let bincode_serialized = bincode::serialize(&value).unwrap();
        let bincode_deserialized = bincode::deserialize(&bincode_serialized).unwrap();
        assert_eq!(value, bincode_deserialized);
    }

    #[test]
    fn test_specialized_serializer_basic() {
        test_roundtrip(Part::from("hello"), 2);
    }

    #[test]
    fn test_specialized_serializer_compound() {
        test_roundtrip(vec![Part::from("hello"), Part::from("world")], 3);
        test_roundtrip((Part::from("hello"), 1, 2, 3, Part::from("world")), 3);
        test_roundtrip(
            {
                #[derive(Serialize, Deserialize, Debug, PartialEq)]
                struct U {
                    parts: Vec<Part>,
                }

                #[derive(Serialize, Deserialize, Debug, PartialEq)]
                struct T {
                    field2: String,
                    field3: Part,
                    field4: Part,
                    field5: Vec<U>,
                }

                T {
                    field2: "hello".to_string(),
                    field3: Part::from("hello"),
                    field4: Part::from("world"),
                    field5: vec![
                        U {
                            parts: vec![Part::from("hello"), Part::from("world")],
                        },
                        U {
                            parts: vec![Part::from("five"), Part::from("six"), Part::from("seven")],
                        },
                    ],
                }
            },
            8,
        );
        test_roundtrip(
            {
                #[derive(Serialize, Deserialize, Debug, PartialEq)]
                struct T {
                    field1: u64,
                    field2: String,
                    field3: Part,
                    field4: Part,
                    field5: u64,
                }
                T {
                    field1: 1,
                    field2: "hello".to_string(),
                    field3: Part::from("hello"),
                    field4: Part::from("world"),
                    field5: 2,
                }
            },
            3,
        );
    }

    #[test]
    fn test_malformed_messages() {
        let message = Message {
            body: Bytes::from_static(b"hello"),
            parts: vec![Part::from("world")],
        };
        let err = deserialize::<String>(message).unwrap_err();

        // Normal bincode errors work:
        assert_matches!(*err, bincode::ErrorKind::Io(err) if err.kind() == std::io::ErrorKind::UnexpectedEof);

        let mut message = serialize(&vec![Part::from("hello"), Part::from("world")]).unwrap();
        message.parts.push(Part::from("foo"));
        let err = deserialize::<Vec<Part>>(message).unwrap_err();
        assert_matches!(*err, bincode::ErrorKind::Custom(message) if message == "multipart overrun while decoding");

        let mut message = serialize(&vec![Part::from("hello"), Part::from("world")]).unwrap();
        let _dropped_message = message.parts.pop().unwrap();
        let err = deserialize::<Vec<Part>>(message).unwrap_err();
        assert_matches!(*err, bincode::ErrorKind::Custom(message) if message == "multipart underrun while decoding");
    }
}
