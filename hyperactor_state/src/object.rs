/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor::Named;
use hyperactor::data::Serialized;
use serde::Deserialize;
use serde::Serialize;

use crate::log_writer::OutputTarget;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Name {
    // Stdout log with hostname and process id
    StdoutLog((String, u32)),
    // Stderr log with hostname and process id
    StderrLog((String, u32)),
}

impl From<OutputTarget> for Name {
    fn from(target: OutputTarget) -> Self {
        let hostname = hostname::get()
            .unwrap_or_else(|_| "unknown_host".into())
            .into_string()
            .unwrap_or("unknown_host".to_string());
        let pid = std::process::id();

        match target {
            OutputTarget::Stdout => Name::StdoutLog((hostname, pid)),
            OutputTarget::Stderr => Name::StderrLog((hostname, pid)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Kind {
    Log,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateMetadata {
    /// Name of the state object.
    pub name: Name,
    /// Kind of the object.
    pub kind: Kind,
}

#[derive(Debug, Serialize, Deserialize, Named)]
pub struct StateObject<S, T> {
    pub metadata: StateMetadata,
    pub spec: S,
    pub state: T,
}

impl<S, T> StateObject<S, T> {
    #[allow(dead_code)]
    pub fn new(metadata: StateMetadata, spec: S, state: T) -> Self {
        Self {
            metadata,
            spec,
            state,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Named)]
pub struct LogSpec;

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, Named)]
pub struct LogState {
    /// A monotonically increasing sequence number.
    seq: u64,
    /// The message in the log as serialized data.
    pub message: Serialized,
}

impl LogState {
    #[allow(dead_code)]
    pub fn new(seq: u64, message: Serialized) -> Self {
        Self { seq, message }
    }

    /// Create a new LogState from a string message
    #[allow(dead_code)]
    pub fn from_string(seq: u64, message: String) -> Result<Self, anyhow::Error> {
        Ok(Self {
            seq,
            message: Serialized::serialize_anon(&message)?,
        })
    }
}

/// A generic state object which is the partially serialized version of a
/// [`StateObject`]. Since [`StateObject`] takes generic types, those type information
/// can be retrieved from the metadata to deserialize [`GenericStateObject`] into
/// a [`StateObject<S, T>`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Named)]
pub struct GenericStateObject {
    metadata: StateMetadata,
    data: Serialized,
}

impl<S, T> TryFrom<StateObject<S, T>> for GenericStateObject
where
    S: Spec,
    T: State,
    StateObject<S, T>: Named,
{
    type Error = anyhow::Error;

    fn try_from(obj: StateObject<S, T>) -> Result<Self, Self::Error> {
        Ok(Self {
            metadata: obj.metadata.clone(),
            data: Serialized::serialize(&obj)?,
        })
    }
}

impl GenericStateObject {
    pub fn metadata(&self) -> &StateMetadata {
        &self.metadata
    }

    pub fn data(&self) -> &Serialized {
        &self.data
    }
}

/// Spec is the define the desired state of an object, defined by the user.
pub trait Spec: Serialize + for<'de> Deserialize<'de> {}
/// State is the current state of an object.
pub trait State: Serialize + for<'de> Deserialize<'de> {}

impl Spec for LogSpec {}
impl State for LogState {}
