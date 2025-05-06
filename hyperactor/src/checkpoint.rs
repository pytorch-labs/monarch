//! Checkpoint functionality for various objects to save and load states.

use std::fmt::Debug;

use async_trait::async_trait;

use crate::RemoteMessage;
use crate::mailbox::log::SeqId;

/// Errors that occur during checkpoint operations.
/// This enum is marked non-exhaustive to allow for extensibility.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum CheckpointError {
    /// An error occured during saving checkpoints.
    #[error("save")]
    Save(#[source] anyhow::Error),

    /// An error occured during loading checkpoints.
    #[error("load: {0}")]
    Load(SeqId, #[source] anyhow::Error),
}

/// [`Checkpoint`] is used to save the state of an instance so that it can be restored later.
#[async_trait]
pub trait Checkpointable: Send + Sync + Sized {
    /// The type of the state that is saved. The state can be serialized and deserialized
    /// from persistent storage.
    type State: RemoteMessage;

    /// Saves the current state.
    async fn save(&self) -> Result<Self::State, CheckpointError>;

    /// Loads the a state to restore the instance.
    async fn load(state: Self::State) -> Result<Self, CheckpointError>;
}
