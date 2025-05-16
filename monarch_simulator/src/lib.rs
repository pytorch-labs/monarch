use hyperactor::actor::ActorError;
use hyperactor::simnet::SimNetError;

pub mod bootstrap;
mod collective_coordinator;
pub mod controller;
pub mod simulator;
pub mod worker;

/// The type of error that can occur on channel operations.
#[derive(thiserror::Error, Debug)]
pub enum SimulatorError {
    /// Error during simnet operation.
    #[error(transparent)]
    SimNetError(#[from] SimNetError),

    /// Error during actor operations.
    #[error(transparent)]
    ActorError(#[from] ActorError),

    /// Simulator cannot find the world with given name.
    #[error("World {0} not found")]
    WorldNotFound(String),

    /// Cannot find the mesh in simulator.
    #[error("Mesh not found {0}")]
    MeshNotFound(String),
}
