//! This module defines a proc allocator interface as well as a multi-process
//! (local) allocator, [`ProcessAllocator`].

pub mod local;
pub mod process;
pub mod remoteprocess;

use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use enum_as_inner::EnumAsInner;
use hyperactor::ActorRef;
use hyperactor::ProcId;
use hyperactor::WorldId;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
pub use local::LocalAlloc;
pub use local::LocalAllocator;
use mockall::predicate::*;
use mockall::*;
use ndslice::Shape;
pub use process::ProcessAlloc;
pub use process::ProcessAllocator;
use serde::Deserialize;
use serde::Serialize;

use crate::proc_mesh::mesh_agent::MeshAgent;

/// Errors that occur during allocation operations.
#[derive(Debug, thiserror::Error)]
pub enum AllocatorError {
    #[error("incomplete allocation; expected: {0}")]
    Incomplete(Shape),

    /// The requested shape is too large for the allocator.
    #[error("not enough resources; requested: {requested:?}, available: {available:?}")]
    NotEnoughResources { requested: Shape, available: Shape },

    /// An uncategorized error from an underlying system.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Constraints on the allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocConstraints {
    /// Aribitrary name/value pairs that are interpreted by individual
    /// allocators to control allocation process.
    pub match_labels: HashMap<String, String>,
}

impl AllocConstraints {
    pub fn none() -> Self {
        Self {
            match_labels: HashMap::new(),
        }
    }
}

/// A specification (desired state) of an alloc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocSpec {
    /// The requested shape of the alloc.
    // We currently assume that this shape is dense.
    // This should be validated, or even enforced by
    // way of types.
    pub shape: Shape,
    /// Constraints on the allocation.
    pub constraints: AllocConstraints,
}

/// The core allocator trait, implemented by all allocators.
#[automock(type Alloc=MockAlloc;)]
#[async_trait]
pub trait Allocator {
    /// The type of [`Alloc`] produced by this allocator.
    type Alloc: Alloc;

    /// Create a new allocation. The allocation itself is generally
    /// returned immediately (after validating parameters, etc.);
    /// the caller is expected to respond to allocation events as
    /// the underlying procs are incrementally allocated.
    async fn allocate(&mut self, spec: AllocSpec) -> Result<Self::Alloc, AllocatorError>;
}

/// A proc's status. A proc can only monotonically move from
/// `Created` to `Running` to `Stopped`.
#[derive(Clone, Debug, PartialEq, EnumAsInner, Serialize, Deserialize)]
pub enum ProcState {
    /// A proc was added to the alloc.
    Created {
        /// The proc's id.
        proc_id: ProcId,
        /// Its assigned coordinates (in the alloc's shape).
        coords: Vec<usize>,
    },
    /// A proc was started.
    Running {
        proc_id: ProcId,
        /// Reference to this proc's mesh agent. In the future, we'll reserve a
        /// 'well known' PID (0) for this purpose.
        mesh_agent: ActorRef<MeshAgent>,
        /// The address of this proc. The endpoint of this address is
        /// the proc's mailbox, which accepts [`hyperactor::mailbox::MessageEnvelope`]s.
        addr: ChannelAddr,
    },
    /// A proc was stopped.
    // TODO: add reason
    Stopped(ProcId),
}

impl fmt::Display for ProcState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProcState::Created { proc_id, coords } => {
                write!(
                    f,
                    "{}: created at ({})",
                    proc_id,
                    coords
                        .iter()
                        .map(|c| c.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            ProcState::Running { proc_id, addr, .. } => {
                write!(f, "{}: running at {}", proc_id, addr)
            }
            ProcState::Stopped(proc_id) => {
                write!(f, "{}: stopped", proc_id)
            }
        }
    }
}

/// An alloc is a specific allocation, returned by an [`Allocator`].
#[automock]
#[async_trait]
pub trait Alloc {
    /// Return the next proc event. `None` indicates that there are
    /// no more events, and that the alloc is stopped.
    async fn next(&mut self) -> Option<ProcState>;

    /// The shape of the alloc.
    fn shape(&self) -> &Shape;

    /// The world id of this alloc, uniquely identifying the alloc.
    /// Note: This will be removed in favor of a different naming scheme,
    /// once we exise "worlds" from hyperactor core.
    fn world_id(&self) -> &WorldId;

    /// The channel transport used the procs in this alloc.
    fn transport(&self) -> ChannelTransport;

    /// Stop this alloc, shutting down all of its procs. A clean
    /// shutdown should result in Stop events from all allocs,
    /// followed by the end of the event stream.
    async fn stop(&mut self) -> Result<(), AllocatorError>;
}

#[cfg(test)]
pub(crate) mod testing {
    use std::collections::HashMap;
    use std::collections::HashSet;

    use ndslice::shape;

    use super::*;

    #[macro_export]
    macro_rules! alloc_test_suite {
        ($allocator:expr) => {
            #[tokio::test]
            async fn test_allocator_basic() {
                $crate::alloc::testing::test_allocator_basic($allocator).await;
            }
        };
    }

    pub(crate) async fn test_allocator_basic(mut allocator: impl Allocator) {
        let mut alloc = allocator
            .allocate(AllocSpec {
                shape: shape! { replica = 4 },
                constraints: AllocConstraints::none(),
            })
            .await
            .unwrap();

        // Get everything up into running state. We require that we get
        // procs 0..4.
        let mut procs = HashMap::new();
        let mut running = HashSet::new();
        while running.len() != 4 {
            match alloc.next().await.unwrap() {
                ProcState::Created { proc_id, coords } => {
                    procs.insert(proc_id, coords);
                }
                ProcState::Running { proc_id, .. } => {
                    assert!(procs.contains_key(&proc_id));
                    assert!(!running.contains(&proc_id));
                    running.insert(proc_id);
                }
                event => panic!("unexpected event: {:?}", event),
            }
        }

        // We should have complete coverage of all coordinates.
        let coords: HashSet<_> = procs.values().collect();
        for x in 0..4 {
            assert!(coords.contains(&vec![x]));
        }

        // Every proc should belong to the same "world" (alloc).
        let worlds: HashSet<_> = procs.keys().map(|proc_id| proc_id.world_id()).collect();
        assert_eq!(worlds.len(), 1);

        // Now, stop the alloc and make sure it shuts down cleanly.

        alloc.stop().await.unwrap();
        let mut stopped = HashSet::new();
        while let Some(ProcState::Stopped(proc_id)) = alloc.next().await {
            stopped.insert(proc_id);
        }
        assert!(alloc.next().await.is_none());
        assert_eq!(stopped, running);
    }
}
