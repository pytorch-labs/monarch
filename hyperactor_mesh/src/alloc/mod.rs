//! This module defines a proc allocator interface as well as a multi-process
//! (local) allocator, [`ProcessAllocator`].

pub mod local;
pub(crate) mod logtailer;
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

use crate::alloc::test_utils::MockAllocWrapper;
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
#[automock(type Alloc=MockAllocWrapper;)]
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
    Stopped {
        proc_id: ProcId,
        reason: ProcStopReason,
    },
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
            ProcState::Stopped { proc_id, reason } => {
                write!(f, "{}: stopped: {}", proc_id, reason)
            }
        }
    }
}

/// The reason a proc stopped.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, EnumAsInner)]
pub enum ProcStopReason {
    /// The proc stopped gracefully, e.g., with exit code 0.
    Stopped,
    /// The proc exited with the provided error code.
    Exited(i32),
    /// The proc was killed. The signal number is indicated;
    /// the flags determines whether there was a core dump.
    Killed(i32, bool),
    /// The proc failed to respond to a watchdog request within a timeout.
    Watchdog,
    /// The proc failed for an unknown reason.
    Unknown,
}

impl fmt::Display for ProcStopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stopped => write!(f, "stopped"),
            Self::Exited(code) => write!(f, "exited with code {}", code),
            Self::Killed(signal, dumped) => {
                write!(f, "killed with signal {} (core dumped={})", signal, dumped)
            }
            Self::Watchdog => write!(f, "watchdog failure"),
            Self::Unknown => write!(f, "unknown"),
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

    /// Stop this alloc and wait for all procs to stop. Call will
    /// block until all ProcState events have been drained.
    async fn stop_and_wait(&mut self) -> Result<(), AllocatorError> {
        self.stop().await?;
        while let Some(event) = self.next().await {
            tracing::debug!("drained event: {:?}", event);
        }
        Ok(())
    }
}

pub mod test_utils {
    use tokio::sync::broadcast::Receiver;
    use tokio::sync::broadcast::Sender;

    use super::*;

    /// Test wrapper around MockAlloc to allow us to block next() calls since
    /// mockall doesn't support returning futures.
    pub struct MockAllocWrapper {
        pub alloc: MockAlloc,
        pub block_next_after: usize,
        notify_tx: Sender<()>,
        _notify_rx: Receiver<()>,
        next_unblocked: bool,
    }

    impl MockAllocWrapper {
        pub fn new(alloc: MockAlloc) -> Self {
            Self::new_block_next(alloc, usize::MAX)
        }

        pub fn new_block_next(alloc: MockAlloc, count: usize) -> Self {
            let (tx, rx) = tokio::sync::broadcast::channel(1);
            Self {
                alloc,
                block_next_after: count,
                notify_tx: tx,
                _notify_rx: rx,
                next_unblocked: false,
            }
        }

        pub fn notify_tx(&self) -> Sender<()> {
            self.notify_tx.clone()
        }
    }

    #[async_trait]
    impl Alloc for MockAllocWrapper {
        async fn next(&mut self) -> Option<ProcState> {
            match self.block_next_after {
                0 => {
                    if !self.next_unblocked {
                        // resubscribe everytime as recv() gets cancelled
                        self.notify_tx.subscribe().recv().await.unwrap();
                        self.next_unblocked = true;
                    }
                }
                1.. => {
                    self.block_next_after -= 1;
                }
            }

            self.alloc.next().await
        }

        fn shape(&self) -> &Shape {
            self.alloc.shape()
        }

        fn world_id(&self) -> &WorldId {
            self.alloc.world_id()
        }

        fn transport(&self) -> ChannelTransport {
            self.alloc.transport()
        }

        async fn stop(&mut self) -> Result<(), AllocatorError> {
            self.alloc.stop().await
        }
    }
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
        while let Some(ProcState::Stopped { proc_id, reason }) = alloc.next().await {
            assert_eq!(reason, ProcStopReason::Stopped);
            stopped.insert(proc_id);
        }
        assert!(alloc.next().await.is_none());
        assert_eq!(stopped, running);
    }
}
