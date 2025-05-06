//! Multiprocess actor system and support.

#![feature(assert_matches)]
#![feature(never_type)]
#![deny(missing_docs)]

/// TODO: add missing doc.
pub mod ping_pong;
pub mod proc_actor;
/// TODO: add missing doc.
pub mod scheduler;
/// TODO: add missing doc.
pub mod supervision;
/// TODO: add missing doc.
pub mod system;
pub mod system_actor;

/// py-spy wrapper.
pub mod pyspy;

pub use hyperactor::actor;
pub use system::System;
