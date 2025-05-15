//! Stub implementation of fbinit for OSS builds
//!
//! This is a minimal implementation that provides the necessary API surface
//! for code that depends on fbinit, but doesn't actually do anything.

/// A stub for the fbinit context
#[derive(Clone, Copy, Debug)]
pub struct FacebookInit;

/// A trait for types that require fbinit
pub trait MainWithFbinit {
    fn init_and_run(self, _fb: FacebookInit) -> i32;
}

/// Initialize the Facebook runtime (stub implementation)
pub fn initialize_with_client_logging(_args: &[&str]) -> FacebookInit {
    FacebookInit
}

/// Initialize the Facebook runtime (stub implementation)
pub fn initialize() -> FacebookInit {
    FacebookInit
}

/// Run a function with fbinit (stub implementation)
pub fn run_with_init<F, R>(f: F) -> R
where
    F: FnOnce(FacebookInit) -> R,
{
    f(FacebookInit)
}
