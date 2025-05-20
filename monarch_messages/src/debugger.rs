// NOTE: Until https://github.com/PyO3/pyo3/pull/4674, `pyo3::pymethods` trigger
// and unsafe-op-in-unsafe-fn warnings.
#![allow(unsafe_op_in_unsafe_fn)]

use derive_more::From;
use hyperactor::Named;
use pyo3::Bound;
use pyo3::PyResult;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;
use serde::Deserialize;
use serde::Serialize;

pub fn register_python_bindings(debugger: &Bound<'_, PyModule>) -> PyResult<()> {
    debugger.add_class::<DebuggerAction>()?;
    Ok(())
}

/// Enumerates the actions relevant to PDB debugging sessions.
#[derive(Debug, Deserialize, Clone, Serialize, PartialEq)]
#[pyo3::pyclass(frozen, module = "monarch._monarch.debugger")]
pub enum DebuggerAction {
    /// Sent from worker to client to indicate that the worker has entered
    /// a pdb debugging session.
    Paused(),

    /// Sent from client to worker to indicate that the client has started
    /// the debugging session.
    Attach(),

    /// Sent to client or to worker to end the debugging session.
    Detach(),

    /// Sent to client or to worker to write bytes to receiver's stdout.
    Write {
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
    },

    /// Sent from worker to client to read bytes from client's stdin.
    Read { requested_size: usize },
}

#[derive(Serialize, Deserialize, Debug, Clone, Named, From)]
pub enum DebuggerMessage {
    Action { action: DebuggerAction },
}
