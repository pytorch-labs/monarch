use std::process::Stdio;
use std::str;

use anyhow::Context;
use hyperactor::serde_json;
use py_spy::stack_trace::Frame;
use py_spy::stack_trace::LocalVariable;
use py_spy::stack_trace::StackTrace;
use serde::Deserialize;
use serde::Serialize;
use tokio::process::Command;

/// A full stack trace from PySpy.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct PySpyTrace {
    /// The process id than generated this stack trace.
    pub pid: i32,
    /// The command line used to start the process.
    pub command_line: String,
    /// The stack traces.
    pub stack_traces: Option<Vec<PySpyStackTrace>>,
    /// The error, if any.
    pub error: Option<String>,
}

/// A stack trace from PySpy.
/// Wrapper is needed to have our own derives.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct PySpyStackTrace {
    /// The process id than generated this stack trace
    pub pid: i32,
    /// The python thread id for this stack trace
    pub thread_id: u64,
    /// The python thread name for this stack trace
    pub thread_name: Option<String>,
    /// The OS thread id for this stack tracee
    pub os_thread_id: Option<u64>,
    /// Whether or not the thread was active
    pub active: bool,
    /// Whether or not the thread held the GIL
    pub owns_gil: bool,
    /// The frames
    pub frames: Vec<PySpyFrame>,
}

/// A frame from PySpy.
/// Wrapper is needed to have our own derives.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct PySpyFrame {
    /// The function name
    pub name: String,
    /// The full filename of the file
    pub filename: String,
    /// The module/shared library the
    pub module: Option<String>,
    /// A short, more readable, representation of the filename
    pub short_filename: Option<String>,
    /// The line number inside the file (or 0 for native frames without line information)
    pub line: i32,
    /// Local Variables associated with the frame
    pub locals: Option<Vec<PySpyLocalVariable>>,
    /// If this is an entry frame. Each entry frame corresponds to one native frame.
    pub is_entry: bool,
}

/// A frame local variable.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct PySpyLocalVariable {
    /// Variable name.
    pub name: String,
    /// Variable address.
    pub addr: usize,
    /// Whether or not the variable is an argument.
    pub arg: bool,
    /// Variable representation.
    pub repr: Option<String>,
}

impl From<StackTrace> for PySpyStackTrace {
    fn from(stack_trace: StackTrace) -> Self {
        Self {
            pid: stack_trace.pid,
            thread_id: stack_trace.thread_id,
            thread_name: stack_trace.thread_name,
            os_thread_id: stack_trace.os_thread_id,
            active: stack_trace.active,
            owns_gil: stack_trace.owns_gil,
            frames: stack_trace
                .frames
                .into_iter()
                .map(|frame| frame.into())
                .collect(),
        }
    }
}

impl From<Frame> for PySpyFrame {
    fn from(frame: Frame) -> Self {
        Self {
            name: frame.name,
            filename: frame.filename,
            module: frame.module,
            short_filename: frame.short_filename,
            line: frame.line,
            locals: frame
                .locals
                .map(|locals| locals.into_iter().map(|local| local.into()).collect()),
            is_entry: frame.is_entry,
        }
    }
}

impl From<LocalVariable> for PySpyLocalVariable {
    fn from(local_variable: LocalVariable) -> Self {
        Self {
            name: local_variable.name,
            addr: local_variable.addr,
            arg: local_variable.arg,
            repr: local_variable.repr,
        }
    }
}

/// Run py-spy and return the stack trace. Py-spy is run has a subprocess
/// to avoid any bad side effects specially when running in non-blocking mode
/// which risks segfaulting py-spy process.
pub async fn py_spy(
    pid: i32,
    native: bool,
    native_all: bool,
    blocking: bool,
) -> Result<PySpyTrace, anyhow::Error> {
    // Unfortunately py-spy exec doesn't produce process information when output is json.
    // We need to collect them ourselves.
    let process =
        remoteprocess::Process::new(pid).context(format!("failed to open process {}", pid))?;

    let command_line = process.cmdline()?.join(" ");
    match run_py_spy(pid, native, native_all, blocking).await {
        Ok(stack_traces) => Ok(PySpyTrace {
            pid,
            command_line,
            stack_traces: Some(stack_traces),
            error: None,
        }),
        Err(e) => Ok(PySpyTrace {
            pid,
            command_line,
            stack_traces: None,
            error: Some(e.to_string()),
        }),
    }
}

async fn run_py_spy(
    pid: i32,
    native: bool,
    native_all: bool,
    blocking: bool,
) -> Result<Vec<PySpyStackTrace>, anyhow::Error> {
    let pid_str = pid.to_string();
    let mut args = vec!["dump", "--pid", &pid_str, "--json"];
    if native {
        args.push("--native");
    }
    if native_all {
        args.push("--native-all");
    }
    if !blocking {
        args.push("--nonblocking");
    }

    let pyspy_bin = std::env::var("PYSPY_BIN").unwrap_or("py-spy".to_string());
    tracing::info!("running {} {}", pyspy_bin, args.join(" "));

    // In some situations when py-spy is run in non-blocking mode, it can segfault due
    // to race condition accessing the process memory which can mutate in the process.
    // Nothing much to do but retry a few times.
    async fn spy_call(pyspy_bin: String, args: Vec<&str>) -> Result<String, anyhow::Error> {
        let retries = 3;
        for _x in 0..retries {
            let child = Command::new(pyspy_bin.clone())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .stdin(Stdio::null())
                .args(args.clone())
                .spawn()
                .context("failed to spawn py-spy process")?;
            let result = child
                .wait_with_output()
                .await
                .context("failed to run py-spy process");
            match result {
                Ok(output) if output.status.success() => {
                    let stdout = str::from_utf8(&output.stdout)
                        .context("failed to get py-spy output as utf8")?;
                    return Ok(stdout.to_string());
                }
                _ => {}
            }

            #[allow(clippy::disallowed_methods)]
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        anyhow::bail!("failed to run py-spy after {} retries", retries);
    }

    let stdout = spy_call(pyspy_bin, args).await?;
    let stack_trace: Vec<PySpyStackTrace> =
        serde_json::from_str(&stdout).context("failed to parse py-spy json output")?;

    Ok(stack_trace)
}
