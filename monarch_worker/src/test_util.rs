use std::io::IsTerminal;

use anyhow::Result;
use pyo3::Python;
use tracing_subscriber::fmt::format::FmtSpan;

pub fn test_setup() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_thread_ids(true)
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(std::io::stderr().is_terminal())
        .with_writer(std::io::stderr)
        .try_init();

    // Redirect NCCL_DEBUG log output to a file so it doesn't clash on stdout.
    // TestX requires stdout to have JSON output on individual lines, and
    // the NCCL output is not JSON. Because it runs in a different thread, it'll
    // race on writing to stdout.
    // Do this regardless of whether NCCL_DEBUG is set or not, because it can
    // be set after this point in the test. If it doesn't get set, NCCL_DEBUG_FILE
    // will be ignored.
    // %h becomes hostname, %p becomes pid.
    let nccl_debug_file = std::env::temp_dir().join("nccl_debug.%h.%p");
    tracing::debug!("Set NCCL_DEBUG_FILE to {:?}", nccl_debug_file);
    std::env::set_var("NCCL_DEBUG_FILE", nccl_debug_file);

    // NOTE(agallagher): Calling `prepare_freethreaded_python` appears to
    // clear `PYTHONPATH` in the env, which we need for test subprocesses
    // to work.  So, manually preserve it.
    let py_path = std::env::var("PYTHONPATH");
    pyo3::prepare_freethreaded_python();
    if let Ok(py_path) = py_path {
        // SAFETY: Re-setting env var cleard by `prepare_freethreaded_python`.
        unsafe { std::env::set_var("PYTHONPATH", py_path) }
    }

    // We need to load torch to initialize some internal structures used by
    // the FFI funcs we use to convert ivalues to/from py objects.
    Python::with_gil(|py| py.run_bound("import torch", None, None))?;

    Ok(())
}
