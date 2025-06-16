/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use monarch_messages::worker::ResolvableFunction;
use monarch_types::PyTree;
use monarch_types::TryIntoPyObject;
use monarch_types::TryIntoPyObjectUnsafe;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use torch_sys::RValue;

use crate::pipe::Pipe;

/// Wrapper around `Pipe` to make it usable in Python.
#[pyclass]
pub struct PyPipe {
    pipe: Box<dyn Pipe<PyTree<RValue>> + Send + Sync>,
    #[pyo3(get)]
    ranks: HashMap<String, usize>,
    #[pyo3(get)]
    sizes: HashMap<String, usize>,
    allow_unsafe_obj_conversion: bool,
}

impl PyPipe {
    pub fn new(
        pipe: Box<dyn Pipe<PyTree<RValue>> + Send + Sync>,
        ranks: HashMap<String, usize>,
        sizes: HashMap<String, usize>,
        allow_unsafe_obj_conversion: bool,
    ) -> Self {
        Self {
            pipe,
            ranks,
            sizes,
            allow_unsafe_obj_conversion,
        }
    }
}

#[pymethods]
impl PyPipe {
    fn send(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = value.extract::<PyTree<RValue>>()?;
        py.allow_threads(move || self.pipe.send(val))?;
        Ok(())
    }

    fn recv<'a>(&mut self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let val = py.allow_threads(|| self.pipe.recv())?;
        if self.allow_unsafe_obj_conversion {
            // SAFETY: A caller who initialized this PyPipe with allow_unsafe_obj_conversion=True
            // asserts that it is safe to use this unsafe method.
            unsafe { val.try_to_object_unsafe(py) }
        } else {
            val.try_to_object(py)
        }
    }
}

/// Run a Python pipe server, which loads a remote function sent over the pipe
/// then delegates to it.
pub fn run_py_pipe(
    pipe: PyPipe,
    func: ResolvableFunction,
    args: Vec<PyTree<RValue>>,
    kwargs: HashMap<String, PyTree<RValue>>,
) -> PyResult<()> {
    Python::with_gil(|py| {
        let pipe_obj: Py<PyPipe> = Py::new(py, pipe)?;
        let func = func.resolve(py)?;
        let mut py_args = vec![pipe_obj.into_bound(py).into_any()];
        py_args.extend(
            args.into_iter()
                .map(|a| a.try_to_object(py))
                .collect::<Result<Vec<_>, _>>()?,
        );
        func.call(PyTuple::new(py, py_args)?, Some(&kwargs.try_to_object(py)?))?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;
    use std::collections::HashMap;

    use anyhow::Result;
    use futures::try_join;
    use indoc::indoc;
    use pyo3::Python;
    use pyo3::ffi::c_str;
    use pyo3::types::PyModule;
    use timed_test::async_timed_test;
    use torch_sys::RValue;

    use super::PyPipe;
    use super::run_py_pipe;
    use crate::pipe::AsyncPipe;
    use crate::pipe::create_local_pipe;

    #[async_timed_test(timeout_secs = 60)]
    async fn test_py_pipe() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        // We need to load torch to initialize some internal structures used by
        // the FFI funcs we use to convert ivalues to/from py objects.
        Python::with_gil(|py| py.run(c_str!("import torch"), None, None))?;

        // Create the Python function that runs as the pipe handler.
        Python::with_gil(|py| {
            let _mod = PyModule::from_code(
                py,
                c_str!(indoc! {r#"
                    def func(pipe):
                        val = pipe.recv()
                        pipe.send(val)
                "#}),
                c_str!("test_helpers.py"),
                c_str!("test_helpers"),
            )?;
            anyhow::Ok(())
        })?;

        let (mut client, server) = create_local_pipe();
        let ((), ()) = try_join!(
            // Startup the pipe server side.
            async move {
                tokio::task::spawn_blocking(move || {
                    run_py_pipe(
                        PyPipe::new(
                            Box::new(server),
                            HashMap::new(),
                            HashMap::new(),
                            false, // allow_unsafe_obj_conversion
                        ),
                        "test_helpers.func".into(),
                        vec![],
                        HashMap::new(),
                    )
                })
                .await??;
                anyhow::Ok(())
            },
            // Run the pipe client side.
            async move {
                client.send(RValue::Int(3).into()).await?;
                let val = client.recv().await?;
                assert_matches!(val.into_leaf().unwrap(), RValue::Int(3));
                anyhow::Ok(())
            },
        )?;

        Ok(())
    }
}
