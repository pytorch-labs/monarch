/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#![allow(unsafe_op_in_unsafe_fn)]
use std::sync::Arc;

use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::Named;
use hyperactor::ProcId;
use monarch_hyperactor::mailbox::PyMailbox;
use monarch_hyperactor::runtime::signal_safe_block_on;
use monarch_rdma::RdmaBuffer;
use monarch_rdma::RdmaManagerActor;
use monarch_rdma::RdmaMemoryRegionView;
use monarch_rdma::ibverbs_supported;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::Deserialize;
use serde::Serialize;

macro_rules! setup_rdma_context {
    ($self:ident, $caller_proc_id:expr) => {{
        let proc_id: ProcId = $caller_proc_id.parse().unwrap();
        let caller_owner_id = ActorId(proc_id, "rdma_manager".to_string(), 0);
        let caller_owner_ref: ActorRef<RdmaManagerActor> = ActorRef::attest(caller_owner_id);
        let buffer = Arc::clone(&$self.buffer);
        (caller_owner_ref, buffer)
    }};
}

#[pyclass(name = "_RdmaBuffer", module = "monarch._rust_bindings.rdma")]
#[derive(Clone, Serialize, Deserialize, Named)]
struct PyRdmaBuffer {
    name: String,
    buffer: Arc<RdmaBuffer>,
    owner_ref: ActorRef<RdmaManagerActor>,
}

async fn create_rdma_buffer(
    name: String,
    addr: usize,
    size: usize,
    proc_id: String,
    client: PyMailbox,
) -> PyResult<PyRdmaBuffer> {
    // Get the owning RdmaManagerActor's ActorRef
    let proc_id: ProcId = proc_id.parse().unwrap();
    let owner_id = ActorId(proc_id, "rdma_manager".to_string(), 0);
    let owner_ref: ActorRef<RdmaManagerActor> = ActorRef::attest(owner_id);

    // Create the RdmaBuffer
    let rdma_buffer = RdmaBuffer::new(
        name.clone(),
        owner_ref.clone(),
        &client.inner,
        RdmaMemoryRegionView::new(addr, size),
    )
    .await?;

    let buffer = Arc::new(rdma_buffer);
    Ok(PyRdmaBuffer {
        name,
        buffer,
        owner_ref,
    })
}

#[pymethods]
impl PyRdmaBuffer {
    #[classmethod]
    fn create_rdma_buffer_blocking<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        name: String,
        addr: usize,
        size: usize,
        proc_id: String,
        client: PyMailbox,
    ) -> PyResult<PyRdmaBuffer> {
        if !ibverbs_supported() {
            return Err(PyException::new_err(
                "ibverbs is not supported on this system",
            ));
        }
        signal_safe_block_on(py, create_rdma_buffer(name, addr, size, proc_id, client))?
    }

    #[classmethod]
    fn create_rdma_buffer_nonblocking<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        name: String,
        addr: usize,
        size: usize,
        proc_id: String,
        client: PyMailbox,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !ibverbs_supported() {
            return Err(PyException::new_err(
                "ibverbs is not supported on this system",
            ));
        }
        pyo3_async_runtimes::tokio::future_into_py(
            py,
            create_rdma_buffer(name, addr, size, proc_id, client),
        )
    }

    #[classmethod]
    fn rdma_supported<'py>(_cls: &Bound<'_, PyType>, py: Python<'py>) -> bool {
        ibverbs_supported()
    }

    #[pyo3(name = "__repr__")]
    fn repr(&self) -> String {
        format!(
            "<RdmaBuffer name='{}',buffer='{:?}'>",
            self.name, self.buffer
        )
    }

    #[pyo3(signature = (addr, size, caller_proc_id, client, timeout=None))]
    fn read_into<'py>(
        &self,
        py: Python<'py>,
        addr: usize,
        size: usize,
        caller_proc_id: String,
        client: PyMailbox,
        timeout: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (caller_owner_ref, buffer) = setup_rdma_context!(self, caller_proc_id);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let _result_ = buffer
                .read_into(
                    RdmaMemoryRegionView::new(addr, size),
                    &client.inner,
                    &caller_owner_ref,
                    timeout,
                )
                .await
                .map_err(|e| PyException::new_err(format!("failed to read into buffer: {}", e)))?;
            Ok(())
        })
    }

    #[pyo3(signature = (addr, size, caller_proc_id, client, timeout=None))]
    fn read_into_blocking<'py>(
        &self,
        py: Python<'py>,
        addr: usize,
        size: usize,
        caller_proc_id: String,
        client: PyMailbox,
        timeout: Option<u32>,
    ) -> PyResult<Option<u64>> {
        let (caller_owner_ref, buffer) = setup_rdma_context!(self, caller_proc_id);
        signal_safe_block_on(py, async move {
            buffer
                .read_into(
                    RdmaMemoryRegionView::new(addr, size),
                    &client.inner,
                    &caller_owner_ref,
                    timeout,
                )
                .await
                .map_err(|e| PyException::new_err(format!("failed to read into buffer: {}", e)))
        })?
    }

    #[pyo3(signature = (addr, size, caller_proc_id, client, timeout=None))]
    fn write_from<'py>(
        &self,
        py: Python<'py>,
        addr: usize,
        size: usize,
        caller_proc_id: String,
        client: PyMailbox,
        timeout: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (caller_owner_ref, buffer) = setup_rdma_context!(self, caller_proc_id);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let _result_ = buffer
                .write_from(
                    RdmaMemoryRegionView::new(addr, size),
                    &client.inner,
                    &caller_owner_ref,
                    timeout,
                )
                .await
                .map_err(|e| PyException::new_err(format!("failed to write from buffer: {}", e)))?;
            Ok(())
        })
    }

    #[pyo3(signature = (addr, size, caller_proc_id, client, timeout=None))]
    fn write_from_blocking<'py>(
        &self,
        py: Python<'py>,
        addr: usize,
        size: usize,
        caller_proc_id: String,
        client: PyMailbox,
        timeout: Option<u32>,
    ) -> PyResult<Option<u64>> {
        let (caller_owner_ref, buffer) = setup_rdma_context!(self, caller_proc_id);
        signal_safe_block_on(py, async move {
            buffer
                .write_from(
                    RdmaMemoryRegionView::new(addr, size),
                    &client.inner,
                    &caller_owner_ref,
                    timeout,
                )
                .await
                .map_err(|e| PyException::new_err(format!("failed to write from buffer: {}", e)))
        })?
    }

    fn release<'py>(
        &self,
        py: Python<'py>,
        addr: usize,
        size: usize,
        caller_proc_id: String,
        client: PyMailbox,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (caller_owner_ref, buffer) = setup_rdma_context!(self, caller_proc_id);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            buffer
                .release(
                    &client.inner,
                    RdmaMemoryRegionView::new(addr, size),
                    &caller_owner_ref,
                )
                .await
                .map_err(|e| PyException::new_err(format!("failed to release buffer: {}", e)))?;
            Ok(())
        })
    }

    fn release_blocking<'py>(
        &self,
        py: Python<'py>,
        addr: usize,
        size: usize,
        caller_proc_id: String,
        client: PyMailbox,
    ) -> PyResult<()> {
        let (caller_owner_ref, buffer) = setup_rdma_context!(self, caller_proc_id);
        signal_safe_block_on(py, async move {
            buffer
                .release(
                    &client.inner,
                    RdmaMemoryRegionView::new(addr, size),
                    &caller_owner_ref,
                )
                .await
                .map_err(|e| PyException::new_err(format!("failed to release buffer: {}", e)))
        })?
    }

    fn __reduce__(&self) -> PyResult<(PyObject, PyObject)> {
        Python::with_gil(|py| {
            let ctor = py.get_type_bound::<PyRdmaBuffer>().to_object(py);
            let json = serde_json::to_string(self).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Serialization failed: {}", e))
            })?;
            Ok((ctor, (json,).to_object(py)))
        })
    }

    #[new]
    fn new_from_json(json: &str) -> PyResult<Self> {
        let deserialized: PyRdmaBuffer = serde_json::from_str(json)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Deserialization failed: {}", e)))?;
        Ok(deserialized)
    }

    fn drop<'py>(&self, py: Python<'py>, client: PyMailbox) -> PyResult<Bound<'py, PyAny>> {
        let buffer = Arc::clone(&self.buffer);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            buffer
                .drop_buffer(&client.inner)
                .await
                .map_err(|e| PyException::new_err(format!("failed to drop buffer: {}", e)))?;
            Ok(())
        })
    }

    fn drop_blocking<'py>(&self, py: Python<'py>, client: PyMailbox) -> PyResult<()> {
        let buffer = Arc::clone(&self.buffer);
        signal_safe_block_on(py, async move {
            buffer
                .drop_buffer(&client.inner)
                .await
                .map_err(|e| PyException::new_err(format!("failed to drop buffer: {}", e)))
        })?
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRdmaBuffer>()?;
    Ok(())
}
