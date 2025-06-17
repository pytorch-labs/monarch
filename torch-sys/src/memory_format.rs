/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use cxx::ExternType;
use cxx::type_id;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use serde::Serialize;

use crate::MemoryFormat;
use crate::bridge::ffi;

// SAFETY: This type is trival, just an i8.
unsafe impl ExternType for MemoryFormat {
    type Id = type_id!("c10::MemoryFormat");
    type Kind = cxx::kind::Trivial;
}

impl MemoryFormat {
    pub(crate) fn from_py_object_or_none(obj: &Bound<'_, PyAny>) -> Option<Self> {
        ffi::py_object_is_memory_format(obj.clone().into())
            .then(|| ffi::memory_format_from_py_object(obj.into()).unwrap())
    }
}

impl FromPyObject<'_> for MemoryFormat {
    fn extract_bound(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        ffi::memory_format_from_py_object(obj.into()).map_err(|e| {
            PyValueError::new_err(format!(
                "Failed extracting {} from py as Layout: {}",
                obj, e
            ))
        })
    }
}

impl IntoPy<PyObject> for MemoryFormat {
    fn into_py(self, py: Python<'_>) -> PyObject {
        ffi::memory_format_to_py_object(self).into_py(py)
    }
}

// Remotely implement Serialize/Deserialize for generated types
// TODO: we should be able to use parse_callbacks + add_derives, (see
// https://github.com/rust-lang/rust-bindgen/pull/2059) and avoid a remote
// implementation, but this is not supported by rust_bindgen_library.
#[derive(Serialize, Deserialize)]
#[serde(remote = "MemoryFormat")]
pub struct MemoryFormatDef(i8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_to_py_and_back() {
        pyo3::prepare_freethreaded_python();
        let memory_format = MemoryFormat::Contiguous;
        let converted_type = Python::with_gil(|py| {
            // import torch to ensure torch.layout types are registered
            py.import("torch").unwrap();
            let obj = memory_format.into_py(py);
            obj.extract::<MemoryFormat>(py).unwrap()
        });
        assert_eq!(converted_type, MemoryFormat::Contiguous);
    }

    #[test]
    fn from_py() {
        pyo3::prepare_freethreaded_python();
        let memory_format = Python::with_gil(|py| {
            let obj = py
                .import("torch")
                .unwrap()
                .getattr("preserve_format")
                .unwrap();
            obj.extract::<MemoryFormat>().unwrap()
        });
        assert_eq!(memory_format, MemoryFormat::Preserve);
    }
}
