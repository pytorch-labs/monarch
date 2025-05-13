use hyperactor::Named;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde::Serialize;

use crate::TryIntoPyObject;

#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub struct PickledPyObject(#[serde(with = "serde_bytes")] Vec<u8>);

impl PickledPyObject {
    pub fn pickle<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let bytes = obj
            .py()
            .import_bound("pickle")?
            .call_method1("dumps", (obj,))?
            .downcast_into::<PyBytes>()?
            .as_bytes()
            .to_vec();
        Ok(Self(bytes))
    }

    pub fn unpickle<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import_bound("pickle")?
            .call_method1("loads", (self.0.as_slice(),))
    }
}

impl TryFrom<&Bound<'_, PyAny>> for PickledPyObject {
    type Error = PyErr;
    fn try_from(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::pickle(obj)
    }
}

impl TryFrom<Bound<'_, PyAny>> for PickledPyObject {
    type Error = PyErr;
    fn try_from(obj: Bound<'_, PyAny>) -> PyResult<Self> {
        Self::pickle(&obj)
    }
}

impl FromPyObject<'_> for PickledPyObject {
    fn extract_bound(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        PickledPyObject::pickle(obj)
    }
}

impl TryIntoPyObject<PyAny> for &PickledPyObject {
    fn try_to_object<'a>(self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        self.unpickle(py)
    }
}
