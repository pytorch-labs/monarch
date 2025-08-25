/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use derive_more::From;
use derive_more::TryInto;
use hyperactor::Named;
use monarch_types::PickledPyObject;
use monarch_types::TryIntoPyObjectUnsafe;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyNone;
use serde::Deserialize;
use serde::Serialize;

use crate::Device;
use crate::IValue;
use crate::IValueKind;
use crate::Layout;
use crate::MemoryFormat;
use crate::ScalarType;
use crate::TensorCell;
use crate::cell::CloneUnsafe;
use crate::ivalue::OpaqueIValueCell;

/// A pure Rust equivalent for [`IValue`]. This is safe to treat like a normal
/// Rust value.
#[derive(Debug, Clone, From, TryInto, Serialize, Deserialize, Named)]
#[try_into(owned, ref, ref_mut)]
pub enum RValue {
    Tensor(TensorCell),
    TensorList(Vec<TensorCell>),
    Int(i64),
    IntList(Vec<i64>),
    Double(f64),
    Bool(bool),
    String(String),
    Device(Device),
    Layout(#[serde(with = "crate::LayoutDef")] Layout),
    ScalarType(#[serde(with = "crate::ScalarTypeDef")] ScalarType),
    MemoryFormat(#[serde(with = "crate::MemoryFormatDef")] MemoryFormat),
    None,
    PyObject(PickledPyObject),
    /// This is meant to be a catch-all for types that we don't support
    /// natively in Rust.
    Opaque(OpaqueIValueCell),
}

// SAFETY: this function creates untracked aliases of tensors. The caller is
// responsible for having acquired the suitable borrows and holding them for the
// entire lifetime of the returned IValue.
pub unsafe fn rvalue_to_ivalue(rvalue: &RValue) -> IValue {
    match rvalue {
        // TODO fix unwrap
        RValue::Tensor(cell) => {
            // SAFETY: caller is responsible for holding a borrow, so the outer
            // function is marked unsafe.
            IValue::from(unsafe { cell.get_unchecked().clone_unsafe() })
        }
        RValue::TensorList(cells) => {
            let mut tensors = Vec::new();
            for cell in cells {
                // SAFETY: caller is responsible for holding a borrow, so the outer
                // function is marked unsafe.
                tensors.push(unsafe { cell.get_unchecked().clone_unsafe() });
            }
            IValue::from(tensors)
        }
        RValue::Int(val) => IValue::from(*val),
        RValue::IntList(val) => IValue::from(val.as_slice()),
        RValue::Double(val) => IValue::from(*val),
        RValue::Bool(val) => IValue::from(*val),
        RValue::String(val) => IValue::from(val),
        RValue::Device(val) => IValue::from(*val),
        // It appears that the enums for Layout/ScalarType/MemoryFormat are just
        // stored as raw ints in `IValue` and that we lose all info about how
        // to convert them back.
        RValue::Layout(val) => IValue::from(val.0 as i64),
        RValue::ScalarType(val) => IValue::from(val.0 as i64),
        RValue::MemoryFormat(val) => IValue::from(val.0 as i64),
        RValue::None => IValue::from(()),
        RValue::PyObject(val) => {
            Python::with_gil(|py| val.unpickle(py).unwrap().extract::<IValue>())
                .expect("unable to convert PyObject to IValue")
        }
        RValue::Opaque(cell) => {
            // SAFETY: caller is responsible for holding a borrow, so the outer
            // function is marked unsafe.
            unsafe { cell.get_unchecked().ivalue() }
        }
    }
}

impl From<IValue> for RValue {
    fn from(ivalue: IValue) -> Self {
        match ivalue.kind() {
            IValueKind::Tensor => RValue::Tensor(TensorCell::new(ivalue.to_tensor().unwrap())),
            IValueKind::Bool => RValue::Bool(ivalue.to_bool().unwrap()),
            IValueKind::Int => RValue::Int(ivalue.to_int().unwrap()),
            IValueKind::IntList => RValue::IntList(ivalue.to_int_list().unwrap()),
            IValueKind::Double => RValue::Double(ivalue.to_double().unwrap()),
            IValueKind::String => RValue::String(ivalue.to_string().unwrap()),
            IValueKind::TensorList => RValue::TensorList(
                ivalue
                    .to_tensor_list()
                    .unwrap()
                    .into_iter()
                    .map(TensorCell::new)
                    .collect(),
            ),
            IValueKind::Device => RValue::Device(ivalue.to_device().unwrap()),
            IValueKind::None => RValue::None,
            IValueKind::Other => RValue::Opaque(OpaqueIValueCell::new(ivalue.to_opaque().unwrap())),
        }
    }
}

impl<'py> IntoPyObject<'py> for RValue {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            RValue::Int(val) => IValue::from(val).into_pyobject(py),
            RValue::IntList(val) => IValue::from(val.as_slice()).into_pyobject(py),
            RValue::Double(val) => IValue::from(val).into_pyobject(py),
            RValue::Bool(val) => IValue::from(val).into_pyobject(py),
            RValue::String(val) => IValue::from(&val).into_pyobject(py),
            RValue::Device(val) => IValue::from(val).into_pyobject(py),
            // Avoid converting layout and scalar type into ivalues, as it appears
            // they just get converted to ints.
            RValue::Layout(val) => val.clone().into_pyobject(py),
            RValue::ScalarType(val) => val.clone().into_pyobject(py),
            RValue::MemoryFormat(val) => val.clone().into_pyobject(py),
            RValue::None => PyNone::get(py).into_bound_py_any(py),
            RValue::PyObject(val) => val.unpickle(py),
            _ => Err(PyErr::new::<PyValueError, _>(format!(
                "cannot safely create py object from {:?}",
                self
            ))),
        }
    }
}

/// Convert into a `PyObject`.
impl<'py> TryIntoPyObjectUnsafe<'py, PyAny> for &RValue {
    unsafe fn try_to_object_unsafe(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            // Avoid converting layout, scalar type, memory format into ivalues, as it appears
            // they just get converted to ints.
            // None and PyObject are also not converted as there is no need to do so.
            RValue::Layout(val) => val.clone().into_pyobject(py),
            RValue::ScalarType(val) => val.clone().into_pyobject(py),
            RValue::MemoryFormat(val) => val.clone().into_pyobject(py),
            RValue::None => PyNone::get(py).into_bound_py_any(py),
            RValue::PyObject(val) => val.unpickle(py),
            // SAFETY: This inherits the unsafety of `rvalue_to_ivalue` (see comment
            // above).
            _ => unsafe { rvalue_to_ivalue(self).into_pyobject(py) },
        }
    }
}

impl FromPyObject<'_> for RValue {
    fn extract_bound(obj: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        // It's crucial for correctness to try converting to IValue after we've
        // tried the other non-PyObject variants, because the IValue conversion
        // will actually succeed when obj is a ScalarType, Layout, or MemoryFormat.
        if let Some(val) = ScalarType::from_py_object_or_none(obj) {
            Ok(RValue::ScalarType(val))
        } else if let Some(val) = Layout::from_py_object_or_none(obj) {
            Ok(RValue::Layout(val))
        } else if let Some(val) = MemoryFormat::from_py_object_or_none(obj) {
            Ok(RValue::MemoryFormat(val))
        } else if let Some(val) = IValue::from_py_object_or_none(obj) {
            Ok(val.into())
        } else {
            Ok(RValue::PyObject(PickledPyObject::pickle(obj)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use anyhow::Result;
    use pyo3::ffi::c_str;
    use pyo3::prelude::*;

    use super::*;

    #[test]
    fn test_py_object() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        let rval = Python::with_gil(|py| {
            // Needed to initialize torch.
            py.import("torch")?;

            // Define the Custom class inline
            py.run(c_str!("class Custom:\n    pass"), None, None)?;

            let obj = py.eval(pyo3::ffi::c_str!("Custom()"), None, None)?;
            RValue::extract_bound(&obj)
        })?;
        // NOTE(agallagher): Among other things, verify this isn't accidentally
        // extracted as an `IValue`.
        assert_matches!(rval, RValue::PyObject(_));
        Ok(())
    }
}
