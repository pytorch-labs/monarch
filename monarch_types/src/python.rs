use std::collections::HashMap;

use pyo3::Bound;
use pyo3::PyAny;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::ToPyObject;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyTuple;
use serde::Deserialize;
use serde::Serialize;

/// A fallible version of `ToPyAny`, which also consumes `self`.
pub trait TryIntoPyObject<P> {
    fn try_to_object<'a>(self, py: Python<'a>) -> PyResult<Bound<'a, P>>;
}

/// Blanket impl for `ToPyAny`.
impl<T: ToPyObject> TryIntoPyObject<PyAny> for T {
    fn try_to_object<'a>(self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        Ok(self.to_object(py).into_bound(py))
    }
}

/// A variant of `TryIntoPyObject` used to wrap unsafe impls and propagates the
/// unsafety to the caller.
pub trait TryIntoPyObjectUnsafe<P> {
    unsafe fn try_to_object_unsafe<'a>(self, py: Python<'a>) -> PyResult<Bound<'a, P>>;
}

/// Helper impl for casting into args for python functions calls.
impl<T> TryIntoPyObject<PyTuple> for &Vec<T>
where
    for<'a> &'a T: TryIntoPyObject<PyAny>,
{
    fn try_to_object<'a>(self, py: Python<'a>) -> PyResult<Bound<'a, PyTuple>> {
        Ok(PyTuple::new_bound(
            py,
            self.iter()
                // SAFETY: Safety requirements are propagated via the `unsafe`
                // tag on this method.
                .map(|v| v.try_to_object(py))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

/// Helper impl for casting into kwargs for python functions calls.
impl<K, V> TryIntoPyObject<PyDict> for &HashMap<K, V>
where
    K: ToPyObject,
    for<'a> &'a V: TryIntoPyObject<PyAny>,
{
    fn try_to_object<'a>(self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let mut elems = vec![];
        for (key, val) in self {
            elems.push((key.to_object(py), val.try_to_object(py)?));
        }
        PyDict::from_sequence_bound(&PyList::new_bound(py, elems))
    }
}

/// Helper impl for casting into args for python functions calls.
impl<T> TryIntoPyObjectUnsafe<PyTuple> for &Vec<T>
where
    for<'a> &'a T: TryIntoPyObjectUnsafe<PyAny>,
{
    unsafe fn try_to_object_unsafe<'a>(self, py: Python<'a>) -> PyResult<Bound<'a, PyTuple>> {
        Ok(PyTuple::new_bound(
            py,
            self.iter()
                // SAFETY: Safety requirements are propagated via the `unsafe`
                // tag on this method.
                .map(|v| unsafe { v.try_to_object_unsafe(py) })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

/// Helper impl for casting into kwargs for python functions calls.
impl<K, V> TryIntoPyObjectUnsafe<PyDict> for &HashMap<K, V>
where
    K: ToPyObject,
    for<'a> &'a V: TryIntoPyObjectUnsafe<PyAny>,
{
    unsafe fn try_to_object_unsafe<'a>(self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let mut elems = vec![];
        for (key, val) in self {
            elems.push((
                key.to_object(py),
                // SAFETY: Safety requirements are propagated via the `unsafe`
                // tag on this method.
                unsafe { val.try_to_object_unsafe(py) }?,
            ));
        }
        PyDict::from_sequence_bound(&PyList::new_bound(py, elems))
    }
}

/// A wrapper around `PyErr` that contains a serialized traceback.
#[derive(Debug, Clone, Serialize, Deserialize, derive_more::Error)]
pub struct SerializablePyErr {
    pub etype: String,
    pub value: String,
    pub traceback: Option<Result<String, String>>,
}

impl SerializablePyErr {
    pub fn from(py: Python, err: &PyErr) -> Self {
        let etype = format!("{}", err.get_type_bound(py));
        let value = format!("{}", err.value_bound(py));
        let traceback = err
            .traceback_bound(py)
            .map(|tb| tb.format().map_err(|e| format!("{}", e)));
        Self {
            etype,
            value,
            traceback,
        }
    }

    pub fn from_fn<'py>(py: Python<'py>) -> impl Fn(PyErr) -> Self + use<'py> {
        move |err| Self::from(py, &err)
    }
}

impl std::fmt::Display for SerializablePyErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(tb_res) = &self.traceback {
            match tb_res {
                Ok(tb) => write!(f, "{}", tb)?,
                Err(err) => write!(f, "Failed to extract traceback: {}", err)?,
            }
        }
        write!(f, "{}: {}", self.etype, self.value)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::Python;
    use pyo3::indoc::indoc;
    use pyo3::prelude::*;
    use timed_test::async_timed_test;

    use crate::SerializablePyErr;

    #[async_timed_test(timeout_secs = 60)]
    async fn test_serializable_py_err() {
        pyo3::prepare_freethreaded_python();
        let _unused = Python::with_gil(|py| {
            let module = PyModule::from_code_bound(
                py,
                indoc! {r#"
                        def func1():
                            raise Exception("test")

                        def func2():
                            func1()

                        def func3():
                            func2()
                    "#},
                "test_helpers.py",
                "test_helpers",
            )?;

            let err = SerializablePyErr::from(py, &module.call_method0("func3").unwrap_err());
            assert_eq!(
                err.traceback.unwrap().unwrap().as_str(),
                indoc! {r#"
                    Traceback (most recent call last):
                      File "test_helpers.py", line 8, in func3
                      File "test_helpers.py", line 5, in func2
                      File "test_helpers.py", line 2, in func1
                "#}
            );

            PyResult::Ok(())
        });
    }
}
