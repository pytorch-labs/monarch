use cxx::type_id;
use monarch_types::TryIntoPyObject;
use pyo3::prelude::*;

#[repr(transparent)]
pub(crate) struct FFIPyObject(*mut pyo3::ffi::PyObject);

// SAFETY: This is just a pointer to a PyObject and the pointer is
// never dereferenced directly. It can only be converted to pyo3::PyObject
// and then dereferenced through that. PyO3 manages the access patterns to
// the underlying PyObject.
// Additionally, we make the assumption that ownership of the underlying
// PyObject is transferred with the it.
// Hence FFIPyObject should always be created from an owned pointer.
unsafe impl cxx::ExternType for FFIPyObject {
    type Id = type_id!("monarch::FFIPyObject");
    type Kind = cxx::kind::Trivial;
}

impl<T> From<Py<T>> for FFIPyObject {
    #[inline]
    fn from(obj: Py<T>) -> Self {
        Self(obj.into_ptr())
    }
}

impl<T> From<Bound<'_, T>> for FFIPyObject {
    #[inline]
    fn from(obj: Bound<'_, T>) -> Self {
        Self(obj.into_ptr())
    }
}

impl<T> From<&Bound<'_, T>> for FFIPyObject {
    #[inline]
    fn from(obj: &Bound<'_, T>) -> Self {
        Self(obj.clone().into_ptr())
    }
}

impl IntoPy<PyObject> for FFIPyObject {
    #[inline]
    fn into_py(self, py: Python<'_>) -> PyObject {
        // SAFETY: Pull in the `PyObject` from C/C++.
        unsafe { PyObject::from_owned_ptr(py, self.0) }
    }
}

impl TryIntoPyObject<pyo3::PyAny> for FFIPyObject {
    #[inline]
    fn try_to_object<'a>(self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        // SAFETY: Pull in the `PyObject` from C/C++.
        Ok(unsafe { PyObject::from_owned_ptr(py, self.0) }.into_bound(py))
    }
}
