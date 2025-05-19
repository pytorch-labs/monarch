use ndslice::selection::Selection;
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyclass(name = "Selection", module = "monarch._monarch.selection", frozen)]
pub struct PySelection {
    inner: Selection,
}

impl From<Selection> for PySelection {
    fn from(inner: Selection) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySelection {
    #[getter]
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    #[classmethod]
    #[pyo3(name = "from_string")]
    pub fn parse(_cls: Bound<'_, PyType>, input: &str) -> PyResult<Self> {
        // TODO: Make this a utility in ndslice.
        use ndslice::selection::parse::expression;
        use nom::combinator::all_consuming;

        let input: String = input.chars().filter(|c| !c.is_whitespace()).collect();
        let (_, selection) = all_consuming(expression)(&input).map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!("parse error: {err}"))
        })?;

        Ok(PySelection::from(selection))
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySelection>()?;
    Ok(())
}
