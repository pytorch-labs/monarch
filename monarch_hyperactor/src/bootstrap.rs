use hyperactor_mesh::bootstrap_or_die;
use pyo3::Bound;
use pyo3::PyAny;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::pyfunction;

#[pyfunction]
#[pyo3(signature = ())]
pub fn bootstrap_main(py: Python) -> PyResult<Bound<PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py::<_, ()>(py, async move {
        bootstrap_or_die().await;
    })
}
