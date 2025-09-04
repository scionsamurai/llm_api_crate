// src/lib.rs
use pyo3::prelude::*;

mod llm;
pub mod openai;
pub mod gemini;
pub mod anthropic;
pub mod errors;
pub mod models;
pub mod token_count;
pub mod structs;
pub mod tests;

use llm::{LLM, Access};

#[pyfunction]
fn call_llm(py: Python<'_>, llm_type: String, messages: Vec<structs::general::Message>, model: Option<String>) -> PyResult<&'_ PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let llm = match llm_type.to_lowercase().as_str() {
            "openai" => LLM::OpenAI,
            "gemini" => LLM::Gemini,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid LLM type. Choose 'openai' or 'gemini'.")),
        };

        let model_str = model.as_deref();

        let result = if messages.len() == 1 {
            // Use send_single_message if only one message is provided
            llm.send_single_message(&messages[0].content.clone(), model_str).await
        } else {
            // Use send_convo_message if multiple messages are provided
            llm.send_convo_message(messages, model_str).await
        };

        match result {
            Ok(response) => Ok(Python::with_gil(|py| response.into_py(py))),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string())),
        }
    })
}


#[pyfunction]
fn get_embedding(py: Python<'_>, input: String, dimensions: Option<u32>) -> PyResult<&'_ PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        openai::get_embedding(input, dimensions)
            .await
            .map(|result| Python::with_gil(|py| result.into_py(py)))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))
    })
}


#[pymodule]
fn llm_api_access(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(call_llm, m)?)?;
    m.add_function(wrap_pyfunction!(get_embedding, m)?)?;
    Ok(())
}