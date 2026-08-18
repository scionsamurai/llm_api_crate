// src/lib.rs

pub mod llm;
pub mod openai;
pub mod gemini;
pub mod anthropic;
pub mod errors;
pub mod models;
pub mod token_count;
pub mod structs;
pub mod tests;
pub mod config;
pub mod llama_server;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module entry point, only compiled when the `python` feature is enabled.
#[cfg(feature = "python")]
#[pymodule]
fn llm_api_access(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Register your classes/functions here when exposing them to Python.
    // e.g., m.add_class::<YourPyClass>()?;
    Ok(())
}