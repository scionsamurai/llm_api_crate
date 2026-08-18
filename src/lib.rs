// path: src/lib.rs
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
#[cfg(feature = "python")]
use crate::llm::{LLM, Access};
#[cfg(feature = "python")]
use crate::config::LlmConfig;
#[cfg(feature = "python")]
use crate::structs::general::{Message, MessageContent, LlmResponse};

/// Python-exposed wrapper for LlmConfig
#[cfg(feature = "python")]
#[pyclass(name = "LlmConfig")]
#[derive(Clone, Default)]
pub struct PyLlmConfig {
    pub inner: LlmConfig,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLlmConfig {
    #[new]
    fn new() -> Self {
        Self { inner: LlmConfig::new() }
    }

    fn with_temperature(&self, temp: f64) -> Self {
        Self { inner: self.inner.clone().with_temperature(temp) }
    }

    fn with_max_tokens(&self, tokens: u32) -> Self {
        Self { inner: self.inner.clone().with_max_tokens(tokens) }
    }

    fn with_server_url(&self, url: String) -> Self {
        Self { inner: self.inner.clone().with_server_url(url) }
    }
}

/// Python-exposed wrapper for LLM providers
#[cfg(feature = "python")]
#[pyclass(name = "LLMProvider")]
#[derive(Clone, Copy)]
pub enum PyLLMProvider {
    OpenAI,
    Gemini,
    Anthropic,
    LlamaServer,
}

#[cfg(feature = "python")]
impl From<PyLLMProvider> for LLM {
    fn from(p: PyLLMProvider) -> Self {
        match p {
            PyLLMProvider::OpenAI => LLM::OpenAI,
            PyLLMProvider::Gemini => LLM::Gemini,
            PyLLMProvider::Anthropic => LLM::Anthropic,
            PyLLMProvider::LlamaServer => LLM::LlamaServer,
        }
    }
}

/// Simple synchronous helper wrapper for Python to run LLM text generation calls easily
#[cfg(feature = "python")]
#[pyclass(name = "LLMClient")]
pub struct PyLLMClient {
    provider: LLM,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLLMClient {
    #[new]
    fn new(provider: PyLLMProvider) -> Self {
        Self { provider: provider.into() }
    }

    /// Send a single text message and get back the response text synchronously
    fn send_message(&self, prompt: String, model: Option<String>, config: Option<PyLlmConfig>) -> PyResult<String> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        rt.block_on(async {
            let cfg = config.map(|c| c.inner);
            let res = self.provider.send_single_message(
                MessageContent::Text(prompt),
                model.as_deref(),
                cfg.as_ref(),
            ).await.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(res.text)
        })
    }
}

/// Python module entry point, only compiled when the `python` feature is enabled.
#[cfg(feature = "python")]
#[pymodule]
fn llm_api_access(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLlmConfig>()?;
    m.add_class::<PyLLMProvider>()?;
    m.add_class::<PyLLMClient>()?;
    Ok(())
}