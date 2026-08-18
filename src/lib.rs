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
use crate::structs::general::{Message, MessageContent, MessagePart, ImageSource, LlmResponse};

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

/// Python-exposed wrapper for constructing chat messages and multimodal inputs
#[cfg(feature = "python")]
#[pyclass(name = "Message")]
#[derive(Clone)]
pub struct PyMessage {
    pub inner: Message,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMessage {
    #[new]
    #[pyo3(signature = (role, text=None, image_base64=None, image_media_type=None, image_url=None))]
    fn new(
        role: String,
        text: Option<String>,
        image_base64: Option<String>,
        image_media_type: Option<String>,
        image_url: Option<String>,
    ) -> Self {
        let mut parts = Vec::new();

        if let Some(t) = text {
            parts.push(MessagePart {
                r#type: "text".to_string(),
                text: Some(t),
                image_url: None,
            });
        }

        if let Some(b64) = image_base64 {
            let media_type = image_media_type.unwrap_or_else(|| "image/jpeg".to_string());
            parts.push(MessagePart {
                r#type: "image_url".to_string(),
                text: None,
                image_url: Some(ImageSource::Base64 { media_type, data: b64 }),
            });
        } else if let Some(url) = image_url {
            parts.push(MessagePart {
                r#type: "image_url".to_string(),
                text: None,
                image_url: Some(ImageSource::Url { url }),
            });
        }

        let content = if parts.len() == 1 && parts[0].r#type == "text" {
            MessageContent::Text(parts[0].text.clone().unwrap_or_default())
        } else {
            MessageContent::Array(parts)
        };

        Self {
            inner: Message { role, content },
        }
    }
}

/// Synchronous client interface for Python to handle single messages, conversations, and multimodal prompts
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

    /// Send a single text message and get back response text
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

    /// Send a conversation history (`Vec<Message>`) or multimodal prompt bundle
    fn send_chat(&self, messages: Vec<PyMessage>, model: Option<String>, config: Option<PyLlmConfig>) -> PyResult<String> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        rt.block_on(async {
            let cfg = config.map(|c| c.inner);
            let rust_messages: Vec<Message> = messages.into_iter().map(|m| m.inner).collect();
            
            let res = self.provider.send_convo_message(
                rust_messages,
                model.as_deref(),
                cfg.as_ref(),
            ).await.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(res.text)
        })
    }
}

/// Python module entry point
#[cfg(feature = "python")]
#[pymodule]
fn llm_api_access(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLlmConfig>()?;
    m.add_class::<PyLLMProvider>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyLLMClient>()?;
    Ok(())
}