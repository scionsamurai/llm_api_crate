// src/llm.rs
use async_trait::async_trait;
use crate::openai::call_gpt;
use crate::gemini::{call_gemini, conversation_gemini_call, get_gemini_model_info, list_gemini_models, count_gemini_tokens};
use crate::anthropic::call_anthropic;
use crate::models::gemini::ModelInfo;
use crate::errors::GeneralError;
use crate::structs::general::{Message, Content, Part};
use crate::config::LlmConfig; // Import LlmConfig

pub enum LLM {
    OpenAI,
    Gemini,
    Anthropic,
    LlamaServer,
}

#[async_trait]
pub trait Access {
    async fn send_single_message(
        &self,
        message: &str,
        model: Option<&str>,
        config: Option<&LlmConfig>, // NEW: optional config parameter
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn send_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        config: Option<&LlmConfig>, // NEW: optional config parameter
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn get_model_info(
        &self,
        model: &str,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>>;
    async fn list_models(&self)
        -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>>;
    async fn count_tokens(
        &self,
        text: &str,
        model: &str,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait]
impl Access for LLM {
    async fn send_single_message(
        &self,
        message: &str,
        model: Option<&str>,
        config: Option<&LlmConfig>, // NEW: optional config parameter
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => {
                let openai_message: Message = Message {
                    role: "user".to_string(),
                    content: message.into(),
                };
                call_gpt(vec![openai_message]).await
            }
            LLM::Gemini => {
                let gemini_message: Message = Message {
                    role: "user".to_string(),
                    content: message.into(),
                };
                // Call call_gemini, then extract the text from the GeminiResponse
                let gemini_response = call_gemini(vec![gemini_message], model, config).await?;
                gemini_response.candidates.into_iter().next()
                    .and_then(|candidate| candidate.content.parts.into_iter().next())
                    .map(|part| part.text)
                    .ok_or_else(|| Box::new(GeneralError {
                        message: "Gemini response did not contain expected text content.".to_string(),
                    }) as Box<dyn std::error::Error + Send + Sync>)
            }
            LLM::Anthropic => {
                let anthropic_message: Message = Message {
                    role: "user".to_string(),
                    content: message.into(),
                };
                call_anthropic(vec![anthropic_message]).await
            }
            LLM::LlamaServer => {
                let llama_message: Message = Message {
                    role: "user".to_string(),
                    content: message.into(),
                };
                crate::llama_server::call_llama_openai_compat(vec![llama_message], config).await
            }
        }
    }
    
    async fn send_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        config: Option<&LlmConfig>, // NEW: optional config parameter
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => call_gpt(messages).await,
            LLM::Gemini => {
                let gemini_messages: Vec<Content> = messages
                    .into_iter()
                    .map(|msg| Content {
                        role: msg.role,
                        parts: vec![Part {
                            text: msg.content.extract_text(),
                        }],
                    })
                    .collect();

                // Call conversation_gemini_call, then extract the text from the GeminiResponse
                let gemini_response = conversation_gemini_call(gemini_messages, model, config).await?;
                let full_text = gemini_response.candidates.into_iter().next()
                    .map(|candidate| {
                        candidate.content.parts.into_iter()
                            .map(|part| part.text)
                            .collect::<Vec<String>>()
                            .join("") // Join all parts into one cohesive string
                    })
                    .ok_or_else(|| Box::new(GeneralError {
                        message: "Gemini response did not contain any candidates.".to_string(),
                    }) as Box<dyn std::error::Error + Send + Sync>)?;

                Ok(full_text)
            }
            LLM::Anthropic => call_anthropic(messages).await,
            LLM::LlamaServer => crate::llama_server::call_llama_openai_compat(messages, config).await,
        }
    }

    async fn get_model_info(
        &self,
        model: &str,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => get_gemini_model_info(model).await,
            _ => Err(Box::new(GeneralError {
                message: format!("Currently only Gemini is implemented for get_model_info func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }

    async fn list_models(
        &self,
    ) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => list_gemini_models().await,
            _ => Err(Box::new(GeneralError {
                message: format!("Currently only Gemini is implemented for list_models func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }

    async fn count_tokens(
        &self,
        text: &str,
        model: &str,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => count_gemini_tokens(text, model).await,
            _ => Err(Box::new(GeneralError {
                message: format!("Currently only Gemini is implemented for count_tokens func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }
}