// src/llm.rs
use async_trait::async_trait;
use crate::openai::call_gpt;
use crate::gemini::{call_gemini, conversation_gemini_call, get_gemini_model_info, list_gemini_models, count_gemini_tokens};
use crate::anthropic::call_anthropic;
use crate::models::gemini::ModelInfo;
use crate::errors::GeneralError;
use crate::structs::general::{Message, Content, Part, LlmResponse}; // Added LlmResponse
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
        config: Option<&LlmConfig>,
    ) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>; // Changed return type
    async fn send_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>; // Changed return type
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
    async fn embed(
        &self,
        text: &str,
        model: Option<&str>,
        dimensions: Option<u32>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait]
impl Access for LLM {
    
    async fn send_single_message(
        &self,
        message: &str,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => {
                let openai_message: Message = Message {
                    role: "user".to_string(),
                    content: message.into(),
                };
                call_gpt(vec![openai_message], model, config).await 
            }
            LLM::Gemini => {
                let gemini_message: Message = Message {
                    role: "user".to_string(),
                    content: message.into(),
                };
                let gemini_response = call_gemini(vec![gemini_message], model, config).await?;
                
                // Extract text and reasoning by iterating through all parts
                let candidate = gemini_response.candidates.into_iter().next()
                    .ok_or_else(|| Box::new(GeneralError { message: "No Gemini candidates".into() }) as Box<dyn std::error::Error + Send + Sync>)?;
                
                let mut text = String::new();
                let mut reasoning = None;

                for part in candidate.content.parts {
                    if let Some(t) = part.text {
                        text.push_str(&t);
                    }
                    if let Some(th) = part.thought {
                        reasoning = Some(th);
                    }
                }

                Ok(LlmResponse { text, reasoning })
            }
            LLM::Anthropic => {
                let anthropic_message: Message = Message {
                    role: "user".to_string(),
                    content: message.into(),
                };
                call_anthropic(vec![anthropic_message], model, config).await 
            }
            LLM::LlamaServer => {
                let llama_message: Message = Message {
                    role: "user".to_string(),
                    content: message.into(),
                };
                crate::llama_server::call_llama_openai_compat(vec![llama_message], model, config).await
            }
        }
    }
    
    async fn send_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => call_gpt(messages, model, config).await,
            LLM::Gemini => {
                let gemini_messages: Vec<Content> = messages
                    .into_iter()
                    .map(|msg| Content {
                        role: msg.role,
                        parts: vec![Part {
                            text: Some(msg.content.extract_text()),
                            thought: None,
                        }],
                    })
                    .collect();

                let gemini_response = conversation_gemini_call(gemini_messages, model, config).await?;
                
                let candidate = gemini_response.candidates.into_iter().next()
                    .ok_or_else(|| Box::new(GeneralError { message: "No Gemini candidates".into() }) as Box<dyn std::error::Error + Send + Sync>)?;

                let mut text = String::new();
                let mut reasoning = None;

                for part in candidate.content.parts {
                    if let Some(t) = part.text {
                        text.push_str(&t);
                    }
                    if let Some(th) = part.thought {
                        reasoning = Some(th);
                    }
                }

                Ok(LlmResponse { text, reasoning })
            }
            LLM::Anthropic => call_anthropic(messages, model, config).await, 
            LLM::LlamaServer => crate::llama_server::call_llama_openai_compat(messages, model, config).await,
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

    async fn embed(
        &self,
        text: &str,
        model: Option<&str>,
        dimensions: Option<u32>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => {
                crate::openai::get_embedding(text.to_string(), model, dimensions).await
            }
            LLM::LlamaServer => {
                crate::llama_server::call_llama_embeddings(text.to_string(), model, dimensions).await
            }
            LLM::Gemini => {
                Err(Box::new(GeneralError {
                    message: "Gemini embeddings not yet implemented in Access trait".into(),
                }) as Box<dyn std::error::Error + Send + Sync>)
            }
            LLM::Anthropic => {
                Err(Box::new(GeneralError {
                    message: "Anthropic embeddings not yet implemented in Access trait".into(),
                }) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    }
}
