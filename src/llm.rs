// src/llm.rs
use async_trait::async_trait;
use crate::openai::call_gpt;
use crate::gemini::{call_gemini, conversation_gemini_call, get_gemini_model_info, list_gemini_models, count_gemini_tokens, gemini_to_llm_response};
use crate::anthropic::call_anthropic;
use crate::models::gemini::ModelInfo;
use crate::errors::GeneralError;
use futures::stream::BoxStream;
use crate::structs::general::{Message, MessageContent, Content, Part, LlmResponse, LlmChunk}; 
use crate::config::LlmConfig;

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
        content: MessageContent,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>;
    async fn send_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>;

    async fn send_streaming_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<BoxStream<'static, Result<LlmChunk, Box<dyn std::error::Error + Send + Sync>>>, Box<dyn std::error::Error + Send + Sync>>;

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
        config: Option<&LlmConfig>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait]
impl Access for LLM {
    
    async fn send_single_message(
        &self,
        content: MessageContent,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> {
        let messages = vec![Message { role: "user".to_string(), content }];
        self.send_convo_message(messages, model, config).await
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
                            inline_data: None,
                            thought: None,
                        }],
                    })
                    .collect();

                let gemini_response = conversation_gemini_call(gemini_messages, model, config).await?;
                // println!("Raw Gemini Response: {:#?}", gemini_response);
                // REFACTORED: Using the centralized helper
                gemini_to_llm_response(gemini_response)
            }
            LLM::Anthropic => call_anthropic(messages, model, config).await, 
            LLM::LlamaServer => crate::llama_server::call_llama_openai_compat(messages, model, config).await,
        }
    }

    async fn send_streaming_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<BoxStream<'static, Result<LlmChunk, Box<dyn std::error::Error + Send + Sync>>>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => crate::openai::call_gpt_stream(messages, model, config).await,
            LLM::LlamaServer => crate::llama_server::call_llama_stream(messages, model, config).await,
            LLM::Anthropic => crate::anthropic::call_anthropic_stream(messages, model, config).await,
            LLM::Gemini => crate::gemini::api::call_gemini_stream(messages, model, config).await,
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
        config: Option<&LlmConfig>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => {
                crate::openai::get_embedding(text.to_string(), model, dimensions, config).await
            }
            LLM::LlamaServer => {
                crate::llama_server::call_llama_embeddings(text.to_string(), model, dimensions, config).await
            }
            LLM::Gemini => {
                crate::gemini::call_gemini_embeddings(text.to_string(), model, dimensions, config).await
            }
            LLM::Anthropic => {
                Err(Box::new(GeneralError {
                    message: "Anthropic embeddings not yet implemented in Access trait".into(),
                }) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    }
}
