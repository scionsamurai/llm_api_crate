// src/llm.rs
use async_trait::async_trait;
use crate::openai::call_gpt;
use crate::gemini::{call_gemini, conversation_gemini_call, get_gemini_model_info, list_gemini_models, count_gemini_tokens};
use crate::anthropic::call_anthropic;
use crate::models::gemini::ModelInfo;
use crate::errors::GeneralError;
use crate::structs::general::{Message, Content, Part};

pub enum LLM {
    OpenAI,
    Gemini,
    Anthropic,
}

#[async_trait]
pub trait Access {
    async fn send_single_message(
        &self,
        message: &str,
        model: Option<&str>,
        api_key: Option<&str>, // Add api_key parameter
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn send_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        api_key: Option<&str>, // Add api_key parameter
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
        api_key: Option<&str>, // Add api_key parameter
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => {
                let openai_message: Message = Message {
                    role: "user".to_string(),
                    content: message.to_string(),
                };
                call_gpt(vec![openai_message], api_key).await // Pass api_key
            }
            LLM::Gemini => {
                let gemini_message: Message = Message {
                    role: "user".to_string(),
                    content: message.to_string(),
                };
                call_gemini(vec![gemini_message], model, api_key).await // Pass api_key
            }
            LLM::Anthropic => {
                let anthropic_message: Message = Message {
                    role: "user".to_string(),
                    content: message.to_string(),
                };
                call_anthropic(vec![anthropic_message], api_key).await // Pass api_key
            }
        }
    }
    
    async fn send_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        api_key: Option<&str>, // Add api_key parameter
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => call_gpt(messages, api_key).await, // Pass api_key
            LLM::Gemini => {
                let gemini_messages: Vec<Content> = messages
                    .into_iter()
                    .map(|msg| Content {
                        role: msg.role,
                        parts: vec![Part {
                            text: msg.content,
                        }],
                    })
                    .collect();
    
                conversation_gemini_call(gemini_messages, model, api_key).await // Pass api_key
            }
            LLM::Anthropic => call_anthropic(messages, api_key).await, // Pass api_key
        }
    }

    async fn get_model_info(
        &self,
        model: &str,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => get_gemini_model_info(model).await, // Note: get_gemini_model_info might also need an api_key if you extend this pattern to it.
            _ => Err(Box::new(GeneralError {
                message: format!("Currently only Gemini is implemented for get_model_info func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }

    async fn list_models(
        &self,
    ) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => list_gemini_models().await, // Note: list_gemini_models might also need an api_key if you extend this pattern to it.
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
            LLM::Gemini => count_gemini_tokens(text, model).await, // Note: count_gemini_tokens might also need an api_key if you extend this pattern to it.
            _ => Err(Box::new(GeneralError {
                message: format!("Currently only Gemini is implemented for count_tokens func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }
}