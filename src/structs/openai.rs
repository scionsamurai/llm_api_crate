use serde::Serialize;
use crate::structs::general::{Message, MessageContent, MessagePart, ImageSource};
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: OpenAiContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum OpenAiContent {
    Text(String),
    Array(Vec<OpenAiContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum OpenAiContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    Image { image_url: OpenAiImageUrl },
}

#[derive(Debug, Serialize)]
pub struct OpenAiImageUrl {
    pub url: String,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct ChatCompletion {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_prompt: Option<bool>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>, 
}

#[derive(Debug, Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    pub encoding_format: String,
}