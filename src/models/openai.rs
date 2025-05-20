// src/models/openai.rs
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct APIMessage {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct APIChoice {
    pub message: APIMessage,
}

#[derive(Debug, Deserialize)]
pub struct APIResponse {
    pub choices: Vec<APIChoice>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
    pub index: u32,
    pub object: String,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub object: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetails,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ErrorDetails {
    pub message: String,
    r#type: String,
    param: Option<String>,
    code: String,
}