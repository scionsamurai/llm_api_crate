// src/structs/llama_server.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct LlamaCompletionRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>, // <-- Added model field
    
    pub prompt: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_predict: Option<u32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_prompt: Option<bool>,
    
    // For Gemma 4 vision inputs on the legacy path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_data: Option<Vec<ImageData>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageData {
    pub id: u32,
    pub data: String, // Base64 encoded string of the image
}

#[derive(Debug, Deserialize)]
pub struct LlamaCompletionResponse {
    pub content: String,
    pub stop: bool,
}