use serde::{Deserialize, Serialize};
// use new_proc::NewStruct;

pub enum LLM {
    OpenAI,
    Gemini,
    Claude,
}

// --- OpenAI

#[derive(Debug, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletion {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: f32,
}

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

// --- Gemini

#[derive(Debug, Serialize)]
pub struct GeminiRequest {
    pub contents: Vec<Content>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Part {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<Candidate>,
    pub prompt_feedback: Option<PromptFeedback>,
}

#[derive(Debug, Deserialize)]
pub struct Candidate {
    pub content: Content,
    pub finish_reason: Option<String>,
    pub index: usize,
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Deserialize)]
pub struct PromptFeedback {
    pub safety_ratings: Vec<SafetyRating>,
}

#[derive(Debug, Deserialize)]
pub struct SafetyRating {
    pub category: String,
    pub probability: String,
}

use std::fmt;

#[derive(Debug)]
pub struct GeminiError {
    pub message: String,
}

impl fmt::Display for GeminiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for GeminiError {}

#[derive(Deserialize, Serialize, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub input_token_limit: Option<u32>,
    pub output_token_limit: Option<u32>,
    pub supported_generation_methods: Option<Vec<String>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<f32>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ListModelsResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct TokenCountPart {
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct TokenCountContent {
    pub parts: Vec<TokenCountPart>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CountTokensRequest {
    pub contents: Vec<TokenCountContent>,
}

#[allow(non_snake_case)]
#[derive(Deserialize, Serialize, Debug)]
pub struct CountTokensResponse {
    pub totalTokens: u32,
}
