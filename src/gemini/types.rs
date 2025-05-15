// src/gemini/types.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize)]
pub struct GeminiRequest {
    pub contents: Vec<Content>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<Candidate>,
    pub prompt_feedback: Option<PromptFeedback>,
    pub usage_metadata: Option<UsageMetadata>,
    pub model_version: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: u32,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: u32,
    #[serde(rename = "promptTokensDetails")]
    pub prompt_tokens_details: Option<Vec<TokenDetails>>,
    #[serde(rename = "candidatesTokensDetails")]
    pub candidates_tokens_details: Option<Vec<TokenDetails>>,
}

#[derive(Debug, Deserialize)]
pub struct TokenDetails {
    pub modality: String,
    #[serde(rename = "tokenCount")]
    pub token_count: u32,
}

#[derive(Debug, Deserialize)]
pub struct Candidate {
    pub content: Content,
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
    #[serde(rename = "avgLogprobs")]
    pub avg_log_probs: Option<f64>,
    pub index: Option<usize>,
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

#[derive(Deserialize, Debug)]
pub struct GeminiErrorResponse {
    pub error: GeminiError,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct GeminiError {
    pub code: u16,
    pub message: String,
    pub status: String,
    pub details: Vec<GeminiErrorDetail>,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct GeminiErrorDetail {
    #[serde(rename = "@type")]
    pub type_: String,
    pub reason: String,
    pub domain: String,
    pub metadata: HashMap<String, String>,
}

use crate::structs::Content;