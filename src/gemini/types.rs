// src/gemini/types.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::structs::general::Content;

// New: Struct for generation configuration
#[derive(Debug, Serialize)]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(rename = "thinkingBudget", skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i32>,
    // Add other generation config options here if needed, e.g., top_p, top_k, candidate_count
}

// New: Struct for tools (like google_search)
#[derive(Debug, Serialize)]
pub struct Tool {
    // google_search tool is an empty object, so use serde_json::Value to represent {}
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search: Option<serde_json::Value>, 
    // Add other tool types here if needed
}

// Updated: GeminiRequest now includes generationConfig and tools
#[derive(Debug, Serialize)]
pub struct GeminiRequest {
    pub contents: Vec<Content>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
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
pub struct GroundingMetadata {
    #[serde(rename = "webSearchQueries")]
    pub web_search_queries: Option<Vec<String>>,
    #[serde(rename = "searchEntryPoint")]
    pub search_entry_point: Option<SearchEntryPoint>,
    // Add other grounding fields if they exist in the response and you need them
    // pub groundingChunks: Option<Vec<GroundingChunk>>,
    // pub groundingSupports: Option<Vec<GroundingSupport>>,
}

#[derive(Debug, Deserialize)]
pub struct SearchEntryPoint {
    #[serde(rename = "renderedContent")]
    pub rendered_content: Option<String>,
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
    #[serde(rename = "groundingMetadata")]
    pub grounding_metadata: Option<GroundingMetadata>,
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
    pub details: Option<Vec<GeminiErrorDetail>>, 
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