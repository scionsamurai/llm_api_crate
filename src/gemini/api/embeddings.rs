// src/gemini/api/embeddings.rs
use std::env;
use dotenv::dotenv;
use crate::errors::GeneralError;
use crate::gemini::types::{GeminiEmbeddingRequest, GeminiEmbeddingContent, GeminiEmbeddingResponse};
use crate::gemini::request::gemini_request;
use crate::config::LlmConfig;
use crate::structs::general::Part;
use reqwest::header::{HeaderMap, HeaderValue};

pub async fn call_gemini_embeddings(
    text: String,
    model: Option<&str>,
    dimensions: Option<u32>,
    _config: Option<&LlmConfig>,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();
    
    // Use default model if none provided. 
    // NOTE: Ensure DEFAULT_GEMINI_MODEL is NOT a chat model like 'gemini-2.5-flash'
    let default_model = env::var("DEFAULT_GEMINI_MODEL").unwrap_or_else(|_| "gemini-embedding-001".to_string());
    let api_key = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GEMINI API KEY not found in environment variables".to_string(),
    })?;

    let model_name = model.unwrap_or(&default_model);
    
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent",
        model_name
    );

    let request = GeminiEmbeddingRequest {
        model: format!("models/{}", model_name),
        content: GeminiEmbeddingContent {
            parts: vec![Part {
                text: Some(text),
                thought: None,
            }],
        },
        output_dimensionality: dimensions,
    };

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    // 1. Perform the request
    let res = gemini_request(&url, &api_key, &request, Some(headers)).await?;

    // 2. Capture the status and the body
    let status = res.status();
    let res_text = res.text().await.map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to read response body: {}", e),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    // 3. Check if it's actually a success before trying to parse the embedding
    if !status.is_success() {
        return Err(Box::new(GeneralError {
            message: format!("Gemini API Error (HTTP {}): {}", status, res_text),
        }) as Box<dyn std::error::Error + Send + Sync>);
    }

    // 4. Parse the successful response
    let gemini_response: GeminiEmbeddingResponse = serde_json::from_str(&res_text).map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to parse successful Gemini embedding response: {} - Body: {}", e, res_text),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    Ok(gemini_response.embedding.values)
}