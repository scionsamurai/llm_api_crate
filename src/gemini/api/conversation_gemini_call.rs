// src/gemini/api/conversation_gemini_call.rs
use reqwest::header::{HeaderMap, HeaderValue};
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::structs::general::Content;
use crate::gemini::types::GeminiRequest;
use crate::gemini::request::gemini_request;
use crate::gemini::response::handle_gemini_error;

pub async fn conversation_gemini_call(
    messages: Vec<Content>,
    model: Option<&str>,
    api_key_override: Option<&str>, // Add api_key_override parameter
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let default_model = env::var("DEFAULT_GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());

    // Use api_key_override if provided, otherwise fall back to environment variable
    let resolved_api_key: String = if let Some(key) = api_key_override {
        key.to_string()
    } else {
        env::var("GEMINI_API_KEY").map_err(|_| {
            Box::new(GeneralError {
                message: "GEMINI_API_KEY not found in environment variables".to_string(),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?
    };

    let model_name = model.unwrap_or(default_model.as_str());
    let url: String = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        model_name
    );

    let request = GeminiRequest { contents: messages };

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    // Pass the resolved_api_key to gemini_request
    let response = gemini_request(&url, &resolved_api_key, &request, Some(headers)).await?;
    let response_body = response.text().await.map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to read response from Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;
    
    handle_gemini_error(&response_body)
}