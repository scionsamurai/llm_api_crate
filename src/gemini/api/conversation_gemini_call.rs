// src/gemini/api/conversation_gemini_call.rs
use reqwest::header::{HeaderMap, HeaderValue};
use std::env;
use dotenv::dotenv;
use serde_json::{json, Map, Value};

use crate::errors::GeneralError;
use crate::structs::general::Content;
use crate::gemini::types::GeminiRequest;
use crate::gemini::request::gemini_request;
use crate::gemini::response::handle_gemini_error;
use crate::config::LlmConfig; // Import LlmConfig

const DEFAULT_GEMINI_MODEL: &str = "gemini-2.0-flash";

pub async fn conversation_gemini_call(
    messages: Vec<Content>,
    model: Option<&str>,
    config: Option<&LlmConfig>, // NEW: optional config parameter
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    let model_name = model.unwrap_or(DEFAULT_GEMINI_MODEL);
    let url: String = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        model_name
    );

    let mut request_body: Map<String, Value> = Map::new();
    request_body.insert("contents".to_string(), json!(messages));

    // Add generationConfig if config is provided
    if let Some(cfg) = config {
        let mut generation_config: Map<String, Value> = Map::new();

        if let Some(thinking_budget) = cfg.thinking_budget {
            generation_config.insert(
                "thinkingBudget".to_string(),
                json!(thinking_budget)
            );
        }

        if let Some(temp) = cfg.temperature {
            generation_config.insert("temperature".to_string(), json!(temp));
        }

        if !generation_config.is_empty() {
            request_body.insert("generationConfig".to_string(), json!(generation_config));
        }
    }

    // Construct the GeminiRequest from the updated request body
    // Convert the Map to a Value
    let request = GeminiRequest { contents: messages };
    

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    let response = gemini_request(&url, &api_key, &request, Some(headers)).await?;
    let response_body = response.text().await.map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to read response from Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;
    
    handle_gemini_error(&response_body)
}
