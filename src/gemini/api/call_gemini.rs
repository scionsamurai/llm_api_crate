// src/gemini/api/call_gemini.rs
use std::env;
use dotenv::dotenv;
use serde_json::{json, Map, Value};

use crate::errors::GeneralError;
use crate::structs::general::{ Message, Content, Part };
use crate::gemini::types::GeminiRequest;
use crate::gemini::request::gemini_request;
use crate::gemini::response::parse_gemini_response;
use crate::gemini::types::GeminiResponse;
use crate::config::LlmConfig; // Import LlmConfig

const DEFAULT_GEMINI_MODEL: &str = "gemini-2.0-flash"; // Or "gemini-2.0-flash" if that's your intended default

pub async fn call_gemini(
    messages: Vec<Message>,
    model: Option<&str>,
    config: Option<&LlmConfig>, // NEW: optional config parameter
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GEMINI API KEY not found in environment variables".to_string(),
    })?;

    let model_name = model.unwrap_or(DEFAULT_GEMINI_MODEL);
    let url: String = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        model_name
    );

    let contents: Vec<Content> = messages
        .iter()
        .map(|msg| Content {
            role: msg.role.clone(),
            parts: vec![Part {
                text: msg.content.clone(),
            }],
        })
        .collect();

    let mut request_body: Map<String, Value> = Map::new();
    request_body.insert("contents".to_string(), json!(contents));

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

    let request = GeminiRequest { contents };

    let response = gemini_request(&url, &api_key, &request, None).await?;
    let gemini_response: GeminiResponse = parse_gemini_response(response).await?;

    Ok(gemini_response.candidates[0].content.parts[0].text.clone())
}