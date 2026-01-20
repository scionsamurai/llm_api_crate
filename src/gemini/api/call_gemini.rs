// src/gemini/api/call_gemini.rs
use std::env;
use dotenv::dotenv;
use serde_json::{json, Map, Value}; // Map and Value are less needed now

use crate::errors::GeneralError;
use crate::structs::general::{ Message, Content, Part };
use crate::gemini::types::{GeminiRequest, GenerationConfig, Tool}; // Import new types
use crate::gemini::request::gemini_request;
use crate::gemini::response::parse_gemini_response;
use crate::gemini::types::GeminiResponse;
use crate::config::LlmConfig;

const DEFAULT_GEMINI_MODEL: &str = "gemini-2.0-flash";

pub async fn call_gemini(
    messages: Vec<Message>,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<GeminiResponse, Box<dyn std::error::Error + Send + Sync>> { // CHANGE RETURN TYPE
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

    let mut generation_config_option: Option<GenerationConfig> = None;
    let mut tools_option: Option<Vec<Tool>> = None;

    if let Some(cfg) = config {
        let mut current_gen_config = GenerationConfig {
            temperature: None,
            thinking_budget: None,
        };

        let mut config_has_options = false; // Flag to check if any generation config is set

        if let Some(thinking_budget) = cfg.thinking_budget {
            current_gen_config.thinking_budget = Some(thinking_budget);
            config_has_options = true;
        }

        if let Some(temp) = cfg.temperature {
            current_gen_config.temperature = Some(temp);
            config_has_options = true;
        }

        if config_has_options {
            generation_config_option = Some(current_gen_config);
        }

        // Add "tools" for grounding with Google Search
        if cfg.grounding_with_search.unwrap_or(false) {
            tools_option = Some(vec![Tool { google_search: Some(json!({})) }]);
        }
    }

    // Construct the GeminiRequest struct with all relevant fields
    let request = GeminiRequest {
        contents,
        generation_config: generation_config_option,
        tools: tools_option,
    };

    let response = gemini_request(&url, &api_key, &request, None).await?;
    let gemini_response: GeminiResponse = parse_gemini_response(response).await?;

    Ok(gemini_response) // Return the full GeminiResponse
}