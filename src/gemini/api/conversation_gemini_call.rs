use reqwest::header::{HeaderMap, HeaderValue};
use std::env;
use std::time::Duration;
use dotenv::dotenv;
use serde_json::json; 

use crate::errors::{GeneralError, with_retry};
use crate::structs::general::{Content, MessageContent, MessagePart, Part, ImageSource};
use crate::gemini::types::{GeminiRequest, GenerationConfig, Tool, GeminiResponse, GeminiInlineData};
use crate::gemini::request::gemini_request;
use crate::gemini::response::parse_gemini_response;
use crate::config::LlmConfig;

fn map_message_content_to_parts(content: &MessageContent) -> Vec<Part> {
    match content {
        MessageContent::Text(text) => vec![Part {
            text: Some(text.clone()),
            inline_data: None,
            thought: None,
        }],
        MessageContent::Array(parts) => parts.iter().map(|p| {
            if p.r#type == "text" {
                Part {
                    text: p.text.clone(),
                    inline_data: None,
                    thought: None,
                }
            } else if p.r#type == "image_url" {
                if let Some(ImageSource::Base64 { mime_type, data }) = &p.image_url {
                    Part {
                        text: None,
                        inline_data: Some(GeminiInlineData {
                            mime_type: mime_type.clone(),
                            data: data.clone(),
                        }),
                        thought: None,
                    }
                } else {
                    Part { text: Some(format!("Image URL: {:?}", p.image_url)), inline_data: None, thought: None }
                }
            } else {
                Part { text: None, inline_data: None, thought: None }
            }
        }).collect(),
    }
}

pub async fn conversation_gemini_call(
    messages_core: Vec<crate::structs::general::Message>,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<GeminiResponse, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();
    let default_gemini_model: String = env::var("DEFAULT_GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    let model_name = model.unwrap_or(&default_gemini_model);
    let url: String = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        model_name
    );

    let messages: Vec<Content> = messages_core.into_iter().map(|m| Content {
        role: m.role,
        parts: map_message_content_to_parts(&m.content),
    }).collect();

    let mut generation_config_option: Option<GenerationConfig> = None;
    let mut tools_option: Option<Vec<Tool>> = None;

    if let Some(cfg) = config {
        let mut current_gen_config = GenerationConfig {
            temperature: None,
            thinking_budget: None,
        };

        let mut config_has_options = false;

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

        if cfg.grounding_with_search.unwrap_or(false) {
            tools_option = Some(vec![Tool { google_search: Some(json!({})) }]);
        }
    }

    let request = GeminiRequest {
        contents: messages,
        generation_config: generation_config_option,
        tools: tools_option,
    };
    
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    with_retry(|| async {
        let response = gemini_request(&url, &api_key, &request, Some(headers.clone())).await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Box::new(GeneralError {
                message: format!("Gemini API returned error {}: {}", status, body),
            }) as Box<dyn std::error::Error + Send + Sync>);
        }

        let gemini_response: GeminiResponse = parse_gemini_response(response).await?;
        Ok(gemini_response)
    }, 3, Duration::from_secs(1)).await
}