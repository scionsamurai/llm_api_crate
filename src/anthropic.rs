// src/anthropic.rs
use serde::{Deserialize, Serialize};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use crate::errors::GeneralError;
use dotenv::dotenv;

use crate::structs::general::{Message, LlmResponse};
use crate::config::LlmConfig; // <-- Import config

// --- NEW: Added Thinking Config struct ---
#[derive(Debug, Serialize, Clone)]
pub struct ThinkingConfig {
    pub r#type: String, // Always "enabled"
    pub budget_tokens: usize,
}

// --- UPDATED: Added thinking and temperature ---
#[derive(Debug, Serialize, Clone)]
pub struct AnthropicRequest {
    pub model: String,
    pub max_tokens: usize,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicResult {
    pub response: AnthropicMessage,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub content: String,
}

const MODEL: &str = "claude-haiku-4-5";
const DEFAULT_MAX_TOKENS: usize = 4096; // Changed from MAX_TOKENS for clarity

use std::str;

#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub role: String,
    pub content: Vec<Content>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(other)]
    Unknown,
}

// --- UPDATED: Signature now accepts model and config ---
pub async fn call_anthropic(
    messages: Vec<Message>,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("ANTHROPIC_API_KEY").map_err(|_| GeneralError {
        message: "ANTHROPIC API KEY not found in environment variables".to_string(),
    })?;

    let url: &str = "https://api.anthropic.com/v1/messages";

    let mut headers: HeaderMap = HeaderMap::new();

    headers.insert(
        "x-api-key",
        HeaderValue::from_str(&api_key)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(GeneralError {
                    message: format!("Invalid Anthropic API key: {}", e.to_string()),
                })
            })?,
    );

    headers.insert(
        "anthropic-version",
        HeaderValue::from_str("2023-06-01") 
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(GeneralError {
                    message: format!("Failed to set Anthropic version header: {}", e.to_string()),
                })
            })?,
    );

    headers.insert(
        "content-type",
        HeaderValue::from_str("application/json")
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(GeneralError {
                    message: format!("Failed to set content-type header: {}", e.to_string()),
                })
            })?,
    );

    let client = Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            Box::new(GeneralError {
                message: format!("Failed to create HTTP client: {}", e.to_string()),
            })
        })?;

    // Convert "model" roles to "assistant" roles
    let processed_messages = messages.into_iter().map(|mut message| {
        if message.role == "model" {
            message.role = "assistant".to_string();
        }
        message
    }).collect::<Vec<Message>>();

    // --- NEW: Dynamic Model and Config Parsing ---
    let mut request = AnthropicRequest {
        model: model.unwrap_or(MODEL).to_string(),
        max_tokens: DEFAULT_MAX_TOKENS,
        messages: processed_messages,
        thinking: None,
        temperature: None,
    };

    if let Some(cfg) = config {
        // Handle max_tokens first
        if let Some(max_t) = cfg.max_tokens {
            request.max_tokens = max_t as usize;
        }

        // Handle Thinking
        if let Some(budget) = cfg.thinking_budget {
            // Anthropic strictly requires a minimum budget of 1024
            let valid_budget = if budget < 1024 { 1024 } else { budget as usize };
            
            request.thinking = Some(ThinkingConfig {
                r#type: "enabled".to_string(),
                budget_tokens: valid_budget,
            });

            // Anthropic strictly requires max_tokens to be larger than the thinking budget
            // We ensure max_tokens provides at least an extra 1000 tokens for the actual text response
            if request.max_tokens <= valid_budget {
                request.max_tokens = valid_budget + 1024; 
            }
        } else if let Some(temp) = cfg.temperature {
            // Only set temperature if thinking is NOT enabled. 
            // Anthropic throws a 400 error if temperature is provided alongside thinking blocks.
            request.temperature = Some(temp as f32);
        }
    }

    let res = client
        .post(url)
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            Box::new(GeneralError {
                message: format!("Failed to send request to Anthropic API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let status = res.status();
    let rspns_strng = res.text().await.map_err(|e: reqwest::Error| {
        Box::new(GeneralError {
            message: format!("Failed to read response from Anthropic API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    // Catch HTTP errors before JSON parsing to surface API complaints (like bad budgets)
    if !status.is_success() {
        return Err(Box::new(GeneralError {
            message: format!("Anthropic API Error (HTTP {}): {}", status, rspns_strng),
        }));
    }

    let res: AnthropicResponse = serde_json::from_str(&rspns_strng).map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to parse response from Anthropic API: {} | Raw: {}", e.to_string(), rspns_strng),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    // Extract text and reasoning blocks
    let mut text_output = String::new();
    let mut reasoning_output = None;

    for block in res.content {
        match block {
            Content::Text { text } => text_output.push_str(&text),
            Content::Thinking { thinking, .. } => reasoning_output = Some(thinking),
            _ => {} // Ignore redacted thinking or unknown blocks
        }
    }

    Ok(LlmResponse {
        text: text_output,
        reasoning: reasoning_output,
    })
}