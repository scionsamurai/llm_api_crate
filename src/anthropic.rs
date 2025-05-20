// src/anthropic.rs
use serde::{Deserialize, Serialize};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use crate::errors::GeneralError;
use dotenv::dotenv;

use crate::structs::general::Message;

// https://docs.anthropic.com/claude/reference/getting-started-with-the-api

#[derive(Debug, Serialize, Clone)]
pub struct AnthropicRequest {
    pub model: String,
    pub max_tokens: usize,
    pub messages: Vec<Message>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicResult {
    pub response: AnthropicMessage,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub content: String,
}

const MODEL: &str = "claude-3-7-sonnet-20250219";
const MAX_TOKENS: usize = 4096;

use std::str;

#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub role: String,
    pub content: Vec<Content>,
    // Other fields omitted for brevity
}

#[derive(Debug, Deserialize)]
pub struct Content {
    pub text: String,
    // Other fields omitted for brevity
}

pub async fn call_anthropic(
    messages: Vec<Message>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
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

    let request: AnthropicRequest = AnthropicRequest {
        model: MODEL.to_string(),
        max_tokens: MAX_TOKENS,
        messages: processed_messages,
    };

    let res = client
        .post(url)
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            println!("{:?}", e);
            Box::new(GeneralError {
                message: format!("Failed to send request to Anthropic API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    // Rest of the function remains the same
    let rspns_strng = res.text().await.map_err(|e: reqwest::Error| {
        Box::new(GeneralError {
            message: format!("Failed to read response from Anthropic API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    // println!("Raw response: {}", rspns_strng);

    if rspns_strng.contains("invalid x-api-key") {
        return Err(Box::new(GeneralError {
            message: "Invalid Anthropic API key. Please check your API key and try again."
                .to_string(),
        }));
    }

    let res: AnthropicResponse = serde_json::from_str(&rspns_strng).map_err(|e| {
        println!("AnthropicResponse res: {:?}", e);
        Box::new(GeneralError {
            message: format!(
                "Failed to parse response from Anthropic API: {}",
                e.to_string()
            ),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    Ok(res.content[0].text.clone())
}
