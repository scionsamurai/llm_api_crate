// src/anthropic.rs
use serde::{Deserialize, Serialize};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use crate::errors::GeneralError;
use dotenv::dotenv;

use futures::stream::{BoxStream, StreamExt};
use async_stream::stream;
use crate::structs::general::{Message, LlmResponse, LlmChunk, MessagePart, ImageSource};
use crate::config::LlmConfig; // <-- Import config

// --- NEW: Added Thinking Config struct ---
#[derive(Debug, Serialize, Clone)]
pub struct ThinkingConfig {
    pub r#type: String, // Always "enabled"
    pub budget_tokens: usize,
}

#[derive(Debug, Serialize, Clone)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String, // "base64"
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text { text: String },
    Image { source: AnthropicImageSource },
}

#[derive(Debug, Serialize, Clone)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicResponseBody {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicResult {
    pub response: AnthropicResponseBody,
}

#[derive(Debug, Serialize, Clone)]
pub struct AnthropicRequest {
    pub model: String,
    pub max_tokens: usize,
    pub messages: Vec<AnthropicMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub role: String,
    pub content: Vec<Content>,
}

const MODEL: &str = "claude-haiku-4-5";
const DEFAULT_MAX_TOKENS: usize = 4096;

use std::str;

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

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum AnthropicEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: serde_json::Value },
    #[serde(rename = "content_block_start")]
    ContentBlockStart { index: u32 },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: AnthropicDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },
    #[serde(rename = "message_stop")]
    MessageStop { stop_reason: String },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    pub text: Option<String>,
    pub thinking: Option<String>,
}

fn map_to_anthropic_messages(messages: Vec<Message>) -> Vec<AnthropicMessage> {
    messages.into_iter().map(|msg| {
        let role = if msg.role == "model" { "assistant".to_string() } else { msg.role };
        let content = msg.content.as_parts().into_iter().map(|part| {
            if part.r#type == "text" {
                AnthropicContentBlock::Text { text: part.text.unwrap_or_default() }
            } else if part.r#type == "image_url" {
                if let Some(ImageSource::Base64 { media_type, data }) = part.image_url {
                    AnthropicContentBlock::Image {
                        source: AnthropicImageSource {
                            source_type: "base64".to_string(),
                            media_type,
                            data,
                        }
                    }
                } else {
                    AnthropicContentBlock::Text { text: "".to_string() }
                }
            } else {
                AnthropicContentBlock::Text { text: "".to_string() }
            }
        }).collect();
        
        AnthropicMessage { role, content }
    }).collect()
}

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

    let anthropic_messages = map_to_anthropic_messages(messages);

    let mut request = AnthropicRequest {
        model: model.unwrap_or(MODEL).to_string(),
        max_tokens: DEFAULT_MAX_TOKENS,
        messages: anthropic_messages,
        stream: false,
        thinking: None,
        temperature: None,
    };

    if let Some(cfg) = config {
        if let Some(max_t) = cfg.max_tokens {
            request.max_tokens = max_t as usize;
        }

        if let Some(budget) = cfg.thinking_budget {
            let valid_budget = if budget < 1024 { 1024 } else { budget as usize };
            
            request.thinking = Some(ThinkingConfig {
                r#type: "enabled".to_string(),
                budget_tokens: valid_budget,
            });

            if request.max_tokens <= valid_budget {
                request.max_tokens = valid_budget + 1024; 
            }
        } else if let Some(temp) = cfg.temperature {
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

    let mut text_output = String::new();
    let mut reasoning_output = None;

    for block in res.content {
        match block {
            Content::Text { text } => text_output.push_str(&text),
            Content::Thinking { thinking, .. } => reasoning_output = Some(thinking),
            _ => {}
        }
    }

    Ok(LlmResponse {
        text: text_output,
        reasoning: reasoning_output,
    })
}

pub async fn call_anthropic_stream(
    messages: Vec<Message>,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<BoxStream<'static, Result<LlmChunk, Box<dyn std::error::Error + Send + Sync>>>, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();
    let api_key = env::var("ANTHROPIC_API_KEY").map_err(|_| GeneralError { message: "ANTHROPIC API KEY not found".into() })?;
    let url = "https://api.anthropic.com/v1/messages";

    let mut headers = HeaderMap::new();
    headers.insert("x-api-key", HeaderValue::from_str(&api_key)?);
    headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
    headers.insert("content-type", HeaderValue::from_static("application/json"));

    let client = Client::builder().default_headers(headers).build()?;

    let anthropic_messages = map_to_anthropic_messages(messages);

    let mut request = AnthropicRequest {
        model: model.unwrap_or(MODEL).to_string(),
        max_tokens: DEFAULT_MAX_TOKENS,
        messages: anthropic_messages,
        stream: true,
        thinking: None,
        temperature: None,
    };

    if let Some(cfg) = config {
        if let Some(max_t) = cfg.max_tokens { request.max_tokens = max_t as usize; }
        if let Some(budget) = cfg.thinking_budget {
            let valid_budget = if budget < 1024 { 1024 } else { budget as usize };
            request.thinking = Some(ThinkingConfig { r#type: "enabled".to_string(), budget_tokens: valid_budget });
            if request.max_tokens <= valid_budget { request.max_tokens = valid_budget + 1024; }
        } else if let Some(temp) = cfg.temperature {
            request.temperature = Some(temp as f32);
        }
    }

    let res = client.post(url).json(&request).send().await?;
    if !res.status().is_success() {
        let err_text = res.text().await?;
        return Err(Box::new(GeneralError { message: format!("Anthropic Stream Error: {}", err_text) }));
    }
    
    let byte_stream = res.bytes_stream();

    let output_stream = stream! {
        let mut buffer = String::new();
        let mut bytes_stream = byte_stream;

        while let Some(item) = bytes_stream.next().await {
            match item {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));
                    while let Some(newline_idx) = buffer.find('\n') {
                        let line = buffer.drain(..newline_idx + 1).collect::<String>().trim().to_string();
                        if line.is_empty() { continue; }
                        if line == "data: [DONE]" {
                            yield Ok(LlmChunk::Done);
                            return; 
                        }
                        if line.starts_with("data: ") {
                            let json_str = &line[6..];

                            if let Ok(event) = serde_json::from_str::<AnthropicEvent>(json_str) {
                                match event {
                                    AnthropicEvent::ContentBlockDelta { delta } => {
                                        if let Some(t) = delta.text { yield Ok(LlmChunk::Text(t)); }
                                        if let Some(th) = delta.thinking { yield Ok(LlmChunk::Reasoning(th)); }
                                    }
                                    AnthropicEvent::MessageStop { .. } => {
                                        yield Ok(LlmChunk::Done);
                                        return;
                                    }
                                    _ => {}
                                }
                            } else {
                                eprintln!("DEBUG: Failed to parse Anthropic event: {}", json_str);
                            }
                        }
                    }
                }
                Err(e) => yield Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
            }
        }
    };

    Ok(Box::pin(output_stream))
}