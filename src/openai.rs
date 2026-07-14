use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use dotenv::dotenv;

use futures::stream::{BoxStream, StreamExt};
use async_stream::stream;
use serde::Deserialize;

use crate::errors::GeneralError;
use crate::structs::general::{Message, MessageContent, MessagePart, ImageSource, LlmResponse, LlmChunk}; 
use crate::structs::openai::{ChatCompletion, EmbeddingRequest, OpenAiMessage, OpenAiContent, OpenAiContentBlock, OpenAiImageUrl};
use crate::models::openai::{APIResponse, ErrorResponse, EmbeddingResponse};
use crate::config::LlmConfig;

const CHAT_COMPLETION_MODEL: &str = "gpt-4o";
const EMBEDDING_MODEL: &str = "text-embedding-3-small";
const EMBEDDING_ENCODING_FORMAT: &str = "float";

pub fn transform_message(msg: Message) -> OpenAiMessage {
    let content = match &msg.content {
        MessageContent::Text(text) => OpenAiContent::Text(text.clone()),
        MessageContent::Array(parts) => {
            let blocks = parts.iter().map(|p| {
                if p.r#type == "text" {
                    OpenAiContentBlock::Text { text: p.text.clone().unwrap_or_default() }
                } else if p.r#type == "image_url" {
                    OpenAiContentBlock::Image { 
                        image_url: OpenAiImageUrl { 
                            url: p.image_url.as_ref().map(|s| s.to_data_url()).unwrap_or_default() 
                        } 
                    }
                } else {
                    OpenAiContentBlock::Text { text: "".to_string() }
                }
            }).collect();
            OpenAiContent::Array(blocks)
        }
    };
    OpenAiMessage { role: msg.role, content }
}

#[derive(Debug, Deserialize)]
pub struct StreamResponse {
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
pub struct StreamChoice {
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StreamDelta {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
}

pub async fn call_gpt_stream(
    messages: Vec<Message>,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<BoxStream<'static, Result<LlmChunk, Box<dyn std::error::Error + Send + Sync>>>, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key = env::var("OPEN_AI_KEY").expect("OPEN AI KEY not found");
    let api_org = env::var("OPEN_AI_ORG").unwrap_or_default();
    let url = "https://api.openai.com/v1/chat/completions";

    let mut headers = HeaderMap::new();
    headers.insert("authorization", HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());
    if !api_org.is_empty() {
        headers.insert("OpenAI-Organization", HeaderValue::from_str(&api_org).unwrap());
    }

    let client = Client::builder().default_headers(headers).build()?;
    let model_name = model.unwrap_or(CHAT_COMPLETION_MODEL).to_string();
    let is_reasoning_model = model_name.starts_with("o1") || model_name.starts_with("o3");

    let mut chat_completion = ChatCompletion {
        model: model_name,
        messages: messages.into_iter().map(transform_message).collect(),
        temperature: None,
        stream: Some(true),
        max_tokens: None,
        max_completion_tokens: None,
        stop: None,
        top_k: None,
        top_p: None,
        cache_prompt: None,
        response_format: None,
    };

    if let Some(cfg) = config {
        if is_reasoning_model {
            chat_completion.max_completion_tokens = cfg.max_tokens;
        } else {
            chat_completion.temperature = cfg.temperature.map(|t| t as f32);
            chat_completion.max_tokens = cfg.max_tokens;
        }
        chat_completion.stop = cfg.stop.clone();
        chat_completion.top_k = cfg.top_k;
        chat_completion.top_p = cfg.top_p;
        chat_completion.response_format = cfg.json_schema.clone();
    }

    let res = client.post(url).json(&chat_completion).send().await?;
    if !res.status().is_success() {
        let err_text = res.text().await?;
        return Err(Box::new(GeneralError { message: format!("OpenAI Stream Error: {}", err_text) }));
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
                            match serde_json::from_str::<StreamResponse>(json_str) {
                                Ok(parsed) => {
                                    if let Some(choice) = parsed.choices.first() {
                                        if let Some(content) = &choice.delta.content {
                                            yield Ok(LlmChunk::Text(content.clone()));
                                        }
                                        if let Some(reasoning) = &choice.delta.reasoning_content {
                                            yield Ok(LlmChunk::Reasoning(reasoning.clone()));
                                        }
                                    }
                                },
                                Err(_) => {}
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

pub async fn call_gpt(
    messages: Vec<Message>,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> { 
    dotenv().ok();
    let api_key: String = env::var("OPEN_AI_KEY").expect("OPEN AI KEY not found");
    let api_org: String = env::var("OPEN_AI_ORG").unwrap_or_default();
    let url: &str = "https://api.openai.com/v1/chat/completions";

    let mut headers: HeaderMap = HeaderMap::new();
    headers.insert("authorization", HeaderValue::from_str(&format!("Bearer {}", api_key)).map_err(|e| Box::new(e))?);
    if !api_org.is_empty() {
        headers.insert("OpenAI-Organization", HeaderValue::from_str(api_org.as_str()).map_err(|e| Box::new(e))?);
    }

    let client = Client::builder().default_headers(headers).build()?;
    let model_name = model.unwrap_or(CHAT_COMPLETION_MODEL).to_string();
    let is_reasoning_model = model_name.starts_with("o1") || model_name.starts_with("o3");

    let mut chat_completion = ChatCompletion {
        model: model_name,
        messages: messages.into_iter().map(transform_message).collect(),
        temperature: None,
        stream: None,
        max_tokens: None,
        max_completion_tokens: None,
        stop: None,
        top_k: None,
        top_p: None,
        cache_prompt: None,
        response_format: None,
    };

    if let Some(cfg) = config {
        if is_reasoning_model {
            chat_completion.max_completion_tokens = cfg.max_tokens;
            if let Some(temp) = cfg.temperature {
                if (temp - 1.0).abs() < f64::EPSILON { chat_completion.temperature = Some(1.0); }
            }
        } else {
            chat_completion.temperature = cfg.temperature.map(|t| t as f32);
            chat_completion.max_tokens = cfg.max_tokens;
        }
        chat_completion.stream = cfg.stream;
        chat_completion.stop = cfg.stop.clone();
        chat_completion.top_k = cfg.top_k;
        chat_completion.top_p = cfg.top_p;
        chat_completion.response_format = cfg.json_schema.clone();
    }

    let res = client.post(url).json(&chat_completion).send().await.map_err(|e| Box::new(GeneralError { message: e.to_string() }))?;
    let status = res.status();
    let rspns_strng = res.text().await.map_err(|e| Box::new(GeneralError { message: e.to_string() }))?;

    if !status.is_success() {
        return Err(Box::new(GeneralError { message: format!("OpenAI API Error (HTTP {}): {}", status, rspns_strng) }));
    }

    match serde_json::from_str::<APIResponse>(&rspns_strng) {
        Ok(api_response) => {
            Ok(LlmResponse {
                text: api_response.choices[0].message.content.clone(),
                reasoning: api_response.choices[0].message.reasoning_content.clone(),
            })
        },
        Err(e) => Err(Box::new(GeneralError { message: format!("Failed to parse response: {} - Raw: {}", e, rspns_strng) })),
    }
}

pub async fn get_embedding(
    input: String,
    model: Option<&str>,
    dimensions: Option<u32>,
    _config: Option<&LlmConfig>,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();
    let api_key: String = env::var("OPEN_AI_KEY").expect("OPEN AI KEY not found");
    let api_org: String = env::var("OPEN_AI_ORG").unwrap_or_default();
    let url: &str = "https://api.openai.com/v1/embeddings";

    let mut headers: HeaderMap = HeaderMap::new();
    headers.insert("authorization", HeaderValue::from_str(&format!("Bearer {}", api_key)).map_err(|e| Box::new(e))?);
    if !api_org.is_empty() {
        headers.insert("OpenAI-Organization", HeaderValue::from_str(api_org.as_str()).map_err(|e| Box::new(e))?);
    }
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    let client = Client::builder().default_headers(headers).build()?;
    let embedding_request = EmbeddingRequest {
        model: model.unwrap_or(EMBEDDING_MODEL).to_string(),
        input,
        dimensions,
        encoding_format: EMBEDDING_ENCODING_FORMAT.to_string(),
    };

    let res = client.post(url).json(&embedding_request).send().await.map_err(|e| Box::new(GeneralError { message: e.to_string() }))?;
    let rspns_strng = res.text().await.map_err(|e| Box::new(GeneralError { message: e.to_string() }))?;

    match serde_json::from_str::<EmbeddingResponse>(&rspns_strng) {
        Ok(api_response) => {
            if let Some(data) = api_response.data.first() {
                Ok(data.embedding.clone())
            } else {
                Err(Box::new(GeneralError { message: "No embedding data found".to_string() }))
            }
        },
        Err(_) => {
            match serde_json::from_str::<ErrorResponse>(&rspns_strng) {
                Ok(err) => Err(Box::new(GeneralError { message: err.error.message })),
                Err(e) => Err(Box::new(GeneralError { message: format!("Failed to parse error: {}", e) })),
            }
        }
    }
}
