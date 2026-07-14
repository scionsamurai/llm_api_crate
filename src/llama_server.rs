use reqwest::Client;
use std::env;
use std::time::Duration;
use dotenv::dotenv;

use futures::stream::{BoxStream, StreamExt};
use async_stream::stream;
use crate::errors::{GeneralError, with_retry};
use crate::structs::general::{ Message, MessageContent, LlmResponse, LlmChunk, MessagePart, ImageSource };
use crate::structs::openai::{ChatCompletion, EmbeddingRequest};
use crate::openai::StreamResponse;
use crate::models::openai::{APIResponse, ErrorResponse, EmbeddingResponse};
use crate::structs::llama_server::{LlamaCompletionRequest, LlamaCompletionResponse, ImageData};
    
use crate::config::LlmConfig;

fn get_server_url() -> String {
    dotenv().ok(); 
    env::var("LLAMA_SERVER_URL").unwrap_or_else(|_| "http://192.168.0.91:8080".to_string())
}

fn parse_raw_reasoning(raw_text: &str) -> (String, Option<String>) {
    if let Some(start_idx) = raw_text.find("<|channel>thought") {
        if let Some(end_idx) = raw_text.find("<channel|>") {
            let reasoning_start = start_idx + "<|channel>thought".len();
            let reasoning = raw_text[reasoning_start..end_idx].trim().to_string();
            let mut text = raw_text[..start_idx].to_string();
            text.push_str(&raw_text[end_idx + "<channel|>".len()..]);
            return (text.trim().to_string(), if reasoning.is_empty() { None } else { Some(reasoning) });
        }
    }
    if let Some(start_idx) = raw_text.find("<think>") {
        if let Some(end_idx) = raw_text.find("</think>") {
            let reasoning_start = start_idx + "<think>".len();
            let reasoning = raw_text[reasoning_start..end_idx].trim().to_string();
            let mut text = raw_text[..start_idx].to_string();
            text.push_str(&raw_text[end_idx + "</think>".len()..]);
            return (text.trim().to_string(), if reasoning.is_empty() { None } else { Some(reasoning) });
        }
    }
    (raw_text.to_string(), None)
}

pub async fn call_llama_openai_compat(
    messages: Vec<Message>,
    model: Option<&str>, 
    config: Option<&LlmConfig>,
) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> {
    let base_url = if let Some(cfg) = config { cfg.server_url.clone().unwrap_or_else(get_server_url) } else { get_server_url() };
    let url = format!("{}/v1/chat/completions", base_url);
    let model_name = model.unwrap_or("gemma-4-26b").to_string();
    let mut processed_messages = messages;

    if let Some(cfg) = config {
        if cfg.thinking_budget.is_some() {
            let has_system = processed_messages.first().map(|m| m.role == "system").unwrap_or(false);
            if has_system {
                let first = processed_messages.first_mut().unwrap();
                if let MessageContent::Text(ref mut text) = first.content {
                    if !text.starts_with("<|think|>") { *text = format!("<|think|>\n{}", text); }
                }
            } else {
                processed_messages.insert(0, Message { role: "system".to_string(), content: MessageContent::Text("<|think|>".to_string()) });
            }
        }
    }

    let mut request_body = ChatCompletion {
        model: model_name, 
        messages: processed_messages.into_iter().map(crate::openai::transform_message).collect(),
        ..Default::default()
    };

    if let Some(cfg) = config {
        request_body.temperature = cfg.temperature.map(|t| t as f32);
        request_body.stream = cfg.stream;
        request_body.max_tokens = cfg.max_tokens;
        request_body.stop = cfg.stop.clone();
        request_body.top_k = cfg.top_k;
        request_body.top_p = cfg.top_p;
        request_body.cache_prompt = cfg.cache_prompt;
        request_body.response_format = cfg.json_schema.clone();
    }

    let client = Client::new();
    with_retry(|| async {
        let res = client.post(&url).json(&request_body).send().await.map_err(|e| Box::new(GeneralError { message: e.to_string() }))?;
        let status = res.status();
        let rspns_strng = res.text().await.unwrap_or_default();
        if !status.is_success() { return Err(Box::new(GeneralError { message: format!("Llama Server HTTP {}: {}", status, rspns_strng) })); }
        match serde_json::from_str::<APIResponse>(&rspns_strng) {
            Ok(api_response) => {
                let message = &api_response.choices[0].message;
                let (final_text, final_reasoning) = if let Some(reasoning) = &message.reasoning_content {
                    (message.content.clone(), Some(reasoning.clone()))
                } else {
                    parse_raw_reasoning(&message.content)
                };
                Ok(LlmResponse { text: final_text, reasoning: final_reasoning })
            },
            Err(_) => Err(Box::new(GeneralError { message: "Failed to parse Llama JSON".into() })),
        }
    }, 3, Duration::from_secs(1)).await
}

pub async fn call_llama_stream(
    messages: Vec<Message>,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<BoxStream<'static, Result<LlmChunk, Box<dyn std::error::Error + Send + Sync>>>, Box<dyn std::error::Error + Send + Sync>> {
    let base_url = if let Some(cfg) = config { cfg.server_url.clone().unwrap_or_else(get_server_url) } else { get_server_url() };
    let url = format!("{}/v1/chat/completions", base_url);
    let model_name = model.unwrap_or("gemma-4-26b").to_string();
    let mut processed_messages = messages;

    if let Some(cfg) = config {
        if cfg.thinking_budget.is_some() {
            let has_system = processed_messages.first().map(|m| m.role == "system").unwrap_or(false);
            if has_system {
                let first = processed_messages.first_mut().unwrap();
                if let MessageContent::Text(ref mut text) = first.content {
                    if !text.starts_with("<|think|>") { *text = format!("<|think|>\n{}", text); }
                }
            } else {
                processed_messages.insert(0, Message { role: "system".to_string(), content: MessageContent::Text("<|think|>".to_string()) });
            }
        }
    }

    let mut request_body = ChatCompletion {
        model: model_name, 
        messages: processed_messages.into_iter().map(crate::openai::transform_message).collect(),
        stream: Some(true),
        ..Default::default()
    };

    if let Some(cfg) = config {
        request_body.temperature = cfg.temperature.map(|t| t as f32);
        request_body.max_tokens = cfg.max_tokens;
        request_body.stop = cfg.stop.clone();
        request_body.top_k = cfg.top_k;
        request_body.top_p = cfg.top_p;
        request_body.response_format = cfg.json_schema.clone();
    }

    let client = Client::new();
    let res = client.post(&url).json(&request_body).send().await?;
    if !res.status().is_success() {
        let err_text = res.text().await?;
        return Err(Box::new(GeneralError { message: format!("Llama Server Stream Error: {}", err_text) }));
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
                        if line == "data: [DONE]" { yield Ok(LlmChunk::Done); return; }
                        if line.starts_with("data: ") {
                            let json_str = &line[6..];
                            if let Ok(parsed) = serde_json::from_str::<StreamResponse>(json_str) {
                                if let Some(choice) = parsed.choices.first() {
                                    if let Some(content) = &choice.delta.content { yield Ok(LlmChunk::Text(content.clone())); }
                                    if let Some(reasoning) = &choice.delta.reasoning_content { yield Ok(LlmChunk::Reasoning(reasoning.clone())); }
                                }
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

pub async fn call_llama_legacy(
    content: MessageContent,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> {
    let base_url = get_server_url();
    let url = format!("{}/completion", base_url);

    let mut processed_prompt = content.extract_text();
    let mut image_data = Vec::new();

    if let MessageContent::Array(parts) = content {
        for (idx, part) in parts.iter().enumerate() {
            if part.r#type == "image_url" {
                if let Some(ImageSource::Base64 { data, .. }) = &part.image_url {
                    image_data.push(ImageData { id: idx as u32, data: data.clone() });
                }
            }
        }
    }

    if let Some(cfg) = config {
        if cfg.thinking_budget.is_some() && !processed_prompt.starts_with("<|think|>") {
            processed_prompt = format!("<|think|>\n{}", processed_prompt);
        }
    }

    let mut request_body = LlamaCompletionRequest {
        model: model.map(|m| m.to_string()), 
        prompt: processed_prompt,
        n_predict: None,
        temperature: None,
        top_k: None,
        top_p: None,
        stream: None,
        stop: None,
        cache_prompt: None,
        image_data: if image_data.is_empty() { None } else { Some(image_data) },
    };

    if let Some(cfg) = config {
        request_body.n_predict = cfg.max_tokens;
        request_body.temperature = cfg.temperature;
        request_body.top_k = cfg.top_k;
        request_body.top_p = cfg.top_p;
        request_body.stream = cfg.stream;
        request_body.stop = cfg.stop.clone();
        request_body.cache_prompt = cfg.cache_prompt;
    }

    let client = Client::new();
    with_retry(|| async {
        let res = client.post(&url).json(&request_body).send().await.map_err(|e| Box::new(GeneralError { message: e.to_string() }))?;
        let status = res.status();
        let rspns_strng = res.text().await.unwrap_or_default();
        if !status.is_success() { return Err(Box::new(GeneralError { message: format!("Llama Legacy HTTP {}: {}", status, rspns_strng) })); }
        let parsed: LlamaCompletionResponse = serde_json::from_str(&rspns_strng).map_err(|e| Box::new(GeneralError { message: e.to_string() }))?;
        let (final_text, final_reasoning) = parse_raw_reasoning(&parsed.content);
        Ok(LlmResponse { text: final_text, reasoning: final_reasoning })
    }, 3, Duration::from_secs(1)).await
}

pub async fn call_llama_embeddings(
    input: String,
    model: Option<&str>,
    dimensions: Option<u32>,
    config: Option<&LlmConfig>,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let base_url = if let Some(cfg) = config { cfg.server_url.clone().unwrap_or_else(get_server_url) } else { get_server_url() };
    let url = format!("{}/v1/embeddings", base_url);
    let client = Client::new();
    let embedding_request = EmbeddingRequest {
        model: model.unwrap_or("embedder").to_string(),
        input,
        dimensions,
        encoding_format: "float".to_string(),
    };
    with_retry(|| async {
        let res = client.post(&url).json(&embedding_request).send().await.map_err(|e| Box::new(GeneralError { message: e.to_string() }))?;
        let status = res.status();
        let rspns_strng = res.text().await.unwrap_or_default();
        if !status.is_success() { return Err(Box::new(GeneralError { message: format!("Llama Embeddings HTTP {}: {}", status, rspns_strng) })); }
        match serde_json::from_str::<EmbeddingResponse>(&rspns_strng) {
            Ok(api_response) => {
                if let Some(data) = api_response.data.first() { Ok(data.embedding.clone()) } else { Err(Box::new(GeneralError { message: "No embedding data".into() })) }
            },
            Err(e) => Err(Box::new(GeneralError { message: e.to_string() })),
        }
    }, 3, Duration::from_secs(1)).await
}
