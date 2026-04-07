// src/llama_server.rs
use reqwest::Client;
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::structs::general::Message;
use crate::structs::openai::ChatCompletion;
use crate::models::openai::{APIResponse, ErrorResponse};
use crate::structs::llama_server::{LlamaCompletionRequest, LlamaCompletionResponse};
use crate::config::LlmConfig;

fn get_server_url() -> String {
    dotenv().ok(); // <-- ADD THIS LINE
    env::var("LLAMA_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string())
}

pub async fn call_llama_openai_compat(
    messages: Vec<Message>,
    config: Option<&LlmConfig>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let base_url = get_server_url();
    let url = format!("{}/v1/chat/completions", base_url);

    let mut request_body = ChatCompletion {
        model: "llama-server".to_string(),
        messages,
        temperature: None,
        stream: None,
        max_tokens: None,
        stop: None,
        top_k: None,
        top_p: None,
        cache_prompt: None,
        response_format: None,
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
    let res = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| {
            Box::new(GeneralError {
                message: format!("Failed to send request to Llama Server (OpenAI compat): {}", e),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let status = res.status();
    let rspns_strng = res.text().await.unwrap_or_default();

    // Check for HTTP errors BEFORE parsing
    if !status.is_success() {
        return Err(Box::new(GeneralError {
            message: format!("Llama Server returned HTTP {}: {}", status, rspns_strng),
        }) as Box<dyn std::error::Error + Send + Sync>);
    }

    match serde_json::from_str::<APIResponse>(&rspns_strng) {
        Ok(api_response) => Ok(api_response.choices[0].message.content.clone()),
        Err(_) => {
            match serde_json::from_str::<ErrorResponse>(&rspns_strng) {
                Ok(err) => Err(Box::new(GeneralError {
                    message: format!("Llama Server API Error: {}", err.error.message),
                }) as Box<dyn std::error::Error + Send + Sync>),
                Err(e) => Err(Box::new(GeneralError {
                    message: format!("Failed to parse JSON response: {} - Raw: {}", e, rspns_strng),
                }) as Box<dyn std::error::Error + Send + Sync>),
            }
        }
    }
}

pub async fn call_llama_legacy(
    prompt: String,
    config: Option<&LlmConfig>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let base_url = get_server_url();
    let url = format!("{}/completion", base_url);

    let mut request_body = LlamaCompletionRequest {
        prompt,
        n_predict: None,
        temperature: None,
        top_k: None,
        top_p: None,
        stream: None,
        stop: None,
        cache_prompt: None,
        image_data: None,
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
    let res = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| {
            Box::new(GeneralError {
                message: format!("Failed to send request to Llama Server (Legacy): {}", e),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let status = res.status();
    let rspns_strng = res.text().await.unwrap_or_default();

    // Check for HTTP errors BEFORE parsing
    if !status.is_success() {
        return Err(Box::new(GeneralError {
            message: format!("Llama Server returned HTTP {}: {}", status, rspns_strng),
        }) as Box<dyn std::error::Error + Send + Sync>);
    }

    let parsed: LlamaCompletionResponse = serde_json::from_str(&rspns_strng).map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to parse legacy JSON: {} - Raw: {}", e, rspns_strng),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    Ok(parsed.content)
}