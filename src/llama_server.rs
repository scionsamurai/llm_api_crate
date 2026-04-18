// src/llama_server.rs
use reqwest::Client;
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::structs::general::{ Message, MessageContent, LlmResponse };
use crate::structs::openai::{ChatCompletion, EmbeddingRequest};
use crate::models::openai::{APIResponse, ErrorResponse, EmbeddingResponse};
use crate::structs::llama_server::{LlamaCompletionRequest, LlamaCompletionResponse};
    
use crate::config::LlmConfig;

fn get_server_url() -> String {
    dotenv().ok(); 
    env::var("LLAMA_SERVER_URL").unwrap_or_else(|_| "http://192.168.0.91:8080".to_string())
}

/// Helper function to manually extract Gemma 4 or DeepSeek reasoning tags 
/// from raw text, just in case the server doesn't parse them into `reasoning_content`.
fn parse_raw_reasoning(raw_text: &str) -> (String, Option<String>) {
    // 1. Check for Gemma 4 tags
    if let Some(start_idx) = raw_text.find("<|channel>thought") {
        if let Some(end_idx) = raw_text.find("<channel|>") {
            let reasoning_start = start_idx + "<|channel>thought".len();
            let reasoning = raw_text[reasoning_start..end_idx].trim().to_string();
            
            let mut text = raw_text[..start_idx].to_string();
            text.push_str(&raw_text[end_idx + "<channel|>".len()..]);
            
            let final_reasoning = if reasoning.is_empty() { None } else { Some(reasoning) };
            return (text.trim().to_string(), final_reasoning);
        }
    }

    // 2. Check for DeepSeek tags (fallback/universal support)
    if let Some(start_idx) = raw_text.find("<think>") {
        if let Some(end_idx) = raw_text.find("</think>") {
            let reasoning_start = start_idx + "<think>".len();
            let reasoning = raw_text[reasoning_start..end_idx].trim().to_string();
            
            let mut text = raw_text[..start_idx].to_string();
            text.push_str(&raw_text[end_idx + "</think>".len()..]);
            
            let final_reasoning = if reasoning.is_empty() { None } else { Some(reasoning) };
            return (text.trim().to_string(), final_reasoning);
        }
    }

    // If no tags found, return the text as-is
    (raw_text.to_string(), None)
}

pub async fn call_llama_openai_compat(
    messages: Vec<Message>,
    model: Option<&str>, 
    config: Option<&LlmConfig>,
) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> {
    let base_url = get_server_url();
    let url = format!("{}/v1/chat/completions", base_url);
    let model_name = model.unwrap_or("gemma-4-26b").to_string();

    let mut processed_messages = messages;

    // --- Inject <|think|> trigger for Gemma 4 ---
    if let Some(cfg) = config {
        if cfg.thinking_budget.is_some() {
            let has_system = processed_messages.first().map(|m| m.role == "system").unwrap_or(false);
            if has_system {
                // Prepend to existing system message
                let first = processed_messages.first_mut().unwrap();
                if let MessageContent::Text(ref mut text) = first.content {
                    if !text.starts_with("<|think|>") {
                        *text = format!("<|think|>\n{}", text);
                    }
                }
            } else {
                // Insert a new system message if one doesn't exist
                processed_messages.insert(0, Message {
                    role: "system".to_string(),
                    content: MessageContent::Text("<|think|>".to_string()),
                });
            }
        }
    }
    // println!("Processed Messages for Llama Server:\n{:#?}", processed_messages);

    let mut request_body = ChatCompletion {
        model: model_name, 
        messages: processed_messages,
        temperature: None,
        stream: None,
        max_tokens: None,
        stop: None,
        top_k: None,
        top_p: None,
        cache_prompt: None,
        response_format: None,
        max_completion_tokens: None,
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
                message: format!("Failed to send request to Llama Server: {}", e),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let status = res.status();
    let rspns_strng = res.text().await.unwrap_or_default();

    if !status.is_success() {
        return Err(Box::new(GeneralError {
            message: format!("Llama Server returned HTTP {}: {}", status, rspns_strng),
        }) as Box<dyn std::error::Error + Send + Sync>);
    }

    match serde_json::from_str::<APIResponse>(&rspns_strng) {
        Ok(api_response) => {
            let message = &api_response.choices[0].message;
            let raw_text = &message.content;
            
            // If the server native-parsed it, use it. Otherwise, run our manual fallback parser!
            let (final_text, final_reasoning) = if let Some(reasoning) = &message.reasoning_content {
                (raw_text.clone(), Some(reasoning.clone()))
            } else {
                parse_raw_reasoning(raw_text)
            };

            Ok(LlmResponse { 
                text: final_text, 
                reasoning: final_reasoning 
            })
        },
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
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> { // <-- RETURN TYPE UPDATED
    let base_url = get_server_url();
    let url = format!("{}/completion", base_url);

    let mut processed_prompt = prompt;

    // --- Inject <|think|> trigger for Gemma 4 ---
    if let Some(cfg) = config {
        if cfg.thinking_budget.is_some() {
            if !processed_prompt.starts_with("<|think|>") {
                processed_prompt = format!("<|think|>\n{}", processed_prompt);
            }
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

    // --- Run the manual fallback parser on legacy output! ---
    let (final_text, final_reasoning) = parse_raw_reasoning(&parsed.content);

    Ok(LlmResponse {
        text: final_text,
        reasoning: final_reasoning,
    })
}

pub async fn call_llama_embeddings(
    input: String,
    model: Option<&str>,
    dimensions: Option<u32>,
    config: Option<&LlmConfig>, // Added this
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    
    // IP OVERRIDE LOGIC:
    // 1. Check if config provides a specific URL
    // 2. If not, fall back to the default get_server_url()
    let base_url = if let Some(cfg) = config {
        // Assuming LlmConfig has a field 'server_url'
        // If it doesn't, we'd use: cfg.server_url.clone().unwrap_or_else(|| get_server_url())
        cfg.server_url.clone().unwrap_or_else(get_server_url)
    } else {
        get_server_url()
    };

    let url = format!("{}/v1/embeddings", base_url);
    let client = Client::new();
    

    let embedding_request = EmbeddingRequest {
        model: model.unwrap_or("embedder").to_string(),
        input,
        dimensions,
        encoding_format: "float".to_string(),
    };

    let res = client
        .post(&url)
        .json(&embedding_request)
        .send()
        .await
        .map_err(|e| {
            Box::new(GeneralError {
                message: format!("Failed to send request to Llama Server Embeddings: {}", e),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let status = res.status();
    let rspns_strng = res.text().await.unwrap_or_default();

    if !status.is_success() {
        return Err(Box::new(GeneralError {
            message: format!("Llama Server returned HTTP {}: {}", status, rspns_strng),
        }) as Box<dyn std::error::Error + Send + Sync>);
    }

    match serde_json::from_str::<EmbeddingResponse>(&rspns_strng) {
        Ok(api_response) => {
            if let Some(data) = api_response.data.first() {
                Ok(data.embedding.clone())
            } else {
                Err(Box::new(GeneralError {
                    message: "No embedding data found in Llama Server response".to_string(),
                }) as Box<dyn std::error::Error + Send + Sync>)
            }
        },
        Err(e) => Err(Box::new(GeneralError {
            message: format!("Failed to parse Llama embedding response: {} - Raw: {}", e, rspns_strng),
        }) as Box<dyn std::error::Error + Send + Sync>),
    }
}
    
