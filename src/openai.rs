// src/openai.rs
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::structs::general::{Message, LlmResponse}; 
use crate::structs::openai::{ChatCompletion, EmbeddingRequest};
use crate::models::openai::{APIResponse, ErrorResponse, EmbeddingResponse};
use crate::config::LlmConfig; // <-- Import config

const CHAT_COMPLETION_MODEL: &str = "gpt-4o"; // Updated default
const EMBEDDING_MODEL: &str = "text-embedding-3-small";
const EMBEDDING_ENCODING_FORMAT: &str = "float";

pub async fn call_gpt(
    messages: Vec<Message>,
    model: Option<&str>, // <-- Added dynamic model
    config: Option<&LlmConfig>, // <-- Added dynamic config
) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> { 
    dotenv().ok();

    let api_key: String =
        env::var("OPEN_AI_KEY").expect("OPEN AI KEY not found in environment variables");
    let api_org: String =
        env::var("OPEN_AI_ORG").unwrap_or_default(); // Made optional since not everyone uses orgs

    let url: &str = "https://api.openai.com/v1/chat/completions";

    let mut headers: HeaderMap = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {}", api_key))
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
    );
    
    if !api_org.is_empty() {
        headers.insert(
            "OpenAI-Organization",
            HeaderValue::from_str(api_org.as_str())
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
        );
    }

    let client = Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    let model_name = model.unwrap_or(CHAT_COMPLETION_MODEL).to_string();
    
    // Check if we are using an OpenAI reasoning model (o1, o3, etc.)
    let is_reasoning_model = model_name.starts_with("o1") || model_name.starts_with("o3");

    let mut chat_completion = ChatCompletion {
        model: model_name,
        messages,
        temperature: None,
        stream: None,
        max_tokens: None,
        max_completion_tokens: None, // NEW
        stop: None,
        top_k: None,
        top_p: None,
        cache_prompt: None,
        response_format: None,
    };

    if let Some(cfg) = config {
        if is_reasoning_model {
            // Reasoning models strictly use max_completion_tokens
            chat_completion.max_completion_tokens = cfg.max_tokens;
            
            // Reasoning models generally reject temperature or only accept 1.0
            if let Some(temp) = cfg.temperature {
                if (temp - 1.0).abs() < f64::EPSILON {
                    chat_completion.temperature = Some(1.0);
                }
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

    let res = client
        .post(url)
        .json(&chat_completion)
        .send()
        .await
        .map_err(|e| {
            println!("{:?}", e);
            Box::new(GeneralError {
                message: format!("Failed to send request to OpenAI Chat Completion API: {}", e),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let status = res.status();
    let rspns_strng = res.text().await.map_err(|e: reqwest::Error| {
        Box::new(GeneralError {
            message: format!("Failed to read response from OpenAI Chat Completion API: {}", e),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    if !status.is_success() {
        return Err(Box::new(GeneralError {
            message: format!("OpenAI API Error (HTTP {}): {}", status, rspns_strng),
        }));
    }

    match serde_json::from_str::<APIResponse>(&rspns_strng) {
        Ok(api_response) => {
            Ok(LlmResponse {
                text: api_response.choices[0].message.content.clone(),
                reasoning: api_response.choices[0].message.reasoning_content.clone(),
            })
        },
        Err(e) => {
            Err(Box::new(GeneralError {
                message: format!("Failed to parse response from OpenAI API: {} - Raw Response: {}", e, rspns_strng),
            }))
        }
    }
}

pub async fn get_embedding(
    input: String,
    model: Option<&str>, // Added model parameter
    dimensions: Option<u32>,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String =
        env::var("OPEN_AI_KEY").expect("OPEN AI KEY not found in environment variables");
    let api_org: String =
        env::var("OPEN_AI_ORG").unwrap_or_default(); // Changed to unwrap_or_default for robustness

    let url: &str = "https://api.openai.com/v1/embeddings";

    let mut headers: HeaderMap = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {}", api_key))
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
    );
    
    if !api_org.is_empty() {
        headers.insert(
            "OpenAI-Organization",
            HeaderValue::from_str(api_org.as_str())
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
        );
    }
    headers.insert(
        "Content-Type",
        HeaderValue::from_static("application/json"),
    );

    let client = Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    let embedding_request = EmbeddingRequest {
        model: model.unwrap_or(EMBEDDING_MODEL).to_string(), // Use provided model or default
        input,
        dimensions,
        encoding_format: EMBEDDING_ENCODING_FORMAT.to_string(),
    };

    let res = client
        .post(url)
        .json(&embedding_request)
        .send()
        .await
        .map_err(|e| {
            println!("{:?}", e);
            Box::new(GeneralError {
                message: format!("Failed to send request to OpenAI Embeddings API: {}", e),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let rspns_strng = res.text().await.map_err(|e: reqwest::Error| {
        Box::new(GeneralError {
            message: format!("Failed to read response from OpenAI Embeddings API: {}", e),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    match serde_json::from_str::<EmbeddingResponse>(&rspns_strng) {
        Ok(api_response) => {
            if let Some(data) = api_response.data.first() {
                Ok(data.embedding.clone())
            } else {
                Err(Box::new(GeneralError {
                    message: "No embedding data found in the OpenAI Embeddings API response".to_string(),
                }) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
        Err(_) => {
            match serde_json::from_str::<ErrorResponse>(&rspns_strng) {
                Ok(err) => Err(Box::new(GeneralError {
                    message: format!("OpenAI Embeddings API Error: {}", err.error.message),
                }) as Box<dyn std::error::Error + Send + Sync>),
                Err(e) => Err(Box::new(GeneralError {
                    message: format!("Failed to parse error response from OpenAI Embeddings API: {} - Raw Response: {}", e, rspns_strng),
                }) as Box<dyn std::error::Error + Send + Sync>), // Ensure this also returns a boxed dynamic error
            }
        }
    }
}
    
