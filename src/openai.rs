// src/openai.rs
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::structs::general::Message;
use crate::structs::openai::{ChatCompletion, EmbeddingRequest};
use crate::models::openai::{APIResponse, ErrorResponse, EmbeddingResponse};

const CHAT_COMPLETION_MODEL: &str = "gpt-4";
const EMBEDDING_MODEL: &str = "text-embedding-3-small";
const EMBEDDING_ENCODING_FORMAT: &str = "float";

pub async fn call_gpt(
    messages: Vec<Message>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String =
        env::var("OPEN_AI_KEY").expect("OPEN AI KEY not found in environment variables");
    let api_org: String =
        env::var("OPEN_AI_ORG").expect("OPEN AI KEY not found in environment variables");

    let url: &str = "https://api.openai.com/v1/chat/completions";

    let mut headers: HeaderMap = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {}", api_key))
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
    );
    headers.insert(
        "OpenAI-Organization",
        HeaderValue::from_str(api_org.as_str())
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
    );

    let client = Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    let chat_completion: ChatCompletion = ChatCompletion {
        model: CHAT_COMPLETION_MODEL.to_string(),
        messages,
        temperature: 0.1,
    };

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

    let rspns_strng = res.text().await.map_err(|e: reqwest::Error| {
        Box::new(GeneralError {
            message: format!("Failed to read response from OpenAI Chat Completion API: {}", e),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    match serde_json::from_str::<APIResponse>(&rspns_strng) {
        Ok(api_response) => Ok(api_response.choices[0].message.content.clone()),
        Err(_) => {
            match serde_json::from_str::<ErrorResponse>(&rspns_strng) {
                Ok(err) => Err(Box::new(GeneralError {
                    message: format!("OpenAI Chat Completion API Error: {}", err.error.message),
                }) as Box<dyn std::error::Error + Send + Sync>),
                Err(e) => Err(Box::new(GeneralError {
                    message: format!("Failed to parse error response from OpenAI Chat Completion API: {} - Raw Response: {}", e, rspns_strng),
                })),
            }
        }
    }
}

pub async fn get_embedding(
    input: String,
    dimensions: Option<u32>,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String =
        env::var("OPEN_AI_KEY").expect("OPEN AI KEY not found in environment variables");
    let api_org: String =
        env::var("OPEN_AI_ORG").expect("OPEN AI KEY not found in environment variables");

    let url: &str = "https://api.openai.com/v1/embeddings";

    let mut headers: HeaderMap = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {}", api_key))
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
    );
    headers.insert(
        "OpenAI-Organization",
        HeaderValue::from_str(api_org.as_str())
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
    );
    headers.insert(
        "Content-Type",
        HeaderValue::from_static("application/json"),
    );

    let client = Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    let embedding_request = EmbeddingRequest {
        model: EMBEDDING_MODEL.to_string(),
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
                }))
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
