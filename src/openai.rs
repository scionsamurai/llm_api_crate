use serde::{Deserialize, Serialize};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::structs::Message;

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletion {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: f32,
}

#[derive(Debug, Deserialize)]
pub struct APIMessage {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct APIChoice {
    pub message: APIMessage,
}

#[derive(Debug, Deserialize)]
pub struct APIResponse {
    pub choices: Vec<APIChoice>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    pub encoding_format: String,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
    pub index: u32,
    pub object: String,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub object: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ErrorDetails,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ErrorDetails {
    message: String,
    r#type: String,
    param: Option<String>,
    code: String,
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_call_gpt() {
        let user_message = Message {
            role: "user".to_string(),
            content: "Hello, can you tell me a joke?".to_string(),
        };
        let messages = vec![user_message];
        let res = call_gpt(messages).await;
        match res {
            Ok(response) => assert!(!response.is_empty(), "Response should not be empty"),
            Err(err) => panic!("Call to OpenAI API failed: {}", err),
        }
    }

    #[tokio::test]
    async fn test_call_gpt_multi_prompt() {
        let system_message = Message {
            role: "system".to_string(),
            content: "You are a helpful coding assistant.".to_string(),
        };
        let user_message_1 = Message {
            role: "user".to_string(),
            content: "Hello, can you write a python function that reverses a string?".to_string(),
        };
        let mut messages = vec![system_message, user_message_1];
        let res = call_gpt(messages.clone()).await;
        match res {
            Ok(response) => {
                assert!(!response.is_empty(), "Response should not be empty");
                let user_message_2 = Message {
                    role: "user".to_string(),
                    content: "Can you also provide an example of how to use that function?".to_string(),
                };
                messages.push(user_message_2);
                let res = call_gpt(messages).await;
                match res {
                    Ok(response) => assert!(!response.is_empty(), "Response should not be empty"),
                    Err(err) => panic!("Call to OpenAI API failed on second prompt: {}", err),
                }
            }
            Err(err) => panic!("Call to OpenAI API failed on first prompt: {}", err),
        }
    }

    #[tokio::test]
    async fn test_get_embedding() {
        let input_text = "This is a test sentence.";
        let res = get_embedding(input_text.to_string(), None).await;
        match res {
            Ok(embedding) => {
                assert!(!embedding.is_empty(), "Embedding should not be empty");
                println!("Embedding vector length: {}", embedding.len());
                // Basic sanity check on the embedding vector (length might change with models)
                assert!(embedding.len() > 100);
            }
            Err(err) => panic!("Failed to get embedding: {}", err),
        }
    }

    #[tokio::test]
    async fn test_get_embedding_with_dimensions() {
        let input_text = "This is another test.";
        let dimensions: u32 = 64;
        let res = get_embedding(input_text.to_string(), Some(dimensions)).await;
        match res {
            Ok(embedding) => {
                assert!(!embedding.is_empty(), "Embedding should not be empty");
                assert_eq!(embedding.len() as u32, dimensions, "Embedding dimension should match requested dimension");
                println!("Embedding vector length: {}", embedding.len());
            }
            Err(err) => panic!("Failed to get embedding with dimensions: {}", err),
        }
    }
}