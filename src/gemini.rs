use serde::{Deserialize, Serialize};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::models::{ListModelsResponse, ModelInfo};
use crate::token_count::{CountTokensRequest, CountTokensResponse, TokenCountContent, TokenCountPart};
use crate::structs::{ Message, Content, Part };

// https://ai.google.dev/tutorials/rest_quickstart

#[derive(Debug, Serialize)]
pub struct GeminiRequest {
    pub contents: Vec<Content>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<Candidate>,
    pub prompt_feedback: Option<PromptFeedback>,
}

#[derive(Debug, Deserialize)]
pub struct Candidate {
    pub content: Content,
    pub finish_reason: Option<String>,
    pub index: usize,
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Deserialize)]
pub struct PromptFeedback {
    pub safety_ratings: Vec<SafetyRating>,
}

#[derive(Debug, Deserialize)]
pub struct SafetyRating {
    pub category: String,
    pub probability: String,
}


pub async fn call_gemini(
    messages: Vec<Message>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GEMINI API KEY not found in environment variables".to_string(),
    })?;

    let url: &str =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent";

    let client = Client::new();

    let contents: Vec<Content> = messages
        .iter()
        .map(|msg| Content {
            role: msg.role.clone(),
            parts: vec![Part {
                text: msg.content.clone(),
            }],
        })
        .collect();

    let request = GeminiRequest { contents };

    let res = client
        .post(&format!("{}?key={}", url, api_key))
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            println!("{:?}", e);
            Box::new(GeneralError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let rspns_strng = res.text().await.map_err(|e: reqwest::Error| {
        println!("------------d------------------{:?}", e);
        Box::new(GeneralError {
            message: format!("Failed to read response from Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    // println!("Raw response: {}", rspns_strng);

    let res: GeminiResponse = serde_json::from_str(&rspns_strng).map_err(|e| {
        println!("{:?}", e);
        Box::new(GeneralError {
            message: format!(
                "Failed to parse response from Gemini API: {}",
                e.to_string()
            ),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    Ok(res.candidates[0].content.parts[0].text.clone())
}

#[derive(Deserialize, Debug)]
struct GeminiErrorResponse {
    error: GeminiError,
}

#[allow(dead_code)] 
#[derive(Deserialize, Debug)]
struct GeminiError {
    code: u16,
    message: String,
    status: String,
    details: Vec<GeminiErrorDetail>,
}

#[allow(dead_code)] 
#[derive(Deserialize, Debug)]
struct GeminiErrorDetail {
    #[serde(rename = "@type")]
    type_: String,
    reason: String,
    domain: String,
    metadata: std::collections::HashMap<String, String>,
}

pub async fn conversation_gemini_call(
    messages: Vec<Content>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    let url: &str =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent";

    let client = Client::new();

    let request = GeminiRequest { contents: messages };

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    let res = client
        .post(&format!("{}?key={}", url, api_key))
        .headers(headers)
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            Box::new(GeneralError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let response_body = res.text().await.map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to read response from Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    // Try to parse the response as a GeminiResponse
    let gemini_response: Result<GeminiResponse, _> = serde_json::from_str(&response_body).map_err(|_| {
        Box::new(GeneralError {
            message: "Failed to parse response from Gemini API".to_string(),
        }) as Box<dyn std::error::Error + Send + Sync>
    });

    match gemini_response {
        Ok(response) => Ok(response.candidates[0].content.parts[0].text.clone()),
        Err(_) => {
            // Try to parse the response as a GeminiErrorResponse
            let error_response: Result<GeminiErrorResponse, _> =
                serde_json::from_str(&response_body).map_err(|e| {
                    Box::new(GeneralError {
                        message: format!(
                            "Failed to parse error response from Gemini API: {}",
                            e.to_string()
                        ),
                    }) as Box<dyn std::error::Error + Send + Sync>
                });

            match error_response {
                Ok(err) => Err(Box::new(GeneralError {
                    message: format!("Gemini API Error: {}", err.error.message),
                }) as Box<dyn std::error::Error + Send + Sync>),
                Err(e) => Err(e),
            }
        }
    }
}

pub async fn get_gemini_model_info(
    model: &str,
) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    let url: String = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}",
        model
    );

    let client = Client::new();

    let res: ModelInfo = client
        .get(&format!("{}?key={}", url, api_key))
        .send()
        .await
        .map_err(|e| {
            Box::new(GeneralError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?
        .json()
        .await
        .map_err(|e| {
            Box::new(GeneralError {
                message: format!(
                    "Failed to parse response from Gemini API: {}",
                    e.to_string()
                ),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;
    Ok(res)
}

pub async fn list_gemini_models() -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>>
{
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    let url: &str = "https://generativelanguage.googleapis.com/v1beta/models";

    let client = Client::new();

    let res: ListModelsResponse = client
        .get(&format!("{}?key={}", url, api_key))
        .send()
        .await
        .map_err(|e| {
            Box::new(GeneralError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?
        .json()
        .await
        .map_err(|e| {
            Box::new(GeneralError {
                message: format!(
                    "Failed to parse response from Gemini API: {}",
                    e.to_string()
                ),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    Ok(res.models)
}

pub async fn count_gemini_tokens(
    text: &str,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    let url: &str =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:countTokens";

    let client = Client::new();

    let request = CountTokensRequest {
        contents: vec![TokenCountContent {
            parts: vec![TokenCountPart {
                text: text.to_string(),
            }],
        }],
    };

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    let res = client
        .post(&format!("{}?key={}", url, api_key))
        .headers(headers)
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            println!("------------------------------{:?}", e);
            Box::new(GeneralError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let rspns_strng = res.text().await.map_err(|e: reqwest::Error| {
        println!("------------d------------------{:?}", e);
        Box::new(GeneralError {
            message: format!("Failed to read response from Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;
    let res: CountTokensResponse = serde_json::from_str(&rspns_strng).unwrap();
    Ok(res.totalTokens)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_call_gemini() {
        let message: Message = Message {
            role: "user".to_string(),
            content: "Hi there, this is a test. Please generate a limrick about the muppets.".to_string(),
        };

        let messages: Vec<Message> = vec![message];

        let res = call_gemini(messages).await;
        match res {
            Ok(res_str) => {
                println!("res: {}", res_str);
                assert!(!res_str.is_empty());
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_conversation_gemini_call() {
        let messages = vec![
            Content {
                role: "user".to_string(),
                parts: vec![Part {
                    text: "Write the first line of a story about a magic backpack.".to_string(),
                }],
            },
            Content {
                role: "model".to_string(),
                parts: vec![Part {
                    text: "In the bustling city of Meadow brook, lived a young girl named Sophie. She was a bright and curious soul with an imaginative mind.".to_string(),
                }],
            },
            Content {
                role: "user".to_string(),
                parts: vec![Part {
                    text: "Can you set it in a quiet village in 1600s France?".to_string(),
                }],
            },
        ];

        let res = conversation_gemini_call(messages).await;
        match res {
            Ok(response) => {
                assert!(!response.is_empty());
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_get_gemini_model_info() {
        let res = get_gemini_model_info("gemini-1.0-pro-001").await;
        match res {
            Ok(model_info) => {
                println!("Ok: {:?}", &model_info);
                assert_eq!(model_info.name, "models/gemini-1.0-pro-001");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_list_gemini_models() {
        let res = list_gemini_models().await;
        match res {
            Ok(models) => {
                println!("Gemini Models\n{:?}", &models);
                assert!(!models.is_empty());
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_count_gemini_tokens() {
        let text = "Write a story about a magic backpack.";
        let res = count_gemini_tokens(text).await;
        match res {
            Ok(token_count) => {
                println!("token_count: {:?}", &token_count);
                assert!(token_count > 0);
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }
}
