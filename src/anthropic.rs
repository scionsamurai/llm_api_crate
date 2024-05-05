use serde::{Deserialize, Serialize};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use dotenv::dotenv;

use crate::structs::Message;

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

const MODEL: &str = "claude-3-opus-20240229";
const MAX_TOKENS: usize = 1024;

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

    let api_key: String = env::var("ANTHROPIC_API_KEY").map_err(|_| AnthropicError {
        message: "ANTHROPIC API KEY not found in environment variables".to_string(),
    })?;

    let url: &str = "https://api.anthropic.com/v1/messages";

    let mut headers: HeaderMap = HeaderMap::new();

    headers.insert(
        "x-api-key",
        HeaderValue::from_str(&api_key)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
    );

    headers.insert(
        "anthropic-version",
        HeaderValue::from_str("2023-06-01")
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
    );

    headers.insert(
        "content-type",
        HeaderValue::from_str("application/json")
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?,
    );

    let client = Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    let request: AnthropicRequest = AnthropicRequest {
        model: MODEL.to_string(),
        max_tokens: MAX_TOKENS,
        messages,
    };

    let res = client
        .post(url)
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            println!("{:?}", e);
            Box::new(AnthropicError {
                message: format!("Failed to send request to Anthropic API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let rspns_strng = res.text().await.map_err(|e: reqwest::Error| {
        // println!("------------d------------------{:?}", e);
        Box::new(AnthropicError {
            message: format!("Failed to read response from Anthropic API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    // println!("Raw response: {}", rspns_strng);

    let res: AnthropicResponse = serde_json::from_str(&rspns_strng).map_err(|e| {
        println!("{:?}", e);
        Box::new(AnthropicError {
            message: format!(
                "Failed to parse response from Anthropic API: {}",
                e.to_string()
            ),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    Ok(res.content[0].text.clone())
}

#[derive(Debug, Clone)]
pub struct AnthropicError {
    pub message: String,
}

impl std::fmt::Display for AnthropicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AnthropicError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_call_anthropic() {
        let user_message = Message {
            role: "user".to_string(),
            content: "Hello, Claude. Can you tell me a joke?".to_string(),
        };

        let messages = vec![user_message];

        let res = call_anthropic(messages).await;
        match res {
            Ok(response) => {
                println!("response: {:}", response);
                assert!(!response.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Anthropic API failed");
            }
        }
    }

    #[tokio::test]
    async fn test_call_anthropic_multi_prompt() {
        let mut messages = vec![
            Message {
                role: "user".to_string(),
                content: "Write the first line of a story about a magic backpack.".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "In the bustling city of Meadow brook, lived a young girl named Sophie. She was a bright and curious soul with an imaginative mind.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Can you set it in a quiet village in 1600s France?".to_string(),
            },
        ];

        let res = call_anthropic(messages.clone()).await;
        match res {
            Ok(response) => {
                assert!(!response.is_empty(), "Response should not be empty");
                println!("Response1: {}", &response);
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: response
                });

                let user_message_2 = Message {
                    role: "user".to_string(),
                    content: "Can you also make the story about pokemon?".to_string(),
                };

                messages.push(user_message_2);

                let res = call_anthropic(messages).await;
                match res {
                    Ok(response) => {
                        assert!(!response.is_empty(), "Response should not be empty");
                        println!("Response2: {}", response);
                    }
                    Err(err) => {
                        println!("Error: {}", err);
                        assert!(false, "Call to Anthropic API failed");
                    }
                }
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Anthropic API failed");
            }
        }
    }
}