use serde::{Deserialize, Serialize};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use dotenv::dotenv;

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

const MODEL: &str = "gpt-4";

pub async fn call_gpt(
    messages: Vec<Message>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    // Extract API Key information
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
        model: MODEL.to_string(),
        messages,
        temperature: 0.1,
    };

    let res: APIResponse = client
        .post(url)
        .json(&chat_completion)
        .send()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?
        .json()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    Ok(res.choices[0].message.content.clone())
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
            Ok(response) => {
                println!("response: {:}", response);
                assert!(!response.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to OpenAI API failed");
            }
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
                println!("Response1: {}", response);

                let user_message_2 = Message {
                    role: "user".to_string(),
                    content: "Can you also provide an example of how to use that function?".to_string(),
                };

                messages.push(user_message_2);

                let res = call_gpt(messages).await;
                match res {
                    Ok(response) => {
                        assert!(!response.is_empty(), "Response should not be empty");
                        println!("Response2: {}", response);
                    }
                    Err(err) => {
                        println!("Error: {}", err);
                        assert!(false, "Call to OpenAI API failed");
                    }
                }
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to OpenAI API failed");
            }
        }
    }
}