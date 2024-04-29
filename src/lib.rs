use dotenv::dotenv;
use reqwest::Client;
use std::env;
use reqwest::header::{HeaderMap, HeaderValue};
use async_trait::async_trait;

pub mod structs;

use crate::structs::{
    APIResponse, ChatCompletion, Content, CountTokensRequest, CountTokensResponse, GeminiError,
    GeminiRequest, GeminiResponse, ListModelsResponse, Message, ModelInfo, Part, TokenCountContent,
    TokenCountPart, LLM
};

#[async_trait]
pub trait Access {
    async fn send_single_message(
        &self,
        messages: Vec<Message>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn send_convo_message(
        &self,
        messages: Vec<Content>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn get_model_info(
        &self,
        model: &str,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>>;
    async fn list_models(&self)
        -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>>;
    async fn count_tokens(
        &self,
        text: &str,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait]
impl Access for LLM {
    async fn send_single_message(
        &self,
        messages: Vec<Message>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => call_gpt(messages).await,
            LLM::Gemini => call_gemini(messages).await,
            LLM::Claude => Ok("Claude not yet implemented in send_single_message func".to_string()),
        }
    }
    async fn send_convo_message(
        &self,
        messages: Vec<Content>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => Ok("OpenAI not yet implemented in send_convo_message func".to_string()),
            LLM::Gemini => conversation_gemini_call(messages).await,
            LLM::Claude => Ok("Claude not yet implemented in send_convo_message func".to_string()),
        }
    }
    async fn get_model_info(
        &self,
        model: &str,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => get_gemini_model_info(model).await,
            _ => Err(Box::new(GeminiError {
                message: format!("Currently only Gemini is implemented for get_model_info func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }
    async fn list_models(
        &self,
    ) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => list_gemini_models().await,
            _ => Err(Box::new(GeminiError {
                message: format!("Currently only Gemini is implemented for list_models func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }
    async fn count_tokens(
        &self,
        text: &str,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => count_gemini_tokens(text).await,
            _ => Err(Box::new(GeminiError {
                message: format!("Currently only Gemini is implemented for count_tokens func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }
}

// https://ai.google.dev/tutorials/rest_quickstart

pub async fn call_gemini(
    messages: Vec<Message>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeminiError {
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

    let res: GeminiResponse = client
        .post(&format!("{}?key={}", url, api_key))
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            println!("{:?}", e);
            Box::new(GeminiError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?
        .json()
        .await
        .map_err(|e| {
            println!("{:?}", e);
            Box::new(GeminiError {
                message: format!(
                    "Failed to parse response from Gemini API: {}",
                    e.to_string()
                ),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    Ok(res.candidates[0].content.parts[0].text.clone())
}

pub async fn conversation_gemini_call(
    messages: Vec<Content>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeminiError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    let url: &str =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent";

    let client = Client::new();

    let request = GeminiRequest { contents: messages };

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    let res: GeminiResponse = client
        .post(&format!("{}?key={}", url, api_key))
        .headers(headers)
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            Box::new(GeminiError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?
        .json()
        .await
        .map_err(|e| {
            Box::new(GeminiError {
                message: format!(
                    "Failed to parse response from Gemini API: {}",
                    e.to_string()
                ),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;
    println!("{:?}", res);

    Ok(res.candidates[0].content.parts[0].text.clone())
}

pub async fn get_gemini_model_info(
    model: &str,
) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeminiError {
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
            Box::new(GeminiError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?
        .json()
        .await
        .map_err(|e| {
            Box::new(GeminiError {
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

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeminiError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    let url: &str = "https://generativelanguage.googleapis.com/v1beta/models";

    let client = Client::new();

    let res: ListModelsResponse = client
        .get(&format!("{}?key={}", url, api_key))
        .send()
        .await
        .map_err(|e| {
            Box::new(GeminiError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?
        .json()
        .await
        .map_err(|e| {
            Box::new(GeminiError {
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

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeminiError {
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
            Box::new(GeminiError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let rspns_strng = res.text().await.map_err(|e: reqwest::Error| {
        println!("------------d------------------{:?}", e);
        Box::new(GeminiError {
            message: format!("Failed to read response from Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;
    let res: CountTokensResponse = serde_json::from_str(&rspns_strng).unwrap();
    Ok(res.totalTokens)
}

// Call Large Language Model (i.e. GPT-4)
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
        model: "gpt-4".to_string(),
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
    async fn test_call_to_gemini() {
        let message: Message = Message {
            role: "user".to_string(),
            content: "Hi there, this is a test. Please generate a limrik.".to_string(),
        };

        let messages: Vec<Message> = vec![message];

        // let res: Result<String, Box<dyn std::error::Error + Send>> = call_gpt(messages).await;
        let res = call_gemini(messages).await;
        match res {
            Ok(res_str) => {
                dbg!(res_str);
                assert!(true);
            }
            Err(_) => {
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
                dbg!(response);
                assert!(true);
            }
            Err(_) => {
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_get_gemini_model_info() {
        let res = get_gemini_model_info("gemini-1.0-pro-001").await;
        match res {
            Ok(model_info) => {
                dbg!(model_info);
                assert!(true);
            }
            Err(_) => {
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_list_gemini_models() {
        let res = list_gemini_models().await;
        match res {
            Ok(models) => {
                dbg!(models);
                assert!(true);
            }
            Err(_) => {
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
                dbg!(token_count);
                assert!(true);
            }
            Err(_) => {
                assert!(false);
            }
        }
    }
}
