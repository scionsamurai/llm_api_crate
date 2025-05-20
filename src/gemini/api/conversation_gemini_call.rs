// src/gemini/api/conversation_gemini_call.rs
use reqwest::header::{HeaderMap, HeaderValue};
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::structs::general::Content;
use crate::gemini::types::GeminiRequest;
use crate::gemini::request::gemini_request;
use crate::gemini::response::handle_gemini_error;

pub async fn conversation_gemini_call(
    messages: Vec<Content>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    let url: &str =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";

    let request = GeminiRequest { contents: messages };

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    let response = gemini_request(url, &api_key, &request, Some(headers)).await?;
    let response_body = response.text().await.map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to read response from Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;
    
    handle_gemini_error(&response_body)
}
