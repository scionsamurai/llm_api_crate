// src/gemini/api/call_gemini.rs
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::structs::general::{ Message, Content, Part };
use crate::gemini::types::GeminiRequest;
use crate::gemini::request::gemini_request;
use crate::gemini::response::parse_gemini_response;
use crate::gemini::types::GeminiResponse;

pub async fn call_gemini(
    messages: Vec<Message>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GEMINI API KEY not found in environment variables".to_string(),
    })?;

    let url: &str =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";

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

    let response = gemini_request(url, &api_key, &request, None).await?;
    let gemini_response: GeminiResponse = parse_gemini_response(response).await?;

    Ok(gemini_response.candidates[0].content.parts[0].text.clone())
}