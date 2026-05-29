// src/gemini/response.rs
use crate::errors::GeneralError;
use serde::de::DeserializeOwned;
use crate::gemini::types::{GeminiResponse, GeminiErrorResponse};
use crate::structs::general::{LlmResponse, ThoughtContent}; // Added ThoughtContent import

pub async fn parse_gemini_response<T: DeserializeOwned>(
    response: reqwest::Response,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
    let response_body = response.text().await.map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to read response from Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    serde_json::from_str(&response_body).map_err(|e| {
        println!("Failed to parse response body: {}", response_body);
        Box::new(GeneralError {
            message: format!("Failed to parse response from Gemini API 1: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })
}

/// Converts a GeminiResponse into a unified LlmResponse.
/// Handles the ThoughtContent enum to extract reasoning as a String.
pub fn gemini_to_llm_response(
    gemini_response: GeminiResponse,
) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> {
    let candidate = gemini_response.candidates.into_iter().next()
        .ok_or_else(|| Box::new(GeneralError { message: "No Gemini candidates".into() }) as Box<dyn std::error::Error + Send + Sync>)?;

    let mut text = String::new();
    // FIX: Explicitly define reasoning as Option<String> to prevent type inference errors
    let mut reasoning: Option<String> = None;

    for part in candidate.content.parts {
        match part.thought {
            // Case 1: Standard Gemini - thought is a string containing the reasoning
            Some(ThoughtContent::String(s)) => {
                reasoning = Some(s);
                if let Some(t) = part.text {
                    text.push_str(&t);
                }
            }
            // Case 2: Gemma - thought is a boolean flag. 
            // If true, the text in this part IS the reasoning content.
            Some(ThoughtContent::Boolean(true)) => {
                if let Some(t) = part.text {
                    reasoning = Some(t);
                }
            }
            // Case 3: Thought is false, or no thought field present - treat as normal text
            _ => {
                if let Some(t) = part.text {
                    text.push_str(&t);
                }
            }
        }
    }

    // println!("Extracted text from Gemini response: {}", text);
    // println!("Extracted reasoning from Gemini response: {:#?}", reasoning);
    Ok(LlmResponse { text, reasoning })
}

pub fn handle_gemini_error(response_body: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let gemini_response: Result<GeminiResponse, _> = serde_json::from_str(&response_body);

    match gemini_response {
        Ok(response) => Ok(response.candidates[0].content.parts[0].text.clone().unwrap()),
        Err(_) => {
            let error_response: Result<GeminiErrorResponse, _> =
                serde_json::from_str(&response_body).map_err(|e| {
                    Box::new(GeneralError {
                        message: format!(
                            "Failed to parse error response from Gemini API 2: {}",
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
