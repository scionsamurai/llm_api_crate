// src/gemini/response.rs
use crate::errors::GeneralError;
use serde::de::DeserializeOwned;
use crate::gemini::types::{GeminiResponse, GeminiErrorResponse};
use crate::structs::general::LlmResponse;
    

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

/// NEW: Converts a GeminiResponse into a unified LlmResponse.
/// This centralizes the extraction of text and reasoning.
pub fn gemini_to_llm_response(
    gemini_response: GeminiResponse,
) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>> {
    let candidate = gemini_response.candidates.into_iter().next()
        .ok_or_else(|| Box::new(GeneralError { message: "No Gemini candidates".into() }) as Box<dyn std::error::Error + Send + Sync>)?;

    let mut text = String::new();
    let mut reasoning = None;

    for part in candidate.content.parts {
        if let Some(t) = part.text {
            text.push_str(&t);
        }
        if let Some(th) = part.thought {
            reasoning = Some(th);
        }
    }

    Ok(LlmResponse { text, reasoning })
}

pub fn handle_gemini_error(response_body: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    
    let gemini_response: Result<GeminiResponse, _> = serde_json::from_str(&response_body);

    match gemini_response {
        Ok(response) => Ok(response.candidates[0].content.parts[0].text.clone().unwrap()),
        Err(_) => {
            // Try to parse the response as a GeminiErrorResponse
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
