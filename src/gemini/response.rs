// src/gemini/response.rs
use crate::errors::GeneralError;
use serde::de::DeserializeOwned;
use crate::gemini::types::{GeminiResponse, GeminiErrorResponse};

pub async fn parse_gemini_response<T: DeserializeOwned>(
    response: reqwest::Response,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
    let response_body = response.text().await.map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to read response from Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    // Add println! to show raw response when parsing into T fails
    serde_json::from_str(&response_body).map_err(|e| {
        eprintln!("Raw Gemini API response (initial parse failure): {}", response_body);
        Box::new(GeneralError {
            message: format!("Failed to parse response from Gemini API 1: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })
}

pub fn handle_gemini_error(response_body: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let gemini_response: Result<GeminiResponse, _> = serde_json::from_str(&response_body);

    match gemini_response {
        Ok(response) => Ok(response.candidates[0].content.parts[0].text.clone()),
        Err(e_gemini_response) => {
            eprintln!("Raw Gemini API response (GeminiResponse parse failure): {}", response_body);
            // Try to parse the response as a GeminiErrorResponse
            let error_response: Result<GeminiErrorResponse, _> =
                serde_json::from_str(&response_body).map_err(|e| {
                    eprintln!("Raw Gemini API response (GeminiErrorResponse parse failure): {}", response_body);
                    Box::new(GeneralError {
                        message: format!(
                            "Failed to parse error response from Gemini API 2: {}",
                            e.to_string()
                        ),
                    }) as Box<dyn std::error::Error + Send + Sync>
                });

            match error_response {
                Ok(err) => {
                    // It seems the GeminiErrorResponse structure might be missing 'details' sometimes.
                    // Let's print the full error message for better context.
                    Err(Box::new(GeneralError {
                        message: format!("Gemini API Error: Code: {}, Status: {}, Message: {}, Details: {:?}",
                                         err.error.code, err.error.status, err.error.message, err.error.details),
                    }) as Box<dyn std::error::Error + Send + Sync>)
                },
                Err(e) => Err(e),
            }
        }
    }
}