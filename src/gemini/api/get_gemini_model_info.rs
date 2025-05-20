// src/gemini/api/get_gemini_model_info.rs
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::models::gemini::ModelInfo;
use reqwest::header::{HeaderMap, HeaderValue};


pub async fn get_gemini_model_info(
    model: &str,
) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| {
        Box::new(GeneralError {
            message: "GEMINI_API_KEY not found in environment variables".to_string(),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    let url: String = format!(
        "https://generativelanguage.googleapis.com/v1beta/{}",
        model
    );

    let client = reqwest::Client::new();

    // Create a HeaderMap and add the API key
    let mut headers = HeaderMap::new();
    let api_key_value = HeaderValue::from_str(&api_key).map_err(|e| {
         Box::new(GeneralError {
            message: format!("Invalid API key format: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;
    headers.insert("x-goog-api-key", api_key_value);

    let response = client
        .get(&url) // Use the URL without the key
        .headers(headers) // Add the headers here
        .send()
        .await
        .map_err(|e| {
            Box::new(GeneralError {
                message: format!("Failed to send request to Gemini API: {}", e.to_string()),
            }) as Box<dyn std::error::Error + Send + Sync>
        })?;

    let response_body = response.text().await.map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to read response body: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    if response_body.is_empty() {
        return Err(Box::new(GeneralError {
            message: "Response body is empty".to_string(),
        }));
    }

    let res: ModelInfo = serde_json::from_str(&response_body).map_err(|e| {
        Box::new(GeneralError {
            message: format!(
                "Failed to parse response from Gemini API: {}",
                e.to_string()
            ),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    Ok(res)
}
