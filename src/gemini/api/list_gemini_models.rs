// src/gemini/api/list_gemini_models.rs
use std::env;
use dotenv::dotenv;
use reqwest::header::{HeaderMap, HeaderValue};

use crate::errors::GeneralError;
use crate::models::{ListModelsResponse, ModelInfo};


pub async fn list_gemini_models() -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>>
{
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| {
        Box::new(GeneralError {
            message: "GEMINI_API_KEY not found in environment variables".to_string(),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;

    let url: &str = "https://generativelanguage.googleapis.com/v1beta/models";

    let client = reqwest::Client::new();

    // Create a HeaderMap and add the API key
    let mut headers = HeaderMap::new();
    let api_key_value = HeaderValue::from_str(&api_key).map_err(|e| {
        Box::new(GeneralError {
            message: format!("Invalid API key format: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;
    headers.insert("x-goog-api-key", api_key_value);


    let res: ListModelsResponse = client
        .get(url) // Removed the key from the URL
        .headers(headers) // Add the headers here
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
