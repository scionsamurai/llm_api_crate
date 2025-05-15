// src/gemini/request.rs
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;

use crate::errors::GeneralError;

pub async fn gemini_request<T: serde::Serialize>(
    url: &str,
    api_key: &str,
    request: &T,
    headers: Option<HeaderMap>, // Make headers mutable
) -> Result<reqwest::Response, Box<dyn std::error::Error + Send + Sync>> {
    let client = Client::new();

    // Create or get the HeaderMap
    let mut final_headers = headers.unwrap_or_else(HeaderMap::new);

    // Add the API key to the headers
    let api_key_value = HeaderValue::from_str(api_key).map_err(|e| {
         Box::new(GeneralError {
            message: format!("Invalid API key format: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?; // Handle potential invalid header value characters
    final_headers.insert("x-goog-api-key", api_key_value);

    // Build the request without the key in the URL
    let req = client
        .post(url) // Removed the key from the URL format
        .json(request)
        .headers(final_headers); // Add the constructed headers

    req.send().await.map_err(|e| {
        Box::new(GeneralError {
            message: format!("Failed to send request to Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })
}