// src/gemini/api/count_gemini_tokens.rs
use reqwest::header::{HeaderMap, HeaderValue};
use std::env;
use dotenv::dotenv;

use crate::errors::GeneralError;
use crate::token_count::{CountTokensRequest, CountTokensResponse, TokenCountContent, TokenCountPart};
use crate::gemini::request::gemini_request;


pub async fn count_gemini_tokens(
    text: &str,
    model: &str,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    let url =
        format!("https://generativelanguage.googleapis.com/v1beta/{}:countTokens", model);

    let request = CountTokensRequest {
        contents: vec![TokenCountContent {
            parts: vec![TokenCountPart {
                text: text.to_string(),
            }],
        }],
    };

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    let res = gemini_request(&url, &api_key, &request, Some(headers)).await?;

    let rspns_strng = res.text().await.map_err(|e: reqwest::Error| {
        println!("------------d------------------{:?}", e);
        Box::new(GeneralError {
            message: format!("Failed to read response from Gemini API: {}", e.to_string()),
        }) as Box<dyn std::error::Error + Send + Sync>
    })?;
    let tok_resp: CountTokensResponse = serde_json::from_str(&rspns_strng)
        .map_err(|e| GeneralError {
            message: format!("Failed to parse token count response: {}", e),
        })?;
    Ok(tok_resp.totalTokens)
}
