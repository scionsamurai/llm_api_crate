// src/gemini/api/conversation_gemini_call.rs
use reqwest::header::{HeaderMap, HeaderValue};
use std::env;
use std::time::Duration;
use dotenv::dotenv;
use serde_json::json; // Removed Map, Value

use crate::errors::{GeneralError, RetryPolicy, RetryDecision, with_policy_retry};
use crate::structs::general::Content;
use crate::gemini::types::{GeminiRequest, GenerationConfig, Tool, GeminiResponse, GeminiErrorResponse};
use crate::gemini::request::gemini_request;
use crate::gemini::response::parse_gemini_response;
use crate::config::LlmConfig;

/// Gemini-aware retry policy that parses Google RPC error details (specifically `retryDelay`)
/// and adds a safety padding (+2s) to prevent immediate secondary quota rejections.
pub struct GeminiRetryPolicy {
    pub max_retries: usize,
    pub safety_padding: Duration,
}

impl RetryPolicy<Box<dyn std::error::Error + Send + Sync>> for GeminiRetryPolicy {
    fn should_retry(&self, error: &Box<dyn std::error::Error + Send + Sync>, attempt: usize) -> RetryDecision {
        if attempt >= self.max_retries {
            return RetryDecision::Abort;
        }

        let err_str = error.to_string();

        // Attempt to extract structured Gemini error JSON from the error message string
        // Error messages typically follow the format: "Gemini API returned error 429 Too Many Requests: { ... }"
        if let Some(json_start) = err_str.find('{') {
            let json_part = &err_str[json_start..];
            if let Ok(error_resp) = serde_json::from_str::<GeminiErrorResponse>(json_part) {
                if let Some(details) = error_resp.error.details {
                    for detail in details {
                        if let Some(delay_str) = detail.retry_delay {
                            // Parse strings like "16.747818564s" or "16s"
                            let clean_delay = delay_str.trim_end_matches('s');
                            if let Ok(secs) = clean_delay.parse::<f64>() {
                                let suggested_duration = Duration::from_secs_f64(secs);
                                let total_delay = suggested_duration + self.safety_padding;
                                return RetryDecision::RetryAfter(total_delay);
                            }
                        }
                    }
                }
            }
        }

        // Fallback to standard exponential backoff if no structured retryDelay was found
        let fallback_delay = Duration::from_secs(1) * 2u32.saturating_pow((attempt - 1) as u32);
        RetryDecision::RetryAfter(fallback_delay)
    }
}

pub async fn conversation_gemini_call(
    messages: Vec<Content>,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<GeminiResponse, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();
    // Fixed snake_case warning
    let default_gemini_model: String = env::var("DEFAULT_GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GOOGLE API KEY not found in environment variables".to_string(),
    })?;

    // Updated reference to snake_case variable
    let model_name = model.unwrap_or(&default_gemini_model);
    let url: String = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        model_name
    );

    let mut generation_config_option: Option<GenerationConfig> = None;
    let mut tools_option: Option<Vec<Tool>> = None;

    if let Some(cfg) = config {
        let mut current_gen_config = GenerationConfig {
            temperature: None,
            thinking_budget: None,
        };

        let mut config_has_options = false;

        if let Some(thinking_budget) = cfg.thinking_budget {
            current_gen_config.thinking_budget = Some(thinking_budget);
            config_has_options = true;
        }

        if let Some(temp) = cfg.temperature {
            current_gen_config.temperature = Some(temp);
            config_has_options = true;
        }

        if config_has_options {
            generation_config_option = Some(current_gen_config);
        }

        if cfg.grounding_with_search.unwrap_or(false) {
            tools_option = Some(vec![Tool { google_search: Some(json!({})) }]);
        }
    }

    let request = GeminiRequest {
        contents: messages,
        generation_config: generation_config_option,
        tools: tools_option,
    };
    
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    let policy = GeminiRetryPolicy {
        max_retries: 3,
        safety_padding: Duration::from_secs(2), // Add 2 seconds padding to server-suggested retryDelay
    };

    // Wrap the request and parsing logic in policy-driven retry
    with_policy_retry(|| async {
        let response = gemini_request(&url, &api_key, &request, Some(headers.clone())).await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Box::new(GeneralError {
                message: format!("Gemini API returned error {}: {}", status, body),
            }) as Box<dyn std::error::Error + Send + Sync>);
        }

        let gemini_response: GeminiResponse = parse_gemini_response(response).await?;
        Ok(gemini_response)
    }, policy).await
}