// src/gemini/api/call_gemini.rs
use std::env;
use std::time::Duration;
use dotenv::dotenv;
use serde_json::json;

    use futures::stream::{BoxStream, StreamExt};
    use async_stream::stream;
    use crate::errors::{GeneralError, with_retry};
    use crate::structs::general::{Message, Content, Part, LlmChunk, MessagePart, ImageSource, GeminiInlineData};
    use crate::gemini::types::{GeminiRequest, GenerationConfig, Tool, GeminiResponse};
    use crate::gemini::request::gemini_request;
    use crate::gemini::response::parse_gemini_response;
    use crate::config::LlmConfig;

pub fn map_message_parts_to_gemini(parts: Vec<MessagePart>) -> Vec<Part> {
    parts.into_iter().map(|p| {
        if p.r#type == "image_url" {
            if let Some(ImageSource::Base64 { media_type, data }) = p.image_url {
                // Apply the shared strip utility
                let clean_data = ImageSource::strip_base64_prefix(&data).to_string();
                return Part { 
                    text: None, 
                    inline_data: Some(GeminiInlineData { mime_type: media_type, data: clean_data }),
                    thought: None 
                };
            }
        }
        Part { text: p.text, inline_data: None, thought: None }
    }).collect()
}

pub async fn call_gemini(
    messages: Vec<Message>,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<GeminiResponse, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();
    // Fixed snake_case warning
    let default_gemini_model: String = env::var("DEFAULT_GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());

    let api_key: String = env::var("GEMINI_API_KEY").map_err(|_| GeneralError {
        message: "GEMINI API KEY not found in environment variables".to_string(),
    })?;

    // Updated reference to snake_case variable
    let model_name = model.unwrap_or(&default_gemini_model);
    let url: String = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        model_name
    );
 
    let contents: Vec<Content> = messages
        .into_iter()
        .map(|msg| Content {
            role: msg.role,
            parts: map_message_parts_to_gemini(msg.content.as_parts()),
        })
        .collect();

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
        contents,
        generation_config: generation_config_option,
        tools: tools_option,
    };

    // Wrap the request and parsing logic in with_retry
    with_retry(|| async {
        let response = gemini_request(&url, &api_key, &request, None).await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Box::new(GeneralError {
                message: format!("Gemini API returned error {}: {}", status, body),
            }) as Box<dyn std::error::Error + Send + Sync>);
        }

        let gemini_response: GeminiResponse = parse_gemini_response(response).await?;
        Ok(gemini_response)
    }, 3, Duration::from_secs(1)).await
}

pub async fn call_gemini_stream(
    messages: Vec<Message>,
    model: Option<&str>,
    config: Option<&LlmConfig>,
) -> Result<BoxStream<'static, Result<LlmChunk, Box<dyn std::error::Error + Send + Sync>>>, Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();
    let default_gemini_model = env::var("DEFAULT_GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.0-flash".to_string());
    let api_key = env::var("GEMINI_API_KEY").map_err(|_| GeneralError { message: "GEMINI API KEY not found".into() })?;
    let model_name = model.unwrap_or(&default_gemini_model);
    
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?alt=sse", 
        model_name
    );

    let contents: Vec<Content> = messages.into_iter().map(|msg| Content {
        role: msg.role,
        parts: map_message_parts_to_gemini(msg.content.as_parts()),
    }).collect();

    let mut generation_config_option = None;
    let mut tools_option = None;
    if let Some(cfg) = config {
        let current_gen_config = GenerationConfig { temperature: cfg.temperature, thinking_budget: cfg.thinking_budget };
        generation_config_option = Some(current_gen_config);
        if cfg.grounding_with_search.unwrap_or(false) {
            tools_option = Some(vec![Tool { google_search: Some(json!({})) }]);
        }
    }

    let request = GeminiRequest { contents, generation_config: generation_config_option, tools: tools_option };
    
    let client = reqwest::Client::new();
    let res = client.post(&url).header("x-goog-api-key", &api_key).json(&request).send().await?;

    if !res.status().is_success() {
        let err_text = res.text().await?;
        return Err(Box::new(GeneralError { message: format!("Gemini Stream Error: {}", err_text) }));
    }

    let byte_stream = res.bytes_stream();

    let output_stream = stream! {
        let mut buffer = String::new();
        let mut bytes_stream = byte_stream;

        while let Some(item) = bytes_stream.next().await {
            match item {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));
                    while let Some(newline_idx) = buffer.find('\n') {
                        let line = buffer.drain(..newline_idx + 1).collect::<String>().trim().to_string();
                        if line.is_empty() { continue; }
                        if line.starts_with("data: ") {
                            let json_str = &line[6..];
                            if let Ok(response) = serde_json::from_str::<GeminiResponse>(json_str) {
                                if let Some(candidate) = response.candidates.first() {
                                    for part in &candidate.content.parts {
                                        if let Some(text) = &part.text { yield Ok(LlmChunk::Text(text.clone())); }
                                        if let Some(thought) = &part.thought {
                                            if let crate::structs::general::ThoughtContent::String(s) = thought {
                                                yield Ok(LlmChunk::Reasoning(s.clone()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => yield Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
            }
        }
        yield Ok(LlmChunk::Done);
    };

    Ok(Box::pin(output_stream))
}