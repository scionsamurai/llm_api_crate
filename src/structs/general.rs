// src/structs/general.rs
use serde::{Deserialize, Serialize};

// --- New Unified Response Type ---
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    pub reasoning: Option<String>,
}

// --- New Multimodal Support ---
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Array(Vec<MessagePart>),
}

// These traits allow us to easily convert Strings into MessageContent
impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        MessageContent::Text(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        MessageContent::Text(text.to_string())
    }
}

impl MessageContent {
    /// Helper to easily extract just the text, ignoring images.
    pub fn extract_text(&self) -> String {
        match self {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Array(parts) => {
                // Combine all text parts into a single string
                parts.iter()
                    .filter_map(|p| p.text.clone())
                    .collect::<Vec<String>>()
                    .join("\n")
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MessagePart {
    pub r#type: String, // "text" or "image_url"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageSource>,
}

// --- New: Enum to handle inconsistent 'thought' types from different models ---
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ThoughtContent {
    Boolean(bool),
    String(String),
}

// --- Updated Message Struct ---
#[derive(Debug, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: MessageContent, // Changed from String
}

// --- Existing Gemini Structs ---
#[derive(Debug, Serialize, Deserialize)]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<GeminiInlineData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<ThoughtContent>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GeminiInlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmChunk {
    Text(String),
    Reasoning(String),
    /// Use this to signal the end of the stream or provide final metadata (like token counts)
    Done,
}
