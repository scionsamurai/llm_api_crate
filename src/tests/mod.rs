pub mod llm_openai;
pub mod llm_gemini;
pub mod llm_anthropic;
pub mod llm_llama;
pub mod embeddings;
pub mod multimodal;
pub mod openai;
pub mod anthropic;

use std::env;
use dotenv::dotenv;


pub fn get_base64_var() -> String {
    dotenv().ok(); 
    env::var("BASE64_DATA").unwrap_or_else(|_| "default_base64_value".to_string())
}