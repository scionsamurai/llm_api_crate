// src/config.rs
use dotenv::dotenv;
use std::env;

pub fn get_openai_api_key() -> Result<String, std::env::VarError> {
    env::var("OPENAI_API_KEY")
}

pub fn get_openai_org() -> Result<String, std::env::VarError> {
    env::var("OPEN_AI_ORG")
}

pub fn get_gemini_api_key() -> Result<String, std::env::VarError> {
    env::var("GEMINI_API_KEY")
}

pub fn get_anthropic_api_key() -> Result<String, std::env::VarError> {
    env::var("ANTHROPIC_API_KEY")
}