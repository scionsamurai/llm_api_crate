// src/token_count.rs
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug)]
pub struct TokenCountPart {
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct TokenCountContent {
    pub parts: Vec<TokenCountPart>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CountTokensRequest {
    pub contents: Vec<TokenCountContent>,
}

#[allow(non_snake_case)]
#[derive(Deserialize, Serialize, Debug)]
pub struct CountTokensResponse {
    pub totalTokens: u32,
}