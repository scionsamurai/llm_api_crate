use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub input_token_limit: Option<u32>,
    pub output_token_limit: Option<u32>,
    pub supported_generation_methods: Option<Vec<String>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<f32>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ListModelsResponse {
    pub models: Vec<ModelInfo>,
}