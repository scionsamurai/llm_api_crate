// src/config.rs
use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub struct LlmConfig {
    pub temperature: Option<f64>,
    pub thinking_budget: Option<i32>,
    pub grounding_with_search: Option<bool>,
    pub server_url: Option<String>,

    // --- New llama-server / Universal Parameters ---
    
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,       // Maps to `n_predict` in legacy
    pub stop: Option<Vec<String>>,
    pub cache_prompt: Option<bool>,
    pub json_schema: Option<Value>,    // For constrained output
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
}

impl LlmConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_thinking_budget(mut self, thinking_budget: i32) -> Self {
        self.thinking_budget = Some(thinking_budget);
        self
    }

    pub fn with_grounding_with_search(mut self, grounding_with_search: bool) -> Self {
        self.grounding_with_search = Some(grounding_with_search);
        self
    }

    pub fn with_server_url(mut self, url: String) -> Self {
        self.server_url = Some(url);
        self
    }

    // --- New Builders ---
    
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    pub fn with_cache_prompt(mut self, cache_prompt: bool) -> Self {
        self.cache_prompt = Some(cache_prompt);
        self
    }

    pub fn with_json_schema(mut self, schema: Value) -> Self {
        self.json_schema = Some(schema);
        self
    }

    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
}
