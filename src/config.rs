// src/config.rs

#[derive(Debug, Clone, Default)]
pub struct LlmConfig {
    pub temperature: Option<f64>,
    pub thinking_budget: Option<i32>,
    // Add other configuration options here
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
}