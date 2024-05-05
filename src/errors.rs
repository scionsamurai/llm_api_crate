use std::fmt;

#[derive(Debug)]
pub struct GeminiError {
    pub message: String,
}

impl fmt::Display for GeminiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for GeminiError {}