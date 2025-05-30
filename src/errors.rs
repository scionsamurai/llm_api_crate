// src/errors.rs
use std::fmt;

#[derive(Debug)]
pub struct GeneralError {
    pub message: String,
}

impl fmt::Display for GeneralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for GeneralError {}