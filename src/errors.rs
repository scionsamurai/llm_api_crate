// src/errors.rs
use std::fmt;
use std::time::Duration;
use tokio::time::sleep;
use std::future::Future;

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

/// A generic retry wrapper for async operations with exponential backoff.
pub async fn with_retry<F, Fut, T, E>(
    mut action: F,
    max_retries: usize,
    initial_delay: Duration,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut attempts = 0;
    let mut delay = initial_delay;

    loop {
        match action().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                attempts += 1;
                if attempts >= max_retries {
                    return Err(err);
                }
                
                eprintln!("Attempt {} failed: {}. Retrying in {:?}...", attempts, err, delay);
                sleep(delay).await;
                
                // Exponential backoff: double the delay each time
                delay *= 2;
            }
        }
    }
}