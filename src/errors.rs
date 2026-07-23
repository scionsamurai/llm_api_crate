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

/// Represents the decision made by a `RetryPolicy` after encountering an error.
#[derive(Debug, Clone)]
pub enum RetryDecision {
    /// Retry after the specified duration.
    RetryAfter(Duration),
    /// Do not retry; propagate the error immediately.
    Abort,
}

/// A trait for provider-aware retry policies that can inspect errors and determine
/// whether (and when) to retry based on provider-specific headers, body structures, or status codes.
pub trait RetryPolicy<E> {
    fn should_retry(&self, error: &E, attempt: usize) -> RetryDecision;
}

/// A standard exponential backoff policy (fallback / default policy).
pub struct ExponentialBackoffPolicy {
    pub max_retries: usize,
    pub initial_delay: Duration,
}

impl<E: std::fmt::Display> RetryPolicy<E> for ExponentialBackoffPolicy {
    fn should_retry(&self, _error: &E, attempt: usize) -> RetryDecision {
        if attempt >= self.max_retries {
            return RetryDecision::Abort;
        }
        // Calculate exponential backoff: initial_delay * 2^(attempt - 1)
        let multiplier = 2u32.saturating_pow((attempt - 1) as u32);
        let delay = self.initial_delay.saturating_mul(multiplier);
        RetryDecision::RetryAfter(delay)
    }
}

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
    let policy = ExponentialBackoffPolicy {
        max_retries,
        initial_delay,
    };
    with_policy_retry(action, policy).await
}

/// An advanced policy-driven retry wrapper for async operations.
pub async fn with_policy_retry<F, Fut, T, E, P>(
    mut action: F,
    policy: P,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
    P: RetryPolicy<E>,
{
    let mut attempts = 0;

    loop {
        match action().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                attempts += 1;
                match policy.should_retry(&err, attempts) {
                    RetryDecision::Abort => return Err(err),
                    RetryDecision::RetryAfter(delay) => {
                        eprintln!("Attempt {} failed: {}. Retrying in {:?}...", attempts, err, delay);
                        sleep(delay).await;
                    }
                }
            }
        }
    }
}