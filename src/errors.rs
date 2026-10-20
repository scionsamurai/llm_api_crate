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

use std::error::Error as StdError;

/// Upper bound on any single backoff sleep, including a server's Retry-After.
pub const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmErrorKind {
    /// A per-attempt timeout (connect + request + body) elapsed.
    Timeout,
    /// Could not connect (refused, DNS, unreachable).
    Connect,
    /// The connection broke after it was established.
    Transport,
    /// The server answered with a non-success status.
    Http,
    /// A success response carried no choices.
    Empty,
    /// A success response could not be understood.
    Malformed,
}

/// Typed transport failure. Boxed into the `Access` error type, so callers can
/// `downcast_ref::<LlmError>()` instead of parsing messages.
#[derive(Debug, Clone)]
pub struct LlmError {
    pub kind: LlmErrorKind,
    pub status: Option<u16>,
    pub retry_after: Option<Duration>,
    pub message: String,
}

pub(crate) fn clip(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

impl LlmError {
    pub fn new(kind: LlmErrorKind, message: impl Into<String>) -> Self {
        Self { kind, status: None, retry_after: None, message: message.into() }
    }

    pub fn http(status: u16, retry_after: Option<Duration>, body: &str) -> Self {
        Self {
            kind: LlmErrorKind::Http,
            status: Some(status),
            retry_after,
            message: format!("HTTP {status}: {}", clip(body.trim(), 400)),
        }
    }

    pub fn from_reqwest(context: &str, e: &reqwest::Error) -> Self {
        let kind = if e.is_timeout() {
            LlmErrorKind::Timeout
        } else if e.is_connect() {
            LlmErrorKind::Connect
        } else {
            LlmErrorKind::Transport
        };
        Self::new(kind, format!("{context}: {e}"))
    }

    /// Whether repeating the same request can plausibly succeed.
    pub fn is_retryable(&self) -> bool {
        match self.kind {
            LlmErrorKind::Timeout | LlmErrorKind::Connect | LlmErrorKind::Transport => true,
            LlmErrorKind::Http => matches!(self.status, Some(408 | 425 | 429 | 500 | 502 | 503 | 504)),
            LlmErrorKind::Empty | LlmErrorKind::Malformed => false,
        }
    }
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl StdError for LlmError {}

/// Retries only errors that are `LlmError::is_retryable`, at most
/// `max_attempts` attempts in total, honouring Retry-After (capped).
pub struct LlmRetryPolicy {
    pub max_attempts: usize,
    pub initial_delay: Duration,
    pub max_delay: Duration,
}

impl RetryPolicy<Box<dyn StdError + Send + Sync>> for LlmRetryPolicy {
    fn should_retry(&self, error: &Box<dyn StdError + Send + Sync>, attempt: usize) -> RetryDecision {
        let Some(e) = error.downcast_ref::<LlmError>() else { return RetryDecision::Abort };
        if !e.is_retryable() || attempt >= self.max_attempts {
            return RetryDecision::Abort;
        }
        let backoff = self.initial_delay.saturating_mul(2u32.saturating_pow((attempt - 1) as u32));
        RetryDecision::RetryAfter(e.retry_after.unwrap_or(backoff).min(self.max_delay))
    }
}