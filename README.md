# llm_api_access

The `llm_api_access` crate provides a unified way to interact with different large language models (LLMs) like OpenAI, Gemini, and Anthropic.

## Current Status

This crate is used to power an open-source coding assistant currently in active development. Gemini has been the main test target; OpenAI (including embeddings) and Anthropic are supported but have been exercised less. Development is self encouraged so updates can be far and few between, open an issue on github if you want something specific.

### LLM Enum

This enum represents the supported LLM providers:

- `OpenAI`: Represents the OpenAI language model.
- `Gemini`: Represents the Gemini language model.
- `Anthropic`: Represents the Anthropic language model.

### Access Trait

The `Access` trait defines asynchronous methods for interacting with LLMs:

- `send_single_message`: Sends a single message and returns the generated response.
  ```rust
  async fn send_single_message(
        &self,
        message: &str,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
  ```
- `send_convo_message`: Sends a list of messages as a conversation and returns the generated response.
  ```rust
  async fn send_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
  ```
- `get_model_info`: Gets information about a specific LLM model.
  ```rust
  async fn get_model_info(
        &self,
        model: &str,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>>;
  ```
- `list_models`: Lists all available LLM models.
  ```rust
  async fn list_models(&self)
        -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>>;
  ```
- `count_tokens`: Counts the number of tokens in a given text.
  ```rust
  async fn count_tokens(
        &self,
        text: &str,
        model: &str,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>>;
  ```

The `LLM` enum implements `Access`, providing specific implementations for each method based on the chosen LLM provider.

**Note:** Currently, `get_model_info`, `list_models`, and `count_tokens` only work for the Gemini LLM. Other providers return an error indicating this functionality is not yet supported.

### LlmConfig

The `LlmConfig` struct allows you to configure provider-specific settings for the LLM calls. It uses a builder pattern for easy customization.

```rust
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
```

**Example Usage:**

```rust
use llm_api_access::config::LlmConfig;

// Default usage (no config)
let config = None;

// With thinking budget
let config = Some(LlmConfig::new().with_thinking_budget(1024));

// With multiple options
let config = Some(LlmConfig::new()
    .with_thinking_budget(2048)
    .with_temperature(0.7));
```

### Loading API Credentials with dotenv

The `llm_api_access` crate uses the `dotenv` library to securely load API credentials from a `.env` file in your project's root directory. This file should contain key-value pairs for each LLM provider you want to use.

**Example Structure:**

```
OPEN_AI_ORG=your_openai_org
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

**Steps:**

1. **Create `.env` File:** Create a file named `.env` at the root of your Rust project directory.
2. **Add API Keys:** Fill in the `.env` file with the following format, replacing placeholders with your actual API keys.

**Important Note:**

* **Never** commit your `.env` file to version control systems like Git. It contains sensitive information like API keys.

## Example Usage

### `send_single_message` Example

```rust
use llm_api_access::llm::{Access, LLM};
use llm_api_access::config::LlmConfig; // Import LlmConfig

#[tokio::main]
async fn main() {
    // Create an instance of the OpenAI LLM
    let llm = LLM::OpenAI;

    // Send a single message to the LLM with no config
    let response = llm.send_single_message("Tell me a joke about programmers", None, None).await;

    match response {
        Ok(joke) => println!("Joke: {}", joke),
        Err(err) => eprintln!("Error: {}", err),
    }

    //Send a single message to the LLM with a config
    let config = Some(LlmConfig::new().with_temperature(0.7));
    let response = llm.send_single_message("Tell me a joke about programmers", None, config.as_ref()).await;

    match response {
        Ok(joke) => println!("Joke: {}", joke),
        Err(err) => eprintln!("Error: {}", err),
    }
}
```

This example creates an instance of the `LLM::OpenAI` provider and sends a single message using the `send_single_message` method. It then matches the result, printing the generated joke or an error message if an error occurred.

### `send_convo_message` Example

```rust
use llm_api_access::llm::{Access, LLM};
use llm_api_access::structs::general::Message;
use llm_api_access::config::LlmConfig; // Import LlmConfig

#[tokio::main]
async fn main() {
    // Create an instance of the Gemini LLM
    let llm = LLM::Gemini;

    // Define the conversation messages
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: "You are a helpful coding assistant.".to_string(),
        },
        Message {
            role: "model".to_string(),
            content: "You got it! I am ready to assist!".to_string(),
        },
        Message {
            role: "user".to_string(),
            content: "Generate a rust function that reverses a string.".to_string(),
        },
    ];

    // Send the conversation messages to the LLM with no config
    let response = llm.send_convo_message(messages, None, None).await;

    match response {
        Ok(code) => println!("Code: {}", code),
        Err(err) => eprintln!("Error: {}", err),
    }

    // Send the conversation messages to the LLM with a config
    let config = Some(LlmConfig::new().with_thinking_budget(2048));
    let response = llm.send_convo_message(messages, None, config.as_ref()).await;

    match response {
        Ok(code) => println!("Code: {}", code),
        Err(err) => eprintln!("Error: {}", err),
    }
}
```

**Note:** This example requires API keys and configuration for the Gemini LLM provider.

## Embeddings

The crate provides support for generating text embeddings through the OpenAI API.

### OpenAI Embeddings

The `openai` module includes functionality to generate vector embeddings:

```rust
pub async fn get_embedding(
    input: String,
    dimensions: Option<u32>,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>
```

This function takes:
- `input`: The text to generate embeddings for
- `dimensions`: Optional parameter to specify the number of dimensions (if omitted, uses the model default)

It returns a vector of floating point values representing the text embedding.

### Example Usage:

```rust
use llm_api_access::openai::get_embedding;

#[tokio::main]
async fn main() {
    // Generate an embedding with default dimensions
    match get_embedding("This is a sample text for embedding".to_string(), None).await {
        Ok(embedding) => {
            println!("Generated embedding with {} dimensions", embedding.len());
            // Use embedding for semantic search, clustering, etc.
        },
        Err(err) => eprintln!("Error generating embedding: {}", err),
    }
    
    // Generate an embedding with custom dimensions
    match get_embedding("Custom dimension embedding".to_string(), Some(64)).await {
        Ok(embedding) => {
            println!("Generated custom embedding with {} dimensions", embedding.len());
            assert_eq!(embedding.len(), 64);
        },
        Err(err) => eprintln!("Error generating embedding: {}", err),
    }
}
```

The function uses the "text-embedding-3-small" model by default and requires the same environment variables as other OpenAI API calls (`OPEN_AI_KEY` and `OPEN_AI_ORG`).

## Testing

The `llm_api_access` crate includes unit tests for various methods in the `Access` trait.  To run the tests, use:

```bash
cargo test
```
