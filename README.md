# llm_api_access

The `llm_api_access` crate provides a unified way to interact with different large language models (LLMs) like OpenAI, Gemini, Anthropic, and local Llama servers. 

## Current Status

This crate is used to power an open-source coding assistant currently in active development. Gemini has been the main test target. OpenAI, Gemini, and Llama Server are supported for both text generation and embeddings. Anthropic is supported for text generation. Recent updates include unified support for "thinking" or "reasoning" blocks from models like OpenAI's `o1`/`o3`, Anthropic's Claude 3.7, and Google's Gemini 2.0 Flash Thinking. Development is self-encouraged so updates can be far and few between, open an issue on github if you want something specific.

### Unified Response Structure

To support models that output both a thought process and a final answer, responses from the text generation methods are returned as an `LlmResponse`:

```rust
pub struct LlmResponse {
    pub text: String,
    pub reasoning: Option<String>,
}
```

### LLM Enum

This enum represents the supported LLM providers:

- `OpenAI`: Represents the OpenAI language models.
- `Gemini`: Represents the Gemini language models.
- `Anthropic`: Represents the Anthropic language models.
- `LlamaServer`: Represents a local or remote Llama-compatible server.

### Access Trait

The `Access` trait defines asynchronous methods for interacting with LLMs:

- `send_single_message`: Sends a single message and returns the generated structured response.
  ```rust
  async fn send_single_message(
        &self,
        message: &str,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>;
  ```
- `send_convo_message`: Sends a list of messages as a conversation and returns the generated structured response.
  ```rust
  async fn send_convo_message(
        &self,
        messages: Vec<Message>,
        model: Option<&str>,
        config: Option<&LlmConfig>,
    ) -> Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>;
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
  - `embed`: Generates a vector embedding for the provided text.
  ```rust
  async fn embed(
        &self,
        text: &str,
        model: Option<&str>,
        dimensions: Option<u32>,
        config: Option<&LlmConfig>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>;

The `LLM` enum implements `Access`, providing specific implementations for each method based on the chosen LLM provider.

**Note:** Currently, `get_model_info`, `list_models`, and `count_tokens` only work for the Gemini LLM. Other providers return an error indicating this functionality is not yet supported.

### LlmConfig

The `LlmConfig` struct allows you to configure provider-specific settings for the LLM calls. It uses a builder pattern for easy customization.

```rust
#[derive(Debug, Clone, Default)]
pub struct LlmConfig {
    pub temperature: Option<f64>,
    pub thinking_budget: Option<i32>,
    pub grounding_with_search: Option<bool>, // Enable grounding with Google Search for Gemini
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,       
    pub stop: Option<Vec<String>,
    pub cache_prompt: Option<bool>,
    pub json_schema: Option<serde_json::Value>,    
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
}
```
**Thinking Budgets & Reasoning:**
Passing a `thinking_budget` automatically configures the underlying provider (like Anthropic) to return reasoning tokens before the final text answer. These reasoning tokens will be populated in the `reasoning` field of the returned `LlmResponse`. 

**Example Usage:**

```rust
use llm_api_access::config::LlmConfig;

// Default usage (no config)
let config = None;

// With thinking budget (Enables reasoning blocks on compatible models)
let config = Some(LlmConfig::new().with_thinking_budget(1024));

// With Google Search grounding enabled for Gemini
let config = Some(LlmConfig::new().with_grounding_with_search(true));

// Universal parameters
let config = Some(LlmConfig::new()
    .with_temperature(0.7)
    .with_max_tokens(2048));
```

### Loading API Credentials with dotenv

The `llm_api_access` crate uses the `dotenv` library to securely load API credentials from a `.env` file in your project's root directory. This file should contain key-value pairs for each LLM provider you want to use.

**Example Structure:**

```
OPEN_AI_ORG=your_openai_org
OPEN_AI_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
LLAMA_SERVER_URL=http://127.0.0.1:8080
```

## Example Usage

### `send_single_message` Example

```rust
use llm_api_access::llm::{Access, LLM};
use llm_api_access::config::LlmConfig; 

#[tokio::main]
async fn main() {
    // Create an instance of the OpenAI LLM
    let llm = LLM::OpenAI;

    // Send a single message to the LLM
    let response = llm.send_single_message("Tell me a joke about programmers", None, None).await;

    match response {
        Ok(res) => println!("Joke: {}", res.text),
        Err(err) => eprintln!("Error: {}", err),
    }

    // Send a message asking for reasoning to a thinking model
    let config = Some(LlmConfig::new().with_thinking_budget(1024));
    let response = llm.send_single_message("Calculate how many ping pong balls fit in a bus.", Some("o3-mini"), config.as_ref()).await;

    match response {
        Ok(res) => {
            if let Some(reasoning) = res.reasoning {
                println!("Thought Process:\n{}", reasoning);
            }
            println!("Final Answer:\n{}", res.text);
        },
        Err(err) => eprintln!("Error: {}", err),
    }
}
```

### `send_convo_message` Example

```rust
use llm_api_access::llm::{Access, LLM};
use llm_api_access::structs::general::Message;
use llm_api_access::config::LlmConfig; 

#[tokio::main]
async fn main() {
    // Create an instance of the Gemini LLM
    let llm = LLM::Gemini;

    // Define the conversation messages
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: "You are a helpful coding assistant.".into(),
        },
        Message {
            role: "model".to_string(),
            content: "You got it! I am ready to assist!".into(),
        },
        Message {
            role: "user".to_string(),
            content: "Generate a rust function that reverses a string.".into(),
        },
    ];

    // Send the conversation messages
    let response = llm.send_convo_message(messages.clone(), None, None).await;

    match response {
        Ok(res) => println!("Code: {}", res.text),
        Err(err) => eprintln!("Error: {}", err),
    }
}
```

## Embeddings

The crate provides unified support for generating text embeddings across multiple providers via the `Access` trait. This allows you to switch between OpenAI, Gemini, or a local Llama server without changing your application logic.

### Supported Providers

| Provider | Default Model | Key Features |
| :--- | :--- | :--- |
| **OpenAI** | `text-embedding-3-small` | High-performance industry standard. |
| **Gemini** | `text-embedding-004` | Supports **Matryoshka Representation Learning** (dimensionality reduction). |
| **LlamaServer** | User-defined | Local, private embedding generation. |

### Unified Usage Example

Regardless of the provider, you use the same `.embed()` method:

```rust
use llm_api_access::llm::{Access, LLM};
use llm_api_access::config::LlmConfig;

#[tokio::main]
async fn main() {
    // 1. Using OpenAI
    let openai = LLM::OpenAI;
    let vec_oa = openai.embed("Hello world", None, None, None).await.unwrap();
    println!("OpenAI dims: {}", vec_oa.len());

    // 2. Using Gemini with dimensionality reduction (Matryoshka)
    let gemini = LLM::Gemini;
    let requested_dims = 256;
    let vec_gem = gemini.embed("Hello world", Some("text-embedding-004"), Some(requested_dims), None).await.unwrap();
    println!("Gemini dims: {}", vec_gem.len());
    assert_eq!(vec_gem.len(), 256);

    // 3. Using a local Llama Server
    let llama = LLM::LlamaServer;
    let vec_llama = llama.embed("Hello world", None, None, None).await.unwrap();
    println!("Llama dims: {}", vec_llama.len());
}
```

## Testing

The `llm_api_access` crate includes unit tests for various methods in the `Access` trait.  To run the tests, use:

```bash
cargo test
```
