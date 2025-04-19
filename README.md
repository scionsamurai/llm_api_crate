The `llm_api_access` crate provides a unified way to interact with different large language models (LLMs) like OpenAI, Gemini, and Anthropic.

### LLM Enum

This enum represents the supported LLM providers:

- `OpenAI`: Represents the OpenAI language model.
- `Gemini`: Represents the Gemini language model.
- `Anthropic`: Represents the Anthropic language model.

### Access Trait

The `Access` trait defines asynchronous methods for interacting with LLMs:

- `send_single_message`: Sends a single message and returns the generated response.
- `send_convo_message`: Sends a list of messages as a conversation and returns the generated response.
- `get_model_info`: Gets information about a specific LLM model.
- `list_models`: Lists all available LLM models.
- `count_tokens`: Counts the number of tokens in a given text.

The `LLM` enum implements `Access`, providing specific implementations for each method based on the chosen LLM provider.

**Note:** Currently, `get_model_info`, `list_models`, and `count_tokens` only work for the Gemini LLM. Other providers return an error indicating this functionality is not yet supported.

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
use llm::{LLM, Access};

#[tokio::main]
async fn main() {
    // Create an instance of the OpenAI LLM
    let llm = LLM::OpenAI;

    // Send a single message to the LLM
    let response = llm.send_single_message("Tell me a joke about programmers").await;

    match response {
        Ok(joke) => println!("Joke: {}", joke),
        Err(err) => eprintln!("Error: {}", err),
    }
}
```

This example creates an instance of the `LLM::OpenAI` provider and sends a single message using the `send_single_message` method. It then matches the result, printing the generated joke or an error message if an error occurred.


### `send_convo_message` Example

```rust
use llm::{LLM, Access, Message};

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

    // Send the conversation messages to the LLM
    let response = llm.send_convo_message(messages).await;

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
use llm_api_crate::openai::get_embedding;

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

The `llm_api_access` crate includes unit tests for various methods in the `Access` trait. These tests showcase usage and expected behavior with different LLM providers.