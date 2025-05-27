# LLM API Access

The `llm_api_access` library (available as a Rust crate and Python package) provides a unified way to interact with different large language models (LLMs) like OpenAI, Gemini, and Anthropic. It aims to simplify the process of sending messages, managing conversations, and generating embeddings across various LLM providers.

## Key Features

* **Unified LLM Access:** Interact with OpenAI, Gemini, and Anthropic through a consistent interface.
* **Conversation Management:** Easily send single messages or manage multi-turn conversations.
* **Embeddings Generation:** Generate text embeddings using OpenAI.
* **Secure Credential Loading:** Utilizes `dotenv` to load API keys securely from a `.env` file.

## Installation

### Rust

Add `llm_api_access` to your `Cargo.toml` so it can install from [Crates](https://crates.io/crates/llm_api_access):

```toml
[dependencies]
llm_api_access = "0.1.XX" # Update this to be the latest version
tokio = { version = "1.28.0", features = ["full"] } # Required for async operations
```

### Python

Install from [PyPI](https://pypi.org/project/llm-api-access/):

```bash
pip install llm-api-access
```

## Loading API Credentials with dotenv

The `llm_api_access` library uses the `dotenv` library to securely load API credentials from a `.env` file in your project's root directory. This file should contain key-value pairs for each LLM provider you want to use.

**Example `.env` Structure:**

```
OPEN_AI_ORG=your_openai_org
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

**Steps:**

1.  **Create `.env` File:** Create a file named `.env` at the root of your project directory.
2.  **Add API Keys:** Fill in the `.env` file with the format shown above, replacing placeholders with your actual API keys.

**Important Note:**

* **Never** commit your `.env` file to version control systems like Git. It contains sensitive information like API keys.

## Rust Usage

The `llm_api_access` crate provides the `LLM` enum and the `Access` trait for interacting with LLMs.

### LLM Enum

This enum represents the supported LLM providers:

* `OpenAI`: Represents the OpenAI language model.
* `Gemini`: Represents the Gemini language model.
* `Anthropic`: Represents the Anthropic language model.

### Access Trait

The `Access` trait defines asynchronous methods for interacting with LLMs:

* `send_single_message`: Sends a single message and returns the generated response.
* `send_convo_message`: Sends a list of messages as a conversation and returns the generated response.
* `get_model_info`: Gets information about a specific LLM model.
* `list_models`: Lists all available LLM models.
* `count_tokens`: Counts the number of tokens in a given text.

The `LLM` enum implements `Access`, providing specific implementations for each method based on the chosen LLM provider.

**Note:** Currently, `get_model_info`, `list_models`, and `count_tokens` only work for the Gemini LLM. Other providers return an error indicating this functionality is not yet supported.

### `send_single_message` Example (Rust)

```rust
use llm_api_access::llm::{Access, LLM};

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

### `send_convo_message` Example (Rust)

```rust
use llm_api_access::llm::{Access, LLM};
use llm_api_access::structs::general::Message;

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

## Python Usage

The Python package exposes two main asynchronous functions: `call_llm` for interacting with LLMs and `get_embedding` for generating OpenAI embeddings.

### `call_llm` Function (Python)

```python
async def call_llm(llm_type: str, messages: list[dict]) -> str
```

This function takes:
* `llm_type`: A string representing the LLM provider ("openai", "gemini", or "anthropic").
* `messages`: A list of dictionaries, where each dictionary represents a message with "role" (e.g., "user", "model", "system") and "content".

It returns a string containing the LLM's response.

### `send_single_message` Example (Python)

```python
import asyncio
from llm_api_access import call_llm

async def main():
    # Send a single message to OpenAI
    messages = [{"role": "user", "content": "Hello, tell me a joke."}]
    response = await call_llm("openai", messages)
    print(f"OpenAI Joke: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### `send_convo_message` Example (Python)

```python
import asyncio
from llm_api_access import call_llm

async def main():
    # Send a conversation to Gemini
    messages = [
        {"role": "user", "content": "Write the first line of a story."},
        {"role": "model", "content": "Once upon a time..."},
        {"role": "user", "content": "Continue the story in 1600s France."},
    ]
    response = await call_llm("gemini", messages)
    print(f"Gemini Story Continuation: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Embeddings

The library provides support for generating text embeddings through the OpenAI API.

### OpenAI Embeddings (Rust)

The `openai` module includes functionality to generate vector embeddings:

```rust
pub async fn get_embedding(
    input: String,
    dimensions: Option<u32>,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>
```

This function takes:
* `input`: The text to generate embeddings for
* `dimensions`: Optional parameter to specify the number of dimensions (if omitted, uses the model default)

It returns a vector of floating point values representing the text embedding.

### Example Usage (Rust):

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

The `get_embedding` function uses the "text-embedding-3-small" model by default and requires the same environment variables as other OpenAI API calls (`OPENAI_API_KEY` and `OPEN_AI_ORG`).

### `get_embedding` Function (Python)

```python
async def get_embedding(input: str, dimensions: int | None) -> list[float]
```

This function takes:
* `input`: The text to generate embeddings for.
* `dimensions`: An optional integer parameter to specify the number of dimensions (if `None`, uses the model default).

It returns a list of floating-point values representing the text embedding.

### Example Usage (Python):

```python
import asyncio
from llm_api_access import get_embedding

async def main():
    # Generate an embedding with default dimensions
    try:
        embedding = await get_embedding("This is a test sentence.", None)
        print(f"Generated embedding with {len(embedding)} dimensions")
        print(f"Embedding snippet: {embedding[:5]}...") # Print first 5 elements
    except Exception as e:
        print(f"Error generating embedding: {e}")

    # Generate an embedding with custom dimensions
    try:
        embedding_with_dims = await get_embedding("Another test sentence.", 64)
        print(f"Generated custom embedding with {len(embedding_with_dims)} dimensions")
        assert len(embedding_with_dims) == 64
    except Exception as e:
        print(f"Error generating custom embedding: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing

The `llm_api_access` library includes unit tests that showcase usage and expected behavior with different LLM providers and the embedding functionality.
