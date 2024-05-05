## llm_api_access Crate

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
// ... (assuming llm_api_access crate is imported)

// Create a Gemini LLM provider instance
let llm = LLM::Gemini;

// Define conversation messages
let messages = vec![
    Message { content: Content::Text("Hi! Can you write a Python function to greet someone?".to_string()) },
    Message { content: Content::Text("Sure, here's an example:".to_string()) },
];

// Send the conversation and handle response
match llm.send_convo_message(messages) {
    Ok(response) => println!("{}", response),
    Err(err) => println!("Error: {}", err),
}
```

**Note:** This example requires API keys and configuration for the Gemini LLM provider.

## Testing

The `llm_api_access` crate includes unit tests for various methods in the `Access` trait. These tests showcase usage and expected behavior with different LLM providers.

**Requirements:** To run the tests, you'll need the necessary API keys and configuration set up for the respective LLM providers.

## Dependencies

The `llm_api_access` crate depends on several other crates:

- `async_trait`: Provides the `async_trait` macro for defining asynchronous traits.
- `llm::openai`: Provides functions for interacting with the OpenAI LLM.
- `llm::gemini`: Provides functions for interacting with the Gemini LLM (calling, conversation flow, model info, model listing, token counting).
- `llm::anthropic`: Provides functions for interacting with the Anthropic LLM.
- `llm::models`: Provides the `ModelInfo` struct for representing model information.
- `llm::errors`: Provides the `GeminiError` struct for handling errors.
- `llm::structs`: Provides structs for representing messages and content (`Message`, `Content`, `Part`).
