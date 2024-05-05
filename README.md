
llm_api_access Crate
The llm crate provides an abstraction layer for interacting with different language models (LLMs) such as OpenAI, Gemini, and Anthropic. It defines a trait Access and an enum LLM that implements this trait for each supported LLM provider.
LLM Enum
The LLM enum represents the different language model providers that are supported. It has the following variants:

OpenAI: Represents the OpenAI language model.
Gemini: Represents the Gemini language model.
Anthropic: Represents the Anthropic language model.

Access Trait
The Access trait defines the following async methods for interacting with language models:

send_single_message(&self, message: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>: Sends a single message to the LLM and returns the generated response.
send_convo_message(&self, messages: Vec<Message>) -> Result<String, Box<dyn std::error::Error + Send + Sync>>: Sends a list of messages as a conversation to the LLM and returns the generated response.
get_model_info(&self, model: &str) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>>: Gets information about a specific LLM model.
list_models(&self) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>>: Lists all available LLM models.
count_tokens(&self, text: &str) -> Result<u32, Box<dyn std::error::Error + Send + Sync>>: Counts the number of tokens in a given text.

The LLM enum implements the Access trait, providing the corresponding implementation for each method based on the LLM provider.
Note
Currently, the get_model_info, list_models, and count_tokens methods are only implemented for the Gemini LLM. For other LLM providers, these methods return a GeminiError indicating that the functionality is not yet implemented.


Loading API Credentials with dotenv

The llm crate utilizes the dotenv library to securely load API credentials from a .env file located at the root of your Rust application. This file should contain key-value pairs for each LLM provider you intend to use. Here's an example structure:

OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

Steps:

    1. Create .env File: Create a file named .env at the root of your Rust project directory.

    2. Add API Keys: Fill in the .env file with the following format, replacing the placeholders with your actual API keys:

Important Note:

    Never commit your .env file to a version control system like Git. It contains sensitive information like API keys.

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
            role: "system".to_string(),
            content: "You are a helpful coding assistant.".to_string(),
        },
        Message {
            role: "user".to_string(),
            content: "Can you write a Python function to reverse a string?".to_string(),
        },
        Message {
            role: "user".to_string(),
            content: "Also, provide an example of how to use that function.".to_string(),
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

In this example, we create an instance of the `LLM::Gemini` provider and define a vector of `Message` structs representing a conversation. We then send these messages to the LLM using the `send_convo_message` method. The generated response (containing the Python function and example) is matched and printed or an error message is displayed if an error occurred.

Note that for these examples to work, you'll need to have the necessary API keys and configuration set up for the respective LLM providers.

Testing
The crate includes a tests module with several test cases for the different methods of the Access trait. These tests demonstrate how to use the crate and verify the expected behavior of the LLM providers.
To run the tests, you'll need to have the necessary API keys and configuration set up for the respective LLM providers.


Dependencies
The llm crate relies on the following dependencies:

async_trait: Provides the async_trait macro for defining async traits.
crate::openai: Provides the call_gpt function for interacting with the OpenAI LLM.
crate::gemini: Provides the call_gemini, conversation_gemini_call, get_gemini_model_info, list_gemini_models, and count_gemini_tokens functions for interacting with the Gemini LLM.
crate::anthropic: Provides the call_anthropic function for interacting with the Anthropic LLM.
crate::models: Provides the ModelInfo struct for representing model information.
crate::errors: Provides the GeminiError struct for handling errors.
crate::structs: Provides the Message, Content, and Part structs for representing messages and content.