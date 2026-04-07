// src/tests/openai.rs

#[cfg(test)]
mod tests {
    use crate::openai::{call_gpt, get_embedding};
    use crate::structs::general::{ Message, MessageContent };

    #[tokio::test]
    async fn test_call_gpt() {
        let user_message = Message {
            role: "user".to_string(),
            content: MessageContent::Text("Hello, can you tell me a joke?".to_string()),
        };
        let messages = vec![user_message];
        
        // UPDATED: Added None, None for model and config
        let res = call_gpt(messages, None, None).await;
        match res {
            Ok(response) => assert!(!response.text.is_empty(), "Response should not be empty"),
            Err(err) => panic!("Call to OpenAI API failed: {}", err),
        }
    }

    #[tokio::test]
    async fn test_call_gpt_multi_prompt() {
        let system_message = Message {
            role: "system".to_string(),
            content: MessageContent::Text("You are a helpful coding assistant.".to_string()),
        };
        let user_message_1 = Message {
            role: "user".to_string(),
            content: MessageContent::Text("Hello, can you write a python function that reverses a string?".to_string()),
        };
        let mut messages = vec![system_message, user_message_1];
        
        // UPDATED: Added None, None for model and config
        let res = call_gpt(messages.clone(), None, None).await;
        match res {
            Ok(response) => {
                assert!(!response.text.is_empty(), "Response should not be empty");
                let user_message_2 = Message {
                    role: "user".to_string(),
                    content: MessageContent::Text("Can you also provide an example of how to use that function?".to_string()),
                };
                messages.push(user_message_2);
                
                // UPDATED: Added None, None for model and config
                let res = call_gpt(messages, None, None).await;
                match res {
                    Ok(response) => assert!(!response.text.is_empty(), "Response should not be empty"),
                    Err(err) => panic!("Call to OpenAI API failed on second prompt: {}", err),
                }
            }
            Err(err) => panic!("Call to OpenAI API failed on first prompt: {}", err),
        }
    }

    // --- NEW TEST: Explicitly test the reasoning implementation ---
    #[tokio::test]
    async fn test_call_gpt_reasoning() {
        let user_message = Message {
            role: "user".to_string(),
            content: MessageContent::Text("Calculate how many ping pong balls fit in a Boeing 747. Show your step-by-step math.".to_string()),
        };

        let messages = vec![user_message];

        // We explicitly test with an o-series model to test reasoning fields
        // Note: Depending on your API tier, o3-mini may or may not return raw reasoning tokens to the developer, 
        // but this verifies the struct parses successfully without crashing.
        let res = call_gpt(messages, Some("o3-mini"), None).await;
        match res {
            Ok(response) => {
                println!("--- response text ---\n{}", response.text);
                println!("--- response reasoning ---\n{:?}", response.reasoning);
                
                assert!(!response.text.is_empty(), "Response text should not be empty");
                // We won't strictly assert response.reasoning.is_some() because OpenAI sometimes hides 
                // the raw reasoning string depending on safety/tier restrictions on o1/o3 models,
                // but we verify the response is formed correctly as an LlmResponse.
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to OpenAI API with reasoning model failed");
            }
        }
    }

    #[tokio::test]
    async fn test_get_embedding() {
        let input_text = "This is a test sentence.";
        let res = get_embedding(input_text.to_string(), None).await;
        match res {
            Ok(embedding) => {
                assert!(!embedding.is_empty(), "Embedding should not be empty");
                println!("Embedding vector length: {}", embedding.len());
                // Basic sanity check on the embedding vector (length might change with models)
                assert!(embedding.len() > 100);
            }
            Err(err) => panic!("Failed to get embedding: {}", err),
        }
    }

    #[tokio::test]
    async fn test_get_embedding_with_dimensions() {
        let input_text = "This is another test.";
        let dimensions: u32 = 64;
        let res = get_embedding(input_text.to_string(), Some(dimensions)).await;
        match res {
            Ok(embedding) => {
                assert!(!embedding.is_empty(), "Embedding should not be empty");
                assert_eq!(embedding.len() as u32, dimensions, "Embedding dimension should match requested dimension");
                println!("Embedding vector length: {}", embedding.len());
            }
            Err(err) => panic!("Failed to get embedding with dimensions: {}", err),
        }
    }
}