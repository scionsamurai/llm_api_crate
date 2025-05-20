
#[cfg(test)]
mod tests {
    use crate::openai::{call_gpt, get_embedding};
    use crate::structs::general::Message;

    #[tokio::test]
    async fn test_call_gpt() {
        let user_message = Message {
            role: "user".to_string(),
            content: "Hello, can you tell me a joke?".to_string(),
        };
        let messages = vec![user_message];
        let res = call_gpt(messages).await;
        match res {
            Ok(response) => assert!(!response.is_empty(), "Response should not be empty"),
            Err(err) => panic!("Call to OpenAI API failed: {}", err),
        }
    }

    #[tokio::test]
    async fn test_call_gpt_multi_prompt() {
        let system_message = Message {
            role: "system".to_string(),
            content: "You are a helpful coding assistant.".to_string(),
        };
        let user_message_1 = Message {
            role: "user".to_string(),
            content: "Hello, can you write a python function that reverses a string?".to_string(),
        };
        let mut messages = vec![system_message, user_message_1];
        let res = call_gpt(messages.clone()).await;
        match res {
            Ok(response) => {
                assert!(!response.is_empty(), "Response should not be empty");
                let user_message_2 = Message {
                    role: "user".to_string(),
                    content: "Can you also provide an example of how to use that function?".to_string(),
                };
                messages.push(user_message_2);
                let res = call_gpt(messages).await;
                match res {
                    Ok(response) => assert!(!response.is_empty(), "Response should not be empty"),
                    Err(err) => panic!("Call to OpenAI API failed on second prompt: {}", err),
                }
            }
            Err(err) => panic!("Call to OpenAI API failed on first prompt: {}", err),
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