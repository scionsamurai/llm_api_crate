#[cfg(test)]
mod tests {
    use crate::anthropic::call_anthropic;
    use crate::structs::general::{ Message, MessageContent };
    use crate::config::LlmConfig; // <-- Added to test thinking blocks

    #[tokio::test]
    async fn test_call_anthropic() {
        let user_message = Message {
            role: "user".to_string(),
            content: MessageContent::Text("Hello, Claude. Can you tell me a joke?".to_string()),
        };

        let messages = vec![user_message];

        // UPDATED: Added None, None for model and config
        let res = call_anthropic(messages, None, None).await;
        match res {
            Ok(response) => {
                println!("response text: {:}", response.text);
                println!("response reasoning: {:?}", response.reasoning);
                assert!(!response.text.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Anthropic API failed");
            }
        }
    }

    #[tokio::test]
    async fn test_call_anthropic_multi_prompt() {
        let mut messages = vec![
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("Write the first line of a story about a magic backpack.".to_string()),
            },
            Message {
                role: "assistant".to_string(),
                content: MessageContent::Text("In the bustling city of Meadow brook, lived a young girl named Sophie. She was a bright and curious soul with an imaginative mind.".to_string()),
            },
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("Can you set it in a quiet village in 1600s France?".to_string()),
            },
        ];

        // UPDATED: Added None, None for model and config
        let res = call_anthropic(messages.clone(), None, None).await;
        match res {
            Ok(response) => {
                assert!(!response.text.is_empty(), "Response should not be empty");
                println!("Response1: {}", &response.text);
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Text(response.text),
                });

                let user_message_2 = Message {
                    role: "user".to_string(),
                    content: MessageContent::Text("Can you also make the story about pokemon?".to_string()),
                };

                messages.push(user_message_2);

                // UPDATED: Added None, None for model and config
                let res = call_anthropic(messages, None, None).await;
                match res {
                    Ok(response) => {
                        assert!(!response.text.is_empty(), "Response should not be empty");
                        println!("Response2: {}", response.text);
                    }
                    Err(err) => {
                        println!("Error: {}", err);
                        assert!(false, "Call to Anthropic API failed");
                    }
                }
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Anthropic API failed");
            }
        }
    }

    // --- NEW TEST: Explicitly test the reasoning implementation ---
    #[tokio::test]
    async fn test_call_anthropic_thinking() {
        let user_message = Message {
            role: "user".to_string(),
            content: MessageContent::Text("Calculate how many golf balls fit in a school bus. Show your work.".to_string()),
        };

        let messages = vec![user_message];

        // Create a config with a thinking budget (minimum 1024 for Anthropic)
        let config = LlmConfig::new().with_thinking_budget(1024);

        // We explicitly test with claude-3-7-sonnet as it supports thinking blocks
        let res = call_anthropic(messages, Some("claude-haiku-4-5"), Some(&config)).await;
        match res {
            Ok(response) => {
                println!("--- response text ---\n{:}", response.text);
                println!("--- response reasoning ---\n{:?}", response.reasoning);
                
                assert!(!response.text.is_empty(), "Response text should not be empty");
                assert!(response.reasoning.is_some(), "Reasoning block should be populated");
                assert!(!response.reasoning.unwrap().is_empty(), "Reasoning string should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Anthropic API with thinking failed");
            }
        }
    }
}