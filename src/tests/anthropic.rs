// src/tests/anthropic.rs

#[cfg(test)]
mod tests {
    use crate::anthropic::call_anthropic;
    use crate::structs::general::Message;


    #[tokio::test]
    async fn test_call_anthropic() {
        let user_message = Message {
            role: "user".to_string(),
            content: "Hello, Claude. Can you tell me a joke?".to_string(),
        };

        let messages = vec![user_message];

        let res = call_anthropic(messages, None).await;
        match res {
            Ok(response) => {
                println!("response: {:}", response);
                assert!(!response.is_empty(), "Response should not be empty");
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
                content: "Write the first line of a story about a magic backpack.".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "In the bustling city of Meadow brook, lived a young girl named Sophie. She was a bright and curious soul with an imaginative mind.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Can you set it in a quiet village in 1600s France?".to_string(),
            },
        ];

        let res = call_anthropic(messages.clone(), None).await;
        match res {
            Ok(response) => {
                assert!(!response.is_empty(), "Response should not be empty");
                println!("Response1: {}", &response);
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: response
                });

                let user_message_2 = Message {
                    role: "user".to_string(),
                    content: "Can you also make the story about pokemon?".to_string(),
                };

                messages.push(user_message_2);

                let res = call_anthropic(messages, None).await;
                match res {
                    Ok(response) => {
                        assert!(!response.is_empty(), "Response should not be empty");
                        println!("Response2: {}", response);
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
}