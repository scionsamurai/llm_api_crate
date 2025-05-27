// src/gemini/mod.rs
pub mod api;
pub mod types;
pub mod request;
pub mod response;

pub use api::*;
pub use types::*;

// https://ai.google.dev/tutorials/rest_quickstart

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::general::{ Message, Content, Part };

    #[tokio::test]
    async fn test_call_gemini() {
        let message: Message = Message {
            role: "user".to_string(),
            content: "Hi there, this is a test. Please generate a limrick about the muppets.".to_string(),
        };

        let messages: Vec<Message> = vec![message];

        let res = call_gemini(messages).await;
        match res {
            Ok(res_str) => {
                println!("res: {}", res_str);
                assert!(!res_str.is_empty());
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_conversation_gemini_call() {
        let messages = vec![
            Content {
                role: "user".to_string(),
                parts: vec![Part {
                    text: "Write the first line of a story about a magic backpack.".to_string(),
                }],
            },
            Content {
                role: "model".to_string(),
                parts: vec![Part {
                    text: "In the bustling city of Meadow brook, lived a young girl named Sophie. She was a bright and curious soul with an imaginative mind.".to_string(),
                }],
            },
            Content {
                role: "user".to_string(),
                parts: vec![Part {
                    text: "Can you set it in a quiet village in 1600s France?".to_string(),
                }],
            },
        ];

        let res = conversation_gemini_call(messages).await;
        match res {
            Ok(response) => {
                assert!(!response.is_empty());
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }

}