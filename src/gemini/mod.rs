// src/gemini.rs
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
    use crate::structs::{ Message, Content, Part };

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

    #[tokio::test]
    async fn test_get_gemini_model_info() {
        let gemini_models = list_gemini_models().await.unwrap();
        let model_name = gemini_models[0].name.clone(); // name like "models/gemini-2.0-flash"
        let res = get_gemini_model_info(&model_name).await;
        match res {
            Ok(model_info) => {
                println!("Ok: {:?}", &model_info);
                assert_eq!(model_info.name, model_name);
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_list_gemini_models() {
        let res = list_gemini_models().await;
        match res {
            Ok(models) => {
                // print model names in human readable format seperated by newlines

                let model_names: String = models
                    .iter()
                    .map(|model| format!("{}\n", model.name))
                    .collect();
                println!("Gemini Models\n{}", &model_names);
                assert!(!models.is_empty());
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_count_gemini_tokens() {
        let text = "Write a story about a magic backpack.";
        let model = "models/gemini-2.0-flash";
        let res = count_gemini_tokens(text, model).await;
        match res {
            Ok(token_count) => {
                println!("token_count: {:?}", &token_count);
                assert!(token_count > 0);
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }
}