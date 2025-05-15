// src/llm.rs
use async_trait::async_trait;
use crate::openai::call_gpt;
use crate::gemini::{call_gemini, conversation_gemini_call, get_gemini_model_info, list_gemini_models, count_gemini_tokens};
use crate::anthropic::call_anthropic;
use crate::models::ModelInfo;
use crate::errors::GeneralError;
use crate::structs::{Message, Content, Part};

pub enum LLM {
    OpenAI,
    Gemini,
    Anthropic,
}

#[async_trait]
pub trait Access {
    async fn send_single_message(
        &self,
        message: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn send_convo_message(
        &self,
        messages: Vec<Message>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn get_model_info(
        &self,
        model: &str,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>>;
    async fn list_models(&self)
        -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>>;
    async fn count_tokens(
        &self,
        text: &str,
        model: &str,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait]
impl Access for LLM {
    async fn send_single_message(
        &self,
        message: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => {
                let openai_message: Message = Message {
                    role: "user".to_string(),
                    content: message.to_string(),
                };
                call_gpt(vec![openai_message]).await
            }
            LLM::Gemini => {
                let gemini_message: Message = Message {
                    role: "user".to_string(),
                    content: message.to_string(),
                };
                call_gemini(vec![gemini_message]).await
            }
            LLM::Anthropic => {
                let anthropic_message: Message = Message {
                    role: "user".to_string(),
                    content: message.to_string(),
                };
                call_anthropic(vec![anthropic_message]).await
            }
        }
    }
    
    async fn send_convo_message(
        &self,
        messages: Vec<Message>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::OpenAI => call_gpt(messages).await,
            LLM::Gemini => {
                let gemini_messages: Vec<Content> = messages
                    .into_iter()
                    .map(|msg| Content {
                        role: msg.role,
                        parts: vec![Part {
                            text: msg.content,
                        }],
                    })
                    .collect();
    
                conversation_gemini_call(gemini_messages).await
            }
            LLM::Anthropic => call_anthropic(messages).await,
        }
    }

    async fn get_model_info(
        &self,
        model: &str,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => get_gemini_model_info(model).await,
            _ => Err(Box::new(GeneralError {
                message: format!("Currently only Gemini is implemented for get_model_info func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }

    async fn list_models(
        &self,
    ) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => list_gemini_models().await,
            _ => Err(Box::new(GeneralError {
                message: format!("Currently only Gemini is implemented for list_models func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }

    async fn count_tokens(
        &self,
        text: &str,
        model: &str,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            LLM::Gemini => count_gemini_tokens(text, model).await,
            _ => Err(Box::new(GeneralError {
                message: format!("Currently only Gemini is implemented for count_tokens func"),
            }) as Box<dyn std::error::Error + Send + Sync>),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_send_single_message_openai() {
        let llm: LLM = LLM::OpenAI;

        let res = llm.send_single_message("Hello, can you tell me a joke?").await;
        match res {
            Ok(response) => {
                println!("Ok: {}", &response);
                assert!(!response.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to OpenAI API failed");
            }
        }
    }

    #[tokio::test]
    async fn test_send_single_message_gemini() {
        let llm: LLM = LLM::Gemini;

        let res = llm.send_single_message("Hi there, this is a test. Please generate a limrik.").await;
        match res {
            Ok(response) => {
                println!("Ok: {}", &response);
                assert!(!response.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Gemini API failed");
            }
        }
    }

    #[tokio::test]
    async fn test_send_single_message_anthropic() {
        let llm: LLM = LLM::Anthropic;

        let res = llm.send_single_message("Hi there, this is a test. Please generate a limrik.").await;
        match res {
            Ok(response) => {
                println!("Ok: {}", &response);
                assert!(!response.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Gemini API failed");
            }
        }
    }

    #[tokio::test]
    async fn test_send_convo_message_openai() {
        let llm = LLM::OpenAI;

        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful coding assistant.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Hello, can you write a python function that reverses a string?".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Can you also provide an example of how to use that function?".to_string(),
            },
        ];

        let res = llm.send_convo_message(messages).await;
        match res {
            Ok(response) => {
                println!("Ok: {}", &response);
                assert!(!response.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to OpenAI API failed");
            }
        }
    }

    #[tokio::test]
    async fn test_send_convo_message_gemini() {
        let llm = LLM::Gemini;
        let messages = vec![
            Message {
                role: "user".to_string(),
                content:"Write the first line of a story about a magic backpack.".to_string(),
            },
            Message {
                role: "model".to_string(),
                content:"In the bustling city of Meadow brook, lived a young girl named Sophie. She was a bright and curious soul with an imaginative mind.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content:"Can you set it in a quiet village in 1600s France?".to_string(),
            },
        ];

        let res = llm.send_convo_message(messages).await;
        match res {
            Ok(response) => {
                println!("Ok: {}", &response);
                assert!(!response.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Gemini API failed");
            }
        }
    }

    #[tokio::test]
    async fn test_send_convo_message_anthropic() {
        let llm: LLM = LLM::Anthropic;
        let messages = vec![
            Message {
                role: "user".to_string(),
                content:"Write the first line of a story about a magic backpack.".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content:"In the bustling city of Meadow brook, lived a young girl named Sophie. She was a bright and curious soul with an imaginative mind.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content:"Can you set it in a quiet village in 1600s France?".to_string(),
            },
        ];

        let res = llm.send_convo_message(messages).await;
        match res {
            Ok(response) => {
                println!("Ok: {}", &response);
                assert!(!response.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Gemini API failed");
            }
        }
    }

    #[tokio::test]
    async fn test_get_model_info() {
        let llm = LLM::Gemini;
        let res = llm.get_model_info("gemini-1.0-pro-001").await;
        match res {
            Ok(model_info) => {
                println!("Ok: {:?}", &model_info);
                assert_eq!(model_info.name, "models/gemini-1.0-pro-001");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Failed to get model info");
            }
        }
    }

    #[tokio::test]
    async fn test_list_models() {
        let llm = LLM::Gemini;
        let res = llm.list_models().await;
        match res {
            Ok(models) => {
                println!("Ok: {:?}", &models);
                assert!(!models.is_empty(), "Models list should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Failed to list models");
            }
        }
    }

    #[tokio::test]
    async fn test_count_tokens() {
        let llm = LLM::Gemini;
        let text = "Write a story about a magic backpack.";
        let model = "models/gemini-2.0-flash";
        let res = llm.count_tokens(text, model).await;
        match res {
            Ok(token_count) => {
                println!("Ok: {}", &token_count);
                assert!(token_count > 0, "Token count should be greater than zero");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Failed to count tokens");
            }
        }
    }
}