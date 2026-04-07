// src/tests/llm_gemini.rs
#[cfg(test)]
mod tests {
    use crate::llm::{Access, LLM};
    use crate::structs::general::{ Message, MessageContent };
    use crate::config::LlmConfig;

    #[tokio::test]
    async fn test_send_single_message_gemini() {
        let llm: LLM = LLM::Gemini;

        // Test without config
        let res = llm.send_single_message("Hi there, this is a test. Please generate a limerick.", None, None).await;
        match res {
            Ok(response) => {
                println!("Ok: {}", &response.text);
                if let Some(reasoning) = &response.reasoning {
                    println!("Reasoning: {}", reasoning);
                }
                assert!(!response.text.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                panic!("Call to Gemini API failed: {}", err);
            }
        }

        // Test with config
        let config = LlmConfig::new().with_temperature(0.5);
        let res = llm.send_single_message("Hi there, this is a test. Please generate a limerick.", None, Some(&config)).await;
        match res {
            Ok(response) => {
                println!("Ok with config: {}", &response.text);
                if let Some(reasoning) = &response.reasoning {
                    println!("Reasoning with config: {}", reasoning);
                }
                assert!(!response.text.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error with config: {}", err);
                panic!("Call to Gemini API with config failed: {}", err);
            }
        }
    }

    #[tokio::test]
    async fn test_send_convo_message_gemini() {
        let llm = LLM::Gemini;
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("Write the first line of a story about a magic backpack.".to_string()),
            },
            Message {
                role: "model".to_string(),
                content: MessageContent::Text("In the bustling city of Meadow brook, lived a young girl named Sophie. She was a bright and curious soul with an imaginative mind.".to_string()),
            },
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("Can you set it in a quiet village in 1600s France?".to_string()),
            },
        ];

        // Test without config
        let res = llm.send_convo_message(messages.clone(), None, None).await;
        match res {
            Ok(response) => {
                println!("Ok: {}", &response.text);
                if let Some(reasoning) = &response.reasoning {
                    println!("Reasoning: {}", reasoning);
                }
                assert!(!response.text.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                panic!("Call to Gemini API failed: {}", err);
            }
        }

        // Test with config
        let config = LlmConfig::new(); 
        let res = llm.send_convo_message(messages, None, Some(&config)).await;
        match res {
            Ok(response) => {
                println!("Ok with config: {}", &response.text);
                if let Some(reasoning) = &response.reasoning {
                    println!("Reasoning with config: {}", reasoning);
                }
                assert!(!response.text.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error with config: {}", err);
                panic!("Call to Gemini API with config failed: {}", err);
            }
        }
    }

    #[tokio::test]
    async fn test_get_model_info() {
        let llm = LLM::Gemini;
        let gemini_models = llm.list_models().await.unwrap();
        let model_name = gemini_models[0].name.clone(); // name like "models/gemini-2.0-flash"
        let res = llm.get_model_info(&model_name).await;
        match res {
            Ok(model_info) => {
                println!("Ok: {:?}", &model_info);
                assert_eq!(model_info.name, model_name);
            }
            Err(err) => {
                println!("Error: {}", err);
                panic!("Failed to get model info: {}", err);
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
                panic!("Failed to list models: {}", err);
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
                panic!("Failed to count tokens: {}", err);
            }
        }
    }
}