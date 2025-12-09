// src/tests/llm_gemini.rs
#[cfg(test)]
mod tests {
    use crate::llm::{Access, LLM};
    use crate::structs::general::Message;

    #[tokio::test]
    async fn test_send_single_message_gemini() {
        let llm: LLM = LLM::Gemini;

        // Pass None for the API key to use the environment variable
        let res = llm.send_single_message("Hi there, this is a test. Please generate a limrik.", None, None).await;
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

        // Pass None for the API key to use the environment variable
        let res = llm.send_convo_message(messages, None, None).await;
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
        // If you were to extend api_key_override to get_model_info,
        // you would pass it here as well. For now, it's not part of the query.
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
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_list_models() {
        let llm = LLM::Gemini;
        // If you were to extend api_key_override to list_models,
        // you would pass it here as well. For now, it's not part of the query.
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
        let model = "models/gemini-2.5-flash";
        // If you were to extend api_key_override to count_tokens,
        // you would pass it here as well. For now, it's not part of the query.
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