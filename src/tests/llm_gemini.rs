// src/tests/llm_gemini.rs
#[cfg(test)]
mod tests {
    use crate::llm::{Access, LLM};
    use crate::structs::general::{Message, MessageContent, LlmChunk};
    use crate::config::LlmConfig;
    use futures::stream::StreamExt;

    #[tokio::test]
    async fn test_send_single_message_gemini() {
        let llm: LLM = LLM::Gemini;

        // Test without config
        let res = llm.send_single_message("Hi there, this is a test. Please generate a limerick.".to_string().into(), None, None).await;
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
        let res = llm.send_single_message("Hi there, this is a test. Please generate a limerick.".to_string().into(), None, Some(&config)).await;
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

    #[tokio::test]
    async fn test_gemini_multimodal_message() {
        use crate::structs::general::{MessagePart, ImageSource};
        let llm = LLM::Gemini;
        let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Array(vec![
                MessagePart {
                    r#type: "text".to_string(),
                    text: Some("Describe this image:".to_string()),
                    image_url: None,
                },
                MessagePart {
                    r#type: "image_url".to_string(),
                    text: None,
                    image_url: Some(ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: base64_data.to_string(),
                    }),
                },
            ]),
        }];

        let res = llm.send_convo_message(messages, None, None).await;
        assert!(res.is_ok(), "Gemini failed to process multimodal message: {:?}", res.err());
    }

    #[tokio::test]
    async fn test_gemini_streaming() {
        let llm = LLM::Gemini;
        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text("Explain quantum entanglement in one sentence.".to_string()),
        }];

        let mut stream = llm.send_streaming_convo_message(messages, None, None)
            .await
            .expect("Failed to initiate Gemini stream");

        let mut full_text = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.expect("Error while streaming Gemini chunk");
            match chunk {
                LlmChunk::Text(t) => {
                    print!("{}", t);
                    full_text.push_str(&t);
                }
                LlmChunk::Reasoning(r) => println!("\n[Thought]: {}", r),
                LlmChunk::Done => break,
            }
        }

        assert!(!full_text.is_empty(), "Gemini stream should return text");
    }
}

#[cfg(test)]
mod embedding_tests {
    use crate::llm::{LLM, Access};

    #[tokio::test]
    async fn test_gemini_embedding_success() {
        let llm = LLM::Gemini;
        let text = "The quick brown fox jumps over the lazy dog.";
        // Using the latest model
        let model = Some("gemini-embedding-001");

        let result = llm.embed(text, model, None, None).await;

        match result {
            Ok(vec) => {
                assert!(!vec.is_empty(), "Embedding vector should not be empty");
                println!("Successfully retrieved embedding of length: {}", vec.len());
            }
            Err(e) => {
                panic!("Gemini embedding failed: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_gemini_embedding_dimensions() {
        let llm = LLM::Gemini;
        let text = "Testing dimensionality reduction with Matryoshka.";
        let model = Some("gemini-embedding-001");
        let requested_dims = 256;

        let result = llm.embed(text, model, Some(requested_dims), None).await;

        match result {
            Ok(vec) => {
                assert_eq!(
                    vec.len(), 
                    requested_dims as usize, 
                    "The embedding vector length should match the requested dimensions"
                );
                println!("Successfully retrieved dimension-reduced embedding: {}", vec.len());
            }
            Err(e) => {
                panic!("Gemini dimension-reduced embedding failed: {}", e);
            }
        }
    }
}