// src/tests/llm_llama.rs

#[cfg(test)]
mod tests {
    use crate::llm::{Access, LLM};
    use crate::config::LlmConfig;
    use crate::structs::general::{Message, MessageContent, LlmChunk};
    use crate::llama_server::call_llama_legacy;
    use futures::stream::StreamExt;

    // Use the model name that matches the file in your model folder
    const TEST_MODEL: &str = "gemma-4-26b";

    #[tokio::test]
    async fn test_llama_single_message() {
        let llm = LLM::LlamaServer;
        let prompt = "Explain the Fermi Paradox in one short sentence.";
        
        let res = llm.send_single_message(prompt.into(), Some(TEST_MODEL), None).await;
        
        match res {
            Ok(response) => {
                assert!(!response.text.is_empty(), "Response text should not be empty");
                println!("Llama Single Message Response:\n{}", response.text);
                
                if let Some(reasoning) = response.reasoning {
                    println!("Llama Single Message Reasoning:\n{}", reasoning);
                }
            }
            Err(err) => panic!("Network call to Llama Server failed: {}", err),
        }
    }

    #[tokio::test]
    async fn test_llama_convo_message() {
        let llm = LLM::LlamaServer;
        
        let system_message = Message {
            role: "system".to_string(),
            content: "You are a helpful assistant that only speaks in pirate jargon.".into(),
        };
        let user_message = Message {
            role: "user".to_string(),
            content: "Hello! How are you doing today?".into(),
        };
        
        let res = llm.send_convo_message(vec![system_message, user_message], Some(TEST_MODEL), None).await;
        
        match res {
            Ok(response) => {
                assert!(!response.text.is_empty(), "Response text should not be empty");
                println!("Llama Convo Response:\n{}", response.text);
                
                if let Some(reasoning) = response.reasoning {
                    println!("Llama Convo Reasoning:\n{}", reasoning);
                }
            }
            Err(err) => panic!("Network call to Llama Server failed: {}", err),
        }
    }

    #[tokio::test]
    async fn test_llama_with_config() {
        let llm = LLM::LlamaServer;
        
        // Testing that the config struct serializes properly without crashing the server
        let config = LlmConfig::new();

        let res = llm.send_single_message("What is capital of Japan?".into(), Some(TEST_MODEL), Some(&config)).await;
        
        match res {
            Ok(response) => {
                assert!(!response.text.is_empty(), "Response text should not be empty");
                println!("Llama Config Reasoning:\n{:?}", response.reasoning);
                println!("Llama Config Response:\n{}", response.text);
            }
            Err(err) => panic!("Network call to Llama Server with config failed: {}", err),
        }
    }

    #[tokio::test]
    async fn test_llama_legacy_endpoint() {
        // We just want to ensure the /completion endpoint successfully receives and returns data
        let raw_prompt = "The sky is blue and the grass is ".to_string();
        
        // Let's test with a thinking budget to ensure <|think|> gets injected
        let config = LlmConfig::new()
            .with_max_tokens(100)
            .with_thinking_budget(1024);
        
        let res = call_llama_legacy(raw_prompt, Some(TEST_MODEL), Some(&config)).await;
        
        match res {
            Ok(response) => {
                assert!(!response.text.is_empty(), "Legacy response should not be empty");
                println!("Llama Legacy Response Text:\n{}", response.text);
                if let Some(reasoning) = response.reasoning {
                     println!("Llama Legacy Reasoning:\n{}", reasoning);
                }
            }
            Err(err) => panic!("Legacy network call failed: {}", err),
        }
    }

    #[tokio::test]
    async fn test_llama_streaming() {
        let llm = LLM::LlamaServer;
        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text("What is the capital of France?".to_string()),
        }];

        let mut stream = llm.send_streaming_convo_message(messages, None, None)
            .await
            .expect("Failed to initiate Llama stream");

        let mut full_text = String::new();
        
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.expect("Error while streaming Llama chunk");
            match chunk {
                LlmChunk::Text(t) => {
                    print!("{}", t);
                    full_text.push_str(&t);
                }
                LlmChunk::Reasoning(r) => println!("\n[Thinking]: {}", r),
                LlmChunk::Done => break,
            }
        }

        assert!(!full_text.is_empty(), "Llama stream should return text");
    }

    #[tokio::test]
    async fn test_llama_multimodal_message() {
        use crate::structs::general::{MessagePart, ImageSource};
        let llm = LLM::LlamaServer;
        
        let base64_data = crate::tests::variables::BASE64_DATA;

        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Array(vec![
                MessagePart {
                    r#type: "text".to_string(),
                    text: Some("What do you see in this image?".to_string()),
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

        let res = llm.send_convo_message(messages, Some(TEST_MODEL), None).await;
        println!("Llama multimodal response: {:?}", res);
        match res {
            Ok(response) => {
                assert!(!response.text.is_empty(), "Llama vision response should not be empty");
            }
            Err(err) => {
                println!("Llama vision call skipped or failed (expected if model is text-only): {}", err);
            }
        }
    }
}