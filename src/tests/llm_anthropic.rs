// src/tests/llm_anthropic.rs
#[cfg(test)]
mod tests {
    use crate::llm::{Access, LLM};
    use crate::structs::general::{Message, MessageContent, LlmChunk};
    use futures::stream::StreamExt;

    #[tokio::test]
    async fn test_send_single_message_anthropic() {
        let llm: LLM = LLM::Anthropic;

        //let res = llm.send_single_message("Hi there, this is a test. Please generate a limrik.", None).await; // Old call
        let res = llm.send_single_message("Hi there, this is a test. Please generate a limrik.".into(), None, None).await; // New call

        match res {
            Ok(response) => {
                println!("Ok: {}", &response.text);
                assert!(!response.text.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Anthropic API failed");
            }
        }
    }

    #[tokio::test]
    async fn test_send_convo_message_anthropic() {
        let llm: LLM = LLM::Anthropic;
        let messages = vec![
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

        //let res = llm.send_convo_message(messages, None).await; // Old call
        let res = llm.send_convo_message(messages, None, None).await; // New call

        match res {
            Ok(response) => {
                println!("Ok: {}", &response.text);
                assert!(!response.text.is_empty(), "Response should not be empty");
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false, "Call to Anthropic API failed");
            }
        }
    }

    #[tokio::test]
    async fn test_anthropic_streaming() {
        let llm = LLM::Anthropic;
        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text("Tell me a joke about AI.".to_string()),
        }];

        let mut stream = llm.send_streaming_convo_message(messages, None, None)
            .await
            .expect("Failed to initiate Anthropic stream");

        let mut full_text = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.expect("Error while streaming Anthropic chunk");
            match chunk {
                LlmChunk::Text(t) => {
                    print!("{}", t);
                    full_text.push_str(&t);
                }
                LlmChunk::Reasoning(r) => println!("\n[Reasoning]: {}", r),
                LlmChunk::Done => break,
            }
        }

        assert!(!full_text.is_empty(), "Anthropic stream should return text");
    }
}