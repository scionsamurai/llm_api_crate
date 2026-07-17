#[cfg(test)]
mod tests {
    use crate::llm::{LLM, Access};
    use crate::structs::general::{Message, MessageContent, MessagePart, ImageSource};

    #[tokio::test]
    async fn test_openai_multimodal_serialization() {
        let llm = LLM::OpenAI;
        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Array(vec![
                MessagePart {
                    r#type: "text".to_string(),
                    text: Some("What is in this image?".to_string()),
                    image_url: None,
                },
                MessagePart {
                    r#type: "image_url".to_string(),
                    text: None,
                    image_url: Some(ImageSource::Url { 
                        url: "https://example.com/image.png".to_string() 
                    }),
                },
            ]),
        }];

        let res = llm.send_convo_message(messages, Some("gpt-4o"), None).await;
        assert!(res.is_ok(), "OpenAI failed to process image URL: {:?}", res.err());
    }
}