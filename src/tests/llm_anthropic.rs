// src/tests/llm_anthropic.rs
#[cfg(test)]
mod tests {
    use crate::llm::{Access, LLM};
    use crate::structs::general::Message;

    #[tokio::test]
    async fn test_send_single_message_anthropic() {
        let llm: LLM = LLM::Anthropic;

        //let res = llm.send_single_message("Hi there, this is a test. Please generate a limrik.", None).await; // Old call
        let res = llm.send_single_message("Hi there, this is a test. Please generate a limrik.", None, None).await; // New call

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

        //let res = llm.send_convo_message(messages, None).await; // Old call
        let res = llm.send_convo_message(messages, None, None).await; // New call

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
}