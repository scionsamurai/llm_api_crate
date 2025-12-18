// src/tests/llm_openai.rs

#[cfg(test)]
mod tests {
    use crate::llm::{Access, LLM};
    use crate::structs::general::Message;

    #[tokio::test]
    async fn test_send_single_message_openai() {
        let llm: LLM = LLM::OpenAI;

        //let res = llm.send_single_message("Hello, can you tell me a joke?", None).await; // Old call
        let res = llm.send_single_message("Hello, can you tell me a joke?", None, None).await; // New call

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

        //let res = llm.send_convo_message(messages, None).await; // Old call
        let res = llm.send_convo_message(messages, None, None).await; // New call

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

}