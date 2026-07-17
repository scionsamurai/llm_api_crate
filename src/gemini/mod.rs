// src/gemini/mod.rs
pub mod api;
pub mod types;
pub mod request;
pub mod response;

pub use api::*;
pub use types::*;
pub use response::*;
    

// https://ai.google.dev/tutorials/rest_quickstart

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::general::{ Message, Content, Part, MessageContent };
    use crate::config::LlmConfig;

    #[tokio::test]
    async fn test_call_gemini() {
        let message: Message = Message {
            role: "user".to_string(),
            content: MessageContent::Text("Hi there, this is a test. Please generate a limrick about the muppets.".to_string()),
        };

        let messages: Vec<Message> = vec![message];

        let res = call_gemini(messages, None, None).await; // Now returns Result<GeminiResponse, ...>

        match res {
            Ok(gemini_response) => {
                let res_str = gemini_response.candidates[0].content.parts[0].text.clone();
                // println!("res: {}", res_str.as_ref().map_or(false, |text| !text.is_empty()));
                assert!(res_str.as_ref().map_or(false, |text| !text.is_empty()));
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
                    text: Some("Write the first line of a story about a magic backpack.".to_string()),
                    inline_data: None,
                    thought: None,
                }],
            },
            Content {
                role: "model".to_string(),
                parts: vec![Part {
                    text: Some("In the bustling city of Meadow brook, lived a young girl named Sophie. She was a bright and curious soul with an imaginative mind.".to_string()),
                    inline_data: None,
                    thought: None,
                }],
            },
            Content {
                role: "user".to_string(),
                parts: vec![Part {
                    text: Some("Can you set it in a quiet village in 1600s France?".to_string()),
                    inline_data: None,
                    thought: None,
                }],
            },
        ];

        let res = conversation_gemini_call(messages, None, None).await; // Now returns Result<GeminiResponse, ...>
        match res {
            Ok(gemini_response) => { // Changed 'response' to 'gemini_response'
                assert!(!gemini_response.candidates.is_empty());
                assert!(gemini_response.candidates[0].content.parts[0].text.as_ref().map_or(false, |text| !text.is_empty()));
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn test_call_gemini_with_grounding() {
        let message: Message = Message {
            role: "user".to_string(),
            content: MessageContent::Text("Who is the current president of the United States?".to_string()),
        };

        let messages: Vec<Message> = vec![message];

        let config = LlmConfig::new().with_grounding_with_search(true);
        // println!("LlmConfig: {:?}", config);
        let res = call_gemini(messages, None, Some(&config)).await; // Now returns Result<GeminiResponse, ...>

        match res {
            Ok(gemini_response) => {
                // println!("Parsed GeminiResponse: {:?}", gemini_response); // Print the full struct for debugging
                // Access text for a basic assertion
                let res_str = gemini_response.candidates.get(0).map(|c| c.content.parts.get(0).map(|p| p.text.clone())).flatten().unwrap_or_default();
                // println!("Extracted text from Gemini API: {:?}", res_str);
                assert!(!res_str.as_ref().map_or(false, |text| !text.is_empty()));

                // Check for groundingMetadata directly from the parsed response
                if let Some(candidate) = gemini_response.candidates.get(0) {
                    if candidate.grounding_metadata.is_some() {
                        // println!("Grounding metadata found!");
                        assert!(true); // Grounding metadata exists
                    } else {
                        // println!("No grounding metadata found in the response.");
                        assert!(false, "Grounding metadata should be present when grounding is enabled.");
                    }
                } else {
                    // println!("No candidates found in the response.");
                    assert!(false, "Expected at least one candidate in the response.");
                }
            }
            Err(err) => {
                println!("Error: {}", err);
                assert!(false);
            }
        }
    }
}
