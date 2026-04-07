// src/tests/llm_llama.rs

#[cfg(test)]
mod tests {
    use crate::llm::{Access, LLM};
    use crate::config::LlmConfig;
    use crate::structs::general::Message;
    use crate::llama_server::call_llama_legacy;

    #[tokio::test]
    async fn test_llama_single_message() {
        let llm = LLM::LlamaServer;
        let prompt = "Explain the Fermi Paradox in one short sentence.";
        
        let res = llm.send_single_message(prompt, None, None).await;
        
        assert!(res.is_ok(), "Network call to Llama Server failed: {:?}", res.err());
        println!("Llama Single Message Response:\n{}", res.unwrap());
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
        
        let res = llm.send_convo_message(vec![system_message, user_message], None, None).await;
        
        assert!(res.is_ok(), "Network call to Llama Server failed: {:?}", res.err());
        println!("Llama Convo Response:\n{}", res.unwrap());
    }

    #[tokio::test]
    async fn test_llama_with_config() {
        let llm = LLM::LlamaServer;
        
        // Testing that the config struct serializes properly without crashing the server
        let config = LlmConfig::new()
            .with_temperature(0.5)
            .with_max_tokens(50) 
            .with_cache_prompt(true); 

        let res = llm.send_single_message("What is the capital of Japan?", None, Some(&config)).await;
        
        assert!(res.is_ok(), "Network call to Llama Server with config failed: {:?}", res.err());
        println!("Llama Config Response:\n{}", res.unwrap());
    }

    #[tokio::test]
    async fn test_llama_legacy_endpoint() {
        // We just want to ensure the /completion endpoint successfully receives and returns data
        let raw_prompt = "The sky is blue and the grass is ".to_string();
        let config = LlmConfig::new().with_max_tokens(10);
        
        let res = call_llama_legacy(raw_prompt, Some(&config)).await;
        
        assert!(res.is_ok(), "Legacy network call failed: {:?}", res.err());
        println!("Llama Legacy Response:\n{}", res.unwrap());
    }
}