
#[cfg(test)]
mod tests {
    use crate::llm::{LLM, Access};
    use std::env;
    use dotenv::dotenv;

    /// Tests that the Access trait correctly routes to OpenAI and returns a valid vector.
    /// Requires `OPEN_AI_KEY` to be set in your environment.
    #[tokio::test]
    async fn test_openai_embedding_integration() {
        dotenv().ok();
        if env::var("OPEN_AI_KEY").is_err() {
            println!("Skipping OpenAI embedding test: OPEN_AI_KEY not found.");
            return;
        }

        let llm = LLM::OpenAI;
        let text = "Testing embedding integration for OpenAI.";
        
        // We use None for model to use the default EMBEDDING_MODEL defined in openai.rs
        let result = llm.embed(text, None, None).await;

        match result {
            Ok(embedding) => {
                assert!(!embedding.is_empty(), "Embedding vector should not be empty");
                // text-embedding-3-small defaults to 1536 dimensions
                assert_eq!(embedding.len(), 1536, "Default OpenAI embedding size mismatch");
            }
            Err(e) => panic!("OpenAI embedding failed: {}", e),
        }
    }

    /// Tests that the Access trait correctly routes to the Llama Server.
    /// Requires a Llama server running at the URL specified in `LLAMA_SERVER_URL`.
    #[tokio::test]
    async fn test_llama_server_embedding_integration() {
        let llm = LLM::LlamaServer;
        let text = "Testing embedding integration for Llama Server.";
        
        let result = llm.embed(text, None, None).await;

        match result {
            Ok(embedding) => {
                assert!(!embedding.is_empty(), "Llama embedding vector should not be empty");
                println!("Llama embedding returned vector of length: {}", embedding.len());
            }
            Err(e) => {
                let err_msg = e.to_string();
                // If we get a connection error, it means the server isn't running.
                // We skip rather than fail so the test suite remains useful in CI.
                if err_msg.contains("connection refused") || err_msg.contains("Failed to send request") {
                    println!("Skipping Llama embedding test: No Llama server detected at default URL.");
                } else {
                    panic!("Llama embedding failed with an unexpected error: {}", e);
                }
            }
        }
    }

    /// Tests that the 'dimensions' parameter is correctly passed through to the provider.
    /// This is a key feature of newer OpenAI models.
    #[tokio::test]
    async fn test_embedding_dimensions_parameter() {
        dotenv().ok();
        if env::var("OPEN_AI_KEY").is_err() {
            println!("Skipping dimension test: OPEN_AI_KEY not found.");
            return;
        }

        let llm = LLM::OpenAI;
        let text = "Testing custom dimensions.";
        let requested_dims = 512;
        
        let result = llm.embed(text, None, Some(requested_dims)).await;

        match result {
            Ok(embedding) => {
                assert_eq!(
                    embedding.len(), 
                    requested_dims as usize, 
                    "Embedding dimension mismatch: expected {}, got {}", 
                    requested_dims, 
                    embedding.len()
                );
            }
            Err(e) => panic!("OpenAI dimension test failed: {}", e),
        }
    }
}
    
    
