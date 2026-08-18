import os
import llm_api_access

def test_openai_client():
    print("Testing OpenAI Client...")
    # Ensure OPEN_AI_KEY is set in your environment or .env
    if not os.getenv("OPEN_AI_KEY"):
        print("Skipping OpenAI test: OPEN_AI_KEY not found in environment.")
        return

    config = llm_api_access.LlmConfig()
    config = config.with_temperature(0.7)
    config = config.with_max_tokens(100)

    client = llm_api_access.LLMClient(llm_api_access.LLMProvider.OpenAI)
    try:
        response = client.send_message("Say 'Hello from Python via Rust!'", model=None, config=config)
        print(f"OpenAI Response: {response}")
    except Exception as e:
        print(f"OpenAI Error: {e}")

def test_llama_client():
    print("\nTesting LlamaServer Client...")
    # Optional: configure custom server URL if needed
    config = llm_api_access.LlmConfig()
    config = config.with_server_url("http://192.168.0.91:8080")

    client = llm_api_access.LLMClient(llm_api_access.LLMProvider.LlamaServer)
    try:
        response = client.send_message("Hello from Python!", model=None, config=config)
        print(f"LlamaServer Response: {response}")
    except Exception as e:
        print(f"LlamaServer Error (is your local server running?): {e}")

if __name__ == "__main__":
    print("--- Starting Python LLM Binding Tests ---")
    test_openai_client()
    test_llama_client()
    print("--- Tests Complete ---")