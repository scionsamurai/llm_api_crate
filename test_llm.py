import os
from dotenv import load_dotenv
import llm_api_access

# Load environment variables from .env file explicitly
load_dotenv()

def test_llama_client():
    print("Testing LlamaServer Client (Primary local target)...")
    config = (
        llm_api_access.LlmConfig()
        .with_server_url(os.getenv("LLAMA_SERVER_URL", "http://192.168.0.91:8080"))
        .with_max_tokens(100)
    )

    client = llm_api_access.LLMClient(llm_api_access.LLMProvider.LlamaServer)
    try:
        # Note: adjust model name to match your local running LlamaServer model if needed
        response = client.send_message("Say 'Hello from Python via LlamaServer!'", model="gemma-4-26b", config=config)
        print(f"LlamaServer Response: {response}")
    except Exception as e:
        print(f"LlamaServer Error (is your local server running?): {e}")

def test_gemini_client():
    print("\nTesting Gemini Client (Secondary target)...")
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("Skipping Gemini test: GEMINI_API_KEY/GOOGLE_API_KEY not found in environment.")
        return

    config = (
        llm_api_access.LlmConfig()
        .with_temperature(0.5)
        .with_max_tokens(100)
    )

    client = llm_api_access.LLMClient(llm_api_access.LLMProvider.Gemini)
    try:
        response = client.send_message("Write a one-sentence greeting from Gemini via Python.", model=None, config=config)
        print(f"Gemini Response: {response}")
    except Exception as e:
        print(f"Gemini Error: {e}")

def test_multimodal_and_convo():
    print("\nTesting Multimodal & Conversation History via LlamaServer / Gemini...")
    
    base64_data = os.getenv("BASE64_DATA")
    if not base64_data or base64_data == "default_base64_value":
        print("Skipping Multimodal test: Valid BASE64_DATA environment variable not found.")
        return

    # Create a multimodal message using the exposed PyMessage class
    image_message = llm_api_access.Message(
        role="user",
        text="What do you see in this attached test image? Answer briefly.",
        image_base64=base64_data,
        image_media_type="image/png"
    )

    # Create follow-up conversation messages
    follow_up_message = llm_api_access.Message(
        role="user",
        text="Now, summarize your previous observation in exactly 3 words."
    )

    convo = [
        image_message,
        llm_api_access.Message(role="assistant", text="I see a test pattern or graphic in the image."),
        follow_up_message
    ]

    # Try testing multimodal against local LlamaServer first
    try:
        print("Sending multi-turn conversation with image payload to LlamaServer...")
        client = llm_api_access.LLMClient(llm_api_access.LLMProvider.LlamaServer)
        config = llm_api_access.LlmConfig().with_server_url(os.getenv("LLAMA_SERVER_URL", "http://192.168.0.91:8080"))
        response = client.send_chat(convo, model="gemma-4-26b", config=config)
        print(f"LlamaServer Chat Response: {response}")
    except Exception as e:
        print(f"LlamaServer Multimodal Chat skipped or failed: {e}")

    # Fallback / secondary multimodal test with Gemini if key exists
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        try:
            print("Sending multi-turn conversation with image payload to Gemini...")
            client = llm_api_access.LLMClient(llm_api_access.LLMProvider.Gemini)
            response = client.send_chat(convo, model=None)
            print(f"Gemini Chat Response: {response}")
        except Exception as e:
            print(f"Gemini Multimodal Chat Error: {e}")

if __name__ == "__main__":
    print("--- Starting Python LLM Binding Tests (Llama & Gemini Focus) ---")
    test_llama_client()
    test_gemini_client()
    test_multimodal_and_convo()
    print("--- Tests Complete ---")