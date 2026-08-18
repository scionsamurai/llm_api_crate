# Python Bindings for `llm_api_access`

This package provides a unified interface to query popular LLM providers (LlamaServer, Gemini, Anthropic, and OpenAI) directly from Python via PyO3 bindings.

---

## 1. Installation

### From PyPI (Recommended for users)
```bash
pip install llm_api_access
```

### For Local Development / Building from Source
If you are modifying the Rust core or building from source:

```bash
# Ensure maturin is installed in your virtual environment
pip install maturin

# Build and develop the package in-place with the python feature enabled
maturin develop --features python
```

---

## 2. Environment Variables & `.env` Support

Because the underlying Rust library uses `dotenv`, environment variables can be loaded from a `.env` file. When invoking from Python, it is recommended to load your `.env` file explicitly using `python-dotenv` at the very entry point of your script to ensure keys like `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `LLAMA_SERVER_URL` are available in the process environment before calling Rust code.

```bash
pip install python-dotenv
```

Example `.env` configuration:
```env
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LLAMA_SERVER_URL=http://192.168.0.91:8080
BASE64_DATA=your_base64_encoded_image_string_here
```

---

## 3. Usage Guide

Here is how to use the primary providers (LlamaServer and Gemini), along with multi-turn conversations and multimodal message handling in Python:

```python
import os
from dotenv import load_dotenv
import llm_api_access

# Load environment variables from .env file first
load_dotenv()

# Configure optional parameters (temperature, max_tokens, server URLs, etc.)
config = (
    llm_api_access.LlmConfig()
    .with_temperature(0.7)
    .with_max_tokens(150)
)

# -------------------------------------------------------------------------
# 1. LlamaServer (Local / Custom URL - Primary Target)
# -------------------------------------------------------------------------
llama_config = (
    llm_api_access.LlmConfig()
    .with_server_url(os.getenv("LLAMA_SERVER_URL", "http://192.168.0.91:8080"))
    .with_max_tokens(100)
)

try:
    llama_client = llm_api_access.LLMClient(llm_api_access.LLMProvider.LlamaServer)
    response = llama_client.send_message(
        prompt="Explain quantum computing in one sentence.",
        model="gemma-4-26b",
        config=llama_config
    )
    print("LlamaServer Response:", response)
except Exception as e:
    print("LlamaServer Error:", e)

# -------------------------------------------------------------------------
# 2. Gemini (Secondary Target)
# -------------------------------------------------------------------------
try:
    gemini_client = llm_api_access.LLMClient(llm_api_access.LLMProvider.Gemini)
    response = gemini_client.send_message(
        prompt="Write a haiku about coding.",
        model=None,
        config=config
    )
    print("Gemini Response:", response)
except Exception as e:
    print("Gemini Error:", e)

# -------------------------------------------------------------------------
# 3. Multi-turn Conversation & Multimodal (Images via Base64)
# -------------------------------------------------------------------------
base64_image = os.getenv("BASE64_DATA")

if base64_image and base64_image != "default_base64_value":
    try:
        # Construct a multimodal user message with an attached base64 image
        image_message = llm_api_access.Message(
            role="user",
            text="What is in this image? Answer briefly.",
            image_base64=base64_image,
            image_media_type="image/png"
        )
        
        # Follow-up message in conversation history
        follow_up = llm_api_access.Message(
            role="user",
            text="Summarize that in 3 words."
        )

        conversation = [
            image_message,
            llm_api_access.Message(role="assistant", text="I see a graphical pattern."),
            follow_up
        ]

        client = llm_api_access.LLMClient(llm_api_access.LLMProvider.Gemini)
        chat_response = client.send_chat(conversation, model=None, config=config)
        print("Multimodal Chat Response:", chat_response)
    except Exception as e:
        print("Multimodal Error:", e)
else:
    print("Skipping multimodal example: BASE64_DATA not found in environment.")
```

---

## 4. Troubleshooting

* **`ModuleNotFoundError: No module named 'llm_api_access'`**: Run `maturin develop --features python` inside your virtual environment to ensure the dynamic library is linked correctly.
* **API Key Errors**: Ensure `python-dotenv` is installed and `load_dotenv()` is called at the top of your python script so environment variables are populated before Rust execution begins.