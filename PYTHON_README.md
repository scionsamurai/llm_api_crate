# Python Bindings for `llm_api_access`

This crate provides a unified interface to query popular LLM providers (OpenAI, Anthropic, Gemini, and LlamaServer) directly from Python via PyO3 bindings.

---

## 1. Installation & Build

To build and install the package into your Python virtual environment using `maturin`:

```bash
# Ensure maturin is installed in your virtual environment
pip install maturin

# Build and develop the package in-place with the python feature enabled
maturin develop --features python
```

---

## 2. Environment Variables & `.env` Support

Because the underlying Rust library uses `dotenv`, environment variables can be loaded from a `.env` file. However, when invoking from Python, it is recommended to load your `.env` file explicitly using `python-dotenv` at the very entry point of your Python script to ensure keys like `OPEN_AI_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` are available in the process environment before calling Rust code.

```bash
pip install python-dotenv
```

Example `.env` configuration:
```env
OPEN_AI_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
LLAMA_SERVER_URL=http://192.168.0.91:8080
```

---

## 3. Usage Guide

Here is how to use all available providers in Python:

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
# 1. OpenAI
# -------------------------------------------------------------------------
try:
    openai_client = llm_api_access.LLMClient(llm_api_access.LLMProvider.OpenAI)
    response = openai_client.send_message(
        prompt="Explain quantum computing in one sentence.",
        model="gpt-4o",
        config=config
    )
    print("OpenAI Response:", response)
except Exception as e:
    print("OpenAI Error:", e)

# -------------------------------------------------------------------------
# 2. Anthropic
# -------------------------------------------------------------------------
try:
    anthropic_client = llm_api_access.LLMClient(llm_api_access.LLMProvider.Anthropic)
    response = anthropic_client.send_message(
        prompt="Why is the sky blue?",
        model=None, # Uses default model
        config=config
    )
    print("Anthropic Response:", response)
except Exception as e:
    print("Anthropic Error:", e)

# -------------------------------------------------------------------------
# 3. Gemini
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
# 4. LlamaServer (Local / Custom URL)
# -------------------------------------------------------------------------
llama_config = (
    llm_api_access.LlmConfig()
    .with_server_url("http://192.168.0.91:8080")
    .with_max_tokens(100)
)

try:
    llama_client = llm_api_access.LLMClient(llm_api_access.LLMProvider.LlamaServer)
    response = llama_client.send_message(
        prompt="Hello local model!",
        model=None,
        config=llama_config
    )
    print("LlamaServer Response:", response)
except Exception as e:
    print("LlamaServer Error:", e)
```

---

## 4. Troubleshooting

* **`ModuleNotFoundError: No module named 'llm_api_access'`**: Run `maturin develop --features python` inside your virtual environment to ensure the dynamic library is linked correctly.
* **API Key Errors**: Ensure `python-dotenv` is installed and `load_dotenv()` is called at the top of your python script so environment variables are populated before Rust execution begins.