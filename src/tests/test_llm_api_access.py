# tests/test_llm_api_access.py
import asyncio, sys, os
# sys.path.insert(0, os.path.abspath('./target/release'))
import llm_api_access
import pytest

async def call_llm_helper(llm_type, messages):
    """Helper function to call call_llm and handle potential exceptions."""
    try:
        response = await llm_api_access.call_llm(llm_type, messages)
        return response
    except Exception as e:
        pytest.fail(f"call_llm failed: {e}")

@pytest.mark.asyncio
async def test_call_llm_single_message_openai():
    messages = [{"role": "user", "content": "Hello, tell me a joke."}]
    response = await call_llm_helper("openai", messages)
    assert isinstance(response, str)
    assert len(response) > 0  # Ensure response is not empty

@pytest.mark.asyncio
async def test_call_llm_single_message_gemini():
    messages = [{"role": "user", "content": "Write the first line of a story."}]
    response = await call_llm_helper("gemini", messages)
    assert isinstance(response, str)
    assert len(response) > 0  # Ensure response is not empty

@pytest.mark.asyncio
async def test_call_llm_convo_message_openai():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    response = await call_llm_helper("openai", messages)
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_call_llm_convo_message_gemini():
    messages = [
        {"role": "user", "content": "Write the first line of a story."},
        {"role": "model", "content": "Once upon a time..."},
        {"role": "user", "content": "Continue the story in 1600s France."},
    ]
    response = await call_llm_helper("gemini", messages)
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_call_llm_invalid_llm_type():
    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(Exception) as excinfo:  # Expect a PyValueError
        await llm_api_access.call_llm("invalid_llm", messages)
    assert "Invalid LLM type" in str(excinfo.value)

@pytest.mark.asyncio
async def test_get_embedding():
    text = "This is a test sentence."
    embedding = await llm_api_access.get_embedding(text, None) #Optional dims
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding) #Check for floats.

    embedding_with_dims = await llm_api_access.get_embedding(text, 1536)
    assert isinstance(embedding_with_dims, list)
    assert all(isinstance(x, float) for x in embedding_with_dims)


async def main():
    messages = [{"role": "user", "content": "Hello, tell me a joke."}]
    response = await llm_api_access.call_llm("openai", messages)
    print(response)

    messages = [{"role": "user", "content": "Write the first line of a story."}]
    response = await llm_api_access.call_llm("gemini", messages)
    print(response)

if __name__ == "__main__":
    # pytest.main([__file__])
    asyncio.run(main())
