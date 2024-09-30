import os
import asyncio
from typing import Tuple, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import time

load_dotenv()

async def test_vendor_model(base_url: str, api_key: str, model_name: str, streaming: bool = True) -> Union[Tuple[str, dict], str]:
    """
    Test the chat model by streaming or non-streaming a prompt and returning the result.

    Args:
        base_url (str): The base URL of the LangChain server
        api_key (str): The API key for the LangChain server
        model_name (str): The name of the model to use
        streaming (bool): Whether to use streaming mode or not

    Returns:
        Union[Tuple[str, dict], str]: The final message and the last chunk metadata if streaming,
                                      or just the final message if not streaming
    """
    llm = ChatOpenAI(base_url=base_url, api_key=api_key, streaming=streaming, model=model_name, max_tokens=4000)
    
    if streaming:
        final_message = ""
        last_chunk = None
        async for chunk in llm.astream("What is the meaning of life?"):
            if chunk.response_metadata:
                last_chunk = chunk
            final_message += chunk.content

        print(last_chunk)
        return final_message, last_chunk
    else:
        final_message = await llm.ainvoke("What is the meaning of life?")
        return final_message.content

def get_credentials(vendor_name: str) -> Tuple[str, str]:
    """
    Get the base URL and API key for a given vendor.

    Args:
        vendor_name (str): The name of the vendor. Must be one of
            ["sambanova", "openai", "groq", "cerebras"]

    Returns:
        Tuple[str, str]: The base URL and API key for the vendor

    Raises:
        ValueError: If the vendor name is not recognized
    """
    vendors = {
        "sambanova": ("https://api.snova.ai/v1/", "SAMBANOVA_API_KEY"),
        "openai": ("https://api.openai.com/v1/", "OPENAI_API_KEY"),
        "groq": ("https://api.groq.com/openai/v1/", "GROQ_API_KEY"),
        "cerebras": ("https://api.cerebras.ai/v1/", "CEREBRAS_API_KEY")
    }

    if vendor_name not in vendors:
        raise ValueError(f"Vendor not supported. Please choose from {list(vendors.keys())}")

    base_url, env_key = vendors[vendor_name]
    api_key = os.environ.get(env_key)
    return base_url, api_key

def test_vendor(vendor_name: str, streaming: bool = True) -> None:
    """
    Test a vendor by calling the test_vendor_model function.
    
    Args:
        vendor_name (str): The name of the vendor to test. Must be one of
            ["sambanova", "openai", "groq", "cerebras"]
        streaming (bool): Whether to use streaming mode or not
    """
    print(f'Testing {vendor_name} with streaming={streaming}...')
    
    models = {
        "openai": "gpt-3.5-turbo",
        "sambanova": "llama3-70b",
        "groq": "llama-3.1-70b-versatile",
        "cerebras": "llama3.1-70b"
    }

    if vendor_name not in models:
        raise ValueError(f"Vendor not supported. Please choose from {list(models.keys())}")

    model_name = models[vendor_name]
    base_url, api_key = get_credentials(vendor_name)
    
    tic = time.time()
    result = asyncio.run(test_vendor_model(base_url, api_key, model_name, streaming))
    toc = time.time()
    print(f'Test complete for {vendor_name} in {toc-tic:.2f} seconds!')
    if not streaming:
        print(f'Result: {result}')
    print('\n')

# Test all vendors with both streaming and non-streaming modes
for vendor in ["sambanova", "groq", "cerebras", "openai"]:
    # test_vendor(vendor, streaming=True)
    test_vendor(vendor, streaming=False)
