import os
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_vendor_model(base_url, api_key, model_name):
    """
    Test the chat model by printing out the result of a stream prompt.

    :param base_url: The base url of the langchain server
    :param api_key: The api key to use for the langchain server
    :param model_name: The name of the model to use
    """    
    llm = ChatOpenAI(base_url=base_url,
                    api_key=api_key,
                    streaming=True,
                    model=model_name)
    
    final_message = ""
    async for chunk in llm.astream("what is the meaning of life?"):
        if chunk.response_metadata:
            last_chunk = chunk
        final_message += chunk.content

    print(last_chunk)
    return final_message, last_chunk

def get_credentials(vendor_name):
    """
    Test the cloud by getting the base url and api key for the vendor.

    :param vendor_name: The name of the vendor to test. Must be one of
        ["sambanova", "openai", "groq", "cerebras"]
    :return: A tuple of the base url and api key for the vendor
    :raises ValueError: If the vendor name is not recognized
    """
    if vendor_name == "sambanova":
        base_url = "https://fast-api.snova.ai/v1/"
        api_key = os.environ.get('SAMBANOVA_API_KEY')
    elif vendor_name == "openai":
        base_url = "https://api.openai.com/v1/"
        api_key = os.environ.get('OPENAI_API_KEY')
    elif vendor_name == "groq":
        base_url = "https://api.groq.com/openai/v1/"
        api_key = os.environ.get('GROQ_API_KEY')
    elif vendor_name == "cerebras":
        base_url = "https://api.cerebras.ai/v1/"
        api_key = os.environ.get('CEREBRAS_API_KEY')
    else:
        raise ValueError("Vendor not supported. Please choose 'sambanova' or 'openai'")

    return base_url, api_key


def test_vendor(vendor_name):

    """
    Test the cloud by calling the test_vendor_model function with the given vendor name.
    
    :param vendor_name: The name of the vendor to test. Must be one of
        ["sambanova", "openai", "groq", "cerebras"]
    """
    print(f'Testing {vendor_name}...')
    if vendor_name == "openai":
        model_name = "gpt-3.5-turbo"
        base_url, api_key = get_credentials(vendor_name)
    elif vendor_name == "sambanova":
        model_name = 'llama3-70b'
        base_url, api_key = get_credentials(vendor_name)
    elif vendor_name == "groq":
        model_name = 'llama-3.1-70b-versatile'
        base_url, api_key = get_credentials(vendor_name)
    elif vendor_name == "cerebras":
        model_name = 'llama3.1-70b'
        base_url, api_key = get_credentials(vendor_name)
    else:
        raise ValueError("Vendor not supported.")
    asyncio.run(test_vendor_model(base_url, api_key, model_name))
    print(f'Test complete for {vendor_name}!\n\n')


test_vendor('sambanova')
test_vendor('groq')
test_vendor('cerebras')
test_vendor('openai')
