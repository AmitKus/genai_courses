import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv("../.env")

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)


async def main() -> None:
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="llama3-70b",
    )
    # Print the response content
    print(chat_completion.choices[0].message.content)


asyncio.run(main())