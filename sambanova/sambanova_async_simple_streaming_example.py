import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv("../.env")
client = AsyncOpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)


async def main():
    stream = await client.chat.completions.create(
        model="llama3-70b",
        messages=[{"role": "user", "content": "Say this is a test"}],
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")


asyncio.run(main())