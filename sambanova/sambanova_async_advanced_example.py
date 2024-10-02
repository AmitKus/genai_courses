import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

async def process_slide(prompt, slide_number):
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b",
        )
        response = chat_completion.choices[0].message.content
        return f"Slide {slide_number}: {response}"
    except Exception as e:
        return f"Error processing slide {slide_number}: {str(e)}"

async def process_all_slides(prompted_slides_text):
    tasks = []
    for index, prompt in enumerate(prompted_slides_text, start=1):
        task = asyncio.create_task(process_slide(prompt, index))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

async def main():
    # Example list of prompts for slides
    prompted_slides_text = [
        "Summarize the key points of this slide about renewable energy.",
        "Explain the graph showing climate change trends over the past century.",
        "List the main benefits of sustainable agriculture mentioned in this slide.",
        # Add more prompts as needed
    ]

    print("Processing slides...")
    results = await process_all_slides(prompted_slides_text)
    
    for result in results:
        print(result)
        print("---")

if __name__ == "__main__":
    asyncio.run(main())