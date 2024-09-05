import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import os

# Constants and Configuration
load_dotenv()
API_KEY = os.getenv("SAMBANOVA_API_KEY")
BASE_URL = "https://fast-api.snova.ai/v1"
MODEL = "llama3-405b"

# OpenAI client setup
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Weather tool definition
WEATHER_TOOL = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    },
}

# Tool prompt
TOOL_PROMPT = f"""
You have access to the following functions:

Use the function '{WEATHER_TOOL["name"]}' to '{WEATHER_TOOL["description"]}':
{json.dumps(WEATHER_TOOL)}

If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
"""

def get_current_weather(location: str) -> str:
    # This would be replaced by a weather API
    weather_data = {
        "San Francisco, CA": "62 degrees and cloudy",
        "Philadelphia, PA": "83 degrees and sunny"
    }
    return weather_data.get(location, "Weather is unknown")

def parse_tool_response(response: str):
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, response)

    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            return {"function": function_name, "arguments": args}
        except json.JSONDecodeError as error:
            print(f"Error parsing function arguments: {error}")
            return None
    return None

def get_streaming_response(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0,
        stream=True
    )
    return ''.join(chunk.choices[0].delta.content or '' for chunk in response)

def main():
    messages = [
        {"role": "system", "content": TOOL_PROMPT},
        {"role": "user", "content": "What is the weather in San Francisco, CA?"},
    ]

    # Get initial response
    message = get_streaming_response(messages)
    parsed_response = parse_tool_response(message)

    if parsed_response:
        available_functions = {"get_current_weather": get_current_weather}
        function_to_call = available_functions[parsed_response["function"]]
        weather = function_to_call(parsed_response["arguments"]["location"])
        messages.append({"role": "tool", "content": weather})
        print("Weather answer:", weather)

        # Get final response
        final_message = get_streaming_response(messages)
        print("Answer from the LLM:", final_message)
    else:
        print("No function call found in the response")

if __name__ == "__main__":
    main()