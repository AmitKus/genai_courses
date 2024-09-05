import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("SAMBANOVA_API_KEY")
MODEL = "llama3-70b"

def get_current_weather(location: str) -> str:
    # This would be replaced by a weather API
    if location == "San Francisco, CA":
        return "62 degrees and cloudy"
    elif location == "Philadelphia, PA":
        return "83 degrees and sunny"
    return "Weather is unknown"

weather_tool = StructuredTool(
    name="get_current_weather",
    description="Get the current weather in a given location",
    func=get_current_weather,
    args_schema={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        }
    }
)

llm = ChatOpenAI(
    model=MODEL,
    openai_api_key=api_key,
    base_url="https://fast-api.snova.ai/v1",
    temperature=0
)

agent = initialize_agent(
    [weather_tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

user_input = "What is the weather in San Francisco, CA?"
response = agent.run(user_input)

print("Answer from the LLM: ", response)