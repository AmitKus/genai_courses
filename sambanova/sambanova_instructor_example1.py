import os
import instructor

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url="https://fast-api.snova.ai/v1",
)
print(os.getenv("SAMBANOVA_API_KEY"))

# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods to support the response_model parameter
client = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)


# Now, we can use the response_model parameter using only a base model
# rather than having to use the OpenAISchema class
class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.chat.completions.create(
    model="llama3-70b",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ]
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"
assert user.name.lower() == "jason"
assert user.age == 25

print(user.model_dump_json(indent=2))