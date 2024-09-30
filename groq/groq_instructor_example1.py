import os
import instructor

from groq import Groq
from pydantic import BaseModel

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods to support the response_model parameter
client = instructor.from_groq(client, mode=instructor.Mode.JSON)


# Now, we can use the response_model parameter using only a base model
# rather than having to use the OpenAISchema class
class UserExtract(BaseModel):
    name: str
    age: int
    email: str


user: UserExtract = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old. His preferred way of contact is jason@gmail.com"},
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"
assert user.name.lower() == "jason"
assert user.age == 25
assert user.email == "jason@gmail.com"
print(user.model_dump_json(indent=2))