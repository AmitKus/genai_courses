from groq import Groq
from dotenv import load_dotenv
import os

# Constants and Configuration
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        # "url": "https://upload.wikimedia.org/wikipedia/commons/8/8d/Euthalia_aconthea_caterpillar.jpg",
                        # "url": "https://media.datacenterdynamics.com/media/images/Sambanova_accenture.width-358.png"
                        "url": "https://sambanova.ai/hs-fs/hubfs/SambaChip_SN40L_Composite_R1_600x300.jpg?width=600&height=300&name=SambaChip_SN40L_Composite_R1_600x300.jpg"
                    },
                },
            ],
        }
    ],
    model="llava-v1.5-7b-4096-preview",
)

print(chat_completion.choices[0].message.content)