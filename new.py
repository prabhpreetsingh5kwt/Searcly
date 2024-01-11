from openai import OpenAI
import os
from api_key import Api_key
os.environ['OPENAI_API_KEY'] = Api_key

client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="a white siamese cat",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url