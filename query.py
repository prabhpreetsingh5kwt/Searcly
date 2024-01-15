import json
from openai import OpenAI
import os
from api_key import Api_key
os.environ['OPENAI_API_KEY'] = Api_key
client = OpenAI()


def relevancy(input_text):
    try:
        response = client.chat.completions.create(
          model="gpt-3.5-turbo-1106",
          response_format={"type": "json_object"},
          messages=[
            {"role": "system", "content": "if given prompt is related to fashion, return 1, else 0 in output as JSON."},
            {"role": "user", "content": input_text}
          ],
          temperature=0,
          max_tokens=250,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          seed=1001
        )
        message = response.choices[0].message.content
        message = json.loads(message)
        value = message['fashion_related']
        return value
    except Exception as e:
        print('relevancy error')
