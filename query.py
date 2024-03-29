import json
from openai import OpenAI
import os
from api_key import Api_key
os.environ['OPENAI_API_KEY'] = Api_key
client = OpenAI()


def relevancy(input_text):
    """This function uses gpt 3.5-turbo model to check the relevancy of
    the given prompt to the preferred domain and returns a json output as 0 as not relevant and 1 as relevant"""
    try:
        response = client.chat.completions.create(
          model="gpt-3.5-turbo-1106",
          response_format={"type": "json_object"},
          messages=[
            {"role": "system", "content": "if given prompt is related to fashion or modelling, return 1, else 0 in output as JSON. Always keep the value strictly as 0 or 1 "},
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
