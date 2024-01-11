import os
from langchain_community.llms import OpenAI as text_openai

from api_key import Api_key
os.environ['OPENAI_API_KEY'] = Api_key
#
# openai_key = Api_key
def category(input_text):
    prompt = f'if above text is related to fashion, return 1 else 0"""{input_text}"""'
    print(input_text)
    llm = text_openai(
        model_name="gpt-3.5-turbo-instruct",
        # openai_api_key=openai_key
    )
    output = llm(prompt)
    output = int(output)
    return output