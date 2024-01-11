from openai import OpenAI
import streamlit as st
import os
import openai
from api_key import Api_key
# from langchain_community.llms import OpenAI as text_openai
# from langchain.llms import OpenAI
from langchain_openai import OpenAI

from query import category
os.environ['OPENAI_API_KEY'] = Api_key


client = OpenAI()


# image_url = response.data[0].url



def generate_using_api(input):
    response = client.images.generate(
        model="dall-e-3",
        prompt=input,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    return image_url


st.title('AI Image Generation Using Searcly')
choice = st.sidebar.selectbox('Select your choice', ['option1', 'option2'])

with st.expander('What is Searcly ?'):
    st.write('Searcy is a text-to-image generation tool which .')

if choice == 'option1':
    st.write('OPTION 1')
elif choice == 'option2':
    st.subheader('Visualize your Fashion Fantasies !')
    input_text = st.text_input('what you have in mind today?')
    if input_text is not None:
        if st.button('Generate Image'):
            st.info(input_text)
            # prompt = f'if above text is related to fashion, return 1 else 0"""{input_text}"""'
            output = category(input_text)


            if output == 1:


                image_url = generate_using_api(input_text)
                st.image(image_url)
            else : st.write('please enter a prompt related to fashion !!')


