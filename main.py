from openai import OpenAI
import streamlit as st
import os
from api_key import Api_key
from query import relevancy
from config import size_dalle2,size_dalle3
os.environ['OPENAI_API_KEY'] = Api_key
client = OpenAI()


def generate_using_api(input_prompt):
    """This function inputs prompt and outputs image url"""
    try:
        response = client.images.generate(
            model=model,
            style="natural",
            prompt=input_prompt,
            size=size,
            quality=option,
            n=num,
        )

        image_urls = [item.url for item in response.data]

        return image_urls

    except Exception as e:
        st.write('Oops! Write Appropriate prompt')


st.title('AI Image Generation Using Searcly')
choice = st.sidebar.selectbox('Select your choice', ['Home Page', 'Text To Image'])

with st.expander('What is Searcly ?'):
    st.write("""Searcly is a cutting-edge text-to-image tool specially designed for fashion aficionados.
     Our innovative platform utilizes the powerful AI text to image engine to bring your fashion dreams to life. 
     With Searcly, you have the power to conceptualize and visualize the perfect dress or apparel by simply typing in 
     your desires.""")

if choice == 'Home Page':
    st.write('How It Works :')
    st.write("""Input Your Desires:
     Whether it's a dreamy wedding gown, a casual summer dress, or a trendy streetwear 
    ensemble,
     let your imagination flow through your fingertips. Type in your specific prompts about the dress or apparel you 
     have in mind.""")
elif choice == 'Text To Image':
    st.subheader('Visualize your Fashion Fantasies !')
    input_text = st.text_input('what you have in mind today?')

    num = st.slider(label='Number of Images:', min_value=1, max_value=10)
    size_list = []
    if num > 1:
        model = 'dall-e-2'
        size_list = size_dalle2

    if num == 1:
        model = 'dall-e-3'
        size_list = size_dalle3

    size = st.selectbox('Select Image Size', size_list)
    option = st.radio(label="Quality", options=('standard', 'hd'))
    if input_text is not None:
        if st.button('Generate Image'):
            st.info(input_text)
            value = relevancy(input_text)
            print('value==', value)

            if value == 1:

                image_urls = generate_using_api(input_text)
                print('image_urls==', image_urls)
                n_cols = 3

                # n_rows = (len(image_urls) + n_cols - 1) // n_cols
                # rows = [st.columns(n_cols) for _ in range(n_rows)]
                # cols = [column for row in rows for column in row]
                # for col,image_url in zip(cols, image_urls):
                #     col.image(image_url)
                for image_url in image_urls:
                    st.image(image_url)

            else:
                st.write('please enter a prompt related to fashion !!')



