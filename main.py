from openai import OpenAI
import streamlit as st
import os
from api_key import Api_key
from query import relevancy
os.environ['OPENAI_API_KEY'] = Api_key
client = OpenAI()


def generate_using_api(input):
    """This function inputs prompt and outputs image url"""
    try:
        response = client.images.generate(
            model="dall-e-3",
            style="natural",
            prompt=input,
            size=size,
            quality=option,
            n=1,
        )

        image_url = response.data[0].url
        return image_url
    except Exception as e:
        st.write('Oops! Write Appropriate prompt')


st.title('AI Image Generation Using Searcly')
choice = st.sidebar.selectbox('Select your choice', ['Home Page', 'Image To Text'])

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
elif choice == 'Image To Text':
    st.subheader('Visualize your Fashion Fantasies !')
    input_text = st.text_input('what you have in mind today?')
    size = st.selectbox('Select Image Size', ['1024x1024', '1024x1792', '1792x1024'])
    option = st.radio(label="Quality",options=('standard','hd'))
    if input_text is not None:
        if st.button('Generate Image'):
            st.info(input_text)
            value = relevancy(input_text)

            if value == 1:

                image_url = generate_using_api(input_text)
                st.image(image_url)

            else:
                st.write('please enter a prompt related to fashion !!')



