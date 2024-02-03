# Importing necessary libraries from imports file
from imports import *
# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = Api_key

# Create an OpenAI client
client = OpenAI()


# Function to generate images using the OpenAI API
def generate_using_api(input_prompt,n):
    """This function inputs a prompt and outputs image URLs"""
    try:
        # Call OpenAI API to generate images based on the input prompt and parameters selected in ui
        response = client.images.generate(
            model=model,
            style="natural",
            prompt=input_prompt,
            size=size,
            quality=option,
            n=num,
        )

        # Extract image URLs from the API response
        image_urls = [item.url for item in response.data]

        return image_urls

    except Exception as e:
        # Handle exceptions and display an error message
        st.write('Oops! Write Appropriate prompt', e)


# Sidebar Dropdown menu to navigate between different sections
choice = st.sidebar.selectbox('Select your choice', ['Home Page', 'Text To Image', 'Fun AI Generation!'])

# Handling different sections based on the user's choice
if choice == 'Home Page':
    # Home Page content
    st.title('AI Image Generation Using Searcly')
    with st.expander('What is Searcly ?'):
        st.write("""Searcly is a cutting-edge text-to-image tool specially designed for fashion aficionados.
         Our innovative platform utilizes the powerful AI text-to-image engine to bring your fashion dreams to life. 
         With Searcly, you have the power to conceptualize and visualize the perfect dress or apparel by simply typing
          in your desires.""")
    st.write('How It Works :')
    st.write("""Input Your Desires:
     Whether it's a dreamy wedding gown, a casual summer dress, or a trendy streetwear 
    ensemble,
     let your imagination flow through your fingertips. Type in your specific prompts about the dress or apparel you 
     have in mind.""")
    st.image(image_path)
    st.title('See the next generation visual product discovery!')

elif choice == 'Text To Image':
    # Section for Text to Image functionality
    st.title('AI Image Generation Using Searcly')
    with st.expander('What is Searcly ?'):
        st.write("""Searcly is a cutting-edge text-to-image tool specially designed for fashion aficionados.
         Our innovative platform utilizes the powerful AI text-to-image engine to bring your fashion dreams to life. 
         With Searcly, you have the power to conceptualize and visualize the perfect dress or apparel by simply typing in 
         your desires.""")

    st.subheader('Visualize your Fashion Fantasies !')

    # Text input for user to input their desires
    input_text = st.text_input('What do you have in mind today?', value=None, key='input_text')

    # Container to display generated images
    image_container = st.empty()

    # Slider to choose the number of images to generate
    num = st.slider(label='Number of Images:', min_value=1, max_value=10)

    # Set the model and size based on the number of images
    size_list = []
    if num > 1:
        model = 'dall-e-2'
        size_list = size_dalle2

    elif num == 1:
        model = 'dall-e-3'
        size_list = size_dalle3
    # Dropdown to select the size of an image
    size = st.selectbox('Select Image Size', size_list)
    # Dropdown to select quality of an image
    option = st.radio(label="Quality", options=('standard', 'hd'))

    # Generate images based on user input

    if st.button('Generate Image'):
        if input_text is not None:
            st.info(input_text)

            # Call function to generate images using the OpenAI API
            image_urls = generate_using_api(input_text,num)
            print('image_urls==', image_urls)

            # Display the generated images
            for image_url in image_urls:
                st.image(image_url)

        else:
            st.write('Please Enter a Prompt')

    # Real-time generation toggle
    on = st.toggle(label="RealTime Generation")

    if on:
        # Function to handle keypress events for real-time generation

        os.environ["REPLICATE_API_TOKEN"] = replicate_ai_token
        value = st_keyup("Enter Text", debounce=500, key="2")

        start = time.time()
        # using sdxl turbo inference api for realtime generation
        output = replicate.run(
            "dhanushreddy291/sdxl-turbo:53a8078c87ad900402a246bf5e724fa7538cf15c76b0a22753594af58850a0e3",
            input={
                "prompt": value,
                "num_outputs": 1,
                "negative_prompt": "3d, cgi, render, bad quality, normal quality, malformed, deformed face,deformed body, deformed hand, nfsw, anime, animated",
                "num_inference_steps": 3,

            }
        )
        print(output)
        st.image(image=output)
        end = time.time()
        print(end - start)
        # st.write(end - start)


elif choice == 'Fun AI Generation!':
    # Section for Fun AI Generation functionality
    st.header('Let Us Fuel Your Imagination🔥 ')
    text = st.text_input(label="Let's have fun!", key='text',)
    st.write('Choose Your Style :')
    # dropdown for image style
    style_choice = st.selectbox(label='Select Style', options=['Natural', 'Anime', 'Digital Art', 'Pixel Art', 'Manga', 'Neo Punk'])
    # Add extra spaces for generate button
    st.text('')
    st.text('')
    st.text('')
    _, col7, _ = st.columns([1, 1, 1])

    # If style choice is natural, prompt is sent to dall-e-3
    if style_choice == "Natural":
        prompt = f'{style_choice}: {text}, 8k, high quality'

        if col7.button(label="Generate"):
            output = replicate.run(
                "fofr/sdxl-emoji:dee76b5afde21b0f01ed7925f0665b7e879c50ee718c5f78a9d38e04d523cc5e",
                input={
                    "prompt": prompt,
                    "negative_prompt": "deformed face, bad quality, ugly, deformed hands, deformed body, anime, digital"
                }
            )
            print(output)
            st.image(image=output)
        #
        #     model = 'dall-e-3'
        #     size = '1024x1024'
        #     num = 1
        #     option = 'standard'
        #     image_urls = generate_using_api(text,n=num)
        #     for image_url in image_urls:
        #         st.image(image_url, width=600)


    else:
        if col7.button(label="Generate"):
            # The prompt includes style selected by the user and the prompt
            prompt = f'{style_choice}: {text}'

            # Stable diffusion 2 inference API (config.py)
            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.content


            image_bytes = query({
                "inputs": prompt,'negative_prompt':'blurry, deformed hands, Ugly, deformed face, deformed body, mutated body parts, disfigured, bad anatomy, deformed body features'
            })

            image = Image.open(io.BytesIO(image_bytes))
            st.image(image)


























