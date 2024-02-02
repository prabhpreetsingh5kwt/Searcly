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
choice = st.sidebar.selectbox('Select your choice', ['Home Page', 'Text To Image', 'Background Changer', 'Fun AI Generation!'])

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
    if input_text is not None:
        if st.button('Generate Image'):
            st.info(input_text)
            # relevancy function checks if the prompt is related to Searcly specific domain (e-commerce and fashion)
            value = relevancy(input_text)
            print('value==', value)

            if value == 1:
                # Call function to generate images using the OpenAI API
                image_urls = generate_using_api(input_text,num)
                print('image_urls==', image_urls)

                # Display the generated images
                for image_url in image_urls:
                    st.image(image_url)
            else:
                st.write('Please enter a prompt related to the fashion industry!!')

    # Real-time generation toggle
    on = st.toggle(label="RealTime Generation")

    if on:
        # Function to handle keypress events for real-time generation

        os.environ["REPLICATE_API_TOKEN"] = "r8_XIqHDe4ZuG1TYMs4bNpnG7UWvJaU5Kk2mbVpG"
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





elif choice == 'Background Changer':
    # Section for Background Changer functionality
    st.title('Background Changer')
    st.subheader('Upload an Image to get started!')

    # File uploader for image input
    file = st.file_uploader('label', label_visibility='hidden', type=['png', 'jpeg', 'jpg'])
    if file is not None:

        # Create a directory to store uploaded images
        new_dir = 'temp/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        newfile_path = f"{new_dir}/{file.name}"
        st.image(file, caption=file.name, width=250)

        # Save the uploaded image to a temporary directory
        with open(newfile_path, "wb") as f:
            f.write(file.getbuffer())

        # Button to delete the uploaded image
        if st.button('Delete Image'):
            try:
                for file in os.listdir(new_dir):
                    os.remove(newfile_path)
            except Exception as e:
                print(e)

        # Load YOLOv5 model for object detection
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model.conf = 0.7

        # Run object detection on the uploaded image
        results = model(newfile_path)

        # Render the image and resize it
        pil_image = Image.fromarray(np.squeeze(results.render()).astype(np.uint8))
        resized_image = pil_image.resize((800, 800))  # Adjust the size as needed

        # Display the detected objects and labels
        df = results.pandas().xyxy[0]
        print(df)

        # Extract label names to a list
        label_names = df['name'].tolist()
        print(label_names)

        # Dropdown to select a label
        selected_label = st.selectbox("Select Label", label_names)
        st.image(resized_image, use_column_width=True)

        # Get the index of the selected label
        selected_index = label_names.index(selected_label)
        selected_boxes = results.xyxy[0].cpu().numpy()[df['name'] == selected_label][:, :4]
        print('selected_boxes', selected_boxes)

        # Load a segmentation model
        sam = load_model()

        if selected_boxes is not None:
            # Dropdown to choose a background
            background_choice = st.selectbox('Choose a Background', ['woodenbg', 'Kitchen', 'Living Room', 'Beach'])

            # Button to change the background
            if st.button(label='Change Background'):

                # Create a SamPredictor object for segmentation
                predictor = SamPredictor(sam)
                image = cv2.cvtColor(cv2.imread(newfile_path), cv2.COLOR_BGR2RGB)
                predictor.set_image(image)
                input_box = np.array(selected_boxes)

                # Predict background masks
                masks = predict_background(image, selected_boxes, predictor)

                # Display the original image with masks and bounding box
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(image)
                show_mask(masks[0], ax)
                show_box(input_box, ax)
                ax.axis('off')

                # Load the chosen background image
                bg_dir = "background"
                bg = f"{bg_dir}/{background_choice}.jpg"
                background_image_bgr = cv2.imread(bg)

                # Process the segmentation mask
                segmentation_mask = masks[0]
                binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

                # Convert background image to RGB
                background_image_rgb = cv2.cvtColor(background_image_bgr, cv2.COLOR_BGR2RGB)

                # Sliders to control the position of the new image
                x_position = st.slider("X Position", 0, image.shape[1] - 1, 0)
                y_position = st.slider("Y Position", 0, image.shape[0] - 1, 0)

                # Resize the background image to match the size of the original image
                background_image_rgb = cv2.resize(background_image_rgb, (image.shape[1], image.shape[0]))

                # Compose the new image by blending the background and the object using the segmentation mask
                new_image = background_image_rgb * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]
                new_image = np.roll(new_image, (y_position, x_position), axis=(0, 1))

                # Convert the NumPy array to a PIL Image
                pil_image = Image.fromarray(new_image.astype(np.uint8))

                # Display the final image in Streamlit
                st.image(pil_image, use_column_width=True)


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

        if col7.button(label="Generate"):

            model = 'dall-e-3'
            size = '1024x1024'
            num = 1
            option = 'standard'
            image_urls = generate_using_api(text,n=num)
            for image_url in image_urls:
                st.image(image_url, width=600)


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


























