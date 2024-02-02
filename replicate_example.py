
import streamlit as st
import time
from st_keyup import st_keyup
import replicate
import os

os.environ["REPLICATE_API_TOKEN"] = "r8_XIqHDe4ZuG1TYMs4bNpnG7UWvJaU5Kk2mbVpG"
value = st_keyup("Enter Text", debounce=500, key="2")

start = time.time()
output = replicate.run(
    "dhanushreddy291/sdxl-turbo:53a8078c87ad900402a246bf5e724fa7538cf15c76b0a22753594af58850a0e3",
    input={
        "prompt": value,
        "num_outputs": 1,
        "negative_prompt": "3d, cgi, render, bad quality, normal quality, malformed, deformed, asian, anime",
        "num_inference_steps": 4,

    }
)
print(output)
st.image(image=output)
end = time.time()
print(end - start)
st.write(end - start)

