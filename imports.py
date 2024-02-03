import requests
import io
from PIL import Image
from openai import OpenAI
import streamlit as st
import os
from query import relevancy
from config import size_dalle2,size_dalle3
from st_keyup import st_keyup
import matplotlib.pyplot as plt
import torch
import  numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from config import API_URL, headers
import time
import replicate
from config import replicate_ai_token
from config import image_path