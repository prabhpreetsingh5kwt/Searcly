import requests
import io
from PIL import Image
from openai import OpenAI
import streamlit as st
import os
from api_key import Api_key
from query import relevancy
from config import size_dalle2,size_dalle3
from st_keyup import st_keyup
import matplotlib.pyplot as plt
import torch
import  numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segmentation import show_mask, show_box, show_points, load_model, predict_background
from config import API_URL, headers
import time
import replicate
from config import replicate_ai_token