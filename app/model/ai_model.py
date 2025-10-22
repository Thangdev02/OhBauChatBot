"""
Model configuration for OhBầu Chatbot.
Handles both text (chat) and image (baby generation) models.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    print("⚠️ Warning: API_KEY not set in environment.")

# Configure Gemini SDK
genai.configure(api_key=API_KEY)

# Define model handles
text_model = genai.GenerativeModel("gemini-2.0-flash")   # For chatbot
image_model = genai.GenerativeModel("imagen-3.0")        # For baby image generation
