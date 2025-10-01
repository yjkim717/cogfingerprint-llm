import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUMAN_DIR = os.path.join(BASE_DIR, "Dataset", "Human")
LLM_DIR = os.path.join(BASE_DIR, "Dataset", "LLM")

# OpenAI API Key (replace later)
OPENAI_API_KEY = "YOUR_API_KEY_HERE"

# Only define styles, not lengths
GENERATION_STYLE = {
    "news": "news article",
    "blogs": "reflective blog post",
    "academic": "academic essay",
    "literary_works": "literary piece",
}
