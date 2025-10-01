import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUMAN_DIR = os.path.join(BASE_DIR, "Dataset", "Human")
LLM_DIR = os.path.join(BASE_DIR, "Dataset", "LLM")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# Style information (length is dynamic, only tone/style is fixed)
GENERATION_STYLE = {
    "News": "news article",
    "Blogs": "reflective blog post",
    "Academic": "academic essay",
    "Literary_works": "literary piece"
}
