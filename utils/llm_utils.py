import openai
from config import OPENAI_API_KEY

# Set API key once (imported from config)
openai.api_key = OPENAI_API_KEY

def generate_text_from_prompt(prompt: str) -> str:
    """
    Call the OpenAI API to generate text from a given prompt.
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # can be swapped to gpt-4, gpt-4o, etc.
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7
    )
    return response.choices[0].message["content"].strip()
