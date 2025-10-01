import openai
from config import OPENAI_API_KEY


openai.api_key = OPENAI_API_KEY

def extract_topic_with_llm(text: str) -> str:
    """
    Use LLM to extract the main topic/title from an article text.
    """
    prompt = f"""
    You are given the full text of an article. 
    Extract the main topic or title of this article in one short phrase.

    Text:
    {text[:1500]} 

    Answer with only the topic.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0
    )
    
    return response.choices[0].message["content"].strip()
