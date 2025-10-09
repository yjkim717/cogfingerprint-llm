# utils/extract_utils.py
from utils.api_utils import chat_json

EXTRACT_SYSTEM = """You are a careful text analyst.
Your task is to read a human-written text and return:
- 5 key domain-specific keywords or phrases (lowercase)
- 1-sentence concise summary (<=30 words)
- the approximate word count of the text
"""

EXTRACT_USER_TMPL = """
HUMAN METADATA
- genre: {genre}
- subfield: {subfield}
- year: {year}

HUMAN TEXT
\"\"\"{text}\"\"\"

Return strict JSON:
{{
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "summary": "one-sentence summary",
  "word_count": 123
}}
"""

def extract_keywords_summary_count(text: str, genre: str, subfield: str, year: int):
    user = EXTRACT_USER_TMPL.format(genre=genre, subfield=subfield, year=year, text=text.strip())
    result = chat_json(EXTRACT_SYSTEM, user, max_tokens=700)
    return {
        "keywords": [k.lower() for k in result.get("keywords", [])][:5],
        "summary": result.get("summary", "").strip(),
        "word_count": result.get("word_count", len(text.split())),
    }
