# utils/extract_utils.py

from typing import Dict, List, Any
from utils.api_utils import chat_json
from utils.prompt_utils import generate_extraction_prompts


def _normalize_keywords(value: Any) -> List[str]:
    """Ensure keywords is a list[str] of up to 5 lowercase items."""
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    else:
        items = []
    cleaned = [x.strip().lower() for x in items if isinstance(x, str) and x.strip()]
    return cleaned[:5]


def _normalize_summary(value: Any) -> str:
    """Return a trimmed string summary."""
    return value.strip() if isinstance(value, str) else ""


def _normalize_word_count(value: Any, fallback_text: str) -> int:
    """Coerce to int; fallback to naive word count."""
    try:
        wc = int(value)
        if wc > 0:
            return wc
    except Exception:
        pass
    return len(fallback_text.split())


def extract_keywords_summary_count(
    text: str,
    genre: str,
    subfield: str,
    year: int,
    level: int = 1,
    identifier: str = "",
) -> Dict[str, Any]:
    """
    Extract 5 keywords, a one-sentence summary, and approximate word count
    using an LLM with Level-specific prompts.

    level:
      1 → Zero-shot
      2 → Genre-based Persona
      3 → Genre-based Persona + Example
    """
    # 1. Generate prompt pair
    system_prompt, user_prompt = generate_extraction_prompts(
        text=text,
        genre=genre,
        subfield=subfield,
        year=year,
        level=level,
        identifier=identifier,
    )

    # 2. Call LLM and expect JSON response
    result = chat_json(system_prompt, user_prompt, max_tokens=700)

    # 3. Normalize safely
    keywords = _normalize_keywords(result.get("keywords"))
    summary = _normalize_summary(result.get("summary"))
    word_count = _normalize_word_count(result.get("word_count"), fallback_text=text)

    return {
        "keywords": keywords,
        "summary": summary,
        "word_count": word_count,
    }
