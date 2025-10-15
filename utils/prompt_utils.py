# utils/prompt_utils.py

"""
Prompt generation utilities for Human→LLM dataset pipeline.

Supports 3 extraction and generation strategies:
  Level 1 → Zero-shot (no persona, no examples)
  Level 2 → Genre-based Persona (adds stylistic guidance)
  Level 3 → Genre-based Persona + Example (adds few-shot examples)
"""

# -------------------------------------------------------------------------
# PERSONA TEMPLATES
# -------------------------------------------------------------------------
def get_persona_by_genre(genre: str) -> str:
    """Return an appropriate persona description for each genre."""
    personas = {
        "Academic": (
            "You are a university researcher skilled in analyzing, summarizing, "
            "and generating academic writing using precise and structured arguments."
        ),
        "Blogs": (
            "You are an experienced content creator who writes engaging, reflective, "
            "and relatable blog posts that connect with readers personally."
        ),
        "News": (
            "You are a professional journalist trained to write concise, neutral, "
            "and factual articles that adhere to high editorial standards."
        )
    }
    return personas.get(
        genre,
        "You are a clear and insightful writer who communicates ideas effectively."
    )


# -------------------------------------------------------------------------
# 🧩 EXAMPLE TEMPLATES
# -------------------------------------------------------------------------
def get_examples_by_genre(genre: str) -> str:
    """Return example Input→Output pairs for each genre."""
    examples = {
        "Academic": [
            (
                "AI ethics focuses on fairness and transparency.",
                "AI ethics emphasizes fairness, accountability, and transparency as core principles in responsible innovation."
            ),
            (
                "Renewable energy reduces carbon emissions.",
                "The global transition to renewable energy has significantly reduced carbon emissions, reshaping industrial policies worldwide."
            )
        ],
        "Blogs": [
            (
                "Morning routines improve focus.",
                "Starting the day with a structured morning routine helps boost clarity, motivation, and productivity."
            ),
            (
                "Traveling broadens perspectives.",
                "Exploring new places expands one’s understanding of cultures and encourages personal growth through new experiences."
            )
        ],
        "News": [
            (
                "Global leaders discuss emission reduction strategies.",
                "World leaders convened to negotiate new frameworks aimed at reducing global emissions and advancing renewable energy policy."
            ),
            (
                "Stock markets show mixed trends amid inflation fears.",
                "Global markets displayed mixed performance this week as investors weighed inflation data against central bank signals."
            )
        ]
    }

    genre_examples = examples.get(genre, [])
    if not genre_examples:
        return ""
    return "\n\n".join([
        f"Example:\nInput: {inp}\nOutput: {out}"
        for inp, out in genre_examples
    ])


# -------------------------------------------------------------------------
# EXTRACTION PROMPT GENERATION (for extract_utils)
# -------------------------------------------------------------------------
def generate_extraction_prompts(text: str, genre: str, subfield: str, year: int, level: int = 1):
    """
    Build (system_prompt, user_prompt) pair for keyword + summary extraction.

    Level meanings:
      1 → Zero-shot baseline
      2 → Genre-based Persona
      3 → Genre-based Persona + Example
    """
    if level == 1:
        # Level 1: Zero-shot baseline
        system_prompt = (
            "You are a careful text analyst.\n"
            "Your task is to read a human-written text and return:\n"
            "- 5 key domain-specific keywords or phrases (lowercase)\n"
            "- 1-sentence concise summary (<=30 words)\n"
            "- the approximate word count of the text"
        )

    elif level == 2:
        # Level 2: Persona guidance
        persona = get_persona_by_genre(genre)
        system_prompt = (
            f"{persona}\nYour task is to analyze the following human-written text and extract:\n"
            "- 5 domain-relevant keywords or phrases (lowercase)\n"
            "- a one-sentence summary (<=30 words)\n"
            "- the approximate word count of the text"
        )

    elif level == 3:
        # Level 3: Persona + Example
        persona = get_persona_by_genre(genre)
        examples = get_examples_by_genre(genre)
        system_prompt = (
            f"{persona}\nBelow are examples of how you have previously analyzed texts:\n\n"
            f"{examples}\n\n"
            "Now, analyze the new text in the same style and format."
        )

    else:
        raise ValueError(f"Unsupported level: {level}")

    # Shared user prompt template
    user_prompt = f"""
HUMAN METADATA
- genre: {genre}
- subfield: {subfield}
- year: {year}

HUMAN TEXT
\"\"\"{text.strip()}\"\"\"

Return strict JSON:
{{
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "summary": "one-sentence summary",
  "word_count": 123
}}
""".strip()

    return system_prompt.strip(), user_prompt.strip()


# -------------------------------------------------------------------------
# GENERATION PROMPT (for LLM text creation)
# -------------------------------------------------------------------------
def generate_prompt_from_summary(
    genre: str,
    subfield: str,
    year: int,
    keywords: list,
    summary: str,
    word_count: int,
    level: int = 1
):
    """
    Build a generation prompt to create new LLM text
    based on extracted keywords and summary.

    Uses the same Level logic:
      1 → Zero-shot (no persona)
      2 → Genre-based Persona
      3 → Persona + Example (few-shot)
    """
    keywords_line = ", ".join(keywords)
    base = f"""Here is a summary of a human-written {genre.lower()} text:

Keywords: {keywords_line}
Summary: {summary}
"""

    # Level-based writing setup
    if level == 1:
        preamble = "Write an original text inspired by these ideas."
    elif level == 2:
        preamble = get_persona_by_genre(genre)
    elif level == 3:
        persona = get_persona_by_genre(genre)
        examples = get_examples_by_genre(genre)
        preamble = f"{persona}\nBelow are writing examples from this genre:\n\n{examples}\n\nNow create a new text inspired by the given summary and keywords."
    else:
        raise ValueError(f"Unsupported level: {level}")

    # Genre-specific tone/style
    if genre == "Academic":
        style = (
            f"Write a formal academic-style abstract in the field of {subfield}. "
            f"Use precise terminology and structured reasoning (~{word_count} words). Year context: {year}."
        )
    elif genre == "Blogs":
        style = (
            f"Write a reflective blog post related to {subfield}. "
            f"Use a conversational tone and clear storytelling (~{word_count} words). Year context: {year}."
        )
    elif genre == "News":
        style = (
            f"Write a news article about {subfield}. Maintain objectivity, "
            f"factual balance, and professional tone (~{word_count} words). Year context: {year}."
        )
    else:
        raise ValueError(f"Unsupported genre: {genre}")

    return f"{base}\n\n{preamble}\n\n{style}".strip()
