from config import GENERATION_STYLE

def generate_prompt(metadata: dict, topic: str, word_count: int) -> str:
    """
    Generate a prompt for the LLM based on metadata, topic, and target length.

    Args:
        metadata (dict): Dictionary containing genre, year, index.
        topic (str): The extracted topic of the text.
        word_count (int): The approximate length of the original human text.

    Returns:
        str: A formatted prompt for the LLM.
    """
    genre = metadata["genre"]
    year = metadata["year"]

    # Select the style from config
    style = GENERATION_STYLE.get(genre, "text")

    # Build the prompt
    prompt = (
        f"Write a {style} on the topic '{topic}' as if it were written in {year}. "
        f"The length should be around {word_count} words, "
        f"and the style should resemble a human-written {style}."
    )

    return prompt
