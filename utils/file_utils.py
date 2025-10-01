def get_word_count(file_path: str) -> int:
    """Count the number of words in a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    words = text.split()
    return len(words)