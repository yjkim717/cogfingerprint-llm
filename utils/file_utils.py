import os

def list_human_files(human_dir: str):
    """
    Traverse the Human dataset directory and return all .txt file paths.
    """
    file_paths = []
    for root, _, files in os.walk(human_dir):
        for file in files:
            if file.endswith(".txt"):
                file_paths.append(os.path.join(root, file))
    return file_paths


def parse_filename(filename: str):
    """
    Parse a filename like 'News_2021_01.txt' into metadata.
    Returns a dict: {genre, year, index}
    """
    base = os.path.splitext(filename)[0]  # Remove .txt
    parts = base.split("_")

    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {filename}")

    genre = parts[0]
    year = parts[1]
    index = parts[2]

    return {
        "genre": genre,
        "year": year,
        "index": index
    }


def get_word_count(file_path: str) -> int:
    """
    Count the number of words in a text file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return len(text.split())

def get_llm_save_path(human_path: str, human_dir: str, llm_dir: str) -> str:
    """
    Create the save path for the LLM-generated file.
    Example:
        Human: Dataset/Human/News/2021/News_2021_01.txt
        LLM:   Dataset/LLM/News/2021/News_llm_2021_01.txt
    """
    rel_path = os.path.relpath(human_path, human_dir)  # News/2021/News_2021_01.txt
    parts = rel_path.split(os.sep)

    # filename transform
    filename = parts[-1]
    base, ext = os.path.splitext(filename)
    genre, year, index = base.split("_")  # expects format Genre_Year_Index

    llm_filename = f"{genre}_llm_{year}_{index}{ext}"

    # replace Human with LLM
    llm_path = os.path.join(llm_dir, *parts[:-1], llm_filename)
    os.makedirs(os.path.dirname(llm_path), exist_ok=True)

    return llm_path
