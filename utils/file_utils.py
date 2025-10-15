import os
import re
from typing import Dict
from config import get_llm_path

# (DeepSeek / Gemma / Llama)
MODEL_TAGS = {
    "DEEPSEEK": "DS",
    "GEMMA_4B": "G4B",
    "GEMMA_12B": "G12B",
    "LLAMA_MAVRICK": "LMK",
}


# -------------------------------------------------------------------------
# BASIC FILE OPERATIONS
# -------------------------------------------------------------------------
def read_text(path: str) -> str:
    """Read UTF-8 text from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str):
    """Write text to file, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# -------------------------------------------------------------------------
# METADATA PARSING
# -------------------------------------------------------------------------
def parse_metadata_from_path(human_path: str) -> Dict[str, str]:
    """
    Parse genre, subfield, year, and index from a Human dataset path.

    Supports multiple folder/file naming styles:
      - Academic: Datasets/Human/Academic/bio/2020/Bio_DS_2020_01.txt
      - Blogs:    Datasets/Human/Blogs/Lifestyle/2022/LIFESTYLE_2022_10_02.txt
      - News:     Datasets/Human/News/politics/2023/NEWS_2023_01.txt
    """
    parts = os.path.normpath(human_path).split(os.sep)

    try:
        idx = parts.index("Human")
        genre = parts[idx + 1]           # e.g. Academic / Blogs / News
        subfield = parts[idx + 2]        # e.g. bio / Lifestyle / politics
        year = parts[idx + 3]            # e.g. 2020
        filename = parts[idx + 4]        # e.g. Bio_DS_2020_01.txt
    except (ValueError, IndexError):
        raise ValueError(f"Unexpected path structure: {human_path}")

    index = "01"  # Default

    # Case 1Academic: Bio_DS_2020_01.txt
    m_acad = re.search(r"_(\d{4})_(\d+)\.txt$", filename)
    if m_acad:
        year = m_acad.group(1)
        index = m_acad.group(2)
        return {"genre": genre, "subfield": subfield.lower(), "year": str(year), "index": str(index)}

    # Case 2️ Blogs: LIFESTYLE_2022_10_02.txt
    m_blog = re.search(r"([A-Za-z]+)_(\d{4})_(\d{2})_(\d{2})\.txt$", filename)
    if m_blog:
        subfield_from_name = m_blog.group(1)
        year = m_blog.group(2)
        month = m_blog.group(3)
        index = m_blog.group(4)
        return {
            "genre": genre,
            "subfield": subfield_from_name.lower(),
            "year": str(year),
            "index": f"{month}_{index}",
        }

    # Case 3️ News: NEWS_2023_01.txt
    m_news = re.search(r"([A-Za-z]+)_(\d{4})_(\d+)\.txt$", filename)
    if m_news:
        subfield_from_name = m_news.group(1)
        year = m_news.group(2)
        index = m_news.group(3)
        return {"genre": genre, "subfield": subfield_from_name.lower(), "year": str(year), "index": str(index)}

    raise ValueError(f"Cannot parse metadata from filename: {filename}")


# -------------------------------------------------------------------------
# OUTPUT FILENAME BUILDER (LV1–3 지원)
# -------------------------------------------------------------------------
def build_llm_filename(meta: Dict[str, str], level: int | None = None) -> str:
    """
    Build output filename for generated LLM files.
    Includes:
      - Model tag (DS / G4B / G12B / LMK)
      - Level suffix (_LV1, _LV2, _LV3)
    """
    provider = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()
    tag = MODEL_TAGS.get(provider, "DS")

    genre = meta["genre"]
    subfield = meta["subfield"].capitalize() if genre.lower() == "blogs" else meta["subfield"]

    tag_with_level = f"{tag}_LV{level}" if level else tag

    return f"{genre}_{subfield}_{tag_with_level}_{meta['year']}_{meta['index']}.txt"

