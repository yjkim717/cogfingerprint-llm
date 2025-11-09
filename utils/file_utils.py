import os
import re
from typing import Dict
from config import get_llm_path

# -------------------------------------------------------------------------
# MODEL TAGS
# -------------------------------------------------------------------------
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
# METADATA PARSING (supports Academic / Blogs / News / custom structures)
# -------------------------------------------------------------------------
def parse_metadata_from_path(human_path: str) -> Dict[str, str]:
    """
    Parse genre, subfield, year, and index from a Human dataset path.

    Supports:
      - Academic: Datasets/Human/Academic/bio/2020/Bio_DS_2020_01.txt
      - Blogs:    Datasets/Human/Blogs/Lifestyle/2022/LIFESTYLE_2022_10_02.txt
      - News:     Datasets/Human/News/politics/2023/NEWS_2023_01.txt
      - Custom:   Datasets/Human/News/3_years/3_years_01_2015_01.txt
    """
    parts = os.path.normpath(human_path).split(os.sep)

    try:
        # Support "Human", "cleaned_human", and "human" directories
        if "Human" in parts:
            idx = parts.index("Human")
        elif "cleaned_human" in parts:
            idx = parts.index("cleaned_human")
        elif "human" in parts:
            idx = parts.index("human")
        else:
            raise ValueError(
                f"No Human/cleaned_human/human directory found in path: {human_path}"
            )

        genre = parts[idx + 1]  # e.g. Academic / Blogs / News
        subfield = (
            parts[idx + 2] if idx + 2 < len(parts) else None
        )  # e.g. bio / Lifestyle / 3_years / filtered_years
        filename = parts[-1]  # Always the filename

        # For News in filtered_years directory or directly in news directory, subfield might be filtered_years or None
        if subfield == "filtered_years" or (
            genre.lower() == "news" and subfield is None
        ):
            # For files directly in news directory, parse from filename
            # Format: News_10years_01_2012_01.txt
            # We'll extract subfield from filename if needed
            subfield = "filtered_years"  # Use as default for news files
    except (ValueError, IndexError):
        raise ValueError(f"Unexpected path structure: {human_path}")

    # --- Case 1: Custom News files (e.g., 3_years_01_2015_01.txt or News_10years_01_2012_01.txt)
    m_custom = re.search(r"([A-Za-z0-9_]+)_(\d{2})_(\d{4})_(\d{2})\.txt$", filename)
    if m_custom:
        subfield_name = m_custom.group(1)  # e.g., 3_years or News_10years
        batch = m_custom.group(2)  # e.g., 01
        year = m_custom.group(3)  # e.g., 2015
        index = m_custom.group(4)  # e.g., 01

        # Remove "News_" prefix if present (for files like News_10years_01_2012_01.txt)
        if subfield_name.lower().startswith("news_"):
            subfield_name = subfield_name[5:]  # Remove "News_" prefix (5 characters)

        return {
            "genre": genre,
            "subfield": subfield_name.lower(),
            "batch": batch,
            "year": str(year),
            "index": str(index),
        }

    # --- Case 2: Blogs (LIFESTYLE_2022_10_02.txt)
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

    # --- Case 3: Academic (Bio_DS_2020_01.txt)
    m_acad = re.search(r"_(\d{4})_(\d+)\.txt$", filename)
    if m_acad:
        year = m_acad.group(1)
        index = m_acad.group(2)
        return {
            "genre": genre,
            "subfield": subfield.lower(),
            "year": str(year),
            "index": str(index),
        }

    # --- Case 4: Standard News (NEWS_2023_01.txt)
    m_news = re.search(r"([A-Za-z]+)_(\d{4})_(\d+)\.txt$", filename)
    if m_news:
        subfield_from_name = m_news.group(1)
        year = m_news.group(2)
        index = m_news.group(3)
        return {
            "genre": genre,
            "subfield": subfield_from_name.lower(),
            "year": str(year),
            "index": str(index),
        }

    # --- Fallback (extract first 4-digit year from anywhere)
    year_match = re.search(r"\d{4}", human_path)
    year = year_match.group(0) if year_match else "0000"
    return {
        "genre": genre,
        "subfield": subfield.lower(),
        "year": str(year),
        "index": "01",
    }


# -------------------------------------------------------------------------
# OUTPUT FILENAME BUILDER
# -------------------------------------------------------------------------
def build_llm_filename(meta: Dict[str, str], level: int | None = None) -> str:
    """
    Build output filename for generated LLM files.
    Includes:
      - Model tag (DS / G4B / G12B / LMK)
      - Level suffix (_LV1, _LV2, _LV3)

    Format for News: {subfield}_{batch}_{year}_{index}_{model}_{level}.txt
    Format for others: {genre}_{subfield}_{model}_{level}_{year}_{index}.txt
    """
    provider = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()
    tag = MODEL_TAGS.get(provider, "DS")

    genre = meta["genre"]
    tag_with_level = f"{tag}_LV{level}" if level else tag

    # News files: News_subfield_batch_year_index_model_level.txt
    if genre.lower() == "news" and "batch" in meta:
        subfield = meta["subfield"]
        batch = meta["batch"]
        year = meta["year"]
        index = meta["index"]
        # Format: News_5years_01_2016_01_DS_LV1.txt (with News_ prefix)
        return f"News_{subfield}_{batch}_{year}_{index}_{tag_with_level}.txt"

    # Other files: genre_subfield_model_level_year_index.txt
    subfield = (
        meta["subfield"].capitalize() if genre.lower() == "blogs" else meta["subfield"]
    )
    return f"{genre}_{subfield}_{tag_with_level}_{meta['year']}_{meta['index']}.txt"
