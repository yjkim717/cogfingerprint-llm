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
    Parse genre, subfield, author_id (batch), year, and index from a Human dataset path.
    
    Unified format for all genres: {GENRE}_{SUBFIELD}_{AUTHOR_ID}_{YEAR}_{ITEM_INDEX}.txt
    Examples:
      - Blogs_LIFESTYLE_01_2020_01.txt
      - News_10years_01_2012_01.txt
      - Academic_CHEMISTRY_03_2022_01.txt
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
            raise ValueError(f"No Human/cleaned_human/human directory found in path: {human_path}")
            
        genre = parts[idx + 1]           # e.g. Academic / Blogs / News
        filename = parts[-1]             # Always the filename
    except (ValueError, IndexError):
        raise ValueError(f"Unexpected path structure: {human_path}")

    # --- Unified format: {GENRE}_{SUBFIELD}_{AUTHOR_ID}_{YEAR}_{ITEM_INDEX}.txt
    # This matches all three genres: Blogs, News, Academic
    # Examples:
    #   - Blogs_LIFESTYLE_01_2020_01.txt
    #   - News_10years_01_2012_01.txt
    #   - Academic_CHEMISTRY_03_2022_01.txt
    unified_pattern = r"^([A-Za-z]+)_([A-Za-z0-9]+)_(\d+)_(\d{4})_(\d+)\.txt$"
    m_unified = re.match(unified_pattern, filename)
    
    if m_unified:
        genre_from_name = m_unified.group(1)  # e.g., Blogs, News, Academic
        subfield_from_name = m_unified.group(2)  # e.g., LIFESTYLE, 10years, CHEMISTRY
        author_id = m_unified.group(3)          # e.g., 01, 03, 14
        year = m_unified.group(4)               # e.g., 2020, 2012, 2022
        item_index = m_unified.group(5)         # e.g., 01, 02, 03
        
        # Use genre from filename if it matches the directory, otherwise use directory genre
        # This handles cases where filename genre might differ from directory structure
        if genre_from_name.lower() == genre.lower():
            final_genre = genre_from_name
        else:
            final_genre = genre
        
        return {
            "genre": final_genre.lower(),  # Normalize to lowercase
            "subfield": subfield_from_name.lower(),  # Normalize to lowercase
            "batch": author_id,  # Author ID (also called batch)
            "year": str(year),
            "index": str(item_index),
        }
    
    # --- Fallback: Try to extract genre from directory if filename doesn't match unified format
    # This handles legacy formats or files that haven't been renamed yet
    year_match = re.search(r"\d{4}", human_path)
    year = year_match.group(0) if year_match else "0000"
    
    # Try to extract subfield from directory structure (for Academic files in subdirectories)
    subfield = None
    try:
        if idx + 2 < len(parts):
            subfield = parts[idx + 2]  # e.g., bio, chemistry, cs
    except (ValueError, IndexError):
        pass
    
    return {
        "genre": genre.lower(),
        "subfield": subfield.lower() if subfield else "unknown",
        "batch": "01",  # Default batch/author_id
        "year": str(year),
        "index": "01",  # Default index
    }


# -------------------------------------------------------------------------
# OUTPUT FILENAME BUILDER
# -------------------------------------------------------------------------
def build_llm_filename(meta: Dict[str, str], level: int | None = None, provider: str | None = None) -> str:
    """
    Build output filename for generated LLM files.
    Uses unified format for all genres: {GENRE}_{SUBFIELD}_{AUTHOR_ID}_{YEAR}_{ITEM_INDEX}_{MODEL}_{LEVEL}.txt
    
    Includes:
      - Model tag (DS / G4B / G12B / LMK)
      - Level suffix (_LV1, _LV2, _LV3)
    
    Args:
        meta: Metadata dictionary with genre, subfield, batch, year, index
        level: Generation level (1, 2, or 3)
        provider: LLM provider (DEEPSEEK, GEMMA_4B, GEMMA_12B, LLAMA_MAVRICK). 
                 If None, reads from environment variable LLM_PROVIDER.
    
    Examples:
      - Blogs_LIFESTYLE_01_2020_01_DS_LV1.txt
      - News_10years_01_2012_01_DS_LV1.txt
      - Academic_CHEMISTRY_03_2022_01_DS_LV1.txt
    """
    # Use provided provider or fall back to environment variable
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()
    else:
        provider = provider.upper()
    tag = MODEL_TAGS.get(provider, "DS")

    genre = meta["genre"]
    tag_with_level = f"{tag}_LV{level}" if level else tag

    # Unified format: {GENRE}_{SUBFIELD}_{AUTHOR_ID}_{YEAR}_{ITEM_INDEX}_{MODEL}_{LEVEL}.txt
    # Capitalize genre (first letter uppercase)
    genre_capitalized = genre.capitalize()  # e.g., blogs -> Blogs, news -> News, academic -> Academic
    
    # Handle subfield capitalization based on genre
    subfield = meta["subfield"]
    if genre.lower() == "blogs":
        # Blogs: SUBFIELD in uppercase (e.g., LIFESTYLE)
        subfield_formatted = subfield.upper()
    elif genre.lower() == "news":
        # News: subfield in lowercase (e.g., 10years)
        subfield_formatted = subfield.lower()
    elif genre.lower() == "academic":
        # Academic: SUBFIELD in uppercase (e.g., CHEMISTRY)
        subfield_formatted = subfield.upper()
    else:
        # Default: capitalize first letter
        subfield_formatted = subfield.capitalize()
    
    # Get author_id (batch)
    author_id = meta.get("batch", "01")  # Default to "01" if batch not present
    year = meta["year"]
    item_index = meta["index"]
    
    # Build unified filename
    return f"{genre_capitalized}_{subfield_formatted}_{author_id}_{year}_{item_index}_{tag_with_level}.txt"
