# utils/file_utils.py

import os
import re
from typing import Dict
from config import get_llm_path

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def parse_metadata_from_path(human_path: str) -> Dict[str, str]:
    parts = os.path.normpath(human_path).split(os.sep)
    idx = parts.index("Human")
    genre_group, subfield, year, filename = parts[idx+1:idx+5]
    genre = genre_group.split("-")[0]
    match = re.search(r"_(\d+)\.txt$", filename)
    index = match.group(1) if match else "01"
    return {"genre": genre, "subfield": subfield, "year": year, "index": index}

def build_llm_filename(meta):
    return f"{meta['genre']}_{meta['subfield']}_DS_{meta['year']}_{meta['index']}.txt"
