"""
Macro-level text generators.

This module provides generators for macro datasets that reuse the existing
BaseGenerator logic but parse metadata from macro_dataset/human paths with
the format:

    {GENRE}_{FIELD}_{YEAR}_{INDEX}.txt

Examples:
    - NEWS_WORLD_2024_200.txt
    - Academic_BIOLOGY_2020_001.txt

These generators are intended to be used with macro_dataset/human as the
input directory and macro_dataset/llm as the output directory.
"""

from typing import Dict

from .news_generator import NewsGenerator
from .blogs_generator import BlogsGenerator
from .academic_generator import AcademicGenerator
from utils.file_utils import parse_macro_metadata_from_path, MODEL_TAGS


class MacroNewsGenerator(NewsGenerator):
    """
    Generator for macro-level News texts.
    
    This reuses all behavior from NewsGenerator / BaseGenerator, but overrides
    metadata extraction to support macro_dataset file naming.
    """

    def extract_metadata(self, human_fp: str) -> Dict[str, str]:
        """Extract metadata from macro_dataset/human/news paths."""
        return parse_macro_metadata_from_path(human_fp)

    def get_output_filename(self, meta: Dict[str, str], level: int) -> str:
        """
        Build output filename for macro News without author/batch segment.
        
        Format:
            News_{subfield}_{year}_{index}_{MODEL}_LV{level}.txt
        Example:
            News_world_2024_200_G12B_LV1.txt
        """
        provider_tag = MODEL_TAGS.get(self.provider, "DS")
        tag_with_level = f"{provider_tag}_LV{level}"

        genre_capitalized = meta["genre"].capitalize()
        subfield = meta["subfield"]
        # Follow same subfield casing rules as build_llm_filename
        if meta["genre"].lower() == "news":
            subfield_formatted = subfield.lower()
        else:
            subfield_formatted = subfield.capitalize()

        year = meta["year"]
        index = meta["index"]

        return f"{genre_capitalized}_{subfield_formatted}_{year}_{index}_{tag_with_level}.txt"


class MacroBlogsGenerator(BlogsGenerator):
    """
    Generator for macro-level Blogs texts.
    
    This reuses all behavior from BlogsGenerator / BaseGenerator, but overrides
    metadata extraction to support macro_dataset file naming.
    """

    def extract_metadata(self, human_fp: str) -> Dict[str, str]:
        """Extract metadata from macro_dataset/human/blogs paths."""
        return parse_macro_metadata_from_path(human_fp)

    def get_output_filename(self, meta: Dict[str, str], level: int) -> str:
        """
        Build output filename for macro Blogs without author/batch segment.
        
        Format:
            Blogs_{SUBFIELD}_{year}_{index}_{MODEL}_LV{level}.txt
        """
        provider_tag = MODEL_TAGS.get(self.provider, "DS")
        tag_with_level = f"{provider_tag}_LV{level}"

        genre_capitalized = meta["genre"].capitalize()
        subfield = meta["subfield"]
        # Blogs: SUBFIELD uppercase
        if meta["genre"].lower() == "blogs":
            subfield_formatted = subfield.upper()
        else:
            subfield_formatted = subfield.capitalize()

        year = meta["year"]
        index = meta["index"]

        return f"{genre_capitalized}_{subfield_formatted}_{year}_{index}_{tag_with_level}.txt"


class MacroAcademicGenerator(AcademicGenerator):
    """
    Generator for macro-level Academic texts.
    
    This reuses all behavior from AcademicGenerator / BaseGenerator, but
    overrides metadata extraction to support macro_dataset file naming.
    """

    def extract_metadata(self, human_fp: str) -> Dict[str, str]:
        """Extract metadata from macro_dataset/human/academic paths."""
        return parse_macro_metadata_from_path(human_fp)

    def get_output_filename(self, meta: Dict[str, str], level: int) -> str:
        """
        Build output filename for macro Academic without author/batch segment.
        
        Target format (user requirement):
            Academic_{SUBFIELD}_{year}_{index}_{MODEL}_LV{level}.txt
        Example:
            Academic_CHEMISTRY_2023_068_G12B_LV1.txt
        """
        provider_tag = MODEL_TAGS.get(self.provider, "DS")
        tag_with_level = f"{provider_tag}_LV{level}"

        genre_capitalized = meta["genre"].capitalize()
        subfield = meta["subfield"]
        # Academic: SUBFIELD uppercase
        if meta["genre"].lower() == "academic":
            subfield_formatted = subfield.upper()
        else:
            subfield_formatted = subfield.capitalize()

        year = meta["year"]
        index = meta["index"]

        return f"{genre_capitalized}_{subfield_formatted}_{year}_{index}_{tag_with_level}.txt"


