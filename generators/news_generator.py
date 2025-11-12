"""
News text generator.

This module provides a generator specifically for News texts.
It inherits from BaseGenerator and can override methods for News-specific logic.
"""

from typing import Dict
from .base_generator import BaseGenerator


class NewsGenerator(BaseGenerator):
    """
    Generator for News texts.
    
    Currently uses all default behavior from BaseGenerator.
    Can be extended with News-specific logic if needed.
    """
    
    @property
    def genre(self) -> str:
        """Return the genre name."""
        return "News"
    
    # Example: If News needs special max_tokens calculation, override here:
    # def calculate_max_tokens(self, word_count: int) -> int:
    #     # Custom max_tokens calculation for News
    #     return min(3000, int(word_count * 2.0))

