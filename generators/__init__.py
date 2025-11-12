"""
Generator modules for LLM text generation.

This package provides genre-specific generators that inherit from BaseGenerator.
"""

from .base_generator import BaseGenerator
from .academic_generator import AcademicGenerator
from .news_generator import NewsGenerator
from .blogs_generator import BlogsGenerator

__all__ = [
    "BaseGenerator",
    "AcademicGenerator",
    "NewsGenerator",
    "BlogsGenerator",
]

