# oci/extraction/__init__.py
"""Extraction module for CDT.

This module provides LLM-based extraction of explicit confounders from clinical text.
"""

from .explicit_confounders import (
    ExplicitConfounderValue,
    VLLMConfounderExtractor,
    build_extraction_prompt,
    parse_extraction_response,
    extract_explicit_confounders,
)
from .cache import ExtractionCache

__all__ = [
    "ExplicitConfounderValue",
    "VLLMConfounderExtractor",
    "build_extraction_prompt",
    "parse_extraction_response",
    "extract_explicit_confounders",
    "ExtractionCache",
]
