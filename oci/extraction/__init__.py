# oci/extraction/__init__.py
"""Extraction module for CDT.

This module provides LLM-based extraction of explicit features from clinical text.
"""

from .explicit_features import (
    ExplicitFeatureValue,
    VLLMFeatureExtractor,
    build_extraction_prompt,
    parse_extraction_response,
    extract_explicit_features,
)
from .cache import ExtractionCache

# Backward-compatible import aliases. Old config keys are still rejected.
ExplicitConfounderValue = ExplicitFeatureValue
VLLMConfounderExtractor = VLLMFeatureExtractor
extract_explicit_confounders = extract_explicit_features

__all__ = [
    "ExplicitFeatureValue",
    "VLLMFeatureExtractor",
    "build_extraction_prompt",
    "parse_extraction_response",
    "extract_explicit_features",
    "ExtractionCache",
    "ExplicitConfounderValue",
    "VLLMConfounderExtractor",
    "extract_explicit_confounders",
]
