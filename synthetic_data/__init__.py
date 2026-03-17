# synthetic_data/__init__.py
"""LLM-based synthetic clinical data generation for causal inference benchmarking."""

from .config import SyntheticDataConfig, LLMConfig, StructuredDataConfig
from .generator import generate_synthetic_dataset
from .structured_data import convert_structured_event_to_text, STRUCTURED_EVENT_TYPES

__all__ = [
    "SyntheticDataConfig",
    "LLMConfig",
    "StructuredDataConfig",
    "generate_synthetic_dataset",
    "convert_structured_event_to_text",
    "STRUCTURED_EVENT_TYPES",
]
