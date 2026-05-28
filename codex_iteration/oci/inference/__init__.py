# oci/inference/__init__.py

"""Inference modules for CDT."""

from .applied import run_applied_inference

# Lazy import for forest inference (requires econml)
def run_applied_inference_forest(*args, **kwargs):
    from .applied_forest import run_applied_inference_forest as _run_forest
    return _run_forest(*args, **kwargs)


def run_agentic_explicit_feature_forest(*args, **kwargs):
    from .agentic_explicit_feature_forest import run_agentic_explicit_feature_forest as _run_agentic
    return _run_agentic(*args, **kwargs)


__all__ = [
    'run_applied_inference',
    'run_applied_inference_forest',
    'run_agentic_explicit_feature_forest',
]
