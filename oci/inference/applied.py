"""Standalone dispatch for the retained explicit-feature workflows.

The production multi-model path is owned by
``ResearchAllEvidenceWorkflow``.  This module intentionally contains no
fallback neural head: every supported standalone model has an explicit,
auditable dispatch branch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch

from ..config import AppliedInferenceConfig

logger = logging.getLogger(__name__)

STANDALONE_MODEL_TYPES = (
    "explicit_feature_forest",
    "agentic_explicit_feature_forest",
    "agentic_attention_variable_forest",
    "multi_model_agentic_forest",
)


def _run_explicit_feature_extraction(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """Materialize investigator-specified explicit features, with caching.

    Uses the shared explicit-feature extraction provider, so endpoint, retry,
    reasoning, temporal, grouping, and context-strategy settings are honored
    exactly as in the agentic explicit-feature runners.
    """

    from .agentic_explicit_feature_forest import VLLMExplicitFeatureExtractionProvider

    specs = list(config.explicit_features.features)
    if not specs:
        raise ValueError(
            "explicit_feature_forest requires at least one explicit feature contract"
        )
    provider = VLLMExplicitFeatureExtractionProvider(
        config=config,
        output_dir=Path(output_path).parent / "explicit_feature_extraction",
    )
    enriched = provider.ensure_features(dataset, specs)
    return enriched, [f"explicit_feat_{spec.name}" for spec in specs]


def run_applied_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    gpu_ids: Optional[List[int]] = None,
    num_workers: int = 1,
) -> None:
    """Dispatch one retained standalone explicit-feature architecture."""

    model_type = str(config.architecture.model_type)

    if model_type == "agentic_explicit_feature_forest":
        from .agentic_explicit_feature_forest import run_agentic_explicit_feature_forest

        run_agentic_explicit_feature_forest(
            dataset=dataset,
            config=config,
            output_path=output_path,
            device=device,
            num_workers=num_workers,
        )
        return

    if model_type == "agentic_attention_variable_forest":
        from .agentic_attention_variable_forest import run_agentic_attention_variable_forest

        run_agentic_attention_variable_forest(
            dataset=dataset,
            config=config,
            output_path=output_path,
            device=device,
            num_workers=num_workers,
            gpu_ids=gpu_ids,
        )
        return

    if model_type == "multi_model_agentic_forest":
        from .multi_model_agentic_forest import run_multi_model_agentic_forest

        run_multi_model_agentic_forest(
            dataset=dataset,
            config=config,
            output_path=output_path,
            device=device,
            gpu_ids=gpu_ids,
            num_workers=num_workers,
        )
        return

    if model_type == "explicit_feature_forest":
        from .applied_explicit_feature_forest import (
            run_applied_inference_explicit_feature_forest,
        )

        explicit_columns = None
        if config.explicit_features.enabled:
            dataset, explicit_columns = _run_explicit_feature_extraction(
                dataset,
                config,
                output_path,
            )
        run_applied_inference_explicit_feature_forest(
            dataset=dataset,
            config=config,
            output_path=output_path,
            device=device,
            num_workers=num_workers,
            explicit_feature_columns=explicit_columns,
        )
        return

    if model_type == "multi_model_forest":
        raise RuntimeError(
            "model_type='multi_model_forest' must run through "
            "ResearchAllEvidenceWorkflow (scripts/run_all_evidence.py)"
        )
    raise ValueError(
        f"Unsupported retired model_type={model_type!r}; retained standalone types: "
        + ", ".join(STANDALONE_MODEL_TYPES)
    )


__all__ = [
    "STANDALONE_MODEL_TYPES",
    "run_applied_inference",
]
