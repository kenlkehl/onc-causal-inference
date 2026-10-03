"""Standalone dispatch for the retained explicit-feature workflows.

The production multi-model path is owned by
``ResearchAllEvidenceWorkflow``.  This module intentionally contains no
fallback neural head: every supported standalone model has an explicit,
auditable dispatch branch.
"""

from __future__ import annotations

import logging
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch

from ..config import AppliedInferenceConfig, ExplicitFeatureSpec
from ..extraction import ExtractionCache, VLLMFeatureExtractor, resolve_vllm_reasoning_parser

logger = logging.getLogger(__name__)

STANDALONE_MODEL_TYPES = (
    "explicit_feature_forest",
    "agentic_explicit_feature_forest",
    "agentic_attention_variable_forest",
    "multi_model_agentic_forest",
)


def _get_explicit_feature_specs(
    config: AppliedInferenceConfig,
) -> Optional[List[ExplicitFeatureSpec]]:
    """Return configured explicit-feature contracts when extraction is enabled."""

    if config.explicit_features.enabled:
        return list(config.explicit_features.features)
    return None


def _run_explicit_feature_extraction(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """Materialize investigator-specified explicit features, with caching."""

    del output_path  # The extraction cache owns its configured location.
    feature_config = config.explicit_features
    specs = list(feature_config.features)
    if not specs:
        raise ValueError(
            "explicit_feature_forest requires at least one explicit feature contract"
        )

    cache = ExtractionCache(cache_dir=feature_config.cache_dir)
    cache_config = {
        "features": specs,
        "prompt_template_version": "explicit_features_v2",
        "vllm_model_name": feature_config.vllm_model_name,
        "vllm_max_model_len": feature_config.vllm_max_model_len,
        "vllm_reasoning_parser": resolve_vllm_reasoning_parser(
            feature_config.vllm_reasoning_parser,
            feature_config.vllm_model_name,
        ),
        "extraction_temperature": feature_config.extraction_temperature,
        "extraction_max_tokens": feature_config.extraction_max_tokens,
        "extraction_max_text_length": feature_config.extraction_max_text_length,
    }
    cache_config["patient_text_hash"] = hashlib.sha256(json.dumps(
        dataset[config.text_column].fillna("").astype(str).tolist(),
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    cache_config["extraction_context_strategy"] = feature_config.extraction_context_strategy
    if feature_config.extraction_context_strategy == "colbert":
        from ..extraction.colbert import retrieval_identity
        cache_config["colbert"] = retrieval_identity(feature_config.colbert)
    cached = None
    if feature_config.cache_enabled:
        cached = cache.load_if_valid(
            config.dataset_path,
            cache_config,
            expected_rows=len(dataset),
        )

    if cached is None:
        extractor = VLLMFeatureExtractor(
            specs=specs,
            mode=feature_config.vllm_mode,
            server_url=feature_config.vllm_server_url or "http://localhost:8000/v1",
            model_name=feature_config.vllm_model_name,
            tensor_parallel_size=feature_config.vllm_tensor_parallel_size,
            gpu_memory_utilization=feature_config.vllm_gpu_memory_utilization,
            download_dir=feature_config.vllm_download_dir,
            max_model_len=feature_config.vllm_max_model_len,
            vllm_reasoning_parser=feature_config.vllm_reasoning_parser,
            max_retries=feature_config.extraction_max_retries,
            temperature=feature_config.extraction_temperature,
            max_tokens=feature_config.extraction_max_tokens,
            max_text_length=feature_config.extraction_max_text_length,
            context_strategy=feature_config.extraction_context_strategy,
            colbert=feature_config.colbert,
        )
        try:
            cached = extractor.extract_to_dataframe(
                dataset[config.text_column].tolist(),
                batch_size=feature_config.extraction_batch_size,
            )
        finally:
            extractor.cleanup()
        if feature_config.cache_enabled:
            cache.save(config.dataset_path, cache_config, cached)

    enriched = dataset.copy()
    for column in cached.columns:
        enriched[column] = cached[column].to_numpy()
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
