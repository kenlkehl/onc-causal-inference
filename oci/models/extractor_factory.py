# oci/models/extractor_factory.py
"""Factory function for creating feature extractors.

This module centralizes feature extractor instantiation logic that was previously
duplicated across CausalText, CausalTextForest, and PropensityOnlyModel.
"""

import logging
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn

from .frozen_llm_pooler_extractor import FrozenLLMPoolerExtractor
from ..config import normalize_feature_extractor_type


logger = logging.getLogger(__name__)


def create_feature_extractor(
    extractor_type: str,
    device: torch.device,
    # Frozen LLM Pooler args
    flp_model_name: str = "Qwen/Qwen3-0.6B-Base",
    flp_max_length: int = 8192,
    flp_freeze_llm: bool = True,
    flp_gated_attention_dim: int = 128,
    flp_projection_dim: int = 128,
    flp_dropout: float = 0.1,
    flp_gradient_checkpointing: bool = True,
    flp_downprojection_dim: Optional[int] = None,
    flp_skip_llm: bool = False,
    flp_cached_hidden_size: int = 0,
    # Numeric feature args
    numeric_features_enabled: bool = False,
    numeric_embedding_dim: int = 32,
    numeric_magnitude_bins: int = 8,
    numeric_type_categories: int = 10,
    # Model type
    model_type: str = "dragonnet",
) -> nn.Module:
    """
    Create a feature extractor based on the specified type.

    Currently only supports frozen_llm_pooler.

    Args:
        extractor_type: Type of feature extractor ("frozen_llm_pooler")
        device: PyTorch device to use
        ... (extractor-specific args)
        model_type: Model type ("dragonnet", "rlearner", etc.)

    Returns:
        nn.Module: The instantiated feature extractor
    """
    normalized_type = normalize_feature_extractor_type(extractor_type)

    if normalized_type != "frozen_llm_pooler":
        raise ValueError(
            f"Unsupported feature extractor type: '{extractor_type}'. "
            f"Only 'frozen_llm_pooler' is supported."
        )

    extractor = FrozenLLMPoolerExtractor(
        model_name=flp_model_name,
        max_length=flp_max_length,
        freeze_llm=flp_freeze_llm,
        gated_attention_dim=flp_gated_attention_dim,
        projection_dim=flp_projection_dim,
        dropout=flp_dropout,
        gradient_checkpointing=flp_gradient_checkpointing,
        downprojection_dim=flp_downprojection_dim,
        numeric_features_enabled=numeric_features_enabled,
        numeric_embedding_dim=numeric_embedding_dim,
        numeric_magnitude_bins=numeric_magnitude_bins,
        numeric_type_categories=numeric_type_categories,
        device=device,
        skip_llm=flp_skip_llm,
        cached_hidden_size=flp_cached_hidden_size,
    )
    mode = "cached" if flp_skip_llm else ("frozen" if flp_freeze_llm else "trainable")
    logger.info(f"Created Frozen LLM Pooler extractor: {flp_model_name} "
               f"({mode}), "
               f"max_length={flp_max_length}, gated_attention_dim={flp_gated_attention_dim}, "
               f"projection_dim={flp_projection_dim}")
    return extractor


def create_feature_extractor_from_config(
    config: Dict[str, Any],
    device: torch.device,
    model_type: str = "dragonnet"
) -> nn.Module:
    """
    Create a feature extractor from a configuration dictionary.

    Args:
        config: Configuration dictionary (typically from CausalText.config)
        device: PyTorch device
        model_type: Model type for task-specific extractors

    Returns:
        nn.Module: The instantiated feature extractor
    """
    return create_feature_extractor(
        extractor_type=config.get('feature_extractor_type', 'frozen_llm_pooler'),
        device=device,
        model_type=model_type,
        # Frozen LLM Pooler args
        flp_model_name=config.get('flp_model_name', 'Qwen/Qwen3-0.6B-Base'),
        flp_max_length=config.get('flp_max_length', 8192),
        flp_freeze_llm=config.get('flp_freeze_llm', True),
        flp_gated_attention_dim=config.get('flp_gated_attention_dim', 128),
        flp_projection_dim=config.get('flp_projection_dim', 128),
        flp_dropout=config.get('flp_dropout', 0.1),
        flp_gradient_checkpointing=config.get('flp_gradient_checkpointing', True),
        flp_downprojection_dim=config.get('flp_downprojection_dim', None),
        flp_skip_llm=config.get('flp_skip_llm', False),
        flp_cached_hidden_size=config.get('flp_cached_hidden_size', 0),
        # Numeric args
        numeric_features_enabled=config.get('numeric_features_enabled', False),
        numeric_embedding_dim=config.get('numeric_embedding_dim', 32),
        numeric_magnitude_bins=config.get('numeric_magnitude_bins', 8),
        numeric_type_categories=config.get('numeric_type_categories', 10),
    )
