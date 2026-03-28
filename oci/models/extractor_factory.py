# oci/models/extractor_factory.py
"""Factory function for creating feature extractors.

This module centralizes feature extractor instantiation logic that was previously
duplicated across CausalText, CausalTextForest, and PropensityOnlyModel.
"""

import logging
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn

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
    flp_chat_template_prompt: Optional[str] = None,
    # Hierarchical LLM args
    hlm_model_name: str = "Qwen/Qwen3-0.6B-Base",
    hlm_chunk_size: int = 2048,
    hlm_chunk_overlap: int = 256,
    hlm_max_chunks: int = 16,
    hlm_freeze_llm: bool = True,
    hlm_gated_attention_dim: int = 128,
    hlm_projection_dim: int = 128,
    hlm_dropout: float = 0.1,
    hlm_gradient_checkpointing: bool = True,
    hlm_downprojection_dim: Optional[int] = None,
    hlm_skip_llm: bool = False,
    hlm_cached_hidden_size: int = 0,
    hlm_chat_template_prompt: Optional[str] = None,
    # Hierarchical CNN args
    hcnn_embedding_dim: int = 256,
    hcnn_conv_dim: int = 256,
    hcnn_kernel_size: int = 5,
    hcnn_num_conv_blocks: int = 4,
    hcnn_chunk_size: int = 512,
    hcnn_chunk_overlap: int = 64,
    hcnn_max_chunks: int = 32,
    hcnn_vocab_size: int = 50000,
    hcnn_gated_attention_dim: int = 128,
    hcnn_projection_dim: int = 128,
    hcnn_dropout: float = 0.1,
    # Hierarchical GRU args
    hgru_embedding_dim: int = 256,
    hgru_gru_hidden_dim: int = 256,
    hgru_num_gru_layers: int = 2,
    hgru_chunk_size: int = 512,
    hgru_chunk_overlap: int = 64,
    hgru_max_chunks: int = 32,
    hgru_vocab_size: int = 50000,
    hgru_gated_attention_dim: int = 128,
    hgru_projection_dim: int = 128,
    hgru_dropout: float = 0.1,
    # Simple CNN args
    scnn_embedding_dim: int = 256,
    scnn_conv_dim: int = 256,
    scnn_kernel_size: int = 5,
    scnn_num_conv_blocks: int = 4,
    scnn_max_length: int = 10000,
    scnn_vocab_size: int = 50000,
    scnn_gated_attention_dim: int = 128,
    scnn_projection_dim: int = 128,
    scnn_dropout: float = 0.1,
    # Model type
    model_type: str = "dragonnet",
) -> nn.Module:
    """
    Create a feature extractor based on the specified type.

    Args:
        extractor_type: Type of feature extractor
        device: PyTorch device to use
        model_type: Model type ("dragonnet", "rlearner", etc.)

    Returns:
        nn.Module: The instantiated feature extractor
    """
    normalized_type = normalize_feature_extractor_type(extractor_type)

    if normalized_type == "frozen_llm_pooler":
        from .frozen_llm_pooler_extractor import FrozenLLMPoolerExtractor
        extractor = FrozenLLMPoolerExtractor(
            model_name=flp_model_name,
            max_length=flp_max_length,
            freeze_llm=flp_freeze_llm,
            gated_attention_dim=flp_gated_attention_dim,
            projection_dim=flp_projection_dim,
            dropout=flp_dropout,
            gradient_checkpointing=flp_gradient_checkpointing,
            downprojection_dim=flp_downprojection_dim,
            device=device,
            skip_llm=flp_skip_llm,
            cached_hidden_size=flp_cached_hidden_size,
            chat_template_prompt=flp_chat_template_prompt,
        )
        mode = "cached" if flp_skip_llm else ("frozen" if flp_freeze_llm else "trainable")
        logger.info(f"Created Frozen LLM Pooler extractor: {flp_model_name} "
                    f"({mode}), max_length={flp_max_length}, "
                    f"projection_dim={flp_projection_dim}")
        return extractor

    elif normalized_type == "hierarchical_llm":
        from .hierarchical_llm_extractor import HierarchicalLLMExtractor
        extractor = HierarchicalLLMExtractor(
            model_name=hlm_model_name,
            chunk_size=hlm_chunk_size,
            chunk_overlap=hlm_chunk_overlap,
            max_chunks=hlm_max_chunks,
            freeze_llm=hlm_freeze_llm,
            gated_attention_dim=hlm_gated_attention_dim,
            projection_dim=hlm_projection_dim,
            dropout=hlm_dropout,
            gradient_checkpointing=hlm_gradient_checkpointing,
            downprojection_dim=hlm_downprojection_dim,
            device=device,
            skip_llm=hlm_skip_llm,
            cached_hidden_size=hlm_cached_hidden_size,
            chat_template_prompt=hlm_chat_template_prompt,
        )
        mode = "cached" if hlm_skip_llm else ("frozen" if hlm_freeze_llm else "trainable")
        logger.info(f"Created Hierarchical LLM extractor: {hlm_model_name} "
                    f"({mode}), chunk_size={hlm_chunk_size}, max_chunks={hlm_max_chunks}, "
                    f"projection_dim={hlm_projection_dim}")
        return extractor

    elif normalized_type == "hierarchical_cnn":
        from .hierarchical_cnn_extractor import HierarchicalCNNExtractor
        extractor = HierarchicalCNNExtractor(
            embedding_dim=hcnn_embedding_dim,
            conv_dim=hcnn_conv_dim,
            kernel_size=hcnn_kernel_size,
            num_conv_blocks=hcnn_num_conv_blocks,
            chunk_size=hcnn_chunk_size,
            chunk_overlap=hcnn_chunk_overlap,
            max_chunks=hcnn_max_chunks,
            vocab_size=hcnn_vocab_size,
            gated_attention_dim=hcnn_gated_attention_dim,
            projection_dim=hcnn_projection_dim,
            dropout=hcnn_dropout,
            device=device,
        )
        logger.info(f"Created Hierarchical CNN extractor: "
                    f"conv_dim={hcnn_conv_dim}, num_blocks={hcnn_num_conv_blocks}, "
                    f"chunk_size={hcnn_chunk_size}, max_chunks={hcnn_max_chunks}, "
                    f"projection_dim={hcnn_projection_dim}")
        return extractor

    elif normalized_type == "hierarchical_gru":
        from .hierarchical_gru_extractor import HierarchicalGRUExtractor
        extractor = HierarchicalGRUExtractor(
            embedding_dim=hgru_embedding_dim,
            gru_hidden_dim=hgru_gru_hidden_dim,
            num_gru_layers=hgru_num_gru_layers,
            chunk_size=hgru_chunk_size,
            chunk_overlap=hgru_chunk_overlap,
            max_chunks=hgru_max_chunks,
            vocab_size=hgru_vocab_size,
            gated_attention_dim=hgru_gated_attention_dim,
            projection_dim=hgru_projection_dim,
            dropout=hgru_dropout,
            device=device,
        )
        logger.info(f"Created Hierarchical GRU extractor: "
                    f"gru_hidden_dim={hgru_gru_hidden_dim}, num_layers={hgru_num_gru_layers}, "
                    f"chunk_size={hgru_chunk_size}, max_chunks={hgru_max_chunks}, "
                    f"projection_dim={hgru_projection_dim}")
        return extractor

    elif normalized_type == "simple_cnn":
        from .simple_cnn_extractor import SimpleCNNExtractor
        extractor = SimpleCNNExtractor(
            embedding_dim=scnn_embedding_dim,
            conv_dim=scnn_conv_dim,
            kernel_size=scnn_kernel_size,
            num_conv_blocks=scnn_num_conv_blocks,
            max_length=scnn_max_length,
            vocab_size=scnn_vocab_size,
            gated_attention_dim=scnn_gated_attention_dim,
            projection_dim=scnn_projection_dim,
            dropout=scnn_dropout,
            device=device,
        )
        logger.info(f"Created Simple CNN extractor: "
                    f"conv_dim={scnn_conv_dim}, num_blocks={scnn_num_conv_blocks}, "
                    f"max_length={scnn_max_length}, projection_dim={scnn_projection_dim}")
        return extractor

    else:
        from ..config import VALID_EXTRACTOR_TYPES
        raise ValueError(
            f"Unsupported feature extractor type: '{extractor_type}'. "
            f"Supported types: {sorted(VALID_EXTRACTOR_TYPES)}"
        )


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
        flp_chat_template_prompt=config.get('flp_chat_template_prompt', None),
        # Hierarchical LLM args
        hlm_model_name=config.get('hlm_model_name', 'Qwen/Qwen3-0.6B-Base'),
        hlm_chunk_size=config.get('hlm_chunk_size', 2048),
        hlm_chunk_overlap=config.get('hlm_chunk_overlap', 256),
        hlm_max_chunks=config.get('hlm_max_chunks', 16),
        hlm_freeze_llm=config.get('hlm_freeze_llm', True),
        hlm_gated_attention_dim=config.get('hlm_gated_attention_dim', 128),
        hlm_projection_dim=config.get('hlm_projection_dim', 128),
        hlm_dropout=config.get('hlm_dropout', 0.1),
        hlm_gradient_checkpointing=config.get('hlm_gradient_checkpointing', True),
        hlm_downprojection_dim=config.get('hlm_downprojection_dim', None),
        hlm_skip_llm=config.get('hlm_skip_llm', False),
        hlm_cached_hidden_size=config.get('hlm_cached_hidden_size', 0),
        hlm_chat_template_prompt=config.get('hlm_chat_template_prompt', None),
        # Hierarchical CNN args
        hcnn_embedding_dim=config.get('hcnn_embedding_dim', 256),
        hcnn_conv_dim=config.get('hcnn_conv_dim', 256),
        hcnn_kernel_size=config.get('hcnn_kernel_size', 5),
        hcnn_num_conv_blocks=config.get('hcnn_num_conv_blocks', 4),
        hcnn_chunk_size=config.get('hcnn_chunk_size', 512),
        hcnn_chunk_overlap=config.get('hcnn_chunk_overlap', 64),
        hcnn_max_chunks=config.get('hcnn_max_chunks', 32),
        hcnn_vocab_size=config.get('hcnn_vocab_size', 50000),
        hcnn_gated_attention_dim=config.get('hcnn_gated_attention_dim', 128),
        hcnn_projection_dim=config.get('hcnn_projection_dim', 128),
        hcnn_dropout=config.get('hcnn_dropout', 0.1),
        # Hierarchical GRU args
        hgru_embedding_dim=config.get('hgru_embedding_dim', 256),
        hgru_gru_hidden_dim=config.get('hgru_gru_hidden_dim', 256),
        hgru_num_gru_layers=config.get('hgru_num_gru_layers', 2),
        hgru_chunk_size=config.get('hgru_chunk_size', 512),
        hgru_chunk_overlap=config.get('hgru_chunk_overlap', 64),
        hgru_max_chunks=config.get('hgru_max_chunks', 32),
        hgru_vocab_size=config.get('hgru_vocab_size', 50000),
        hgru_gated_attention_dim=config.get('hgru_gated_attention_dim', 128),
        hgru_projection_dim=config.get('hgru_projection_dim', 128),
        hgru_dropout=config.get('hgru_dropout', 0.1),
        # Simple CNN args
        scnn_embedding_dim=config.get('scnn_embedding_dim', 256),
        scnn_conv_dim=config.get('scnn_conv_dim', 256),
        scnn_kernel_size=config.get('scnn_kernel_size', 5),
        scnn_num_conv_blocks=config.get('scnn_num_conv_blocks', 4),
        scnn_max_length=config.get('scnn_max_length', 10000),
        scnn_vocab_size=config.get('scnn_vocab_size', 50000),
        scnn_gated_attention_dim=config.get('scnn_gated_attention_dim', 128),
        scnn_projection_dim=config.get('scnn_projection_dim', 128),
        scnn_dropout=config.get('scnn_dropout', 0.1),
    )
