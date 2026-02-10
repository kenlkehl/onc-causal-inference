# cdt/models/extractor_factory.py
"""Factory function for creating feature extractors.

This module centralizes feature extractor instantiation logic that was previously
duplicated across CausalText, CausalTextForest, and PropensityOnlyModel.
"""

import logging
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn

from .cnn_extractor import CNNFeatureExtractor
from .bert_extractor import BertFeatureExtractor
from .gru_extractor import GRUFeatureExtractor
from .confounder_extractor import (
    ConfounderExtractor,
    HierarchicalConfounderExtractor,
    GRUHierarchicalConfounderExtractor
)
from .hierarchical_transformer_extractor import HierarchicalTransformerExtractor
from .gated_mil_hierarchical_extractor import GatedMILHierarchicalExtractor
from .gru_transformer_mil_extractor import GRUTransformerMILExtractor
from .gru_pool_extractor import GRUPoolExtractor
from .bert_cross_chunk_extractor import BertCrossChunkExtractor
from .llm_extractor import LLMFeatureExtractor
from ..config import normalize_feature_extractor_type


logger = logging.getLogger(__name__)


def create_feature_extractor(
    extractor_type: str,
    device: torch.device,
    # CNN-specific args
    embedding_dim: int = 128,
    kernel_sizes: List[int] = None,
    explicit_filter_concepts: Optional[Dict[str, List[str]]] = None,
    num_kmeans_filters: int = 64,
    num_random_filters: int = 0,
    cnn_dropout: float = 0.1,
    max_length: int = 2048,
    min_word_freq: int = 2,
    max_vocab_size: Optional[int] = 50000,
    projection_dim: Optional[int] = 128,
    # BERT-specific args
    bert_model_name: str = "bert-base-uncased",
    bert_max_length: int = 512,
    bert_projection_dim: Optional[int] = 128,
    bert_dropout: float = 0.1,
    bert_freeze_encoder: bool = False,
    bert_gradient_checkpointing: bool = False,
    # GRU-specific args
    gru_hidden_dim: int = 256,
    gru_num_layers: int = 2,
    gru_dropout: float = 0.1,
    gru_bidirectional: bool = True,
    gru_attention_dim: Optional[int] = None,
    gru_projection_dim: Optional[int] = 128,
    # Confounder extractor args
    confounder_num_latents: int = 4,
    confounder_explicit_texts: Optional[List[str]] = None,
    confounder_value_dim: int = 128,
    confounder_sentence_model: str = "all-MiniLM-L6-v2",
    confounder_freeze_encoder: bool = True,
    confounder_max_sentences: int = 100,
    confounder_num_heads: int = 4,
    confounder_num_iterations: int = 2,
    confounder_use_self_attention: bool = True,
    confounder_sparse_attention: bool = True,
    confounder_sparse_method: str = "entmax",
    confounder_sparse_alpha: float = 1.5,
    confounder_top_k: int = 5,
    confounder_dropout: float = 0.1,
    # Hierarchical confounder args (token-level attention)
    confounder_hierarchical: bool = False,
    confounder_token_encoder: str = "distilbert-base-uncased",
    confounder_freeze_token_encoder: bool = True,
    confounder_max_sentence_tokens: int = 128,
    # GRU-based hierarchical confounder args (learns from scratch)
    confounder_use_gru: bool = False,
    confounder_gru_embedding_dim: int = 128,
    confounder_gru_hidden_dim: int = 128,
    confounder_gru_num_layers: int = 1,
    confounder_gru_bidirectional: bool = True,
    confounder_gru_dropout: float = 0.1,
    confounder_gru_max_vocab: int = 50000,
    confounder_gru_min_word_freq: int = 2,
    confounder_gru_max_sentence_length: int = 128,
    # Hierarchical Transformer args
    hier_transformer_sentence_model: str = "prajjwal1/bert-tiny",
    hier_transformer_freeze_sentence_encoder: bool = True,
    hier_transformer_max_chunks: int = 100,
    hier_transformer_chunk_size: int = 128,
    hier_transformer_chunk_overlap: int = 32,
    hier_transformer_num_layers: int = 2,
    hier_transformer_num_heads: int = 4,
    hier_transformer_dim: int = 256,
    hier_transformer_dropout: float = 0.1,
    hier_transformer_projection_dim: int = 128,
    # Gated MIL Hierarchical args
    gated_mil_sentence_model: str = "prajjwal1/bert-tiny",
    gated_mil_freeze_sentence_encoder: bool = True,
    gated_mil_max_chunks: int = 100,
    gated_mil_chunk_size: int = 128,
    gated_mil_chunk_overlap: int = 32,
    gated_mil_hidden_dim: int = 128,
    gated_mil_num_confounders: int = 4,
    gated_mil_dropout: float = 0.1,
    gated_mil_projection_dim: int = 128,
    gated_mil_hierarchical: bool = False,
    gated_mil_token_hidden_dim: int = 64,
    gated_mil_use_mean_pooling: bool = False,
    # GRU-Transformer-MIL args
    gru_mil_embedding_dim: int = 128,
    gru_mil_gru_hidden_dim: int = 128,
    gru_mil_gru_num_layers: int = 1,
    gru_mil_gru_bidirectional: bool = True,
    gru_mil_gru_dropout: float = 0.1,
    gru_mil_max_chunks: int = 100,
    gru_mil_chunk_size: int = 128,
    gru_mil_chunk_overlap: int = 32,
    gru_mil_transformer_layers: int = 2,
    gru_mil_transformer_heads: int = 4,
    gru_mil_transformer_dim: int = 256,
    gru_mil_num_confounders: int = 4,
    gru_mil_mil_hidden_dim: int = 128,
    gru_mil_projection_dim: int = 128,
    gru_mil_max_vocab: int = 50000,
    gru_mil_min_word_freq: int = 2,
    # GRU-Pool args
    gru_pool_embedding_dim: int = 128,
    gru_pool_gru_hidden_dim: int = 128,
    gru_pool_gru_num_layers: int = 1,
    gru_pool_gru_bidirectional: bool = True,
    gru_pool_gru_dropout: float = 0.1,
    gru_pool_max_chunks: int = 100,
    gru_pool_chunk_size: int = 128,
    gru_pool_chunk_overlap: int = 32,
    gru_pool_transformer_layers: int = 2,
    gru_pool_transformer_heads: int = 4,
    gru_pool_transformer_dim: int = 256,
    gru_pool_gated_attention_dim: int = 128,
    gru_pool_projection_dim: int = 128,
    gru_pool_max_vocab: int = 50000,
    gru_pool_min_word_freq: int = 2,
    # BERT Cross-Chunk args
    bcc_sentence_model: str = "prajjwal1/bert-tiny",
    bcc_freeze_sentence_encoder: bool = False,
    bcc_max_chunks: int = 100,
    bcc_chunk_size: int = 128,
    bcc_chunk_overlap: int = 32,
    bcc_num_cross_layers: int = 2,
    bcc_num_attention_heads: int = 4,
    bcc_cross_chunk_dim: int = 256,
    bcc_cross_chunk_dropout: float = 0.1,
    bcc_gated_attention_dim: int = 128,
    bcc_projection_dim: int = 128,
    # LLM Feature Extractor args
    llm_model_name: str = "Qwen/Qwen3-0.6B-Base",
    llm_max_length: int = 8192,
    llm_projection_dim: Optional[int] = 128,
    llm_dropout: float = 0.1,
    llm_gradient_checkpointing: bool = True,
    llm_use_pretrained: bool = False,
    # Numeric feature args
    numeric_features_enabled: bool = False,
    numeric_embedding_dim: int = 32,
    numeric_magnitude_bins: int = 8,
    numeric_type_categories: int = 10,
    # Model type (needed for task-specific extractors like gated_mil_hierarchical)
    model_type: str = "dragonnet",
) -> nn.Module:
    """
    Create a feature extractor based on the specified type.

    This factory function centralizes the instantiation logic for all supported
    feature extractors, reducing code duplication across CausalText,
    CausalTextForest, and PropensityOnlyModel.

    Args:
        extractor_type: Type of feature extractor to create. Supported types:
            - "cnn": CNN feature extractor (default)
            - "bert": BERT feature extractor
            - "gru": GRU feature extractor
            - "confounder": Confounder extractor (with hierarchical/GRU variants)
            - "hierarchical_transformer": Hierarchical transformer extractor
            - "gated_mil_hierarchical": Gated MIL hierarchical extractor
            - "gru_transformer_mil": GRU-Transformer-MIL extractor
            - "gru_pool": GRU-Pool extractor
            - "llm": LLM feature extractor (decoder-only with random init)
        device: PyTorch device to use
        ... (extractor-specific args)
        model_type: Model type for task-specific extractors ("dragonnet", "rlearner", etc.)

    Returns:
        nn.Module: The instantiated feature extractor
    """
    if kernel_sizes is None:
        kernel_sizes = [3, 4, 5, 7]

    # Normalize the extractor type
    normalized_type = normalize_feature_extractor_type(extractor_type)

    if normalized_type == "bert":
        extractor = BertFeatureExtractor(
            model_name=bert_model_name,
            projection_dim=bert_projection_dim,
            max_length=bert_max_length,
            dropout=bert_dropout,
            freeze_encoder=bert_freeze_encoder,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
            device=device
        )
        if bert_gradient_checkpointing and hasattr(extractor, 'gradient_checkpointing_enable'):
            extractor.gradient_checkpointing_enable()
        logger.info(f"Created BERT feature extractor: {bert_model_name}")
        return extractor

    elif normalized_type == "gru":
        extractor = GRUFeatureExtractor(
            embedding_dim=embedding_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
            bidirectional=gru_bidirectional,
            attention_dim=gru_attention_dim,
            projection_dim=gru_projection_dim,
            max_length=max_length,
            min_word_freq=min_word_freq,
            max_vocab_size=max_vocab_size,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
            device=device
        )
        logger.info(f"Created GRU feature extractor: {gru_num_layers} layers, "
                   f"hidden_dim={gru_hidden_dim}, bidirectional={gru_bidirectional}")
        return extractor

    elif normalized_type == "confounder":
        if confounder_use_gru:
            # GRU-based hierarchical extractor (learns from scratch)
            extractor = GRUHierarchicalConfounderExtractor(
                vocab_size=confounder_gru_max_vocab,
                embedding_dim=confounder_gru_embedding_dim,
                min_word_freq=confounder_gru_min_word_freq,
                max_sentence_length=confounder_gru_max_sentence_length,
                gru_hidden_dim=confounder_gru_hidden_dim,
                gru_num_layers=confounder_gru_num_layers,
                gru_bidirectional=confounder_gru_bidirectional,
                gru_dropout=confounder_gru_dropout,
                num_latent_confounders=confounder_num_latents,
                num_attention_heads=confounder_num_heads,
                sparse_attention=confounder_sparse_attention,
                sparse_alpha=confounder_sparse_alpha,
                sparse_method=confounder_sparse_method,
                top_k=confounder_top_k,
                max_sentences=confounder_max_sentences,
                value_dim=confounder_value_dim,
                dropout=confounder_dropout,
                model_type=model_type,
                numeric_features_enabled=numeric_features_enabled,
                numeric_embedding_dim=numeric_embedding_dim,
                numeric_magnitude_bins=numeric_magnitude_bins,
                numeric_type_categories=numeric_type_categories,
                device=device
            )
            logger.info(f"Created GRU Hierarchical Confounder extractor: {confounder_num_latents} latents, "
                       f"GRU hidden_dim={confounder_gru_hidden_dim}, sparse={confounder_sparse_attention}")
        elif confounder_hierarchical:
            # Hierarchical extractor with token-level attention (BERT-based)
            extractor = HierarchicalConfounderExtractor(
                num_latent_confounders=confounder_num_latents,
                explicit_confounder_texts=confounder_explicit_texts,
                value_dim=confounder_value_dim,
                token_encoder_model=confounder_token_encoder,
                freeze_token_encoder=confounder_freeze_token_encoder,
                max_sentences=confounder_max_sentences,
                max_sentence_tokens=confounder_max_sentence_tokens,
                num_attention_heads=confounder_num_heads,
                sparse_attention=confounder_sparse_attention,
                sparse_method=confounder_sparse_method,
                sparse_alpha=confounder_sparse_alpha,
                top_k=confounder_top_k,
                dropout=confounder_dropout,
                model_type=model_type,
                numeric_features_enabled=numeric_features_enabled,
                numeric_embedding_dim=numeric_embedding_dim,
                numeric_magnitude_bins=numeric_magnitude_bins,
                numeric_type_categories=numeric_type_categories,
                device=device
            )
            logger.info(f"Created Hierarchical Confounder extractor: {confounder_num_latents} latents, "
                       f"token_encoder={confounder_token_encoder}, sparse={confounder_sparse_attention}")
        else:
            # Standard sentence-level extractor
            extractor = ConfounderExtractor(
                num_latent_confounders=confounder_num_latents,
                explicit_confounder_texts=confounder_explicit_texts,
                value_dim=confounder_value_dim,
                sentence_transformer_model=confounder_sentence_model,
                freeze_sentence_encoder=confounder_freeze_encoder,
                max_sentences=confounder_max_sentences,
                num_attention_heads=confounder_num_heads,
                num_iterations=confounder_num_iterations,
                use_self_attention=confounder_use_self_attention,
                sparse_attention=confounder_sparse_attention,
                sparse_method=confounder_sparse_method,
                sparse_alpha=confounder_sparse_alpha,
                top_k=confounder_top_k,
                dropout=confounder_dropout,
                numeric_features_enabled=numeric_features_enabled,
                numeric_embedding_dim=numeric_embedding_dim,
                numeric_magnitude_bins=numeric_magnitude_bins,
                numeric_type_categories=numeric_type_categories,
                device=device
            )
            logger.info(f"Created Confounder extractor: {confounder_num_latents} latents, "
                       f"{confounder_num_iterations} iterations, sparse={confounder_sparse_attention}")
        return extractor

    elif normalized_type == "hierarchical_transformer":
        extractor = HierarchicalTransformerExtractor(
            sentence_encoder_model=hier_transformer_sentence_model,
            freeze_sentence_encoder=hier_transformer_freeze_sentence_encoder,
            max_chunks=hier_transformer_max_chunks,
            chunk_size=hier_transformer_chunk_size,
            chunk_overlap=hier_transformer_chunk_overlap,
            num_transformer_layers=hier_transformer_num_layers,
            num_attention_heads=hier_transformer_num_heads,
            transformer_dim=hier_transformer_dim,
            transformer_dropout=hier_transformer_dropout,
            projection_dim=hier_transformer_projection_dim,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
            device=device
        )
        logger.info(f"Created Hierarchical Transformer extractor: {hier_transformer_sentence_model}, "
                   f"{hier_transformer_num_layers} layers, chunk_size={hier_transformer_chunk_size}, "
                   f"projection_dim={hier_transformer_projection_dim}")
        return extractor

    elif normalized_type == "gated_mil_hierarchical":
        extractor = GatedMILHierarchicalExtractor(
            sentence_encoder_model=gated_mil_sentence_model,
            freeze_sentence_encoder=gated_mil_freeze_sentence_encoder,
            max_chunks=gated_mil_max_chunks,
            chunk_size=gated_mil_chunk_size,
            chunk_overlap=gated_mil_chunk_overlap,
            mil_hidden_dim=gated_mil_hidden_dim,
            num_confounders=gated_mil_num_confounders,
            model_type=model_type,
            projection_dim=gated_mil_projection_dim,
            dropout=gated_mil_dropout,
            hierarchical=gated_mil_hierarchical,
            token_hidden_dim=gated_mil_token_hidden_dim,
            use_mean_pooling=gated_mil_use_mean_pooling,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
            device=device
        )
        logger.info(f"Created Gated MIL Hierarchical extractor: {gated_mil_sentence_model}, "
                   f"{gated_mil_num_confounders} confounders, projection_dim={gated_mil_projection_dim}, "
                   f"hierarchical={gated_mil_hierarchical}, mean_pooling={gated_mil_use_mean_pooling}")
        return extractor

    elif normalized_type == "gru_transformer_mil":
        extractor = GRUTransformerMILExtractor(
            embedding_dim=gru_mil_embedding_dim,
            gru_hidden_dim=gru_mil_gru_hidden_dim,
            gru_num_layers=gru_mil_gru_num_layers,
            gru_bidirectional=gru_mil_gru_bidirectional,
            gru_dropout=gru_mil_gru_dropout,
            max_chunks=gru_mil_max_chunks,
            chunk_size=gru_mil_chunk_size,
            chunk_overlap=gru_mil_chunk_overlap,
            transformer_layers=gru_mil_transformer_layers,
            transformer_heads=gru_mil_transformer_heads,
            transformer_dim=gru_mil_transformer_dim,
            num_confounders=gru_mil_num_confounders,
            mil_hidden_dim=gru_mil_mil_hidden_dim,
            projection_dim=gru_mil_projection_dim,
            max_vocab_size=gru_mil_max_vocab,
            min_word_freq=gru_mil_min_word_freq,
            model_type=model_type,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
            device=device
        )
        logger.info(f"Created GRU-Transformer-MIL extractor: "
                   f"GRU {gru_mil_gru_hidden_dim}x{2 if gru_mil_gru_bidirectional else 1}, "
                   f"{gru_mil_transformer_layers} transformer layers, "
                   f"{gru_mil_num_confounders} confounders, projection_dim={gru_mil_projection_dim}")
        return extractor

    elif normalized_type == "gru_pool":
        extractor = GRUPoolExtractor(
            embedding_dim=gru_pool_embedding_dim,
            gru_hidden_dim=gru_pool_gru_hidden_dim,
            gru_num_layers=gru_pool_gru_num_layers,
            gru_bidirectional=gru_pool_gru_bidirectional,
            gru_dropout=gru_pool_gru_dropout,
            max_chunks=gru_pool_max_chunks,
            chunk_size=gru_pool_chunk_size,
            chunk_overlap=gru_pool_chunk_overlap,
            transformer_layers=gru_pool_transformer_layers,
            transformer_heads=gru_pool_transformer_heads,
            transformer_dim=gru_pool_transformer_dim,
            gated_attention_dim=gru_pool_gated_attention_dim,
            projection_dim=gru_pool_projection_dim,
            max_vocab_size=gru_pool_max_vocab,
            min_word_freq=gru_pool_min_word_freq,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
            device=device
        )
        logger.info(f"Created GRU-Pool extractor: "
                   f"GRU {gru_pool_gru_hidden_dim}x{2 if gru_pool_gru_bidirectional else 1}, "
                   f"{gru_pool_transformer_layers} transformer layers, "
                   f"gated_attention_dim={gru_pool_gated_attention_dim}, projection_dim={gru_pool_projection_dim}")
        return extractor

    elif normalized_type == "bert_cross_chunk":
        extractor = BertCrossChunkExtractor(
            sentence_encoder_model=bcc_sentence_model,
            freeze_sentence_encoder=bcc_freeze_sentence_encoder,
            max_chunks=bcc_max_chunks,
            chunk_size=bcc_chunk_size,
            chunk_overlap=bcc_chunk_overlap,
            num_cross_layers=bcc_num_cross_layers,
            num_attention_heads=bcc_num_attention_heads,
            cross_chunk_dim=bcc_cross_chunk_dim,
            cross_chunk_dropout=bcc_cross_chunk_dropout,
            gated_attention_dim=bcc_gated_attention_dim,
            projection_dim=bcc_projection_dim,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
            device=device
        )
        logger.info(f"Created BERT Cross-Chunk extractor: {bcc_sentence_model}, "
                   f"{bcc_num_cross_layers} cross-chunk layers, chunk_size={bcc_chunk_size}, "
                   f"cross_chunk_dim={bcc_cross_chunk_dim}, projection_dim={bcc_projection_dim}")
        return extractor

    elif normalized_type == "llm":
        extractor = LLMFeatureExtractor(
            model_name=llm_model_name,
            max_length=llm_max_length,
            projection_dim=llm_projection_dim,
            dropout=llm_dropout,
            gradient_checkpointing=llm_gradient_checkpointing,
            use_pretrained=llm_use_pretrained,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
            device=device
        )
        init_mode = "pretrained" if llm_use_pretrained else "random init"
        logger.info(f"Created LLM feature extractor: {llm_model_name} ({init_mode}), "
                   f"max_length={llm_max_length}, projection_dim={llm_projection_dim}")
        return extractor

    else:
        # CNN feature extractor (default)
        extractor = CNNFeatureExtractor(
            embedding_dim=embedding_dim,
            kernel_sizes=kernel_sizes,
            explicit_filter_concepts=explicit_filter_concepts,
            num_kmeans_filters=num_kmeans_filters,
            num_random_filters=num_random_filters,
            projection_dim=projection_dim,
            dropout=cnn_dropout,
            max_length=max_length,
            min_word_freq=min_word_freq,
            max_vocab_size=max_vocab_size,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
            device=device
        )
        logger.info("Created CNN feature extractor")
        return extractor


def create_feature_extractor_from_config(
    config: Dict[str, Any],
    device: torch.device,
    model_type: str = "dragonnet"
) -> nn.Module:
    """
    Create a feature extractor from a configuration dictionary.

    This is a convenience function that extracts the relevant parameters
    from a config dict and calls create_feature_extractor.

    Args:
        config: Configuration dictionary (typically from CausalText.config)
        device: PyTorch device
        model_type: Model type for task-specific extractors

    Returns:
        nn.Module: The instantiated feature extractor
    """
    return create_feature_extractor(
        extractor_type=config.get('feature_extractor_type', 'cnn'),
        device=device,
        model_type=model_type,
        # CNN args
        embedding_dim=config.get('embedding_dim', 128),
        kernel_sizes=config.get('kernel_sizes', [3, 4, 5, 7]),
        explicit_filter_concepts=config.get('explicit_filter_concepts'),
        num_kmeans_filters=config.get('num_kmeans_filters', 64),
        num_random_filters=config.get('num_random_filters', 0),
        cnn_dropout=config.get('cnn_dropout', 0.1),
        max_length=config.get('max_length', 2048),
        min_word_freq=config.get('min_word_freq', 2),
        max_vocab_size=config.get('max_vocab_size', 50000),
        projection_dim=config.get('projection_dim', 128),
        # BERT args
        bert_model_name=config.get('bert_model_name', 'bert-base-uncased'),
        bert_max_length=config.get('bert_max_length', 512),
        bert_projection_dim=config.get('bert_projection_dim', 128),
        bert_dropout=config.get('bert_dropout', 0.1),
        bert_freeze_encoder=config.get('bert_freeze_encoder', False),
        bert_gradient_checkpointing=config.get('bert_gradient_checkpointing', False),
        # GRU args
        gru_hidden_dim=config.get('gru_hidden_dim', 256),
        gru_num_layers=config.get('gru_num_layers', 2),
        gru_dropout=config.get('gru_dropout', 0.1),
        gru_bidirectional=config.get('gru_bidirectional', True),
        gru_attention_dim=config.get('gru_attention_dim'),
        gru_projection_dim=config.get('gru_projection_dim', 128),
        # Confounder args
        confounder_num_latents=config.get('confounder_num_latents', 4),
        confounder_explicit_texts=config.get('confounder_explicit_texts'),
        confounder_value_dim=config.get('confounder_value_dim', 128),
        confounder_sentence_model=config.get('confounder_sentence_model', 'all-MiniLM-L6-v2'),
        confounder_freeze_encoder=config.get('confounder_freeze_encoder', True),
        confounder_max_sentences=config.get('confounder_max_sentences', 100),
        confounder_num_heads=config.get('confounder_num_heads', 4),
        confounder_num_iterations=config.get('confounder_num_iterations', 2),
        confounder_use_self_attention=config.get('confounder_use_self_attention', True),
        confounder_sparse_attention=config.get('confounder_sparse_attention', True),
        confounder_sparse_method=config.get('confounder_sparse_method', 'entmax'),
        confounder_sparse_alpha=config.get('confounder_sparse_alpha', 1.5),
        confounder_top_k=config.get('confounder_top_k', 5),
        confounder_dropout=config.get('confounder_dropout', 0.1),
        confounder_hierarchical=config.get('confounder_hierarchical', False),
        confounder_token_encoder=config.get('confounder_token_encoder', 'distilbert-base-uncased'),
        confounder_freeze_token_encoder=config.get('confounder_freeze_token_encoder', True),
        confounder_max_sentence_tokens=config.get('confounder_max_sentence_tokens', 128),
        confounder_use_gru=config.get('confounder_use_gru', False),
        confounder_gru_embedding_dim=config.get('confounder_gru_embedding_dim', 128),
        confounder_gru_hidden_dim=config.get('confounder_gru_hidden_dim', 128),
        confounder_gru_num_layers=config.get('confounder_gru_num_layers', 1),
        confounder_gru_bidirectional=config.get('confounder_gru_bidirectional', True),
        confounder_gru_dropout=config.get('confounder_gru_dropout', 0.1),
        confounder_gru_max_vocab=config.get('confounder_gru_max_vocab', 50000),
        confounder_gru_min_word_freq=config.get('confounder_gru_min_word_freq', 2),
        confounder_gru_max_sentence_length=config.get('confounder_gru_max_sentence_length', 128),
        # Hierarchical Transformer args
        hier_transformer_sentence_model=config.get('hier_transformer_sentence_model', 'prajjwal1/bert-tiny'),
        hier_transformer_freeze_sentence_encoder=config.get('hier_transformer_freeze_sentence_encoder', True),
        hier_transformer_max_chunks=config.get('hier_transformer_max_chunks', 100),
        hier_transformer_chunk_size=config.get('hier_transformer_chunk_size', 128),
        hier_transformer_chunk_overlap=config.get('hier_transformer_chunk_overlap', 32),
        hier_transformer_num_layers=config.get('hier_transformer_num_layers', 2),
        hier_transformer_num_heads=config.get('hier_transformer_num_heads', 4),
        hier_transformer_dim=config.get('hier_transformer_dim', 256),
        hier_transformer_dropout=config.get('hier_transformer_dropout', 0.1),
        hier_transformer_projection_dim=config.get('hier_transformer_projection_dim', 128),
        # Gated MIL args
        gated_mil_sentence_model=config.get('gated_mil_sentence_model', 'prajjwal1/bert-tiny'),
        gated_mil_freeze_sentence_encoder=config.get('gated_mil_freeze_sentence_encoder', True),
        gated_mil_max_chunks=config.get('gated_mil_max_chunks', 100),
        gated_mil_chunk_size=config.get('gated_mil_chunk_size', 128),
        gated_mil_chunk_overlap=config.get('gated_mil_chunk_overlap', 32),
        gated_mil_hidden_dim=config.get('gated_mil_hidden_dim', 128),
        gated_mil_num_confounders=config.get('gated_mil_num_confounders', 4),
        gated_mil_dropout=config.get('gated_mil_dropout', 0.1),
        gated_mil_projection_dim=config.get('gated_mil_projection_dim', 128),
        gated_mil_hierarchical=config.get('gated_mil_hierarchical', False),
        gated_mil_token_hidden_dim=config.get('gated_mil_token_hidden_dim', 64),
        gated_mil_use_mean_pooling=config.get('gated_mil_use_mean_pooling', False),
        # GRU-Transformer-MIL args
        gru_mil_embedding_dim=config.get('gru_mil_embedding_dim', 128),
        gru_mil_gru_hidden_dim=config.get('gru_mil_gru_hidden_dim', 128),
        gru_mil_gru_num_layers=config.get('gru_mil_gru_num_layers', 1),
        gru_mil_gru_bidirectional=config.get('gru_mil_gru_bidirectional', True),
        gru_mil_gru_dropout=config.get('gru_mil_gru_dropout', 0.1),
        gru_mil_max_chunks=config.get('gru_mil_max_chunks', 100),
        gru_mil_chunk_size=config.get('gru_mil_chunk_size', 128),
        gru_mil_chunk_overlap=config.get('gru_mil_chunk_overlap', 32),
        gru_mil_transformer_layers=config.get('gru_mil_transformer_layers', 2),
        gru_mil_transformer_heads=config.get('gru_mil_transformer_heads', 4),
        gru_mil_transformer_dim=config.get('gru_mil_transformer_dim', 256),
        gru_mil_num_confounders=config.get('gru_mil_num_confounders', 4),
        gru_mil_mil_hidden_dim=config.get('gru_mil_mil_hidden_dim', 128),
        gru_mil_projection_dim=config.get('gru_mil_projection_dim', 128),
        gru_mil_max_vocab=config.get('gru_mil_max_vocab', 50000),
        gru_mil_min_word_freq=config.get('gru_mil_min_word_freq', 2),
        # GRU-Pool args
        gru_pool_embedding_dim=config.get('gru_pool_embedding_dim', 128),
        gru_pool_gru_hidden_dim=config.get('gru_pool_gru_hidden_dim', 128),
        gru_pool_gru_num_layers=config.get('gru_pool_gru_num_layers', 1),
        gru_pool_gru_bidirectional=config.get('gru_pool_gru_bidirectional', True),
        gru_pool_gru_dropout=config.get('gru_pool_gru_dropout', 0.1),
        gru_pool_max_chunks=config.get('gru_pool_max_chunks', 100),
        gru_pool_chunk_size=config.get('gru_pool_chunk_size', 128),
        gru_pool_chunk_overlap=config.get('gru_pool_chunk_overlap', 32),
        gru_pool_transformer_layers=config.get('gru_pool_transformer_layers', 2),
        gru_pool_transformer_heads=config.get('gru_pool_transformer_heads', 4),
        gru_pool_transformer_dim=config.get('gru_pool_transformer_dim', 256),
        gru_pool_gated_attention_dim=config.get('gru_pool_gated_attention_dim', 128),
        gru_pool_projection_dim=config.get('gru_pool_projection_dim', 128),
        gru_pool_max_vocab=config.get('gru_pool_max_vocab', 50000),
        gru_pool_min_word_freq=config.get('gru_pool_min_word_freq', 2),
        # BERT Cross-Chunk args
        bcc_sentence_model=config.get('bcc_sentence_model', 'prajjwal1/bert-tiny'),
        bcc_freeze_sentence_encoder=config.get('bcc_freeze_sentence_encoder', False),
        bcc_max_chunks=config.get('bcc_max_chunks', 100),
        bcc_chunk_size=config.get('bcc_chunk_size', 128),
        bcc_chunk_overlap=config.get('bcc_chunk_overlap', 32),
        bcc_num_cross_layers=config.get('bcc_num_cross_layers', 2),
        bcc_num_attention_heads=config.get('bcc_num_attention_heads', 4),
        bcc_cross_chunk_dim=config.get('bcc_cross_chunk_dim', 256),
        bcc_cross_chunk_dropout=config.get('bcc_cross_chunk_dropout', 0.1),
        bcc_gated_attention_dim=config.get('bcc_gated_attention_dim', 128),
        bcc_projection_dim=config.get('bcc_projection_dim', 128),
        # LLM args
        llm_model_name=config.get('llm_model_name', 'Qwen/Qwen3-0.6B-Base'),
        llm_max_length=config.get('llm_max_length', 8192),
        llm_projection_dim=config.get('llm_projection_dim', 128),
        llm_dropout=config.get('llm_dropout', 0.1),
        llm_gradient_checkpointing=config.get('llm_gradient_checkpointing', True),
        llm_use_pretrained=config.get('llm_use_pretrained', False),
        # Numeric args
        numeric_features_enabled=config.get('numeric_features_enabled', False),
        numeric_embedding_dim=config.get('numeric_embedding_dim', 32),
        numeric_magnitude_bins=config.get('numeric_magnitude_bins', 8),
        numeric_type_categories=config.get('numeric_type_categories', 10),
    )
