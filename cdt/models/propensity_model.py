# cdt/models/propensity_model.py
"""Propensity-only model for dataset trimming before causal inference."""

import gc
import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractor_factory import create_feature_extractor
from ..config import normalize_feature_extractor_type


logger = logging.getLogger(__name__)


class PropensityNet(nn.Module):
    """
    Propensity prediction network with same representation as DragonNet.

    Uses 6-layer representation followed by a single propensity head.
    """

    def __init__(self, input_dim: int, representation_dim: int = 200):
        super().__init__()

        # Shared representation layers (same as DragonNet)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)

        # Single propensity head (same as DragonNet)
        self.propensity_fc1 = nn.Linear(representation_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the propensity network.

        Args:
            features: Feature tensor from feature extractor (batch, input_dim)

        Returns:
            t_logit: Propensity logits (batch, 1)
        """
        h = F.relu(self.representation_fc1(features))
        h = F.elu(self.representation_fc2(h))

        t_logit = self.propensity_fc1(h)

        return t_logit


class PropensityOnlyModel(nn.Module):
    """
    Propensity-score-only model for dataset trimming.

    Uses same architecture as CausalText/DragonNet:
    - Feature extractor (CNN or BERT)
    - 6-layer representation network
    - Single propensity head

    This model is trained to predict P(T=1|X) using binary cross-entropy loss.
    Used for generating propensity scores for trimming before DragonNet training.
    """

    def __init__(
        self,
        # Feature extractor type
        feature_extractor_type: str = "cnn",
        # CNN-specific args
        embedding_dim: int = 128,
        kernel_sizes: List[int] = [3, 4, 5, 7],
        explicit_filter_concepts: Optional[Dict[str, List[str]]] = None,
        num_kmeans_filters: int = 0,
        num_random_filters: int = 256,
        cnn_dropout: float = 0.0,
        max_length: int = 8192,
        min_word_freq: int = 2,
        max_vocab_size: Optional[int] = 20000,
        projection_dim: Optional[int] = 128,
        # BERT-specific args
        bert_model_name: str = "bert-base-uncased",
        bert_max_length: int = 512,
        bert_projection_dim: Optional[int] = 128,
        bert_dropout: float = 0.1,
        bert_freeze_encoder: bool = False,
        bert_gradient_checkpointing: bool = False,
        # GRU-specific args
        gru_embedding_dim: int = 256,
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.1,
        gru_bidirectional: bool = True,
        gru_attention_dim: Optional[int] = None,
        gru_projection_dim: Optional[int] = 128,
        gru_max_length: int = 8192,
        gru_min_word_freq: int = 2,
        gru_max_vocab_size: Optional[int] = 50000,
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
        # Conv-Pool args
        conv_pool_embedding_dim: int = 128,
        conv_pool_conv_dim: int = 256,
        conv_pool_kernel_size: int = 3,
        conv_pool_num_blocks: int = 4,
        conv_pool_dropout: float = 0.1,
        conv_pool_max_chunks: int = 100,
        conv_pool_chunk_size: int = 128,
        conv_pool_chunk_overlap: int = 32,
        conv_pool_transformer_layers: int = 2,
        conv_pool_transformer_heads: int = 4,
        conv_pool_transformer_dim: int = 256,
        conv_pool_transformer_dropout: float = 0.1,
        conv_pool_gated_attention_dim: int = 128,
        conv_pool_projection_dim: int = 128,
        conv_pool_max_vocab: int = 50000,
        conv_pool_min_word_freq: int = 2,
        # Conv1d-Transformer Hybrid args
        c1d_hybrid_embedding_dim: int = 128,
        c1d_hybrid_conv_dim: int = 256,
        c1d_hybrid_kernel_size: int = 3,
        c1d_hybrid_num_blocks: int = 4,
        c1d_hybrid_conv_dropout: float = 0.1,
        c1d_hybrid_pool_stride: int = 2,
        c1d_hybrid_max_length: int = 8192,
        c1d_hybrid_transformer_layers: int = 2,
        c1d_hybrid_transformer_heads: int = 4,
        c1d_hybrid_transformer_dim: int = 256,
        c1d_hybrid_transformer_dropout: float = 0.1,
        c1d_hybrid_gated_attention_dim: int = 128,
        c1d_hybrid_projection_dim: int = 128,
        c1d_hybrid_max_vocab: int = 50000,
        c1d_hybrid_min_word_freq: int = 2,
        # Transformer Pool args (learned tokenizer + token transformer + chunk transformer + gated pooling)
        tp_embedding_dim: int = 128,
        tp_token_transformer_layers: int = 2,
        tp_token_transformer_heads: int = 4,
        tp_token_transformer_dim: int = 256,
        tp_token_transformer_dropout: float = 0.1,
        tp_chunk_transformer_layers: int = 2,
        tp_chunk_transformer_heads: int = 4,
        tp_chunk_transformer_dim: int = 256,
        tp_chunk_transformer_dropout: float = 0.1,
        tp_gated_attention_dim: int = 128,
        tp_projection_dim: int = 128,
        tp_chunk_size: int = 128,
        tp_chunk_overlap: int = 32,
        tp_max_chunks: int = 100,
        tp_max_vocab: int = 50000,
        tp_min_word_freq: int = 2,
        # BERT Pool args
        bert_pool_sentence_model: str = "prajjwal1/bert-tiny",
        bert_pool_freeze_sentence_encoder: bool = False,
        bert_pool_use_pretrained: bool = True,
        bert_pool_max_chunks: int = 100,
        bert_pool_chunk_size: int = 128,
        bert_pool_chunk_overlap: int = 32,
        bert_pool_transformer_layers: int = 2,
        bert_pool_transformer_heads: int = 4,
        bert_pool_transformer_dim: int = 256,
        bert_pool_transformer_dropout: float = 0.1,
        bert_pool_gated_attention_dim: int = 128,
        bert_pool_projection_dim: int = 128,
        # LLM args
        llm_model_name: str = "Qwen/Qwen3-0.6B-Base",
        llm_max_length: int = 8192,
        llm_projection_dim: Optional[int] = 128,
        llm_dropout: float = 0.1,
        llm_gradient_checkpointing: bool = True,
        llm_use_pretrained: bool = False,
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
        # Propensity network args
        representation_dim: int = 128,
        device: str = "cuda:0"
    ):
        """
        Initialize propensity-only model.

        Args:
            feature_extractor_type: "cnn" or "bert"
            embedding_dim: (CNN) Dimension of word embeddings
            kernel_sizes: (CNN) List of kernel sizes for n-gram capture
            explicit_filter_concepts: (CNN) Dict mapping kernel_size to concept phrases
            num_kmeans_filters: (CNN) Number of k-means derived filters per kernel size
            num_random_filters: (CNN) Number of randomly initialized filters per kernel size
            cnn_dropout: (CNN) Dropout rate
            max_length: (CNN) Maximum sequence length in tokens
            min_word_freq: (CNN) Minimum word frequency for vocabulary inclusion
            max_vocab_size: (CNN) Maximum vocabulary size
            projection_dim: (CNN) Dimension to project CNN output to
            bert_model_name: (BERT) HuggingFace model name or path
            bert_max_length: (BERT) Maximum sequence length in subword tokens
            bert_projection_dim: (BERT) Projection dimension after CLS token
            bert_dropout: (BERT) Dropout rate for projection layer
            bert_freeze_encoder: (BERT) Whether to freeze transformer weights
            bert_gradient_checkpointing: (BERT) Enable gradient checkpointing
            representation_dim: Dimension of representation layers
            device: Device string
        """
        super().__init__()

        self._device = torch.device(device)
        # Normalize feature extractor type (e.g., "modernbert" -> "bert")
        self.feature_extractor_type = normalize_feature_extractor_type(feature_extractor_type)

        # Store config for checkpointing
        self.config = {
            'feature_extractor_type': feature_extractor_type,
            'embedding_dim': embedding_dim,
            'kernel_sizes': kernel_sizes,
            'explicit_filter_concepts': explicit_filter_concepts,
            'num_kmeans_filters': num_kmeans_filters,
            'num_random_filters': num_random_filters,
            'cnn_dropout': cnn_dropout,
            'max_length': max_length,
            'min_word_freq': min_word_freq,
            'max_vocab_size': max_vocab_size,
            'projection_dim': projection_dim,
            'bert_model_name': bert_model_name,
            'bert_max_length': bert_max_length,
            'bert_projection_dim': bert_projection_dim,
            'bert_dropout': bert_dropout,
            'bert_freeze_encoder': bert_freeze_encoder,
            'bert_gradient_checkpointing': bert_gradient_checkpointing,
            'gru_embedding_dim': gru_embedding_dim,
            'gru_hidden_dim': gru_hidden_dim,
            'gru_num_layers': gru_num_layers,
            'gru_dropout': gru_dropout,
            'gru_bidirectional': gru_bidirectional,
            'gru_attention_dim': gru_attention_dim,
            'gru_projection_dim': gru_projection_dim,
            'gru_max_length': gru_max_length,
            'gru_min_word_freq': gru_min_word_freq,
            'gru_max_vocab_size': gru_max_vocab_size,
            'hier_transformer_sentence_model': hier_transformer_sentence_model,
            'hier_transformer_freeze_sentence_encoder': hier_transformer_freeze_sentence_encoder,
            'hier_transformer_max_chunks': hier_transformer_max_chunks,
            'hier_transformer_chunk_size': hier_transformer_chunk_size,
            'hier_transformer_chunk_overlap': hier_transformer_chunk_overlap,
            'hier_transformer_num_layers': hier_transformer_num_layers,
            'hier_transformer_num_heads': hier_transformer_num_heads,
            'hier_transformer_dim': hier_transformer_dim,
            'hier_transformer_dropout': hier_transformer_dropout,
            'hier_transformer_projection_dim': hier_transformer_projection_dim,
            'bcc_sentence_model': bcc_sentence_model,
            'bcc_freeze_sentence_encoder': bcc_freeze_sentence_encoder,
            'bcc_max_chunks': bcc_max_chunks,
            'bcc_chunk_size': bcc_chunk_size,
            'bcc_chunk_overlap': bcc_chunk_overlap,
            'bcc_num_cross_layers': bcc_num_cross_layers,
            'bcc_num_attention_heads': bcc_num_attention_heads,
            'bcc_cross_chunk_dim': bcc_cross_chunk_dim,
            'bcc_cross_chunk_dropout': bcc_cross_chunk_dropout,
            'bcc_gated_attention_dim': bcc_gated_attention_dim,
            'bcc_projection_dim': bcc_projection_dim,
            'gru_mil_embedding_dim': gru_mil_embedding_dim,
            'gru_mil_gru_hidden_dim': gru_mil_gru_hidden_dim,
            'gru_mil_gru_num_layers': gru_mil_gru_num_layers,
            'gru_mil_gru_bidirectional': gru_mil_gru_bidirectional,
            'gru_mil_gru_dropout': gru_mil_gru_dropout,
            'gru_mil_max_chunks': gru_mil_max_chunks,
            'gru_mil_chunk_size': gru_mil_chunk_size,
            'gru_mil_chunk_overlap': gru_mil_chunk_overlap,
            'gru_mil_transformer_layers': gru_mil_transformer_layers,
            'gru_mil_transformer_heads': gru_mil_transformer_heads,
            'gru_mil_transformer_dim': gru_mil_transformer_dim,
            'gru_mil_num_confounders': gru_mil_num_confounders,
            'gru_mil_mil_hidden_dim': gru_mil_mil_hidden_dim,
            'gru_mil_projection_dim': gru_mil_projection_dim,
            'gru_mil_max_vocab': gru_mil_max_vocab,
            'gru_mil_min_word_freq': gru_mil_min_word_freq,
            'gru_pool_embedding_dim': gru_pool_embedding_dim,
            'gru_pool_gru_hidden_dim': gru_pool_gru_hidden_dim,
            'gru_pool_gru_num_layers': gru_pool_gru_num_layers,
            'gru_pool_gru_bidirectional': gru_pool_gru_bidirectional,
            'gru_pool_gru_dropout': gru_pool_gru_dropout,
            'gru_pool_max_chunks': gru_pool_max_chunks,
            'gru_pool_chunk_size': gru_pool_chunk_size,
            'gru_pool_chunk_overlap': gru_pool_chunk_overlap,
            'gru_pool_transformer_layers': gru_pool_transformer_layers,
            'gru_pool_transformer_heads': gru_pool_transformer_heads,
            'gru_pool_transformer_dim': gru_pool_transformer_dim,
            'gru_pool_gated_attention_dim': gru_pool_gated_attention_dim,
            'gru_pool_projection_dim': gru_pool_projection_dim,
            'gru_pool_max_vocab': gru_pool_max_vocab,
            'gru_pool_min_word_freq': gru_pool_min_word_freq,
            'conv_pool_embedding_dim': conv_pool_embedding_dim,
            'conv_pool_conv_dim': conv_pool_conv_dim,
            'conv_pool_kernel_size': conv_pool_kernel_size,
            'conv_pool_num_blocks': conv_pool_num_blocks,
            'conv_pool_dropout': conv_pool_dropout,
            'conv_pool_max_chunks': conv_pool_max_chunks,
            'conv_pool_chunk_size': conv_pool_chunk_size,
            'conv_pool_chunk_overlap': conv_pool_chunk_overlap,
            'conv_pool_transformer_layers': conv_pool_transformer_layers,
            'conv_pool_transformer_heads': conv_pool_transformer_heads,
            'conv_pool_transformer_dim': conv_pool_transformer_dim,
            'conv_pool_transformer_dropout': conv_pool_transformer_dropout,
            'conv_pool_gated_attention_dim': conv_pool_gated_attention_dim,
            'conv_pool_projection_dim': conv_pool_projection_dim,
            'conv_pool_max_vocab': conv_pool_max_vocab,
            'conv_pool_min_word_freq': conv_pool_min_word_freq,
            'c1d_hybrid_embedding_dim': c1d_hybrid_embedding_dim,
            'c1d_hybrid_conv_dim': c1d_hybrid_conv_dim,
            'c1d_hybrid_kernel_size': c1d_hybrid_kernel_size,
            'c1d_hybrid_num_blocks': c1d_hybrid_num_blocks,
            'c1d_hybrid_conv_dropout': c1d_hybrid_conv_dropout,
            'c1d_hybrid_pool_stride': c1d_hybrid_pool_stride,
            'c1d_hybrid_max_length': c1d_hybrid_max_length,
            'c1d_hybrid_transformer_layers': c1d_hybrid_transformer_layers,
            'c1d_hybrid_transformer_heads': c1d_hybrid_transformer_heads,
            'c1d_hybrid_transformer_dim': c1d_hybrid_transformer_dim,
            'c1d_hybrid_transformer_dropout': c1d_hybrid_transformer_dropout,
            'c1d_hybrid_gated_attention_dim': c1d_hybrid_gated_attention_dim,
            'c1d_hybrid_projection_dim': c1d_hybrid_projection_dim,
            'c1d_hybrid_max_vocab': c1d_hybrid_max_vocab,
            'c1d_hybrid_min_word_freq': c1d_hybrid_min_word_freq,
            'tp_embedding_dim': tp_embedding_dim,
            'tp_token_transformer_layers': tp_token_transformer_layers,
            'tp_token_transformer_heads': tp_token_transformer_heads,
            'tp_token_transformer_dim': tp_token_transformer_dim,
            'tp_token_transformer_dropout': tp_token_transformer_dropout,
            'tp_chunk_transformer_layers': tp_chunk_transformer_layers,
            'tp_chunk_transformer_heads': tp_chunk_transformer_heads,
            'tp_chunk_transformer_dim': tp_chunk_transformer_dim,
            'tp_chunk_transformer_dropout': tp_chunk_transformer_dropout,
            'tp_gated_attention_dim': tp_gated_attention_dim,
            'tp_projection_dim': tp_projection_dim,
            'tp_chunk_size': tp_chunk_size,
            'tp_chunk_overlap': tp_chunk_overlap,
            'tp_max_chunks': tp_max_chunks,
            'tp_max_vocab': tp_max_vocab,
            'tp_min_word_freq': tp_min_word_freq,
            'bert_pool_sentence_model': bert_pool_sentence_model,
            'bert_pool_freeze_sentence_encoder': bert_pool_freeze_sentence_encoder,
            'bert_pool_use_pretrained': bert_pool_use_pretrained,
            'bert_pool_max_chunks': bert_pool_max_chunks,
            'bert_pool_chunk_size': bert_pool_chunk_size,
            'bert_pool_chunk_overlap': bert_pool_chunk_overlap,
            'bert_pool_transformer_layers': bert_pool_transformer_layers,
            'bert_pool_transformer_heads': bert_pool_transformer_heads,
            'bert_pool_transformer_dim': bert_pool_transformer_dim,
            'bert_pool_transformer_dropout': bert_pool_transformer_dropout,
            'bert_pool_gated_attention_dim': bert_pool_gated_attention_dim,
            'bert_pool_projection_dim': bert_pool_projection_dim,
            'llm_model_name': llm_model_name,
            'llm_max_length': llm_max_length,
            'llm_projection_dim': llm_projection_dim,
            'llm_dropout': llm_dropout,
            'llm_gradient_checkpointing': llm_gradient_checkpointing,
            'llm_use_pretrained': llm_use_pretrained,
            'flp_model_name': flp_model_name,
            'flp_max_length': flp_max_length,
            'flp_freeze_llm': flp_freeze_llm,
            'flp_gated_attention_dim': flp_gated_attention_dim,
            'flp_projection_dim': flp_projection_dim,
            'flp_dropout': flp_dropout,
            'flp_gradient_checkpointing': flp_gradient_checkpointing,
            'flp_downprojection_dim': flp_downprojection_dim,
            'flp_skip_llm': flp_skip_llm,
            'flp_cached_hidden_size': flp_cached_hidden_size,
            'numeric_features_enabled': numeric_features_enabled,
            'numeric_embedding_dim': numeric_embedding_dim,
            'numeric_magnitude_bins': numeric_magnitude_bins,
            'numeric_type_categories': numeric_type_categories,
            'representation_dim': representation_dim
        }

        # Initialize feature extractor using factory
        self.feature_extractor = create_feature_extractor(
            extractor_type=self.feature_extractor_type,
            device=self._device,
            model_type="rlearner",  # Propensity model uses rlearner-style weighting for task-specific extractors
            embedding_dim=embedding_dim,
            kernel_sizes=kernel_sizes,
            explicit_filter_concepts=explicit_filter_concepts,
            num_kmeans_filters=num_kmeans_filters,
            num_random_filters=num_random_filters,
            cnn_dropout=cnn_dropout,
            max_length=max_length,
            min_word_freq=min_word_freq,
            max_vocab_size=max_vocab_size,
            projection_dim=projection_dim,
            bert_model_name=bert_model_name,
            bert_max_length=bert_max_length,
            bert_projection_dim=bert_projection_dim,
            bert_dropout=bert_dropout,
            bert_freeze_encoder=bert_freeze_encoder,
            bert_gradient_checkpointing=bert_gradient_checkpointing,
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            gru_dropout=gru_dropout,
            gru_bidirectional=gru_bidirectional,
            gru_attention_dim=gru_attention_dim,
            gru_projection_dim=gru_projection_dim,
            hier_transformer_sentence_model=hier_transformer_sentence_model,
            hier_transformer_freeze_sentence_encoder=hier_transformer_freeze_sentence_encoder,
            hier_transformer_max_chunks=hier_transformer_max_chunks,
            hier_transformer_chunk_size=hier_transformer_chunk_size,
            hier_transformer_chunk_overlap=hier_transformer_chunk_overlap,
            hier_transformer_num_layers=hier_transformer_num_layers,
            hier_transformer_num_heads=hier_transformer_num_heads,
            hier_transformer_dim=hier_transformer_dim,
            hier_transformer_dropout=hier_transformer_dropout,
            hier_transformer_projection_dim=hier_transformer_projection_dim,
            bcc_sentence_model=bcc_sentence_model,
            bcc_freeze_sentence_encoder=bcc_freeze_sentence_encoder,
            bcc_max_chunks=bcc_max_chunks,
            bcc_chunk_size=bcc_chunk_size,
            bcc_chunk_overlap=bcc_chunk_overlap,
            bcc_num_cross_layers=bcc_num_cross_layers,
            bcc_num_attention_heads=bcc_num_attention_heads,
            bcc_cross_chunk_dim=bcc_cross_chunk_dim,
            bcc_cross_chunk_dropout=bcc_cross_chunk_dropout,
            bcc_gated_attention_dim=bcc_gated_attention_dim,
            bcc_projection_dim=bcc_projection_dim,
            gru_mil_embedding_dim=gru_mil_embedding_dim,
            gru_mil_gru_hidden_dim=gru_mil_gru_hidden_dim,
            gru_mil_gru_num_layers=gru_mil_gru_num_layers,
            gru_mil_gru_bidirectional=gru_mil_gru_bidirectional,
            gru_mil_gru_dropout=gru_mil_gru_dropout,
            gru_mil_max_chunks=gru_mil_max_chunks,
            gru_mil_chunk_size=gru_mil_chunk_size,
            gru_mil_chunk_overlap=gru_mil_chunk_overlap,
            gru_mil_transformer_layers=gru_mil_transformer_layers,
            gru_mil_transformer_heads=gru_mil_transformer_heads,
            gru_mil_transformer_dim=gru_mil_transformer_dim,
            gru_mil_num_confounders=gru_mil_num_confounders,
            gru_mil_mil_hidden_dim=gru_mil_mil_hidden_dim,
            gru_mil_projection_dim=gru_mil_projection_dim,
            gru_mil_max_vocab=gru_mil_max_vocab,
            gru_mil_min_word_freq=gru_mil_min_word_freq,
            gru_pool_embedding_dim=gru_pool_embedding_dim,
            gru_pool_gru_hidden_dim=gru_pool_gru_hidden_dim,
            gru_pool_gru_num_layers=gru_pool_gru_num_layers,
            gru_pool_gru_bidirectional=gru_pool_gru_bidirectional,
            gru_pool_gru_dropout=gru_pool_gru_dropout,
            gru_pool_max_chunks=gru_pool_max_chunks,
            gru_pool_chunk_size=gru_pool_chunk_size,
            gru_pool_chunk_overlap=gru_pool_chunk_overlap,
            gru_pool_transformer_layers=gru_pool_transformer_layers,
            gru_pool_transformer_heads=gru_pool_transformer_heads,
            gru_pool_transformer_dim=gru_pool_transformer_dim,
            gru_pool_gated_attention_dim=gru_pool_gated_attention_dim,
            gru_pool_projection_dim=gru_pool_projection_dim,
            gru_pool_max_vocab=gru_pool_max_vocab,
            gru_pool_min_word_freq=gru_pool_min_word_freq,
            conv_pool_embedding_dim=conv_pool_embedding_dim,
            conv_pool_conv_dim=conv_pool_conv_dim,
            conv_pool_kernel_size=conv_pool_kernel_size,
            conv_pool_num_blocks=conv_pool_num_blocks,
            conv_pool_dropout=conv_pool_dropout,
            conv_pool_max_chunks=conv_pool_max_chunks,
            conv_pool_chunk_size=conv_pool_chunk_size,
            conv_pool_chunk_overlap=conv_pool_chunk_overlap,
            conv_pool_transformer_layers=conv_pool_transformer_layers,
            conv_pool_transformer_heads=conv_pool_transformer_heads,
            conv_pool_transformer_dim=conv_pool_transformer_dim,
            conv_pool_transformer_dropout=conv_pool_transformer_dropout,
            conv_pool_gated_attention_dim=conv_pool_gated_attention_dim,
            conv_pool_projection_dim=conv_pool_projection_dim,
            conv_pool_max_vocab=conv_pool_max_vocab,
            conv_pool_min_word_freq=conv_pool_min_word_freq,
            c1d_hybrid_embedding_dim=c1d_hybrid_embedding_dim,
            c1d_hybrid_conv_dim=c1d_hybrid_conv_dim,
            c1d_hybrid_kernel_size=c1d_hybrid_kernel_size,
            c1d_hybrid_num_blocks=c1d_hybrid_num_blocks,
            c1d_hybrid_conv_dropout=c1d_hybrid_conv_dropout,
            c1d_hybrid_pool_stride=c1d_hybrid_pool_stride,
            c1d_hybrid_max_length=c1d_hybrid_max_length,
            c1d_hybrid_transformer_layers=c1d_hybrid_transformer_layers,
            c1d_hybrid_transformer_heads=c1d_hybrid_transformer_heads,
            c1d_hybrid_transformer_dim=c1d_hybrid_transformer_dim,
            c1d_hybrid_transformer_dropout=c1d_hybrid_transformer_dropout,
            c1d_hybrid_gated_attention_dim=c1d_hybrid_gated_attention_dim,
            c1d_hybrid_projection_dim=c1d_hybrid_projection_dim,
            c1d_hybrid_max_vocab=c1d_hybrid_max_vocab,
            c1d_hybrid_min_word_freq=c1d_hybrid_min_word_freq,
            # Transformer Pool args
            tp_embedding_dim=tp_embedding_dim,
            tp_token_transformer_layers=tp_token_transformer_layers,
            tp_token_transformer_heads=tp_token_transformer_heads,
            tp_token_transformer_dim=tp_token_transformer_dim,
            tp_token_transformer_dropout=tp_token_transformer_dropout,
            tp_chunk_transformer_layers=tp_chunk_transformer_layers,
            tp_chunk_transformer_heads=tp_chunk_transformer_heads,
            tp_chunk_transformer_dim=tp_chunk_transformer_dim,
            tp_chunk_transformer_dropout=tp_chunk_transformer_dropout,
            tp_gated_attention_dim=tp_gated_attention_dim,
            tp_projection_dim=tp_projection_dim,
            tp_chunk_size=tp_chunk_size,
            tp_chunk_overlap=tp_chunk_overlap,
            tp_max_chunks=tp_max_chunks,
            tp_max_vocab=tp_max_vocab,
            tp_min_word_freq=tp_min_word_freq,
            bert_pool_sentence_model=bert_pool_sentence_model,
            bert_pool_freeze_sentence_encoder=bert_pool_freeze_sentence_encoder,
            bert_pool_use_pretrained=bert_pool_use_pretrained,
            bert_pool_max_chunks=bert_pool_max_chunks,
            bert_pool_chunk_size=bert_pool_chunk_size,
            bert_pool_chunk_overlap=bert_pool_chunk_overlap,
            bert_pool_transformer_layers=bert_pool_transformer_layers,
            bert_pool_transformer_heads=bert_pool_transformer_heads,
            bert_pool_transformer_dim=bert_pool_transformer_dim,
            bert_pool_transformer_dropout=bert_pool_transformer_dropout,
            bert_pool_gated_attention_dim=bert_pool_gated_attention_dim,
            bert_pool_projection_dim=bert_pool_projection_dim,
            llm_model_name=llm_model_name,
            llm_max_length=llm_max_length,
            llm_projection_dim=llm_projection_dim,
            llm_dropout=llm_dropout,
            llm_gradient_checkpointing=llm_gradient_checkpointing,
            llm_use_pretrained=llm_use_pretrained,
            flp_model_name=flp_model_name,
            flp_max_length=flp_max_length,
            flp_freeze_llm=flp_freeze_llm,
            flp_gated_attention_dim=flp_gated_attention_dim,
            flp_projection_dim=flp_projection_dim,
            flp_dropout=flp_dropout,
            flp_gradient_checkpointing=flp_gradient_checkpointing,
            flp_downprojection_dim=flp_downprojection_dim,
            flp_skip_llm=flp_skip_llm,
            flp_cached_hidden_size=flp_cached_hidden_size,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
        )
        logger.info(f"Propensity model using {self.feature_extractor_type.upper()} feature extractor")

        # Propensity network
        input_dim = self.feature_extractor.output_dim
        self.propensity_net = PropensityNet(
            input_dim=input_dim,
            representation_dim=representation_dim
        )

        # Move to device
        self.to(self._device)

        logger.info(f"PropensityOnlyModel initialized:")
        logger.info(f"  Feature extractor: {self.feature_extractor_type}")
        logger.info(f"  Feature extractor output: {input_dim}")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Device: {self._device}")

    @staticmethod
    def _get_extractor_input(batch, texts):
        """Return preprocessed batch if available, otherwise raw texts."""
        if 'cached_hidden_states' in batch:
            return {
                'cached_hidden_states': batch['cached_hidden_states'],
                'cached_attention_mask': batch['cached_attention_mask'],
                'texts': texts,
            }
        if 'chunk_input_ids' in batch or 'chunk_token_ids' in batch:
            return batch
        return texts

    def forward(self, texts_or_batch) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            texts_or_batch: List of text strings or preprocessed batch dict

        Returns:
            t_logit: Propensity logits (batch, 1)
        """
        features = self.feature_extractor(texts_or_batch)
        t_logit = self.propensity_net(features)
        return t_logit

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Args:
            batch: Dictionary with 'texts' and 'treatment' keys

        Returns:
            Dictionary with loss and predictions
        """
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        extractor_input = self._get_extractor_input(batch, texts)

        # Forward pass
        t_logit = self.forward(extractor_input)

        # Binary cross-entropy loss for treatment prediction
        loss = F.binary_cross_entropy_with_logits(
            t_logit.squeeze(-1),
            treatments
        )

        return {
            'loss': loss,
            't_logit': t_logit.detach()
        }

    def predict(self, texts_or_batch) -> torch.Tensor:
        """
        Predict propensity scores.

        Args:
            texts_or_batch: List of text strings or preprocessed batch dict from DataLoader

        Returns:
            Propensity probabilities (batch,)
        """
        with torch.no_grad():
            if isinstance(texts_or_batch, dict):
                texts = texts_or_batch['texts']
                extractor_input = self._get_extractor_input(texts_or_batch, texts)
            else:
                extractor_input = texts_or_batch
            t_logit = self.forward(extractor_input)
            propensity = torch.sigmoid(t_logit).squeeze(-1)
            return propensity

    def fit_tokenizer(self, texts: List[str]) -> 'PropensityOnlyModel':
        """
        Fit the word tokenizer on training texts.

        For CNN: This MUST be called before using the model for training or inference.
        For BERT: This is a no-op (BERT uses its pretrained tokenizer).

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        if hasattr(self.feature_extractor, 'fit_tokenizer'):
            self.feature_extractor.fit_tokenizer(texts)
        # BERT uses pretrained tokenizer, no fitting needed
        return self

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)


def create_propensity_model_from_config(
    arch_config,
    representation_dim: int,
    device: torch.device,
    flp_skip_llm: bool = False,
    flp_cached_hidden_size: int = 0
) -> PropensityOnlyModel:
    """
    Create a PropensityOnlyModel from architecture config.

    Args:
        arch_config: ModelArchitectureConfig instance
        representation_dim: Dimension for representation layers
        device: PyTorch device

    Returns:
        PropensityOnlyModel instance
    """
    feature_extractor_type = getattr(arch_config, 'feature_extractor_type', 'cnn')

    model = PropensityOnlyModel(
        feature_extractor_type=feature_extractor_type,
        # CNN args
        embedding_dim=arch_config.cnn_embedding_dim,
        kernel_sizes=arch_config.cnn_kernel_sizes,
        explicit_filter_concepts=arch_config.cnn_explicit_filter_concepts,
        num_kmeans_filters=arch_config.cnn_num_kmeans_filters,
        num_random_filters=arch_config.cnn_num_random_filters,
        cnn_dropout=arch_config.cnn_dropout,
        max_length=arch_config.cnn_max_length,
        min_word_freq=getattr(arch_config, 'cnn_min_word_freq', 2),
        max_vocab_size=getattr(arch_config, 'cnn_max_vocab_size', 50000),
        projection_dim=arch_config.causal_head_representation_dim,
        # BERT args
        bert_model_name=getattr(arch_config, 'bert_model_name', 'bert-base-uncased'),
        bert_max_length=getattr(arch_config, 'bert_max_length', 512),
        bert_projection_dim=getattr(arch_config, 'bert_projection_dim', 128),
        bert_dropout=getattr(arch_config, 'bert_dropout', 0.1),
        bert_freeze_encoder=getattr(arch_config, 'bert_freeze_encoder', False),
        bert_gradient_checkpointing=getattr(arch_config, 'bert_gradient_checkpointing', False),
        # GRU args
        gru_embedding_dim=getattr(arch_config, 'gru_embedding_dim', 256),
        gru_hidden_dim=getattr(arch_config, 'gru_hidden_dim', 256),
        gru_num_layers=getattr(arch_config, 'gru_num_layers', 2),
        gru_dropout=getattr(arch_config, 'gru_dropout', 0.1),
        gru_bidirectional=getattr(arch_config, 'gru_bidirectional', True),
        gru_attention_dim=getattr(arch_config, 'gru_attention_dim', None),
        gru_projection_dim=getattr(arch_config, 'gru_projection_dim', 128),
        gru_max_length=getattr(arch_config, 'gru_max_length', 8192),
        gru_min_word_freq=getattr(arch_config, 'gru_min_word_freq', 2),
        gru_max_vocab_size=getattr(arch_config, 'gru_max_vocab_size', 50000),
        # Hierarchical Transformer args
        hier_transformer_sentence_model=getattr(arch_config, 'hier_transformer_sentence_model', 'prajjwal1/bert-tiny'),
        hier_transformer_freeze_sentence_encoder=getattr(arch_config, 'hier_transformer_freeze_sentence_encoder', True),
        hier_transformer_max_chunks=getattr(arch_config, 'hier_transformer_max_chunks', 100),
        hier_transformer_chunk_size=getattr(arch_config, 'hier_transformer_chunk_size', 128),
        hier_transformer_chunk_overlap=getattr(arch_config, 'hier_transformer_chunk_overlap', 32),
        hier_transformer_num_layers=getattr(arch_config, 'hier_transformer_num_layers', 2),
        hier_transformer_num_heads=getattr(arch_config, 'hier_transformer_num_heads', 4),
        hier_transformer_dim=getattr(arch_config, 'hier_transformer_dim', 256),
        hier_transformer_dropout=getattr(arch_config, 'hier_transformer_dropout', 0.1),
        hier_transformer_projection_dim=getattr(arch_config, 'hier_transformer_projection_dim', 128),
        # BERT Cross-Chunk args
        bcc_sentence_model=getattr(arch_config, 'bcc_sentence_model', 'prajjwal1/bert-tiny'),
        bcc_freeze_sentence_encoder=getattr(arch_config, 'bcc_freeze_sentence_encoder', False),
        bcc_max_chunks=getattr(arch_config, 'bcc_max_chunks', 100),
        bcc_chunk_size=getattr(arch_config, 'bcc_chunk_size', 128),
        bcc_chunk_overlap=getattr(arch_config, 'bcc_chunk_overlap', 32),
        bcc_num_cross_layers=getattr(arch_config, 'bcc_num_cross_layers', 2),
        bcc_num_attention_heads=getattr(arch_config, 'bcc_num_attention_heads', 4),
        bcc_cross_chunk_dim=getattr(arch_config, 'bcc_cross_chunk_dim', 256),
        bcc_cross_chunk_dropout=getattr(arch_config, 'bcc_cross_chunk_dropout', 0.1),
        bcc_gated_attention_dim=getattr(arch_config, 'bcc_gated_attention_dim', 128),
        bcc_projection_dim=getattr(arch_config, 'bcc_projection_dim', 128),
        # GRU-Transformer-MIL args
        gru_mil_embedding_dim=getattr(arch_config, 'gru_mil_embedding_dim', 128),
        gru_mil_gru_hidden_dim=getattr(arch_config, 'gru_mil_gru_hidden_dim', 128),
        gru_mil_gru_num_layers=getattr(arch_config, 'gru_mil_gru_num_layers', 1),
        gru_mil_gru_bidirectional=getattr(arch_config, 'gru_mil_gru_bidirectional', True),
        gru_mil_gru_dropout=getattr(arch_config, 'gru_mil_gru_dropout', 0.1),
        gru_mil_max_chunks=getattr(arch_config, 'gru_mil_max_chunks', 100),
        gru_mil_chunk_size=getattr(arch_config, 'gru_mil_chunk_size', 128),
        gru_mil_chunk_overlap=getattr(arch_config, 'gru_mil_chunk_overlap', 32),
        gru_mil_transformer_layers=getattr(arch_config, 'gru_mil_transformer_layers', 2),
        gru_mil_transformer_heads=getattr(arch_config, 'gru_mil_transformer_heads', 4),
        gru_mil_transformer_dim=getattr(arch_config, 'gru_mil_transformer_dim', 256),
        gru_mil_num_confounders=getattr(arch_config, 'gru_mil_num_confounders', 4),
        gru_mil_mil_hidden_dim=getattr(arch_config, 'gru_mil_mil_hidden_dim', 128),
        gru_mil_projection_dim=getattr(arch_config, 'gru_mil_projection_dim', 128),
        gru_mil_max_vocab=getattr(arch_config, 'gru_mil_max_vocab', 50000),
        gru_mil_min_word_freq=getattr(arch_config, 'gru_mil_min_word_freq', 2),
        # GRU-Pool args
        gru_pool_embedding_dim=getattr(arch_config, 'gru_pool_embedding_dim', 128),
        gru_pool_gru_hidden_dim=getattr(arch_config, 'gru_pool_gru_hidden_dim', 128),
        gru_pool_gru_num_layers=getattr(arch_config, 'gru_pool_gru_num_layers', 1),
        gru_pool_gru_bidirectional=getattr(arch_config, 'gru_pool_gru_bidirectional', True),
        gru_pool_gru_dropout=getattr(arch_config, 'gru_pool_gru_dropout', 0.1),
        gru_pool_max_chunks=getattr(arch_config, 'gru_pool_max_chunks', 100),
        gru_pool_chunk_size=getattr(arch_config, 'gru_pool_chunk_size', 128),
        gru_pool_chunk_overlap=getattr(arch_config, 'gru_pool_chunk_overlap', 32),
        gru_pool_transformer_layers=getattr(arch_config, 'gru_pool_transformer_layers', 2),
        gru_pool_transformer_heads=getattr(arch_config, 'gru_pool_transformer_heads', 4),
        gru_pool_transformer_dim=getattr(arch_config, 'gru_pool_transformer_dim', 256),
        gru_pool_gated_attention_dim=getattr(arch_config, 'gru_pool_gated_attention_dim', 128),
        gru_pool_projection_dim=getattr(arch_config, 'gru_pool_projection_dim', 128),
        gru_pool_max_vocab=getattr(arch_config, 'gru_pool_max_vocab', 50000),
        gru_pool_min_word_freq=getattr(arch_config, 'gru_pool_min_word_freq', 2),
        # Conv-Pool args
        conv_pool_embedding_dim=getattr(arch_config, 'conv_pool_embedding_dim', 128),
        conv_pool_conv_dim=getattr(arch_config, 'conv_pool_conv_dim', 256),
        conv_pool_kernel_size=getattr(arch_config, 'conv_pool_kernel_size', 3),
        conv_pool_num_blocks=getattr(arch_config, 'conv_pool_num_blocks', 4),
        conv_pool_dropout=getattr(arch_config, 'conv_pool_dropout', 0.1),
        conv_pool_max_chunks=getattr(arch_config, 'conv_pool_max_chunks', 100),
        conv_pool_chunk_size=getattr(arch_config, 'conv_pool_chunk_size', 128),
        conv_pool_chunk_overlap=getattr(arch_config, 'conv_pool_chunk_overlap', 32),
        conv_pool_transformer_layers=getattr(arch_config, 'conv_pool_transformer_layers', 2),
        conv_pool_transformer_heads=getattr(arch_config, 'conv_pool_transformer_heads', 4),
        conv_pool_transformer_dim=getattr(arch_config, 'conv_pool_transformer_dim', 256),
        conv_pool_transformer_dropout=getattr(arch_config, 'conv_pool_transformer_dropout', 0.1),
        conv_pool_gated_attention_dim=getattr(arch_config, 'conv_pool_gated_attention_dim', 128),
        conv_pool_projection_dim=getattr(arch_config, 'conv_pool_projection_dim', 128),
        conv_pool_max_vocab=getattr(arch_config, 'conv_pool_max_vocab', 50000),
        conv_pool_min_word_freq=getattr(arch_config, 'conv_pool_min_word_freq', 2),
        # Conv1d-Transformer Hybrid args
        c1d_hybrid_embedding_dim=getattr(arch_config, 'c1d_hybrid_embedding_dim', 128),
        c1d_hybrid_conv_dim=getattr(arch_config, 'c1d_hybrid_conv_dim', 256),
        c1d_hybrid_kernel_size=getattr(arch_config, 'c1d_hybrid_kernel_size', 3),
        c1d_hybrid_num_blocks=getattr(arch_config, 'c1d_hybrid_num_blocks', 4),
        c1d_hybrid_conv_dropout=getattr(arch_config, 'c1d_hybrid_conv_dropout', 0.1),
        c1d_hybrid_pool_stride=getattr(arch_config, 'c1d_hybrid_pool_stride', 2),
        c1d_hybrid_max_length=getattr(arch_config, 'c1d_hybrid_max_length', 8192),
        c1d_hybrid_transformer_layers=getattr(arch_config, 'c1d_hybrid_transformer_layers', 2),
        c1d_hybrid_transformer_heads=getattr(arch_config, 'c1d_hybrid_transformer_heads', 4),
        c1d_hybrid_transformer_dim=getattr(arch_config, 'c1d_hybrid_transformer_dim', 256),
        c1d_hybrid_transformer_dropout=getattr(arch_config, 'c1d_hybrid_transformer_dropout', 0.1),
        c1d_hybrid_gated_attention_dim=getattr(arch_config, 'c1d_hybrid_gated_attention_dim', 128),
        c1d_hybrid_projection_dim=getattr(arch_config, 'c1d_hybrid_projection_dim', 128),
        c1d_hybrid_max_vocab=getattr(arch_config, 'c1d_hybrid_max_vocab', 50000),
        c1d_hybrid_min_word_freq=getattr(arch_config, 'c1d_hybrid_min_word_freq', 2),
        # Transformer Pool args
        tp_embedding_dim=getattr(arch_config, 'tp_embedding_dim', 128),
        tp_token_transformer_layers=getattr(arch_config, 'tp_token_transformer_layers', 2),
        tp_token_transformer_heads=getattr(arch_config, 'tp_token_transformer_heads', 4),
        tp_token_transformer_dim=getattr(arch_config, 'tp_token_transformer_dim', 256),
        tp_token_transformer_dropout=getattr(arch_config, 'tp_token_transformer_dropout', 0.1),
        tp_chunk_transformer_layers=getattr(arch_config, 'tp_chunk_transformer_layers', 2),
        tp_chunk_transformer_heads=getattr(arch_config, 'tp_chunk_transformer_heads', 4),
        tp_chunk_transformer_dim=getattr(arch_config, 'tp_chunk_transformer_dim', 256),
        tp_chunk_transformer_dropout=getattr(arch_config, 'tp_chunk_transformer_dropout', 0.1),
        tp_gated_attention_dim=getattr(arch_config, 'tp_gated_attention_dim', 128),
        tp_projection_dim=getattr(arch_config, 'tp_projection_dim', 128),
        tp_chunk_size=getattr(arch_config, 'tp_chunk_size', 128),
        tp_chunk_overlap=getattr(arch_config, 'tp_chunk_overlap', 32),
        tp_max_chunks=getattr(arch_config, 'tp_max_chunks', 100),
        tp_max_vocab=getattr(arch_config, 'tp_max_vocab', 50000),
        tp_min_word_freq=getattr(arch_config, 'tp_min_word_freq', 2),
        # BERT Pool args
        bert_pool_sentence_model=getattr(arch_config, 'bert_pool_sentence_model', 'prajjwal1/bert-tiny'),
        bert_pool_freeze_sentence_encoder=getattr(arch_config, 'bert_pool_freeze_sentence_encoder', False),
        bert_pool_use_pretrained=getattr(arch_config, 'bert_pool_use_pretrained', True),
        bert_pool_max_chunks=getattr(arch_config, 'bert_pool_max_chunks', 100),
        bert_pool_chunk_size=getattr(arch_config, 'bert_pool_chunk_size', 128),
        bert_pool_chunk_overlap=getattr(arch_config, 'bert_pool_chunk_overlap', 32),
        bert_pool_transformer_layers=getattr(arch_config, 'bert_pool_transformer_layers', 2),
        bert_pool_transformer_heads=getattr(arch_config, 'bert_pool_transformer_heads', 4),
        bert_pool_transformer_dim=getattr(arch_config, 'bert_pool_transformer_dim', 256),
        bert_pool_transformer_dropout=getattr(arch_config, 'bert_pool_transformer_dropout', 0.1),
        bert_pool_gated_attention_dim=getattr(arch_config, 'bert_pool_gated_attention_dim', 128),
        bert_pool_projection_dim=getattr(arch_config, 'bert_pool_projection_dim', 128),
        # LLM args
        llm_model_name=getattr(arch_config, 'llm_model_name', 'Qwen/Qwen3-0.6B-Base'),
        llm_max_length=getattr(arch_config, 'llm_max_length', 8192),
        llm_projection_dim=getattr(arch_config, 'llm_projection_dim', 128),
        llm_dropout=getattr(arch_config, 'llm_dropout', 0.1),
        llm_gradient_checkpointing=getattr(arch_config, 'llm_gradient_checkpointing', True),
        llm_use_pretrained=getattr(arch_config, 'llm_use_pretrained', False),
        # Frozen LLM Pooler args
        flp_model_name=getattr(arch_config, 'flp_model_name', 'Qwen/Qwen3-0.6B-Base'),
        flp_max_length=getattr(arch_config, 'flp_max_length', 8192),
        flp_freeze_llm=getattr(arch_config, 'flp_freeze_llm', True),
        flp_gated_attention_dim=getattr(arch_config, 'flp_gated_attention_dim', 128),
        flp_projection_dim=getattr(arch_config, 'flp_projection_dim', 128),
        flp_dropout=getattr(arch_config, 'flp_dropout', 0.1),
        flp_gradient_checkpointing=getattr(arch_config, 'flp_gradient_checkpointing', True),
        flp_downprojection_dim=getattr(arch_config, 'flp_downprojection_dim', None),
        flp_skip_llm=flp_skip_llm,
        flp_cached_hidden_size=flp_cached_hidden_size,
        # Numeric feature args
        numeric_features_enabled=getattr(arch_config, 'numeric_features_enabled', False),
        numeric_embedding_dim=getattr(arch_config, 'numeric_embedding_dim', 32),
        numeric_magnitude_bins=getattr(arch_config, 'numeric_magnitude_bins', 8),
        numeric_type_categories=getattr(arch_config, 'numeric_type_categories', 10),
        # Propensity network args
        representation_dim=representation_dim,
        device=str(device)
    )

    return model
