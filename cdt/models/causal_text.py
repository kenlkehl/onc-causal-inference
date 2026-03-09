# cdt/models/causal_text.py
"""Causal inference model using simple 1D CNN for text representation."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dragonnet import DragonNet
from .uplift import UpliftNet
from .rlearner import RLearnerNet
from .traditional_logreg import TraditionalLogRegNet
from .dr_moce import DRMoCENet, NuisancePredictionBuffer, compute_dr_pseudo_outcome
from .explicit_confounder_featurizer import ExplicitConfounderFeaturizer
from .intra_batch_contrastive import IntraBatchContrastiveLoss
from .extractor_factory import create_feature_extractor
from ..config import normalize_feature_extractor_type, ExplicitConfounderSpec


logger = logging.getLogger(__name__)


class CausalText(nn.Module):
    """
    Causal inference model for text using various feature extraction methods.

    Architecture:
    - Feature extractor (CNN, BERT, GRU, or Confounder) encodes text into feature vector
    - DragonNet/UpliftNet/RLearnerNet predicts outcomes and propensity

    CNN mode (feature_extractor_type="cnn"):
    - 1D CNN with word-level tokenization
    - Much faster to train than transformers
    - IMPORTANT: Call fit_tokenizer(texts) with training data before use

    BERT mode (feature_extractor_type="bert"):
    - HuggingFace transformer with CLS token extraction
    - Fine-tuning or frozen encoder options
    - No fit_tokenizer() needed (uses pretrained tokenizer)
    - O(N^2) attention - may struggle with very long sequences

    GRU mode (feature_extractor_type="gru"):
    - Bidirectional GRU with attention pooling
    - O(N) complexity - efficient for long sequences
    - Attention weights provide interpretability
    - IMPORTANT: Call fit_tokenizer(texts) with training data before use

    Confounder mode (feature_extractor_type="confounder"):
    - Perceiver-style cross-attention with sparse attention
    - Hierarchical option: sentence-level + token-level attention
    - Designed for extracting confounders from long clinical text
    - No fit_tokenizer() needed (uses pretrained encoders)
    """

    def __init__(
        self,
        # Feature extractor type
        feature_extractor_type: str = "cnn",
        # CNN-specific args
        embedding_dim: int = 128,
        kernel_sizes: List[int] = [3, 4, 5, 7],
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
        # Conv-Pool args (dilated convolution variant of GRU-Pool)
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
        # Conv1D Transformer Hybrid args (full-document dilated conv + stride downsample + transformer)
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
        # LLM Feature Extractor args
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
        # CLAM instance-level loss args (for GRU-Pool extractor)
        clam_enabled: bool = False,
        clam_num_instances: int = 5,
        clam_instance_hidden_dim: int = 64,
        # Intra-batch contrastive learning args
        contrastive_enabled: bool = False,
        contrastive_num_clusters: int = 4,
        contrastive_temperature: float = 0.1,
        contrastive_label_mode: str = "joint",
        contrastive_projection_dim: int = 64,
        contrastive_min_cluster_size: int = 2,
        contrastive_clustering_method: str = "kmeans",
        # Causal head args (applies to all causal heads: DragonNet, RLearner, UpliftNet, etc.)
        causal_head_representation_dim: int = 128,
        causal_head_hidden_outcome_dim: int = 64,
        causal_head_dropout: float = 0.2,
        device: str = "cuda:0",
        model_type: str = "dragonnet",  # "dragonnet", "uplift", or "rlearner"
        # Auxiliary features (for hybrid text + categorical models)
        auxiliary_dim: int = 0,  # Dimension of auxiliary categorical features (0 = no auxiliary)
        # Numeric feature args
        numeric_features_enabled: bool = False,
        numeric_embedding_dim: int = 32,
        numeric_magnitude_bins: int = 8,
        numeric_type_categories: int = 10,
        # Explicit confounder featurizer args
        explicit_confounder_specs: Optional[List[ExplicitConfounderSpec]] = None,
        explicit_confounder_output_dim: int = 64,
        explicit_confounder_hidden_dim: int = 128,
        explicit_confounder_dropout: float = 0.1,
        # R-Learner dual extractor mode
        rlearner_dual_extractors: bool = False,
        # Uplift dual extractor mode
        uplift_dual_extractors: bool = False,
        # DR-MoCE args
        dr_moce_num_experts: int = 8,
        dr_moce_router_temperature: float = 1.0,
        dr_moce_propensity_clip: float = 0.01,
        dr_moce_het_weight: float = 0.1,
        dr_moce_balance_weight: float = 0.01,
        dr_moce_crossfit_buffer_size: int = 1024,
        # Outcome type
        outcome_type: str = "binary",  # "binary" or "continuous"
    ):
        """
        Initialize causal inference model with CNN, BERT, or GRU feature extractor.

        Args:
            feature_extractor_type: "cnn", "bert", or "gru"
            embedding_dim: (CNN/GRU) Dimension of word embeddings
            kernel_sizes: (CNN) List of kernel sizes for n-gram capture
            explicit_filter_concepts: (CNN) Dict mapping kernel_size to concept phrases
            num_kmeans_filters: (CNN) Number of k-means derived filters per kernel size
            num_random_filters: (CNN) Number of randomly initialized filters per kernel size
            cnn_dropout: (CNN) Dropout rate
            max_length: (CNN/GRU) Maximum sequence length in tokens
            min_word_freq: (CNN/GRU) Minimum word frequency for vocabulary inclusion
            max_vocab_size: (CNN/GRU) Maximum vocabulary size
            projection_dim: (CNN) Dimension to project CNN output to
            bert_model_name: (BERT) HuggingFace model name or path
            bert_max_length: (BERT) Maximum sequence length in subword tokens
            bert_projection_dim: (BERT) Projection dimension after CLS token
            bert_dropout: (BERT) Dropout rate for projection layer
            bert_freeze_encoder: (BERT) Whether to freeze transformer weights
            bert_gradient_checkpointing: (BERT) Enable gradient checkpointing
            gru_hidden_dim: (GRU) Hidden state dimension per direction
            gru_num_layers: (GRU) Number of stacked GRU layers
            gru_dropout: (GRU) Dropout rate
            gru_bidirectional: (GRU) Use bidirectional GRU
            gru_attention_dim: (GRU) Attention hidden dimension (default: 2*hidden_dim)
            gru_projection_dim: (GRU) Output projection dimension
            causal_head_representation_dim: Causal head representation dimension
            causal_head_hidden_outcome_dim: Causal head outcome hidden dimension
            causal_head_dropout: Dropout rate for causal head layers
            device: Device string
            model_type: Architecture type ("dragonnet", "uplift", or "rlearner")
            auxiliary_dim: Dimension of auxiliary categorical features (0 = disabled)
        """
        super().__init__()

        self._device = torch.device(device)
        self.model_type = model_type
        self.outcome_type = outcome_type
        # Normalize feature extractor type (e.g., "modernbert" -> "bert")
        self.feature_extractor_type = normalize_feature_extractor_type(feature_extractor_type)

        # Store config for checkpointing (store original type for reproducibility)
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
            'gru_hidden_dim': gru_hidden_dim,
            'gru_num_layers': gru_num_layers,
            'gru_dropout': gru_dropout,
            'gru_bidirectional': gru_bidirectional,
            'gru_attention_dim': gru_attention_dim,
            'gru_projection_dim': gru_projection_dim,
            'confounder_num_latents': confounder_num_latents,
            'confounder_explicit_texts': confounder_explicit_texts,
            'confounder_value_dim': confounder_value_dim,
            'confounder_sentence_model': confounder_sentence_model,
            'confounder_freeze_encoder': confounder_freeze_encoder,
            'confounder_max_sentences': confounder_max_sentences,
            'confounder_num_heads': confounder_num_heads,
            'confounder_num_iterations': confounder_num_iterations,
            'confounder_use_self_attention': confounder_use_self_attention,
            'confounder_sparse_attention': confounder_sparse_attention,
            'confounder_sparse_method': confounder_sparse_method,
            'confounder_sparse_alpha': confounder_sparse_alpha,
            'confounder_top_k': confounder_top_k,
            'confounder_dropout': confounder_dropout,
            'confounder_hierarchical': confounder_hierarchical,
            'confounder_token_encoder': confounder_token_encoder,
            'confounder_freeze_token_encoder': confounder_freeze_token_encoder,
            'confounder_max_sentence_tokens': confounder_max_sentence_tokens,
            'confounder_use_gru': confounder_use_gru,
            'confounder_gru_embedding_dim': confounder_gru_embedding_dim,
            'confounder_gru_hidden_dim': confounder_gru_hidden_dim,
            'confounder_gru_num_layers': confounder_gru_num_layers,
            'confounder_gru_bidirectional': confounder_gru_bidirectional,
            'confounder_gru_dropout': confounder_gru_dropout,
            'confounder_gru_max_vocab': confounder_gru_max_vocab,
            'confounder_gru_min_word_freq': confounder_gru_min_word_freq,
            'confounder_gru_max_sentence_length': confounder_gru_max_sentence_length,
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
            'gated_mil_sentence_model': gated_mil_sentence_model,
            'gated_mil_freeze_sentence_encoder': gated_mil_freeze_sentence_encoder,
            'gated_mil_max_chunks': gated_mil_max_chunks,
            'gated_mil_chunk_size': gated_mil_chunk_size,
            'gated_mil_chunk_overlap': gated_mil_chunk_overlap,
            'gated_mil_hidden_dim': gated_mil_hidden_dim,
            'gated_mil_num_confounders': gated_mil_num_confounders,
            'gated_mil_dropout': gated_mil_dropout,
            'gated_mil_projection_dim': gated_mil_projection_dim,
            'gated_mil_hierarchical': gated_mil_hierarchical,
            'gated_mil_token_hidden_dim': gated_mil_token_hidden_dim,
            'gated_mil_use_mean_pooling': gated_mil_use_mean_pooling,
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
            'clam_enabled': clam_enabled,
            'clam_num_instances': clam_num_instances,
            'clam_instance_hidden_dim': clam_instance_hidden_dim,
            'contrastive_enabled': contrastive_enabled,
            'contrastive_num_clusters': contrastive_num_clusters,
            'contrastive_temperature': contrastive_temperature,
            'contrastive_label_mode': contrastive_label_mode,
            'contrastive_projection_dim': contrastive_projection_dim,
            'contrastive_min_cluster_size': contrastive_min_cluster_size,
            'contrastive_clustering_method': contrastive_clustering_method,
            'causal_head_representation_dim': causal_head_representation_dim,
            'causal_head_hidden_outcome_dim': causal_head_hidden_outcome_dim,
            'causal_head_dropout': causal_head_dropout,
            'model_type': model_type,
            'auxiliary_dim': auxiliary_dim,
            'numeric_features_enabled': numeric_features_enabled,
            'numeric_embedding_dim': numeric_embedding_dim,
            'numeric_magnitude_bins': numeric_magnitude_bins,
            'numeric_type_categories': numeric_type_categories,
            'explicit_confounder_specs': explicit_confounder_specs,
            'explicit_confounder_output_dim': explicit_confounder_output_dim,
            'explicit_confounder_hidden_dim': explicit_confounder_hidden_dim,
            'explicit_confounder_dropout': explicit_confounder_dropout,
            'rlearner_dual_extractors': rlearner_dual_extractors,
            'uplift_dual_extractors': uplift_dual_extractors,
            'dr_moce_num_experts': dr_moce_num_experts,
            'dr_moce_router_temperature': dr_moce_router_temperature,
            'dr_moce_propensity_clip': dr_moce_propensity_clip,
            'dr_moce_het_weight': dr_moce_het_weight,
            'dr_moce_balance_weight': dr_moce_balance_weight,
            'dr_moce_crossfit_buffer_size': dr_moce_crossfit_buffer_size,
            'outcome_type': outcome_type,
        }

        # Store auxiliary dimension
        self.auxiliary_dim = auxiliary_dim

        # Initialize feature extractor using factory
        self.feature_extractor = create_feature_extractor(
            extractor_type=self.feature_extractor_type,
            device=self._device,
            model_type=model_type,
            # CNN args
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
            # BERT args
            bert_model_name=bert_model_name,
            bert_max_length=bert_max_length,
            bert_projection_dim=bert_projection_dim,
            bert_dropout=bert_dropout,
            bert_freeze_encoder=bert_freeze_encoder,
            bert_gradient_checkpointing=bert_gradient_checkpointing,
            # GRU args
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            gru_dropout=gru_dropout,
            gru_bidirectional=gru_bidirectional,
            gru_attention_dim=gru_attention_dim,
            gru_projection_dim=gru_projection_dim,
            # Confounder args
            confounder_num_latents=confounder_num_latents,
            confounder_explicit_texts=confounder_explicit_texts,
            confounder_value_dim=confounder_value_dim,
            confounder_sentence_model=confounder_sentence_model,
            confounder_freeze_encoder=confounder_freeze_encoder,
            confounder_max_sentences=confounder_max_sentences,
            confounder_num_heads=confounder_num_heads,
            confounder_num_iterations=confounder_num_iterations,
            confounder_use_self_attention=confounder_use_self_attention,
            confounder_sparse_attention=confounder_sparse_attention,
            confounder_sparse_method=confounder_sparse_method,
            confounder_sparse_alpha=confounder_sparse_alpha,
            confounder_top_k=confounder_top_k,
            confounder_dropout=confounder_dropout,
            confounder_hierarchical=confounder_hierarchical,
            confounder_token_encoder=confounder_token_encoder,
            confounder_freeze_token_encoder=confounder_freeze_token_encoder,
            confounder_max_sentence_tokens=confounder_max_sentence_tokens,
            confounder_use_gru=confounder_use_gru,
            confounder_gru_embedding_dim=confounder_gru_embedding_dim,
            confounder_gru_hidden_dim=confounder_gru_hidden_dim,
            confounder_gru_num_layers=confounder_gru_num_layers,
            confounder_gru_bidirectional=confounder_gru_bidirectional,
            confounder_gru_dropout=confounder_gru_dropout,
            confounder_gru_max_vocab=confounder_gru_max_vocab,
            confounder_gru_min_word_freq=confounder_gru_min_word_freq,
            confounder_gru_max_sentence_length=confounder_gru_max_sentence_length,
            # Hierarchical Transformer args
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
            # BERT Cross-Chunk args
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
            # Gated MIL args
            gated_mil_sentence_model=gated_mil_sentence_model,
            gated_mil_freeze_sentence_encoder=gated_mil_freeze_sentence_encoder,
            gated_mil_max_chunks=gated_mil_max_chunks,
            gated_mil_chunk_size=gated_mil_chunk_size,
            gated_mil_chunk_overlap=gated_mil_chunk_overlap,
            gated_mil_hidden_dim=gated_mil_hidden_dim,
            gated_mil_num_confounders=gated_mil_num_confounders,
            gated_mil_dropout=gated_mil_dropout,
            gated_mil_projection_dim=gated_mil_projection_dim,
            gated_mil_hierarchical=gated_mil_hierarchical,
            gated_mil_token_hidden_dim=gated_mil_token_hidden_dim,
            gated_mil_use_mean_pooling=gated_mil_use_mean_pooling,
            # GRU-Transformer-MIL args
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
            # GRU-Pool args
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
            # Conv-Pool args
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
            # Conv1D Transformer Hybrid args
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
            # BERT Pool args
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
            # LLM args
            llm_model_name=llm_model_name,
            llm_max_length=llm_max_length,
            llm_projection_dim=llm_projection_dim,
            llm_dropout=llm_dropout,
            llm_gradient_checkpointing=llm_gradient_checkpointing,
            llm_use_pretrained=llm_use_pretrained,
            # Frozen LLM Pooler args
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
            # Numeric args
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
        )

        # Auxiliary feature projection (if enabled)
        if auxiliary_dim > 0:
            self.auxiliary_projection = nn.Sequential(
                nn.Linear(auxiliary_dim, causal_head_representation_dim // 2),
                nn.LayerNorm(causal_head_representation_dim // 2),
                nn.ReLU(),
                nn.Dropout(causal_head_dropout)
            )
            logger.info(f"Auxiliary features enabled: {auxiliary_dim} -> {causal_head_representation_dim // 2}")
        else:
            self.auxiliary_projection = None

        # Explicit confounder featurizer (if specs provided)
        self.explicit_confounder_specs = explicit_confounder_specs
        if explicit_confounder_specs and len(explicit_confounder_specs) > 0:
            self.explicit_confounder_featurizer = ExplicitConfounderFeaturizer(
                specs=explicit_confounder_specs,
                output_dim=explicit_confounder_output_dim,
                hidden_dim=explicit_confounder_hidden_dim,
                dropout=explicit_confounder_dropout,
                device=str(self._device)
            )
            logger.info(f"Explicit confounder featurizer enabled: {len(explicit_confounder_specs)} confounders, "
                       f"output_dim={explicit_confounder_output_dim}")
        else:
            self.explicit_confounder_featurizer = None

        # Binary treatment Causal Inference Net
        # Input dim = text features + auxiliary features (if any) + explicit confounder features (if any)
        input_dim = self.feature_extractor.output_dim
        if auxiliary_dim > 0:
            input_dim += causal_head_representation_dim // 2
        if self.explicit_confounder_featurizer is not None:
            input_dim += explicit_confounder_output_dim

        if model_type == "uplift":
            self.net = UpliftNet(
                input_dim=input_dim,
                representation_dim=causal_head_representation_dim,
                hidden_outcome_dim=causal_head_hidden_outcome_dim,
                dropout=causal_head_dropout
            )
            logger.info("Using UpliftNet architecture (Base + ITE parametrization)")
        elif model_type == "rlearner":
            self.net = RLearnerNet(
                input_dim=input_dim,
                representation_dim=causal_head_representation_dim,
                hidden_outcome_dim=causal_head_hidden_outcome_dim,
                dropout=causal_head_dropout
            )
            logger.info("Using R-Learner architecture (direct tau optimization)")
        elif model_type == "traditional_logreg":
            self.net = TraditionalLogRegNet(
                input_dim=input_dim,
                representation_dim=causal_head_representation_dim,
                hidden_outcome_dim=causal_head_hidden_outcome_dim,
                dropout=causal_head_dropout
            )
            logger.info("Using Traditional LogReg architecture (treatment as feature)")
        elif model_type == "dr_moce":
            self.net = DRMoCENet(
                input_dim=input_dim,
                representation_dim=causal_head_representation_dim,
                hidden_outcome_dim=causal_head_hidden_outcome_dim,
                num_experts=dr_moce_num_experts,
                router_temperature=dr_moce_router_temperature,
                dropout=causal_head_dropout
            )
            # Store DR-MoCE specific config
            self.dr_moce_propensity_clip = dr_moce_propensity_clip
            self.dr_moce_het_weight = dr_moce_het_weight
            self.dr_moce_balance_weight = dr_moce_balance_weight
            # Initialize prediction buffer for cross-fitting
            if dr_moce_crossfit_buffer_size > 0:
                self.dr_moce_buffer = NuisancePredictionBuffer(
                    buffer_size=dr_moce_crossfit_buffer_size
                )
            else:
                self.dr_moce_buffer = None
            logger.info(f"Using DR-MoCE architecture ({dr_moce_num_experts} experts, "
                       f"temp={dr_moce_router_temperature}, "
                       f"buffer={'enabled' if dr_moce_crossfit_buffer_size > 0 else 'disabled'})")
        else:
            self.net = DragonNet(
                input_dim=input_dim,
                representation_dim=causal_head_representation_dim,
                hidden_outcome_dim=causal_head_hidden_outcome_dim,
                dropout=causal_head_dropout
            )
            logger.info("Using classic DragonNet architecture")

        # Alias for backward compatibility
        self.dragonnet = self.net

        # CLAM instance-level loss head (for hierarchical extractors)
        # Creates a separate, lightweight causal head for top-attended chunks
        # Supported extractors: gru_pool, conv_pool, hierarchical_transformer, gated_mil_hierarchical, gru_transformer_mil
        self.clam_enabled = clam_enabled
        self.clam_num_instances = clam_num_instances
        self.clam_instance_hidden_dim = clam_instance_hidden_dim
        self.instance_head = None

        # Define which extractors support CLAM and their instance input dimensions
        clam_supported_extractors = {
            "gru_pool": gru_pool_transformer_dim,
            "conv_pool": conv_pool_transformer_dim,
            "conv1d_transformer_hybrid": c1d_hybrid_transformer_dim,
            "transformer_pool": tp_chunk_transformer_dim,
            "bert_pool": bert_pool_transformer_dim,
            "hierarchical_transformer": hier_transformer_dim,
            "gated_mil_hierarchical": None,  # Needs lazy initialization (sentence_dim from BERT)
            "gru_transformer_mil": gru_mil_transformer_dim,
        }

        if clam_enabled:
            if self.feature_extractor_type not in clam_supported_extractors:
                logger.warning(f"CLAM instance loss is not supported for {self.feature_extractor_type} extractor. "
                              f"Supported extractors: {list(clam_supported_extractors.keys())}. Disabling CLAM.")
                self.clam_enabled = False
            else:
                # Get instance input dimension based on extractor type
                instance_input_dim = clam_supported_extractors[self.feature_extractor_type]

                # For gated_mil_hierarchical, we need to get sentence_dim after lazy init
                # Store the dimension source for lazy initialization
                if instance_input_dim is None:
                    # Will be set during first forward pass when feature_extractor is initialized
                    # For now, use a typical BERT-tiny hidden size as placeholder
                    # The actual dimension is the sentence encoder's hidden size
                    self._clam_instance_dim_source = "gated_mil_sentence_dim"
                    # bert-tiny has hidden_size=128, we'll verify this during training
                    instance_input_dim = 128  # Default for bert-tiny
                    logger.info(f"CLAM: Using estimated sentence_dim={instance_input_dim} for gated_mil_hierarchical. "
                               f"Will be verified during initialization.")
                else:
                    self._clam_instance_dim_source = None

                # Use the same causal head type as the main model
                if model_type == "rlearner":
                    self.instance_head = RLearnerNet(
                        input_dim=instance_input_dim,
                        representation_dim=clam_instance_hidden_dim,
                        hidden_outcome_dim=clam_instance_hidden_dim // 2,
                        dropout=causal_head_dropout
                    )
                elif model_type == "uplift":
                    self.instance_head = UpliftNet(
                        input_dim=instance_input_dim,
                        representation_dim=clam_instance_hidden_dim,
                        hidden_outcome_dim=clam_instance_hidden_dim // 2,
                        dropout=causal_head_dropout
                    )
                elif model_type == "traditional_logreg":
                    self.instance_head = TraditionalLogRegNet(
                        input_dim=instance_input_dim,
                        representation_dim=clam_instance_hidden_dim,
                        hidden_outcome_dim=clam_instance_hidden_dim // 2,
                        dropout=causal_head_dropout
                    )
                else:
                    self.instance_head = DragonNet(
                        input_dim=instance_input_dim,
                        representation_dim=clam_instance_hidden_dim,
                        hidden_outcome_dim=clam_instance_hidden_dim // 2,
                        dropout=causal_head_dropout
                    )
                logger.info(f"CLAM instance-level loss enabled: {clam_num_instances} top chunks, "
                           f"instance_input_dim={instance_input_dim}, instance_head_dim={clam_instance_hidden_dim}")

        # R-Learner and Uplift dual extractor mode
        # When enabled, creates a second independent feature extractor for τ(X)
        # The nuisance extractor (self.feature_extractor) handles e(X) and m(X)/Y0(X)
        # The effect extractor (self.effect_feature_extractor) handles τ(X)
        self.rlearner_dual_extractors = rlearner_dual_extractors
        self.uplift_dual_extractors = uplift_dual_extractors
        self.effect_feature_extractor = None
        self.effect_mlp = None

        # Check for dual extractor mode (R-Learner or Uplift)
        dual_mode_enabled = (
            (rlearner_dual_extractors and model_type == "rlearner") or
            (uplift_dual_extractors and model_type == "uplift")
        )

        if dual_mode_enabled:
            # Create second feature extractor with same architecture using factory
            self.effect_feature_extractor = create_feature_extractor(
                extractor_type=self.feature_extractor_type,
                device=self._device,
                model_type=model_type,
                # CNN args
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
                # BERT args
                bert_model_name=bert_model_name,
                bert_max_length=bert_max_length,
                bert_projection_dim=bert_projection_dim,
                bert_dropout=bert_dropout,
                bert_freeze_encoder=bert_freeze_encoder,
                bert_gradient_checkpointing=bert_gradient_checkpointing,
                # GRU args
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                gru_dropout=gru_dropout,
                gru_bidirectional=gru_bidirectional,
                gru_attention_dim=gru_attention_dim,
                gru_projection_dim=gru_projection_dim,
                # Confounder args
                confounder_num_latents=confounder_num_latents,
                confounder_explicit_texts=confounder_explicit_texts,
                confounder_value_dim=confounder_value_dim,
                confounder_sentence_model=confounder_sentence_model,
                confounder_freeze_encoder=confounder_freeze_encoder,
                confounder_max_sentences=confounder_max_sentences,
                confounder_num_heads=confounder_num_heads,
                confounder_num_iterations=confounder_num_iterations,
                confounder_use_self_attention=confounder_use_self_attention,
                confounder_sparse_attention=confounder_sparse_attention,
                confounder_sparse_method=confounder_sparse_method,
                confounder_sparse_alpha=confounder_sparse_alpha,
                confounder_top_k=confounder_top_k,
                confounder_dropout=confounder_dropout,
                confounder_hierarchical=confounder_hierarchical,
                confounder_token_encoder=confounder_token_encoder,
                confounder_freeze_token_encoder=confounder_freeze_token_encoder,
                confounder_max_sentence_tokens=confounder_max_sentence_tokens,
                confounder_use_gru=confounder_use_gru,
                confounder_gru_embedding_dim=confounder_gru_embedding_dim,
                confounder_gru_hidden_dim=confounder_gru_hidden_dim,
                confounder_gru_num_layers=confounder_gru_num_layers,
                confounder_gru_bidirectional=confounder_gru_bidirectional,
                confounder_gru_dropout=confounder_gru_dropout,
                confounder_gru_max_vocab=confounder_gru_max_vocab,
                confounder_gru_min_word_freq=confounder_gru_min_word_freq,
                confounder_gru_max_sentence_length=confounder_gru_max_sentence_length,
                # Hierarchical Transformer args
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
                # Gated MIL args
                gated_mil_sentence_model=gated_mil_sentence_model,
                gated_mil_freeze_sentence_encoder=gated_mil_freeze_sentence_encoder,
                gated_mil_max_chunks=gated_mil_max_chunks,
                gated_mil_chunk_size=gated_mil_chunk_size,
                gated_mil_chunk_overlap=gated_mil_chunk_overlap,
                gated_mil_hidden_dim=gated_mil_hidden_dim,
                gated_mil_num_confounders=gated_mil_num_confounders,
                gated_mil_dropout=gated_mil_dropout,
                gated_mil_projection_dim=gated_mil_projection_dim,
                gated_mil_hierarchical=gated_mil_hierarchical,
                gated_mil_token_hidden_dim=gated_mil_token_hidden_dim,
                gated_mil_use_mean_pooling=gated_mil_use_mean_pooling,
                # GRU-Transformer-MIL args
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
                # GRU-Pool args
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
                # Conv-Pool args
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
                # BERT Pool args
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
                # LLM args
                llm_model_name=llm_model_name,
                llm_max_length=llm_max_length,
                llm_projection_dim=llm_projection_dim,
                llm_dropout=llm_dropout,
                llm_gradient_checkpointing=llm_gradient_checkpointing,
                llm_use_pretrained=llm_use_pretrained,
                # Frozen LLM Pooler args
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
                # Numeric args
                numeric_features_enabled=numeric_features_enabled,
                numeric_embedding_dim=numeric_embedding_dim,
                numeric_magnitude_bins=numeric_magnitude_bins,
                numeric_type_categories=numeric_type_categories,
            )

            # Simple MLP for τ(X) - takes effect extractor output, predicts treatment effect
            # Note: τ is unbounded (can be negative) - no final activation
            effect_input_dim = self.effect_feature_extractor.output_dim
            self.effect_mlp = nn.Sequential(
                nn.Linear(effect_input_dim, causal_head_hidden_outcome_dim),
                nn.ReLU(),
                nn.Dropout(causal_head_dropout),
                nn.Linear(causal_head_hidden_outcome_dim, causal_head_hidden_outcome_dim),
                nn.ELU(),
                nn.Dropout(causal_head_dropout),
                nn.Linear(causal_head_hidden_outcome_dim, 1)  # τ is unbounded
            )

            if model_type == "rlearner":
                logger.info(f"R-Learner dual extractor mode enabled:")
                logger.info(f"  Nuisance extractor: {self.feature_extractor_type} -> e(X), m(X)")
                logger.info(f"  Effect extractor: {self.feature_extractor_type} -> τ(X)")
                logger.info(f"  Effect MLP: {effect_input_dim} -> {causal_head_hidden_outcome_dim} -> 1")
            else:  # uplift
                logger.info(f"Uplift dual extractor mode enabled:")
                logger.info(f"  Nuisance extractor: {self.feature_extractor_type} -> e(X), Y0(X)")
                logger.info(f"  Effect extractor: {self.feature_extractor_type} -> τ(X)")
                logger.info(f"  Effect MLP: {effect_input_dim} -> {causal_head_hidden_outcome_dim} -> 1")

        # Intra-batch contrastive learning module
        self.contrastive_enabled = contrastive_enabled
        self.contrastive_loss_module = None

        if contrastive_enabled:
            self.contrastive_loss_module = IntraBatchContrastiveLoss(
                feature_dim=self.feature_extractor.output_dim,
                num_clusters=contrastive_num_clusters,
                temperature=contrastive_temperature,
                label_mode=contrastive_label_mode,
                projection_dim=contrastive_projection_dim,
                min_cluster_size=contrastive_min_cluster_size,
                clustering_method=contrastive_clustering_method,
            )
            logger.info(f"Intra-batch contrastive loss enabled: K={contrastive_num_clusters}, "
                       f"mode={contrastive_label_mode}, temp={contrastive_temperature}")

        # Move to device
        self.to(self._device)

        logger.info(f"CausalText initialized:")
        logger.info(f"  Feature extractor: {self.feature_extractor_type}")
        logger.info(f"  Feature extractor output: {input_dim}")
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

    def forward(
        self,
        texts: List[str],
        auxiliary_features: Optional[torch.Tensor] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            texts: List of text strings
            auxiliary_features: Optional tensor of auxiliary features (batch, auxiliary_dim)
            explicit_confounder_values: Optional list of dicts with explicit confounder values

        Returns:
            y0_logit: (batch, 1) - outcome prediction under control
            y1_logit: (batch, 1) - outcome prediction under treatment
            t_logit: (batch, 1) - treatment propensity logit
            final_common_layer: (batch, representation_dim) - shared representation
        """
        # Extract features from texts using CNN
        features = self.feature_extractor(texts)

        # Concatenate auxiliary features if provided
        if self.auxiliary_projection is not None and auxiliary_features is not None:
            aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
            features = torch.cat([features, aux_projected], dim=1)

        # Concatenate explicit confounder features if provided
        if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
            conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
            features = torch.cat([features, conf_features], dim=1)

        if self.model_type == "uplift":
            # UpliftNet returns: y0_logit, tau_logit, t_logit, final_common_layer
            y0_logit, tau_logit, t_logit, final_common_layer = self.net(features)
            # Reconstruct y1_logit = y0_logit + tau_logit
            y1_logit = y0_logit + tau_logit
        elif self.model_type == "rlearner":
            # RLearnerNet returns: m_logit, tau, t_logit, final_common_layer
            # Returns native outputs - caller handles interpretation
            m_logit, tau, t_logit, final_common_layer = self.net(features)
            # For forward() compatibility, return in same tuple format
            # But these are semantically different: m_logit is marginal, tau is effect
            return m_logit, tau, t_logit, final_common_layer
        elif self.model_type == "dr_moce":
            # DRMoCENet returns: mu0_logit, mu1_logit, tau, sigma2, t_logit, g, expert_means, phi
            mu0_logit, mu1_logit, tau, sigma2, t_logit, g, expert_means, phi = self.net(features)
            return mu0_logit, mu1_logit, t_logit, phi
        elif self.model_type == "traditional_logreg":
            # TraditionalLogRegNet in counterfactual mode returns: y0_logit, y1_logit, t_logit, phi
            y0_logit, y1_logit, t_logit, final_common_layer = self.net(features, treatment=None)
        else:
            # DragonNet returns: y0_logit, y1_logit, t_logit, final_common_layer
            y0_logit, y1_logit, t_logit, final_common_layer = self.net(features)

        return y0_logit, y1_logit, t_logit, final_common_layer

    def _outcome_loss(self, logit, target):
        """BCE for binary outcomes, MSE for continuous outcomes."""
        if self.outcome_type == "continuous":
            return F.mse_loss(logit, target)
        return F.binary_cross_entropy_with_logits(logit, target)

    def _outcome_activation(self, logit):
        """Sigmoid for binary outcomes, identity for continuous outcomes."""
        if self.outcome_type == "continuous":
            return logit
        return torch.sigmoid(logit)

    def train_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        beta_targreg: float = 0.1,
        gamma_rlearner: float = 1.0,
        gamma_dr: float = 1.0,
        label_smoothing: float = 0.0,
        stop_grad_propensity: bool = False,
        attention_entropy_weight: float = 0.0,
        clam_instance_weight: float = 0.5,
        contrastive_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys.
                   Optional 'auxiliary_features' for hybrid models.
                   Optional 'explicit_confounder_values' for explicit confounders.
            alpha_propensity: Weight for propensity loss
            beta_targreg: Weight for targeted regularization (dragonnet/uplift)
            gamma_rlearner: Weight for R-learner loss (rlearner only)
            gamma_dr: Weight for DR effect loss (dr_moce only)
            label_smoothing: Label smoothing factor (0 = no smoothing)
            stop_grad_propensity: If True, detach features before propensity loss
                so propensity optimization doesn't affect the feature extractor.
                This forces the representation to optimize for tau/outcome.
            attention_entropy_weight: Weight for attention entropy regularization.
                Higher = stronger penalty on diffuse attention. Only applies to
                gated_mil_hierarchical extractor.
            clam_instance_weight: Weight for CLAM instance-level loss. Only applies
                when clam_enabled=True and using gru_pool extractor.
            contrastive_weight: Weight for intra-batch contrastive loss. Only applies
                when contrastive_enabled=True.

        Returns:
            Dictionary with loss components and detached predictions
        """
        # Dispatch to specialized training step for dr_moce
        if self.model_type == "dr_moce":
            return self._train_step_dr_moce(
                batch, alpha_propensity, gamma_dr, label_smoothing,
                stop_grad_propensity, contrastive_weight
            )

        # Dispatch to specialized training step for rlearner
        if self.model_type == "rlearner":
            return self._train_step_rlearner(
                batch, alpha_propensity, gamma_rlearner, label_smoothing,
                stop_grad_propensity, attention_entropy_weight, clam_instance_weight,
                contrastive_weight
            )

        # Dispatch to specialized training step for uplift
        if self.model_type == "uplift":
            return self._train_step_uplift(
                batch, alpha_propensity, beta_targreg, label_smoothing,
                stop_grad_propensity, attention_entropy_weight, clam_instance_weight,
                contrastive_weight
            )

        # Dispatch to specialized training step for traditional_logreg
        if self.model_type == "traditional_logreg":
            return self._train_step_traditional_logreg(
                batch, alpha_propensity, label_smoothing,
                stop_grad_propensity, attention_entropy_weight, clam_instance_weight,
                contrastive_weight
            )

        # DragonNet (default)
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)
        auxiliary_features = batch.get('auxiliary_features', None)
        explicit_confounder_values = batch.get('explicit_confounder_values', None)
        extractor_input = self._get_extractor_input(batch, texts)

        # Apply label smoothing if enabled (skip outcome smoothing for continuous)
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            if self.outcome_type == "binary":
                outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
            else:
                outcomes_smooth = outcomes
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Extract features
        # Use forward_with_instances when CLAM is active to avoid double forward pass
        if self.clam_enabled and self.instance_head is not None and hasattr(self.feature_extractor, 'forward_with_instances'):
            features, _clam_chunk_embs, _clam_attn_weights = self.feature_extractor.forward_with_instances(extractor_input)
        else:
            features = self.feature_extractor(extractor_input)
            _clam_chunk_embs = None
            _clam_attn_weights = None

        # Intra-batch contrastive loss (on raw extractor features before concatenation)
        contrastive_loss = torch.tensor(0.0, device=self._device)
        if self.contrastive_enabled and self.contrastive_loss_module is not None and contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss_module(features, treatments, outcomes)

        # Concatenate auxiliary features if provided
        if self.auxiliary_projection is not None and auxiliary_features is not None:
            aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
            features = torch.cat([features, aux_projected], dim=1)

        # Concatenate explicit confounder features if provided
        if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
            conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
            features = torch.cat([features, conf_features], dim=1)

        # DragonNet: handle stop_grad_propensity
        if stop_grad_propensity:
            # Detach features for propensity computation to prevent propensity
            # from dominating the representation learning
            features_detached = features.detach()

            # Compute representation with detached features for propensity
            phi_detached = self.net.get_representation(features_detached)
            t_logit_for_loss = self.net.propensity_from_representation(phi_detached)

            # Compute full forward pass with regular features for outcome heads
            y0_logit, y1_logit, t_logit, phi = self.net(features)
        else:
            # Standard forward pass
            y0_logit, y1_logit, t_logit, phi = self.net(features)
            t_logit_for_loss = t_logit

        # Propensity loss - use t_logit_for_loss (detached features if stop_grad)
        propensity_loss = F.binary_cross_entropy_with_logits(
            t_logit_for_loss.squeeze(-1),
            treatments_smooth
        )

        # Outcome loss - factual outcome only
        factual_logit = torch.where(
            treatments.unsqueeze(1) > 0.5,
            y1_logit,
            y0_logit
        )

        outcome_loss = self._outcome_loss(
            factual_logit.squeeze(-1),
            outcomes_smooth
        )

        # Targeted regularization (R-loss)
        if beta_targreg > 0:
            with torch.no_grad():
                propensity = torch.sigmoid(t_logit).clamp(1e-3, 1 - 1e-3)
                H = (treatments.unsqueeze(1) / propensity) - \
                    ((1 - treatments.unsqueeze(1)) / (1 - propensity))

            factual_prob = self._outcome_activation(factual_logit)
            moment = torch.mean((outcomes.unsqueeze(1) - factual_prob) * H)
            targreg_loss = moment ** 2
        else:
            targreg_loss = torch.tensor(0.0, device=self._device)

        # CLAM instance-level loss (if enabled)
        instance_loss = torch.tensor(0.0, device=self._device)
        if self.clam_enabled and clam_instance_weight > 0 and self.instance_head is not None:
            # Use pre-computed chunk embeddings from forward_with_instances (avoids double forward pass)
            chunk_embs_list = _clam_chunk_embs
            attn_weights_list = _clam_attn_weights

            all_top_chunks = []
            expanded_treatments = []
            expanded_outcomes = []

            if chunk_embs_list is not None and attn_weights_list is not None:
                for i, (chunk_embs, attn_weights) in enumerate(zip(chunk_embs_list, attn_weights_list)):
                    if chunk_embs.size(0) == 0:
                        continue
                    B = min(self.clam_num_instances, chunk_embs.size(0))
                    top_indices = torch.topk(attn_weights, B).indices
                    top_chunks = chunk_embs[top_indices]  # (B, transformer_dim)

                    all_top_chunks.append(top_chunks)
                    expanded_treatments.extend([treatments[i]] * B)
                    expanded_outcomes.extend([outcomes[i]] * B)

            if all_top_chunks:
                stacked_chunks = torch.cat(all_top_chunks, dim=0)
                exp_treatments = torch.stack(expanded_treatments)
                exp_outcomes = torch.stack(expanded_outcomes)

                # Forward through instance head (DragonNet)
                inst_y0, inst_y1, inst_t, _ = self.instance_head(stacked_chunks)

                # Instance propensity loss
                instance_propensity_loss = F.binary_cross_entropy_with_logits(
                    inst_t.squeeze(-1), exp_treatments
                )

                # Instance outcome loss (factual only)
                inst_factual = torch.where(
                    exp_treatments.unsqueeze(1) > 0.5, inst_y1, inst_y0
                )
                instance_outcome_loss = self._outcome_loss(
                    inst_factual.squeeze(-1), exp_outcomes
                )

                instance_loss = instance_outcome_loss + alpha_propensity * instance_propensity_loss

        # Total loss
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            beta_targreg * targreg_loss +
            clam_instance_weight * instance_loss +
            contrastive_weight * contrastive_loss
        )

        result = {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'targreg_loss': targreg_loss.detach() if isinstance(targreg_loss, torch.Tensor) else targreg_loss,
            'y0_logit': y0_logit.detach(),
            'y1_logit': y1_logit.detach(),
            't_logit': t_logit.detach()
        }

        if self.clam_enabled:
            result['instance_loss'] = instance_loss.detach() if isinstance(instance_loss, torch.Tensor) else instance_loss

        if self.contrastive_enabled:
            result['contrastive_loss'] = contrastive_loss.detach()

        return result

    def _train_step_rlearner(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float,
        gamma_rlearner: float,
        label_smoothing: float,
        stop_grad_propensity: bool = False,
        attention_entropy_weight: float = 0.0,
        clam_instance_weight: float = 0.5,
        contrastive_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Perform R-learner training step with three-headed loss.

        R-learner loss decomposes into:
        1. Propensity loss: BCE for e(X) = P(T=1|X)
        2. Marginal outcome loss: BCE for m(X) = E[Y|X]
        3. R-loss: ((Y - m(X)) - tau(X) * (T - e(X)))^2

        The key insight is that e(X) and m(X) are DETACHED in the R-loss,
        so gradients from effect estimation flow only through tau(X).
        This provides stronger gradient signal for learning treatment
        effect modifiers from text.

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys
            alpha_propensity: Weight for propensity loss
            gamma_rlearner: Weight for R-learner loss
            label_smoothing: Label smoothing factor
            stop_grad_propensity: If True, detach features before propensity loss
            attention_entropy_weight: Weight for attention entropy regularization

        Returns:
            Dictionary with loss components and predictions
        """
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)
        auxiliary_features = batch.get('auxiliary_features', None)
        explicit_confounder_values = batch.get('explicit_confounder_values', None)
        extractor_input = self._get_extractor_input(batch, texts)

        # Apply label smoothing if enabled (skip outcome smoothing for continuous)
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            if self.outcome_type == "binary":
                outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
            else:
                outcomes_smooth = outcomes
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Check for dual extractor mode
        if self.rlearner_dual_extractors and self.effect_feature_extractor is not None:
            # DUAL EXTRACTOR MODE:
            # - Nuisance extractor (self.feature_extractor) -> e(X), m(X)
            # - Effect extractor (self.effect_feature_extractor) + effect_mlp -> τ(X)

            # Nuisance path: extract features for e(X) and m(X)
            # Use forward_with_instances when CLAM is active to avoid double forward pass
            if self.clam_enabled and self.instance_head is not None and hasattr(self.feature_extractor, 'forward_with_instances'):
                nuisance_features, _clam_chunk_embs, _clam_attn_weights = self.feature_extractor.forward_with_instances(extractor_input)
            else:
                nuisance_features = self.feature_extractor(extractor_input)
                _clam_chunk_embs = None
                _clam_attn_weights = None

            # Intra-batch contrastive loss (on nuisance extractor features)
            contrastive_loss = torch.tensor(0.0, device=self._device)
            if self.contrastive_enabled and self.contrastive_loss_module is not None and contrastive_weight > 0:
                contrastive_loss = self.contrastive_loss_module(nuisance_features, treatments, outcomes)

            # Compute attention entropy loss if enabled and extractor supports it
            entropy_loss = torch.tensor(0.0, device=self._device)
            if attention_entropy_weight > 0 and hasattr(self.feature_extractor, 'compute_attention_entropy_loss'):
                _, attention_info = self.feature_extractor.forward_with_attention(texts)
                entropy_loss = attention_info['attention_entropy']

            # Concatenate auxiliary features if provided
            if self.auxiliary_projection is not None and auxiliary_features is not None:
                aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
                nuisance_features = torch.cat([nuisance_features, aux_projected], dim=1)

            # Concatenate explicit confounder features if provided
            if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
                conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
                nuisance_features = torch.cat([nuisance_features, conf_features], dim=1)

            # Nuisance heads: propensity e(X) and marginal outcome m(X)
            # Note: We use the RLearnerNet's shared layers but only for nuisance functions
            m_logit, _, t_logit, phi = self.net(nuisance_features)

            # Effect path: extract features for τ(X)
            effect_features = self.effect_feature_extractor(extractor_input)

            # τ(X) from separate effect MLP
            tau = self.effect_mlp(effect_features)

            # Handle stop_grad_propensity (detach nuisance features for propensity loss)
            if stop_grad_propensity:
                nuisance_features_detached = nuisance_features.detach()
                phi_detached = self.net.get_representation(nuisance_features_detached)
                t_logit_for_loss = self.net.propensity_from_representation(phi_detached)

                propensity_loss = F.binary_cross_entropy_with_logits(
                    t_logit_for_loss.squeeze(-1),
                    treatments_smooth
                )
            else:
                propensity_loss = F.binary_cross_entropy_with_logits(
                    t_logit.squeeze(-1),
                    treatments_smooth
                )

            # Loss 2: Marginal outcome loss - BCE/MSE for m(X) = E[Y|X]
            outcome_loss = self._outcome_loss(
                m_logit.squeeze(-1),
                outcomes_smooth
            )

            # Loss 3: R-learner loss
            # CRITICAL: Nuisance functions are detached - gradients flow only through τ
            # In dual mode, τ comes from separate effect extractor + MLP
            e_X = torch.sigmoid(t_logit).detach().clamp(0.01, 0.99)
            m_X = self._outcome_activation(m_logit).detach()

            # Compute residuals
            Y_residual = outcomes - m_X.squeeze(-1)  # Y - m(X)
            T_residual = treatments - e_X.squeeze(-1)  # T - e(X)

            # R-loss: E[((Y - m(X)) - tau(X) * (T - e(X)))^2]
            r_loss = ((Y_residual - tau.squeeze(-1) * T_residual) ** 2).mean()

            # Set features variable for downstream use (CLAM, etc.)
            features = nuisance_features

        else:
            # STANDARD SINGLE EXTRACTOR MODE

            # Extract features
            # Use forward_with_instances when CLAM is active to avoid double forward pass
            if self.clam_enabled and self.instance_head is not None and hasattr(self.feature_extractor, 'forward_with_instances'):
                features, _clam_chunk_embs, _clam_attn_weights = self.feature_extractor.forward_with_instances(extractor_input)
            else:
                features = self.feature_extractor(extractor_input)
                _clam_chunk_embs = None
                _clam_attn_weights = None

            # Intra-batch contrastive loss (on raw extractor features)
            contrastive_loss = torch.tensor(0.0, device=self._device)
            if self.contrastive_enabled and self.contrastive_loss_module is not None and contrastive_weight > 0:
                contrastive_loss = self.contrastive_loss_module(features, treatments, outcomes)

            # Compute attention entropy loss if enabled and extractor supports it
            entropy_loss = torch.tensor(0.0, device=self._device)
            if attention_entropy_weight > 0 and hasattr(self.feature_extractor, 'compute_attention_entropy_loss'):
                # Use forward_with_attention to get entropy
                _, attention_info = self.feature_extractor.forward_with_attention(texts)
                entropy_loss = attention_info['attention_entropy']

            # Concatenate auxiliary features if provided
            if self.auxiliary_projection is not None and auxiliary_features is not None:
                aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
                features = torch.cat([features, aux_projected], dim=1)

            # Concatenate explicit confounder features if provided
            if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
                conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
                features = torch.cat([features, conf_features], dim=1)

            if stop_grad_propensity:
                # CRITICAL: Detach features for propensity to prevent propensity
                # from dominating the representation learning
                features_detached = features.detach()

                # Forward pass with regular features for outcome/tau
                m_logit, tau, t_logit, phi = self.net(features)

                # Re-compute propensity with detached features using helper methods
                phi_detached = self.net.get_representation(features_detached)
                t_logit_for_loss = self.net.propensity_from_representation(phi_detached)

                # Loss 1: Propensity loss with DETACHED features
                propensity_loss = F.binary_cross_entropy_with_logits(
                    t_logit_for_loss.squeeze(-1),
                    treatments_smooth
                )
            else:
                # Standard forward pass
                m_logit, tau, t_logit, phi = self.net(features)

                # Loss 1: Propensity loss - BCE for e(X)
                propensity_loss = F.binary_cross_entropy_with_logits(
                    t_logit.squeeze(-1),
                    treatments_smooth
                )

            # Loss 2: Marginal outcome loss - BCE/MSE for m(X) = E[Y|X]
            outcome_loss = self._outcome_loss(
                m_logit.squeeze(-1),
                outcomes_smooth
            )

            # Loss 3: R-learner loss
            # CRITICAL: Detach nuisance functions so gradients flow only through tau
            e_X = torch.sigmoid(t_logit).detach().clamp(0.01, 0.99)
            m_X = self._outcome_activation(m_logit).detach()

            # Compute residuals
            Y_residual = outcomes - m_X.squeeze(-1)  # Y - m(X)
            T_residual = treatments - e_X.squeeze(-1)  # T - e(X)

            # R-loss: E[((Y - m(X)) - tau(X) * (T - e(X)))^2]
            r_loss = ((Y_residual - tau.squeeze(-1) * T_residual) ** 2).mean()

        # CLAM instance-level loss (if enabled)
        instance_loss = torch.tensor(0.0, device=self._device)
        if self.clam_enabled and clam_instance_weight > 0 and self.instance_head is not None:
            # Use pre-computed chunk embeddings from forward_with_instances (avoids double forward pass)
            chunk_embs_list = _clam_chunk_embs
            attn_weights_list = _clam_attn_weights

            all_top_chunks = []
            expanded_treatments = []
            expanded_outcomes = []

            if chunk_embs_list is not None and attn_weights_list is not None:
                for i, (chunk_embs, attn_weights) in enumerate(zip(chunk_embs_list, attn_weights_list)):
                    if chunk_embs.size(0) == 0:
                        continue
                    B = min(self.clam_num_instances, chunk_embs.size(0))
                    top_indices = torch.topk(attn_weights, B).indices
                    top_chunks = chunk_embs[top_indices]  # (B, transformer_dim)

                    all_top_chunks.append(top_chunks)
                    expanded_treatments.extend([treatments[i]] * B)
                    expanded_outcomes.extend([outcomes[i]] * B)

            if all_top_chunks:
                stacked_chunks = torch.cat(all_top_chunks, dim=0)
                exp_treatments = torch.stack(expanded_treatments)
                exp_outcomes = torch.stack(expanded_outcomes)

                # Forward through instance head (R-Learner)
                # RLearnerNet returns: m_logit, tau, t_logit, phi
                inst_m, inst_tau, inst_t, _ = self.instance_head(stacked_chunks)

                # Instance propensity loss
                instance_propensity_loss = F.binary_cross_entropy_with_logits(
                    inst_t.squeeze(-1), exp_treatments
                )

                # Instance marginal outcome loss
                instance_outcome_loss = self._outcome_loss(
                    inst_m.squeeze(-1), exp_outcomes
                )

                # Instance R-loss (with detached nuisance functions)
                inst_e = torch.sigmoid(inst_t).detach().clamp(0.01, 0.99)
                inst_m_prob = self._outcome_activation(inst_m).detach()
                inst_Y_residual = exp_outcomes - inst_m_prob.squeeze(-1)
                inst_T_residual = exp_treatments - inst_e.squeeze(-1)
                instance_r_loss = ((inst_Y_residual - inst_tau.squeeze(-1) * inst_T_residual) ** 2).mean()

                instance_loss = (
                    instance_outcome_loss +
                    alpha_propensity * instance_propensity_loss +
                    gamma_rlearner * instance_r_loss
                )

        # Total loss
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            gamma_rlearner * r_loss +
            attention_entropy_weight * entropy_loss +
            clam_instance_weight * instance_loss +
            contrastive_weight * contrastive_loss
        )

        # Derive y0/y1 for backward-compatible metrics
        # From: m = e*y1 + (1-e)*y0 and tau = y1 - y0
        # Solving: y0 = m - e*tau, y1 = m + (1-e)*tau
        with torch.no_grad():
            m_prob = self._outcome_activation(m_logit)
            prop = torch.sigmoid(t_logit)
            tau_val = tau
            y0_logit_approx = m_logit - prop * tau_val
            y1_logit_approx = m_logit + (1 - prop) * tau_val

        result = {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'r_loss': r_loss.detach(),
            'targreg_loss': r_loss.detach(),  # Alias for compatibility
            'm_logit': m_logit.detach(),
            'tau': tau.detach(),
            't_logit': t_logit.detach(),
            # Backward compatible outputs (derived)
            'y0_logit': y0_logit_approx.detach(),
            'y1_logit': y1_logit_approx.detach()
        }

        # Add entropy loss if computed
        if attention_entropy_weight > 0:
            result['entropy_loss'] = entropy_loss.detach() if isinstance(entropy_loss, torch.Tensor) else entropy_loss

        # Add contrastive loss if enabled
        if self.contrastive_enabled:
            result['contrastive_loss'] = contrastive_loss.detach()

        # Add instance loss if CLAM enabled
        if self.clam_enabled:
            result['instance_loss'] = instance_loss.detach() if isinstance(instance_loss, torch.Tensor) else instance_loss

        return result

    def _train_step_uplift(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float,
        beta_targreg: float,
        label_smoothing: float,
        stop_grad_propensity: bool = False,
        attention_entropy_weight: float = 0.0,
        clam_instance_weight: float = 0.5,
        contrastive_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Perform Uplift training step with optional dual extractor mode.

        UpliftNet parametrizes outcomes as:
        - Y0(X): baseline outcome under control
        - τ(X): treatment effect
        - Y1(X) = Y0(X) + τ(X)

        In dual extractor mode:
        - Nuisance extractor: e(X), Y0(X)
        - Effect extractor + MLP: τ(X)

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys
            alpha_propensity: Weight for propensity loss
            beta_targreg: Weight for targeted regularization
            label_smoothing: Label smoothing factor
            stop_grad_propensity: If True, detach features before propensity loss
            attention_entropy_weight: Weight for attention entropy regularization
            clam_instance_weight: Weight for CLAM instance-level loss

        Returns:
            Dictionary with loss components and predictions
        """
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)
        auxiliary_features = batch.get('auxiliary_features', None)
        explicit_confounder_values = batch.get('explicit_confounder_values', None)
        extractor_input = self._get_extractor_input(batch, texts)

        # Apply label smoothing if enabled (skip outcome smoothing for continuous)
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            if self.outcome_type == "binary":
                outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
            else:
                outcomes_smooth = outcomes
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Check for dual extractor mode
        if self.uplift_dual_extractors and self.effect_feature_extractor is not None:
            # DUAL EXTRACTOR MODE:
            # - Nuisance extractor (self.feature_extractor) -> e(X), Y0(X)
            # - Effect extractor (self.effect_feature_extractor) + effect_mlp -> τ(X)

            # Nuisance path: extract features for e(X) and Y0(X)
            # Use forward_with_instances when CLAM is active to avoid double forward pass
            if self.clam_enabled and self.instance_head is not None and hasattr(self.feature_extractor, 'forward_with_instances'):
                nuisance_features, _clam_chunk_embs, _clam_attn_weights = self.feature_extractor.forward_with_instances(extractor_input)
            else:
                nuisance_features = self.feature_extractor(extractor_input)
                _clam_chunk_embs = None
                _clam_attn_weights = None

            # Intra-batch contrastive loss (on nuisance extractor features)
            contrastive_loss = torch.tensor(0.0, device=self._device)
            if self.contrastive_enabled and self.contrastive_loss_module is not None and contrastive_weight > 0:
                contrastive_loss = self.contrastive_loss_module(nuisance_features, treatments, outcomes)

            # Compute attention entropy loss if enabled and extractor supports it
            entropy_loss = torch.tensor(0.0, device=self._device)
            if attention_entropy_weight > 0 and hasattr(self.feature_extractor, 'compute_attention_entropy_loss'):
                _, attention_info = self.feature_extractor.forward_with_attention(texts)
                entropy_loss = attention_info['attention_entropy']

            # Concatenate auxiliary features if provided
            if self.auxiliary_projection is not None and auxiliary_features is not None:
                aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
                nuisance_features = torch.cat([nuisance_features, aux_projected], dim=1)

            # Concatenate explicit confounder features if provided
            if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
                conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
                nuisance_features = torch.cat([nuisance_features, conf_features], dim=1)

            # Nuisance heads: propensity e(X) and baseline outcome Y0(X)
            # UpliftNet returns: y0_logit, tau_logit, t_logit, phi
            # In dual mode, we ignore tau_logit from UpliftNet and use effect_mlp instead
            y0_logit, _, t_logit, phi = self.net(nuisance_features)

            # Effect path: extract features for τ(X)
            effect_features = self.effect_feature_extractor(extractor_input)

            # τ(X) from separate effect MLP
            tau_logit = self.effect_mlp(effect_features)

            # Y1 = Y0 + τ
            y1_logit = y0_logit + tau_logit

            # Handle stop_grad_propensity (detach nuisance features for propensity loss)
            if stop_grad_propensity:
                nuisance_features_detached = nuisance_features.detach()
                phi_detached = self.net.get_representation(nuisance_features_detached)
                t_logit_for_loss = self.net.propensity_from_representation(phi_detached)

                propensity_loss = F.binary_cross_entropy_with_logits(
                    t_logit_for_loss.squeeze(-1),
                    treatments_smooth
                )
            else:
                propensity_loss = F.binary_cross_entropy_with_logits(
                    t_logit.squeeze(-1),
                    treatments_smooth
                )

            # Set features variable for downstream use (CLAM, etc.)
            features = nuisance_features

        else:
            # STANDARD SINGLE EXTRACTOR MODE

            # Extract features
            # Use forward_with_instances when CLAM is active to avoid double forward pass
            if self.clam_enabled and self.instance_head is not None and hasattr(self.feature_extractor, 'forward_with_instances'):
                features, _clam_chunk_embs, _clam_attn_weights = self.feature_extractor.forward_with_instances(extractor_input)
            else:
                features = self.feature_extractor(extractor_input)
                _clam_chunk_embs = None
                _clam_attn_weights = None

            # Intra-batch contrastive loss (on raw extractor features)
            contrastive_loss = torch.tensor(0.0, device=self._device)
            if self.contrastive_enabled and self.contrastive_loss_module is not None and contrastive_weight > 0:
                contrastive_loss = self.contrastive_loss_module(features, treatments, outcomes)

            # Compute attention entropy loss if enabled and extractor supports it
            entropy_loss = torch.tensor(0.0, device=self._device)
            if attention_entropy_weight > 0 and hasattr(self.feature_extractor, 'compute_attention_entropy_loss'):
                _, attention_info = self.feature_extractor.forward_with_attention(texts)
                entropy_loss = attention_info['attention_entropy']

            # Concatenate auxiliary features if provided
            if self.auxiliary_projection is not None and auxiliary_features is not None:
                aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
                features = torch.cat([features, aux_projected], dim=1)

            # Concatenate explicit confounder features if provided
            if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
                conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
                features = torch.cat([features, conf_features], dim=1)

            if stop_grad_propensity:
                # Detach features for propensity to prevent propensity from dominating
                features_detached = features.detach()

                # Forward pass with regular features for outcome/tau
                y0_logit, tau_logit, t_logit, phi = self.net(features)
                y1_logit = y0_logit + tau_logit

                # Re-compute propensity with detached features using helper methods
                phi_detached = self.net.get_representation(features_detached)
                t_logit_for_loss = self.net.propensity_from_representation(phi_detached)

                propensity_loss = F.binary_cross_entropy_with_logits(
                    t_logit_for_loss.squeeze(-1),
                    treatments_smooth
                )
            else:
                # Standard forward pass
                y0_logit, tau_logit, t_logit, phi = self.net(features)
                y1_logit = y0_logit + tau_logit

                propensity_loss = F.binary_cross_entropy_with_logits(
                    t_logit.squeeze(-1),
                    treatments_smooth
                )

        # Outcome loss - factual outcome only
        factual_logit = torch.where(
            treatments.unsqueeze(1) > 0.5,
            y1_logit,
            y0_logit
        )

        outcome_loss = self._outcome_loss(
            factual_logit.squeeze(-1),
            outcomes_smooth
        )

        # Targeted regularization
        if beta_targreg > 0:
            with torch.no_grad():
                propensity = torch.sigmoid(t_logit).clamp(1e-3, 1 - 1e-3)
                H = (treatments.unsqueeze(1) / propensity) - \
                    ((1 - treatments.unsqueeze(1)) / (1 - propensity))

            factual_prob = self._outcome_activation(factual_logit)
            moment = torch.mean((outcomes.unsqueeze(1) - factual_prob) * H)
            targreg_loss = moment ** 2
        else:
            targreg_loss = torch.tensor(0.0, device=self._device)

        # CLAM instance-level loss (if enabled)
        instance_loss = torch.tensor(0.0, device=self._device)
        if self.clam_enabled and clam_instance_weight > 0 and self.instance_head is not None:
            # Use pre-computed chunk embeddings from forward_with_instances (avoids double forward pass)
            chunk_embs_list = _clam_chunk_embs
            attn_weights_list = _clam_attn_weights

            all_top_chunks = []
            expanded_treatments = []
            expanded_outcomes = []

            if chunk_embs_list is not None and attn_weights_list is not None:
                for i, (chunk_embs, attn_weights) in enumerate(zip(chunk_embs_list, attn_weights_list)):
                    if chunk_embs.size(0) == 0:
                        continue
                    B = min(self.clam_num_instances, chunk_embs.size(0))
                    top_indices = torch.topk(attn_weights, B).indices
                    top_chunks = chunk_embs[top_indices]  # (B, transformer_dim)

                    all_top_chunks.append(top_chunks)
                    expanded_treatments.extend([treatments[i]] * B)
                    expanded_outcomes.extend([outcomes[i]] * B)

            if all_top_chunks:
                stacked_chunks = torch.cat(all_top_chunks, dim=0)
                exp_treatments = torch.stack(expanded_treatments)
                exp_outcomes = torch.stack(expanded_outcomes)

                # Forward through instance head (UpliftNet)
                # UpliftNet returns: y0_logit, tau_logit, t_logit, phi
                inst_y0, inst_tau, inst_t, _ = self.instance_head(stacked_chunks)
                inst_y1 = inst_y0 + inst_tau

                # Instance propensity loss
                instance_propensity_loss = F.binary_cross_entropy_with_logits(
                    inst_t.squeeze(-1), exp_treatments
                )

                # Instance outcome loss (factual only)
                inst_factual = torch.where(
                    exp_treatments.unsqueeze(1) > 0.5, inst_y1, inst_y0
                )
                instance_outcome_loss = self._outcome_loss(
                    inst_factual.squeeze(-1), exp_outcomes
                )

                instance_loss = instance_outcome_loss + alpha_propensity * instance_propensity_loss

        # Total loss
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            beta_targreg * targreg_loss +
            attention_entropy_weight * entropy_loss +
            clam_instance_weight * instance_loss +
            contrastive_weight * contrastive_loss
        )

        result = {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'targreg_loss': targreg_loss.detach() if isinstance(targreg_loss, torch.Tensor) else targreg_loss,
            'y0_logit': y0_logit.detach(),
            'y1_logit': y1_logit.detach(),
            't_logit': t_logit.detach(),
            'tau_logit': tau_logit.detach(),
        }

        # Add entropy loss if computed
        if attention_entropy_weight > 0:
            result['entropy_loss'] = entropy_loss.detach() if isinstance(entropy_loss, torch.Tensor) else entropy_loss

        # Add contrastive loss if enabled
        if self.contrastive_enabled:
            result['contrastive_loss'] = contrastive_loss.detach()

        # Add instance loss if CLAM enabled
        if self.clam_enabled:
            result['instance_loss'] = instance_loss.detach() if isinstance(instance_loss, torch.Tensor) else instance_loss

        return result

    def _train_step_traditional_logreg(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float,
        label_smoothing: float,
        stop_grad_propensity: bool = False,
        attention_entropy_weight: float = 0.0,
        clam_instance_weight: float = 0.5,
        contrastive_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Perform training step for traditional logistic regression head.

        Traditional approach:
        1. Propensity loss: BCE for P(T|X)
        2. Outcome loss: BCE for P(Y|X, T_observed) - treatment is concatenated as feature

        This is simpler than DragonNet since we only predict the factual outcome
        conditioned on observed treatment, not counterfactuals.

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys
            alpha_propensity: Weight for propensity loss
            label_smoothing: Label smoothing factor
            stop_grad_propensity: If True, detach features before propensity loss
            attention_entropy_weight: Weight for attention entropy regularization
            clam_instance_weight: Weight for CLAM instance-level loss

        Returns:
            Dictionary with loss components and predictions
        """
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)
        auxiliary_features = batch.get('auxiliary_features', None)
        explicit_confounder_values = batch.get('explicit_confounder_values', None)
        extractor_input = self._get_extractor_input(batch, texts)

        # Apply label smoothing if enabled (skip outcome smoothing for continuous)
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            if self.outcome_type == "binary":
                outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
            else:
                outcomes_smooth = outcomes
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Extract features
        # Use forward_with_instances when CLAM is active to avoid double forward pass
        if self.clam_enabled and self.instance_head is not None and hasattr(self.feature_extractor, 'forward_with_instances'):
            features, _clam_chunk_embs, _clam_attn_weights = self.feature_extractor.forward_with_instances(extractor_input)
        else:
            features = self.feature_extractor(extractor_input)
            _clam_chunk_embs = None
            _clam_attn_weights = None

        # Intra-batch contrastive loss (on raw extractor features)
        contrastive_loss = torch.tensor(0.0, device=self._device)
        if self.contrastive_enabled and self.contrastive_loss_module is not None and contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss_module(features, treatments, outcomes)

        # Compute attention entropy loss if enabled and extractor supports it
        entropy_loss = torch.tensor(0.0, device=self._device)
        if attention_entropy_weight > 0 and hasattr(self.feature_extractor, 'compute_attention_entropy_loss'):
            _, attention_info = self.feature_extractor.forward_with_attention(texts)
            entropy_loss = attention_info['attention_entropy']

        # Concatenate auxiliary features if provided
        if self.auxiliary_projection is not None and auxiliary_features is not None:
            aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
            features = torch.cat([features, aux_projected], dim=1)

        # Concatenate explicit confounder features if provided
        if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
            conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
            features = torch.cat([features, conf_features], dim=1)

        if stop_grad_propensity:
            # Detach features for propensity to prevent propensity from dominating
            features_detached = features.detach()
            phi_detached = self.net.get_representation(features_detached)
            t_logit_for_loss = self.net.propensity_from_representation(phi_detached)

            # Forward pass with regular features for outcome
            y_logit, t_logit, phi = self.net(features, treatment=treatments)
        else:
            # Standard forward pass with observed treatment
            y_logit, t_logit, phi = self.net(features, treatment=treatments)
            t_logit_for_loss = t_logit

        # Propensity loss
        propensity_loss = F.binary_cross_entropy_with_logits(
            t_logit_for_loss.squeeze(-1),
            treatments_smooth
        )

        # Outcome loss - factual outcome conditioned on observed treatment
        outcome_loss = self._outcome_loss(
            y_logit.squeeze(-1),
            outcomes_smooth
        )

        # Get counterfactual predictions for logging/metrics
        with torch.no_grad():
            y0_logit, y1_logit, _, _ = self.net(features, treatment=None)

        # CLAM instance-level loss (if enabled)
        instance_loss = torch.tensor(0.0, device=self._device)
        if self.clam_enabled and clam_instance_weight > 0 and self.instance_head is not None:
            # Use pre-computed chunk embeddings from forward_with_instances (avoids double forward pass)
            chunk_embs_list = _clam_chunk_embs
            attn_weights_list = _clam_attn_weights

            all_top_chunks = []
            expanded_treatments = []
            expanded_outcomes = []

            if chunk_embs_list is not None and attn_weights_list is not None:
                for i, (chunk_embs, attn_weights) in enumerate(zip(chunk_embs_list, attn_weights_list)):
                    if chunk_embs.size(0) == 0:
                        continue
                    B = min(self.clam_num_instances, chunk_embs.size(0))
                    top_indices = torch.topk(attn_weights, B).indices
                    top_chunks = chunk_embs[top_indices]

                    all_top_chunks.append(top_chunks)
                    expanded_treatments.extend([treatments[i]] * B)
                    expanded_outcomes.extend([outcomes[i]] * B)

            if all_top_chunks:
                stacked_chunks = torch.cat(all_top_chunks, dim=0)
                exp_treatments = torch.stack(expanded_treatments)
                exp_outcomes = torch.stack(expanded_outcomes)

                # Forward through instance head with observed treatment
                inst_y, inst_t, _ = self.instance_head(stacked_chunks, treatment=exp_treatments)

                # Instance propensity loss
                instance_propensity_loss = F.binary_cross_entropy_with_logits(
                    inst_t.squeeze(-1), exp_treatments
                )

                # Instance outcome loss
                instance_outcome_loss = self._outcome_loss(
                    inst_y.squeeze(-1), exp_outcomes
                )

                instance_loss = instance_outcome_loss + alpha_propensity * instance_propensity_loss

        # Total loss (no targeted regularization for traditional approach)
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            attention_entropy_weight * entropy_loss +
            clam_instance_weight * instance_loss +
            contrastive_weight * contrastive_loss
        )

        result = {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'targreg_loss': torch.tensor(0.0, device=self._device),  # No targreg for traditional
            'y0_logit': y0_logit.detach(),
            'y1_logit': y1_logit.detach(),
            't_logit': t_logit.detach()
        }

        if attention_entropy_weight > 0:
            result['entropy_loss'] = entropy_loss.detach() if isinstance(entropy_loss, torch.Tensor) else entropy_loss

        if self.contrastive_enabled:
            result['contrastive_loss'] = contrastive_loss.detach()

        if self.clam_enabled:
            result['instance_loss'] = instance_loss.detach() if isinstance(instance_loss, torch.Tensor) else instance_loss

        return result

    def _train_step_dr_moce(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float,
        gamma_dr: float,
        label_smoothing: float,
        stop_grad_propensity: bool = False,
        contrastive_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Perform DR-MoCE training step.

        Loss = L_nuisance + gamma_dr * L_DR + lambda_het * L_het + lambda_bal * L_bal

        where:
          L_nuisance = BCE(e(X), T) + BCE(mu0(X), Y|T=0) + BCE(mu1(X), Y|T=1)
          L_DR = heteroscedastic Gaussian NLL of Gamma (DR pseudo-outcome)
          L_het = -Var_k[mean_k(X)]  (encourage expert specialization)
          L_bal = KL(avg routing || Uniform(K))  (load balancing)

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys
            alpha_propensity: Weight for propensity loss
            gamma_dr: Weight for DR effect loss
            label_smoothing: Label smoothing factor
            stop_grad_propensity: If True, detach features before propensity

        Returns:
            Dictionary with loss components and predictions
        """
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)
        explicit_confounder_values = batch.get('explicit_confounder_values', None)
        extractor_input = self._get_extractor_input(batch, texts)

        # Apply label smoothing if enabled (skip outcome smoothing for continuous)
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            if self.outcome_type == "binary":
                outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
            else:
                outcomes_smooth = outcomes
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Extract features
        features = self.feature_extractor(extractor_input)

        # Intra-batch contrastive loss (on raw extractor features)
        contrastive_loss = torch.tensor(0.0, device=self._device)
        if self.contrastive_enabled and self.contrastive_loss_module is not None and contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss_module(features, treatments, outcomes)

        # Concatenate explicit confounder features if provided
        if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
            conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
            features = torch.cat([features, conf_features], dim=1)

        # Full forward pass through DR-MoCE
        mu0_logit, mu1_logit, tau, sigma2, t_logit, g, expert_means, phi = self.net(features)

        # --- Nuisance losses ---
        # Propensity loss
        if stop_grad_propensity:
            features_detached = features.detach()
            phi_detached = self.net.get_representation(features_detached)
            t_logit_for_loss = self.net.propensity_from_representation(phi_detached)
        else:
            t_logit_for_loss = t_logit

        propensity_loss = F.binary_cross_entropy_with_logits(
            t_logit_for_loss.squeeze(-1),
            treatments_smooth
        )

        # Factual outcome loss for mu0, mu1 (only on observed treatment arm)
        treated_mask = (treatments > 0.5)
        control_mask = ~treated_mask

        outcome_loss = torch.tensor(0.0, device=self._device)
        n_terms = 0
        if control_mask.any():
            outcome_loss = outcome_loss + self._outcome_loss(
                mu0_logit[control_mask].squeeze(-1),
                outcomes_smooth[control_mask]
            )
            n_terms += 1
        if treated_mask.any():
            outcome_loss = outcome_loss + self._outcome_loss(
                mu1_logit[treated_mask].squeeze(-1),
                outcomes_smooth[treated_mask]
            )
            n_terms += 1
        if n_terms > 1:
            outcome_loss = outcome_loss / n_terms

        # --- DR pseudo-outcome with optional cross-fitting ---
        with torch.no_grad():
            e = torch.sigmoid(t_logit).squeeze(-1)
            mu0_prob = self._outcome_activation(mu0_logit).squeeze(-1)
            mu1_prob = self._outcome_activation(mu1_logit).squeeze(-1)

            # Use prediction buffer for cross-fitting if available
            if (self.dr_moce_buffer is not None
                    and self.dr_moce_buffer.is_ready()
                    and self.training):
                # Push current predictions to buffer
                self.dr_moce_buffer.push(
                    e.detach().cpu(),
                    mu0_prob.detach().cpu(),
                    mu1_prob.detach().cpu(),
                    outcomes.detach().cpu(),
                    treatments.detach().cpu()
                )

            # AIPW pseudo-outcome (using detached current-batch nuisance)
            Gamma = compute_dr_pseudo_outcome(
                Y=outcomes,
                T=treatments,
                e=e,
                mu0=mu0_prob,
                mu1=mu1_prob,
                clip=self.dr_moce_propensity_clip
            )

        # --- DR effect loss (heteroscedastic NLL) ---
        sigma2_clamped = sigma2.clamp(min=1e-6)
        dr_loss = (
            0.5 * (Gamma - tau) ** 2 / sigma2_clamped
            + 0.5 * torch.log(sigma2_clamped)
        ).mean()

        # --- Regularization losses ---
        # Expert heterogeneity: encourage different experts to learn different effects
        # expert_means shape: (batch, K) - variance across experts for each sample
        het_loss = -expert_means.var(dim=1).mean()

        # Load balancing: routing should be roughly uniform on average
        avg_routing = g.mean(dim=0)  # (K,)
        uniform = torch.ones_like(avg_routing) / avg_routing.size(0)
        # KL(avg_routing || uniform)
        bal_loss = (avg_routing * (avg_routing / uniform + 1e-8).log()).sum()

        # --- Total loss ---
        total_loss = (
            outcome_loss
            + alpha_propensity * propensity_loss
            + gamma_dr * dr_loss
            + self.dr_moce_het_weight * het_loss
            + self.dr_moce_balance_weight * bal_loss
            + contrastive_weight * contrastive_loss
        )

        # Push to buffer after computing loss (for next iteration's cross-fitting)
        if (self.dr_moce_buffer is not None
                and not self.dr_moce_buffer.is_ready()
                and self.training):
            with torch.no_grad():
                self.dr_moce_buffer.push(
                    e.detach().cpu(),
                    mu0_prob.detach().cpu(),
                    mu1_prob.detach().cpu(),
                    outcomes.detach().cpu(),
                    treatments.detach().cpu()
                )

        # Derive y0/y1 logits for backward-compatible metrics
        result = {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'dr_loss': dr_loss.detach(),
            'het_loss': het_loss.detach(),
            'bal_loss': bal_loss.detach(),
            'tau': tau.detach(),
            'sigma2': sigma2.detach(),
            'y0_logit': mu0_logit.detach(),
            'y1_logit': mu1_logit.detach(),
            't_logit': t_logit.detach(),
            'routing_weights': g.detach(),
        }

        if self.contrastive_enabled:
            result['contrastive_loss'] = contrastive_loss.detach()

        return result

    def predict(
        self,
        texts_or_batch,
        auxiliary_features: Optional[torch.Tensor] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for inference.

        Args:
            texts_or_batch: List of text strings or preprocessed batch dict from DataLoader
            auxiliary_features: Optional tensor of auxiliary features (batch, auxiliary_dim)
            explicit_confounder_values: Optional list of dicts with explicit confounder values.
                If texts_or_batch is a batch dict, confounder values are extracted from it
                automatically (unless explicitly overridden).

        Returns:
            Dictionary with prediction outputs
        """
        with torch.no_grad():
            if isinstance(texts_or_batch, dict):
                texts = texts_or_batch['texts']
                extractor_input = self._get_extractor_input(texts_or_batch, texts)
                if explicit_confounder_values is None:
                    explicit_confounder_values = texts_or_batch.get('explicit_confounder_values', None)
            else:
                texts = texts_or_batch
                extractor_input = texts

            features = self.feature_extractor(extractor_input)

            # Concatenate auxiliary features if provided
            if self.auxiliary_projection is not None and auxiliary_features is not None:
                aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
                features = torch.cat([features, aux_projected], dim=1)

            # Concatenate explicit confounder features if provided
            if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
                conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
                features = torch.cat([features, conf_features], dim=1)

            if self.model_type == "dr_moce":
                # DRMoCENet returns: mu0_logit, mu1_logit, tau, sigma2, t_logit, g, expert_means, phi
                mu0_logit, mu1_logit, tau, sigma2, t_logit, g, expert_means, phi = self.net(features)

                sigma = torch.sqrt(sigma2.clamp(min=1e-6))
                y0_prob = self._outcome_activation(mu0_logit).squeeze(-1)
                y1_prob = self._outcome_activation(mu1_logit).squeeze(-1)
                propensity = torch.sigmoid(t_logit).squeeze(-1)

                return {
                    'tau_pred': tau,
                    'tau_lower': tau - 1.96 * sigma,
                    'tau_upper': tau + 1.96 * sigma,
                    'tau_std': sigma,
                    'y0_prob': y0_prob,
                    'y1_prob': y1_prob,
                    'propensity': propensity,
                    'routing_weights': g,
                    'y0_logit': mu0_logit.squeeze(-1),
                    'y1_logit': mu1_logit.squeeze(-1),
                    't_logit': t_logit.squeeze(-1),
                    'final_common_layer': phi,
                }

            elif self.model_type == "uplift":
                # Check for dual extractor mode
                if self.uplift_dual_extractors and self.effect_feature_extractor is not None:
                    # DUAL EXTRACTOR MODE:
                    # - Nuisance from main extractor + UpliftNet -> e(X), Y0(X)
                    # - τ from effect extractor + effect_mlp
                    y0_logit, _, t_logit, final_common_layer = self.net(features)

                    # Get τ from effect extractor
                    effect_features = self.effect_feature_extractor(extractor_input)
                    tau_logit = self.effect_mlp(effect_features)

                    y1_logit = y0_logit + tau_logit
                    tau_pred = tau_logit.squeeze(-1)
                else:
                    # STANDARD MODE
                    y0_logit, tau_logit, t_logit, final_common_layer = self.net(features)
                    y1_logit = y0_logit + tau_logit
                    tau_pred = tau_logit.squeeze(-1)
            elif self.model_type == "rlearner":
                # Check for dual extractor mode
                if self.rlearner_dual_extractors and self.effect_feature_extractor is not None:
                    # DUAL EXTRACTOR MODE:
                    # - Nuisance from main extractor + RLearnerNet
                    # - τ from effect extractor + effect_mlp
                    m_logit, _, t_logit, final_common_layer = self.net(features)

                    # Get τ from effect extractor
                    effect_features = self.effect_feature_extractor(extractor_input)
                    tau = self.effect_mlp(effect_features)

                    m_prob = self._outcome_activation(m_logit).squeeze(-1)  # E[Y|X]
                    tau_val = tau.squeeze(-1)  # τ(X)
                    prop = torch.sigmoid(t_logit).squeeze(-1)  # e(X)

                else:
                    # STANDARD MODE:
                    # RLearnerNet returns: m_logit, tau, t_logit, final_common_layer
                    m_logit, tau, t_logit, final_common_layer = self.net(features)

                    m_prob = self._outcome_activation(m_logit).squeeze(-1)  # E[Y|X]
                    tau_val = tau.squeeze(-1)  # τ(X)
                    prop = torch.sigmoid(t_logit).squeeze(-1)  # e(X)

                # Derive Y0/Y1 from m and τ for backward compatibility:
                # From: m = e*y1 + (1-e)*y0 and tau = y1 - y0
                # Solving: y0 = m - e*tau, y1 = m + (1-e)*tau
                y0_prob = (m_prob - prop * tau_val)
                y1_prob = (m_prob + (1 - prop) * tau_val)
                if self.outcome_type == "binary":
                    y0_prob = y0_prob.clamp(0, 1)
                    y1_prob = y1_prob.clamp(0, 1)

                return {
                    'y0_prob': y0_prob,
                    'y1_prob': y1_prob,
                    'propensity': prop,
                    'm_prob': m_prob,  # Native E[Y|X]
                    'tau_pred': tau_val,  # Native τ(X)
                    't_logit': t_logit.squeeze(-1),
                    'm_logit': m_logit.squeeze(-1),
                    'final_common_layer': final_common_layer,
                    # Approximate logits for compatibility
                    'y0_logit': torch.logit(y0_prob.clamp(1e-6, 1 - 1e-6)) if self.outcome_type == "binary" else y0_prob,
                    'y1_logit': torch.logit(y1_prob.clamp(1e-6, 1 - 1e-6)) if self.outcome_type == "binary" else y1_prob,
                }
            elif self.model_type == "traditional_logreg":
                # TraditionalLogRegNet in counterfactual mode returns: y0_logit, y1_logit, t_logit, phi
                y0_logit, y1_logit, t_logit, final_common_layer = self.net(features, treatment=None)
                tau_pred = (y1_logit - y0_logit).squeeze(-1)
            else:
                y0_logit, y1_logit, t_logit, final_common_layer = self.net(features)
                tau_pred = (y1_logit - y0_logit).squeeze(-1)

            # Convert to probabilities (or identity for continuous)
            y0_prob = self._outcome_activation(y0_logit).squeeze(-1)
            y1_prob = self._outcome_activation(y1_logit).squeeze(-1)
            propensity = torch.sigmoid(t_logit).squeeze(-1)

            return {
                'y0_prob': y0_prob,
                'y1_prob': y1_prob,
                'propensity': propensity,
                'y0_logit': y0_logit.squeeze(-1),
                'y1_logit': y1_logit.squeeze(-1),
                't_logit': t_logit.squeeze(-1),
                'final_common_layer': final_common_layer,
                'tau_pred': tau_pred
            }

    def get_features(
        self,
        texts_or_batch,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None
    ) -> torch.Tensor:
        """
        Extract feature representations from texts.

        Args:
            texts_or_batch: List of text strings or preprocessed batch dict from DataLoader
            explicit_confounder_values: Optional list of dicts with explicit confounder values

        Returns:
            Feature tensor: (batch, output_dim)
        """
        with torch.no_grad():
            if isinstance(texts_or_batch, dict):
                texts = texts_or_batch['texts']
                extractor_input = self._get_extractor_input(texts_or_batch, texts)
                if explicit_confounder_values is None:
                    explicit_confounder_values = texts_or_batch.get('explicit_confounder_values', None)
            else:
                extractor_input = texts_or_batch

            features = self.feature_extractor(extractor_input)

            # Concatenate explicit confounder features if provided
            if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
                conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
                features = torch.cat([features, conf_features], dim=1)

            return features

    def init_extractor(self, texts: List[str]) -> 'CausalText':
        """
        Initialize the feature extractor with training texts.

        This method performs different operations depending on the feature extractor type:
        - CNN/GRU: Builds vocabulary from texts and initializes embeddings (required)
        - BERT: No-op (uses pretrained tokenizer)
        - ConfounderExtractor: Triggers lazy initialization of pretrained encoder
        - HierarchicalTransformer: Triggers lazy initialization of sentence encoder

        MUST be called before training for CNN/GRU/GRUConfounder extractors.
        For pretrained extractors (BERT, Confounder, HierarchicalTransformer), this
        triggers lazy initialization but the texts argument is not used.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        if hasattr(self.feature_extractor, 'fit_tokenizer'):
            self.feature_extractor.fit_tokenizer(texts)
        # BERT uses pretrained tokenizer, no fitting needed

        # Initialize effect extractor if in dual mode (R-Learner or Uplift)
        dual_mode = (
            (self.rlearner_dual_extractors and self.model_type == "rlearner") or
            (self.uplift_dual_extractors and self.model_type == "uplift")
        )
        if dual_mode and self.effect_feature_extractor is not None:
            if hasattr(self.effect_feature_extractor, 'fit_tokenizer'):
                self.effect_feature_extractor.fit_tokenizer(texts)
            mode_name = "R-Learner" if self.model_type == "rlearner" else "Uplift"
            logger.info(f"Effect extractor initialized (dual {mode_name} mode)")

        # After feature extractor initialization, verify/update CLAM instance head dimensions
        # for gated_mil_hierarchical which has lazy initialization
        if (self.clam_enabled and
            hasattr(self, '_clam_instance_dim_source') and
            self._clam_instance_dim_source == "gated_mil_sentence_dim"):
            # Get the actual sentence_dim from the initialized feature extractor
            if hasattr(self.feature_extractor, '_sentence_dim') and self.feature_extractor._sentence_dim is not None:
                actual_dim = self.feature_extractor._sentence_dim
                # Check if we need to reinitialize the instance head
                current_input_dim = self.instance_head.shared_layers[0].in_features if hasattr(self.instance_head, 'shared_layers') else None
                if current_input_dim != actual_dim:
                    logger.info(f"CLAM: Reinitializing instance head with actual sentence_dim={actual_dim}")
                    # Reinitialize instance head with correct dimension
                    if self.model_type == "rlearner":
                        self.instance_head = RLearnerNet(
                            input_dim=actual_dim,
                            representation_dim=self.clam_instance_hidden_dim,
                            hidden_outcome_dim=self.clam_instance_hidden_dim // 2,
                            dropout=self.config['causal_head_dropout']
                        )
                    elif self.model_type == "uplift":
                        self.instance_head = UpliftNet(
                            input_dim=actual_dim,
                            representation_dim=self.clam_instance_hidden_dim,
                            hidden_outcome_dim=self.clam_instance_hidden_dim // 2,
                            dropout=self.config['causal_head_dropout']
                        )
                    elif self.model_type == "traditional_logreg":
                        self.instance_head = TraditionalLogRegNet(
                            input_dim=actual_dim,
                            representation_dim=self.clam_instance_hidden_dim,
                            hidden_outcome_dim=self.clam_instance_hidden_dim // 2,
                            dropout=self.config['causal_head_dropout']
                        )
                    else:
                        self.instance_head = DragonNet(
                            input_dim=actual_dim,
                            representation_dim=self.clam_instance_hidden_dim,
                            hidden_outcome_dim=self.clam_instance_hidden_dim // 2,
                            dropout=self.config['causal_head_dropout']
                        )
                    self.instance_head.to(self._device)

        return self

    def fit_tokenizer(self, texts: List[str]) -> 'CausalText':
        """
        Alias for init_extractor() for backward compatibility.

        See init_extractor() for documentation.
        """
        return self.init_extractor(texts)

    def fit_explicit_confounder_featurizer(
        self,
        confounder_values_list: List[Dict[str, Any]]
    ) -> 'CausalText':
        """
        Fit the explicit confounder featurizer on training data.

        This computes normalization statistics (mean/std) for continuous confounders.
        Must be called before training if explicit confounders are used.

        Args:
            confounder_values_list: List of dicts with confounder values from training data.
                Each dict should have "{name}" and "{name}_missing" keys.

        Returns:
            self for method chaining
        """
        if self.explicit_confounder_featurizer is not None:
            self.explicit_confounder_featurizer.fit(confounder_values_list)
        return self

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model checkpoint including tokenizer state.
        """
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'feature_extractor': self.feature_extractor.state_dict(),
            'dragonnet': self.net.state_dict(),
            'feature_extractor_type': self.feature_extractor_type,
        }

        # Save tokenizer state for CNN, or extractor state for BERT
        if self.feature_extractor_type == "cnn":
            checkpoint['tokenizer_state'] = self.feature_extractor.get_tokenizer_state()
        else:
            checkpoint['extractor_state'] = self.feature_extractor.get_state()

        # Save explicit confounder featurizer state if enabled
        if self.explicit_confounder_featurizer is not None:
            checkpoint['explicit_confounder_featurizer_state'] = self.explicit_confounder_featurizer.get_state()

        # Save effect extractor and effect MLP state if in dual mode (R-Learner or Uplift)
        dual_mode = (
            (self.rlearner_dual_extractors and self.model_type == "rlearner") or
            (self.uplift_dual_extractors and self.model_type == "uplift")
        )
        if dual_mode and self.effect_feature_extractor is not None:
            checkpoint['effect_feature_extractor'] = self.effect_feature_extractor.state_dict()
            checkpoint['effect_mlp'] = self.effect_mlp.state_dict()
            if hasattr(self.effect_feature_extractor, 'get_state'):
                checkpoint['effect_extractor_state'] = self.effect_feature_extractor.get_state()
            elif hasattr(self.effect_feature_extractor, 'get_tokenizer_state'):
                checkpoint['effect_tokenizer_state'] = self.effect_feature_extractor.get_tokenizer_state()

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        device: Optional[str] = None
    ) -> 'CausalText':
        """
        Load model from checkpoint including tokenizer state.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

        if device is not None:
            config['device'] = device

        # Create model
        model = cls(**config)

        # Load tokenizer state for CNN (rebuilds embedding layer with correct vocab size)
        if model.feature_extractor_type == "cnn" and 'tokenizer_state' in checkpoint:
            model.feature_extractor.load_tokenizer_state(checkpoint['tokenizer_state'])

        # Load effect extractor tokenizer/state BEFORE loading model_state_dict
        # This ensures embedding layers have correct dimensions
        # Check for dual mode (R-Learner or Uplift)
        dual_mode = (
            (model.rlearner_dual_extractors and model.model_type == "rlearner") or
            (model.uplift_dual_extractors and model.model_type == "uplift")
        )
        if dual_mode and model.effect_feature_extractor is not None:
            if 'effect_tokenizer_state' in checkpoint:
                if hasattr(model.effect_feature_extractor, 'load_tokenizer_state'):
                    model.effect_feature_extractor.load_tokenizer_state(checkpoint['effect_tokenizer_state'])
            elif 'effect_extractor_state' in checkpoint:
                if hasattr(model.effect_feature_extractor, 'load_state'):
                    model.effect_feature_extractor.load_state(checkpoint['effect_extractor_state'])

        # Load state dict (after tokenizer so embedding has correct size)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            if 'feature_extractor' in checkpoint:
                model.feature_extractor.load_state_dict(
                    checkpoint['feature_extractor'],
                    strict=False
                )
            if 'dragonnet' in checkpoint:
                model.net.load_state_dict(
                    checkpoint['dragonnet'],
                    strict=False
                )
            # Load effect extractor weights separately if not using model_state_dict
            if dual_mode and model.effect_feature_extractor is not None:
                if 'effect_feature_extractor' in checkpoint:
                    model.effect_feature_extractor.load_state_dict(
                        checkpoint['effect_feature_extractor'],
                        strict=False
                    )
                if 'effect_mlp' in checkpoint:
                    model.effect_mlp.load_state_dict(checkpoint['effect_mlp'])

        # Load explicit confounder featurizer state if present
        if 'explicit_confounder_featurizer_state' in checkpoint and model.explicit_confounder_featurizer is not None:
            model.explicit_confounder_featurizer.load_state(checkpoint['explicit_confounder_featurizer_state'])

        logger.info(f"Model loaded from {path}")
        return model

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)


# Backward compatibility alias
CausalCNNText = CausalText
