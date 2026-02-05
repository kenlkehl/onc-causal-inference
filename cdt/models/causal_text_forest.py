# cdt/models/causal_text_forest.py
"""Two-stage causal text model combining neural feature extraction with causal forests."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .causal_forest_head import CausalForestHead, ECONML_AVAILABLE
from .explicit_confounder_featurizer import get_raw_confounder_features, ExplicitConfounderFeaturizer
from .extractor_factory import create_feature_extractor
from ..config import normalize_feature_extractor_type, ExplicitConfounderSpec


logger = logging.getLogger(__name__)


class CausalTextForest(nn.Module):
    """
    Two-stage causal text model combining neural feature extraction with causal forests.

    Architecture:
        Stage 1 (Neural): Feature extractor + propensity/outcome heads
            - Learns to extract confounders from text
            - Trained with propensity + outcome BCE losses
            - Any existing feature extractor (gru_pool, gated_mil_hierarchical, etc.)

        Stage 2 (Causal Forest): Effect estimation
            - Extracts learned representations as fixed features
            - Trains CausalForestDML on those features
            - Estimates τ(X) = E[Y(1) - Y(0) | X] directly

    Advantages:
        - Separation of concerns: Neural net for text→representation, causal forest for effects
        - Doubly-robust estimation with asymptotic guarantees
        - No gradient competition between propensity/outcome/treatment effect
        - Confidence intervals for treatment effects
        - Designed for heterogeneous treatment effect estimation

    Training Flow:
        1. Train representation (propensity + outcome loss)
        2. Extract features for all samples
        3. Fit causal forest on extracted features
        4. Predict ITEs with confidence intervals

    References:
        Athey, Tibshirani, Wager (2019). Generalized Random Forests.
        Chernozhukov et al. (2018). Double/Debiased Machine Learning.
    """

    def __init__(
        self,
        # Feature extractor type
        feature_extractor_type: str = "gru_pool",
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
        # LLM args (decoder-only with random init)
        llm_model_name: str = "Qwen/Qwen3-0.6B-Base",
        llm_max_length: int = 8192,
        llm_projection_dim: Optional[int] = 128,
        llm_dropout: float = 0.1,
        llm_gradient_checkpointing: bool = True,
        # Simple heads args
        representation_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        # Causal Forest args
        cf_n_estimators: int = 100,
        cf_max_depth: Optional[int] = None,
        cf_min_samples_leaf: int = 5,
        cf_max_features: str = "sqrt",
        cf_honest: bool = True,
        cf_inference: bool = True,
        cf_random_state: int = 42,
        # R-learner representation training args
        cf_use_rlearner_representation: bool = False,
        cf_gamma_rlearner: float = 1.0,
        # R-learner dual extractor mode (separate extractors for nuisance vs effect)
        cf_rlearner_dual_extractors: bool = False,
        # Numeric feature args
        numeric_features_enabled: bool = False,
        numeric_embedding_dim: int = 32,
        numeric_magnitude_bins: int = 8,
        numeric_type_categories: int = 10,
        # Explicit confounder args (raw features for causal forest, MLP for Stage 1 training)
        explicit_confounder_specs: Optional[List[ExplicitConfounderSpec]] = None,
        explicit_confounder_output_dim: int = 64,
        explicit_confounder_hidden_dim: int = 128,
        explicit_confounder_dropout: float = 0.1,
        # CLAM instance-level loss args (for GRU-Pool and other hierarchical extractors)
        clam_enabled: bool = False,
        clam_num_instances: int = 5,
        clam_instance_hidden_dim: int = 64,
        # Device
        device: str = "cuda:0"
    ):
        """
        Initialize two-stage causal text model.

        Args:
            feature_extractor_type: Type of neural feature extractor
            ... (feature extractor args - same as CausalText)
            representation_dim: Dimension for propensity/outcome heads
            hidden_dim: Hidden dimension for heads
            dropout: Dropout rate
            cf_n_estimators: Number of trees in causal forest (must be divisible by 4)
            cf_max_depth: Max tree depth (None = unlimited)
            cf_min_samples_leaf: Minimum samples per leaf
            cf_max_features: Feature subset strategy
            cf_honest: Use honest estimation
            cf_inference: Enable confidence intervals
            cf_random_state: Random seed
            cf_use_rlearner_representation: Add τ head and R-loss to representation training
            cf_gamma_rlearner: Weight for R-learner loss
            device: PyTorch device
        """
        super().__init__()

        if not ECONML_AVAILABLE:
            raise ImportError(
                "econml is required for CausalTextForest. "
                "Install with: pip install econml"
            )

        self._device = torch.device(device)
        self.feature_extractor_type = normalize_feature_extractor_type(feature_extractor_type)

        # Store config
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
            'llm_model_name': llm_model_name,
            'llm_max_length': llm_max_length,
            'llm_projection_dim': llm_projection_dim,
            'llm_dropout': llm_dropout,
            'llm_gradient_checkpointing': llm_gradient_checkpointing,
            'representation_dim': representation_dim,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'cf_n_estimators': cf_n_estimators,
            'cf_max_depth': cf_max_depth,
            'cf_min_samples_leaf': cf_min_samples_leaf,
            'cf_max_features': cf_max_features,
            'cf_honest': cf_honest,
            'cf_inference': cf_inference,
            'cf_random_state': cf_random_state,
            'cf_use_rlearner_representation': cf_use_rlearner_representation,
            'cf_gamma_rlearner': cf_gamma_rlearner,
            'cf_rlearner_dual_extractors': cf_rlearner_dual_extractors,
            'numeric_features_enabled': numeric_features_enabled,
            'numeric_embedding_dim': numeric_embedding_dim,
            'numeric_magnitude_bins': numeric_magnitude_bins,
            'numeric_type_categories': numeric_type_categories,
            'explicit_confounder_specs': explicit_confounder_specs,
            'explicit_confounder_output_dim': explicit_confounder_output_dim,
            'explicit_confounder_hidden_dim': explicit_confounder_hidden_dim,
            'explicit_confounder_dropout': explicit_confounder_dropout,
            'clam_enabled': clam_enabled,
            'clam_num_instances': clam_num_instances,
            'clam_instance_hidden_dim': clam_instance_hidden_dim,
        }

        # Store explicit confounder output dim for head input calculation
        self._explicit_confounder_output_dim = explicit_confounder_output_dim

        # Explicit confounder support (raw features for interpretability)
        self.explicit_confounder_specs = explicit_confounder_specs or []
        self._explicit_confounder_means = {}
        self._explicit_confounder_stds = {}
        self._explicit_confounders_fitted = False

        if self.explicit_confounder_specs:
            # Calculate raw feature dimension
            raw_dim = 0
            for spec in self.explicit_confounder_specs:
                if spec.type == "categorical":
                    n_cats = len(spec.categories) if spec.categories else 2
                    raw_dim += (n_cats - 1) + 1  # k-1 dummies + missing
                else:
                    raw_dim += 2  # value + missing
            self._explicit_confounder_raw_dim = raw_dim
            logger.info(f"Explicit confounders: {len(self.explicit_confounder_specs)} specs, "
                       f"raw feature dim: {raw_dim}")
        else:
            self._explicit_confounder_raw_dim = 0

        # Initialize ExplicitConfounderFeaturizer (MLP) for Stage 1 training
        # This allows the neural network to learn from explicit confounders during representation learning
        if self.explicit_confounder_specs:
            self.explicit_confounder_featurizer = ExplicitConfounderFeaturizer(
                specs=self.explicit_confounder_specs,
                output_dim=explicit_confounder_output_dim,
                hidden_dim=explicit_confounder_hidden_dim,
                dropout=explicit_confounder_dropout,
                device=str(self._device)
            )
            logger.info(f"ExplicitConfounderFeaturizer for Stage 1: output_dim={explicit_confounder_output_dim}")
        else:
            self.explicit_confounder_featurizer = None

        # Initialize feature extractor using factory
        self.feature_extractor = create_feature_extractor(
            extractor_type=self.feature_extractor_type,
            device=self._device,
            model_type="dragonnet",  # Use dragonnet style for representation
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
            llm_model_name=llm_model_name,
            llm_max_length=llm_max_length,
            llm_projection_dim=llm_projection_dim,
            llm_dropout=llm_dropout,
            llm_gradient_checkpointing=llm_gradient_checkpointing,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
        )

        logger.info(f"Using {self.feature_extractor_type.upper()} feature extractor")

        # Simple propensity head for representation learning
        # Input dim = text features + explicit confounder features (if any)
        input_dim = self.feature_extractor.output_dim
        if self.explicit_confounder_featurizer is not None:
            input_dim += explicit_confounder_output_dim
        self._head_input_dim = input_dim  # Store for logging

        self.propensity_head = nn.Sequential(
            nn.Linear(input_dim, representation_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(representation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Simple outcome head for representation learning
        self.outcome_head = nn.Sequential(
            nn.Linear(input_dim, representation_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(representation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Optional: Treatment effect head for R-learner representation training
        # When enabled, adds R-loss to encourage embeddings to capture τ heterogeneity
        self.use_rlearner_representation = cf_use_rlearner_representation
        self.cf_gamma_rlearner = cf_gamma_rlearner
        self.rlearner_dual_extractors = cf_rlearner_dual_extractors
        self.effect_feature_extractor = None
        self.effect_mlp = None

        if cf_use_rlearner_representation:
            if cf_rlearner_dual_extractors:
                # DUAL EXTRACTOR MODE: Create a second feature extractor for τ(X)
                # This provides complete separation between nuisance and effect learning
                self.effect_feature_extractor = create_feature_extractor(
                    extractor_type=self.feature_extractor_type,
                    device=self._device,
                    model_type="dragonnet",
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
                    llm_model_name=llm_model_name,
                    llm_max_length=llm_max_length,
                    llm_projection_dim=llm_projection_dim,
                    llm_dropout=llm_dropout,
                    llm_gradient_checkpointing=llm_gradient_checkpointing,
                    numeric_features_enabled=numeric_features_enabled,
                    numeric_embedding_dim=numeric_embedding_dim,
                    numeric_magnitude_bins=numeric_magnitude_bins,
                    numeric_type_categories=numeric_type_categories,
                )

                # Effect MLP: takes effect extractor output, predicts τ
                effect_input_dim = self.effect_feature_extractor.output_dim
                self.effect_mlp = nn.Sequential(
                    nn.Linear(effect_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)  # τ is unbounded
                )

                logger.info("  R-learner representation training: ENABLED (DUAL EXTRACTOR MODE)")
                logger.info(f"    Nuisance extractor: {self.feature_extractor_type} -> e(X), m(X)")
                logger.info(f"    Effect extractor: {self.feature_extractor_type} -> τ(X)")
                logger.info(f"    Effect MLP: {effect_input_dim} -> {hidden_dim} -> 1")

                # effect_head is not used in dual mode, but set to None for clarity
                self.effect_head = None
            else:
                # SINGLE EXTRACTOR MODE: Use shared features with separate effect head
                self.effect_head = nn.Sequential(
                    nn.Linear(input_dim, representation_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(representation_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)  # No activation - τ can be negative
                )
                logger.info("  R-learner representation training: ENABLED (single extractor)")
        else:
            self.effect_head = None

        # CLAM instance-level loss head (for hierarchical extractors)
        # Creates a separate, lightweight head for top-attended chunks
        self.clam_enabled = clam_enabled
        self.clam_num_instances = clam_num_instances
        self.clam_instance_hidden_dim = clam_instance_hidden_dim
        self.instance_propensity_head = None
        self.instance_outcome_head = None

        # Define which extractors support CLAM and their instance input dimensions
        clam_supported_extractors = {
            "gru_pool": gru_pool_transformer_dim,
            "hierarchical_transformer": hier_transformer_dim,
            "gated_mil_hierarchical": None,  # Needs lazy init
            "gru_transformer_mil": gru_pool_transformer_dim,  # Uses same transformer dim
        }

        if clam_enabled:
            if self.feature_extractor_type not in clam_supported_extractors:
                logger.warning(f"CLAM instance loss is not supported for {self.feature_extractor_type} extractor. "
                              f"Supported extractors: {list(clam_supported_extractors.keys())}. Disabling CLAM.")
                self.clam_enabled = False
            else:
                # Get instance input dimension based on extractor type
                instance_input_dim = clam_supported_extractors[self.feature_extractor_type]

                # For gated_mil_hierarchical, use default BERT-tiny hidden size
                if instance_input_dim is None:
                    instance_input_dim = 128  # Default for bert-tiny

                # Create simple propensity and outcome heads for instances
                # These are lightweight heads that supervise top-attended chunks
                self.instance_propensity_head = nn.Sequential(
                    nn.Linear(instance_input_dim, clam_instance_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(clam_instance_hidden_dim, 1)
                )

                self.instance_outcome_head = nn.Sequential(
                    nn.Linear(instance_input_dim, clam_instance_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(clam_instance_hidden_dim, 1)
                )

                logger.info(f"CLAM instance-level loss enabled: {clam_num_instances} top chunks, "
                           f"instance_input_dim={instance_input_dim}, instance_head_dim={clam_instance_hidden_dim}")

        # Causal forest (non-neural, trained separately)
        self.causal_forest = CausalForestHead(
            n_estimators=cf_n_estimators,
            max_depth=cf_max_depth,
            min_samples_leaf=cf_min_samples_leaf,
            max_features=cf_max_features,
            honest=cf_honest,
            inference=cf_inference,
            random_state=cf_random_state
        )

        # Move to device
        self.to(self._device)

        logger.info(f"CausalTextForest initialized:")
        logger.info(f"  Feature extractor: {self.feature_extractor_type}")
        logger.info(f"  Feature dim: {input_dim}")
        logger.info(f"  Causal forest: {cf_n_estimators} trees, honest={cf_honest}")

    def forward(
        self,
        texts: List[str],
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through neural components.

        Args:
            texts: List of text strings
            explicit_confounder_values: Optional list of dicts with explicit confounder values

        Returns:
            features: Extracted features (batch, feature_dim)
            propensity_logit: Propensity prediction (batch, 1)
            outcome_logit: Outcome prediction (batch, 1)
        """
        features = self.feature_extractor(texts)

        # Concatenate explicit confounder features if provided
        if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
            conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
            features = torch.cat([features, conf_features], dim=1)

        propensity_logit = self.propensity_head(features)
        outcome_logit = self.outcome_head(features)
        return features, propensity_logit, outcome_logit

    def train_representation_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        gamma_rlearner: float = 1.0,
        label_smoothing: float = 0.0,
        stop_grad_propensity: bool = False,
        clam_instance_weight: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single representation training step.

        This stage learns to extract confounders from text by training
        propensity and outcome prediction heads. Optionally includes R-learner
        loss to encourage embeddings to capture treatment effect heterogeneity.

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys.
                   Optional 'explicit_confounder_values' for explicit confounders.
            alpha_propensity: Weight for propensity loss
            gamma_rlearner: Weight for R-learner loss (only used if use_rlearner_representation=True)
            label_smoothing: Label smoothing factor
            stop_grad_propensity: If True, detach features before propensity
            clam_instance_weight: Weight for CLAM instance-level loss (only used if clam_enabled=True)

        Returns:
            Dictionary with loss components
        """
        texts = batch['texts']
        treatments = batch['treatment']
        outcomes = batch['outcome']
        explicit_confounder_values = batch.get('explicit_confounder_values', None)

        # Apply label smoothing
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Extract features from text
        features = self.feature_extractor(texts)

        # Concatenate explicit confounder features if provided
        if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
            conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
            features = torch.cat([features, conf_features], dim=1)

        # Propensity prediction
        if stop_grad_propensity:
            propensity_logit = self.propensity_head(features.detach())
        else:
            propensity_logit = self.propensity_head(features)

        # Outcome prediction (marginal E[Y|X])
        outcome_logit = self.outcome_head(features)

        # Losses
        propensity_loss = F.binary_cross_entropy_with_logits(
            propensity_logit.squeeze(-1),
            treatments_smooth
        )

        outcome_loss = F.binary_cross_entropy_with_logits(
            outcome_logit.squeeze(-1),
            outcomes_smooth
        )

        # R-learner loss (optional): encourages features to capture τ heterogeneity
        # R-loss: E[((Y - m(X)) - τ(X)(T - e(X)))²]
        # CRITICAL: Nuisance functions (e, m) are DETACHED so gradients flow only through τ
        r_loss = torch.tensor(0.0, device=self._device)
        if self.use_rlearner_representation:
            if self.rlearner_dual_extractors and self.effect_feature_extractor is not None:
                # DUAL EXTRACTOR MODE:
                # - Nuisance extractor (self.feature_extractor) already computed features for e(X), m(X)
                # - Effect extractor (self.effect_feature_extractor) + effect_mlp -> τ(X)

                # Effect path: extract features for τ(X) using separate extractor
                effect_features = self.effect_feature_extractor(texts)

                # Compute τ(X) from effect MLP
                tau = self.effect_mlp(effect_features)

                # Detach nuisance functions - gradients flow only through effect extractor + MLP
                e_X = torch.sigmoid(propensity_logit).detach().clamp(0.01, 0.99)
                m_X = torch.sigmoid(outcome_logit).detach()

                # R-loss: pseudo-outcome regression
                Y_residual = outcomes - m_X.squeeze(-1)
                T_residual = treatments - e_X.squeeze(-1)
                r_loss = ((Y_residual - tau.squeeze(-1) * T_residual) ** 2).mean()

            elif self.effect_head is not None:
                # SINGLE EXTRACTOR MODE: Use shared features with separate effect head
                tau = self.effect_head(features)

                # Detach nuisance functions - this is the key to R-learner
                # Gradients only flow through τ, not through e or m estimates
                e_X = torch.sigmoid(propensity_logit).detach().clamp(0.01, 0.99)
                m_X = torch.sigmoid(outcome_logit).detach()

                # R-loss: pseudo-outcome regression
                Y_residual = outcomes - m_X.squeeze(-1)
                T_residual = treatments - e_X.squeeze(-1)
                r_loss = ((Y_residual - tau.squeeze(-1) * T_residual) ** 2).mean()

        # CLAM instance-level loss (if enabled)
        # Supervises top-attended chunks with document-level labels
        instance_loss = torch.tensor(0.0, device=self._device)
        if self.clam_enabled and clam_instance_weight > 0 and self.instance_propensity_head is not None:
            # Check if feature extractor supports forward_with_instances
            if hasattr(self.feature_extractor, 'forward_with_instances'):
                # Get chunk embeddings and attention weights
                _, chunk_embs_list, attn_weights_list = self.feature_extractor.forward_with_instances(texts)

                all_top_chunks = []
                expanded_treatments = []
                expanded_outcomes = []

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

                    # Forward through instance heads
                    inst_propensity = self.instance_propensity_head(stacked_chunks)
                    inst_outcome = self.instance_outcome_head(stacked_chunks)

                    # Instance propensity loss
                    instance_propensity_loss = F.binary_cross_entropy_with_logits(
                        inst_propensity.squeeze(-1), exp_treatments
                    )

                    # Instance outcome loss
                    instance_outcome_loss = F.binary_cross_entropy_with_logits(
                        inst_outcome.squeeze(-1), exp_outcomes
                    )

                    instance_loss = instance_outcome_loss + alpha_propensity * instance_propensity_loss

        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            gamma_rlearner * r_loss +
            clam_instance_weight * instance_loss
        )

        result = {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'r_loss': r_loss.detach() if isinstance(r_loss, torch.Tensor) else torch.tensor(r_loss),
            'propensity_logit': propensity_logit.detach(),
            'outcome_logit': outcome_logit.detach()
        }

        if self.clam_enabled:
            result['instance_loss'] = instance_loss.detach() if isinstance(instance_loss, torch.Tensor) else instance_loss

        return result

    # Alias for API consistency with CausalText
    def train_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        gamma_rlearner: float = 1.0,
        label_smoothing: float = 0.0,
        stop_grad_propensity: bool = False,
        clam_instance_weight: float = 0.0,
        **kwargs  # Ignore extra args like beta_targreg for compatibility
    ) -> Dict[str, torch.Tensor]:
        """Alias for train_representation_step for API consistency."""
        return self.train_representation_step(
            batch=batch,
            alpha_propensity=alpha_propensity,
            gamma_rlearner=gamma_rlearner,
            label_smoothing=label_smoothing,
            stop_grad_propensity=stop_grad_propensity,
            clam_instance_weight=clam_instance_weight
        )

    def extract_features(
        self,
        texts: List[str],
        batch_size: int = 32,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features and nuisance predictions for all texts.

        In dual extractor mode, features are extracted from the effect extractor
        (optimized for treatment effect heterogeneity via R-loss). Propensity
        and outcome predictions still come from the nuisance extractor.

        Args:
            texts: List of all text strings
            batch_size: Batch size for processing
            explicit_confounder_values: Optional list of dicts with confounder values.
                If provided and explicit_confounder_specs is set, raw confounder features
                are concatenated to neural features.

        Returns:
            features: Feature matrix (n_samples, feature_dim + confounder_dim)
            propensity: Propensity predictions (n_samples,)
            outcome_pred: Outcome predictions (n_samples,)
        """
        self.eval()
        all_text_features = []
        all_propensity = []
        all_outcome = []

        # Determine which extractor to use for features
        # In dual mode, use effect extractor (optimized for τ)
        # Otherwise, use nuisance extractor (feature_extractor)
        use_effect_extractor = (
            self.rlearner_dual_extractors and
            self.effect_feature_extractor is not None
        )

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Get batch slice of confounder values if provided
                batch_conf_values = None
                if explicit_confounder_values is not None:
                    batch_conf_values = explicit_confounder_values[i:i + batch_size]

                # Extract text features for causal forest input
                # In dual mode: use effect extractor (learned specifically for τ)
                # In single mode: use nuisance extractor (feature_extractor)
                if use_effect_extractor:
                    text_features = self.effect_feature_extractor(batch_texts)
                else:
                    text_features = self.feature_extractor(batch_texts)
                all_text_features.append(text_features.cpu().numpy())

                # Get propensity/outcome predictions from nuisance extractor (full forward)
                # These are used for nuisance estimation in causal forest
                _, prop_logit, outcome_logit = self.forward(
                    batch_texts,
                    explicit_confounder_values=batch_conf_values
                )
                all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                all_outcome.append(torch.sigmoid(outcome_logit).cpu().numpy())

        neural_features = np.vstack(all_text_features)

        # Concatenate raw confounder features if provided
        if explicit_confounder_values is not None and self.explicit_confounder_specs:
            raw_conf_features = self._get_raw_confounder_features(explicit_confounder_values)
            combined_features = np.hstack([neural_features, raw_conf_features])
        else:
            combined_features = neural_features

        return (
            combined_features,
            np.vstack(all_propensity).flatten(),
            np.vstack(all_outcome).flatten()
        )

    def train_causal_forest(
        self,
        texts: List[str],
        T: np.ndarray,
        Y: np.ndarray,
        batch_size: int = 32,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None
    ) -> 'CausalTextForest':
        """
        Train causal forest on extracted features.

        Should be called after representation training is complete.
        The causal forest uses sklearn random forests for nuisance estimation
        on the neural network's learned features.

        Args:
            texts: List of training texts
            T: Treatment indicators
            Y: Outcome indicators
            batch_size: Batch size for feature extraction
            explicit_confounder_values: Optional list of dicts with confounder values.
                If provided, raw confounder features are concatenated to neural features.

        Returns:
            self
        """
        logger.info("Extracting features for causal forest training...")
        features, _, _ = self.extract_features(
            texts, batch_size,
            explicit_confounder_values=explicit_confounder_values
        )

        if self.explicit_confounder_specs and explicit_confounder_values is not None:
            logger.info(f"  Neural features: {features.shape[1] - self._explicit_confounder_raw_dim}, "
                       f"Raw confounder features: {self._explicit_confounder_raw_dim}")

        # Fit causal forest on neural network features (+ raw confounder features if provided)
        # Nuisance functions are estimated internally using random forests
        self.causal_forest.fit(
            X=features,
            T=T,
            Y=Y
        )

        return self

    def predict(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_ci: bool = True,
        alpha: float = 0.05,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict ITEs using trained causal forest.

        Args:
            texts: List of text strings
            batch_size: Batch size for feature extraction
            return_ci: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            explicit_confounder_values: Optional list of dicts with confounder values.
                Must be provided if model was trained with explicit confounders.

        Returns:
            Dictionary with predictions:
                - tau_pred: ITE estimates
                - propensity: Propensity scores from neural network
                - outcome_pred: Outcome predictions from neural network
                - tau_lower, tau_upper: Confidence intervals (if return_ci)
        """
        # Extract features (with raw confounder features if provided)
        features, propensity, outcome_pred = self.extract_features(
            texts, batch_size,
            explicit_confounder_values=explicit_confounder_values
        )

        # Get ITE predictions from causal forest
        cf_preds = self.causal_forest.predict(features, return_ci=return_ci, alpha=alpha)

        result = {
            'tau_pred': cf_preds['tau_pred'],
            'propensity_prob': propensity,
            'outcome_pred': outcome_pred,
            # For compatibility with existing prediction format
            'pred_ite_prob': cf_preds['tau_pred'],
            'pred_propensity_prob': propensity,
        }

        # Derive Y0/Y1 estimates from τ and m
        # From: m = e*y1 + (1-e)*y0 and tau = y1 - y0
        # Solving: y0 = m - e*tau, y1 = m + (1-e)*tau
        tau = cf_preds['tau_pred']
        y0_prob = np.clip(outcome_pred - propensity * tau, 0, 1)
        y1_prob = np.clip(outcome_pred + (1 - propensity) * tau, 0, 1)

        result['pred_y0_prob'] = y0_prob
        result['pred_y1_prob'] = y1_prob

        if 'tau_lower' in cf_preds:
            result['tau_lower'] = cf_preds['tau_lower']
            result['tau_upper'] = cf_preds['tau_upper']

        return result

    def fit_tokenizer(self, texts: List[str]) -> 'CausalTextForest':
        """
        Initialize the feature extractor(s) with training texts.

        Required for CNN, GRU, and GRU-based extractors.
        In dual extractor mode, initializes both nuisance and effect extractors.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        if hasattr(self.feature_extractor, 'fit_tokenizer'):
            self.feature_extractor.fit_tokenizer(texts)

        # Initialize effect extractor if in dual mode
        if self.rlearner_dual_extractors and self.effect_feature_extractor is not None:
            if hasattr(self.effect_feature_extractor, 'fit_tokenizer'):
                self.effect_feature_extractor.fit_tokenizer(texts)
            logger.info("Effect extractor initialized (dual R-Learner mode)")

        return self

    def fit_explicit_confounder_featurizer(
        self,
        confounder_values_list: List[Dict[str, Any]]
    ) -> 'CausalTextForest':
        """
        Fit the explicit confounder featurizer (MLP) on training data.

        This computes normalization statistics (mean/std) for continuous confounders
        used during Stage 1 neural network training. Must be called before training
        if explicit confounders are used.

        Args:
            confounder_values_list: List of dicts with confounder values from training data.
                Each dict should have "{name}" and "{name}_missing" keys.

        Returns:
            self for method chaining
        """
        if self.explicit_confounder_featurizer is not None:
            self.explicit_confounder_featurizer.fit(confounder_values_list)
            logger.info("Fitted ExplicitConfounderFeaturizer for Stage 1 training")
        return self

    def fit_explicit_confounders(
        self,
        confounder_values_list: List[Dict[str, Any]]
    ) -> 'CausalTextForest':
        """
        Compute normalization statistics for explicit confounders from training data.

        For causal forest, we use raw features (no MLP) for interpretability.
        This method computes mean/std for continuous confounders.

        Args:
            confounder_values_list: List of dicts with confounder values.
                Keys should match spec.name (e.g., "age", not "explicit_conf_age").

        Returns:
            self for method chaining
        """
        if not self.explicit_confounder_specs:
            return self

        # Collect continuous values
        continuous_values = {
            spec.name: [] for spec in self.explicit_confounder_specs if spec.type == "continuous"
        }

        for values in confounder_values_list:
            for spec in self.explicit_confounder_specs:
                if spec.type == "continuous":
                    val = values.get(spec.name)
                    missing = values.get(f"{spec.name}_missing", val is None)
                    if not missing and val is not None:
                        continuous_values[spec.name].append(float(val))

        # Compute mean and std for each continuous confounder
        for name, vals in continuous_values.items():
            if vals:
                self._explicit_confounder_means[name] = sum(vals) / len(vals)
                variance = sum((v - self._explicit_confounder_means[name]) ** 2 for v in vals) / len(vals)
                self._explicit_confounder_stds[name] = max(variance ** 0.5, 1e-6)
            else:
                self._explicit_confounder_means[name] = 0.0
                self._explicit_confounder_stds[name] = 1.0

        self._explicit_confounders_fitted = True
        logger.info(f"Fitted explicit confounders on {len(confounder_values_list)} samples")
        return self

    def _get_raw_confounder_features(
        self,
        confounder_values_list: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Get raw confounder features as numpy array.

        Args:
            confounder_values_list: List of dicts with confounder values

        Returns:
            (n_samples, raw_dim) numpy array
        """
        if not self.explicit_confounder_specs:
            return np.zeros((len(confounder_values_list), 0))

        features, _ = get_raw_confounder_features(
            confounder_values_list,
            self.explicit_confounder_specs,
            continuous_means=self._explicit_confounder_means,
            continuous_stds=self._explicit_confounder_stds
        )
        return np.array(features)

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)

    def get_features(self, texts: List[str]) -> torch.Tensor:
        """
        Extract feature representations from texts.

        Args:
            texts: List of text strings

        Returns:
            Feature tensor: (batch, output_dim)
        """
        with torch.no_grad():
            return self.feature_extractor(texts)

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model checkpoint."""
        import pickle

        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'feature_extractor_type': self.feature_extractor_type,
            'causal_forest_state': self.causal_forest.get_state(),
        }

        # Save tokenizer state if applicable
        if hasattr(self.feature_extractor, 'get_tokenizer_state'):
            checkpoint['tokenizer_state'] = self.feature_extractor.get_tokenizer_state()
        elif hasattr(self.feature_extractor, 'get_state'):
            checkpoint['extractor_state'] = self.feature_extractor.get_state()

        # Save causal forest model (pickled)
        if self.causal_forest._fitted:
            checkpoint['causal_forest_model'] = pickle.dumps(self.causal_forest.model)

        # Save explicit confounder featurizer state if enabled
        if self.explicit_confounder_featurizer is not None:
            checkpoint['explicit_confounder_featurizer_state'] = self.explicit_confounder_featurizer.get_state()

        # Save effect extractor and effect MLP state if in dual mode
        if self.rlearner_dual_extractors and self.effect_feature_extractor is not None:
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
