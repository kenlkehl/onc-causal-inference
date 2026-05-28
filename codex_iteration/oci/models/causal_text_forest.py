# oci/models/causal_text_forest.py
"""Two-stage causal text model combining neural feature extraction with causal forests."""

import logging
from itertools import chain
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .causal_forest_head import CausalForestHead, ECONML_AVAILABLE
from .explicit_feature_featurizer import (
    ExplicitFeatureFeaturizer,
    filter_specs_by_role,
    get_raw_explicit_features,
)
from .extractor_factory import create_feature_extractor
from ..config import normalize_feature_extractor_type, ExplicitFeatureSpec
from ..data.cached_hidden_state_dataset import prepare_cached_batch


logger = logging.getLogger(__name__)


class CausalTextForest(nn.Module):
    """
    Two-stage causal text model combining neural feature extraction with causal forests.

    Architecture:
        Stage 1 (Neural): Feature extractor + propensity/outcome heads
            - Learns to extract confounders from text
            - Trained with propensity + outcome BCE losses
            - Frozen LLM Pooler feature extractor

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
        feature_extractor_type: str = "frozen_llm_pooler",
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
        flp_attention_slots: int = 1,
        flp_document_window: str = "tail",
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
        # Byte CNN args
        byte_embedding_dim: int = 32,
        byte_conv_dim: int = 64,
        byte_kernel_size: int = 7,
        byte_num_conv_blocks: int = 4,
        byte_chunk_size: int = 512,
        byte_chunk_overlap: int = 64,
        byte_max_chunks: int = 128,
        byte_gated_attention_dim: int = 64,
        byte_projection_dim: int = 128,
        byte_dropout: float = 0.1,
        # Simple heads args
        representation_dim: int = 128,
        hidden_dim: int = 64,
        wx_nuisance_hidden_dim: Optional[int] = None,
        wx_effect_hidden_dim: Optional[int] = None,
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
        cf_effect_aux_outcome_weight: float = 0.0,
        cf_nuisance_potential_weight: float = 0.0,
        cf_effect_dr_weight: float = 0.0,
        cf_effect_dr_clip: float = 1.0,
        cf_effect_attention_entropy_weight: float = 0.0,
        cf_export_nuisance_potential_features: bool = False,
        cf_export_shared_text_features: bool = True,
        cf_forest_x_mode: str = "x_hidden_plus_tau",
        # Explicit confounder args (raw features for causal forest, MLP for Stage 1 training)
        explicit_feature_specs: Optional[List[ExplicitFeatureSpec]] = None,
        explicit_feature_output_dim: int = 64,
        explicit_feature_hidden_dim: int = 128,
        explicit_feature_dropout: float = 0.1,
        # Backward-compatible aliases for older direct callers.
        explicit_confounder_specs: Optional[List[ExplicitFeatureSpec]] = None,
        explicit_confounder_output_dim: int = 64,
        explicit_confounder_hidden_dim: int = 128,
        explicit_confounder_dropout: float = 0.1,
        # Device
        device: str = "cuda:0",
        # Outcome type
        outcome_type: str = "binary",  # "binary" or "continuous"
    ):
        """Initialize two-stage causal text model."""
        super().__init__()

        if not ECONML_AVAILABLE:
            raise ImportError(
                "econml is required for CausalTextForest. "
                "Install with: pip install econml"
            )

        self._device = torch.device(device)
        self.outcome_type = outcome_type
        self.feature_extractor_type = normalize_feature_extractor_type(feature_extractor_type)
        if explicit_feature_specs is None:
            explicit_feature_specs = explicit_confounder_specs
            explicit_feature_output_dim = explicit_confounder_output_dim
            explicit_feature_hidden_dim = explicit_confounder_hidden_dim
            explicit_feature_dropout = explicit_confounder_dropout
        wx_nuisance_hidden_dim = wx_nuisance_hidden_dim or hidden_dim
        wx_effect_hidden_dim = wx_effect_hidden_dim or hidden_dim
        valid_forest_x_modes = {"x_hidden_plus_tau", "x_hidden", "tau"}
        if cf_forest_x_mode not in valid_forest_x_modes:
            raise ValueError(
                f"cf_forest_x_mode must be one of {sorted(valid_forest_x_modes)}, "
                f"got {cf_forest_x_mode!r}"
            )

        # Store config
        self.config = {
            'feature_extractor_type': feature_extractor_type,
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
            'flp_chat_template_prompt': flp_chat_template_prompt,
            'flp_attention_slots': flp_attention_slots,
            'flp_document_window': flp_document_window,
            'hlm_model_name': hlm_model_name,
            'hlm_chunk_size': hlm_chunk_size,
            'hlm_chunk_overlap': hlm_chunk_overlap,
            'hlm_max_chunks': hlm_max_chunks,
            'hlm_freeze_llm': hlm_freeze_llm,
            'hlm_gated_attention_dim': hlm_gated_attention_dim,
            'hlm_projection_dim': hlm_projection_dim,
            'hlm_dropout': hlm_dropout,
            'hlm_gradient_checkpointing': hlm_gradient_checkpointing,
            'hlm_downprojection_dim': hlm_downprojection_dim,
            'hlm_skip_llm': hlm_skip_llm,
            'hlm_cached_hidden_size': hlm_cached_hidden_size,
            'hlm_chat_template_prompt': hlm_chat_template_prompt,
            'hcnn_embedding_dim': hcnn_embedding_dim,
            'hcnn_conv_dim': hcnn_conv_dim,
            'hcnn_kernel_size': hcnn_kernel_size,
            'hcnn_num_conv_blocks': hcnn_num_conv_blocks,
            'hcnn_chunk_size': hcnn_chunk_size,
            'hcnn_chunk_overlap': hcnn_chunk_overlap,
            'hcnn_max_chunks': hcnn_max_chunks,
            'hcnn_vocab_size': hcnn_vocab_size,
            'hcnn_gated_attention_dim': hcnn_gated_attention_dim,
            'hcnn_projection_dim': hcnn_projection_dim,
            'hcnn_dropout': hcnn_dropout,
            'hgru_embedding_dim': hgru_embedding_dim,
            'hgru_gru_hidden_dim': hgru_gru_hidden_dim,
            'hgru_num_gru_layers': hgru_num_gru_layers,
            'hgru_chunk_size': hgru_chunk_size,
            'hgru_chunk_overlap': hgru_chunk_overlap,
            'hgru_max_chunks': hgru_max_chunks,
            'hgru_vocab_size': hgru_vocab_size,
            'hgru_gated_attention_dim': hgru_gated_attention_dim,
            'hgru_projection_dim': hgru_projection_dim,
            'hgru_dropout': hgru_dropout,
            'scnn_embedding_dim': scnn_embedding_dim,
            'scnn_conv_dim': scnn_conv_dim,
            'scnn_kernel_size': scnn_kernel_size,
            'scnn_num_conv_blocks': scnn_num_conv_blocks,
            'scnn_max_length': scnn_max_length,
            'scnn_vocab_size': scnn_vocab_size,
            'scnn_gated_attention_dim': scnn_gated_attention_dim,
            'scnn_projection_dim': scnn_projection_dim,
            'scnn_dropout': scnn_dropout,
            'byte_embedding_dim': byte_embedding_dim,
            'byte_conv_dim': byte_conv_dim,
            'byte_kernel_size': byte_kernel_size,
            'byte_num_conv_blocks': byte_num_conv_blocks,
            'byte_chunk_size': byte_chunk_size,
            'byte_chunk_overlap': byte_chunk_overlap,
            'byte_max_chunks': byte_max_chunks,
            'byte_gated_attention_dim': byte_gated_attention_dim,
            'byte_projection_dim': byte_projection_dim,
            'byte_dropout': byte_dropout,
            'representation_dim': representation_dim,
            'hidden_dim': hidden_dim,
            'wx_nuisance_hidden_dim': wx_nuisance_hidden_dim,
            'wx_effect_hidden_dim': wx_effect_hidden_dim,
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
            'cf_effect_aux_outcome_weight': cf_effect_aux_outcome_weight,
            'cf_nuisance_potential_weight': cf_nuisance_potential_weight,
            'cf_effect_dr_weight': cf_effect_dr_weight,
            'cf_effect_dr_clip': cf_effect_dr_clip,
            'cf_effect_attention_entropy_weight': cf_effect_attention_entropy_weight,
            'cf_export_nuisance_potential_features': cf_export_nuisance_potential_features,
            'cf_export_shared_text_features': cf_export_shared_text_features,
            'cf_forest_x_mode': cf_forest_x_mode,
            'explicit_feature_specs': explicit_feature_specs,
            'explicit_feature_output_dim': explicit_feature_output_dim,
            'explicit_feature_hidden_dim': explicit_feature_hidden_dim,
            'explicit_feature_dropout': explicit_feature_dropout,
            'outcome_type': outcome_type,
        }

        # Store explicit confounder output dim for head input calculation
        self._explicit_feature_output_dim = explicit_feature_output_dim

        # Explicit confounder support (raw features for interpretability)
        self.explicit_feature_specs = explicit_feature_specs or []
        self._explicit_feature_means = {}
        self._explicit_feature_stds = {}
        self._explicit_features_fitted = False

        if self.explicit_feature_specs:
            # Calculate raw feature dimension
            raw_dim = 0
            for spec in self.explicit_feature_specs:
                if spec.type == "categorical":
                    n_cats = len(spec.categories) if spec.categories else 2
                    raw_dim += (n_cats - 1) + 1  # k-1 dummies + missing
                else:
                    raw_dim += 2  # value + missing
            self._explicit_feature_raw_dim = raw_dim
            logger.info(f"Explicit confounders: {len(self.explicit_feature_specs)} specs, "
                       f"raw feature dim: {raw_dim}")
        else:
            self._explicit_feature_raw_dim = 0

        self.explicit_nuisance_specs = filter_specs_by_role(self.explicit_feature_specs, "confounder")
        self.explicit_effect_specs = filter_specs_by_role(self.explicit_feature_specs, "effect_modifier")

        def make_role_featurizer(specs: List[ExplicitFeatureSpec], role: str):
            if not specs:
                return None
            logger.info(
                f"ExplicitFeatureFeaturizer for {role}: {len(specs)} features, "
                f"output_dim={explicit_feature_output_dim}"
            )
            return ExplicitFeatureFeaturizer(
                specs=specs,
                output_dim=explicit_feature_output_dim,
                hidden_dim=explicit_feature_hidden_dim,
                dropout=explicit_feature_dropout,
                device=str(self._device),
            )

        self.explicit_nuisance_featurizer = make_role_featurizer(
            self.explicit_nuisance_specs, "confounder/W"
        )
        self.explicit_effect_featurizer = make_role_featurizer(
            self.explicit_effect_specs, "effect_modifier/X"
        )
        self.explicit_feature_featurizer = self.explicit_nuisance_featurizer

        # Initialize feature extractor using factory
        self.feature_extractor = create_feature_extractor(
            extractor_type=self.feature_extractor_type,
            device=self._device,
            model_type="dragonnet",
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
            flp_chat_template_prompt=flp_chat_template_prompt,
            flp_attention_slots=flp_attention_slots,
            flp_document_window=flp_document_window,
            hlm_model_name=hlm_model_name,
            hlm_chunk_size=hlm_chunk_size,
            hlm_chunk_overlap=hlm_chunk_overlap,
            hlm_max_chunks=hlm_max_chunks,
            hlm_freeze_llm=hlm_freeze_llm,
            hlm_gated_attention_dim=hlm_gated_attention_dim,
            hlm_projection_dim=hlm_projection_dim,
            hlm_dropout=hlm_dropout,
            hlm_gradient_checkpointing=hlm_gradient_checkpointing,
            hlm_downprojection_dim=hlm_downprojection_dim,
            hlm_skip_llm=hlm_skip_llm,
            hlm_cached_hidden_size=hlm_cached_hidden_size,
            hlm_chat_template_prompt=hlm_chat_template_prompt,
            hcnn_embedding_dim=hcnn_embedding_dim,
            hcnn_conv_dim=hcnn_conv_dim,
            hcnn_kernel_size=hcnn_kernel_size,
            hcnn_num_conv_blocks=hcnn_num_conv_blocks,
            hcnn_chunk_size=hcnn_chunk_size,
            hcnn_chunk_overlap=hcnn_chunk_overlap,
            hcnn_max_chunks=hcnn_max_chunks,
            hcnn_vocab_size=hcnn_vocab_size,
            hcnn_gated_attention_dim=hcnn_gated_attention_dim,
            hcnn_projection_dim=hcnn_projection_dim,
            hcnn_dropout=hcnn_dropout,
            hgru_embedding_dim=hgru_embedding_dim,
            hgru_gru_hidden_dim=hgru_gru_hidden_dim,
            hgru_num_gru_layers=hgru_num_gru_layers,
            hgru_chunk_size=hgru_chunk_size,
            hgru_chunk_overlap=hgru_chunk_overlap,
            hgru_max_chunks=hgru_max_chunks,
            hgru_vocab_size=hgru_vocab_size,
            hgru_gated_attention_dim=hgru_gated_attention_dim,
            hgru_projection_dim=hgru_projection_dim,
            hgru_dropout=hgru_dropout,
            scnn_embedding_dim=scnn_embedding_dim,
            scnn_conv_dim=scnn_conv_dim,
            scnn_kernel_size=scnn_kernel_size,
            scnn_num_conv_blocks=scnn_num_conv_blocks,
            scnn_max_length=scnn_max_length,
            scnn_vocab_size=scnn_vocab_size,
            scnn_gated_attention_dim=scnn_gated_attention_dim,
            scnn_projection_dim=scnn_projection_dim,
            scnn_dropout=scnn_dropout,
            byte_embedding_dim=byte_embedding_dim,
            byte_conv_dim=byte_conv_dim,
            byte_kernel_size=byte_kernel_size,
            byte_num_conv_blocks=byte_num_conv_blocks,
            byte_chunk_size=byte_chunk_size,
            byte_chunk_overlap=byte_chunk_overlap,
            byte_max_chunks=byte_max_chunks,
            byte_gated_attention_dim=byte_gated_attention_dim,
            byte_projection_dim=byte_projection_dim,
            byte_dropout=byte_dropout,
        )

        logger.info(f"Using {self.feature_extractor_type.upper()} feature extractor")

        # Branch-specific Stage 1 heads.
        # W hidden state jointly predicts propensity and marginal outcome.
        # X hidden state predicts the R-learner tau head.
        base_input_dim = self.feature_extractor.output_dim
        nuisance_input_dim = base_input_dim
        effect_input_dim = base_input_dim
        if self.explicit_nuisance_featurizer is not None:
            nuisance_input_dim += explicit_feature_output_dim
        if self.explicit_effect_featurizer is not None:
            effect_input_dim += explicit_feature_output_dim
        self._head_input_dim = base_input_dim
        self._nuisance_input_dim = nuisance_input_dim
        self._effect_input_dim = effect_input_dim
        self._wx_hidden_dim = hidden_dim
        self._wx_nuisance_hidden_dim = wx_nuisance_hidden_dim
        self._wx_effect_hidden_dim = wx_effect_hidden_dim

        self.nuisance_branch = nn.Sequential(
            nn.Linear(nuisance_input_dim, wx_nuisance_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.propensity_head = nn.Linear(wx_nuisance_hidden_dim, 1)
        self.outcome_head = nn.Linear(wx_nuisance_hidden_dim, 1)
        self.nuisance_y0_head = nn.Linear(wx_nuisance_hidden_dim, 1)
        self.nuisance_y1_head = nn.Linear(wx_nuisance_hidden_dim, 1)

        self.effect_branch = nn.Sequential(
            nn.Linear(effect_input_dim, wx_effect_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.effect_tau_head = nn.Linear(wx_effect_hidden_dim, 1)
        self.effect_y0_head = nn.Linear(wx_effect_hidden_dim, 1)
        self.effect_y1_head = nn.Linear(wx_effect_hidden_dim, 1)
        input_dim = nuisance_input_dim

        # Optional staged R-learner representation training.
        # When enabled, the nuisance extractor/branch learns W for e(W), m(W),
        # while a separate effect extractor/branch learns X for tau(X) from
        # fixed out-of-fold nuisance predictions.
        self.use_rlearner_representation = cf_use_rlearner_representation
        self.cf_gamma_rlearner = cf_gamma_rlearner
        self.cf_effect_aux_outcome_weight = cf_effect_aux_outcome_weight
        self.cf_nuisance_potential_weight = cf_nuisance_potential_weight
        self.cf_effect_dr_weight = cf_effect_dr_weight
        self.cf_effect_dr_clip = cf_effect_dr_clip
        self.cf_effect_attention_entropy_weight = cf_effect_attention_entropy_weight
        self.cf_export_nuisance_potential_features = cf_export_nuisance_potential_features
        self.cf_export_shared_text_features = cf_export_shared_text_features
        self.cf_forest_x_mode = cf_forest_x_mode
        self.effect_feature_extractor = None

        if cf_use_rlearner_representation:
            self.effect_feature_extractor = create_feature_extractor(
                extractor_type=self.feature_extractor_type,
                device=self._device,
                model_type="dragonnet",
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
                flp_chat_template_prompt=flp_chat_template_prompt,
                flp_attention_slots=flp_attention_slots,
                flp_document_window=flp_document_window,
                hlm_model_name=hlm_model_name,
                hlm_chunk_size=hlm_chunk_size,
                hlm_chunk_overlap=hlm_chunk_overlap,
                hlm_max_chunks=hlm_max_chunks,
                hlm_freeze_llm=hlm_freeze_llm,
                hlm_gated_attention_dim=hlm_gated_attention_dim,
                hlm_projection_dim=hlm_projection_dim,
                hlm_dropout=hlm_dropout,
                hlm_gradient_checkpointing=hlm_gradient_checkpointing,
                hlm_downprojection_dim=hlm_downprojection_dim,
                hlm_skip_llm=hlm_skip_llm,
                hlm_cached_hidden_size=hlm_cached_hidden_size,
                hlm_chat_template_prompt=hlm_chat_template_prompt,
                hcnn_embedding_dim=hcnn_embedding_dim,
                hcnn_conv_dim=hcnn_conv_dim,
                hcnn_kernel_size=hcnn_kernel_size,
                hcnn_num_conv_blocks=hcnn_num_conv_blocks,
                hcnn_chunk_size=hcnn_chunk_size,
                hcnn_chunk_overlap=hcnn_chunk_overlap,
                hcnn_max_chunks=hcnn_max_chunks,
                hcnn_vocab_size=hcnn_vocab_size,
                hcnn_gated_attention_dim=hcnn_gated_attention_dim,
                hcnn_projection_dim=hcnn_projection_dim,
                hcnn_dropout=hcnn_dropout,
                hgru_embedding_dim=hgru_embedding_dim,
                hgru_gru_hidden_dim=hgru_gru_hidden_dim,
                hgru_num_gru_layers=hgru_num_gru_layers,
                hgru_chunk_size=hgru_chunk_size,
                hgru_chunk_overlap=hgru_chunk_overlap,
                hgru_max_chunks=hgru_max_chunks,
                hgru_vocab_size=hgru_vocab_size,
                hgru_gated_attention_dim=hgru_gated_attention_dim,
                hgru_projection_dim=hgru_projection_dim,
                hgru_dropout=hgru_dropout,
                scnn_embedding_dim=scnn_embedding_dim,
                scnn_conv_dim=scnn_conv_dim,
                scnn_kernel_size=scnn_kernel_size,
                scnn_num_conv_blocks=scnn_num_conv_blocks,
                scnn_max_length=scnn_max_length,
                scnn_vocab_size=scnn_vocab_size,
                scnn_gated_attention_dim=scnn_gated_attention_dim,
                scnn_projection_dim=scnn_projection_dim,
                scnn_dropout=scnn_dropout,
                byte_embedding_dim=byte_embedding_dim,
                byte_conv_dim=byte_conv_dim,
                byte_kernel_size=byte_kernel_size,
                byte_num_conv_blocks=byte_num_conv_blocks,
                byte_chunk_size=byte_chunk_size,
                byte_chunk_overlap=byte_chunk_overlap,
                byte_max_chunks=byte_max_chunks,
                byte_gated_attention_dim=byte_gated_attention_dim,
                byte_projection_dim=byte_projection_dim,
                byte_dropout=byte_dropout,
            )
            self.effect_head = None
            logger.info("  R-learner representation training: ENABLED (staged separate nets)")
            logger.info(f"    Nuisance extractor: {self.feature_extractor_type} -> W, e(W), m(W)")
            logger.info(f"    Effect extractor: {self.feature_extractor_type} -> X, tau(X)")
            if cf_effect_aux_outcome_weight > 0:
                logger.info(
                    "    Effect auxiliary factual outcome loss weight: %.3g",
                    cf_effect_aux_outcome_weight,
                )
            if cf_effect_dr_weight > 0:
                logger.info(
                    "    Effect doubly robust pseudo-outcome loss weight: %.3g",
                    cf_effect_dr_weight,
                )
            if cf_effect_attention_entropy_weight > 0:
                logger.info(
                    "    Effect attention entropy loss weight: %.3g",
                    cf_effect_attention_entropy_weight,
                )
        else:
            self.effect_head = None

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

    def fit_tokenizer(self, texts):
        """Fit tokenizer for trainable-from-scratch extractors. No-op for LLM-based."""
        if hasattr(self.feature_extractor, 'fit_tokenizer'):
            self.feature_extractor.fit_tokenizer(texts)
        if self.effect_feature_extractor is not None:
            if hasattr(self.effect_feature_extractor, 'fit_tokenizer'):
                self.effect_feature_extractor.fit_tokenizer(texts)

    def initialize_effect_from_nuisance(self) -> bool:
        """Warm-start the effect text encoder from the trained nuisance encoder."""
        if self.effect_feature_extractor is None:
            return False

        self.effect_feature_extractor.load_state_dict(
            self.feature_extractor.state_dict(),
            strict=True,
        )
        copied_branch = False
        if self._nuisance_input_dim == self._effect_input_dim:
            self.effect_branch.load_state_dict(self.nuisance_branch.state_dict(), strict=True)
            copied_branch = True

        logger.info(
            "Initialized effect encoder from nuisance encoder%s",
            " and copied branch weights" if copied_branch else "",
        )
        return True

    def nuisance_parameters(self):
        """Return trainable parameters used by the nuisance stage."""
        modules = [
            self.feature_extractor,
            self.nuisance_branch,
            self.propensity_head,
            self.outcome_head,
            self.nuisance_y0_head,
            self.nuisance_y1_head,
        ]
        if self.explicit_nuisance_featurizer is not None:
            modules.append(self.explicit_nuisance_featurizer)
        return chain.from_iterable(module.parameters() for module in modules)

    def effect_parameters(self):
        """Return trainable parameters used by the effect/R-loss stage."""
        modules = [
            self.effect_branch,
            self.effect_tau_head,
            self.effect_y0_head,
            self.effect_y1_head,
        ]
        if self.effect_feature_extractor is not None:
            modules.insert(0, self.effect_feature_extractor)
        else:
            modules.insert(0, self.feature_extractor)
        if self.explicit_effect_featurizer is not None:
            modules.append(self.explicit_effect_featurizer)
        return chain.from_iterable(module.parameters() for module in modules)

    @staticmethod
    def _get_extractor_input(batch, texts):
        """Return preprocessed batch if available, otherwise raw texts."""
        if 'cached_hidden_states' in batch:
            result = {
                'cached_hidden_states': batch['cached_hidden_states'],
                'cached_attention_mask': batch['cached_attention_mask'],
                'texts': texts,
            }
            if 'sample_chunk_counts' in batch:
                result['sample_chunk_counts'] = batch['sample_chunk_counts']
            return result
        if 'chunk_input_ids' in batch or 'chunk_token_ids' in batch:
            return batch
        return texts

    def _append_role_features(
        self,
        text_features: torch.Tensor,
        explicit_feature_values: Optional[List[Dict[str, Any]]],
        role: str,
    ) -> torch.Tensor:
        """Append role-specific explicit feature embeddings to text features."""
        featurizer = (
            self.explicit_nuisance_featurizer
            if role == "confounder"
            else self.explicit_effect_featurizer
        )
        if featurizer is None:
            return text_features
        if explicit_feature_values is None:
            raise ValueError(
                f"Explicit features with role '{role}' are configured, but no "
                "explicit_feature_values were provided."
            )
        role_features = featurizer(explicit_feature_values)
        return torch.cat([text_features, role_features], dim=1)

    def _nuisance_forward(
        self,
        text_features: torch.Tensor,
        explicit_feature_values: Optional[List[Dict[str, Any]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return W hidden state plus propensity/outcome logits."""
        nuisance_input = self._append_role_features(
            text_features, explicit_feature_values, "confounder"
        )
        w_hidden = self.nuisance_branch(nuisance_input)
        propensity_logit = self.propensity_head(w_hidden)
        outcome_logit = self.outcome_head(w_hidden)
        return w_hidden, propensity_logit, outcome_logit

    def _nuisance_potential_forward(
        self,
        w_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return treatment-specific nuisance outcome logits from W."""
        return self.nuisance_y0_head(w_hidden), self.nuisance_y1_head(w_hidden)

    def _effect_forward(
        self,
        text_features: torch.Tensor,
        explicit_feature_values: Optional[List[Dict[str, Any]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return X hidden state plus tau prediction."""
        effect_input = self._append_role_features(
            text_features, explicit_feature_values, "effect_modifier"
        )
        x_hidden = self.effect_branch(effect_input)
        tau = self.effect_tau_head(x_hidden)
        return x_hidden, tau

    def forward(
        self,
        texts_or_batch,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through neural components.

        Args:
            texts_or_batch: List of text strings or preprocessed batch dict from DataLoader
            explicit_feature_values: Optional list of dicts with explicit confounder values

        Returns:
            features: Extracted features (batch, feature_dim)
            propensity_logit: Propensity prediction (batch, 1)
            outcome_logit: Outcome prediction (batch, 1)
        """
        if isinstance(texts_or_batch, dict):
            texts = texts_or_batch['texts']
            extractor_input = self._get_extractor_input(texts_or_batch, texts)
            if explicit_feature_values is None:
                explicit_feature_values = texts_or_batch.get('explicit_feature_values', None)
        else:
            extractor_input = texts_or_batch

        text_features = self.feature_extractor(extractor_input)
        w_hidden, propensity_logit, outcome_logit = self._nuisance_forward(
            text_features, explicit_feature_values
        )
        return w_hidden, propensity_logit, outcome_logit

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

    def train_representation_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        gamma_rlearner: float = 1.0,
        label_smoothing: float = 0.0,
        stop_grad_propensity: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compatibility wrapper for nuisance representation training.

        Staged R-learner representation training now uses train_nuisance_step()
        followed by train_effect_r_step() with out-of-fold nuisance predictions.
        This method intentionally trains only e(W) and m(W).

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys.
                   Optional 'explicit_feature_values' for explicit confounders.
            alpha_propensity: Weight for propensity loss
            gamma_rlearner: Weight for R-learner loss (only used if use_rlearner_representation=True)
            label_smoothing: Label smoothing factor
            stop_grad_propensity: If True, detach features before propensity

        Returns:
            Dictionary with loss components
        """
        del gamma_rlearner
        result = self.train_nuisance_step(
            batch=batch,
            alpha_propensity=alpha_propensity,
            label_smoothing=label_smoothing,
            stop_grad_propensity=stop_grad_propensity,
        )
        result['r_loss'] = torch.tensor(0.0, device=self._device)
        return result

    def train_nuisance_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        label_smoothing: float = 0.0,
        stop_grad_propensity: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Train e(W) and m(W) from the nuisance representation."""
        texts = batch['texts']
        treatments = batch['treatment']
        outcomes = batch['outcome']
        explicit_feature_values = batch.get('explicit_feature_values', None)
        extractor_input = self._get_extractor_input(batch, texts)

        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            outcomes_smooth = (
                outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
                if self.outcome_type == "binary"
                else outcomes
            )
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        text_features = self.feature_extractor(extractor_input)
        w_hidden, propensity_logit, outcome_logit = self._nuisance_forward(
            text_features, explicit_feature_values
        )

        if stop_grad_propensity:
            _, detached_propensity_logit, _ = self._nuisance_forward(
                text_features.detach(), explicit_feature_values
            )
            propensity_logit_for_loss = detached_propensity_logit
        else:
            propensity_logit_for_loss = propensity_logit

        propensity_loss = F.binary_cross_entropy_with_logits(
            propensity_logit_for_loss.squeeze(-1),
            treatments_smooth
        )
        outcome_loss = self._outcome_loss(outcome_logit.squeeze(-1), outcomes_smooth)
        potential_loss = torch.tensor(0.0, device=self._device)
        if self.cf_nuisance_potential_weight > 0:
            y0_logit, y1_logit = self._nuisance_potential_forward(w_hidden)
            factual_logit = torch.where(treatments.unsqueeze(1) > 0.5, y1_logit, y0_logit)
            potential_loss = self._outcome_loss(factual_logit.squeeze(-1), outcomes_smooth)
        total_loss = (
            outcome_loss
            + alpha_propensity * propensity_loss
            + self.cf_nuisance_potential_weight * potential_loss
        )

        return {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'potential_loss': potential_loss.detach(),
            'propensity_logit': propensity_logit.detach(),
            'outcome_logit': outcome_logit.detach(),
            'w_hidden': w_hidden.detach(),
        }

    def train_effect_r_step(
        self,
        batch: Dict[str, Any],
        e_hat: torch.Tensor,
        m_hat: torch.Tensor,
        gamma_rlearner: float = 1.0,
        mu0_hat: Optional[torch.Tensor] = None,
        mu1_hat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train tau(X) from fixed nuisance predictions using the R-loss."""
        texts = batch['texts']
        treatments = batch['treatment']
        outcomes = batch['outcome']
        explicit_feature_values = batch.get('explicit_feature_values', None)
        extractor_input = self._get_extractor_input(batch, texts)

        extractor = self.effect_feature_extractor or self.feature_extractor
        effect_text_features = extractor(extractor_input)
        x_hidden, tau = self._effect_forward(effect_text_features, explicit_feature_values)

        e_hat = e_hat.to(self._device).float().clamp(0.01, 0.99)
        m_hat = m_hat.to(self._device).float()
        y_residual = outcomes - m_hat
        t_residual = treatments - e_hat
        r_loss = ((y_residual - tau.squeeze(-1) * t_residual) ** 2).mean()
        total_loss = gamma_rlearner * r_loss

        dr_loss = torch.tensor(0.0, device=self._device)
        if self.cf_effect_dr_weight > 0 and mu0_hat is not None and mu1_hat is not None:
            mu0_hat = mu0_hat.to(self._device).float()
            mu1_hat = mu1_hat.to(self._device).float()
            dr_tau = (
                mu1_hat - mu0_hat
                + treatments * (outcomes - mu1_hat) / e_hat
                - (1.0 - treatments) * (outcomes - mu0_hat) / (1.0 - e_hat)
            )
            if self.cf_effect_dr_clip is not None and self.cf_effect_dr_clip > 0:
                dr_tau = dr_tau.clamp(-self.cf_effect_dr_clip, self.cf_effect_dr_clip)
            overlap_weight = (e_hat * (1.0 - e_hat)).detach().clamp_min(1e-4)
            overlap_weight = overlap_weight / overlap_weight.mean().clamp_min(1e-6)
            dr_loss = (overlap_weight * (tau.squeeze(-1) - dr_tau.detach()) ** 2).mean()
            total_loss = total_loss + self.cf_effect_dr_weight * dr_loss

        effect_outcome_loss = torch.tensor(0.0, device=self._device)
        if self.cf_effect_aux_outcome_weight > 0:
            y0_logit = self.effect_y0_head(x_hidden).squeeze(-1)
            y1_logit = self.effect_y1_head(x_hidden).squeeze(-1)
            factual_logit = torch.where(treatments > 0.5, y1_logit, y0_logit)
            effect_outcome_loss = self._outcome_loss(factual_logit, outcomes)
            total_loss = total_loss + self.cf_effect_aux_outcome_weight * effect_outcome_loss

        attention_entropy_loss = torch.tensor(0.0, device=self._device)
        if self.cf_effect_attention_entropy_weight > 0:
            entropy_fn = getattr(extractor, "attention_entropy_loss", None)
            if entropy_fn is not None:
                entropy = entropy_fn()
                if entropy is not None:
                    attention_entropy_loss = entropy
                    total_loss = (
                        total_loss
                        + self.cf_effect_attention_entropy_weight * attention_entropy_loss
                    )

        return {
            'loss': total_loss,
            'r_loss': r_loss.detach(),
            'dr_loss': dr_loss.detach(),
            'effect_outcome_loss': effect_outcome_loss.detach(),
            'attention_entropy_loss': attention_entropy_loss.detach(),
            'tau': tau.detach(),
            'x_hidden': x_hidden.detach(),
        }

    def predict_nuisance(
        self,
        texts_or_loader,
        batch_size: int = 32,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        gpu_store=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ordered propensity and marginal-outcome predictions."""
        from torch.utils.data import DataLoader

        if explicit_feature_values is None:
            explicit_feature_values = explicit_confounder_values

        self.eval()
        all_propensity = []
        all_outcome = []
        is_batch_iterable = isinstance(texts_or_loader, DataLoader) or (
            hasattr(texts_or_loader, '__iter__') and not isinstance(texts_or_loader, (list, str))
        )

        with torch.no_grad():
            if is_batch_iterable:
                for batch in texts_or_loader:
                    prepare_cached_batch(batch, self._device, gpu_store=gpu_store)
                    texts = batch['texts']
                    extractor_input = self._get_extractor_input(batch, texts)
                    batch_feature_values = batch.get('explicit_feature_values', None)
                    text_features = self.feature_extractor(extractor_input)
                    _, prop_logit, outcome_logit = self._nuisance_forward(
                        text_features, batch_feature_values
                    )
                    all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                    all_outcome.append(self._outcome_activation(outcome_logit).cpu().numpy())
            else:
                texts = texts_or_loader
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_feature_values = None
                    if explicit_feature_values is not None:
                        batch_feature_values = explicit_feature_values[i:i + batch_size]
                    text_features = self.feature_extractor(batch_texts)
                    _, prop_logit, outcome_logit = self._nuisance_forward(
                        text_features, batch_feature_values
                    )
                    all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                    all_outcome.append(self._outcome_activation(outcome_logit).cpu().numpy())

        return (
            np.vstack(all_propensity).flatten(),
            np.vstack(all_outcome).flatten(),
        )

    def predict_nuisance_components(
        self,
        texts_or_loader,
        batch_size: int = 32,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        gpu_store=None,
    ) -> Dict[str, np.ndarray]:
        """Return propensity, marginal outcome, and treatment-specific nuisance means."""
        from torch.utils.data import DataLoader

        if explicit_feature_values is None:
            explicit_feature_values = explicit_confounder_values

        self.eval()
        all_propensity = []
        all_outcome = []
        all_mu0 = []
        all_mu1 = []
        is_batch_iterable = isinstance(texts_or_loader, DataLoader) or (
            hasattr(texts_or_loader, '__iter__') and not isinstance(texts_or_loader, (list, str))
        )

        def process_batch(extractor_input, batch_feature_values):
            text_features = self.feature_extractor(extractor_input)
            w_hidden, prop_logit, outcome_logit = self._nuisance_forward(
                text_features, batch_feature_values
            )
            y0_logit, y1_logit = self._nuisance_potential_forward(w_hidden)
            return prop_logit, outcome_logit, y0_logit, y1_logit

        with torch.no_grad():
            if is_batch_iterable:
                for batch in texts_or_loader:
                    prepare_cached_batch(batch, self._device, gpu_store=gpu_store)
                    texts = batch['texts']
                    extractor_input = self._get_extractor_input(batch, texts)
                    batch_feature_values = batch.get('explicit_feature_values', None)
                    prop_logit, outcome_logit, y0_logit, y1_logit = process_batch(
                        extractor_input, batch_feature_values
                    )
                    all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                    all_outcome.append(self._outcome_activation(outcome_logit).cpu().numpy())
                    all_mu0.append(self._outcome_activation(y0_logit).cpu().numpy())
                    all_mu1.append(self._outcome_activation(y1_logit).cpu().numpy())
            else:
                texts = texts_or_loader
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_feature_values = None
                    if explicit_feature_values is not None:
                        batch_feature_values = explicit_feature_values[i:i + batch_size]
                    prop_logit, outcome_logit, y0_logit, y1_logit = process_batch(
                        batch_texts, batch_feature_values
                    )
                    all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                    all_outcome.append(self._outcome_activation(outcome_logit).cpu().numpy())
                    all_mu0.append(self._outcome_activation(y0_logit).cpu().numpy())
                    all_mu1.append(self._outcome_activation(y1_logit).cpu().numpy())

        return {
            'propensity': np.vstack(all_propensity).flatten(),
            'outcome': np.vstack(all_outcome).flatten(),
            'mu0': np.vstack(all_mu0).flatten(),
            'mu1': np.vstack(all_mu1).flatten(),
        }

    # Alias for API consistency with CausalText
    def train_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        gamma_rlearner: float = 1.0,
        label_smoothing: float = 0.0,
        stop_grad_propensity: bool = False,
        **kwargs  # Ignore extra args like beta_targreg for compatibility
    ) -> Dict[str, torch.Tensor]:
        """Alias for train_representation_step for API consistency."""
        return self.train_representation_step(
            batch=batch,
            alpha_propensity=alpha_propensity,
            gamma_rlearner=gamma_rlearner,
            label_smoothing=label_smoothing,
            stop_grad_propensity=stop_grad_propensity
        )

    @staticmethod
    def _hstack_optional(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Horizontally stack optional 2D matrices, treating zero-width as absent."""
        if right is not None and right.shape[1] == 0:
            right = None
        if left is not None and left.shape[1] == 0:
            left = None
        if left is None:
            return right
        if right is None:
            return left
        return np.hstack([left, right])

    def extract_forest_features(
        self,
        texts_or_loader,
        batch_size: int = 32,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        gpu_store=None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Extract EconML X/W feature matrices plus nuisance predictions.

        X receives effect-modifier information: learned X activations and raw
        explicit features with role="effect_modifier". W receives confounder
        information: learned W activations and raw explicit features with
        role="confounder". Both-role explicit features appear in both matrices.
        """
        from torch.utils.data import DataLoader

        if explicit_feature_values is None:
            explicit_feature_values = explicit_confounder_values

        self.eval()
        all_x = []
        all_w = []
        all_propensity = []
        all_outcome = []
        all_feature_values = []

        use_wx_activations = self.use_rlearner_representation

        is_batch_iterable = isinstance(texts_or_loader, DataLoader) or (
            hasattr(texts_or_loader, '__iter__') and not isinstance(texts_or_loader, (list, str))
        )

        def process_batch(extractor_input, batch_feature_values):
            text_features = self.feature_extractor(extractor_input)
            w_hidden, prop_logit, outcome_logit = self._nuisance_forward(
                text_features, batch_feature_values
            )
            role_features = None
            if hasattr(self.feature_extractor, "extract_role_features"):
                role_features = self.feature_extractor.extract_role_features(extractor_input)
            shared_w_features = None
            shared_x_features = None
            if (
                self.cf_export_shared_text_features
                and hasattr(self.feature_extractor, "extract_shared_forest_features")
            ):
                shared_w_features = self.feature_extractor.extract_shared_forest_features(
                    extractor_input,
                    text_features=text_features,
                )

            if use_wx_activations:
                effect_extractor = self.effect_feature_extractor or self.feature_extractor
                effect_text_features = (
                    effect_extractor(extractor_input)
                    if effect_extractor is not self.feature_extractor
                    else text_features
                )
                x_hidden, tau_pred = self._effect_forward(effect_text_features, batch_feature_values)
                if self.cf_forest_x_mode == "x_hidden":
                    x_matrix = x_hidden
                elif self.cf_forest_x_mode == "tau":
                    x_matrix = tau_pred
                else:
                    x_matrix = torch.cat([x_hidden, tau_pred], dim=1)
                w_matrix = w_hidden
                if (
                    self.cf_export_shared_text_features
                    and hasattr(effect_extractor, "extract_shared_forest_features")
                ):
                    shared_x_features = effect_extractor.extract_shared_forest_features(
                        extractor_input,
                        text_features=effect_text_features,
                    )
            else:
                x_matrix = text_features
                w_matrix = None
                shared_x_features = shared_w_features

            if shared_x_features is not None and shared_x_features.shape[1] > 0:
                x_matrix = torch.cat(
                    [x_matrix, shared_x_features.to(x_matrix.device)],
                    dim=1,
                )
            if shared_w_features is not None and shared_w_features.shape[1] > 0:
                if w_matrix is None:
                    w_matrix = shared_w_features.to(x_matrix.device)
                else:
                    w_matrix = torch.cat(
                        [w_matrix, shared_w_features.to(w_matrix.device)],
                        dim=1,
                    )

            if self.cf_export_nuisance_potential_features:
                y0_logit, y1_logit = self._nuisance_potential_forward(w_hidden)
                mu0 = self._outcome_activation(y0_logit)
                mu1 = self._outcome_activation(y1_logit)
                nuisance_effect_features = torch.cat([mu0, mu1, mu1 - mu0], dim=1)
                x_matrix = torch.cat(
                    [x_matrix, nuisance_effect_features.to(x_matrix.device)],
                    dim=1,
                )

            if role_features is not None:
                raw_x = role_features.get("effect_modifier")
                raw_w = role_features.get("confounder")
                if raw_x is not None and raw_x.shape[1] > 0:
                    x_matrix = torch.cat([x_matrix, raw_x.to(x_matrix.device)], dim=1)
                if raw_w is not None and raw_w.shape[1] > 0:
                    if w_matrix is None:
                        w_matrix = raw_w.to(x_matrix.device)
                    else:
                        w_matrix = torch.cat([w_matrix, raw_w.to(w_matrix.device)], dim=1)

            return x_matrix, w_matrix, prop_logit, outcome_logit

        with torch.no_grad():
            if is_batch_iterable:
                for batch in texts_or_loader:
                    prepare_cached_batch(batch, self._device, gpu_store=gpu_store)
                    texts = batch['texts']
                    extractor_input = self._get_extractor_input(batch, texts)
                    batch_feature_values = batch.get('explicit_feature_values', None)
                    x_matrix, w_matrix, prop_logit, outcome_logit = process_batch(
                        extractor_input, batch_feature_values
                    )
                    all_x.append(x_matrix.cpu().numpy())
                    if w_matrix is not None:
                        all_w.append(w_matrix.cpu().numpy())
                    all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                    all_outcome.append(self._outcome_activation(outcome_logit).cpu().numpy())
                    if batch_feature_values is not None:
                        all_feature_values.extend(batch_feature_values)
            else:
                texts = texts_or_loader
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_feature_values = None
                    if explicit_feature_values is not None:
                        batch_feature_values = explicit_feature_values[i:i + batch_size]
                    x_matrix, w_matrix, prop_logit, outcome_logit = process_batch(
                        batch_texts, batch_feature_values
                    )
                    all_x.append(x_matrix.cpu().numpy())
                    if w_matrix is not None:
                        all_w.append(w_matrix.cpu().numpy())
                    all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                    all_outcome.append(self._outcome_activation(outcome_logit).cpu().numpy())

        x_features = np.vstack(all_x)
        w_features = np.vstack(all_w) if all_w else None

        feature_values_for_raw = all_feature_values if all_feature_values else explicit_feature_values
        if feature_values_for_raw is not None and self.explicit_feature_specs:
            raw_w = self._get_raw_explicit_features(feature_values_for_raw, role="confounder")
            raw_x = self._get_raw_explicit_features(feature_values_for_raw, role="effect_modifier")
            x_features = self._hstack_optional(x_features, raw_x)
            w_features = self._hstack_optional(w_features, raw_w)

        return (
            x_features,
            w_features,
            np.vstack(all_propensity).flatten(),
            np.vstack(all_outcome).flatten(),
        )

    def extract_features(
        self,
        texts_or_loader,
        batch_size: int = 32,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        gpu_store=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features and nuisance predictions for all texts.

        In staged R-learner representation mode, features are extracted from
        the effect branch (X). Propensity and outcome predictions still come
        from the nuisance branch.

        Args:
            texts_or_loader: List of all text strings, or a DataLoader yielding batch dicts
            batch_size: Batch size for processing (only used when texts_or_loader is a list)
            explicit_feature_values: Optional list of dicts with confounder values.
                If provided and explicit_feature_specs is set, raw confounder features
                are concatenated to neural features. Ignored when using DataLoader
                (confounder values come from batch dicts).

        Returns:
            features: Feature matrix (n_samples, feature_dim + confounder_dim)
            propensity: Propensity predictions (n_samples,)
            outcome_pred: Outcome predictions (n_samples,)
        """
        x_features, _, propensity, outcome_pred = self.extract_forest_features(
            texts_or_loader,
            batch_size=batch_size,
            explicit_feature_values=explicit_feature_values or explicit_confounder_values,
            gpu_store=gpu_store,
        )
        return x_features, propensity, outcome_pred


    def train_causal_forest(
        self,
        texts_or_loader,
        T: np.ndarray,
        Y: np.ndarray,
        batch_size: int = 32,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        gpu_store=None
    ) -> 'CausalTextForest':
        """
        Train causal forest on extracted features.

        Should be called after representation training is complete.
        The causal forest uses sklearn random forests for nuisance estimation
        on the neural network's learned features.

        Args:
            texts_or_loader: List of training texts, or a DataLoader yielding batch dicts
            T: Treatment indicators
            Y: Outcome indicators
            batch_size: Batch size for feature extraction (only used with raw texts)
            explicit_feature_values: Optional list of dicts with confounder values.
                If provided, raw confounder features are concatenated to neural features.
            gpu_store: Optional GPUHiddenStateStore for GPU-resident hidden states.

        Returns:
            self
        """
        logger.info("Extracting X/W features for causal forest training...")
        x_features, w_features, _, _ = self.extract_forest_features(
            texts_or_loader, batch_size,
            explicit_feature_values=explicit_feature_values or explicit_confounder_values,
            gpu_store=gpu_store
        )

        logger.info(
            f"  Causal forest X features: {x_features.shape[1]}, "
            f"W controls: {w_features.shape[1] if w_features is not None else 0}"
        )

        self.causal_forest.fit(
            X=x_features,
            W=w_features,
            T=T,
            Y=Y
        )

        return self

    def predict(
        self,
        texts_or_loader,
        batch_size: int = 32,
        return_ci: bool = True,
        alpha: float = 0.05,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        gpu_store=None
    ) -> Dict[str, np.ndarray]:
        """
        Predict ITEs using trained causal forest.

        Args:
            texts_or_loader: List of text strings, or a DataLoader yielding batch dicts
            batch_size: Batch size for feature extraction (only used with raw texts)
            return_ci: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            explicit_feature_values: Optional list of dicts with confounder values.
                Must be provided if model was trained with explicit confounders.
            gpu_store: Optional GPUHiddenStateStore for GPU-resident hidden states.

        Returns:
            Dictionary with predictions:
                - tau_pred: ITE estimates
                - propensity: Propensity scores from neural network
                - outcome_pred: Outcome predictions from neural network
                - tau_lower, tau_upper: Confidence intervals (if return_ci)
        """
        # Extract X/W features (W is only needed at fit time; predictions use X).
        x_features, _, propensity, outcome_pred = self.extract_forest_features(
            texts_or_loader, batch_size,
            explicit_feature_values=explicit_feature_values or explicit_confounder_values,
            gpu_store=gpu_store
        )

        # Get ITE predictions from causal forest
        cf_preds = self.causal_forest.predict(x_features, return_ci=return_ci, alpha=alpha)

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
        y0_prob = outcome_pred - propensity * tau
        y1_prob = outcome_pred + (1 - propensity) * tau
        if self.outcome_type == "binary":
            y0_prob = np.clip(y0_prob, 0, 1)
            y1_prob = np.clip(y1_prob, 0, 1)

        result['pred_y0_prob'] = y0_prob
        result['pred_y1_prob'] = y1_prob

        if 'tau_lower' in cf_preds:
            result['tau_lower'] = cf_preds['tau_lower']
            result['tau_upper'] = cf_preds['tau_upper']

        return result

    def fit_explicit_feature_featurizer(
        self,
        feature_values_list: List[Dict[str, Any]]
    ) -> 'CausalTextForest':
        """
        Fit the explicit confounder featurizer (MLP) on training data.

        This computes normalization statistics (mean/std) for continuous confounders
        used during Stage 1 neural network training. Must be called before training
        if explicit confounders are used.

        Args:
            feature_values_list: List of dicts with confounder values from training data.
                Each dict should have "{name}" and "{name}_missing" keys.

        Returns:
            self for method chaining
        """
        if self.explicit_nuisance_featurizer is not None:
            self.explicit_nuisance_featurizer.fit(feature_values_list)
            logger.info("Fitted confounder-role ExplicitFeatureFeaturizer for Stage 1 training")
        if self.explicit_effect_featurizer is not None and self.explicit_effect_featurizer is not self.explicit_nuisance_featurizer:
            self.explicit_effect_featurizer.fit(feature_values_list)
            logger.info("Fitted effect-modifier-role ExplicitFeatureFeaturizer for Stage 1 training")
        return self

    def fit_explicit_confounder_featurizer(
        self,
        confounder_values_list: List[Dict[str, Any]]
    ) -> 'CausalTextForest':
        """Backward-compatible alias for fit_explicit_feature_featurizer."""
        return self.fit_explicit_feature_featurizer(confounder_values_list)

    def fit_explicit_features(
        self,
        feature_values_list: List[Dict[str, Any]]
    ) -> 'CausalTextForest':
        """
        Compute normalization statistics for explicit confounders from training data.

        For causal forest, we use raw features (no MLP) for interpretability.
        This method computes mean/std for continuous confounders.

        Args:
            feature_values_list: List of dicts with confounder values.
                Keys should match spec.name (e.g., "age", not "explicit_conf_age").

        Returns:
            self for method chaining
        """
        if not self.explicit_feature_specs:
            return self

        # Collect continuous values
        continuous_values = {
            spec.name: [] for spec in self.explicit_feature_specs if spec.type == "continuous"
        }

        for values in feature_values_list:
            for spec in self.explicit_feature_specs:
                if spec.type == "continuous":
                    val = values.get(spec.name)
                    missing = values.get(f"{spec.name}_missing", val is None)
                    if not missing and val is not None:
                        continuous_values[spec.name].append(float(val))

        # Compute mean and std for each continuous confounder
        for name, vals in continuous_values.items():
            if vals:
                self._explicit_feature_means[name] = sum(vals) / len(vals)
                variance = sum((v - self._explicit_feature_means[name]) ** 2 for v in vals) / len(vals)
                self._explicit_feature_stds[name] = max(variance ** 0.5, 1e-6)
            else:
                self._explicit_feature_means[name] = 0.0
                self._explicit_feature_stds[name] = 1.0

        self._explicit_features_fitted = True
        logger.info(f"Fitted explicit confounders on {len(feature_values_list)} samples")
        return self

    def fit_explicit_confounders(
        self,
        confounder_values_list: List[Dict[str, Any]]
    ) -> 'CausalTextForest':
        """Backward-compatible alias for fit_explicit_features."""
        return self.fit_explicit_features(confounder_values_list)

    def _get_raw_explicit_features(
        self,
        feature_values_list: List[Dict[str, Any]],
        role: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get raw confounder features as numpy array.

        Args:
            feature_values_list: List of dicts with confounder values

        Returns:
            (n_samples, raw_dim) numpy array
        """
        if not self.explicit_feature_specs:
            return np.zeros((len(feature_values_list), 0))

        features, _ = get_raw_explicit_features(
            feature_values_list,
            self.explicit_feature_specs,
            continuous_means=self._explicit_feature_means,
            continuous_stds=self._explicit_feature_stds,
            role=role,
        )
        return np.array(features)

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)

    def get_features(self, texts_or_batch) -> torch.Tensor:
        """
        Extract feature representations from texts or batch dict.

        Args:
            texts_or_batch: List of text strings or preprocessed batch dict

        Returns:
            Feature tensor: (batch, output_dim)
        """
        with torch.no_grad():
            if isinstance(texts_or_batch, dict):
                texts = texts_or_batch['texts']
                extractor_input = self._get_extractor_input(texts_or_batch, texts)
            else:
                extractor_input = texts_or_batch
            return self.feature_extractor(extractor_input)

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

        # Save role-specific explicit feature featurizer state if enabled
        if self.explicit_nuisance_featurizer is not None:
            checkpoint['explicit_nuisance_featurizer_state'] = self.explicit_nuisance_featurizer.get_state()
        if self.explicit_effect_featurizer is not None:
            checkpoint['explicit_effect_featurizer_state'] = self.explicit_effect_featurizer.get_state()

        # Save staged effect extractor state when enabled.
        if self.effect_feature_extractor is not None:
            checkpoint['effect_feature_extractor'] = self.effect_feature_extractor.state_dict()
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
