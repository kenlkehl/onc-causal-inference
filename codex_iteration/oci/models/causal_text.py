# oci/models/causal_text.py
"""Causal inference model using frozen LLM pooler for text representation."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dragonnet import DragonNet
from .rlearner import RLearnerNet
from .explicit_feature_featurizer import ExplicitFeatureFeaturizer
from .extractor_factory import create_feature_extractor
from ..config import normalize_feature_extractor_type, ExplicitFeatureSpec


logger = logging.getLogger(__name__)


class CausalText(nn.Module):
    """
    Causal inference model for text using frozen LLM pooler feature extraction.

    Architecture:
    - Frozen LLM Pooler extracts features from text via gated attention pooling
    - DragonNet or RLearnerNet predicts outcomes and propensity

    Frozen LLM Pooler mode (feature_extractor_type="frozen_llm_pooler"):
    - Pretrained decoder-only LLM with frozen weights
    - GatedAttentionPooling over all token hidden states
    - No fit_tokenizer() needed (uses pretrained tokenizer)
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
        # Causal head args (applies to all causal heads: DragonNet, RLearner)
        causal_head_representation_dim: int = 128,
        causal_head_hidden_outcome_dim: int = 64,
        causal_head_dropout: float = 0.2,
        device: str = "cuda:0",
        model_type: str = "dragonnet",  # "dragonnet" or "rlearner"
        # Auxiliary features (for hybrid text + categorical models)
        auxiliary_dim: int = 0,  # Dimension of auxiliary categorical features (0 = no auxiliary)
        # Explicit feature featurizer args
        explicit_feature_specs: Optional[List[ExplicitFeatureSpec]] = None,
        explicit_feature_output_dim: int = 64,
        explicit_feature_hidden_dim: int = 128,
        explicit_feature_dropout: float = 0.1,
        # Backward-compatible aliases for old checkpoints/callers.
        explicit_confounder_specs: Optional[List[ExplicitFeatureSpec]] = None,
        explicit_confounder_output_dim: int = 64,
        explicit_confounder_hidden_dim: int = 128,
        explicit_confounder_dropout: float = 0.1,
        # Outcome type
        outcome_type: str = "binary",  # "binary" or "continuous"
    ):
        """
        Initialize causal inference model with frozen LLM pooler feature extractor.

        Args:
            feature_extractor_type: Feature extractor type (default: "frozen_llm_pooler")
            flp_model_name: HuggingFace model name for frozen LLM
            flp_max_length: Maximum sequence length
            flp_freeze_llm: Whether to freeze LLM weights
            flp_gated_attention_dim: Gated attention hidden dimension
            flp_projection_dim: Output projection dimension
            flp_dropout: Dropout rate
            flp_gradient_checkpointing: Enable gradient checkpointing
            flp_downprojection_dim: Optional downprojection dimension before pooling
            flp_skip_llm: Skip LLM forward (for cached mode)
            flp_cached_hidden_size: Hidden size for cached hidden states
            causal_head_representation_dim: Causal head representation dimension
            causal_head_hidden_outcome_dim: Causal head outcome hidden dimension
            causal_head_dropout: Dropout rate for causal head layers
            device: Device string
            model_type: Architecture type ("dragonnet" or "rlearner")
            auxiliary_dim: Dimension of auxiliary categorical features (0 = disabled)
        """
        super().__init__()

        self._device = torch.device(device)
        self.model_type = model_type
        self.outcome_type = outcome_type
        # Normalize feature extractor type
        self.feature_extractor_type = normalize_feature_extractor_type(feature_extractor_type)
        if explicit_feature_specs is None:
            explicit_feature_specs = explicit_confounder_specs
            explicit_feature_output_dim = explicit_confounder_output_dim
            explicit_feature_hidden_dim = explicit_confounder_hidden_dim
            explicit_feature_dropout = explicit_confounder_dropout

        # Store config for checkpointing (store original type for reproducibility)
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
            'causal_head_representation_dim': causal_head_representation_dim,
            'causal_head_hidden_outcome_dim': causal_head_hidden_outcome_dim,
            'causal_head_dropout': causal_head_dropout,
            'model_type': model_type,
            'auxiliary_dim': auxiliary_dim,
            'explicit_feature_specs': explicit_feature_specs,
            'explicit_feature_output_dim': explicit_feature_output_dim,
            'explicit_feature_hidden_dim': explicit_feature_hidden_dim,
            'explicit_feature_dropout': explicit_feature_dropout,
            'outcome_type': outcome_type,
        }

        # Store auxiliary dimension
        self.auxiliary_dim = auxiliary_dim

        # Initialize feature extractor using factory
        self.feature_extractor = create_feature_extractor(
            extractor_type=self.feature_extractor_type,
            device=self._device,
            model_type=model_type,
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
            flp_chat_template_prompt=flp_chat_template_prompt,
            flp_attention_slots=flp_attention_slots,
            flp_document_window=flp_document_window,
            # Hierarchical LLM args
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
            # Hierarchical CNN args
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
            # Hierarchical GRU args
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
            # Simple CNN args
            scnn_embedding_dim=scnn_embedding_dim,
            scnn_conv_dim=scnn_conv_dim,
            scnn_kernel_size=scnn_kernel_size,
            scnn_num_conv_blocks=scnn_num_conv_blocks,
            scnn_max_length=scnn_max_length,
            scnn_vocab_size=scnn_vocab_size,
            scnn_gated_attention_dim=scnn_gated_attention_dim,
            scnn_projection_dim=scnn_projection_dim,
            scnn_dropout=scnn_dropout,
            # Byte CNN args
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

        # Explicit feature featurizer (if specs provided)
        self.explicit_feature_specs = explicit_feature_specs
        self.explicit_confounder_specs = explicit_feature_specs
        if explicit_feature_specs and len(explicit_feature_specs) > 0:
            self.explicit_feature_featurizer = ExplicitFeatureFeaturizer(
                specs=explicit_feature_specs,
                output_dim=explicit_feature_output_dim,
                hidden_dim=explicit_feature_hidden_dim,
                dropout=explicit_feature_dropout,
                device=str(self._device)
            )
            logger.info(
                f"Explicit feature featurizer enabled: {len(explicit_feature_specs)} features, "
                f"output_dim={explicit_feature_output_dim}"
            )
        else:
            self.explicit_feature_featurizer = None
        self.explicit_confounder_featurizer = self.explicit_feature_featurizer

        # Binary treatment Causal Inference Net
        # Input dim = text features + auxiliary features (if any) + explicit confounder features (if any)
        input_dim = self.feature_extractor.output_dim
        if auxiliary_dim > 0:
            input_dim += causal_head_representation_dim // 2
        if self.explicit_feature_featurizer is not None:
            input_dim += explicit_feature_output_dim

        if model_type == "rlearner":
            self.net = RLearnerNet(
                input_dim=input_dim,
                representation_dim=causal_head_representation_dim,
                hidden_outcome_dim=causal_head_hidden_outcome_dim,
                dropout=causal_head_dropout
            )
            logger.info("Using R-Learner architecture (direct tau optimization)")
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

    @staticmethod
    def _get_explicit_feature_values(
        batch: Dict[str, Any],
        override: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return explicit feature values from the new key, with legacy fallback."""
        if override is not None:
            return override
        return batch.get('explicit_feature_values', batch.get('explicit_confounder_values', None))

    def _append_explicit_features(
        self,
        features: torch.Tensor,
        explicit_feature_values: Optional[List[Dict[str, Any]]],
    ) -> torch.Tensor:
        """Append explicit feature embeddings when configured and present."""
        if self.explicit_feature_featurizer is None:
            return features
        if explicit_feature_values is None:
            return features
        explicit_features = self.explicit_feature_featurizer(explicit_feature_values)
        return torch.cat([features, explicit_features], dim=1)

    def forward(
        self,
        texts: List[str],
        auxiliary_features: Optional[torch.Tensor] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
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
        # Extract features from texts
        features = self.feature_extractor(texts)

        # Concatenate auxiliary features if provided
        if self.auxiliary_projection is not None and auxiliary_features is not None:
            aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
            features = torch.cat([features, aux_projected], dim=1)

        explicit_feature_values = explicit_feature_values or explicit_confounder_values
        features = self._append_explicit_features(features, explicit_feature_values)

        if self.model_type == "rlearner":
            # RLearnerNet returns: m_logit, tau, t_logit, final_common_layer
            # Returns native outputs - caller handles interpretation
            m_logit, tau, t_logit, final_common_layer = self.net(features)
            # For forward() compatibility, return in same tuple format
            # But these are semantically different: m_logit is marginal, tau is effect
            return m_logit, tau, t_logit, final_common_layer
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
        **kwargs  # Ignore extra args for API compatibility
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys.
                   Optional 'auxiliary_features' for hybrid models.
                   Optional 'explicit_confounder_values' for explicit confounders.
            alpha_propensity: Weight for propensity loss
            beta_targreg: Weight for targeted regularization (dragonnet)
            gamma_rlearner: Weight for R-learner loss (rlearner only)
            gamma_dr: Unused, kept for API compatibility
            label_smoothing: Label smoothing factor (0 = no smoothing)
            stop_grad_propensity: If True, detach features before propensity loss
                so propensity optimization doesn't affect the feature extractor.
                This forces the representation to optimize for tau/outcome.
            attention_entropy_weight: Weight for attention entropy regularization.

        Returns:
            Dictionary with loss components and detached predictions
        """
        # Dispatch to specialized training step for rlearner
        if self.model_type == "rlearner":
            return self._train_step_rlearner(
                batch, alpha_propensity, gamma_rlearner, label_smoothing,
                stop_grad_propensity, attention_entropy_weight
            )

        # DragonNet (default)
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)
        auxiliary_features = batch.get('auxiliary_features', None)
        explicit_feature_values = self._get_explicit_feature_values(batch)
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

        # Concatenate auxiliary features if provided
        if self.auxiliary_projection is not None and auxiliary_features is not None:
            aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
            features = torch.cat([features, aux_projected], dim=1)

        features = self._append_explicit_features(features, explicit_feature_values)

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

        # Total loss
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            beta_targreg * targreg_loss
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

        return result

    def _train_step_rlearner(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float,
        gamma_rlearner: float,
        label_smoothing: float,
        stop_grad_propensity: bool = False,
        attention_entropy_weight: float = 0.0
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
        explicit_feature_values = self._get_explicit_feature_values(batch)
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

        # Compute attention entropy loss if enabled and extractor supports it
        entropy_loss = torch.tensor(0.0, device=self._device)
        if attention_entropy_weight > 0 and hasattr(self.feature_extractor, 'compute_attention_entropy_loss'):
            _, attention_info = self.feature_extractor.forward_with_attention(texts)
            entropy_loss = attention_info['attention_entropy']

        # Concatenate auxiliary features if provided
        if self.auxiliary_projection is not None and auxiliary_features is not None:
            aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
            features = torch.cat([features, aux_projected], dim=1)

        features = self._append_explicit_features(features, explicit_feature_values)

        if stop_grad_propensity:
            features_detached = features.detach()
            m_logit, tau, t_logit, phi = self.net(features)
            phi_detached = self.net.get_representation(features_detached)
            t_logit_for_loss = self.net.propensity_from_representation(phi_detached)
            propensity_loss = F.binary_cross_entropy_with_logits(
                t_logit_for_loss.squeeze(-1),
                treatments_smooth
            )
        else:
            m_logit, tau, t_logit, phi = self.net(features)
            propensity_loss = F.binary_cross_entropy_with_logits(
                t_logit.squeeze(-1),
                treatments_smooth
            )

        outcome_loss = self._outcome_loss(m_logit.squeeze(-1), outcomes_smooth)

        e_X = torch.sigmoid(t_logit).detach().clamp(0.01, 0.99)
        m_X = self._outcome_activation(m_logit).detach()
        Y_residual = outcomes - m_X.squeeze(-1)
        T_residual = treatments - e_X.squeeze(-1)
        r_loss = ((Y_residual - tau.squeeze(-1) * T_residual) ** 2).mean()

        # Total loss
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            gamma_rlearner * r_loss +
            attention_entropy_weight * entropy_loss
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

        return result

    def predict(
        self,
        texts_or_batch,
        auxiliary_features: Optional[torch.Tensor] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
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
                explicit_feature_values = self._get_explicit_feature_values(
                    texts_or_batch,
                    override=explicit_feature_values or explicit_confounder_values,
                )
            else:
                texts = texts_or_batch
                extractor_input = texts
                explicit_feature_values = explicit_feature_values or explicit_confounder_values

            features = self.feature_extractor(extractor_input)

            # Concatenate auxiliary features if provided
            if self.auxiliary_projection is not None and auxiliary_features is not None:
                aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
                features = torch.cat([features, aux_projected], dim=1)

            features = self._append_explicit_features(features, explicit_feature_values)

            if self.model_type == "rlearner":
                # RLearnerNet returns: m_logit, tau, t_logit, final_common_layer
                m_logit, tau, t_logit, final_common_layer = self.net(features)

                m_prob = self._outcome_activation(m_logit).squeeze(-1)  # E[Y|X]
                tau_val = tau.squeeze(-1)  # tau(X)
                prop = torch.sigmoid(t_logit).squeeze(-1)  # e(X)

                # Derive Y0/Y1 from m and tau for backward compatibility:
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
                    'tau_pred': tau_val,  # Native tau(X)
                    't_logit': t_logit.squeeze(-1),
                    'm_logit': m_logit.squeeze(-1),
                    'final_common_layer': final_common_layer,
                    # Approximate logits for compatibility
                    'y0_logit': torch.logit(y0_prob.clamp(1e-6, 1 - 1e-6)) if self.outcome_type == "binary" else y0_prob,
                    'y1_logit': torch.logit(y1_prob.clamp(1e-6, 1 - 1e-6)) if self.outcome_type == "binary" else y1_prob,
                }
            else:
                # DragonNet
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
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
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
                explicit_feature_values = self._get_explicit_feature_values(
                    texts_or_batch,
                    override=explicit_feature_values or explicit_confounder_values,
                )
            else:
                extractor_input = texts_or_batch
                explicit_feature_values = explicit_feature_values or explicit_confounder_values

            features = self.feature_extractor(extractor_input)

            features = self._append_explicit_features(features, explicit_feature_values)

            return features

    def init_extractor(self, texts: List[str]) -> 'CausalText':
        """
        Initialize the feature extractor with training texts.

        For frozen_llm_pooler, this is a no-op since it uses a pretrained tokenizer.
        Kept for API compatibility.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        return self

    def fit_tokenizer(self, texts):
        """Fit tokenizer for trainable-from-scratch extractors. No-op for LLM-based."""
        if hasattr(self.feature_extractor, 'fit_tokenizer'):
            self.feature_extractor.fit_tokenizer(texts)

    def fit_explicit_feature_featurizer(
        self,
        feature_values_list: List[Dict[str, Any]]
    ) -> 'CausalText':
        """
        Fit the explicit feature featurizer on training data.

        This computes normalization statistics (mean/std) for continuous features.
        Must be called before training if explicit features are used.

        Args:
            feature_values_list: List of dicts with feature values from training data.
                Each dict should have "{name}" and "{name}_missing" keys.

        Returns:
            self for method chaining
        """
        if self.explicit_feature_featurizer is not None:
            self.explicit_feature_featurizer.fit(feature_values_list)
        return self

    def fit_explicit_confounder_featurizer(
        self,
        confounder_values_list: List[Dict[str, Any]]
    ) -> 'CausalText':
        """Backward-compatible alias for fit_explicit_feature_featurizer."""
        return self.fit_explicit_feature_featurizer(confounder_values_list)

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

        # Save extractor state
        checkpoint['extractor_state'] = self.feature_extractor.get_state()

        # Save explicit feature featurizer state if enabled
        if self.explicit_feature_featurizer is not None:
            checkpoint['explicit_feature_featurizer_state'] = self.explicit_feature_featurizer.get_state()
            checkpoint['explicit_confounder_featurizer_state'] = checkpoint['explicit_feature_featurizer_state']

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

        # Load state dict
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
        # Load explicit feature featurizer state if present
        feature_state = checkpoint.get(
            'explicit_feature_featurizer_state',
            checkpoint.get('explicit_confounder_featurizer_state')
        )
        if feature_state is not None and model.explicit_feature_featurizer is not None:
            model.explicit_feature_featurizer.load_state(feature_state)

        logger.info(f"Model loaded from {path}")
        return model

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)


# Backward compatibility alias
CausalCNNText = CausalText
