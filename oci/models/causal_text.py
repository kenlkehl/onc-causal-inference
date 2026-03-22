# oci/models/causal_text.py
"""Causal inference model using frozen LLM pooler for text representation."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dragonnet import DragonNet
from .rlearner import RLearnerNet
from .explicit_confounder_featurizer import ExplicitConfounderFeaturizer
from .extractor_factory import create_feature_extractor
from ..config import normalize_feature_extractor_type, ExplicitConfounderSpec


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
        # Causal head args (applies to all causal heads: DragonNet, RLearner)
        causal_head_representation_dim: int = 128,
        causal_head_hidden_outcome_dim: int = 64,
        causal_head_dropout: float = 0.2,
        device: str = "cuda:0",
        model_type: str = "dragonnet",  # "dragonnet" or "rlearner"
        # Auxiliary features (for hybrid text + categorical models)
        auxiliary_dim: int = 0,  # Dimension of auxiliary categorical features (0 = no auxiliary)
        # Explicit confounder featurizer args
        explicit_confounder_specs: Optional[List[ExplicitConfounderSpec]] = None,
        explicit_confounder_output_dim: int = 64,
        explicit_confounder_hidden_dim: int = 128,
        explicit_confounder_dropout: float = 0.1,
        # R-Learner dual extractor mode
        rlearner_dual_extractors: bool = False,
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
            'causal_head_representation_dim': causal_head_representation_dim,
            'causal_head_hidden_outcome_dim': causal_head_hidden_outcome_dim,
            'causal_head_dropout': causal_head_dropout,
            'model_type': model_type,
            'auxiliary_dim': auxiliary_dim,
            'explicit_confounder_specs': explicit_confounder_specs,
            'explicit_confounder_output_dim': explicit_confounder_output_dim,
            'explicit_confounder_hidden_dim': explicit_confounder_hidden_dim,
            'explicit_confounder_dropout': explicit_confounder_dropout,
            'rlearner_dual_extractors': rlearner_dual_extractors,
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

        # R-Learner dual extractor mode
        # When enabled, creates a second independent feature extractor for tau(X)
        # The nuisance extractor (self.feature_extractor) handles e(X) and m(X)
        # The effect extractor (self.effect_feature_extractor) handles tau(X)
        self.rlearner_dual_extractors = rlearner_dual_extractors
        self.effect_feature_extractor = None
        self.effect_mlp = None

        # Check for dual extractor mode (R-Learner only)
        dual_mode_enabled = (rlearner_dual_extractors and model_type == "rlearner")

        if dual_mode_enabled:
            # Create second feature extractor with same architecture using factory
            self.effect_feature_extractor = create_feature_extractor(
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
            )

            # Simple MLP for tau(X) - takes effect extractor output, predicts treatment effect
            # Note: tau is unbounded (can be negative) - no final activation
            effect_input_dim = self.effect_feature_extractor.output_dim
            self.effect_mlp = nn.Sequential(
                nn.Linear(effect_input_dim, causal_head_hidden_outcome_dim),
                nn.ReLU(),
                nn.Dropout(causal_head_dropout),
                nn.Linear(causal_head_hidden_outcome_dim, causal_head_hidden_outcome_dim),
                nn.ELU(),
                nn.Dropout(causal_head_dropout),
                nn.Linear(causal_head_hidden_outcome_dim, 1)  # tau is unbounded
            )

            logger.info(f"R-Learner dual extractor mode enabled:")
            logger.info(f"  Nuisance extractor: {self.feature_extractor_type} -> e(X), m(X)")
            logger.info(f"  Effect extractor: {self.feature_extractor_type} -> tau(X)")
            logger.info(f"  Effect MLP: {effect_input_dim} -> {causal_head_hidden_outcome_dim} -> 1")

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
        # Extract features from texts
        features = self.feature_extractor(texts)

        # Concatenate auxiliary features if provided
        if self.auxiliary_projection is not None and auxiliary_features is not None:
            aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
            features = torch.cat([features, aux_projected], dim=1)

        # Concatenate explicit confounder features if provided
        if self.explicit_confounder_featurizer is not None and explicit_confounder_values is not None:
            conf_features = self.explicit_confounder_featurizer(explicit_confounder_values)
            features = torch.cat([features, conf_features], dim=1)

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
            # - Effect extractor (self.effect_feature_extractor) + effect_mlp -> tau(X)

            # Nuisance path: extract features for e(X) and m(X)
            nuisance_features = self.feature_extractor(extractor_input)

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

            # Effect path: extract features for tau(X)
            effect_features = self.effect_feature_extractor(extractor_input)

            # tau(X) from separate effect MLP
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
            # CRITICAL: Nuisance functions are detached - gradients flow only through tau
            # In dual mode, tau comes from separate effect extractor + MLP
            e_X = torch.sigmoid(t_logit).detach().clamp(0.01, 0.99)
            m_X = self._outcome_activation(m_logit).detach()

            # Compute residuals
            Y_residual = outcomes - m_X.squeeze(-1)  # Y - m(X)
            T_residual = treatments - e_X.squeeze(-1)  # T - e(X)

            # R-loss: E[((Y - m(X)) - tau(X) * (T - e(X)))^2]
            r_loss = ((Y_residual - tau.squeeze(-1) * T_residual) ** 2).mean()

            # Set features variable for downstream use
            features = nuisance_features

        else:
            # STANDARD SINGLE EXTRACTOR MODE

            # Extract features
            features = self.feature_extractor(extractor_input)

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

            if self.model_type == "rlearner":
                # Check for dual extractor mode
                if self.rlearner_dual_extractors and self.effect_feature_extractor is not None:
                    # DUAL EXTRACTOR MODE:
                    # - Nuisance from main extractor + RLearnerNet
                    # - tau from effect extractor + effect_mlp
                    m_logit, _, t_logit, final_common_layer = self.net(features)

                    # Get tau from effect extractor
                    effect_features = self.effect_feature_extractor(extractor_input)
                    tau = self.effect_mlp(effect_features)

                    m_prob = self._outcome_activation(m_logit).squeeze(-1)  # E[Y|X]
                    tau_val = tau.squeeze(-1)  # tau(X)
                    prop = torch.sigmoid(t_logit).squeeze(-1)  # e(X)

                else:
                    # STANDARD MODE:
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

        For frozen_llm_pooler, this is a no-op since it uses a pretrained tokenizer.
        Kept for API compatibility.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        return self

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

        # Save extractor state
        checkpoint['extractor_state'] = self.feature_extractor.get_state()

        # Save explicit confounder featurizer state if enabled
        if self.explicit_confounder_featurizer is not None:
            checkpoint['explicit_confounder_featurizer_state'] = self.explicit_confounder_featurizer.get_state()

        # Save effect extractor and effect MLP state if in dual mode (R-Learner)
        if self.rlearner_dual_extractors and self.model_type == "rlearner":
            if self.effect_feature_extractor is not None:
                checkpoint['effect_feature_extractor'] = self.effect_feature_extractor.state_dict()
                checkpoint['effect_mlp'] = self.effect_mlp.state_dict()
                if hasattr(self.effect_feature_extractor, 'get_state'):
                    checkpoint['effect_extractor_state'] = self.effect_feature_extractor.get_state()

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

        # Load effect extractor state BEFORE loading model_state_dict
        # This ensures embedding layers have correct dimensions
        if model.rlearner_dual_extractors and model.model_type == "rlearner":
            if model.effect_feature_extractor is not None:
                if 'effect_extractor_state' in checkpoint:
                    if hasattr(model.effect_feature_extractor, 'load_state'):
                        model.effect_feature_extractor.load_state(checkpoint['effect_extractor_state'])

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
            # Load effect extractor weights separately if not using model_state_dict
            if model.rlearner_dual_extractors and model.model_type == "rlearner":
                if model.effect_feature_extractor is not None:
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
