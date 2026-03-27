# oci/models/causal_text_forest.py
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
        # Explicit confounder args (raw features for causal forest, MLP for Stage 1 training)
        explicit_confounder_specs: Optional[List[ExplicitConfounderSpec]] = None,
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
            'explicit_confounder_specs': explicit_confounder_specs,
            'explicit_confounder_output_dim': explicit_confounder_output_dim,
            'explicit_confounder_hidden_dim': explicit_confounder_hidden_dim,
            'explicit_confounder_dropout': explicit_confounder_dropout,
            'outcome_type': outcome_type,
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
        texts_or_batch,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through neural components.

        Args:
            texts_or_batch: List of text strings or preprocessed batch dict from DataLoader
            explicit_confounder_values: Optional list of dicts with explicit confounder values

        Returns:
            features: Extracted features (batch, feature_dim)
            propensity_logit: Propensity prediction (batch, 1)
            outcome_logit: Outcome prediction (batch, 1)
        """
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

        propensity_logit = self.propensity_head(features)
        outcome_logit = self.outcome_head(features)
        return features, propensity_logit, outcome_logit

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

        Returns:
            Dictionary with loss components
        """
        texts = batch['texts']
        treatments = batch['treatment']
        outcomes = batch['outcome']
        explicit_confounder_values = batch.get('explicit_confounder_values', None)
        extractor_input = self._get_extractor_input(batch, texts)

        # Apply label smoothing (skip outcome smoothing for continuous)
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            if self.outcome_type == "binary":
                outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
            else:
                outcomes_smooth = outcomes
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Extract features from text
        features = self.feature_extractor(extractor_input)

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

        outcome_loss = self._outcome_loss(
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
                effect_features = self.effect_feature_extractor(extractor_input)

                # Compute τ(X) from effect MLP
                tau = self.effect_mlp(effect_features)

                # Detach nuisance functions - gradients flow only through effect extractor + MLP
                e_X = torch.sigmoid(propensity_logit).detach().clamp(0.01, 0.99)
                m_X = self._outcome_activation(outcome_logit).detach()

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
                m_X = self._outcome_activation(outcome_logit).detach()

                # R-loss: pseudo-outcome regression
                Y_residual = outcomes - m_X.squeeze(-1)
                T_residual = treatments - e_X.squeeze(-1)
                r_loss = ((Y_residual - tau.squeeze(-1) * T_residual) ** 2).mean()

        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            gamma_rlearner * r_loss
        )

        result = {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'r_loss': r_loss.detach() if isinstance(r_loss, torch.Tensor) else torch.tensor(r_loss),
            'propensity_logit': propensity_logit.detach(),
            'outcome_logit': outcome_logit.detach()
        }

        return result

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

    def extract_features(
        self,
        texts_or_loader,
        batch_size: int = 32,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        gpu_store=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features and nuisance predictions for all texts.

        In dual extractor mode, features are extracted from the effect extractor
        (optimized for treatment effect heterogeneity via R-loss). Propensity
        and outcome predictions still come from the nuisance extractor.

        Args:
            texts_or_loader: List of all text strings, or a DataLoader yielding batch dicts
            batch_size: Batch size for processing (only used when texts_or_loader is a list)
            explicit_confounder_values: Optional list of dicts with confounder values.
                If provided and explicit_confounder_specs is set, raw confounder features
                are concatenated to neural features. Ignored when using DataLoader
                (confounder values come from batch dicts).

        Returns:
            features: Feature matrix (n_samples, feature_dim + confounder_dim)
            propensity: Propensity predictions (n_samples,)
            outcome_pred: Outcome predictions (n_samples,)
        """
        from torch.utils.data import DataLoader

        self.eval()
        all_text_features = []
        all_propensity = []
        all_outcome = []
        all_conf_values = []

        # Determine which extractor to use for features
        # In dual mode, use effect extractor (optimized for τ)
        # Otherwise, use nuisance extractor (feature_extractor)
        use_effect_extractor = (
            self.rlearner_dual_extractors and
            self.effect_feature_extractor is not None
        )

        # Accept DataLoader or any iterable yielding batch dicts (e.g. generator)
        is_batch_iterable = isinstance(texts_or_loader, DataLoader) or (
            hasattr(texts_or_loader, '__iter__') and not isinstance(texts_or_loader, (list, str))
        )
        if is_batch_iterable:
            # DataLoader / batch iterable path: iterate over preprocessed batches
            with torch.no_grad():
                for batch in texts_or_loader:
                    # Move cached hidden states to device if present (from DataLoader)
                    prepare_cached_batch(batch, self._device, gpu_store=gpu_store)
                    texts = batch['texts']
                    extractor_input = self._get_extractor_input(batch, texts)
                    batch_conf_values = batch.get('explicit_confounder_values', None)

                    if use_effect_extractor:
                        text_features = self.effect_feature_extractor(extractor_input)
                    else:
                        text_features = self.feature_extractor(extractor_input)
                    all_text_features.append(text_features.cpu().numpy())

                    _, prop_logit, outcome_logit = self.forward(
                        batch,
                        explicit_confounder_values=batch_conf_values
                    )
                    all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                    all_outcome.append(self._outcome_activation(outcome_logit).cpu().numpy())

                    if batch_conf_values is not None:
                        all_conf_values.extend(batch_conf_values)
        else:
            # Raw texts path (backward compatible)
            texts = texts_or_loader
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]

                    batch_conf_values = None
                    if explicit_confounder_values is not None:
                        batch_conf_values = explicit_confounder_values[i:i + batch_size]

                    if use_effect_extractor:
                        text_features = self.effect_feature_extractor(batch_texts)
                    else:
                        text_features = self.feature_extractor(batch_texts)
                    all_text_features.append(text_features.cpu().numpy())

                    _, prop_logit, outcome_logit = self.forward(
                        batch_texts,
                        explicit_confounder_values=batch_conf_values
                    )
                    all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                    all_outcome.append(self._outcome_activation(outcome_logit).cpu().numpy())

        neural_features = np.vstack(all_text_features)

        # Concatenate raw confounder features if provided
        # Use collected confounder values from DataLoader batches, or the provided list
        conf_values_for_raw = all_conf_values if all_conf_values else explicit_confounder_values
        if conf_values_for_raw is not None and self.explicit_confounder_specs:
            raw_conf_features = self._get_raw_confounder_features(conf_values_for_raw)
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
        texts_or_loader,
        T: np.ndarray,
        Y: np.ndarray,
        batch_size: int = 32,
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
            explicit_confounder_values: Optional list of dicts with confounder values.
                If provided, raw confounder features are concatenated to neural features.
            gpu_store: Optional GPUHiddenStateStore for GPU-resident hidden states.

        Returns:
            self
        """
        logger.info("Extracting features for causal forest training...")
        features, _, _ = self.extract_features(
            texts_or_loader, batch_size,
            explicit_confounder_values=explicit_confounder_values,
            gpu_store=gpu_store
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
        texts_or_loader,
        batch_size: int = 32,
        return_ci: bool = True,
        alpha: float = 0.05,
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
            explicit_confounder_values: Optional list of dicts with confounder values.
                Must be provided if model was trained with explicit confounders.
            gpu_store: Optional GPUHiddenStateStore for GPU-resident hidden states.

        Returns:
            Dictionary with predictions:
                - tau_pred: ITE estimates
                - propensity: Propensity scores from neural network
                - outcome_pred: Outcome predictions from neural network
                - tau_lower, tau_upper: Confidence intervals (if return_ci)
        """
        # Extract features (with raw confounder features if provided)
        features, propensity, outcome_pred = self.extract_features(
            texts_or_loader, batch_size,
            explicit_confounder_values=explicit_confounder_values,
            gpu_store=gpu_store
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
