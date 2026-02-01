# cdt/models/causal_text_forest.py
"""Two-stage causal text model combining neural feature extraction with causal forests."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .cnn_extractor import CNNFeatureExtractor
from .bert_extractor import BertFeatureExtractor
from .gru_extractor import GRUFeatureExtractor
from .confounder_extractor import ConfounderExtractor, HierarchicalConfounderExtractor, GRUHierarchicalConfounderExtractor
from .hierarchical_transformer_extractor import HierarchicalTransformerExtractor
from .gated_mil_hierarchical_extractor import GatedMILHierarchicalExtractor
from .gru_transformer_mil_extractor import GRUTransformerMILExtractor
from .gru_pool_extractor import GRUPoolExtractor
from .llm_extractor import LLMFeatureExtractor
from .causal_forest_head import CausalForestHead, ECONML_AVAILABLE
from ..config import normalize_feature_extractor_type


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
        }

        # Initialize feature extractor (reuse existing implementations)
        if self.feature_extractor_type == "bert":
            self.feature_extractor = BertFeatureExtractor(
                model_name=bert_model_name,
                projection_dim=bert_projection_dim,
                max_length=bert_max_length,
                dropout=bert_dropout,
                freeze_encoder=bert_freeze_encoder,
                device=self._device
            )
            if bert_gradient_checkpointing:
                self.feature_extractor.gradient_checkpointing_enable()
        elif self.feature_extractor_type == "gru":
            self.feature_extractor = GRUFeatureExtractor(
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
                device=self._device
            )
        elif self.feature_extractor_type == "hierarchical_transformer":
            self.feature_extractor = HierarchicalTransformerExtractor(
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
                device=self._device
            )
        elif self.feature_extractor_type == "gated_mil_hierarchical":
            self.feature_extractor = GatedMILHierarchicalExtractor(
                sentence_encoder_model=gated_mil_sentence_model,
                freeze_sentence_encoder=gated_mil_freeze_sentence_encoder,
                max_chunks=gated_mil_max_chunks,
                chunk_size=gated_mil_chunk_size,
                chunk_overlap=gated_mil_chunk_overlap,
                mil_hidden_dim=gated_mil_hidden_dim,
                num_confounders=gated_mil_num_confounders,
                model_type="dragonnet",  # Use dragonnet style for representation
                projection_dim=gated_mil_projection_dim,
                dropout=gated_mil_dropout,
                hierarchical=gated_mil_hierarchical,
                token_hidden_dim=gated_mil_token_hidden_dim,
                use_mean_pooling=gated_mil_use_mean_pooling,
                device=self._device
            )
        elif self.feature_extractor_type == "gru_pool":
            self.feature_extractor = GRUPoolExtractor(
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
                device=self._device
            )
        elif self.feature_extractor_type == "llm":
            self.feature_extractor = LLMFeatureExtractor(
                model_name=llm_model_name,
                max_length=llm_max_length,
                projection_dim=llm_projection_dim,
                dropout=llm_dropout,
                gradient_checkpointing=llm_gradient_checkpointing,
                device=self._device
            )
        elif self.feature_extractor_type == "cnn":
            self.feature_extractor = CNNFeatureExtractor(
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
                device=self._device
            )
        else:
            raise ValueError(f"Unsupported feature extractor type: {feature_extractor_type}")

        logger.info(f"Using {self.feature_extractor_type.upper()} feature extractor")

        # Simple propensity head for representation learning
        input_dim = self.feature_extractor.output_dim
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
        if cf_use_rlearner_representation:
            self.effect_head = nn.Sequential(
                nn.Linear(input_dim, representation_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(representation_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)  # No activation - τ can be negative
            )
            logger.info("  R-learner representation training: ENABLED")
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

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through neural components.

        Args:
            texts: List of text strings

        Returns:
            features: Extracted features (batch, feature_dim)
            propensity_logit: Propensity prediction (batch, 1)
            outcome_logit: Outcome prediction (batch, 1)
        """
        features = self.feature_extractor(texts)
        propensity_logit = self.propensity_head(features)
        outcome_logit = self.outcome_head(features)
        return features, propensity_logit, outcome_logit

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
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys
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

        # Apply label smoothing
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Extract features
        features = self.feature_extractor(texts)

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
        if self.use_rlearner_representation and self.effect_head is not None:
            # Compute τ(X) from the effect head
            tau = self.effect_head(features)

            # Detach nuisance functions - this is the key to R-learner
            # Gradients only flow through τ, not through e or m estimates
            e_X = torch.sigmoid(propensity_logit).detach().clamp(0.01, 0.99)
            m_X = torch.sigmoid(outcome_logit).detach()

            # R-loss: pseudo-outcome regression
            Y_residual = outcomes - m_X.squeeze(-1)
            T_residual = treatments - e_X.squeeze(-1)
            r_loss = ((Y_residual - tau.squeeze(-1) * T_residual) ** 2).mean()

        total_loss = outcome_loss + alpha_propensity * propensity_loss + gamma_rlearner * r_loss

        return {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'r_loss': r_loss.detach() if isinstance(r_loss, torch.Tensor) else torch.tensor(r_loss),
            'propensity_logit': propensity_logit.detach(),
            'outcome_logit': outcome_logit.detach()
        }

    def extract_features(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features and nuisance predictions for all texts.

        Args:
            texts: List of all text strings
            batch_size: Batch size for processing

        Returns:
            features: Feature matrix (n_samples, feature_dim)
            propensity: Propensity predictions (n_samples,)
            outcome_pred: Outcome predictions (n_samples,)
        """
        self.eval()
        all_features = []
        all_propensity = []
        all_outcome = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                features, prop_logit, outcome_logit = self.forward(batch_texts)

                all_features.append(features.cpu().numpy())
                all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                all_outcome.append(torch.sigmoid(outcome_logit).cpu().numpy())

        return (
            np.vstack(all_features),
            np.vstack(all_propensity).flatten(),
            np.vstack(all_outcome).flatten()
        )

    def train_causal_forest(
        self,
        texts: List[str],
        T: np.ndarray,
        Y: np.ndarray,
        batch_size: int = 32
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

        Returns:
            self
        """
        logger.info("Extracting features for causal forest training...")
        features, _, _ = self.extract_features(texts, batch_size)

        # Fit causal forest on neural network features
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
        alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Predict ITEs using trained causal forest.

        Args:
            texts: List of text strings
            batch_size: Batch size for feature extraction
            return_ci: Whether to return confidence intervals
            alpha: Significance level for confidence intervals

        Returns:
            Dictionary with predictions:
                - tau_pred: ITE estimates
                - propensity: Propensity scores from neural network
                - outcome_pred: Outcome predictions from neural network
                - tau_lower, tau_upper: Confidence intervals (if return_ci)
        """
        # Extract features
        features, propensity, outcome_pred = self.extract_features(texts, batch_size)

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
        Initialize the feature extractor with training texts.

        Required for CNN, GRU, and GRU-based extractors.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        if hasattr(self.feature_extractor, 'fit_tokenizer'):
            self.feature_extractor.fit_tokenizer(texts)
        return self

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

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
