# cdt/models/matched_pair_ite.py
"""Matched pair ITE estimation models.

This module implements a two-stage approach for Individual Treatment Effect (ITE)
estimation using propensity matching:

Stage 1: PropensityMatchingModel
    - Train propensity model using hierarchical attention (bert-tiny -> CLS -> POOL)
    - Extract representations for embedding-based matching
    - Freeze representation after training

Stage 2: MatchedPairOutcomeModel
    - Train outcome + tau model on matched pairs only
    - Shared outcome head for Y_U and Y_T prediction
    - Tau head predicts treatment effect from untreated embedding only
    - Target: log-odds difference between matched pair outcomes
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical_transformer_extractor import HierarchicalTransformerExtractor


logger = logging.getLogger(__name__)


class PropensityMatchingModel(nn.Module):
    """
    Propensity model with representation extraction for matching.

    Architecture:
        HierarchicalTransformerExtractor (bert-tiny -> CLS per sentence -> Transformer POOL)
        -> representation_layers (Linear -> ELU -> Linear)
        -> propensity_head (Linear -> sigmoid)

    The representation layer can be frozen after propensity training to preserve
    covariate balance when used for matching and outcome/tau training.

    Args:
        sentence_model: HuggingFace model name for sentence encoding (default: prajjwal1/bert-tiny)
        freeze_sentence_encoder: Whether to freeze the sentence encoder weights
        max_sentences: Maximum number of sentences to process per document
        max_sentence_length: Maximum tokens per sentence for BERT encoding
        transformer_dim: Hidden dimension for transformer pooling layers
        num_transformer_layers: Number of transformer layers for pooling
        num_attention_heads: Number of attention heads in transformer layers
        transformer_dropout: Dropout rate for transformer layers
        representation_dim: Dimension of the learned representation layer
        device: PyTorch device
    """

    def __init__(
        self,
        sentence_model: str = "prajjwal1/bert-tiny",
        freeze_sentence_encoder: bool = True,
        max_sentences: int = 100,
        max_sentence_length: int = 128,
        transformer_dim: int = 256,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        transformer_dropout: float = 0.1,
        representation_dim: int = 256,
        device: str = "cuda:0"
    ):
        super().__init__()

        self._device = torch.device(device) if isinstance(device, str) else device
        self._representation_dim = representation_dim
        self._representation_frozen = False

        # Feature extractor: HierarchicalTransformerExtractor
        # The projection_dim from the extractor feeds into our representation layers
        self.feature_extractor = HierarchicalTransformerExtractor(
            sentence_encoder_model=sentence_model,
            freeze_sentence_encoder=freeze_sentence_encoder,
            max_sentences=max_sentences,
            max_sentence_length=max_sentence_length,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            transformer_dim=transformer_dim,
            transformer_dropout=transformer_dropout,
            projection_dim=transformer_dim,  # Use transformer_dim as intermediate
            device=self._device
        )

        # Representation layers (can be frozen after propensity training)
        self.repr_fc1 = nn.Linear(transformer_dim, representation_dim)
        self.repr_fc2 = nn.Linear(representation_dim, representation_dim)
        self.repr_norm = nn.LayerNorm(representation_dim)

        # Propensity head
        self.propensity_head = nn.Sequential(
            nn.Linear(representation_dim, representation_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(representation_dim // 2, 1)
        )

        logger.info(f"PropensityMatchingModel initialized:")
        logger.info(f"  Sentence encoder: {sentence_model}")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Transformer dim: {transformer_dim}")
        logger.info(f"  Device: {self._device}")

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Forward pass returning propensity logits.

        Args:
            texts: List of document texts

        Returns:
            Propensity logits of shape (batch_size, 1)
        """
        repr = self.get_representation(texts)
        return self.propensity_head(repr)

    def get_representation(self, texts: List[str]) -> torch.Tensor:
        """
        Extract representation for matching/outcome training.

        Args:
            texts: List of document texts

        Returns:
            Representation tensor of shape (batch_size, representation_dim)
        """
        # Get features from hierarchical transformer
        features = self.feature_extractor(texts)  # (B, transformer_dim)

        # Apply representation layers
        h = F.relu(self.repr_fc1(features))
        h = F.elu(self.repr_fc2(h))
        repr = self.repr_norm(h)

        return repr

    def predict_propensity(self, texts: List[str]) -> torch.Tensor:
        """
        Predict propensity scores (probabilities).

        Args:
            texts: List of document texts

        Returns:
            Propensity probabilities of shape (batch_size,)
        """
        logits = self.forward(texts)
        return torch.sigmoid(logits).squeeze(-1)

    def freeze_representation(self) -> None:
        """
        Freeze feature extractor and representation layers.

        Call this after propensity training to preserve covariate balance
        when training the outcome/tau model.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.repr_fc1.parameters():
            param.requires_grad = False
        for param in self.repr_fc2.parameters():
            param.requires_grad = False
        for param in self.repr_norm.parameters():
            param.requires_grad = False

        self._representation_frozen = True
        logger.info("Representation layers frozen")

    def unfreeze_representation(self) -> None:
        """Unfreeze representation layers (e.g., for fine-tuning)."""
        for param in self.repr_fc1.parameters():
            param.requires_grad = True
        for param in self.repr_fc2.parameters():
            param.requires_grad = True
        for param in self.repr_norm.parameters():
            param.requires_grad = True
        # Note: feature_extractor may still be frozen if freeze_sentence_encoder=True

        self._representation_frozen = False
        logger.info("Representation layers unfrozen")

    @property
    def representation_dim(self) -> int:
        """Return the representation dimension."""
        return self._representation_dim

    @property
    def is_representation_frozen(self) -> bool:
        """Check if representation is frozen."""
        return self._representation_frozen

    def fit_tokenizer(self, texts: List[str]) -> 'PropensityMatchingModel':
        """
        Initialize the feature extractor (triggers lazy initialization).

        Args:
            texts: List of training text strings (triggers initialization)

        Returns:
            self for method chaining
        """
        self.feature_extractor.init_extractor(texts)
        return self

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.feature_extractor = self.feature_extractor.to(device)
        return super().to(device)

    def get_state(self) -> Dict[str, Any]:
        """Get model state for checkpoint saving."""
        return {
            'representation_dim': self._representation_dim,
            'representation_frozen': self._representation_frozen,
            'feature_extractor_state': self.feature_extractor.get_state()
        }


class MatchedPairOutcomeModel(nn.Module):
    """
    Outcome + Tau model for matched pairs.

    Takes frozen representations as input (not text). Trains on matched pairs
    where each pair consists of (untreated patient U, treated patient T).

    Architecture:
        repr_U, repr_T (frozen, from PropensityMatchingModel)
        -> outcome_head (shared): Linear -> ReLU -> Linear -> ReLU -> Linear
        -> tau_head (U only): Linear -> ReLU -> Linear -> ReLU -> Linear

    The tau target is the log-odds difference: logit(P_T) - logit(P_U),
    which is unbounded and stable for optimization.

    Args:
        representation_dim: Dimension of input representations
        hidden_dim: Hidden dimension for outcome and tau heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        representation_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()

        self._representation_dim = representation_dim
        self._hidden_dim = hidden_dim

        # Shared outcome head (predicts P(Y=1|repr))
        self.outcome_fc1 = nn.Linear(representation_dim, hidden_dim)
        self.outcome_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.outcome_fc3 = nn.Linear(hidden_dim, 1)

        # Tau head (predicts treatment effect from U repr only)
        # Output is unbounded (log-odds scale) to allow negative effects
        self.tau_fc1 = nn.Linear(representation_dim, hidden_dim)
        self.tau_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tau_fc3 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

        logger.info(f"MatchedPairOutcomeModel initialized:")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Dropout: {dropout}")

    def forward(
        self,
        repr_U: torch.Tensor,
        repr_T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for matched pair training.

        Args:
            repr_U: Untreated patient representations (B, D)
            repr_T: Treated patient representations (B, D)

        Returns:
            y_U_logit: Outcome logit for untreated (B, 1)
            y_T_logit: Outcome logit for treated (B, 1)
            tau_pred: Treatment effect prediction on log-odds scale (B, 1)
        """
        # Outcome predictions (shared weights)
        y_U_logit = self._outcome_forward(repr_U)
        y_T_logit = self._outcome_forward(repr_T)

        # Tau prediction (from U only)
        tau_pred = self._tau_forward(repr_U)

        return y_U_logit, y_T_logit, tau_pred

    def _outcome_forward(self, repr: torch.Tensor) -> torch.Tensor:
        """Shared outcome prediction head."""
        h = F.relu(self.outcome_fc1(repr))
        h = self.dropout(h)
        h = F.relu(self.outcome_fc2(h))
        h = self.dropout(h)
        return self.outcome_fc3(h)

    def _tau_forward(self, repr: torch.Tensor) -> torch.Tensor:
        """Tau (treatment effect) prediction head."""
        h = F.relu(self.tau_fc1(repr))
        h = self.dropout(h)
        h = F.relu(self.tau_fc2(h))
        h = self.dropout(h)
        return self.tau_fc3(h)

    def predict_ite(self, repr: torch.Tensor) -> torch.Tensor:
        """
        Predict ITE for any patient given their representation.

        The output is tau on log-odds scale. To convert to probability scale,
        use the approximation:
            ite_prob ≈ tau_logodds * p * (1 - p)
        where p is the baseline outcome probability.

        Or use the exact conversion:
            P(Y=1|T=1) - P(Y=1|T=0) = sigmoid(logit_0 + tau) - sigmoid(logit_0)

        Args:
            repr: Patient representations of shape (B, D)

        Returns:
            tau on log-odds scale of shape (B, 1)
        """
        return self._tau_forward(repr)

    def predict_outcome(self, repr: torch.Tensor) -> torch.Tensor:
        """
        Predict outcome probability for a patient.

        Args:
            repr: Patient representations of shape (B, D)

        Returns:
            Outcome probability of shape (B, 1)
        """
        logit = self._outcome_forward(repr)
        return torch.sigmoid(logit)

    def predict_potential_outcomes(
        self,
        repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict potential outcomes Y(0), Y(1) and ITE for a patient.

        Uses the outcome head for Y(0) and tau head to derive Y(1).

        Args:
            repr: Patient representations of shape (B, D)

        Returns:
            Tuple of (y0_prob, y1_prob, ite_prob), each of shape (B, 1)
        """
        y0_logit = self._outcome_forward(repr)
        tau_logodds = self._tau_forward(repr)

        # Y(1) = Y(0) + tau in logit space
        y1_logit = y0_logit + tau_logodds

        y0_prob = torch.sigmoid(y0_logit)
        y1_prob = torch.sigmoid(y1_logit)
        ite_prob = y1_prob - y0_prob

        return y0_prob, y1_prob, ite_prob


class CombinedMatchedPairModel(nn.Module):
    """
    Combined model for end-to-end prediction after training.

    Wraps PropensityMatchingModel and MatchedPairOutcomeModel for inference,
    taking raw text as input and producing ITE predictions.

    Args:
        propensity_model: Trained PropensityMatchingModel
        outcome_model: Trained MatchedPairOutcomeModel
    """

    def __init__(
        self,
        propensity_model: PropensityMatchingModel,
        outcome_model: MatchedPairOutcomeModel
    ):
        super().__init__()
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model

    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Full inference pipeline: text -> predictions.

        Args:
            texts: List of document texts

        Returns:
            Dictionary with predictions:
                - 'propensity': Propensity probability (B,)
                - 'tau_logodds': Treatment effect on log-odds scale (B, 1)
                - 'y0_prob': Potential outcome under control (B, 1)
                - 'y1_prob': Potential outcome under treatment (B, 1)
                - 'ite_prob': Individual treatment effect on probability scale (B, 1)
        """
        self.propensity_model.eval()
        self.outcome_model.eval()

        with torch.no_grad():
            # Get representations
            repr = self.propensity_model.get_representation(texts)

            # Get propensity
            propensity = self.propensity_model.predict_propensity(texts)

            # Get potential outcomes and ITE
            y0_prob, y1_prob, ite_prob = self.outcome_model.predict_potential_outcomes(repr)

            # Get raw tau
            tau_logodds = self.outcome_model.predict_ite(repr)

        return {
            'propensity': propensity,
            'tau_logodds': tau_logodds,
            'y0_prob': y0_prob,
            'y1_prob': y1_prob,
            'ite_prob': ite_prob
        }

    def predict(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Alias for forward() for API consistency."""
        return self.forward(texts)
