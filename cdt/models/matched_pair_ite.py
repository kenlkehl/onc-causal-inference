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
from .hierarchical_gru_transformer_extractor import HierarchicalGRUTransformerExtractor
from .residual_cross_encoder import ResidualCrossEncoder
from .bert_gated_pool_extractor import BERTGatedPoolExtractor
from .gru_pool_extractor import GRUPoolExtractor


logger = logging.getLogger(__name__)


class PropensityMatchingModel(nn.Module):
    """
    Propensity model with representation extraction for matching.

    Architecture:
        HierarchicalTransformerExtractor (bert-tiny -> CLS per sentence -> Transformer POOL)
        OR HierarchicalGRUTransformerExtractor (overlapping token chunks -> BiGRU + attention)
        -> representation_layers (Linear -> ELU -> Linear)
        -> propensity_head (Linear -> sigmoid)

    The representation layer can be frozen after propensity training to preserve
    covariate balance when used for matching and outcome/tau training.

    The chunk encoder type is selected via the `chunk_encoder` parameter:
    - "bert" (default): Uses HierarchicalTransformerExtractor with sentence-level BERT
    - "gru": Uses HierarchicalGRUTransformerExtractor with overlapping token chunks
    - "bert_gated_pool": Uses BERTGatedPoolExtractor with gated attention pooling
    - "gru_pool": Uses GRUPoolExtractor with gated attention pooling

    Args:
        sentence_model: HuggingFace model name for sentence encoding (default: prajjwal1/bert-tiny)
        freeze_sentence_encoder: Whether to freeze the sentence encoder weights
        max_sentences: Maximum number of sentences/chunks to process per document
        max_sentence_length: Maximum tokens per sentence for BERT encoding
        transformer_dim: Hidden dimension for transformer pooling layers
        num_transformer_layers: Number of transformer layers for pooling
        num_attention_heads: Number of attention heads in transformer layers
        transformer_dropout: Dropout rate for transformer layers
        representation_dim: Dimension of the learned representation layer
        joint_outcome_training: Whether to jointly train on outcome prediction
        chunk_encoder: Encoder type - "bert", "gru", "bert_gated_pool", or "gru_pool"
        gru_chunk_size: Tokens per chunk (GRU encoder only)
        gru_chunk_overlap: Overlap between chunks (GRU encoder only)
        gru_embedding_dim: Word embedding dimension (GRU encoder only)
        gru_hidden_dim: BiGRU hidden dimension per direction (GRU encoder only)
        gru_num_layers: Number of GRU layers (GRU encoder only)
        gru_max_vocab_size: Maximum vocabulary size (GRU encoder only)
        gru_min_word_freq: Minimum word frequency (GRU encoder only)
        bert_gated_pool_model: HuggingFace model for bert_gated_pool encoder
        bert_gated_pool_freeze_encoder: Whether to freeze BERT in bert_gated_pool
        bert_gated_pool_chunk_size: Tokens per chunk for bert_gated_pool
        bert_gated_pool_chunk_overlap: Overlap for bert_gated_pool
        bert_gated_pool_transformer_layers: Transformer layers for bert_gated_pool
        bert_gated_pool_transformer_heads: Attention heads for bert_gated_pool
        bert_gated_pool_transformer_dim: Hidden dim for bert_gated_pool
        bert_gated_pool_gated_attention_dim: Gated attention dim for bert_gated_pool
        bert_gated_pool_use_mean_pooling: Use mean pooling vs [CLS] for bert_gated_pool
        gru_pool_gated_attention_dim: Gated attention dim for gru_pool
        gru_pool_transformer_layers: Transformer layers for gru_pool
        gru_pool_transformer_heads: Attention heads for gru_pool
        gru_pool_transformer_dim: Hidden dim for gru_pool
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
        joint_outcome_training: bool = False,
        # Chunk encoder selection
        chunk_encoder: str = "bert",
        # GRU-specific parameters (for "gru" encoder)
        gru_chunk_size: int = 128,
        gru_chunk_overlap: int = 32,
        gru_embedding_dim: int = 128,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        gru_max_vocab_size: int = 50000,
        gru_min_word_freq: int = 2,
        # BERT Gated Pool parameters (for "bert_gated_pool" encoder)
        bert_gated_pool_model: str = "prajjwal1/bert-tiny",
        bert_gated_pool_freeze_encoder: bool = True,
        bert_gated_pool_chunk_size: int = 128,
        bert_gated_pool_chunk_overlap: int = 32,
        bert_gated_pool_transformer_layers: int = 2,
        bert_gated_pool_transformer_heads: int = 4,
        bert_gated_pool_transformer_dim: int = 256,
        bert_gated_pool_gated_attention_dim: int = 128,
        bert_gated_pool_use_mean_pooling: bool = False,
        # GRU Pool parameters (for "gru_pool" encoder)
        gru_pool_gated_attention_dim: int = 128,
        gru_pool_transformer_layers: int = 2,
        gru_pool_transformer_heads: int = 4,
        gru_pool_transformer_dim: int = 256,
        device: str = "cuda:0"
    ):
        super().__init__()

        self._device = torch.device(device) if isinstance(device, str) else device
        self._representation_dim = representation_dim
        self._representation_frozen = False
        self._joint_outcome_training = joint_outcome_training
        self._chunk_encoder_type = chunk_encoder

        # Feature extractor: based on chunk_encoder type
        if chunk_encoder == "bert_gated_pool":
            self.feature_extractor = BERTGatedPoolExtractor(
                bert_model=bert_gated_pool_model,
                freeze_encoder=bert_gated_pool_freeze_encoder,
                use_mean_pooling=bert_gated_pool_use_mean_pooling,
                max_chunks=max_sentences,  # Reuse max_sentences parameter
                chunk_size=bert_gated_pool_chunk_size,
                chunk_overlap=bert_gated_pool_chunk_overlap,
                transformer_layers=bert_gated_pool_transformer_layers,
                transformer_heads=bert_gated_pool_transformer_heads,
                transformer_dim=bert_gated_pool_transformer_dim,
                transformer_dropout=transformer_dropout,
                gated_attention_dim=bert_gated_pool_gated_attention_dim,
                projection_dim=transformer_dim,  # Use transformer_dim as intermediate
                device=self._device
            )
        elif chunk_encoder == "gru_pool":
            self.feature_extractor = GRUPoolExtractor(
                embedding_dim=gru_embedding_dim,
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                gru_bidirectional=True,
                gru_dropout=transformer_dropout,
                max_chunks=max_sentences,  # Reuse max_sentences parameter
                chunk_size=gru_chunk_size,
                chunk_overlap=gru_chunk_overlap,
                transformer_layers=gru_pool_transformer_layers,
                transformer_heads=gru_pool_transformer_heads,
                transformer_dim=gru_pool_transformer_dim,
                transformer_dropout=transformer_dropout,
                gated_attention_dim=gru_pool_gated_attention_dim,
                projection_dim=transformer_dim,  # Use transformer_dim as intermediate
                max_vocab_size=gru_max_vocab_size,
                min_word_freq=gru_min_word_freq,
                device=self._device
            )
        elif chunk_encoder == "gru":
            self.feature_extractor = HierarchicalGRUTransformerExtractor(
                chunk_size=gru_chunk_size,
                chunk_overlap=gru_chunk_overlap,
                max_chunks=max_sentences,  # Reuse max_sentences parameter
                embedding_dim=gru_embedding_dim,
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                chunk_dim=transformer_dim,
                num_transformer_layers=num_transformer_layers,
                num_attention_heads=num_attention_heads,
                transformer_dropout=transformer_dropout,
                projection_dim=transformer_dim,  # Use transformer_dim as intermediate
                max_vocab_size=gru_max_vocab_size,
                min_word_freq=gru_min_word_freq,
                device=self._device
            )
        else:  # "bert" (default)
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

        # Outcome head (only if joint training enabled)
        # This predicts P(Y=1|X), NOT potential outcomes
        self.outcome_head = None
        if joint_outcome_training:
            self.outcome_head = nn.Sequential(
                nn.Linear(representation_dim, representation_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(representation_dim // 2, 1)
            )

        logger.info(f"PropensityMatchingModel initialized:")
        logger.info(f"  Chunk encoder: {chunk_encoder}")
        if chunk_encoder == "bert_gated_pool":
            logger.info(f"  BERT model: {bert_gated_pool_model}")
            logger.info(f"  Chunk size: {bert_gated_pool_chunk_size}, overlap: {bert_gated_pool_chunk_overlap}")
            logger.info(f"  Gated attention dim: {bert_gated_pool_gated_attention_dim}")
        elif chunk_encoder == "gru_pool":
            logger.info(f"  GRU: chunk_size={gru_chunk_size}, overlap={gru_chunk_overlap}")
            logger.info(f"  GRU: embedding_dim={gru_embedding_dim}, hidden_dim={gru_hidden_dim}, layers={gru_num_layers}")
            logger.info(f"  Gated attention dim: {gru_pool_gated_attention_dim}")
        elif chunk_encoder == "gru":
            logger.info(f"  GRU: chunk_size={gru_chunk_size}, overlap={gru_chunk_overlap}")
            logger.info(f"  GRU: embedding_dim={gru_embedding_dim}, hidden_dim={gru_hidden_dim}, layers={gru_num_layers}")
        else:
            logger.info(f"  Sentence encoder: {sentence_model}")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Transformer dim: {transformer_dim}")
        logger.info(f"  Joint outcome training: {joint_outcome_training}")
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

    def forward_joint(self, texts: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass returning both propensity and outcome logits.

        Used during joint training when joint_outcome_training=True.

        Args:
            texts: List of document texts

        Returns:
            Tuple of (t_logit, y_logit) where:
                - t_logit: Propensity logits of shape (batch_size, 1)
                - y_logit: Outcome logits of shape (batch_size, 1), or None if
                          joint_outcome_training=False
        """
        repr = self.get_representation(texts)
        t_logit = self.propensity_head(repr)
        y_logit = self.outcome_head(repr) if self._joint_outcome_training else None
        return t_logit, y_logit

    def predict_outcome(self, texts: List[str]) -> torch.Tensor:
        """
        Predict outcome probabilities (requires joint_outcome_training=True).

        Note: This predicts P(Y=1|X), not potential outcomes Y(0) or Y(1).

        Args:
            texts: List of document texts

        Returns:
            Outcome probabilities of shape (batch_size,)

        Raises:
            ValueError: If joint_outcome_training was not enabled
        """
        if not self._joint_outcome_training:
            raise ValueError("Outcome prediction requires joint_outcome_training=True")
        repr = self.get_representation(texts)
        return torch.sigmoid(self.outcome_head(repr)).squeeze(-1)

    @property
    def joint_outcome_training(self) -> bool:
        """Check if joint outcome training is enabled."""
        return self._joint_outcome_training

    def forward_with_instances(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass returning representations AND chunk-level info for CLAM supervision.

        This method is only available when using gated pool extractors
        (chunk_encoder="bert_gated_pool" or "gru_pool").

        Args:
            texts: List of document texts

        Returns:
            repr: (B, representation_dim) - document-level representations
            chunk_embeddings_list: List of (C_i, transformer_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - gated attention weights per doc

        Raises:
            RuntimeError: If the feature extractor does not support forward_with_instances
        """
        if not hasattr(self.feature_extractor, 'forward_with_instances'):
            raise RuntimeError(
                "forward_with_instances is only supported with gated pool extractors "
                "(chunk_encoder='bert_gated_pool' or 'gru_pool')"
            )

        # Get features and instance info from extractor
        features, chunk_embeddings_list, attention_weights_list = \
            self.feature_extractor.forward_with_instances(texts)  # (B, transformer_dim)

        # Apply representation layers
        h = F.relu(self.repr_fc1(features))
        h = F.elu(self.repr_fc2(h))
        repr = self.repr_norm(h)

        return repr, chunk_embeddings_list, attention_weights_list

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

    @property
    def chunk_encoder_type(self) -> str:
        """Return the chunk encoder type ('bert' or 'gru')."""
        return self._chunk_encoder_type

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
            'joint_outcome_training': self._joint_outcome_training,
            'chunk_encoder_type': self._chunk_encoder_type,
            'feature_extractor_state': self.feature_extractor.get_state()
        }


class InstanceCausalHead(nn.Module):
    """
    Lightweight causal head for CLAM instance-level supervision.

    Supervises top-B attended chunks with document-level labels.
    Separate from document-level head (no weight sharing).

    This enables instance-level learning where individual chunks
    are encouraged to predict both treatment and outcome, forcing
    the model to attend to causally relevant text.

    Args:
        input_dim: Dimension of chunk embeddings (transformer_dim)
        hidden_dim: Hidden dimension for the heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()

        self.propensity_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.outcome_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, chunk_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict propensity and outcome for chunk embeddings.

        Args:
            chunk_embeddings: (N, input_dim) - embeddings of top-attended chunks

        Returns:
            t_logit: (N, 1) - propensity logits for each chunk
            y_logit: (N, 1) - outcome logits for each chunk
        """
        t_logit = self.propensity_head(chunk_embeddings)
        y_logit = self.outcome_head(chunk_embeddings)
        return t_logit, y_logit


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


class EnhancedMatchedPairOutcomeModel(nn.Module):
    """
    Outcome + Tau model with cross-encoder residual features.

    Extends MatchedPairOutcomeModel by incorporating a ResidualCrossEncoder
    that identifies discriminative features between matched pairs at the
    sentence level. These residual features enhance tau prediction.

    Architecture:
        repr_U, repr_T (frozen, from PropensityMatchingModel)
        -> outcome_head (shared): Same as MatchedPairOutcomeModel
        -> cross_encoder: sent_T, sent_U -> residual_features
        -> tau_head (enhanced): repr_U + residual_features -> tau

    Args:
        representation_dim: Dimension of input representations
        hidden_dim: Hidden dimension for outcome and tau heads
        dropout: Dropout rate
        use_cross_encoder: Whether to use cross-encoder for residual features
        cross_encoder_num_queries: Number of discriminative queries in cross-encoder
        cross_encoder_num_heads: Number of attention heads in cross-encoder
        cross_encoder_hidden_dim: Hidden dimension for cross-encoder
        cross_encoder_use_gating: Whether to use gated attention in cross-encoder
        residual_weight: Optional learnable weight for residual features
    """

    def __init__(
        self,
        representation_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_cross_encoder: bool = True,
        cross_encoder_num_queries: int = 4,
        cross_encoder_num_heads: int = 4,
        cross_encoder_hidden_dim: int = 128,
        cross_encoder_use_gating: bool = True,
        residual_weight: Optional[float] = None
    ):
        super().__init__()

        self._representation_dim = representation_dim
        self._hidden_dim = hidden_dim
        self._use_cross_encoder = use_cross_encoder

        # Shared outcome head (predicts P(Y=1|repr)) - same as base model
        self.outcome_fc1 = nn.Linear(representation_dim, hidden_dim)
        self.outcome_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.outcome_fc3 = nn.Linear(hidden_dim, 1)

        # Cross-encoder for residual features
        self.cross_encoder = None
        if use_cross_encoder:
            self.cross_encoder = ResidualCrossEncoder(
                sentence_dim=representation_dim,
                hidden_dim=cross_encoder_hidden_dim,
                num_heads=cross_encoder_num_heads,
                num_discriminative_queries=cross_encoder_num_queries,
                dropout=dropout,
                use_gated_attention=cross_encoder_use_gating
            )

        # Enhanced tau head: takes repr_U + residual_features
        # Input dimension is representation_dim + representation_dim if using cross-encoder
        tau_input_dim = representation_dim + representation_dim if use_cross_encoder else representation_dim
        self.tau_fc1 = nn.Linear(tau_input_dim, hidden_dim)
        self.tau_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tau_fc3 = nn.Linear(hidden_dim, 1)

        # Learnable residual weight (optional)
        if residual_weight is not None:
            self.residual_weight = nn.Parameter(torch.tensor(residual_weight))
        else:
            self.residual_weight = None

        self.dropout = nn.Dropout(dropout)

        logger.info(f"EnhancedMatchedPairOutcomeModel initialized:")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Cross-encoder: {use_cross_encoder}")
        if use_cross_encoder:
            logger.info(f"    Num queries: {cross_encoder_num_queries}")
            logger.info(f"    Num heads: {cross_encoder_num_heads}")
            logger.info(f"    Gated attention: {cross_encoder_use_gating}")

    def forward(
        self,
        repr_U: torch.Tensor,
        repr_T: torch.Tensor,
        sent_U: Optional[List[torch.Tensor]] = None,
        sent_T: Optional[List[torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Forward pass for matched pair training.

        Args:
            repr_U: Untreated patient representations (B, D)
            repr_T: Treated patient representations (B, D)
            sent_U: List of sentence embeddings for untreated patients [(S_Ui, D), ...]
            sent_T: List of sentence embeddings for treated patients [(S_Ti, D), ...]
            return_attention: Whether to return cross-encoder attention weights

        Returns:
            y_U_logit: Outcome logit for untreated (B, 1)
            y_T_logit: Outcome logit for treated (B, 1)
            tau_pred: Treatment effect prediction on log-odds scale (B, 1)
            attention_info: List of attention dicts if return_attention=True
        """
        # Outcome predictions (shared weights) - same as base model
        y_U_logit = self._outcome_forward(repr_U)
        y_T_logit = self._outcome_forward(repr_T)

        # Cross-encoder residual features
        attention_info = None
        if self._use_cross_encoder and sent_U is not None and sent_T is not None:
            residual_features, attention_info = self.cross_encoder.forward_batch(
                sent_T, sent_U, return_attention=return_attention
            )

            # Apply residual weight if specified
            if self.residual_weight is not None:
                residual_features = self.residual_weight * residual_features

            # Enhanced tau input: repr_U + residual_features
            tau_input = torch.cat([repr_U, residual_features], dim=-1)
        else:
            # Fallback to base behavior if no sentence embeddings provided
            tau_input = torch.cat([repr_U, torch.zeros_like(repr_U)], dim=-1) if self._use_cross_encoder else repr_U

        # Tau prediction (enhanced)
        tau_pred = self._tau_forward(tau_input)

        return y_U_logit, y_T_logit, tau_pred, attention_info

    def _outcome_forward(self, repr: torch.Tensor) -> torch.Tensor:
        """Shared outcome prediction head."""
        h = F.relu(self.outcome_fc1(repr))
        h = self.dropout(h)
        h = F.relu(self.outcome_fc2(h))
        h = self.dropout(h)
        return self.outcome_fc3(h)

    def _tau_forward(self, tau_input: torch.Tensor) -> torch.Tensor:
        """Enhanced tau (treatment effect) prediction head."""
        h = F.relu(self.tau_fc1(tau_input))
        h = self.dropout(h)
        h = F.relu(self.tau_fc2(h))
        h = self.dropout(h)
        return self.tau_fc3(h)

    def predict_ite(
        self,
        repr: torch.Tensor,
        sent: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Predict ITE for patients given their representation.

        When no sentence embeddings are provided, uses zero residual features.

        Args:
            repr: Patient representations of shape (B, D)
            sent: Optional list of sentence embeddings

        Returns:
            tau on log-odds scale of shape (B, 1)
        """
        if self._use_cross_encoder:
            # Without sentence embeddings, use zeros for residual features
            residual_features = torch.zeros_like(repr)
            tau_input = torch.cat([repr, residual_features], dim=-1)
        else:
            tau_input = repr

        return self._tau_forward(tau_input)

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
        repr: torch.Tensor,
        sent: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict potential outcomes Y(0), Y(1) and ITE for a patient.

        Uses the outcome head for Y(0) and tau head to derive Y(1).

        Args:
            repr: Patient representations of shape (B, D)
            sent: Optional list of sentence embeddings

        Returns:
            Tuple of (y0_prob, y1_prob, ite_prob), each of shape (B, 1)
        """
        y0_logit = self._outcome_forward(repr)
        tau_logodds = self.predict_ite(repr, sent)

        # Y(1) = Y(0) + tau in logit space
        y1_logit = y0_logit + tau_logodds

        y0_prob = torch.sigmoid(y0_logit)
        y1_prob = torch.sigmoid(y1_logit)
        ite_prob = y1_prob - y0_prob

        return y0_prob, y1_prob, ite_prob

    def predict_treatment_from_residual(
        self,
        sent_T: List[torch.Tensor],
        sent_U: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Predict treatment status from residual features (auxiliary task).

        Args:
            sent_T: List of sentence embeddings for treated patients
            sent_U: List of sentence embeddings for untreated patients

        Returns:
            treatment_logit: Treatment prediction logit (B, 1)
        """
        if not self._use_cross_encoder:
            raise ValueError("Treatment prediction requires use_cross_encoder=True")

        residual_features, _ = self.cross_encoder.forward_batch(sent_T, sent_U, return_attention=False)
        return self.cross_encoder.predict_treatment(residual_features)

    @property
    def uses_cross_encoder(self) -> bool:
        """Check if cross-encoder is enabled."""
        return self._use_cross_encoder


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


class MeanEmbeddingITEModel(nn.Module):
    """
    ITE head trained on mean of matched pair embeddings.

    Key design:
    - Takes a FROZEN outcome_head from Stage 1 propensity model
    - Only the ite_head is trainable
    - Training uses mean(repr_T, repr_U), inference uses patient's own repr

    Architecture:
        ite_head: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear

    Training predictions (on matched pairs):
        mean_repr = (repr_T + repr_U) / 2
        Y_T_logit = frozen_outcome(mean_repr) + ite_head(mean_repr)
        Y_U_logit = frozen_outcome(mean_repr) - ite_head(mean_repr)

    Inference predictions (single patient):
        Y1_logit = frozen_outcome(repr) + ite_head(repr)
        Y0_logit = frozen_outcome(repr) - ite_head(repr)
        ITE = Y1_prob - Y0_prob

    Args:
        representation_dim: Dimension of input representations
        hidden_dim: Hidden dimension for ITE head
        dropout: Dropout rate
        frozen_outcome_head: nn.Module from Stage 1 (will be frozen)
    """

    def __init__(
        self,
        representation_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        frozen_outcome_head: Optional[nn.Module] = None
    ):
        super().__init__()

        self._representation_dim = representation_dim
        self._hidden_dim = hidden_dim

        # Store frozen outcome head (not trainable)
        if frozen_outcome_head is not None:
            self.frozen_outcome_head = frozen_outcome_head
            for param in self.frozen_outcome_head.parameters():
                param.requires_grad = False
        else:
            # Create a placeholder if not provided (will be set later)
            self.frozen_outcome_head = None

        # Only ite_head is trainable
        self.ite_head = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        logger.info(f"MeanEmbeddingITEModel initialized:")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Frozen outcome head: {'provided' if frozen_outcome_head else 'None'}")

    def set_frozen_outcome_head(self, outcome_head: nn.Module) -> None:
        """
        Set the frozen outcome head after construction.

        Args:
            outcome_head: nn.Module from Stage 1 (will be frozen)
        """
        self.frozen_outcome_head = outcome_head
        for param in self.frozen_outcome_head.parameters():
            param.requires_grad = False
        logger.info("Frozen outcome head set")

    def forward_training(
        self,
        repr_U: torch.Tensor,
        repr_T: torch.Tensor,
        external_outcome_head: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass on matched pairs.

        Computes predictions using mean of paired embeddings:
            mean_repr = (repr_T + repr_U) / 2
            base_logit = outcome_head(mean_repr)  [frozen or external/trainable]
            ite_half = ite_head(mean_repr)        [trainable]
            Y_U_logit = base_logit - ite_half
            Y_T_logit = base_logit + ite_half

        Args:
            repr_U: Untreated patient representations (B, D)
            repr_T: Treated patient representations (B, D)
            external_outcome_head: Optional external outcome head to use instead of
                frozen_outcome_head. When provided, this head is used WITH gradients
                (for unfrozen training mode). When None, uses self.frozen_outcome_head
                without gradients.

        Returns:
            Tuple of (Y_U_logit, Y_T_logit, ite_half_logit)
        """
        mean_repr = (repr_U + repr_T) / 2

        if external_outcome_head is not None:
            # Unfrozen mode: use external outcome head WITH gradients
            base_logit = external_outcome_head(mean_repr)
        else:
            # Frozen mode: use frozen_outcome_head WITHOUT gradients
            if self.frozen_outcome_head is None:
                raise ValueError("No outcome head available. Either provide external_outcome_head "
                               "or call set_frozen_outcome_head() first.")
            with torch.no_grad():
                base_logit = self.frozen_outcome_head(mean_repr)

        ite_half = self.ite_head(mean_repr)

        Y_U_logit = base_logit - ite_half
        Y_T_logit = base_logit + ite_half

        return Y_U_logit, Y_T_logit, ite_half

    def forward(
        self,
        repr_U: torch.Tensor,
        repr_T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alias for forward_training for API consistency."""
        return self.forward_training(repr_U, repr_T)

    def predict_potential_outcomes(
        self,
        repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference on single patient.

        Computes potential outcomes using patient's own representation:
            base_logit = frozen_outcome_head(repr)  [no gradient]
            ite_half = ite_head(repr)               [inference mode]
            Y0_logit = base_logit - ite_half
            Y1_logit = base_logit + ite_half

        Args:
            repr: Patient representations of shape (B, D)

        Returns:
            Tuple of (y0_prob, y1_prob, ite_prob)
        """
        if self.frozen_outcome_head is None:
            raise ValueError("Frozen outcome head not set. Call set_frozen_outcome_head() first.")

        with torch.no_grad():
            base_logit = self.frozen_outcome_head(repr)

        ite_half = self.ite_head(repr)

        y0_logit = base_logit - ite_half
        y1_logit = base_logit + ite_half

        y0_prob = torch.sigmoid(y0_logit)
        y1_prob = torch.sigmoid(y1_logit)
        ite_prob = y1_prob - y0_prob

        return y0_prob, y1_prob, ite_prob

    def predict_ite_logit(self, repr: torch.Tensor) -> torch.Tensor:
        """
        Predict ITE for single patient (logit scale = 2*ite_half).

        This gives the full ITE on logit scale, not just half.

        Args:
            repr: Patient representations of shape (B, D)

        Returns:
            ITE on logit scale of shape (B, 1)
        """
        return 2 * self.ite_head(repr)

    def predict_ite(self, repr: torch.Tensor) -> torch.Tensor:
        """
        Alias for predict_ite_logit for API consistency.

        Args:
            repr: Patient representations of shape (B, D)

        Returns:
            ITE on logit scale of shape (B, 1)
        """
        return self.predict_ite_logit(repr)

    def predict_outcome(self, repr: torch.Tensor) -> torch.Tensor:
        """
        Predict base outcome probability for a patient.

        This returns the frozen outcome head's prediction (P(Y=1|X)),
        which represents the baseline outcome without treatment effect adjustment.

        Args:
            repr: Patient representations of shape (B, D)

        Returns:
            Outcome probability of shape (B, 1)
        """
        if self.frozen_outcome_head is None:
            raise ValueError("Frozen outcome head not set. Call set_frozen_outcome_head() first.")

        with torch.no_grad():
            logit = self.frozen_outcome_head(repr)

        return torch.sigmoid(logit)


class EndToEndMatchedPairModel(nn.Module):
    """
    Unified model for end-to-end matched pair ITE estimation.

    Combines feature extraction, propensity prediction, outcome prediction,
    and tau prediction in a single model for joint training. Unlike the
    3-stage approach (PropensityMatchingModel + MatchedPairOutcomeModel),
    this model trains all components together with periodic re-matching.

    Key differences from 3-stage approach:
    - Single unified model with shared feature extractor
    - Propensity loss applied throughout training (not just Stage 1)
    - Representation always trainable (never frozen)
    - Re-matching is mandatory (computed periodically as model improves)

    Architecture:
        HierarchicalTransformerExtractor
        -> repr_layers (Linear -> ELU -> Linear -> LayerNorm)
        -> propensity_head (Linear -> ReLU -> Dropout -> Linear)
        -> outcome_head (shared for Y_U and Y_T prediction)
        -> tau_head (predicts ITE from untreated repr)

    Args:
        sentence_model: HuggingFace model name for sentence encoding
        freeze_sentence_encoder: Whether to freeze the sentence encoder weights
        max_sentences: Maximum number of sentences to process per document
        max_sentence_length: Maximum tokens per sentence for BERT encoding
        transformer_dim: Hidden dimension for transformer pooling layers
        num_transformer_layers: Number of transformer layers for pooling
        num_attention_heads: Number of attention heads in transformer layers
        transformer_dropout: Dropout rate for transformer layers
        representation_dim: Dimension of the learned representation layer
        hidden_outcome_dim: Hidden dimension for outcome and tau heads
        dropout: Dropout rate for heads
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
        hidden_outcome_dim: int = 128,
        dropout: float = 0.2,
        device: str = "cuda:0"
    ):
        super().__init__()

        self._device = torch.device(device) if isinstance(device, str) else device
        self._representation_dim = representation_dim
        self._hidden_outcome_dim = hidden_outcome_dim

        # Feature extractor: HierarchicalTransformerExtractor
        self.feature_extractor = HierarchicalTransformerExtractor(
            sentence_encoder_model=sentence_model,
            freeze_sentence_encoder=freeze_sentence_encoder,
            max_sentences=max_sentences,
            max_sentence_length=max_sentence_length,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            transformer_dim=transformer_dim,
            transformer_dropout=transformer_dropout,
            projection_dim=transformer_dim,
            device=self._device
        )

        # Representation layers
        self.repr_fc1 = nn.Linear(transformer_dim, representation_dim)
        self.repr_fc2 = nn.Linear(representation_dim, representation_dim)
        self.repr_norm = nn.LayerNorm(representation_dim)

        # Propensity head
        self.propensity_head = nn.Sequential(
            nn.Linear(representation_dim, representation_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(representation_dim // 2, 1)
        )

        # Outcome head (shared for Y_U and Y_T)
        self.outcome_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.outcome_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Tau head (predicts ITE from untreated repr)
        self.tau_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.tau_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.tau_fc3 = nn.Linear(hidden_outcome_dim, 1)

        self.dropout = nn.Dropout(dropout)

        logger.info(f"EndToEndMatchedPairModel initialized:")
        logger.info(f"  Sentence encoder: {sentence_model}")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Transformer dim: {transformer_dim}")
        logger.info(f"  Hidden outcome dim: {hidden_outcome_dim}")
        logger.info(f"  Device: {self._device}")

    def get_representation(self, texts: List[str]) -> torch.Tensor:
        """
        Extract representation from text.

        Args:
            texts: List of document texts

        Returns:
            Representation tensor of shape (batch_size, representation_dim)
        """
        features = self.feature_extractor(texts)  # (B, transformer_dim)
        h = F.relu(self.repr_fc1(features))
        h = F.elu(self.repr_fc2(h))
        return self.repr_norm(h)

    def predict_propensity(self, texts: List[str]) -> torch.Tensor:
        """
        Predict propensity score (probability of treatment).

        Args:
            texts: List of document texts

        Returns:
            Propensity probabilities of shape (batch_size,)
        """
        repr = self.get_representation(texts)
        return torch.sigmoid(self.propensity_head(repr)).squeeze(-1)

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

    def forward_matched_pair(
        self,
        texts_T: List[str],
        texts_U: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for matched pair training.

        Takes paired treated (T) and untreated (U) texts, computes all
        predictions needed for the joint loss.

        Args:
            texts_T: List of treated patient texts
            texts_U: List of untreated patient texts (matched to T)

        Returns:
            Dictionary with:
                - t_logit_T: Propensity logit for treated (B, 1)
                - t_logit_U: Propensity logit for untreated (B, 1)
                - y_U_logit: Outcome logit for untreated (B, 1)
                - y_T_logit: Outcome logit for treated (B, 1)
                - tau_pred: Treatment effect prediction (B, 1)
                - repr_T: Treated representation (B, D)
                - repr_U: Untreated representation (B, D)
        """
        repr_T = self.get_representation(texts_T)
        repr_U = self.get_representation(texts_U)

        # Propensity predictions
        t_logit_T = self.propensity_head(repr_T)
        t_logit_U = self.propensity_head(repr_U)

        # Outcome predictions (shared head)
        y_U_logit = self._outcome_forward(repr_U)
        y_T_logit = self._outcome_forward(repr_T)

        # Tau prediction (from U repr only)
        tau_pred = self._tau_forward(repr_U)

        return {
            't_logit_T': t_logit_T,
            't_logit_U': t_logit_U,
            'y_U_logit': y_U_logit,
            'y_T_logit': y_T_logit,
            'tau_pred': tau_pred,
            'repr_T': repr_T,
            'repr_U': repr_U
        }

    def predict_potential_outcomes(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict potential outcomes Y(0), Y(1) and ITE for inference.

        Uses the outcome head for Y(0) and tau head to derive Y(1):
            Y(1) = Y(0) + tau in logit space

        Args:
            texts: List of document texts

        Returns:
            Tuple of (y0_prob, y1_prob, ite_prob), each of shape (B, 1)
        """
        repr = self.get_representation(texts)
        y0_logit = self._outcome_forward(repr)
        tau_logodds = self._tau_forward(repr)

        # Y(1) = Y(0) + tau in logit space
        y1_logit = y0_logit + tau_logodds

        y0_prob = torch.sigmoid(y0_logit)
        y1_prob = torch.sigmoid(y1_logit)
        ite_prob = y1_prob - y0_prob

        return y0_prob, y1_prob, ite_prob

    def predict_tau(self, texts: List[str]) -> torch.Tensor:
        """
        Predict tau (ITE on log-odds scale) for any patient.

        Args:
            texts: List of document texts

        Returns:
            Tau on log-odds scale of shape (B, 1)
        """
        repr = self.get_representation(texts)
        return self._tau_forward(repr)

    def fit_tokenizer(self, texts: List[str]) -> 'EndToEndMatchedPairModel':
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

    @property
    def representation_dim(self) -> int:
        """Return the representation dimension."""
        return self._representation_dim

    def get_state(self) -> Dict[str, Any]:
        """Get model state for checkpoint saving."""
        return {
            'representation_dim': self._representation_dim,
            'hidden_outcome_dim': self._hidden_outcome_dim,
            'feature_extractor_state': self.feature_extractor.get_state()
        }


class EndToEndMatchedPairModelGRU(nn.Module):
    """
    End-to-end matched pair ITE model using GRU chunk encoder.

    Same architecture as EndToEndMatchedPairModel but uses
    HierarchicalGRUTransformerExtractor instead of HierarchicalTransformerExtractor.
    This processes text as overlapping token chunks with BiGRU + attention
    rather than sentences with bert-tiny.

    Key differences from EndToEndMatchedPairModel:
    - Uses overlapping fixed-size chunks (default: 128 tokens, 32 overlap)
    - BiGRU with learned attention per chunk instead of frozen bert-tiny [CLS]
    - Requires fit_tokenizer() to build vocabulary from training text
    - Guarantees confounder text appears fully in at least one chunk

    Args:
        chunk_size: Number of tokens per chunk (default: 128)
        chunk_overlap: Overlap between consecutive chunks (default: 32)
        max_chunks: Maximum number of chunks to process per document
        gru_embedding_dim: Dimension of word embeddings
        gru_hidden_dim: Hidden dimension for BiGRU
        gru_num_layers: Number of GRU layers
        chunk_dim: Output dimension of chunk encoder (input to transformer)
        num_transformer_layers: Number of transformer layers for pooling
        num_attention_heads: Number of attention heads in transformer layers
        transformer_dropout: Dropout rate for transformer layers
        representation_dim: Dimension of the learned representation layer
        hidden_outcome_dim: Hidden dimension for outcome and tau heads
        dropout: Dropout rate for heads
        max_vocab_size: Maximum vocabulary size
        min_word_freq: Minimum word frequency to include in vocabulary
        device: PyTorch device
    """

    def __init__(
        self,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        max_chunks: int = 100,
        gru_embedding_dim: int = 128,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        chunk_dim: int = 256,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        transformer_dropout: float = 0.1,
        representation_dim: int = 256,
        hidden_outcome_dim: int = 128,
        dropout: float = 0.2,
        max_vocab_size: int = 50000,
        min_word_freq: int = 2,
        device: str = "cuda:0"
    ):
        super().__init__()

        self._device = torch.device(device) if isinstance(device, str) else device
        self._representation_dim = representation_dim
        self._hidden_outcome_dim = hidden_outcome_dim
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Feature extractor: HierarchicalGRUTransformerExtractor
        self.feature_extractor = HierarchicalGRUTransformerExtractor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            embedding_dim=gru_embedding_dim,
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            chunk_dim=chunk_dim,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            transformer_dropout=transformer_dropout,
            projection_dim=chunk_dim,  # Use chunk_dim as intermediate
            max_vocab_size=max_vocab_size,
            min_word_freq=min_word_freq,
            device=self._device
        )

        # Representation layers
        self.repr_fc1 = nn.Linear(chunk_dim, representation_dim)
        self.repr_fc2 = nn.Linear(representation_dim, representation_dim)
        self.repr_norm = nn.LayerNorm(representation_dim)

        # Propensity head
        self.propensity_head = nn.Sequential(
            nn.Linear(representation_dim, representation_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(representation_dim // 2, 1)
        )

        # Outcome head (shared for Y_U and Y_T)
        self.outcome_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.outcome_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Tau head (predicts ITE from untreated repr)
        self.tau_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.tau_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.tau_fc3 = nn.Linear(hidden_outcome_dim, 1)

        self.dropout = nn.Dropout(dropout)

        logger.info(f"EndToEndMatchedPairModelGRU initialized:")
        logger.info(f"  Chunk size: {chunk_size}, overlap: {chunk_overlap}")
        logger.info(f"  GRU: embedding_dim={gru_embedding_dim}, hidden_dim={gru_hidden_dim}, layers={gru_num_layers}")
        logger.info(f"  Chunk dim: {chunk_dim}")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Hidden outcome dim: {hidden_outcome_dim}")
        logger.info(f"  Device: {self._device}")

    def get_representation(self, texts: List[str]) -> torch.Tensor:
        """
        Extract representation from text.

        Args:
            texts: List of document texts

        Returns:
            Representation tensor of shape (batch_size, representation_dim)
        """
        features = self.feature_extractor(texts)  # (B, chunk_dim)
        h = F.relu(self.repr_fc1(features))
        h = F.elu(self.repr_fc2(h))
        return self.repr_norm(h)

    def predict_propensity(self, texts: List[str]) -> torch.Tensor:
        """
        Predict propensity score (probability of treatment).

        Args:
            texts: List of document texts

        Returns:
            Propensity probabilities of shape (batch_size,)
        """
        repr = self.get_representation(texts)
        return torch.sigmoid(self.propensity_head(repr)).squeeze(-1)

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

    def forward_matched_pair(
        self,
        texts_T: List[str],
        texts_U: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for matched pair training.

        Takes paired treated (T) and untreated (U) texts, computes all
        predictions needed for the joint loss.

        Args:
            texts_T: List of treated patient texts
            texts_U: List of untreated patient texts (matched to T)

        Returns:
            Dictionary with:
                - t_logit_T: Propensity logit for treated (B, 1)
                - t_logit_U: Propensity logit for untreated (B, 1)
                - y_U_logit: Outcome logit for untreated (B, 1)
                - y_T_logit: Outcome logit for treated (B, 1)
                - tau_pred: Treatment effect prediction (B, 1)
                - repr_T: Treated representation (B, D)
                - repr_U: Untreated representation (B, D)
        """
        repr_T = self.get_representation(texts_T)
        repr_U = self.get_representation(texts_U)

        # Propensity predictions
        t_logit_T = self.propensity_head(repr_T)
        t_logit_U = self.propensity_head(repr_U)

        # Outcome predictions (shared head)
        y_U_logit = self._outcome_forward(repr_U)
        y_T_logit = self._outcome_forward(repr_T)

        # Tau prediction (from U repr only)
        tau_pred = self._tau_forward(repr_U)

        return {
            't_logit_T': t_logit_T,
            't_logit_U': t_logit_U,
            'y_U_logit': y_U_logit,
            'y_T_logit': y_T_logit,
            'tau_pred': tau_pred,
            'repr_T': repr_T,
            'repr_U': repr_U
        }

    def predict_potential_outcomes(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict potential outcomes Y(0), Y(1) and ITE for inference.

        Uses the outcome head for Y(0) and tau head to derive Y(1):
            Y(1) = Y(0) + tau in logit space

        Args:
            texts: List of document texts

        Returns:
            Tuple of (y0_prob, y1_prob, ite_prob), each of shape (B, 1)
        """
        repr = self.get_representation(texts)
        y0_logit = self._outcome_forward(repr)
        tau_logodds = self._tau_forward(repr)

        # Y(1) = Y(0) + tau in logit space
        y1_logit = y0_logit + tau_logodds

        y0_prob = torch.sigmoid(y0_logit)
        y1_prob = torch.sigmoid(y1_logit)
        ite_prob = y1_prob - y0_prob

        return y0_prob, y1_prob, ite_prob

    def predict_tau(self, texts: List[str]) -> torch.Tensor:
        """
        Predict tau (ITE on log-odds scale) for any patient.

        Args:
            texts: List of document texts

        Returns:
            Tau on log-odds scale of shape (B, 1)
        """
        repr = self.get_representation(texts)
        return self._tau_forward(repr)

    def fit_tokenizer(self, texts: List[str]) -> 'EndToEndMatchedPairModelGRU':
        """
        Fit the tokenizer and initialize the feature extractor.

        MUST be called before training or inference.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        self.feature_extractor.fit_tokenizer(texts)
        return self

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.feature_extractor = self.feature_extractor.to(device)
        return super().to(device)

    @property
    def representation_dim(self) -> int:
        """Return the representation dimension."""
        return self._representation_dim

    def get_state(self) -> Dict[str, Any]:
        """Get model state for checkpoint saving."""
        return {
            'representation_dim': self._representation_dim,
            'hidden_outcome_dim': self._hidden_outcome_dim,
            'chunk_size': self._chunk_size,
            'chunk_overlap': self._chunk_overlap,
            'feature_extractor_state': self.feature_extractor.get_state()
        }

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of chunk and token attention.

        Delegates to the feature extractor's interpret_attention method.

        Args:
            texts: List of document texts
            top_k: Number of top-attended chunks to show

        Returns:
            List of attention interpretations per document
        """
        return self.feature_extractor.interpret_attention(texts, top_k=top_k)
