# cdt/models/outcome_model.py
"""Outcome-only model for assessing prognostic signal in data."""

import gc
import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_extractor import CNNFeatureExtractor
from .bert_extractor import BertFeatureExtractor
from .gru_extractor import GRUFeatureExtractor
from ..config import normalize_feature_extractor_type


logger = logging.getLogger(__name__)


class OutcomeNet(nn.Module):
    """
    Outcome prediction network with same representation as DragonNet.

    Uses 2-layer representation followed by a single outcome head.
    """

    def __init__(self, input_dim: int, representation_dim: int = 200):
        super().__init__()

        # Shared representation layers (same as PropensityNet)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc6 = nn.Linear(representation_dim, representation_dim)

        # Single outcome head
        self.outcome_fc1 = nn.Linear(representation_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the outcome network.

        Args:
            features: Feature tensor from feature extractor (batch, input_dim)

        Returns:
            y_logit: Outcome logits (batch, 1)
        """
        h = F.relu(self.representation_fc1(features))
        h = F.elu(self.representation_fc6(h))

        y_logit = self.outcome_fc1(h)

        return y_logit


class OutcomeOnlyModel(nn.Module):
    """
    Outcome-only model for assessing prognostic signal.

    Uses same architecture as CausalText/DragonNet:
    - Feature extractor (CNN or BERT)
    - 2-layer representation network
    - Single outcome head

    This model is trained to predict P(Y=1|X) using binary cross-entropy loss.
    Used for understanding prognostic signal in data before DragonNet training.
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
        # Outcome network args
        representation_dim: int = 128,
        device: str = "cuda:0"
    ):
        """
        Initialize outcome-only model.

        Args:
            feature_extractor_type: "cnn", "bert", or "gru"
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
            'representation_dim': representation_dim
        }

        # Initialize feature extractor based on normalized type
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
            logger.info(f"Outcome model using BERT feature extractor: {bert_model_name}")
        elif self.feature_extractor_type == "gru":
            self.feature_extractor = GRUFeatureExtractor(
                embedding_dim=gru_embedding_dim,
                hidden_dim=gru_hidden_dim,
                num_layers=gru_num_layers,
                dropout=gru_dropout,
                bidirectional=gru_bidirectional,
                attention_dim=gru_attention_dim,
                projection_dim=gru_projection_dim,
                max_length=gru_max_length,
                min_word_freq=gru_min_word_freq,
                max_vocab_size=gru_max_vocab_size,
                device=self._device
            )
            logger.info("Outcome model using GRU feature extractor")
        else:
            # CNN feature extractor (default)
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
            logger.info("Outcome model using CNN feature extractor")

        # Outcome network
        input_dim = self.feature_extractor.output_dim
        self.outcome_net = OutcomeNet(
            input_dim=input_dim,
            representation_dim=representation_dim
        )

        # Move to device
        self.to(self._device)

        logger.info(f"OutcomeOnlyModel initialized:")
        logger.info(f"  Feature extractor: {feature_extractor_type}")
        logger.info(f"  Feature extractor output: {input_dim}")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Device: {self._device}")

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            texts: List of text strings

        Returns:
            y_logit: Outcome logits (batch, 1)
        """
        features = self.feature_extractor(texts)
        y_logit = self.outcome_net(features)
        return y_logit

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Args:
            batch: Dictionary with 'texts' and 'outcome' keys

        Returns:
            Dictionary with loss and predictions
        """
        texts = batch['texts']
        outcomes = batch['outcome']  # (batch,)

        # Forward pass
        y_logit = self.forward(texts)

        # Binary cross-entropy loss for outcome prediction
        loss = F.binary_cross_entropy_with_logits(
            y_logit.squeeze(-1),
            outcomes
        )

        return {
            'loss': loss,
            'y_logit': y_logit.detach()
        }

    def predict(self, texts: List[str]) -> torch.Tensor:
        """
        Predict outcome probabilities.

        Args:
            texts: List of text strings

        Returns:
            Outcome probabilities (batch,)
        """
        with torch.no_grad():
            y_logit = self.forward(texts)
            outcome_prob = torch.sigmoid(y_logit).squeeze(-1)
            return outcome_prob

    def fit_tokenizer(self, texts: List[str]) -> 'OutcomeOnlyModel':
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


def create_outcome_model_from_config(
    arch_config,
    representation_dim: int,
    device: torch.device
) -> OutcomeOnlyModel:
    """
    Create an OutcomeOnlyModel from architecture config.

    Args:
        arch_config: ModelArchitectureConfig instance
        representation_dim: Dimension for representation layers
        device: PyTorch device

    Returns:
        OutcomeOnlyModel instance
    """
    feature_extractor_type = getattr(arch_config, 'feature_extractor_type', 'cnn')

    model = OutcomeOnlyModel(
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
        # Outcome network args
        representation_dim=representation_dim,
        device=str(device)
    )

    return model
