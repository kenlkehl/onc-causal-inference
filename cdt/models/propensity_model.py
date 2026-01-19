# cdt/models/propensity_model.py
"""Propensity score model with optional joint outcome prediction for learning true confounders."""

import logging
import math
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .feature_extractor import FeatureExtractor, pad_chunks


logger = logging.getLogger(__name__)


class CNNEncoder(nn.Module):
    """1D CNN encoder for text chunk embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_filters: int = 128,
        kernel_sizes: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.fc = nn.Linear(num_filters * len(kernel_sizes), hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embedding_dim)
            mask: (batch, 1, seq_len) boolean mask where True = padding

        Returns:
            (batch, hidden_dim)
        """
        # x: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)

        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            h = F.relu(conv(x))  # (B, num_filters, L)
            # Mask out padding
            h = h.masked_fill(mask.transpose(1, 2).expand_as(h), float('-inf'))
            # Max pooling over sequence
            h = h.max(dim=2)[0]  # (B, num_filters)
            conv_outputs.append(h)

        # Concatenate and project
        h = torch.cat(conv_outputs, dim=1)  # (B, num_filters * len(kernel_sizes))
        h = self.dropout(h)
        h = F.relu(self.fc(h))

        return h


class TransformerEncoder(nn.Module):
    """Transformer encoder for text chunk embeddings (BERT-style)."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embedding_dim)
            mask: (batch, 1, seq_len) boolean mask where True = padding

        Returns:
            (batch, hidden_dim)
        """
        # Convert mask to transformer format (True = ignore)
        src_key_padding_mask = mask.squeeze(1)  # (B, L)

        # Apply transformer
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, L, D)

        # Mean pooling over non-padded positions
        valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()  # (B, L, 1)
        h = (h * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # (B, D)

        h = self.dropout(h)
        h = F.relu(self.fc(h))

        return h


class GRUAttentionEncoder(nn.Module):
    """Bidirectional GRU with attention mechanism."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embedding_dim)
            mask: (batch, 1, seq_len) boolean mask where True = padding

        Returns:
            (batch, hidden_dim)
        """
        # Pack padded sequence for efficiency
        lengths = (~mask.squeeze(1)).sum(dim=1).cpu()

        # GRU forward
        h, _ = self.gru(x)  # (B, L, hidden_dim)

        # Attention scores
        attn_scores = self.attention(h).squeeze(-1)  # (B, L)
        attn_scores = attn_scores.masked_fill(mask.squeeze(1), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, L)

        # Weighted sum
        h = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, hidden_dim)
        h = self.dropout(h)

        return h


class PropensityHead(nn.Module):
    """Propensity score prediction head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns propensity logit."""
        return self.network(x)


class OutcomeHead(nn.Module):
    """Outcome prediction head for joint training."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns outcome logit."""
        return self.network(x)


class PropensityModel(nn.Module):
    """
    Propensity score model with optional joint outcome prediction.

    Supports three encoder architectures:
    - 'cnn': 1D CNN with multiple kernel sizes
    - 'transformer': BERT-style transformer encoder
    - 'gru': Bidirectional GRU with attention

    Can optionally jointly predict outcomes to encourage learning of true confounders.
    """

    def __init__(
        self,
        sentence_transformer_model_name: str = 'all-MiniLM-L6-v2',
        encoder_type: str = 'gru',  # 'cnn', 'transformer', 'gru'
        hidden_dim: int = 256,
        num_latent_confounders: int = 20,
        features_per_confounder: int = 1,
        explicit_confounder_texts: Optional[List[str]] = None,
        explicit_confounder_embeddings: Optional[torch.Tensor] = None,
        aggregator_mode: str = 'attn',
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        dropout: float = 0.1,
        joint_outcome_prediction: bool = False,
        outcome_weight: float = 0.5,  # Weight for outcome loss in joint training
        use_confounder_features: bool = True,  # Use FeatureExtractor or raw embeddings
        arctanh_transform: bool = False,
        device: str = "cuda:0"
    ):
        """
        Initialize propensity model.

        Args:
            sentence_transformer_model_name: Name of sentence transformer model
            encoder_type: Type of encoder ('cnn', 'transformer', 'gru')
            hidden_dim: Hidden dimension for encoder and heads
            num_latent_confounders: Number of learnable confounder patterns
            features_per_confounder: Features per confounder
            explicit_confounder_texts: Optional explicit confounder queries
            explicit_confounder_embeddings: Optional pre-computed embeddings
            aggregator_mode: Aggregation mode for confounder features
            chunk_size: Text chunk size in words
            chunk_overlap: Overlap between chunks
            dropout: Dropout rate
            joint_outcome_prediction: Whether to jointly predict outcomes
            outcome_weight: Weight for outcome loss (0-1)
            use_confounder_features: If True, use FeatureExtractor; else use encoder on raw embeddings
            arctanh_transform: Apply arctanh to cosine similarities
            device: Device string
        """
        super().__init__()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._device = torch.device(device)
        self.encoder_type = encoder_type
        self.joint_outcome_prediction = joint_outcome_prediction
        self.outcome_weight = outcome_weight
        self.use_confounder_features = use_confounder_features

        # Store config for checkpointing
        self.config = {
            'sentence_transformer_model_name': sentence_transformer_model_name,
            'encoder_type': encoder_type,
            'hidden_dim': hidden_dim,
            'num_latent_confounders': num_latent_confounders,
            'features_per_confounder': features_per_confounder,
            'explicit_confounder_texts': explicit_confounder_texts,
            'aggregator_mode': aggregator_mode,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'dropout': dropout,
            'joint_outcome_prediction': joint_outcome_prediction,
            'outcome_weight': outcome_weight,
            'use_confounder_features': use_confounder_features,
            'arctanh_transform': arctanh_transform
        }

        # Load sentence transformer
        self.sentence_transformer_model = SentenceTransformer(
            sentence_transformer_model_name,
            device=self._device
        )
        self.embedding_dim = self.sentence_transformer_model.get_sentence_embedding_dimension()

        # Feature extractor (if using confounder features)
        if use_confounder_features:
            self.feature_extractor = FeatureExtractor(
                embedding_dim=self.embedding_dim,
                num_latent_confounders=num_latent_confounders,
                explicit_confounder_texts=explicit_confounder_texts,
                features_per_confounder=features_per_confounder,
                aggregator_mode=aggregator_mode,
                sentence_transformer_model=self.sentence_transformer_model,
                phantom_confounders=0,
                device=self._device,
                explicit_confounder_embeddings=explicit_confounder_embeddings,
                arctanh_transform=arctanh_transform
            )
            encoder_input_dim = self.feature_extractor.output_dim
        else:
            self.feature_extractor = None
            encoder_input_dim = self.embedding_dim

        # Create encoder based on type
        if encoder_type == 'cnn':
            self.encoder = CNNEncoder(
                embedding_dim=encoder_input_dim if not use_confounder_features else self.embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                embedding_dim=encoder_input_dim if not use_confounder_features else self.embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif encoder_type == 'gru':
            self.encoder = GRUAttentionEncoder(
                embedding_dim=encoder_input_dim if not use_confounder_features else self.embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Determine propensity head input dimension
        if use_confounder_features:
            # Use both confounder features and encoder output
            propensity_input_dim = self.feature_extractor.output_dim + hidden_dim
        else:
            propensity_input_dim = hidden_dim

        # Propensity head
        self.propensity_head = PropensityHead(
            input_dim=propensity_input_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )

        # Optional outcome head for joint training
        if joint_outcome_prediction:
            self.outcome_head = OutcomeHead(
                input_dim=propensity_input_dim,
                hidden_dim=hidden_dim // 2,
                dropout=dropout
            )
        else:
            self.outcome_head = None

        # Move to device
        self.to(self._device)

        logger.info(f"PropensityModel initialized:")
        logger.info(f"  Encoder type: {encoder_type}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Joint outcome prediction: {joint_outcome_prediction}")
        logger.info(f"  Use confounder features: {use_confounder_features}")
        logger.info(f"  Device: {self._device}")

    def forward(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            chunk_embeddings_list: List of tensors, each (num_chunks, embedding_dim)

        Returns:
            Dictionary with:
                - propensity_logit: (batch, 1)
                - outcome_logit: (batch, 1) if joint_outcome_prediction else None
                - representation: (batch, hidden_dim)
                - confounder_features: (batch, num_features) if use_confounder_features else None
        """
        device = chunk_embeddings_list[0].device
        padded_chunks, mask = pad_chunks(chunk_embeddings_list, device)

        # Get encoder representation
        encoder_output = self.encoder(padded_chunks, mask)  # (B, hidden_dim)

        # Get confounder features if using
        if self.use_confounder_features:
            confounder_features = self.feature_extractor(chunk_embeddings_list)  # (B, num_features)
            # Concatenate for propensity prediction
            combined = torch.cat([confounder_features, encoder_output], dim=1)
        else:
            confounder_features = None
            combined = encoder_output

        # Propensity prediction
        propensity_logit = self.propensity_head(combined)

        # Optional outcome prediction
        if self.joint_outcome_prediction and self.outcome_head is not None:
            outcome_logit = self.outcome_head(combined)
        else:
            outcome_logit = None

        return {
            'propensity_logit': propensity_logit,
            'outcome_logit': outcome_logit,
            'representation': encoder_output,
            'confounder_features': confounder_features
        }

    def train_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Returns:
            Dictionary with loss components and detached predictions
        """
        chunk_embeddings_list = batch['chunk_embeddings']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch.get('outcome')  # (batch,) - may be None if not joint training

        # Forward pass
        outputs = self.forward(chunk_embeddings_list)

        # Propensity loss
        propensity_loss = F.binary_cross_entropy_with_logits(
            outputs['propensity_logit'].squeeze(-1),
            treatments
        )

        # Outcome loss (if joint training)
        if self.joint_outcome_prediction and outcomes is not None and outputs['outcome_logit'] is not None:
            outcome_loss = F.binary_cross_entropy_with_logits(
                outputs['outcome_logit'].squeeze(-1),
                outcomes
            )
            total_loss = (1 - self.outcome_weight) * propensity_loss + self.outcome_weight * outcome_loss
        else:
            outcome_loss = torch.tensor(0.0, device=self._device)
            total_loss = propensity_loss

        return {
            'loss': total_loss,
            'propensity_loss': propensity_loss.detach(),
            'outcome_loss': outcome_loss.detach() if isinstance(outcome_loss, torch.Tensor) else outcome_loss,
            'propensity_logit': outputs['propensity_logit'].detach(),
            'outcome_logit': outputs['outcome_logit'].detach() if outputs['outcome_logit'] is not None else None,
            'representation': outputs['representation'].detach(),
            'confounder_features': outputs['confounder_features'].detach() if outputs['confounder_features'] is not None else None
        }

    def predict(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for inference.

        Returns:
            Dictionary with propensity scores and representations
        """
        with torch.no_grad():
            outputs = self.forward(chunk_embeddings_list)

            propensity = torch.sigmoid(outputs['propensity_logit']).squeeze(-1)

            result = {
                'propensity': propensity,
                'propensity_logit': outputs['propensity_logit'].squeeze(-1),
                'representation': outputs['representation']
            }

            if outputs['confounder_features'] is not None:
                result['confounder_features'] = outputs['confounder_features']

            if outputs['outcome_logit'] is not None:
                result['outcome_prob'] = torch.sigmoid(outputs['outcome_logit']).squeeze(-1)
                result['outcome_logit'] = outputs['outcome_logit'].squeeze(-1)

            return result

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict()
        }

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
    ) -> 'PropensityModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

        if device is not None:
            config['device'] = device

        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        logger.info(f"Model loaded from {path}")
        return model
