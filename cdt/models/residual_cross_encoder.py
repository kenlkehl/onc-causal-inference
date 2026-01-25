# cdt/models/residual_cross_encoder.py
"""Residual Cross-Encoder for capturing discriminative features between matched pairs.

This module implements a cross-encoder that uses bidirectional cross-attention
between sentence embeddings from matched pairs (treated, untreated) to identify
discriminative features that may represent residual confounders missed by
propensity matching.

Architecture:
    sent_T (S_T, D) ──┐
                      ├── Bidirectional Cross-Attention ──> residual_features (D,)
    sent_U (S_U, D) ──┘

The cross-encoder learns to identify which sentences in each patient's record
most distinguish them from their matched counterpart. These residual features
are then used to enhance tau (treatment effect) estimation.
"""

import logging
from typing import Optional, List, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class ResidualCrossEncoder(nn.Module):
    """
    Cross-encoder for extracting discriminative features between matched pairs.

    Uses bidirectional cross-attention between sentence embeddings from treated
    and untreated patients to identify features that distinguish them despite
    having similar propensity scores.

    Architecture:
        1. Cross-attention: T sentences attend to U sentences
        2. Cross-attention: U sentences attend to T sentences
        3. Gated attention (tanh * sigmoid) for focused aggregation
        4. Discriminative query aggregation
        5. Output projection to residual features

    Args:
        sentence_dim: Dimension of input sentence embeddings
        hidden_dim: Hidden dimension for cross-attention layers
        num_heads: Number of attention heads
        num_discriminative_queries: Number of learnable discriminative queries (K)
        dropout: Dropout rate
        use_gated_attention: Whether to use gated attention (tanh * sigmoid)
    """

    def __init__(
        self,
        sentence_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_discriminative_queries: int = 4,
        dropout: float = 0.1,
        use_gated_attention: bool = True
    ):
        super().__init__()

        self._sentence_dim = sentence_dim
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._num_queries = num_discriminative_queries
        self._use_gated_attention = use_gated_attention

        # Project sentence embeddings to hidden dim if needed
        self.input_proj = nn.Linear(sentence_dim, hidden_dim) if sentence_dim != hidden_dim else nn.Identity()

        # Bidirectional cross-attention layers
        # T attends to U: find what in U is relevant to understand T
        self.cross_attn_T_to_U = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # U attends to T: find what in T is relevant to understand U
        self.cross_attn_U_to_T = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norms for cross-attention
        self.norm_T = nn.LayerNorm(hidden_dim)
        self.norm_U = nn.LayerNorm(hidden_dim)

        # Gated attention components
        if use_gated_attention:
            # tanh gate
            self.gate_tanh = nn.Linear(hidden_dim, hidden_dim)
            # sigmoid gate
            self.gate_sigmoid = nn.Linear(hidden_dim, hidden_dim)

        # Discriminative queries for aggregating cross-attended features
        # These learn to extract specific discriminative patterns
        self.discriminative_queries = nn.Parameter(
            torch.randn(num_discriminative_queries, hidden_dim) * 0.02
        )

        # Aggregation attention: queries attend to cross-attended features
        self.aggregation_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection: aggregate queries -> residual features
        self.output_proj = nn.Sequential(
            nn.Linear(num_discriminative_queries * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, sentence_dim)  # Project back to sentence_dim for compatibility
        )

        # Optional: treatment discrimination head (auxiliary loss)
        # Predicts which patient is treated based on residual features
        self.discrimination_head = nn.Sequential(
            nn.Linear(sentence_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)

        logger.info(f"ResidualCrossEncoder initialized:")
        logger.info(f"  Sentence dim: {sentence_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Num heads: {num_heads}")
        logger.info(f"  Num discriminative queries: {num_discriminative_queries}")
        logger.info(f"  Gated attention: {use_gated_attention}")

    def forward(
        self,
        sent_T: torch.Tensor,
        sent_U: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for a single matched pair.

        Args:
            sent_T: Sentence embeddings for treated patient (S_T, D) or (B, S_T, D)
            sent_U: Sentence embeddings for untreated patient (S_U, D) or (B, S_U, D)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of:
                - residual_features: Discriminative features (D,) or (B, D)
                - attention_info: Dict with attention weights if return_attention=True
        """
        # Handle both single sample and batch
        single_sample = sent_T.dim() == 2
        if single_sample:
            sent_T = sent_T.unsqueeze(0)  # (1, S_T, D)
            sent_U = sent_U.unsqueeze(0)  # (1, S_U, D)

        batch_size = sent_T.size(0)

        # Project to hidden dim
        T_proj = self.input_proj(sent_T)  # (B, S_T, H)
        U_proj = self.input_proj(sent_U)  # (B, S_U, H)

        # Cross-attention: T attends to U
        T_cross, attn_T_to_U = self.cross_attn_T_to_U(
            query=T_proj, key=U_proj, value=U_proj,
            need_weights=return_attention
        )
        T_cross = self.norm_T(T_proj + self.dropout(T_cross))  # (B, S_T, H)

        # Cross-attention: U attends to T
        U_cross, attn_U_to_T = self.cross_attn_U_to_T(
            query=U_proj, key=T_proj, value=T_proj,
            need_weights=return_attention
        )
        U_cross = self.norm_U(U_proj + self.dropout(U_cross))  # (B, S_U, H)

        # Concatenate cross-attended features
        # This contains features from both perspectives
        cross_features = torch.cat([T_cross, U_cross], dim=1)  # (B, S_T+S_U, H)

        # Apply gated attention
        if self._use_gated_attention:
            gate_tanh = torch.tanh(self.gate_tanh(cross_features))
            gate_sigmoid = torch.sigmoid(self.gate_sigmoid(cross_features))
            cross_features = cross_features * gate_tanh * gate_sigmoid

        # Expand discriminative queries for batch
        queries = self.discriminative_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, K, H)

        # Aggregate with discriminative queries
        aggregated, attn_agg = self.aggregation_attn(
            query=queries, key=cross_features, value=cross_features,
            need_weights=return_attention
        )  # (B, K, H)

        # Flatten and project to output
        aggregated_flat = aggregated.reshape(batch_size, -1)  # (B, K*H)
        residual_features = self.output_proj(aggregated_flat)  # (B, D)

        if single_sample:
            residual_features = residual_features.squeeze(0)  # (D,)

        attention_info = None
        if return_attention:
            attention_info = {
                'attn_T_to_U': attn_T_to_U,  # (B, S_T, S_U)
                'attn_U_to_T': attn_U_to_T,  # (B, S_U, S_T)
                'attn_aggregation': attn_agg  # (B, K, S_T+S_U)
            }

        return residual_features, attention_info

    def forward_batch(
        self,
        sent_T_list: List[torch.Tensor],
        sent_U_list: List[torch.Tensor],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Forward pass for a batch of matched pairs with variable-length sentences.

        Args:
            sent_T_list: List of sentence embeddings for treated patients [(S_Ti, D), ...]
            sent_U_list: List of sentence embeddings for untreated patients [(S_Ui, D), ...]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of:
                - residual_features: Discriminative features (B, D)
                - attention_info_list: List of attention dicts if return_attention=True
        """
        batch_size = len(sent_T_list)
        residual_features_list = []
        attention_info_list = [] if return_attention else None

        for i in range(batch_size):
            residual, attn_info = self.forward(
                sent_T_list[i], sent_U_list[i], return_attention
            )
            residual_features_list.append(residual)
            if return_attention:
                attention_info_list.append(attn_info)

        residual_features = torch.stack(residual_features_list, dim=0)  # (B, D)

        return residual_features, attention_info_list

    def predict_treatment(self, residual_features: torch.Tensor) -> torch.Tensor:
        """
        Predict treatment status from residual features (auxiliary task).

        This head is trained to discriminate treated from untreated patients
        based on residual features, encouraging the cross-encoder to learn
        discriminative features.

        Args:
            residual_features: Discriminative features (B, D) or (D,)

        Returns:
            treatment_logit: Treatment prediction logit (B, 1) or (1,)
        """
        return self.discrimination_head(residual_features)

    def interpret_discrimination(
        self,
        sent_T: torch.Tensor,
        sent_U: torch.Tensor,
        sentences_T: List[str],
        sentences_U: List[str],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Interpret which sentences are most discriminative between matched pairs.

        Analyzes cross-attention weights to identify:
        1. Which T sentences most attended to which U sentences
        2. Which U sentences most attended to which T sentences
        3. Which sentences received highest aggregation attention

        Args:
            sent_T: Sentence embeddings for treated patient (S_T, D)
            sent_U: Sentence embeddings for untreated patient (S_U, D)
            sentences_T: List of sentence strings for treated patient
            sentences_U: List of sentence strings for untreated patient
            top_k: Number of top sentence pairs to return

        Returns:
            Dict with interpretation results:
                - top_T_sentences: Top sentences from T that distinguish it from U
                - top_U_sentences: Top sentences from U that distinguish it from T
                - top_cross_pairs: Top (T sentence, U sentence) pairs by attention
                - aggregation_weights: Which sentences received most query attention
        """
        self.eval()
        with torch.no_grad():
            residual, attn_info = self.forward(sent_T, sent_U, return_attention=True)

        # Ensure tensors are on CPU for processing
        attn_T_to_U = attn_info['attn_T_to_U'].squeeze(0).cpu().numpy()  # (S_T, S_U)
        attn_U_to_T = attn_info['attn_U_to_T'].squeeze(0).cpu().numpy()  # (S_U, S_T)
        attn_agg = attn_info['attn_aggregation'].squeeze(0).cpu().numpy()  # (K, S_T+S_U)

        # Find top T sentences (those that attend most strongly to U)
        T_importance = attn_T_to_U.max(axis=1)  # Max attention per T sentence
        top_T_indices = T_importance.argsort()[-top_k:][::-1]
        top_T_sentences = [
            {'sentence': sentences_T[i], 'importance': float(T_importance[i]), 'index': int(i)}
            for i in top_T_indices if i < len(sentences_T)
        ]

        # Find top U sentences (those that attend most strongly to T)
        U_importance = attn_U_to_T.max(axis=1)  # Max attention per U sentence
        top_U_indices = U_importance.argsort()[-top_k:][::-1]
        top_U_sentences = [
            {'sentence': sentences_U[i], 'importance': float(U_importance[i]), 'index': int(i)}
            for i in top_U_indices if i < len(sentences_U)
        ]

        # Find top cross-attention pairs
        n_T, n_U = attn_T_to_U.shape
        flat_indices = attn_T_to_U.flatten().argsort()[-top_k:][::-1]
        top_cross_pairs = []
        for flat_idx in flat_indices:
            t_idx = flat_idx // n_U
            u_idx = flat_idx % n_U
            if t_idx < len(sentences_T) and u_idx < len(sentences_U):
                top_cross_pairs.append({
                    'T_sentence': sentences_T[t_idx],
                    'U_sentence': sentences_U[u_idx],
                    'attention': float(attn_T_to_U[t_idx, u_idx]),
                    'T_index': int(t_idx),
                    'U_index': int(u_idx)
                })

        # Aggregation weights: average across queries
        agg_weights = attn_agg.mean(axis=0)  # (S_T+S_U,)
        n_T_sent = len(sentences_T)

        # Split aggregation weights for T and U
        agg_T = agg_weights[:n_T_sent] if n_T_sent <= len(agg_weights) else agg_weights[:len(agg_weights)//2]
        agg_U = agg_weights[n_T_sent:] if n_T_sent <= len(agg_weights) else agg_weights[len(agg_weights)//2:]

        return {
            'top_T_sentences': top_T_sentences,
            'top_U_sentences': top_U_sentences,
            'top_cross_pairs': top_cross_pairs,
            'aggregation_weights_T': agg_T.tolist(),
            'aggregation_weights_U': agg_U.tolist(),
            'residual_features_norm': float(residual.norm().cpu())
        }

    @property
    def output_dim(self) -> int:
        """Return the output dimension of residual features."""
        return self._sentence_dim
