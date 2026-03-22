# oci/models/gated_attention_pooling.py
"""Gated attention pooling module for aggregating sequences."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPooling(nn.Module):
    """
    Gated attention pooling (tanh x sigmoid) for aggregating sequences.

    Used for final document aggregation (chunks -> document vector).
    The gating mechanism allows learning which chunks to suppress (via sigmoid gate)
    while extracting content (via tanh).

    Formula:
        g = tanh(V(h)) * sigmoid(U(h))  # Gated features
        g' = LayerNorm(g)
        s = v(W(g'))  # Attention scores
        w = softmax(s)  # Attention weights
        output = sum(w * h)  # Weighted sum of original embeddings
    """

    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None):
        """
        Initialize gated attention pooling.

        Args:
            hidden_dim: Dimension of input hidden states
            attention_dim: Dimension of attention hidden layer (default: hidden_dim)
        """
        super().__init__()
        attention_dim = attention_dim or hidden_dim

        # Gating transforms
        self.V = nn.Linear(hidden_dim, attention_dim)  # tanh branch (content)
        self.U = nn.Linear(hidden_dim, attention_dim)  # sigmoid branch (gate)

        # Attention computation
        self.W = nn.Linear(attention_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

        self.layer_norm = nn.LayerNorm(attention_dim)

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gated attention pooling.

        Args:
            chunk_embeddings: (C, D) or (B, C, D) - transformed chunk embeddings
            attention_mask: Optional mask for valid chunks (1 = valid, 0 = padding)

        Returns:
            pooled: (D,) or (B, D) - document representation
            weights: (C,) or (B, C) - attention weights over chunks
        """
        # Gated features: g = tanh(V(h)) * sigmoid(U(h))
        tanh_branch = torch.tanh(self.V(chunk_embeddings))
        sigmoid_branch = torch.sigmoid(self.U(chunk_embeddings))
        gated = tanh_branch * sigmoid_branch
        gated = self.layer_norm(gated)

        # Attention scores
        scores = self.v(self.W(gated)).squeeze(-1)

        # Apply mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)

        # Handle both single sample and batch
        if chunk_embeddings.dim() == 2:
            pooled = (weights.unsqueeze(-1) * chunk_embeddings).sum(dim=0)
        else:
            pooled = torch.bmm(weights.unsqueeze(1), chunk_embeddings).squeeze(1)

        return pooled, weights
