# cdt/models/components.py
"""Aggregator modules for pooling chunk embeddings into confounder representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionAggregator(nn.Module):
    """
    Aggregate chunk embeddings using confounder-driven cross-attention.

    Unlike the old ConfounderAggregator which outputs scalar similarities,
    this aggregates transformed chunk content using cosine similarity as
    attention weights.

    Architecture:
    - Cosine similarities between confounders and chunks serve as attention scores
    - Chunks are projected to value vectors via learnable W_v
    - Multi-head attention allows capturing multiple aspects per confounder
    - Output is the weighted sum of value vectors per confounder
    """

    def __init__(
        self,
        embedding_dim: int,
        value_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention aggregator.

        Args:
            embedding_dim: Dimension of chunk embeddings (e.g., 384 for MiniLM)
            value_dim: Output dimension per confounder (e.g., 128)
            num_heads: Number of attention heads per confounder
            dropout: Dropout rate on attention weights
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = value_dim // num_heads

        if value_dim % num_heads != 0:
            raise ValueError(f"value_dim ({value_dim}) must be divisible by num_heads ({num_heads})")

        # Value projection: chunks (B, L, D) -> values (B, L, d_v)
        self.W_v = nn.Linear(embedding_dim, value_dim, bias=False)

        # Learnable temperature per head (for attention softmax)
        self.log_tau = nn.Parameter(torch.zeros(num_heads))

        # Output projection after concatenating heads
        self.out_proj = nn.Linear(value_dim, value_dim)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(value_dim)

    def forward(
        self,
        attn_scores: torch.Tensor,
        chunk_embeddings: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-attention aggregation.

        Args:
            attn_scores: Cosine similarities from confounder matching (B, C, L)
            chunk_embeddings: Raw chunk embeddings (B, L, D)
            mask: Padding mask (B, 1, L) where True = padding

        Returns:
            Aggregated features: (B, C, d_v) - value_dim per confounder
        """
        B, C, L = attn_scores.shape

        # 1. Project chunks to values: (B, L, d_v)
        V = self.W_v(chunk_embeddings)

        # 2. Reshape for multi-head: (B, L, H, d_h)
        V = V.view(B, L, self.num_heads, self.head_dim)

        # 3. Temperature-scaled softmax attention
        # Expand attn_scores for multi-head: (B, C, H, L)
        tau = torch.exp(self.log_tau).view(1, 1, self.num_heads, 1)
        attn = attn_scores.unsqueeze(2).expand(B, C, self.num_heads, L)
        attn = attn / tau

        # Mask padding positions
        mask_expanded = mask.unsqueeze(1)  # (B, 1, 1, L)
        attn = attn.masked_fill(mask_expanded, float('-inf'))

        # Softmax over sequence dimension
        attn_weights = F.softmax(attn, dim=-1)  # (B, C, H, L)
        attn_weights = self.dropout(attn_weights)

        # 4. Weighted aggregation: (B, C, H, L) @ (B, L, H, d_h) -> (B, C, H, d_h)
        output = torch.einsum('bchl,blhd->bchd', attn_weights, V)

        # 5. Concatenate heads: (B, C, d_v)
        output = output.reshape(B, C, self.value_dim)

        # 6. Output projection + layer norm (per-confounder)
        output = self.out_proj(output)
        output = self.layer_norm(output)

        return output
