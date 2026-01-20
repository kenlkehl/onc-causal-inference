# cdt/models/sparse_attention.py
"""Sparse attention utilities for confounder extraction.

This module provides alternatives to standard softmax attention that produce
sparse attention weights, forcing the model to concentrate on fewer positions.

Key utilities:
- entmax: Generalized softmax that can produce exact zeros (alpha=1.5 is entmax15)
- sparsemax: Equivalent to entmax with alpha=2.0
- top_k_attention: Hard top-k selection with straight-through gradient

References:
- Peters et al. (2019): "Sparse Sequence-to-Sequence Models"
- Correia et al. (2019): "Adaptively Sparse Transformers"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import entmax library, fall back to custom implementations if not available
try:
    from entmax import entmax15 as _entmax15
    from entmax import sparsemax as _sparsemax
    HAS_ENTMAX = True
except ImportError:
    HAS_ENTMAX = False


def _sparsemax_fallback(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax implementation for when entmax library is not available.

    Sparsemax projects onto the probability simplex, resulting in sparse outputs.
    Based on Martins & Astudillo (2016).

    Args:
        scores: Input scores (any shape)
        dim: Dimension to normalize

    Returns:
        Sparse probability distribution (same shape as input)
    """
    # Move dim to last position for easier computation
    scores = scores.transpose(dim, -1)
    original_shape = scores.shape
    scores = scores.reshape(-1, scores.size(-1))

    # Sort scores in descending order
    sorted_scores, _ = torch.sort(scores, descending=True, dim=-1)

    # Compute cumulative sums
    cumsum = torch.cumsum(sorted_scores, dim=-1)
    k = torch.arange(1, scores.size(-1) + 1, device=scores.device, dtype=scores.dtype)

    # Find the threshold
    # k_z = max{k : 1 + k * z_k > sum_{i<=k} z_i}
    check = 1 + k * sorted_scores > cumsum
    k_max = check.sum(dim=-1, keepdim=True).clamp(min=1)

    # Gather cumsum at k_max position
    cumsum_at_k = cumsum.gather(-1, k_max.long() - 1)

    # Compute threshold tau
    tau = (cumsum_at_k - 1) / k_max.float()

    # Apply threshold
    output = torch.clamp(scores - tau, min=0)

    # Reshape back
    output = output.reshape(original_shape)
    output = output.transpose(dim, -1)

    return output


def _entmax15_fallback(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Entmax-1.5 approximation when entmax library is not available.

    This is a simplified implementation using the ReLU-based formula.
    For production use, install the entmax library for exact implementation.

    Args:
        scores: Input scores (any shape)
        dim: Dimension to normalize

    Returns:
        Sparse probability distribution (same shape as input)
    """
    # Normalize scores for numerical stability
    scores = scores - scores.max(dim=dim, keepdim=True)[0]

    # Apply ReLU-squared transformation (approximates entmax-1.5)
    # This is not exact entmax but provides similar sparsity properties
    relu_scores = F.relu(scores)
    squared = relu_scores ** 2

    # Normalize
    sum_squared = squared.sum(dim=dim, keepdim=True).clamp(min=1e-12)
    output = squared / sum_squared

    return output


def sparse_softmax(
    scores: torch.Tensor,
    dim: int = -1,
    alpha: float = 1.5,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply sparse softmax (entmax) or fall back to standard softmax.

    Args:
        scores: Attention scores (any shape)
        dim: Dimension to normalize
        alpha: Sparsity parameter
            - 1.0 = standard softmax (dense)
            - 1.5 = entmax15 (moderately sparse)
            - 2.0 = sparsemax (very sparse)
        temperature: Temperature for scaling scores before normalization

    Returns:
        Attention weights (same shape as scores)
    """
    # Apply temperature scaling
    if temperature != 1.0:
        scores = scores / temperature

    # Standard softmax
    if alpha == 1.0:
        return F.softmax(scores, dim=dim)

    # Use entmax library if available
    if HAS_ENTMAX:
        if alpha == 2.0:
            return _sparsemax(scores, dim=dim)
        else:
            return _entmax15(scores, dim=dim)

    # Fallback implementations
    if alpha == 2.0:
        return _sparsemax_fallback(scores, dim=dim)
    else:
        return _entmax15_fallback(scores, dim=dim)


def top_k_attention(
    scores: torch.Tensor,
    k: int,
    dim: int = -1,
    straight_through: bool = True
) -> torch.Tensor:
    """
    Hard top-k selection with optional straight-through gradient.

    Only the top-k positions receive non-zero attention weights.
    Other positions are exactly zero.

    Args:
        scores: Attention scores (B, ..., L)
        k: Number of positions to keep
        dim: Dimension to select from
        straight_through: If True, use straight-through estimator for gradients
            (forward uses hard top-k, backward passes gradients through softmax)

    Returns:
        Sparse attention weights (same shape, zeros except top-k positions)
    """
    # Ensure k doesn't exceed sequence length
    k = min(k, scores.size(dim))

    if k <= 0:
        return torch.zeros_like(scores)

    # Get top-k indices
    _, top_indices = torch.topk(scores, k=k, dim=dim)

    # Create sparse mask
    mask = torch.zeros_like(scores)
    mask.scatter_(dim, top_indices, 1.0)

    # Apply mask and compute softmax over selected positions
    masked_scores = scores.clone()
    masked_scores[mask == 0] = float('-inf')
    sparse_weights = F.softmax(masked_scores, dim=dim)

    # Replace NaN with 0 (in case all values were -inf)
    sparse_weights = torch.nan_to_num(sparse_weights, nan=0.0)

    if straight_through:
        # Straight-through: forward uses sparse, backward uses dense softmax gradient
        dense_weights = F.softmax(scores, dim=dim)
        sparse_weights = sparse_weights + dense_weights - dense_weights.detach()

    return sparse_weights


def adaptive_top_k(
    scores: torch.Tensor,
    min_k: int = 1,
    max_k: int = 10,
    threshold: float = 0.05,
    dim: int = -1
) -> torch.Tensor:
    """
    Adaptive top-k selection based on attention weight distribution.

    Selects positions until cumulative attention exceeds a threshold,
    with constraints on minimum and maximum number of positions.

    Args:
        scores: Attention scores (B, ..., L)
        min_k: Minimum number of positions to keep
        max_k: Maximum number of positions to keep
        threshold: Cumulative attention threshold (0.05 = keep top 95%)
        dim: Dimension to select from

    Returns:
        Sparse attention weights
    """
    # Compute softmax to get probabilities
    probs = F.softmax(scores, dim=dim)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=dim, descending=True)

    # Compute cumulative sum
    cumsum = torch.cumsum(sorted_probs, dim=dim)

    # Find where cumsum exceeds (1 - threshold)
    # This gives us the number of positions needed to capture (1 - threshold) of mass
    target = 1.0 - threshold
    exceed_mask = cumsum >= target

    # Get the index where we first exceed threshold
    # Add 1 because we want to include that position
    k_adaptive = exceed_mask.int().argmax(dim=dim, keepdim=True) + 1

    # Clamp to min/max
    k_adaptive = k_adaptive.clamp(min=min_k, max=max_k)

    # Create mask for top-k positions (varying k per sample)
    # This is more complex - for simplicity, use max of k_adaptive
    k_max = k_adaptive.max().item()
    k_max = min(max(k_max, min_k), max_k)

    return top_k_attention(scores, k=k_max, dim=dim)


class SparseCrossAttention(nn.Module):
    """
    Cross-attention module with sparse attention weights.

    Supports multiple sparsity mechanisms:
    - entmax (soft sparsity)
    - top-k (hard sparsity)
    - adaptive top-k (dynamic sparsity)
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        sparse_method: str = "entmax",  # "entmax", "topk", "adaptive", "softmax"
        sparse_alpha: float = 1.5,  # For entmax
        top_k: int = 5,  # For topk methods
        adaptive_threshold: float = 0.05  # For adaptive
    ):
        """
        Initialize sparse cross-attention.

        Args:
            query_dim: Dimension of query vectors
            key_dim: Dimension of key vectors (typically same as query)
            value_dim: Dimension of value vectors
            num_heads: Number of attention heads
            dropout: Dropout rate on attention weights
            sparse_method: Sparsity method ("entmax", "topk", "adaptive", "softmax")
            sparse_alpha: Alpha parameter for entmax (1.0=softmax, 1.5=entmax15, 2.0=sparsemax)
            top_k: Number of positions for top-k methods
            adaptive_threshold: Threshold for adaptive top-k
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = value_dim // num_heads
        self.sparse_method = sparse_method
        self.sparse_alpha = sparse_alpha
        self.top_k = top_k
        self.adaptive_threshold = adaptive_threshold

        if value_dim % num_heads != 0:
            raise ValueError(f"value_dim ({value_dim}) must be divisible by num_heads ({num_heads})")

        # Projection layers
        self.W_q = nn.Linear(query_dim, value_dim, bias=False)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim, bias=False)
        self.W_o = nn.Linear(value_dim, value_dim)

        # Learnable temperature per head
        self.log_temperature = nn.Parameter(torch.zeros(num_heads))

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(value_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply sparse cross-attention.

        Args:
            queries: Query vectors (B, Q, query_dim)
            keys: Key vectors (B, K, key_dim)
            values: Value vectors (B, K, key_dim) - typically same as keys
            mask: Optional mask (B, K) where True = ignore position
            return_attention: Whether to return attention weights

        Returns:
            output: Attended output (B, Q, value_dim)
            attention_weights: Optional attention weights (B, num_heads, Q, K)
        """
        B, Q, _ = queries.shape
        K = keys.size(1)

        # Project to multi-head
        q = self.W_q(queries).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Q, D)
        k = self.W_k(keys).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, K, D)
        v = self.W_v(values).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, K, D)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, Q, K)

        # Apply temperature
        temperature = torch.exp(self.log_temperature).view(1, self.num_heads, 1, 1)
        scores = scores / temperature

        # Apply mask (set masked positions to -inf before attention)
        if mask is not None:
            # Expand mask to (B, 1, 1, K)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, float('-inf'))

        # Apply sparse attention
        if self.sparse_method == "entmax":
            attention_weights = sparse_softmax(scores, dim=-1, alpha=self.sparse_alpha)
        elif self.sparse_method == "topk":
            attention_weights = top_k_attention(scores, k=self.top_k, dim=-1)
        elif self.sparse_method == "adaptive":
            attention_weights = adaptive_top_k(scores, threshold=self.adaptive_threshold, dim=-1)
        else:  # softmax
            attention_weights = F.softmax(scores, dim=-1)

        # Handle all-masked case (replace NaN with 0)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum of values
        output = torch.matmul(attention_weights, v)  # (B, H, Q, D)

        # Reshape and project
        output = output.transpose(1, 2).reshape(B, Q, -1)  # (B, Q, value_dim)
        output = self.W_o(output)
        output = self.layer_norm(output)

        if return_attention:
            return output, attention_weights
        return output, None

    def get_attention_sparsity(self, attention_weights: torch.Tensor) -> dict:
        """
        Compute sparsity statistics for attention weights.

        Args:
            attention_weights: Attention weights (B, H, Q, K)

        Returns:
            Dictionary with sparsity metrics
        """
        # Count zeros (< 1e-6)
        zero_mask = attention_weights.abs() < 1e-6
        sparsity = zero_mask.float().mean().item()

        # Effective number of attended positions (entropy-based)
        # Higher = more diffuse attention
        log_weights = torch.log(attention_weights.clamp(min=1e-12))
        entropy = -(attention_weights * log_weights).sum(dim=-1).mean().item()
        effective_k = torch.exp(torch.tensor(entropy)).item()

        # Max attention weight (higher = more concentrated)
        max_weight = attention_weights.max(dim=-1)[0].mean().item()

        return {
            'sparsity': sparsity,
            'entropy': entropy,
            'effective_k': effective_k,
            'max_weight': max_weight
        }
