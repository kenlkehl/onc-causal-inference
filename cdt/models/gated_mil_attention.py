# cdt/models/gated_mil_attention.py
"""Gated MIL (Multiple Instance Learning) attention modules for feature aggregation.

This module implements gated attention mechanisms from "Attention-based Deep MIL"
(Ilse et al. 2018) for aggregating sentence embeddings into document representations.

Key insight: Gated attention (tanh * sigmoid) can learn to suppress irrelevant
instances while highlighting informative ones, making it well-suited for extracting
confounder signals from long clinical documents.

Architecture:
    Sentence Embeddings (S x D)
            |
    SHARED Gated Features: h = tanh(V @ x) * sigmoid(U @ x)
            |
    K Confounder Queries attending to h
            |
    K Confounder Representations (K x D)
            |
    Task-Specific Weighting (propensity, tau, outcome)
            |
    Concatenated Output (3 x D) -> MLP -> Causal Head

References:
- Ilse et al. (2018): "Attention-based Deep Multiple Instance Learning"
- Lu et al. (2021): "Data-efficient and weakly supervised computational pathology"
"""

import logging
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class GatedMILAttention(nn.Module):
    """
    Gated MIL attention for extracting multiple confounder representations.

    Computes: a = softmax(Q @ W @ (tanh(V @ x) * sigmoid(U @ x)).T)

    The gating mechanism (tanh * sigmoid) allows the model to learn which
    sentences are informative while suppressing irrelevant ones.

    SHARED gating: All K confounder queries attend to the same gated features,
    but each query can focus on different aspects via learned query vectors.

    Args:
        input_dim: Dimension of input sentence embeddings
        hidden_dim: Hidden dimension for gated attention
        num_confounders: Number of confounder queries (K)
        dropout: Dropout rate for attention weights
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_confounders: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_confounders = num_confounders

        # SHARED gating transforms (applied to all sentences)
        # V: tanh branch - captures semantic content
        self.V = nn.Linear(input_dim, hidden_dim)
        # U: sigmoid branch - learns to gate/suppress
        self.U = nn.Linear(input_dim, hidden_dim)

        # Query projection for computing attention scores
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # K learnable confounder query vectors
        self.confounder_queries = nn.Parameter(
            torch.randn(num_confounders, hidden_dim) * 0.02
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        logger.info(f"GatedMILAttention initialized: "
                   f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_confounders={num_confounders}")

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply gated MIL attention to extract confounder representations.

        Args:
            x: Sentence embeddings of shape (S, D) where S is num sentences, D is input_dim
            return_attention: Whether to return attention weights

        Returns:
            confounders: Tensor of shape (K, D) with K confounder representations
            attention_weights: Optional tensor of shape (K, S) with attention weights
        """
        S, D = x.shape

        # 1. Compute gated features (shared across all confounders)
        # h = tanh(V @ x) * sigmoid(U @ x)
        tanh_branch = torch.tanh(self.V(x))  # (S, hidden_dim)
        sigmoid_branch = torch.sigmoid(self.U(x))  # (S, hidden_dim)
        h = tanh_branch * sigmoid_branch  # (S, hidden_dim)
        h = self.layer_norm(h)

        # 2. Project gated features for attention computation
        h_projected = self.W(h)  # (S, hidden_dim)

        # 3. Compute attention scores for each confounder query
        # scores[k, s] = dot(confounder_queries[k], h_projected[s])
        scores = torch.matmul(self.confounder_queries, h_projected.T)  # (K, S)

        # 4. Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (K, S)
        attention_weights = self.dropout(attention_weights)

        # 5. Compute weighted sum of original sentence embeddings
        # confounders[k] = sum_s(attention_weights[k, s] * x[s])
        confounders = torch.matmul(attention_weights, x)  # (K, D)

        if return_attention:
            return confounders, attention_weights
        return confounders, None

    def forward_batch(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Batched forward pass for multiple documents.

        Args:
            x: Sentence embeddings of shape (B, S, D)
            mask: Optional mask of shape (B, S) where True indicates valid sentences
            return_attention: Whether to return attention weights

        Returns:
            confounders: Tensor of shape (B, K, D)
            attention_weights: Optional tensor of shape (B, K, S)
        """
        B, S, D = x.shape

        # 1. Compute gated features
        tanh_branch = torch.tanh(self.V(x))  # (B, S, hidden_dim)
        sigmoid_branch = torch.sigmoid(self.U(x))  # (B, S, hidden_dim)
        h = tanh_branch * sigmoid_branch  # (B, S, hidden_dim)
        h = self.layer_norm(h)

        # 2. Project gated features
        h_projected = self.W(h)  # (B, S, hidden_dim)

        # 3. Compute attention scores
        # (B, S, hidden_dim) @ (hidden_dim, K) -> (B, S, K) -> transpose -> (B, K, S)
        scores = torch.einsum('bsh,kh->bks', h_projected, self.confounder_queries)

        # 4. Apply mask if provided
        if mask is not None:
            # mask: (B, S) -> expand to (B, K, S)
            mask_expanded = mask.unsqueeze(1).expand(-1, self.num_confounders, -1)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        # 5. Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # (B, K, S)
        attention_weights = self.dropout(attention_weights)

        # 6. Compute weighted sum
        confounders = torch.bmm(attention_weights, x)  # (B, K, D)

        if return_attention:
            return confounders, attention_weights
        return confounders, None


class TaskSpecificConfounderWeighting(nn.Module):
    """
    Task-specific weighting of SHARED K confounders.

    The same K confounders feed into all tasks, but each task learns
    different weights over them. This is causally coherent: confounders
    are patient characteristics that affect both treatment and outcome,
    but different tasks may weight them differently.

    For R-Learner:
        - propensity_weights: How to weight confounders for P(T=1|X)
        - tau_weights: How to weight confounders for treatment effect tau(X)
        - outcome_weights: How to weight confounders for marginal outcome E[Y|X]

    For DragonNet:
        - propensity_weights: How to weight confounders for P(T=1|X)
        - y0_weights: How to weight confounders for Y(0)
        - y1_weights: How to weight confounders for Y(1)

    Args:
        confounder_dim: Dimension of each confounder representation (D)
        num_confounders: Number of confounders (K)
        model_type: "rlearner" or "dragonnet"
    """

    def __init__(
        self,
        confounder_dim: int,
        num_confounders: int,
        model_type: str = "rlearner"
    ):
        super().__init__()

        self.confounder_dim = confounder_dim
        self.num_confounders = num_confounders
        self.model_type = model_type

        # Learnable weight logits for each task
        # Using logits -> softmax to get normalized weights
        self.propensity_weight_logits = nn.Parameter(torch.zeros(num_confounders))

        if model_type == "rlearner":
            self.tau_weight_logits = nn.Parameter(torch.zeros(num_confounders))
            self.outcome_weight_logits = nn.Parameter(torch.zeros(num_confounders))
        else:  # dragonnet
            self.y0_weight_logits = nn.Parameter(torch.zeros(num_confounders))
            self.y1_weight_logits = nn.Parameter(torch.zeros(num_confounders))

        logger.info(f"TaskSpecificConfounderWeighting initialized: "
                   f"num_confounders={num_confounders}, model_type={model_type}")

    def forward(
        self,
        confounders: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute task-specific weighted sums of confounders.

        Args:
            confounders: Tensor of shape (K, D) or (B, K, D)

        Returns:
            propensity_repr: Shape (D,) or (B, D)
            task2_repr: Shape (D,) or (B, D) - tau for rlearner, y0 for dragonnet
            task3_repr: Shape (D,) or (B, D) - outcome for rlearner, y1 for dragonnet
        """
        # Handle both single sample and batch
        is_batch = confounders.dim() == 3

        # Compute normalized weights via softmax
        prop_weights = F.softmax(self.propensity_weight_logits, dim=0)  # (K,)

        if self.model_type == "rlearner":
            tau_weights = F.softmax(self.tau_weight_logits, dim=0)
            outcome_weights = F.softmax(self.outcome_weight_logits, dim=0)
        else:
            y0_weights = F.softmax(self.y0_weight_logits, dim=0)
            y1_weights = F.softmax(self.y1_weight_logits, dim=0)

        if is_batch:
            # confounders: (B, K, D)
            # weights: (K,) -> expand for batch matmul
            propensity_repr = torch.einsum('bkd,k->bd', confounders, prop_weights)

            if self.model_type == "rlearner":
                tau_repr = torch.einsum('bkd,k->bd', confounders, tau_weights)
                outcome_repr = torch.einsum('bkd,k->bd', confounders, outcome_weights)
                return propensity_repr, tau_repr, outcome_repr
            else:
                y0_repr = torch.einsum('bkd,k->bd', confounders, y0_weights)
                y1_repr = torch.einsum('bkd,k->bd', confounders, y1_weights)
                return propensity_repr, y0_repr, y1_repr
        else:
            # confounders: (K, D)
            propensity_repr = torch.einsum('kd,k->d', confounders, prop_weights)

            if self.model_type == "rlearner":
                tau_repr = torch.einsum('kd,k->d', confounders, tau_weights)
                outcome_repr = torch.einsum('kd,k->d', confounders, outcome_weights)
                return propensity_repr, tau_repr, outcome_repr
            else:
                y0_repr = torch.einsum('kd,k->d', confounders, y0_weights)
                y1_repr = torch.einsum('kd,k->d', confounders, y1_weights)
                return propensity_repr, y0_repr, y1_repr

    def get_weights(self) -> dict:
        """Get the normalized weights for each task (for interpretability)."""
        weights = {
            'propensity': F.softmax(self.propensity_weight_logits, dim=0).detach().cpu().tolist()
        }

        if self.model_type == "rlearner":
            weights['tau'] = F.softmax(self.tau_weight_logits, dim=0).detach().cpu().tolist()
            weights['outcome'] = F.softmax(self.outcome_weight_logits, dim=0).detach().cpu().tolist()
        else:
            weights['y0'] = F.softmax(self.y0_weight_logits, dim=0).detach().cpu().tolist()
            weights['y1'] = F.softmax(self.y1_weight_logits, dim=0).detach().cpu().tolist()

        return weights


class TokenLevelGatedPooling(nn.Module):
    """
    Token-level gated attention for creating confounder-specific sentence representations.

    Instead of using [CLS] tokens from BERT, this module applies gated attention
    over tokens within each sentence. Each confounder query attends to different
    tokens, producing K distinct sentence representations per sentence.

    This preserves fine-grained distinctions that [CLS] embeddings may lose,
    such as "ECOG PS 0" vs "ECOG PS 2" or "no metastatic disease" vs "metastatic disease".

    Architecture:
        Token Embeddings (L x D)
                |
        Gated Features: g = tanh(V @ tokens) * sigmoid(U @ tokens)
                |
        For each confounder k:
            b_k = softmax(q_k @ W @ g.T)  # (L,) token weights
            r_k = sum(b_k * tokens)       # (D,) confounder-specific representation
                |
        Output: K confounder-specific representations (K x D)

    Args:
        input_dim: Dimension of token embeddings (D)
        hidden_dim: Hidden dimension for gated attention
        num_confounders: Number of confounder queries (K)
        dropout: Dropout rate for attention weights
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_confounders: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_confounders = num_confounders

        # Gating transforms
        # V: tanh branch - captures semantic content
        self.V = nn.Linear(input_dim, hidden_dim)
        # U: sigmoid branch - learns to gate/suppress
        self.U = nn.Linear(input_dim, hidden_dim)

        # Query projection for attention scores
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # K learnable confounder query vectors for token attention
        self.confounder_queries = nn.Parameter(
            torch.randn(num_confounders, hidden_dim) * 0.02
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        logger.info(f"TokenLevelGatedPooling initialized: "
                   f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_confounders={num_confounders}")

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply token-level gated attention to create K confounder-specific representations.

        Args:
            tokens: Token embeddings of shape (L, D) where L is num tokens
            attention_mask: Optional mask of shape (L,) where True indicates valid tokens
            return_attention: Whether to return attention weights

        Returns:
            representations: Tensor of shape (K, D) with K confounder-specific representations
            attention_weights: Optional tensor of shape (K, L) with token attention weights
        """
        L, D = tokens.shape

        # 1. Compute gated features
        # g = tanh(V @ tokens) * sigmoid(U @ tokens)
        tanh_branch = torch.tanh(self.V(tokens))  # (L, hidden_dim)
        sigmoid_branch = torch.sigmoid(self.U(tokens))  # (L, hidden_dim)
        g = tanh_branch * sigmoid_branch  # (L, hidden_dim)
        g = self.layer_norm(g)

        # 2. Project gated features
        g_projected = self.W(g)  # (L, hidden_dim)

        # 3. Compute attention scores for each confounder query
        # scores[k, l] = dot(confounder_queries[k], g_projected[l])
        scores = torch.matmul(self.confounder_queries, g_projected.T)  # (K, L)

        # 4. Apply mask if provided
        if attention_mask is not None:
            # Mask out padding tokens
            scores = scores.masked_fill(~attention_mask.unsqueeze(0), float('-inf'))

        # 5. Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (K, L)
        attention_weights = self.dropout(attention_weights)

        # 6. Compute weighted sum of original token embeddings
        # representations[k] = sum_l(attention_weights[k, l] * tokens[l])
        representations = torch.matmul(attention_weights, tokens)  # (K, D)

        if return_attention:
            return representations, attention_weights
        return representations, None

    def forward_batch(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Batched forward pass for multiple sentences.

        Args:
            tokens: Token embeddings of shape (S, L, D) where S is num sentences
            attention_mask: Optional mask of shape (S, L) where True indicates valid tokens
            return_attention: Whether to return attention weights

        Returns:
            representations: Tensor of shape (S, K, D) with K representations per sentence
            attention_weights: Optional tensor of shape (S, K, L)
        """
        S, L, D = tokens.shape

        # 1. Compute gated features
        tanh_branch = torch.tanh(self.V(tokens))  # (S, L, hidden_dim)
        sigmoid_branch = torch.sigmoid(self.U(tokens))  # (S, L, hidden_dim)
        g = tanh_branch * sigmoid_branch  # (S, L, hidden_dim)
        g = self.layer_norm(g)

        # 2. Project gated features
        g_projected = self.W(g)  # (S, L, hidden_dim)

        # 3. Compute attention scores
        # (S, L, hidden_dim) @ (hidden_dim, K) -> (S, L, K) -> transpose -> (S, K, L)
        scores = torch.einsum('slh,kh->skl', g_projected, self.confounder_queries)

        # 4. Apply mask if provided
        if attention_mask is not None:
            # attention_mask: (S, L) -> expand to (S, K, L)
            mask_expanded = attention_mask.unsqueeze(1).expand(-1, self.num_confounders, -1)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        # 5. Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # (S, K, L)
        attention_weights = self.dropout(attention_weights)

        # 6. Compute weighted sum
        # (S, K, L) @ (S, L, D) -> (S, K, D)
        representations = torch.bmm(attention_weights, tokens)

        if return_attention:
            return representations, attention_weights
        return representations, None
