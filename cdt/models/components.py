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


class ConfounderAggregator(nn.Module):
    """
    Pool feature maps (B, C_total, L) -> (B, C_total * features_per_agg_output)
    
    Modes: 'max', 'lsep', 'gem', 'topk', 'attn', 'stats', 'noisyor'
    """
    
    def __init__(
        self,
        mode: str = 'attn',
        temperature: float = 0.5,
        topk: int = 3,
        per_confounder_params: bool = True
    ):
        """
        Initialize aggregator.
        
        Args:
            mode: Aggregation mode
            temperature: Temperature parameter for soft pooling
            topk: Number of top chunks for topk mode
            per_confounder_params: Use separate parameters per confounder
        """
        super().__init__()
        
        valid_modes = {'max', 'lsep', 'gem', 'topk', 'attn', 'stats', 'noisyor'}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
        
        self.mode = mode
        self.topk = topk
        self.per_confounder_params = per_confounder_params
        self._initialized = False
        
        if not per_confounder_params:
            self.tau = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.tau = None
    
    def _lazy_init(self, num_confounders: int, device: torch.device):
        """Initialize parameters based on input dimensions."""
        if self.mode == 'gem':
            self.raw_p = nn.Parameter(torch.zeros(1, device=device))
            self.features_per_conf = 1
        elif self.mode == 'stats':
            self.features_per_conf = 2
        else:
            self.features_per_conf = 1
        
        if self.per_confounder_params:
            if self.mode in {'lsep', 'noisyor', 'attn'}:  # Add modes that need it
                self.log_tau = nn.Parameter(torch.zeros(num_confounders, 1, device=device))
            else:
                self.log_tau = None  # Explicitly set to None for other modes
       
        self._initialized = True
    
    def forward(
        self,
        feature_maps: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool feature maps.
        
        Args:
            feature_maps: (batch, confounders, sequence_length)
            mask: (batch, 1, sequence_length) boolean mask
        
        Returns:
            Pooled features: (batch, confounders * features_per_conf)
        """
        batch_size, num_confounders, seq_len = feature_maps.shape
        device = feature_maps.device
        
        if not self._initialized:
            self._lazy_init(num_confounders, device)
        
        fm = feature_maps.clone()
        fill_value = float('-inf') if self.mode in {'max', 'lsep', 'topk', 'attn'} else 0.0
        fm.masked_fill_(mask, fill_value)
        
        if self.mode == 'max':
            pooled = torch.amax(fm, dim=2)
        
        elif self.mode == 'lsep':
            tau = self._get_tau(num_confounders, device)
            z = (fm / tau).logsumexp(dim=2)
            pooled = tau.squeeze(-1) * z
        
        elif self.mode == 'gem':
            p = 1.0 + F.softplus(self.raw_p.squeeze())
            x = F.relu(fm) + 1e-6
            pooled = torch.pow(torch.mean(torch.pow(x, p), dim=2), 1.0 / p)
        
        elif self.mode == 'topk':
            k = min(self.topk, seq_len)
            vals, _ = torch.topk(fm, k=k, dim=2)
            pooled = torch.mean(vals, dim=2)
        
        elif self.mode == 'attn':
            tau = self._get_tau(num_confounders, device)
            scores = fm / tau
            scores.masked_fill_(mask, float('-inf'))
            weights = torch.softmax(scores, dim=2)
            pooled = torch.sum(
                weights * feature_maps.clamp_min(-50).clamp_max(50),
                dim=2
            )
        
        elif self.mode == 'stats':
            valid = (~mask).float()
            denom = valid.sum(dim=2).clamp_min(1.0)
            mean = (fm * valid).sum(dim=2) / denom
            
            x = feature_maps.clone()
            x.masked_fill_(mask, 0.0)
            mean_for_var = (x * valid).sum(dim=2) / denom
            var = ((x - mean_for_var.unsqueeze(-1))**2 * valid).sum(dim=2) / denom
            std = torch.sqrt(var + 1e-6)
            pooled = torch.cat([mean, std], dim=1)
        
        elif self.mode == 'noisyor':
            tau = self._get_tau(num_confounders, device)
            probs = torch.sigmoid(fm / tau).clamp(1e-6, 1 - 1e-6)
            probs.masked_fill_(mask, 0.0)
            log1m = torch.log1p(-probs)
            log_prod = torch.sum(log1m, dim=2)
            any_prob = 1.0 - torch.exp(log_prod)
            pooled = torch.log(any_prob) - torch.log1p(-any_prob)
        
        return pooled
    
    def _get_tau(self, num_confounders: int, device: torch.device) -> torch.Tensor:
        """Get temperature parameter."""
        if self.per_confounder_params:
            if self.log_tau is not None:
                return torch.exp(self.log_tau).view(1, num_confounders, 1)
            else:
                return torch.tensor(0.5, device=device).view(1, 1, 1)
