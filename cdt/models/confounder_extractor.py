# cdt/models/confounder_extractor.py
"""Confounder-aware feature extractor using cross-attention with sparse attention.

This module implements a Perceiver-style architecture for extracting confounder
representations from long clinical text. The key insight is that confounders are
often mentioned in specific sentences, so the model should learn to focus
attention on those sentences rather than spreading attention across the entire document.

Architecture:
1. Split text into sentences (chunks)
2. Encode each sentence with a sentence transformer
3. Use learnable latent vectors (confounders) to cross-attend to sentence embeddings
4. Sparse attention (entmax) forces each latent to focus on few sentences
5. Iterative refinement allows latents to progressively focus on relevant content
6. Output is concatenation of refined latent vectors

Key features:
- Sparse attention via entmax (forces exact zeros on irrelevant sentences)
- Iterative cross-attention (Perceiver-IO style refinement)
- Optional self-attention between latents (allows confounders to share information)
- Explicit confounder initialization from clinical concept phrases

References:
- Jaegle et al. (2021): "Perceiver: General Perception with Iterative Attention"
- Nie & Wager (2021): "Quasi-oracle estimation of heterogeneous treatment effects"
"""

import logging
import re
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_attention import sparse_softmax, SparseCrossAttention


logger = logging.getLogger(__name__)


def split_into_sentences(text: str, max_sentences: int = 100) -> List[str]:
    """
    Split text into sentences using simple regex-based splitting.

    Args:
        text: Input text
        max_sentences: Maximum number of sentences to return

    Returns:
        List of sentence strings
    """
    # Simple sentence splitting on period, exclamation, question mark
    # followed by space and capital letter, or end of string
    pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    sentences = re.split(pattern, text.strip())

    # Filter empty sentences and limit count
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences[:max_sentences]


class SentenceEncoder(nn.Module):
    """
    Encode sentences using a sentence transformer model.

    This is a wrapper around sentence-transformers that handles batching
    and device management.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
        max_length: int = 256,
        freeze: bool = True
    ):
        """
        Initialize sentence encoder.

        Args:
            model_name: HuggingFace model name or path
            device: Device to place model on
            max_length: Maximum tokens per sentence
            freeze: Whether to freeze encoder weights
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self._device = device or torch.device('cpu')
        self._freeze = freeze

        # Lazy loading of sentence-transformers
        self._encoder = None
        self._output_dim = None

    def _ensure_loaded(self):
        """Lazily load the sentence transformer model."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for ConfounderExtractor. "
                    "Install with: pip install sentence-transformers"
                )

            self._encoder = SentenceTransformer(self.model_name, device=str(self._device))
            self._output_dim = self._encoder.get_sentence_embedding_dimension()

            if self._freeze:
                for param in self._encoder.parameters():
                    param.requires_grad = False

            logger.info(f"SentenceEncoder loaded: {self.model_name}, dim={self._output_dim}")

    @property
    def output_dim(self) -> int:
        """Get output embedding dimension."""
        self._ensure_loaded()
        return self._output_dim

    def forward(self, sentences: List[str]) -> torch.Tensor:
        """
        Encode a list of sentences.

        Args:
            sentences: List of sentence strings

        Returns:
            Tensor of shape (num_sentences, embedding_dim)
        """
        self._ensure_loaded()

        if not sentences:
            return torch.zeros(0, self._output_dim, device=self._device)

        # Encode with sentence transformer
        with torch.set_grad_enabled(not self._freeze):
            embeddings = self._encoder.encode(
                sentences,
                convert_to_tensor=True,
                device=str(self._device),
                show_progress_bar=False
            )

        return embeddings.to(self._device)

    def to(self, device):
        """Move encoder to device."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        if self._encoder is not None:
            self._encoder = self._encoder.to(str(self._device))
        return super().to(device)


class IterativeCrossAttention(nn.Module):
    """
    Stack of cross-attention layers with optional self-attention for iterative refinement.

    This implements Perceiver-IO style iterative attention where latent vectors
    repeatedly attend to the input sequence, progressively refining their representations.
    """

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        num_heads: int = 4,
        num_iterations: int = 2,
        use_self_attention: bool = True,
        sparse_method: str = "entmax",
        sparse_alpha: float = 1.5,
        top_k: int = 5,
        dropout: float = 0.1
    ):
        """
        Initialize iterative cross-attention.

        Args:
            latent_dim: Dimension of latent vectors
            input_dim: Dimension of input (sentence) embeddings
            num_heads: Number of attention heads
            num_iterations: Number of refinement passes
            use_self_attention: Whether latents attend to each other between iterations
            sparse_method: Attention sparsity method ("entmax", "topk", "softmax")
            sparse_alpha: Alpha for entmax (1.5 = entmax15, 2.0 = sparsemax)
            top_k: K for top-k attention
            dropout: Dropout rate
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.use_self_attention = use_self_attention

        # Cross-attention layers (latents attend to input)
        self.cross_attn_layers = nn.ModuleList([
            SparseCrossAttention(
                query_dim=latent_dim,
                key_dim=input_dim,
                value_dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout,
                sparse_method=sparse_method,
                sparse_alpha=sparse_alpha,
                top_k=top_k
            )
            for _ in range(num_iterations)
        ])

        # Optional self-attention layers (latents attend to each other)
        if use_self_attention:
            self.self_attn_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_iterations)
            ])
            self.self_attn_norms = nn.ModuleList([
                nn.LayerNorm(latent_dim)
                for _ in range(num_iterations)
            ])
        else:
            self.self_attn_layers = None
            self.self_attn_norms = None

        # Feed-forward layers after each iteration
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim * 4, latent_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_iterations)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim)
            for _ in range(num_iterations)
        ])

    def forward(
        self,
        latents: torch.Tensor,
        input_embeddings: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Iteratively refine latent representations.

        Args:
            latents: Initial latent vectors (B, K, D) or (K, D) for shared initialization
            input_embeddings: Input sequence embeddings (B, L, input_dim)
            input_mask: Mask for input positions (B, L) where True = ignore
            return_attention: Whether to return attention weights from each iteration

        Returns:
            refined_latents: Refined latent vectors (B, K, D)
            attention_weights: Optional list of attention weights per iteration
        """
        B = input_embeddings.size(0)

        # Expand shared latents to batch dimension if needed
        if latents.dim() == 2:
            latents = latents.unsqueeze(0).expand(B, -1, -1)

        attention_list = [] if return_attention else None

        for i in range(self.num_iterations):
            # Cross-attention: latents attend to input
            cross_output, attn_weights = self.cross_attn_layers[i](
                queries=latents,
                keys=input_embeddings,
                values=input_embeddings,
                mask=input_mask,
                return_attention=return_attention
            )
            latents = latents + cross_output  # Residual connection

            if return_attention and attn_weights is not None:
                attention_list.append(attn_weights)

            # Self-attention: latents attend to each other
            if self.self_attn_layers is not None:
                self_output, _ = self.self_attn_layers[i](
                    latents, latents, latents
                )
                latents = self.self_attn_norms[i](latents + self_output)

            # Feed-forward
            ffn_output = self.ffn_layers[i](latents)
            latents = self.ffn_norms[i](latents + ffn_output)

        return latents, attention_list


class ConfounderExtractor(nn.Module):
    """
    Extract confounder representations from clinical text using cross-attention.

    This is the main feature extractor class that combines:
    1. Sentence-level encoding (SentenceTransformer)
    2. Learnable latent confounder vectors
    3. Sparse cross-attention (entmax/top-k)
    4. Iterative refinement (Perceiver-style)

    The output is a fixed-size representation regardless of input text length,
    making it suitable for downstream causal inference heads (DragonNet, R-Learner).
    """

    def __init__(
        self,
        # Core architecture
        num_latent_confounders: int = 4,
        explicit_confounder_texts: Optional[List[str]] = None,
        value_dim: int = 128,
        # Sentence encoder
        sentence_transformer_model: str = "all-MiniLM-L6-v2",
        freeze_sentence_encoder: bool = True,
        max_sentences: int = 100,
        # Cross-attention
        num_attention_heads: int = 4,
        num_iterations: int = 2,
        use_self_attention: bool = True,
        # Sparse attention
        sparse_attention: bool = True,
        sparse_alpha: float = 1.5,
        sparse_method: str = "entmax",  # "entmax", "topk", "softmax"
        top_k: int = 5,
        # Regularization
        dropout: float = 0.1,
        # Device
        device: Optional[torch.device] = None
    ):
        """
        Initialize confounder extractor.

        Args:
            num_latent_confounders: Number of learnable latent confounder vectors
            explicit_confounder_texts: Optional list of explicit confounder concept texts
                (e.g., ["metastatic sites", "performance status"]). These are encoded
                and used as additional confounder queries.
            value_dim: Output dimension per confounder (and latent dimension)
            sentence_transformer_model: HuggingFace model for sentence encoding
            freeze_sentence_encoder: Whether to freeze sentence encoder weights
            max_sentences: Maximum sentences to process per document
            num_attention_heads: Number of attention heads in cross-attention
            num_iterations: Number of iterative refinement passes
            use_self_attention: Whether latents attend to each other
            sparse_attention: Whether to use sparse attention (entmax/top-k)
            sparse_alpha: Alpha for entmax (1.5=entmax15, 2.0=sparsemax)
            sparse_method: Sparsity method ("entmax", "topk", "softmax")
            top_k: K for top-k attention method
            dropout: Dropout rate
            device: Device to place model on
        """
        super().__init__()

        self._device = device or torch.device('cpu')
        self.max_sentences = max_sentences
        self.num_latent_confounders = num_latent_confounders
        self.value_dim = value_dim

        # Sentence encoder
        self.sentence_encoder = SentenceEncoder(
            model_name=sentence_transformer_model,
            device=self._device,
            freeze=freeze_sentence_encoder
        )

        # Get sentence embedding dimension (triggers lazy loading)
        # We'll do this lazily to avoid loading model at init time
        self._sentence_dim = None

        # Explicit confounder texts (will be encoded lazily)
        self._explicit_confounder_texts = explicit_confounder_texts or []
        self._explicit_embeddings = None

        # Learnable latent confounders
        # Initialized as learnable parameters
        self.latent_confounders = nn.Parameter(
            torch.randn(num_latent_confounders, value_dim) * 0.1
        )

        # Total number of confounders = latent + explicit
        self.num_explicit_confounders = len(self._explicit_confounder_texts)
        self.total_confounders = num_latent_confounders + self.num_explicit_confounders

        # Input projection (from sentence embedding dim to value_dim)
        # Created lazily when we know sentence_dim
        self._input_projection = None

        # Iterative cross-attention
        self.cross_attention = IterativeCrossAttention(
            latent_dim=value_dim,
            input_dim=value_dim,  # After projection
            num_heads=num_attention_heads,
            num_iterations=num_iterations,
            use_self_attention=use_self_attention,
            sparse_method=sparse_method if sparse_attention else "softmax",
            sparse_alpha=sparse_alpha,
            top_k=top_k,
            dropout=dropout
        )

        # Output projection MLP
        output_dim = value_dim * self.total_confounders
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, value_dim),
            nn.LayerNorm(value_dim)
        )

        # Store output dimension
        self._output_dim = value_dim

        logger.info(f"ConfounderExtractor initialized:")
        logger.info(f"  Latent confounders: {num_latent_confounders}")
        logger.info(f"  Explicit confounders: {self.num_explicit_confounders}")
        logger.info(f"  Total confounders: {self.total_confounders}")
        logger.info(f"  Value dim: {value_dim}")
        logger.info(f"  Output dim: {self._output_dim}")
        logger.info(f"  Iterations: {num_iterations}")
        logger.info(f"  Sparse attention: {sparse_attention} ({sparse_method}, alpha={sparse_alpha})")

    def _ensure_initialized(self):
        """Lazily initialize components that depend on sentence encoder dimension."""
        if self._sentence_dim is not None:
            return

        self._sentence_dim = self.sentence_encoder.output_dim

        # Create input projection
        self._input_projection = nn.Linear(self._sentence_dim, self.value_dim).to(self._device)

        # Encode explicit confounder texts
        if self._explicit_confounder_texts:
            with torch.no_grad():
                explicit_emb = self.sentence_encoder(self._explicit_confounder_texts)
                explicit_emb = self._input_projection(explicit_emb)
            self.register_buffer('explicit_confounders', explicit_emb)
        else:
            self.register_buffer('explicit_confounders', torch.zeros(0, self.value_dim))

        logger.info(f"  Sentence encoder dim: {self._sentence_dim}")

    @property
    def output_dim(self) -> int:
        """Get output embedding dimension."""
        return self._output_dim

    def _encode_sentences(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split texts into sentences and encode them.

        Args:
            texts: List of document texts

        Returns:
            sentence_embeddings: Padded tensor (B, max_sentences, dim)
            mask: Padding mask (B, max_sentences) where True = padding
        """
        self._ensure_initialized()

        batch_size = len(texts)
        all_sentences = []
        sentence_counts = []

        for text in texts:
            sentences = split_into_sentences(text, self.max_sentences)
            if not sentences:
                sentences = [text[:500]]  # Fallback: use truncated text
            all_sentences.append(sentences)
            sentence_counts.append(len(sentences))

        # Flatten for batch encoding
        flat_sentences = [s for doc_sentences in all_sentences for s in doc_sentences]

        if not flat_sentences:
            # Empty batch
            dummy = torch.zeros(batch_size, 1, self.value_dim, device=self._device)
            mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=self._device)
            return dummy, mask

        # Encode all sentences
        flat_embeddings = self.sentence_encoder(flat_sentences)
        # Clone to avoid inference mode issues with autograd
        flat_embeddings = flat_embeddings.clone()
        flat_embeddings = self._input_projection(flat_embeddings)

        # Reshape back to batch
        max_sentences = max(sentence_counts)
        padded_embeddings = torch.zeros(
            batch_size, max_sentences, self.value_dim,
            device=self._device
        )
        mask = torch.ones(batch_size, max_sentences, dtype=torch.bool, device=self._device)

        idx = 0
        for i, count in enumerate(sentence_counts):
            padded_embeddings[i, :count] = flat_embeddings[idx:idx + count]
            mask[i, :count] = False  # Not padding
            idx += count

        return padded_embeddings, mask

    def forward(
        self,
        texts: List[str],
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Extract confounder representations from texts.

        Args:
            texts: List of document texts
            return_attention: Whether to return attention weights (for visualization)

        Returns:
            features: Feature tensor (batch, output_dim)
            attention_weights: Optional attention weights if return_attention=True
        """
        self._ensure_initialized()
        batch_size = len(texts)

        # Encode sentences
        sentence_embeddings, mask = self._encode_sentences(texts)

        # Combine latent and explicit confounders
        latents = self.latent_confounders  # (K_latent, D)
        if self.explicit_confounders.size(0) > 0:
            all_confounders = torch.cat([latents, self.explicit_confounders], dim=0)
        else:
            all_confounders = latents

        # Expand to batch dimension
        all_confounders = all_confounders.unsqueeze(0).expand(batch_size, -1, -1)

        # Iterative cross-attention
        refined_confounders, attention_weights = self.cross_attention(
            latents=all_confounders,
            input_embeddings=sentence_embeddings,
            input_mask=mask,
            return_attention=return_attention
        )

        # Flatten confounders: (B, C, D) -> (B, C*D)
        flat_confounders = refined_confounders.reshape(batch_size, -1)

        # Project to output dimension
        features = self.output_projection(flat_confounders)

        if return_attention:
            return features, attention_weights
        return features

    def get_attention_weights(
        self,
        texts: List[str]
    ) -> Dict[str, Any]:
        """
        Get attention weights for visualization and interpretation.

        Args:
            texts: List of document texts

        Returns:
            Dictionary with:
                - sentences: List of sentence lists per document
                - attention_weights: Attention weights per iteration
                - confounder_names: Names of confounders (latent_0, ..., explicit_0, ...)
        """
        self._ensure_initialized()

        # Get sentences
        all_sentences = [split_into_sentences(t, self.max_sentences) for t in texts]

        # Forward with attention - use no_grad to avoid inference mode issues
        with torch.no_grad():
            _, attention_weights = self.forward(texts, return_attention=True)

        # Build confounder names
        confounder_names = [f"latent_{i}" for i in range(self.num_latent_confounders)]
        confounder_names += [f"explicit_{i}_{t[:20]}" for i, t in enumerate(self._explicit_confounder_texts)]

        return {
            'sentences': all_sentences,
            'attention_weights': attention_weights,
            'confounder_names': confounder_names
        }

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of what each confounder attends to.

        Args:
            texts: List of document texts
            top_k: Number of top-attended sentences to show per confounder

        Returns:
            List of dictionaries per document, each containing per-confounder interpretations
        """
        result = self.get_attention_weights(texts)
        sentences = result['sentences']
        attention_weights = result['attention_weights']
        confounder_names = result['confounder_names']

        if not attention_weights:
            return [{} for _ in texts]

        # Use last iteration's attention
        final_attn = attention_weights[-1]  # (B, H, C, L) - averaged over heads
        # Average over heads
        final_attn = final_attn.mean(dim=1)  # (B, C, L)

        interpretations = []
        for doc_idx in range(len(texts)):
            doc_sentences = sentences[doc_idx]
            doc_attn = final_attn[doc_idx]  # (C, L)

            doc_interp = {}
            for conf_idx, conf_name in enumerate(confounder_names):
                conf_attn = doc_attn[conf_idx, :len(doc_sentences)]  # (L,)

                # Get top-k attended sentences
                top_vals, top_indices = torch.topk(conf_attn, min(top_k, len(doc_sentences)))

                top_sentences = []
                for val, idx in zip(top_vals.tolist(), top_indices.tolist()):
                    if val > 0.001:  # Only include non-trivial attention
                        top_sentences.append({
                            'sentence': doc_sentences[idx],
                            'attention': val
                        })

                doc_interp[conf_name] = top_sentences

            interpretations.append(doc_interp)

        return interpretations

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.sentence_encoder = self.sentence_encoder.to(device)
        if self._input_projection is not None:
            self._input_projection = self._input_projection.to(device)
        return super().to(device)

    def get_state(self) -> Dict[str, Any]:
        """Get extractor state for checkpoint saving."""
        return {
            'num_latent_confounders': self.num_latent_confounders,
            'explicit_confounder_texts': self._explicit_confounder_texts,
            'value_dim': self.value_dim,
            'max_sentences': self.max_sentences,
            'output_dim': self._output_dim,
        }

    # Compatibility with CausalCNNText interface
    def fit_tokenizer(self, texts: List[str]) -> 'ConfounderExtractor':
        """No-op for compatibility. ConfounderExtractor uses pretrained sentence encoder."""
        # Trigger lazy initialization
        self._ensure_initialized()
        return self
