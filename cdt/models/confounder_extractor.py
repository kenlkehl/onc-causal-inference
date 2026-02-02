# cdt/models/confounder_extractor.py
"""Confounder-aware feature extractor using cross-attention with sparse attention.

This module implements a Perceiver-style architecture for extracting confounder
representations from long clinical text. The key insight is that confounders are
often mentioned in specific chunks, so the model should learn to focus
attention on those chunks rather than spreading attention across the entire document.

Architecture:
1. Split text into overlapping token chunks
2. Encode each chunk with a sentence transformer or BERT
3. Use learnable latent vectors (confounders) to cross-attend to chunk embeddings
4. Sparse attention (entmax) forces each latent to focus on few chunks
5. Iterative refinement allows latents to progressively focus on relevant content
6. Output is concatenation of refined latent vectors

Key features:
- Sparse attention via entmax (forces exact zeros on irrelevant chunks)
- Iterative cross-attention (Perceiver-IO style refinement)
- Optional self-attention between latents (allows confounders to share information)
- Explicit confounder initialization from clinical concept phrases

References:
- Jaegle et al. (2021): "Perceiver: General Perception with Iterative Attention"
- Nie & Wager (2021): "Quasi-oracle estimation of heterogeneous treatment effects"
"""

import logging
import re
import warnings
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_attention import sparse_softmax, SparseCrossAttention, top_k_attention
from .chunking import split_into_chunks_hf, split_into_chunks_vocab
from .numeric_features import NumericFeatureVector


logger = logging.getLogger(__name__)


def split_into_sentences(text: str, max_sentences: int = 100) -> List[str]:
    """
    DEPRECATED: Use split_into_chunks_hf from chunking.py instead.

    Split text into sentences using simple regex-based splitting.
    Kept for backward compatibility only.

    Args:
        text: Input text
        max_sentences: Maximum number of sentences to return

    Returns:
        List of sentence strings
    """
    warnings.warn(
        "split_into_sentences is deprecated. Use split_into_chunks_hf from "
        "cdt.models.chunking instead for token-based chunking.",
        DeprecationWarning,
        stacklevel=2
    )
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
        device: Optional[torch.device] = None,
        # Numeric features
        numeric_features_enabled: bool = False,
        numeric_embedding_dim: int = 32,
        numeric_magnitude_bins: int = 8,
        numeric_type_categories: int = 10
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
            numeric_features_enabled: Whether to extract and merge numeric features from text
            numeric_embedding_dim: Output dimension of numeric feature vector
            numeric_magnitude_bins: Number of magnitude bins for numeric encoding
            numeric_type_categories: Number of numeric type categories
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

        # Numeric feature vector
        self.numeric_features_enabled = numeric_features_enabled
        self.numeric_feature_vector = None
        if numeric_features_enabled:
            self.numeric_feature_vector = NumericFeatureVector(
                num_magnitude_bins=numeric_magnitude_bins,
                num_type_categories=numeric_type_categories,
                output_dim=numeric_embedding_dim
            )
            # Merge layer: C*D + numeric_dim -> C*D (before output projection)
            output_dim = value_dim * self.total_confounders
            self._numeric_merge = nn.Sequential(
                nn.Linear(output_dim + numeric_embedding_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )

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

        # Add numeric features before output projection
        if self.numeric_features_enabled and self.numeric_feature_vector is not None:
            numeric_feats = self.numeric_feature_vector(texts)
            flat_confounders = self._numeric_merge(
                torch.cat([flat_confounders, numeric_feats], dim=1)
            )

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

    # Compatibility with CausalText interface
    def fit_tokenizer(self, texts: List[str]) -> 'ConfounderExtractor':
        """No-op for compatibility. ConfounderExtractor uses pretrained sentence encoder."""
        # Trigger lazy initialization
        self._ensure_initialized()
        return self


class TokenEncoder(nn.Module):
    """
    Encode sentences with BERT, keeping token-level embeddings.

    Unlike SentenceEncoder which only returns mean-pooled embeddings,
    this encoder returns both token embeddings and sentence embeddings.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: Optional[torch.device] = None,
        max_length: int = 128,
        freeze: bool = True
    ):
        """
        Initialize token encoder.

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

        # Lazy loading
        self._encoder = None
        self._tokenizer = None
        self._output_dim = None

    def _ensure_loaded(self):
        """Lazily load the transformer model."""
        if self._encoder is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError:
                raise ImportError(
                    "transformers is required for HierarchicalConfounderExtractor. "
                    "Install with: pip install transformers"
                )

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._encoder = AutoModel.from_pretrained(self.model_name)
            self._encoder = self._encoder.to(self._device)
            self._output_dim = self._encoder.config.hidden_size

            if self._freeze:
                for param in self._encoder.parameters():
                    param.requires_grad = False

            logger.info(f"TokenEncoder loaded: {self.model_name}, dim={self._output_dim}")

    @property
    def output_dim(self) -> int:
        """Get output embedding dimension."""
        self._ensure_loaded()
        return self._output_dim

    def encode_sentence(
        self,
        sentence: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single sentence, returning both token and sentence embeddings.

        Args:
            sentence: Input sentence string

        Returns:
            token_embeddings: (L, D) tensor of token embeddings
            sentence_embedding: (D,) tensor of mean-pooled sentence embedding
        """
        self._ensure_loaded()

        # Tokenize
        inputs = self._tokenizer(
            sentence,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Encode
        with torch.set_grad_enabled(not self._freeze):
            outputs = self._encoder(**inputs)

        # Get token embeddings (exclude CLS and SEP tokens)
        token_embs = outputs.last_hidden_state[0]  # (L, D)

        # Mean pool for sentence embedding (including CLS/SEP for now)
        # Use attention mask for proper masking
        attention_mask = inputs['attention_mask'][0]  # (L,)
        masked_embs = token_embs * attention_mask.unsqueeze(-1)
        sentence_emb = masked_embs.sum(dim=0) / attention_mask.sum()

        return token_embs, sentence_emb

    def encode_sentences_batch(
        self,
        sentences: List[str]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Encode multiple sentences, returning token embeddings per sentence
        and batched sentence embeddings.

        Args:
            sentences: List of sentence strings

        Returns:
            token_embeddings_list: List of (L_i, D) tensors for each sentence
            sentence_embeddings: (S, D) tensor of sentence embeddings
        """
        self._ensure_loaded()

        if not sentences:
            return [], torch.zeros(0, self._output_dim, device=self._device)

        token_embeddings_list = []
        sentence_embeddings = []

        for sentence in sentences:
            token_embs, sentence_emb = self.encode_sentence(sentence)
            token_embeddings_list.append(token_embs)
            sentence_embeddings.append(sentence_emb)

        return token_embeddings_list, torch.stack(sentence_embeddings)

    def to(self, device):
        """Move encoder to device."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        if self._encoder is not None:
            self._encoder = self._encoder.to(self._device)
        return super().to(device)


class HierarchicalConfounderExtractor(nn.Module):
    """
    Hierarchical confounder extractor with both sentence-level and token-level attention.

    This architecture preserves token-level signal while using sentence-level attention
    as a focusing mechanism. Key insight: sentence embeddings may lose fine-grained
    signal (e.g., "ECOG PS is 0" vs "ECOG PS is 2" may have similar sentence embeddings).

    Architecture:
    1. Split text into sentences
    2. Encode EACH sentence with BERT → S × (L_s tokens × 768)
    3. Mean-pool each sentence → Sentence Embeddings (S × 768)
    4. Sentence-Level Sparse Attention (entmax) → Sentence Weights (K × S)
    5. Token-Level Cross-Attention (within each sentence, gated by sentence weights)
    6. K Confounder Representations → Causal Head

    The key advantage is that sentence-level sparse attention reduces the search space,
    while token-level attention preserves fine-grained distinctions.
    """

    def __init__(
        self,
        # Core architecture
        num_latent_confounders: int = 4,
        explicit_confounder_texts: Optional[List[str]] = None,
        value_dim: int = 128,
        # Token encoder
        token_encoder_model: str = "distilbert-base-uncased",
        freeze_token_encoder: bool = True,
        max_sentences: int = 100,
        max_sentence_tokens: int = 128,
        # Cross-attention
        num_attention_heads: int = 4,
        # Sparse attention
        sparse_attention: bool = True,
        sparse_alpha: float = 1.5,
        sparse_method: str = "entmax",  # "entmax", "topk", "softmax"
        top_k: int = 5,
        # Regularization
        dropout: float = 0.1,
        # Model type for task-specific aggregation
        model_type: str = "dragonnet",
        # Device
        device: Optional[torch.device] = None,
        # Numeric features
        numeric_features_enabled: bool = False,
        numeric_embedding_dim: int = 32,
        numeric_magnitude_bins: int = 8,
        numeric_type_categories: int = 10
    ):
        """
        Initialize hierarchical confounder extractor.

        Args:
            num_latent_confounders: Number of learnable latent confounder vectors
            explicit_confounder_texts: Optional list of explicit confounder concept texts
            value_dim: Output dimension per confounder
            token_encoder_model: HuggingFace model for token encoding
            freeze_token_encoder: Whether to freeze token encoder weights
            max_sentences: Maximum sentences to process per document
            max_sentence_tokens: Maximum tokens per sentence
            num_attention_heads: Number of attention heads for token cross-attention
            sparse_attention: Whether to use sparse attention (entmax/top-k)
            sparse_alpha: Alpha for entmax (1.5=entmax15, 2.0=sparsemax)
            sparse_method: Sparsity method ("entmax", "topk", "softmax")
            top_k: K for top-k attention method
            dropout: Dropout rate
            model_type: Architecture type ("dragonnet", "uplift", or "rlearner") for
                       task-specific aggregation. DragonNet uses propensity/Y0/Y1 queries,
                       R-Learner uses propensity/outcome/tau queries.
            device: Device to place model on
            numeric_features_enabled: Whether to extract and merge numeric features from text
            numeric_embedding_dim: Output dimension of numeric feature vector
            numeric_magnitude_bins: Number of magnitude bins for numeric encoding
            numeric_type_categories: Number of numeric type categories
        """
        super().__init__()

        self._device = device or torch.device('cpu')
        self.max_sentences = max_sentences
        self.max_sentence_tokens = max_sentence_tokens
        self.num_latent_confounders = num_latent_confounders
        self.value_dim = value_dim
        self.num_attention_heads = num_attention_heads
        self.sparse_attention = sparse_attention
        self.sparse_alpha = sparse_alpha
        self.sparse_method = sparse_method
        self.top_k = top_k
        self.model_type = model_type
        self.numeric_features_enabled = numeric_features_enabled
        self._numeric_embedding_dim = numeric_embedding_dim
        self._numeric_magnitude_bins = numeric_magnitude_bins
        self._numeric_type_categories = numeric_type_categories

        # Token encoder (BERT-based)
        self.token_encoder = TokenEncoder(
            model_name=token_encoder_model,
            device=self._device,
            max_length=max_sentence_tokens,
            freeze=freeze_token_encoder
        )

        # Lazy initialization of dimension-dependent components
        self._encoder_dim = None
        self._input_projection = None

        # Explicit confounder texts (will be encoded lazily)
        self._explicit_confounder_texts = explicit_confounder_texts or []
        self._explicit_embeddings = None

        # Total confounders
        self.num_explicit_confounders = len(self._explicit_confounder_texts)
        self.total_confounders = num_latent_confounders + self.num_explicit_confounders

        # Latent confounders (will be initialized to correct dimension lazily)
        self._latent_confounders = None

        # Token-level cross-attention projections (will be created lazily)
        self._W_q = None
        self._W_k = None
        self._W_v = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection MLP (will be created lazily)
        self._output_projection = None
        self._output_dim = value_dim

        logger.info(f"HierarchicalConfounderExtractor initialized:")
        logger.info(f"  Latent confounders: {num_latent_confounders}")
        logger.info(f"  Explicit confounders: {self.num_explicit_confounders}")
        logger.info(f"  Total confounders: {self.total_confounders}")
        logger.info(f"  Token encoder: {token_encoder_model}")
        logger.info(f"  Sparse attention: {sparse_attention} ({sparse_method}, alpha={sparse_alpha})")

    def _ensure_initialized(self):
        """Lazily initialize components that depend on encoder dimension."""
        if self._encoder_dim is not None:
            return

        self._encoder_dim = self.token_encoder.output_dim

        # Latent confounders: learnable, in BERT embedding space
        self._latent_confounders = nn.Parameter(
            torch.randn(self.num_latent_confounders, self._encoder_dim, device=self._device) * 0.02
        )

        # Encode explicit confounders
        if self._explicit_confounder_texts:
            with torch.no_grad():
                _, explicit_embs = self.token_encoder.encode_sentences_batch(
                    self._explicit_confounder_texts
                )
            self.register_buffer('explicit_confounders', explicit_embs)
        else:
            self.register_buffer('explicit_confounders', torch.zeros(0, self._encoder_dim, device=self._device))

        # Confounder-specific sentence pooling projections
        # Each confounder uses its own attention query to pool token embeddings
        self._W_pool_k = nn.Linear(self._encoder_dim, self._encoder_dim, bias=False).to(self._device)
        self._W_pool_v = nn.Linear(self._encoder_dim, self._encoder_dim, bias=False).to(self._device)

        # Token-level cross-attention projections
        head_dim = self._encoder_dim // self.num_attention_heads
        self._W_q = nn.Linear(self._encoder_dim, self._encoder_dim, bias=False).to(self._device)
        self._W_k = nn.Linear(self._encoder_dim, self._encoder_dim, bias=False).to(self._device)
        self._W_v = nn.Linear(self._encoder_dim, self._encoder_dim, bias=False).to(self._device)

        # Task-specific aggregation queries (3 queries for 3-way aggregation)
        # These allow different weighting of confounders for each task head
        self._propensity_query = nn.Parameter(
            torch.randn(self._encoder_dim, device=self._device) * 0.02
        )
        if self.model_type == "rlearner":
            # R-Learner: propensity, marginal outcome m(X), treatment effect τ(X)
            self._outcome_query = nn.Parameter(
                torch.randn(self._encoder_dim, device=self._device) * 0.02
            )
            self._tau_query = nn.Parameter(
                torch.randn(self._encoder_dim, device=self._device) * 0.02
            )
        else:
            # DragonNet/UpliftNet: propensity, Y0 (outcome under control), Y1 (outcome under treatment)
            self._y0_query = nn.Parameter(
                torch.randn(self._encoder_dim, device=self._device) * 0.02
            )
            self._y1_query = nn.Parameter(
                torch.randn(self._encoder_dim, device=self._device) * 0.02
            )

        # Output projection
        # Task-specific aggregation produces 3*D instead of K*D
        # (propensity + y0/y1 for DragonNet, or propensity + outcome + tau for R-Learner)
        output_input_dim = self._encoder_dim * 3
        self._output_projection = nn.Sequential(
            nn.Linear(output_input_dim, output_input_dim),
            nn.LayerNorm(output_input_dim),
            nn.GELU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(output_input_dim, self.value_dim),
            nn.LayerNorm(self.value_dim)
        ).to(self._device)

        # Numeric feature vector
        self._numeric_feature_vector = None
        if self.numeric_features_enabled:
            self._numeric_feature_vector = NumericFeatureVector(
                num_magnitude_bins=self._numeric_magnitude_bins,
                num_type_categories=self._numeric_type_categories,
                output_dim=self._numeric_embedding_dim
            ).to(self._device)
            output_input_dim = self._encoder_dim * 3
            self._numeric_merge = nn.Sequential(
                nn.Linear(output_input_dim + self._numeric_embedding_dim, output_input_dim),
                nn.LayerNorm(output_input_dim),
                nn.ReLU(),
            ).to(self._device)

        logger.info(f"  Encoder dim: {self._encoder_dim}")

    @property
    def output_dim(self) -> int:
        """Get output embedding dimension."""
        return self._output_dim

    @property
    def latent_confounders(self) -> nn.Parameter:
        """Get latent confounders, ensuring they're initialized."""
        self._ensure_initialized()
        return self._latent_confounders

    def _get_all_confounders(self) -> torch.Tensor:
        """Get combined latent and explicit confounders."""
        self._ensure_initialized()
        if self.explicit_confounders.size(0) > 0:
            return torch.cat([self._latent_confounders, self.explicit_confounders], dim=0)
        return self._latent_confounders

    def _compute_confounder_sentence_embeddings(
        self,
        confounders: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute K confounder-specific sentence embeddings from BERT token embeddings.

        Each confounder uses its own attention query to pool the token embeddings,
        creating K different sentence embeddings per sentence. This allows
        each confounder to focus on different aspects of the sentence.

        Args:
            confounders: (K, D) confounder vectors used as attention queries
            token_embeddings: (L, D) BERT hidden states for one sentence
            attention_mask: (L,) attention mask where 1 = valid token, 0 = padding

        Returns:
            sentence_embs: (K, D) one embedding per confounder
        """
        K = confounders.size(0)

        # Project to keys and values
        keys = self._W_pool_k(token_embeddings)    # (L, D)
        values = self._W_pool_v(token_embeddings)  # (L, D)

        # Each confounder queries the sentence: (K, D) @ (D, L) -> (K, L)
        scores = torch.matmul(confounders, keys.T) / (keys.size(-1) ** 0.5)

        # Mask padding tokens
        scores = scores.masked_fill(attention_mask.unsqueeze(0) == 0, -1e9)

        # Softmax per confounder
        weights = F.softmax(scores, dim=-1)  # (K, L)

        # Weighted sum: (K, L) @ (L, D) -> (K, D)
        sentence_embs = torch.matmul(weights, values)

        return sentence_embs

    def _compute_sentence_attention(
        self,
        confounders: torch.Tensor,
        sentence_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sparse sentence-level attention with confounder-specific views.

        Each confounder k attends to its own confounder-specific sentence embeddings,
        so different confounders can have different relevance assessments of the
        same sentence.

        Args:
            confounders: (K, D) confounder vectors
            sentence_embeddings: (S, K, D) confounder-specific sentence embeddings
            mask: (S,) optional mask where True = ignore

        Returns:
            sentence_weights: (K, S) sparse attention weights
        """
        K, D = confounders.shape
        S = sentence_embeddings.size(0)

        # Each confounder k attends to its own view of sentences
        # scores[k, s] = confounders[k] · sentence_embeddings[s, k]
        # Efficient computation: batch dot products
        # confounders: (K, D), sentence_embeddings: (S, K, D)
        # We want: for each k, compute confounders[k] @ sentence_embeddings[:, k, :].T
        # This gives us (K, S) scores

        # Reshape for batch matmul: (K, 1, D) @ (K, D, S) -> (K, 1, S) -> (K, S)
        confounders_expanded = confounders.unsqueeze(1)  # (K, 1, D)
        sentence_embs_transposed = sentence_embeddings.permute(1, 2, 0)  # (K, D, S)
        scores = torch.bmm(confounders_expanded, sentence_embs_transposed).squeeze(1)  # (K, S)
        scores = scores / (D ** 0.5)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

        # Apply sparse attention
        if self.sparse_attention:
            if self.sparse_method == "topk":
                weights = top_k_attention(scores, k=self.top_k, dim=-1)
            else:
                weights = sparse_softmax(scores, dim=-1, alpha=self.sparse_alpha)

            # Fallback to softmax if sparse attention produces all zeros
            # This can happen early in training when embeddings are random
            if weights.sum() < 1e-6:
                weights = F.softmax(scores, dim=-1)
        else:
            weights = F.softmax(scores, dim=-1)

        return weights

    def _compute_token_attention(
        self,
        confounder: torch.Tensor,
        token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute token-level cross-attention within a sentence.

        Args:
            confounder: (D,) single confounder vector
            token_embeddings: (L, D) token embeddings for one sentence

        Returns:
            attended: (D,) weighted sum of tokens
        """
        # Project query (confounder), keys, values
        q = self._W_q(confounder.unsqueeze(0))  # (1, D)
        k = self._W_k(token_embeddings)  # (L, D)
        v = self._W_v(token_embeddings)  # (L, D)

        # Compute attention scores
        scores = torch.matmul(q, k.T) / (q.size(-1) ** 0.5)  # (1, L)

        # Softmax (token attention is typically dense within sentence)
        weights = F.softmax(scores, dim=-1)  # (1, L)
        weights = self.dropout(weights)

        # Weighted sum
        attended = torch.matmul(weights, v)  # (1, D)

        return attended.squeeze(0)

    def _aggregate_confounders(
        self,
        confounders: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate K confounder representations using 3 task-specific queries.

        For DragonNet/UpliftNet: propensity, Y0 (control outcome), Y1 (treatment outcome)
        For R-Learner: propensity, m(X) (marginal outcome), τ(X) (treatment effect)

        This 3-way aggregation allows:
        - DragonNet: Different confounders for Y0 vs Y1 (treatment effect modifiers)
        - R-Learner: Distinguish prognostic factors (m) from effect modifiers (τ)

        Args:
            confounders: (K, D) confounder representations for one document

        Returns:
            aggregated: (3*D,) concatenated task-specific representations
            prop_weights: (K,) propensity task confounder weights
            weight2: (K,) y0 weights (DragonNet) or outcome weights (R-Learner)
            weight3: (K,) y1 weights (DragonNet) or tau weights (R-Learner)
        """
        # Propensity task aggregation (same for both model types)
        prop_scores = torch.matmul(confounders, self._propensity_query)  # (K,)
        prop_weights = F.softmax(prop_scores, dim=0)  # (K,)
        prop_repr = torch.matmul(prop_weights, confounders)  # (D,)

        if self.model_type == "rlearner":
            # Outcome aggregation: m(X)
            out_scores = torch.matmul(confounders, self._outcome_query)  # (K,)
            out_weights = F.softmax(out_scores, dim=0)  # (K,)
            out_repr = torch.matmul(out_weights, confounders)  # (D,)

            # Tau aggregation: τ(X) - treatment effect modifiers
            tau_scores = torch.matmul(confounders, self._tau_query)  # (K,)
            tau_weights = F.softmax(tau_scores, dim=0)  # (K,)
            tau_repr = torch.matmul(tau_weights, confounders)  # (D,)

            aggregated = torch.cat([prop_repr, out_repr, tau_repr], dim=0)  # (3*D,)
            return aggregated, prop_weights, out_weights, tau_weights
        else:
            # Y0 aggregation: outcome under control
            y0_scores = torch.matmul(confounders, self._y0_query)  # (K,)
            y0_weights = F.softmax(y0_scores, dim=0)  # (K,)
            y0_repr = torch.matmul(y0_weights, confounders)  # (D,)

            # Y1 aggregation: outcome under treatment
            y1_scores = torch.matmul(confounders, self._y1_query)  # (K,)
            y1_weights = F.softmax(y1_scores, dim=0)  # (K,)
            y1_repr = torch.matmul(y1_weights, confounders)  # (D,)

            aggregated = torch.cat([prop_repr, y0_repr, y1_repr], dim=0)  # (3*D,)
            return aggregated, prop_weights, y0_weights, y1_weights

    def forward(
        self,
        texts: List[str],
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Extract confounder representations from texts using hierarchical attention.

        Uses confounder-specific sentence pooling: each confounder uses its own
        attention query to create a different view of each sentence before
        computing sentence-level sparse attention.

        Args:
            texts: List of document texts
            return_attention: Whether to return attention weights

        Returns:
            features: Feature tensor (batch, output_dim)
        """
        self._ensure_initialized()
        batch_size = len(texts)
        batch_results = []
        all_attention_weights = [] if return_attention else None

        for text in texts:
            # 1. Split into sentences
            sentences = split_into_sentences(text, self.max_sentences)
            if not sentences:
                sentences = [text[:500]]  # Fallback

            # 2. Get all confounders (needed for confounder-specific pooling)
            all_confounders = self._get_all_confounders()  # (K, D)

            # 3. Encode each sentence and compute confounder-specific embeddings
            # token_embeddings_list: List[(L_i, D)], one per sentence
            # sentence_embeddings: (S, K, D), one embedding per confounder per sentence
            token_embeddings_list, _ = self.token_encoder.encode_sentences_batch(sentences)

            # Compute confounder-specific sentence embeddings
            confounder_sentence_embeddings = []
            for sent_tokens in token_embeddings_list:
                # Create a simple attention mask (all valid since we have the tokens)
                attention_mask = torch.ones(sent_tokens.size(0), device=self._device)
                conf_sent_emb = self._compute_confounder_sentence_embeddings(
                    all_confounders, sent_tokens, attention_mask
                )  # (K, D)
                confounder_sentence_embeddings.append(conf_sent_emb)

            # Stack to (S, K, D)
            sentence_embeddings = torch.stack(confounder_sentence_embeddings)

            # 4. Sentence-level sparse attention (each confounder uses its own view)
            sentence_weights = self._compute_sentence_attention(
                all_confounders, sentence_embeddings
            )  # (K, S)

            # 5. Token-level cross-attention per sentence, gated by sentence weights
            confounder_reprs = []
            for k in range(self.total_confounders):
                weighted_repr = torch.zeros(self._encoder_dim, device=self._device)

                for s, sent_tokens in enumerate(token_embeddings_list):
                    weight = sentence_weights[k, s].item()
                    if weight < 1e-6:
                        continue  # Skip zero-weight sentences (sparse!)

                    # Token attention within sentence
                    sent_repr = self._compute_token_attention(
                        all_confounders[k], sent_tokens
                    )

                    # Gate by sentence importance
                    weighted_repr = weighted_repr + sentence_weights[k, s] * sent_repr

                confounder_reprs.append(weighted_repr)

            # Stack confounders: (K, D)
            doc_confounders = torch.stack(confounder_reprs)

            # 6. Task-specific aggregation: (K, D) -> (3*D,)
            aggregated, prop_weights, weight2, weight3 = self._aggregate_confounders(doc_confounders)
            batch_results.append(aggregated)

            if return_attention:
                attn_info = {
                    'sentence_weights': sentence_weights.detach().cpu(),
                    'sentences': sentences,
                    'propensity_confounder_weights': prop_weights.detach().cpu()
                }
                if self.model_type == "rlearner":
                    attn_info['outcome_confounder_weights'] = weight2.detach().cpu()
                    attn_info['tau_confounder_weights'] = weight3.detach().cpu()
                else:
                    attn_info['y0_confounder_weights'] = weight2.detach().cpu()
                    attn_info['y1_confounder_weights'] = weight3.detach().cpu()
                all_attention_weights.append(attn_info)

        # Stack batch: (B, 3*D)
        batch_aggregated = torch.stack(batch_results)

        # Add numeric features before output projection
        if self.numeric_features_enabled and self._numeric_feature_vector is not None:
            numeric_feats = self._numeric_feature_vector(texts)
            batch_aggregated = self._numeric_merge(
                torch.cat([batch_aggregated, numeric_feats], dim=1)
            )

        # Project to output: (B, 3*D) -> (B, value_dim)
        features = self._output_projection(batch_aggregated)

        if return_attention:
            return features, all_attention_weights
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
            Dictionary with attention information per document
        """
        _, attention_info = self.forward(texts, return_attention=True)

        confounder_names = [f"latent_{i}" for i in range(self.num_latent_confounders)]
        confounder_names += [f"explicit_{i}_{t[:20]}" for i, t in enumerate(self._explicit_confounder_texts)]

        return {
            'attention_info': attention_info,
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
            List of dictionaries per document with interpretations
        """
        result = self.get_attention_weights(texts)
        attention_info = result['attention_info']
        confounder_names = result['confounder_names']

        interpretations = []
        for doc_idx, doc_info in enumerate(attention_info):
            sentences = doc_info['sentences']
            sentence_weights = doc_info['sentence_weights']  # (K, S)

            doc_interp = {}
            for conf_idx, conf_name in enumerate(confounder_names):
                conf_weights = sentence_weights[conf_idx]  # (S,)

                # Get top-k
                k_actual = min(top_k, len(sentences))
                top_vals, top_indices = torch.topk(conf_weights, k_actual)

                top_sentences = []
                for val, idx in zip(top_vals.tolist(), top_indices.tolist()):
                    if val > 0.001:
                        top_sentences.append({
                            'sentence': sentences[idx],
                            'attention': val
                        })

                doc_interp[conf_name] = top_sentences

            interpretations.append(doc_interp)

        return interpretations

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.token_encoder = self.token_encoder.to(device)
        if hasattr(self, '_W_pool_k') and self._W_pool_k is not None:
            self._W_pool_k = self._W_pool_k.to(device)
            self._W_pool_v = self._W_pool_v.to(device)
        if self._W_q is not None:
            self._W_q = self._W_q.to(device)
            self._W_k = self._W_k.to(device)
            self._W_v = self._W_v.to(device)
        if self._output_projection is not None:
            self._output_projection = self._output_projection.to(device)
        # Move task-specific aggregation query parameters
        if hasattr(self, '_propensity_query') and self._propensity_query is not None:
            self._propensity_query.data = self._propensity_query.data.to(device)
        # R-Learner queries
        if hasattr(self, '_outcome_query') and self._outcome_query is not None:
            self._outcome_query.data = self._outcome_query.data.to(device)
        if hasattr(self, '_tau_query') and self._tau_query is not None:
            self._tau_query.data = self._tau_query.data.to(device)
        # DragonNet queries
        if hasattr(self, '_y0_query') and self._y0_query is not None:
            self._y0_query.data = self._y0_query.data.to(device)
        if hasattr(self, '_y1_query') and self._y1_query is not None:
            self._y1_query.data = self._y1_query.data.to(device)
        return super().to(device)

    def get_state(self) -> Dict[str, Any]:
        """Get extractor state for checkpoint saving."""
        return {
            'num_latent_confounders': self.num_latent_confounders,
            'explicit_confounder_texts': self._explicit_confounder_texts,
            'value_dim': self.value_dim,
            'max_sentences': self.max_sentences,
            'output_dim': self._output_dim,
            'hierarchical': True,
            'model_type': self.model_type
        }

    # Compatibility with CausalText interface
    def fit_tokenizer(self, texts: List[str]) -> 'HierarchicalConfounderExtractor':
        """No-op for compatibility. Uses pretrained token encoder."""
        self._ensure_initialized()
        return self


class GRUHierarchicalConfounderExtractor(nn.Module):
    """
    GRU-based hierarchical confounder extractor that learns entirely from scratch.

    Unlike HierarchicalConfounderExtractor which uses pretrained BERT for token encoding,
    this extractor uses a BiGRU with learnable word embeddings. All parameters (embeddings,
    GRU weights, attention layers, latent confounders) are learned together via the causal
    loss. This ensures that confounder representations are optimized directly for causal
    inference rather than general language understanding.

    Architecture:
    1. Split text into sentences
    2. Tokenize each sentence with word-level tokenizer
    3. Embed tokens with learnable embeddings
    4. Encode each sentence with BiGRU + attention pooling
    5. Sentence-Level Sparse Attention (entmax) → Sentence Weights (K × S)
    6. Token-Level Cross-Attention (within each sentence, gated by sentence weights)
    7. K Confounder Representations → Causal Head

    Key advantages:
    - All parameters learn together from causal objective
    - No domain mismatch from pretrained encoder
    - Latent confounders adapt to the specific clinical context
    - Lighter weight than BERT-based approaches
    """

    def __init__(
        self,
        # Vocabulary / tokenizer
        vocab_size: int = 50000,
        embedding_dim: int = 128,
        min_word_freq: int = 2,
        max_sentence_length: int = 128,
        # GRU encoder
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        gru_bidirectional: bool = True,
        gru_dropout: float = 0.1,
        # Confounder architecture
        num_latent_confounders: int = 8,
        num_attention_heads: int = 4,
        sparse_attention: bool = True,
        sparse_alpha: float = 1.5,
        sparse_method: str = "entmax",
        top_k: int = 5,
        max_sentences: int = 100,
        value_dim: int = 128,
        # Regularization
        dropout: float = 0.1,
        # Model type for task-specific aggregation
        model_type: str = "dragonnet",
        # Device
        device: Optional[torch.device] = None,
        # Numeric features
        numeric_features_enabled: bool = False,
        numeric_embedding_dim: int = 32,
        numeric_magnitude_bins: int = 8,
        numeric_type_categories: int = 10
    ):
        """
        Initialize GRU-based hierarchical confounder extractor.

        Args:
            vocab_size: Maximum vocabulary size
            embedding_dim: Dimension of word embeddings
            min_word_freq: Minimum word frequency for vocabulary inclusion
            max_sentence_length: Maximum tokens per sentence
            gru_hidden_dim: GRU hidden state dimension per direction
            gru_num_layers: Number of stacked GRU layers
            gru_bidirectional: Use bidirectional GRU
            gru_dropout: Dropout rate in GRU
            num_latent_confounders: Number of learnable latent confounder vectors
            num_attention_heads: Number of attention heads for token cross-attention
            sparse_attention: Whether to use sparse attention (entmax/top-k)
            sparse_alpha: Alpha for entmax (1.5=entmax15, 2.0=sparsemax)
            sparse_method: Sparsity method ("entmax", "topk", "softmax")
            top_k: K for top-k attention method
            max_sentences: Maximum sentences to process per document
            value_dim: Output dimension per confounder
            dropout: Dropout rate
            model_type: Architecture type ("dragonnet", "uplift", or "rlearner") for
                       task-specific aggregation. DragonNet uses propensity/Y0/Y1 queries,
                       R-Learner uses propensity/outcome/tau queries.
            device: Device to place model on
            numeric_features_enabled: Whether to extract and merge numeric features from text
            numeric_embedding_dim: Output dimension of numeric feature vector
            numeric_magnitude_bins: Number of magnitude bins for numeric encoding
            numeric_type_categories: Number of numeric type categories
        """
        super().__init__()

        self._device = device or torch.device('cpu')
        self.max_sentences = max_sentences
        self.max_sentence_length = max_sentence_length
        self.num_latent_confounders = num_latent_confounders
        self.value_dim = value_dim
        self.num_attention_heads = num_attention_heads
        self.sparse_attention = sparse_attention
        self.sparse_alpha = sparse_alpha
        self.sparse_method = sparse_method
        self.top_k = top_k
        self.gru_bidirectional = gru_bidirectional
        self.model_type = model_type
        self.numeric_features_enabled = numeric_features_enabled
        self._numeric_embedding_dim = numeric_embedding_dim
        self._numeric_magnitude_bins = numeric_magnitude_bins
        self._numeric_type_categories = numeric_type_categories

        # Import WordTokenizer from cnn_extractor
        from .cnn_extractor import WordTokenizer

        # Word tokenizer (learns vocabulary from training data)
        self.tokenizer = WordTokenizer(
            max_length=max_sentence_length,
            min_freq=min_word_freq,
            max_vocab_size=vocab_size
        )

        # Store config for later initialization
        self._embedding_dim = embedding_dim
        self._gru_hidden_dim = gru_hidden_dim
        self._gru_num_layers = gru_num_layers
        self._gru_dropout = gru_dropout
        self._dropout_rate = dropout

        # Compute GRU output dimension
        self.num_directions = 2 if gru_bidirectional else 1
        self._encoder_dim = gru_hidden_dim * self.num_directions

        # These will be initialized after fit_tokenizer()
        self.embedding = None
        self.gru = None
        self._W_pool_k = None
        self._W_pool_v = None
        self._initialized = False

        # Latent confounders: learnable, in GRU embedding space
        self.latent_confounders = nn.Parameter(
            torch.randn(num_latent_confounders, self._encoder_dim, device=self._device) * 0.02
        )

        # Task-specific aggregation queries (3 queries for 3-way aggregation)
        # These allow different weighting of confounders for each task head
        self._propensity_query = nn.Parameter(
            torch.randn(self._encoder_dim, device=self._device) * 0.02
        )
        if model_type == "rlearner":
            # R-Learner: propensity, marginal outcome m(X), treatment effect τ(X)
            self._outcome_query = nn.Parameter(
                torch.randn(self._encoder_dim, device=self._device) * 0.02
            )
            self._tau_query = nn.Parameter(
                torch.randn(self._encoder_dim, device=self._device) * 0.02
            )
        else:
            # DragonNet/UpliftNet: propensity, Y0 (outcome under control), Y1 (outcome under treatment)
            self._y0_query = nn.Parameter(
                torch.randn(self._encoder_dim, device=self._device) * 0.02
            )
            self._y1_query = nn.Parameter(
                torch.randn(self._encoder_dim, device=self._device) * 0.02
            )

        # Token-level cross-attention projections
        self._W_q = nn.Linear(self._encoder_dim, self._encoder_dim, bias=False)
        self._W_k = nn.Linear(self._encoder_dim, self._encoder_dim, bias=False)
        self._W_v = nn.Linear(self._encoder_dim, self._encoder_dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        # Task-specific aggregation produces 3*D instead of K*D
        # (propensity + y0/y1 for DragonNet, or propensity + outcome + tau for R-Learner)
        output_input_dim = self._encoder_dim * 3
        self._output_projection = nn.Sequential(
            nn.Linear(output_input_dim, output_input_dim),
            nn.LayerNorm(output_input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_input_dim, value_dim),
            nn.LayerNorm(value_dim)
        )

        # Numeric feature vector
        self._numeric_feature_vector = None
        if numeric_features_enabled:
            self._numeric_feature_vector = NumericFeatureVector(
                num_magnitude_bins=numeric_magnitude_bins,
                num_type_categories=numeric_type_categories,
                output_dim=numeric_embedding_dim
            )
            output_input_dim = self._encoder_dim * 3
            self._numeric_merge = nn.Sequential(
                nn.Linear(output_input_dim + numeric_embedding_dim, output_input_dim),
                nn.LayerNorm(output_input_dim),
                nn.ReLU(),
            )

        self._output_dim = value_dim

        # No explicit confounders for GRU version (all learned from scratch)
        self.num_explicit_confounders = 0
        self.total_confounders = num_latent_confounders

        logger.info(f"GRUHierarchicalConfounderExtractor initialized:")
        logger.info(f"  Latent confounders: {num_latent_confounders}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  GRU hidden dim: {gru_hidden_dim} x {self.num_directions}")
        logger.info(f"  Encoder dim: {self._encoder_dim}")
        logger.info(f"  Sparse attention: {sparse_attention} ({sparse_method}, alpha={sparse_alpha})")

    def _ensure_initialized(self):
        """Ensure embedding and GRU are initialized after tokenizer is fitted."""
        if self._initialized:
            return

        if self.tokenizer.vocab_size == 0:
            raise RuntimeError("Must call fit_tokenizer() before using the model")

        # Initialize embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=self._embedding_dim,
            padding_idx=self.tokenizer.pad_token
        )
        self.embedding.to(self._device)

        # Initialize GRU
        self.gru = nn.GRU(
            input_size=self._embedding_dim,
            hidden_size=self._gru_hidden_dim,
            num_layers=self._gru_num_layers,
            batch_first=True,
            dropout=self._gru_dropout if self._gru_num_layers > 1 else 0,
            bidirectional=self.gru_bidirectional
        )
        self.gru.to(self._device)

        # Confounder-specific sentence pooling projections
        # Instead of a single shared attention pooling, each confounder uses its own
        # attention query to pool the sentence, creating K different sentence embeddings
        self._W_pool_k = nn.Linear(self._encoder_dim, self._encoder_dim, bias=False)
        self._W_pool_v = nn.Linear(self._encoder_dim, self._encoder_dim, bias=False)
        self._W_pool_k.to(self._device)
        self._W_pool_v.to(self._device)

        # Embedding layer norm and dropout
        self._embed_layer_norm = nn.LayerNorm(self._embedding_dim)
        self._embed_layer_norm.to(self._device)
        self._embed_dropout = nn.Dropout(self._dropout_rate)

        self._initialized = True
        logger.info(f"  Vocab size: {self.tokenizer.vocab_size}")

    @property
    def output_dim(self) -> int:
        """Get output embedding dimension."""
        return self._output_dim

    def _compute_confounder_sentence_embeddings(
        self,
        confounders: torch.Tensor,
        gru_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute K confounder-specific sentence embeddings from GRU output.

        Each confounder uses its own attention query to pool the sentence,
        creating K different sentence embeddings per sentence. This allows
        each confounder to focus on different aspects of the sentence.

        Args:
            confounders: (K, D) confounder vectors used as attention queries
            gru_output: (L, D) GRU hidden states for one sentence
            attention_mask: (L,) attention mask where 1 = valid token, 0 = padding

        Returns:
            sentence_embs: (K, D) one embedding per confounder
        """
        K = confounders.size(0)

        # Project to keys and values
        keys = self._W_pool_k(gru_output)    # (L, D)
        values = self._W_pool_v(gru_output)  # (L, D)

        # Each confounder queries the sentence: (K, D) @ (D, L) -> (K, L)
        scores = torch.matmul(confounders, keys.T) / (keys.size(-1) ** 0.5)

        # Mask padding tokens
        scores = scores.masked_fill(attention_mask.unsqueeze(0) == 0, -1e9)

        # Softmax per confounder
        weights = F.softmax(scores, dim=-1)  # (K, L)

        # Weighted sum: (K, L) @ (L, D) -> (K, D)
        sentence_embs = torch.matmul(weights, values)

        return sentence_embs

    def fit_tokenizer(self, texts: List[str]) -> 'GRUHierarchicalConfounderExtractor':
        """
        Fit tokenizer on training texts and initialize embedding layer.

        MUST be called before using the model for training or inference.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        # Collect all sentences from all documents
        all_sentences = []
        for text in texts:
            sentences = split_into_sentences(text, self.max_sentences)
            all_sentences.extend(sentences)

        # Fit tokenizer on all sentences
        self.tokenizer.fit(all_sentences)

        # Initialize embedding and GRU
        self._ensure_initialized()

        logger.info(f"Tokenizer fitted on {len(all_sentences)} sentences, vocab size = {self.tokenizer.vocab_size}")

        return self

    def _encode_sentence_tokens(
        self,
        sentence: str,
        confounders: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single sentence, returning token embeddings and confounder-specific
        sentence embeddings.

        Args:
            sentence: Input sentence string
            confounders: (K, D) confounder vectors used as attention queries

        Returns:
            token_embeddings: (L, encoder_dim) tensor of token embeddings from GRU
            sentence_embeddings: (K, encoder_dim) tensor with one embedding per confounder
        """
        self._ensure_initialized()

        # Tokenize
        encoded = self.tokenizer(
            [sentence],
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self._device)  # (1, L)
        attention_mask = encoded['attention_mask'].to(self._device)  # (1, L)

        # Embed tokens
        embeddings = self.embedding(input_ids)  # (1, L, embedding_dim)
        embeddings = self._embed_layer_norm(embeddings)
        embeddings = self._embed_dropout(embeddings)

        # GRU forward
        gru_output, _ = self.gru(embeddings)  # (1, L, encoder_dim)

        # Token embeddings
        token_embs = gru_output[0]  # (L, encoder_dim)

        # Confounder-specific sentence embeddings using query-based pooling
        sentence_embs = self._compute_confounder_sentence_embeddings(
            confounders, token_embs, attention_mask[0]
        )  # (K, encoder_dim)

        return token_embs, sentence_embs

    def _encode_sentences_batch(
        self,
        sentences: List[str],
        confounders: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Encode multiple sentences, returning token embeddings per sentence
        and confounder-specific sentence embeddings.

        Args:
            sentences: List of sentence strings
            confounders: (K, D) confounder vectors used as attention queries

        Returns:
            token_embeddings_list: List of (L_i, encoder_dim) tensors for each sentence
            sentence_embeddings: (S, K, encoder_dim) tensor with one embedding per
                                 confounder per sentence
        """
        K = confounders.size(0)
        if not sentences:
            return [], torch.zeros(0, K, self._encoder_dim, device=self._device)

        token_embeddings_list = []
        sentence_embeddings = []

        for sentence in sentences:
            token_embs, sentence_emb = self._encode_sentence_tokens(sentence, confounders)
            token_embeddings_list.append(token_embs)
            sentence_embeddings.append(sentence_emb)  # (K, encoder_dim)

        # Stack to (S, K, encoder_dim)
        return token_embeddings_list, torch.stack(sentence_embeddings)

    def _compute_sentence_attention(
        self,
        confounders: torch.Tensor,
        sentence_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sparse sentence-level attention with confounder-specific views.

        Each confounder k attends to its own confounder-specific sentence embeddings,
        so different confounders can have different relevance assessments of the
        same sentence.

        Args:
            confounders: (K, D) confounder vectors
            sentence_embeddings: (S, K, D) confounder-specific sentence embeddings
            mask: (S,) optional mask where True = ignore

        Returns:
            sentence_weights: (K, S) sparse attention weights
        """
        K, D = confounders.shape
        S = sentence_embeddings.size(0)

        # Each confounder k attends to its own view of sentences
        # scores[k, s] = confounders[k] · sentence_embeddings[s, k]
        # Efficient computation: batch dot products
        # confounders: (K, D), sentence_embeddings: (S, K, D)
        # We want: for each k, compute confounders[k] @ sentence_embeddings[:, k, :].T
        # This gives us (K, S) scores

        # Reshape for batch matmul: (K, 1, D) @ (K, D, S) -> (K, 1, S) -> (K, S)
        confounders_expanded = confounders.unsqueeze(1)  # (K, 1, D)
        sentence_embs_transposed = sentence_embeddings.permute(1, 2, 0)  # (K, D, S)
        scores = torch.bmm(confounders_expanded, sentence_embs_transposed).squeeze(1)  # (K, S)
        scores = scores / (D ** 0.5)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

        # Apply sparse attention
        if self.sparse_attention:
            if self.sparse_method == "topk":
                weights = top_k_attention(scores, k=self.top_k, dim=-1)
            else:
                weights = sparse_softmax(scores, dim=-1, alpha=self.sparse_alpha)

            # Fallback to softmax if sparse attention produces all zeros
            # This can happen early in training when embeddings are random
            if weights.sum() < 1e-6:
                weights = F.softmax(scores, dim=-1)
        else:
            weights = F.softmax(scores, dim=-1)

        return weights

    def _compute_token_attention(
        self,
        confounder: torch.Tensor,
        token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute token-level cross-attention within a sentence.

        Args:
            confounder: (D,) single confounder vector
            token_embeddings: (L, D) token embeddings for one sentence

        Returns:
            attended: (D,) weighted sum of tokens
        """
        # Project query (confounder), keys, values
        q = self._W_q(confounder.unsqueeze(0))  # (1, D)
        k = self._W_k(token_embeddings)  # (L, D)
        v = self._W_v(token_embeddings)  # (L, D)

        # Compute attention scores
        scores = torch.matmul(q, k.T) / (q.size(-1) ** 0.5)  # (1, L)

        # Softmax (token attention is typically dense within sentence)
        weights = F.softmax(scores, dim=-1)  # (1, L)
        weights = self.dropout(weights)

        # Weighted sum
        attended = torch.matmul(weights, v)  # (1, D)

        return attended.squeeze(0)

    def _aggregate_confounders(
        self,
        confounders: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate K confounder representations using 3 task-specific queries.

        For DragonNet/UpliftNet: propensity, Y0 (control outcome), Y1 (treatment outcome)
        For R-Learner: propensity, m(X) (marginal outcome), τ(X) (treatment effect)

        This 3-way aggregation allows:
        - DragonNet: Different confounders for Y0 vs Y1 (treatment effect modifiers)
        - R-Learner: Distinguish prognostic factors (m) from effect modifiers (τ)

        Args:
            confounders: (K, D) confounder representations for one document

        Returns:
            aggregated: (3*D,) concatenated task-specific representations
            prop_weights: (K,) propensity task confounder weights
            weight2: (K,) y0 weights (DragonNet) or outcome weights (R-Learner)
            weight3: (K,) y1 weights (DragonNet) or tau weights (R-Learner)
        """
        # Propensity task aggregation (same for both model types)
        prop_scores = torch.matmul(confounders, self._propensity_query)  # (K,)
        prop_weights = F.softmax(prop_scores, dim=0)  # (K,)
        prop_repr = torch.matmul(prop_weights, confounders)  # (D,)

        if self.model_type == "rlearner":
            # Outcome aggregation: m(X)
            out_scores = torch.matmul(confounders, self._outcome_query)  # (K,)
            out_weights = F.softmax(out_scores, dim=0)  # (K,)
            out_repr = torch.matmul(out_weights, confounders)  # (D,)

            # Tau aggregation: τ(X) - treatment effect modifiers
            tau_scores = torch.matmul(confounders, self._tau_query)  # (K,)
            tau_weights = F.softmax(tau_scores, dim=0)  # (K,)
            tau_repr = torch.matmul(tau_weights, confounders)  # (D,)

            aggregated = torch.cat([prop_repr, out_repr, tau_repr], dim=0)  # (3*D,)
            return aggregated, prop_weights, out_weights, tau_weights
        else:
            # Y0 aggregation: outcome under control
            y0_scores = torch.matmul(confounders, self._y0_query)  # (K,)
            y0_weights = F.softmax(y0_scores, dim=0)  # (K,)
            y0_repr = torch.matmul(y0_weights, confounders)  # (D,)

            # Y1 aggregation: outcome under treatment
            y1_scores = torch.matmul(confounders, self._y1_query)  # (K,)
            y1_weights = F.softmax(y1_scores, dim=0)  # (K,)
            y1_repr = torch.matmul(y1_weights, confounders)  # (D,)

            aggregated = torch.cat([prop_repr, y0_repr, y1_repr], dim=0)  # (3*D,)
            return aggregated, prop_weights, y0_weights, y1_weights

    def forward(
        self,
        texts: List[str],
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Extract confounder representations from texts using hierarchical attention.

        Uses confounder-specific sentence pooling: each confounder uses its own
        attention query to create a different view of each sentence before
        computing sentence-level sparse attention.

        Args:
            texts: List of document texts
            return_attention: Whether to return attention weights

        Returns:
            features: Feature tensor (batch, output_dim)
        """
        self._ensure_initialized()
        batch_size = len(texts)
        batch_results = []
        all_attention_weights = [] if return_attention else None

        for text in texts:
            # 1. Split into sentences
            sentences = split_into_sentences(text, self.max_sentences)
            if not sentences:
                sentences = [text[:500]]  # Fallback

            # 2. Get latent confounders (needed for confounder-specific pooling)
            all_confounders = self.latent_confounders  # (K, D)

            # 3. Encode each sentence with confounder-specific pooling
            # Returns: token_embeddings_list: List[(L_i, D)], sentence_embeddings: (S, K, D)
            token_embeddings_list, sentence_embeddings = self._encode_sentences_batch(
                sentences, all_confounders
            )

            # 4. Sentence-level sparse attention (each confounder uses its own view)
            sentence_weights = self._compute_sentence_attention(
                all_confounders, sentence_embeddings
            )  # (K, S)

            # 5. Token-level cross-attention per sentence, gated by sentence weights
            confounder_reprs = []
            for k in range(self.total_confounders):
                weighted_repr = torch.zeros(self._encoder_dim, device=self._device)

                for s, sent_tokens in enumerate(token_embeddings_list):
                    weight = sentence_weights[k, s].item()
                    if weight < 1e-6:
                        continue  # Skip zero-weight sentences (sparse!)

                    # Token attention within sentence
                    sent_repr = self._compute_token_attention(
                        all_confounders[k], sent_tokens
                    )

                    # Gate by sentence importance
                    weighted_repr = weighted_repr + sentence_weights[k, s] * sent_repr

                confounder_reprs.append(weighted_repr)

            # Stack confounders: (K, D)
            doc_confounders = torch.stack(confounder_reprs)

            # 6. Task-specific aggregation: (K, D) -> (3*D,)
            aggregated, prop_weights, weight2, weight3 = self._aggregate_confounders(doc_confounders)
            batch_results.append(aggregated)

            if return_attention:
                attn_info = {
                    'sentence_weights': sentence_weights.detach().cpu(),
                    'sentences': sentences,
                    'propensity_confounder_weights': prop_weights.detach().cpu()
                }
                if self.model_type == "rlearner":
                    attn_info['outcome_confounder_weights'] = weight2.detach().cpu()
                    attn_info['tau_confounder_weights'] = weight3.detach().cpu()
                else:
                    attn_info['y0_confounder_weights'] = weight2.detach().cpu()
                    attn_info['y1_confounder_weights'] = weight3.detach().cpu()
                all_attention_weights.append(attn_info)

        # Stack batch: (B, 3*D)
        batch_aggregated = torch.stack(batch_results)

        # Add numeric features before output projection
        if self.numeric_features_enabled and self._numeric_feature_vector is not None:
            numeric_feats = self._numeric_feature_vector(texts)
            batch_aggregated = self._numeric_merge(
                torch.cat([batch_aggregated, numeric_feats], dim=1)
            )

        # Project to output: (B, 3*D) -> (B, value_dim)
        features = self._output_projection(batch_aggregated)

        if return_attention:
            return features, all_attention_weights
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
            Dictionary with attention information per document
        """
        _, attention_info = self.forward(texts, return_attention=True)

        confounder_names = [f"latent_{i}" for i in range(self.num_latent_confounders)]

        return {
            'attention_info': attention_info,
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
            List of dictionaries per document with interpretations
        """
        result = self.get_attention_weights(texts)
        attention_info = result['attention_info']
        confounder_names = result['confounder_names']

        interpretations = []
        for doc_idx, doc_info in enumerate(attention_info):
            sentences = doc_info['sentences']
            sentence_weights = doc_info['sentence_weights']  # (K, S)

            doc_interp = {}
            for conf_idx, conf_name in enumerate(confounder_names):
                conf_weights = sentence_weights[conf_idx]  # (S,)

                # Get top-k
                k_actual = min(top_k, len(sentences))
                top_vals, top_indices = torch.topk(conf_weights, k_actual)

                top_sentences = []
                for val, idx in zip(top_vals.tolist(), top_indices.tolist()):
                    if val > 0.001:
                        top_sentences.append({
                            'sentence': sentences[idx],
                            'attention': val
                        })

                doc_interp[conf_name] = top_sentences

            interpretations.append(doc_interp)

        return interpretations

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        if self.embedding is not None:
            self.embedding = self.embedding.to(device)
        if self.gru is not None:
            self.gru = self.gru.to(device)
        if hasattr(self, '_W_pool_k') and self._W_pool_k is not None:
            self._W_pool_k = self._W_pool_k.to(device)
            self._W_pool_v = self._W_pool_v.to(device)
        if hasattr(self, '_embed_layer_norm') and self._embed_layer_norm is not None:
            self._embed_layer_norm = self._embed_layer_norm.to(device)
        self._W_q = self._W_q.to(device)
        self._W_k = self._W_k.to(device)
        self._W_v = self._W_v.to(device)
        self._output_projection = self._output_projection.to(device)
        # Move task-specific aggregation query parameters
        if hasattr(self, '_propensity_query'):
            self._propensity_query.data = self._propensity_query.data.to(device)
        # R-Learner queries
        if hasattr(self, '_outcome_query') and self._outcome_query is not None:
            self._outcome_query.data = self._outcome_query.data.to(device)
        if hasattr(self, '_tau_query') and self._tau_query is not None:
            self._tau_query.data = self._tau_query.data.to(device)
        # DragonNet queries
        if hasattr(self, '_y0_query') and self._y0_query is not None:
            self._y0_query.data = self._y0_query.data.to(device)
        if hasattr(self, '_y1_query') and self._y1_query is not None:
            self._y1_query.data = self._y1_query.data.to(device)
        return super().to(device)

    def get_state(self) -> Dict[str, Any]:
        """Get extractor state for checkpoint saving."""
        return {
            'num_latent_confounders': self.num_latent_confounders,
            'value_dim': self.value_dim,
            'max_sentences': self.max_sentences,
            'output_dim': self._output_dim,
            'gru_hierarchical': True,
            'model_type': self.model_type,
            'vocab_size': self.tokenizer.vocab_size if self._initialized else 0
        }

    def get_tokenizer_state(self) -> Dict[str, Any]:
        """Get tokenizer state for checkpoint saving."""
        return {
            'word_to_id': self.tokenizer.word_to_id,
            'id_to_word': self.tokenizer.id_to_word,
            'vocab_size': self.tokenizer.vocab_size,
            'max_length': self.tokenizer.max_length,
            'min_freq': self.tokenizer.min_freq,
            'max_vocab_size': self.tokenizer.max_vocab_size
        }

    def load_tokenizer_state(self, state: Dict[str, Any]) -> None:
        """Load tokenizer state from checkpoint."""
        self.tokenizer.word_to_id = state['word_to_id']
        self.tokenizer.id_to_word = state['id_to_word']
        self.tokenizer._vocab_size = state['vocab_size']
        self._ensure_initialized()
