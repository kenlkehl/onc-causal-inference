# cdt/models/gated_mil_hierarchical_extractor.py
"""Gated MIL Hierarchical feature extractor using sentence-level BERT + gated attention.

This module implements a hierarchical approach for extracting features from long
clinical text using gated MIL (Multiple Instance Learning) attention:

**Sentence-level mode (hierarchical=False, default):**
1. Split text into sentences
2. Encode each sentence with a tiny BERT (e.g., prajjwal1/bert-tiny), taking the [CLS] token
3. Apply gated MIL attention with K learnable confounder queries
4. Task-specific weighting of confounders (propensity, tau/y0, outcome/y1)
5. Concatenate and project to output dimension

**Token-level mode (hierarchical=True):**
1. Split text into sentences
2. Encode each sentence with BERT, keeping ALL token embeddings
3. Apply token-level gated pooling to create K confounder-specific sentence representations
4. Apply sentence-level gated MIL attention over the K-view representations
5. Task-specific weighting of confounders
6. Concatenate and project to output dimension

Token-level mode preserves fine-grained distinctions that [CLS] embeddings may lose,
such as "ECOG PS 0" vs "ECOG PS 2" or "no metastatic disease" vs "metastatic disease".

Key insight: Confounders are patient characteristics (metastatic sites, performance status)
that affect both treatment and outcome. The same K confounders feed into all tasks,
but each task can weight them differently:
- Propensity: "Which confounders predict treatment?"
- Tau: "Which confounders modify treatment effect?"
- Outcome: "Which confounders predict baseline outcome?"

Sentence-level Architecture:
    Long Clinical Text
            |
    Split into Sentences (S sentences)
            |
    Tiny BERT per Sentence -> [CLS] token (S x D)
            |
    Gated MIL Attention with K Confounder Queries
            |
    K Confounder Representations (K x D)
            |
    Task-Specific Weighting -> (3 x D)
            |
    MLP Projection -> Final Representation

Token-level Architecture (hierarchical=True):
    Long Clinical Text
            |
    Split into Sentences (S sentences)
            |
    Tiny BERT per Sentence -> ALL tokens (S x L x D)
            |
    Token-Level Gated Pooling (K queries per sentence)
            |
    S x K confounder-specific sentence embeddings
            |
    Sentence-Level Gated MIL Attention (per confounder view)
            |
    K Confounder Representations (K x D)
            |
    Task-Specific Weighting -> (3 x D)
            |
    MLP Projection -> Final Representation

References:
- Ilse et al. (2018): "Attention-based Deep Multiple Instance Learning"
- Lu et al. (2021): "Data-efficient and weakly supervised computational pathology"
"""

import logging
import math
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .confounder_extractor import split_into_sentences
from .gated_mil_attention import GatedMILAttention, TaskSpecificConfounderWeighting, TokenLevelGatedPooling


logger = logging.getLogger(__name__)


class GatedMILHierarchicalExtractor(nn.Module):
    """
    Hierarchical feature extractor using gated MIL attention.

    Combines:
    - Sentence-level BERT encoding (tiny BERT for efficiency)
    - Gated MIL attention with K learnable confounder queries
    - Task-specific weighting of shared confounders
    - Optional token-level gated pooling for fine-grained signal preservation

    Args:
        sentence_encoder_model: HuggingFace model name for sentence encoding
        freeze_sentence_encoder: Whether to freeze the sentence encoder weights
        max_sentences: Maximum number of sentences to process per document
        max_sentence_length: Maximum tokens per sentence for BERT encoding
        mil_hidden_dim: Hidden dimension for gated MIL attention
        num_confounders: Number of confounder queries (K)
        model_type: "rlearner" or "dragonnet"
        projection_dim: Final output dimension
        dropout: Dropout rate
        hierarchical: Whether to use token-level gated pooling (preserves fine-grained signal)
        token_hidden_dim: Hidden dimension for token-level gated attention
        device: PyTorch device
    """

    def __init__(
        self,
        sentence_encoder_model: str = "prajjwal1/bert-tiny",
        freeze_sentence_encoder: bool = True,
        max_sentences: int = 100,
        max_sentence_length: int = 128,
        mil_hidden_dim: int = 128,
        num_confounders: int = 4,
        model_type: str = "rlearner",
        projection_dim: int = 128,
        dropout: float = 0.1,
        hierarchical: bool = False,
        token_hidden_dim: int = 64,
        use_mean_pooling: bool = False,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self._device = device or torch.device('cpu')
        self._sentence_encoder_model = sentence_encoder_model
        self._freeze = freeze_sentence_encoder
        self._max_sentences = max_sentences
        self._max_sentence_length = max_sentence_length
        self._mil_hidden_dim = mil_hidden_dim
        self._num_confounders = num_confounders
        self._model_type = model_type
        self._projection_dim = projection_dim
        self._dropout = dropout
        self._hierarchical = hierarchical
        self._token_hidden_dim = token_hidden_dim
        self._use_mean_pooling = use_mean_pooling

        # Lazy initialization
        self._sentence_encoder = None
        self._tokenizer = None
        self._sentence_dim = None
        self._gated_mil_attention = None
        self._task_weighting = None
        self._output_projection = None
        self._token_level_pooling = None  # Only used when hierarchical=True
        self._initialized = False

        logger.info(f"GatedMILHierarchicalExtractor initialized:")
        logger.info(f"  Sentence encoder: {sentence_encoder_model}")
        logger.info(f"  Freeze encoder: {freeze_sentence_encoder}")
        logger.info(f"  Num confounders: {num_confounders}")
        logger.info(f"  MIL hidden dim: {mil_hidden_dim}")
        logger.info(f"  Model type: {model_type}")
        logger.info(f"  Projection dim: {projection_dim}")
        logger.info(f"  Hierarchical (token-level): {hierarchical}")
        logger.info(f"  Use mean pooling (not [CLS]): {use_mean_pooling}")
        if hierarchical:
            logger.info(f"  Token hidden dim: {token_hidden_dim}")

    def _ensure_initialized(self):
        """Lazily initialize components."""
        if self._initialized:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading sentence encoder: {self._sentence_encoder_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._sentence_encoder_model)
        self._sentence_encoder = AutoModel.from_pretrained(self._sentence_encoder_model)
        self._sentence_encoder = self._sentence_encoder.to(self._device)
        self._sentence_dim = self._sentence_encoder.config.hidden_size
        logger.info(f"  Sentence encoder dim: {self._sentence_dim}")

        if self._freeze:
            for param in self._sentence_encoder.parameters():
                param.requires_grad = False
            logger.info("  Sentence encoder frozen")

        # Token-level gated pooling (only for hierarchical mode)
        if self._hierarchical:
            self._token_level_pooling = TokenLevelGatedPooling(
                input_dim=self._sentence_dim,
                hidden_dim=self._token_hidden_dim,
                num_confounders=self._num_confounders,
                dropout=self._dropout
            ).to(self._device)
            logger.info(f"  Token-level pooling initialized: hidden_dim={self._token_hidden_dim}")

        # Gated MIL attention
        self._gated_mil_attention = GatedMILAttention(
            input_dim=self._sentence_dim,
            hidden_dim=self._mil_hidden_dim,
            num_confounders=self._num_confounders,
            dropout=self._dropout
        ).to(self._device)

        # Task-specific weighting
        self._task_weighting = TaskSpecificConfounderWeighting(
            confounder_dim=self._sentence_dim,
            num_confounders=self._num_confounders,
            model_type=self._model_type
        ).to(self._device)

        # Output projection: 3 * sentence_dim -> projection_dim
        # (propensity_repr || tau_repr || outcome_repr) -> final output
        self._output_projection = nn.Sequential(
            nn.Linear(3 * self._sentence_dim, self._projection_dim * 2),
            nn.LayerNorm(self._projection_dim * 2),
            nn.GELU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._projection_dim * 2, self._projection_dim),
            nn.LayerNorm(self._projection_dim)
        ).to(self._device)

        self._initialized = True
        logger.info("GatedMILHierarchicalExtractor initialization complete")

    @property
    def output_dim(self) -> int:
        """Return the output dimension of this feature extractor."""
        return self._projection_dim

    def _encode_sentences_batch(self, sentences: List[str]) -> torch.Tensor:
        """
        Encode sentences with BERT, returning [CLS] tokens or mean-pooled embeddings.

        When use_mean_pooling=True, computes mean over all tokens (excluding padding)
        instead of using [CLS] token. This can provide more robust sentence
        representations that capture the full sentence content.

        Args:
            sentences: List of sentence strings

        Returns:
            Tensor of shape (num_sentences, sentence_dim) containing sentence embeddings
        """
        if not sentences:
            self._ensure_initialized()
            return torch.zeros(0, self._sentence_dim, device=self._device)

        encoded = self._tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self._max_sentence_length,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        with torch.set_grad_enabled(not self._freeze):
            outputs = self._sentence_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        if self._use_mean_pooling:
            # Mean pooling: average over all tokens (excluding padding)
            # outputs.last_hidden_state: (batch, seq_len, hidden_dim)
            # attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding
            token_embeddings = outputs.last_hidden_state
            # Expand mask for broadcasting: (batch, seq_len, 1)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            # Sum embeddings for valid tokens
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            # Count valid tokens
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            # Mean pooling
            return sum_embeddings / sum_mask
        else:
            # [CLS] token at position 0
            return outputs.last_hidden_state[:, 0, :]

    def _encode_sentence_tokens_batch(
        self,
        sentences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sentences with BERT, returning ALL token embeddings.

        Used in hierarchical mode to preserve fine-grained token signal.

        Args:
            sentences: List of sentence strings

        Returns:
            Tuple of:
                - token_embeddings: (num_sentences, max_len, sentence_dim)
                - attention_mask: (num_sentences, max_len) boolean mask
        """
        if not sentences:
            self._ensure_initialized()
            return (
                torch.zeros(0, self._max_sentence_length, self._sentence_dim, device=self._device),
                torch.zeros(0, self._max_sentence_length, dtype=torch.bool, device=self._device)
            )

        encoded = self._tokenizer(
            sentences,
            padding='max_length',  # Pad to max length for consistent tensor shapes
            truncation=True,
            max_length=self._max_sentence_length,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        with torch.set_grad_enabled(not self._freeze):
            outputs = self._sentence_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Return all token embeddings and the mask
        return outputs.last_hidden_state, attention_mask.bool()

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from texts.

        Args:
            texts: List of document texts

        Returns:
            Feature tensor of shape (batch_size, projection_dim)
        """
        self._ensure_initialized()
        batch_outputs = []

        for text in texts:
            # 1. Split into sentences
            sentences = split_into_sentences(text, self._max_sentences)
            if not sentences:
                sentences = [text[:500]]  # Fallback for short/malformed text

            if self._hierarchical:
                # Hierarchical (token-level) mode
                combined = self._forward_hierarchical(sentences)
            else:
                # Standard sentence-level mode
                combined = self._forward_sentence_level(sentences)

            batch_outputs.append(combined)

        # Stack batch
        batch_outputs = torch.stack(batch_outputs)  # (B, 3 * sentence_dim)

        # Project to output dimension
        features = self._output_projection(batch_outputs)  # (B, projection_dim)

        return features

    def _forward_sentence_level(self, sentences: List[str]) -> torch.Tensor:
        """
        Standard sentence-level forward pass using [CLS] tokens.

        Args:
            sentences: List of sentence strings

        Returns:
            Combined representation of shape (3 * sentence_dim,)
        """
        # Encode sentences with BERT [CLS]
        sentence_embeddings = self._encode_sentences_batch(sentences)  # (S, sentence_dim)

        # Apply gated MIL attention to get K confounders
        confounders, _ = self._gated_mil_attention(sentence_embeddings)  # (K, sentence_dim)

        # Apply task-specific weighting
        prop_repr, task2_repr, task3_repr = self._task_weighting(confounders)
        # Each is (sentence_dim,)

        # Concatenate task representations
        return torch.cat([prop_repr, task2_repr, task3_repr], dim=0)  # (3 * sentence_dim,)

    def _forward_hierarchical(self, sentences: List[str]) -> torch.Tensor:
        """
        Hierarchical forward pass with token-level gated pooling.

        Each confounder query attends to tokens within sentences to create
        confounder-specific sentence representations, then sentence-level
        gated attention aggregates these into K confounders.

        Args:
            sentences: List of sentence strings

        Returns:
            Combined representation of shape (3 * sentence_dim,)
        """
        S = len(sentences)
        K = self._num_confounders

        # 1. Encode all sentences with BERT (full token embeddings)
        token_embeddings, attention_mask = self._encode_sentence_tokens_batch(sentences)
        # token_embeddings: (S, L, D)
        # attention_mask: (S, L) boolean

        # 2. Apply token-level gated pooling to each sentence
        # Each confounder query produces its own sentence representation
        # Result: (S, K, D) - K confounder-specific representations per sentence
        confounder_sentence_embeddings, _ = self._token_level_pooling.forward_batch(
            token_embeddings, attention_mask
        )

        # 3. For each confounder k, apply sentence-level gated MIL attention
        # over that confounder's view of the sentences
        # This is equivalent to K parallel applications of sentence-level attention

        # Rearrange to (K, S, D) for per-confounder attention
        confounder_views = confounder_sentence_embeddings.permute(1, 0, 2)  # (K, S, D)

        # Apply sentence-level gated attention for each confounder view
        # We process each confounder's view through the shared gated MIL attention
        all_confounders = []
        for k in range(K):
            view_k = confounder_views[k]  # (S, D) - confounder k's view of all sentences
            # Use the gated MIL attention to aggregate this view
            # Note: We only take the k-th output to avoid redundancy
            conf_k, _ = self._gated_mil_attention(view_k)  # (K, D)
            # Since all K queries attend to the same view, we take the mean
            # to collapse to a single representation for this confounder
            all_confounders.append(conf_k.mean(dim=0))  # (D,)

        confounders = torch.stack(all_confounders, dim=0)  # (K, D)

        # 4. Apply task-specific weighting
        prop_repr, task2_repr, task3_repr = self._task_weighting(confounders)
        # Each is (sentence_dim,)

        # 5. Concatenate task representations
        return torch.cat([prop_repr, task2_repr, task3_repr], dim=0)  # (3 * sentence_dim,)

    def init_extractor(self, texts: List[str]) -> 'GatedMILHierarchicalExtractor':
        """
        Initialize the feature extractor (triggers lazy initialization).

        For GatedMILHierarchicalExtractor, this loads the pretrained sentence
        encoder and initializes the gated MIL attention components. The texts
        argument is not used since we use pretrained tokenizers.

        Args:
            texts: List of training text strings (not used, kept for API compatibility)

        Returns:
            self for method chaining
        """
        self._ensure_initialized()
        return self

    def fit_tokenizer(self, texts: List[str]) -> 'GatedMILHierarchicalExtractor':
        """Alias for init_extractor() for backward compatibility."""
        return self.init_extractor(texts)

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if self._sentence_encoder is not None:
            self._sentence_encoder = self._sentence_encoder.to(self._device)
        if self._gated_mil_attention is not None:
            self._gated_mil_attention = self._gated_mil_attention.to(self._device)
        if self._task_weighting is not None:
            self._task_weighting = self._task_weighting.to(self._device)
        if self._output_projection is not None:
            self._output_projection = self._output_projection.to(self._device)
        if self._token_level_pooling is not None:
            self._token_level_pooling = self._token_level_pooling.to(self._device)

        return super().to(device)

    def get_state(self) -> Dict[str, Any]:
        """
        Get extractor state for checkpoint saving.

        Returns:
            Dictionary containing configuration for reconstruction
        """
        return {
            'sentence_encoder_model': self._sentence_encoder_model,
            'freeze_sentence_encoder': self._freeze,
            'max_sentences': self._max_sentences,
            'max_sentence_length': self._max_sentence_length,
            'mil_hidden_dim': self._mil_hidden_dim,
            'num_confounders': self._num_confounders,
            'model_type': self._model_type,
            'projection_dim': self._projection_dim,
            'dropout': self._dropout,
            'hierarchical': self._hierarchical,
            'token_hidden_dim': self._token_hidden_dim,
            'use_mean_pooling': self._use_mean_pooling
        }

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5,
        include_token_attention: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of gated MIL attention.

        This extracts attention weights from each confounder query to sentences,
        showing which sentences each confounder focuses on, plus the task-specific
        weights for each confounder.

        For hierarchical (token-level) mode, also includes:
        - Token-level attention weights within each sentence
        - Which specific tokens each confounder attends to
        - Attention entropy metrics (low = focused, high = diffuse)

        Args:
            texts: List of document texts
            top_k: Number of top-attended sentences to show per confounder
            include_token_attention: Whether to include token-level attention (only for hierarchical mode)

        Returns:
            List of dicts per document with attention interpretations:
            - 'sentences': List of sentence strings
            - 'confounder_attention': Dict mapping confounder index to attention weights
            - 'top_sentences_per_confounder': Top-k sentences per confounder
            - 'task_weights': Task-specific weights for each confounder
            - 'attention_entropy': Dict with entropy metrics per confounder (lower = more focused)
            - 'token_attention' (hierarchical only): Token-level attention info
        """
        self._ensure_initialized()
        interpretations = []

        # Get task weights (shared across all documents)
        task_weights = self._task_weighting.get_weights()

        with torch.no_grad():
            for text in texts:
                sentences = split_into_sentences(text, self._max_sentences)
                if not sentences:
                    sentences = [text[:500]]

                if self._hierarchical and include_token_attention:
                    # Hierarchical (token-level) mode
                    interp = self._interpret_hierarchical(sentences, top_k, task_weights)
                else:
                    # Standard sentence-level mode
                    interp = self._interpret_sentence_level(sentences, top_k, task_weights)

                interpretations.append(interp)

        return interpretations

    def _interpret_sentence_level(
        self,
        sentences: List[str],
        top_k: int,
        task_weights: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Interpret attention for sentence-level mode."""
        # Encode sentences
        sentence_embeddings = self._encode_sentences_batch(sentences)

        # Get attention weights
        _, attention_weights = self._gated_mil_attention(
            sentence_embeddings, return_attention=True
        )  # (K, S)

        if attention_weights is not None and len(sentences) > 0:
            confounder_attention = {}
            top_sentences_per_confounder = {}
            attention_entropy = {}

            for k in range(self._num_confounders):
                attn = attention_weights[k].cpu()  # (S,)
                confounder_attention[f'confounder_{k}'] = attn.tolist()

                # Compute attention entropy (lower = more focused)
                entropy = self._compute_attention_entropy(attn)
                attention_entropy[f'confounder_{k}'] = entropy

                # Get top-k sentences for this confounder
                k_actual = min(top_k, len(sentences))
                top_vals, top_indices = torch.topk(attn, k_actual)

                top_sentences_per_confounder[f'confounder_{k}'] = [
                    {
                        'sentence': sentences[idx],
                        'attention': val.item(),
                        'idx': int(idx)
                    }
                    for val, idx in zip(top_vals, top_indices)
                ]

            return {
                'sentences': sentences,
                'confounder_attention': confounder_attention,
                'top_sentences_per_confounder': top_sentences_per_confounder,
                'task_weights': task_weights,
                'attention_entropy': attention_entropy,
                'hierarchical_mode': False
            }
        else:
            return {
                'sentences': sentences,
                'confounder_attention': {},
                'top_sentences_per_confounder': {},
                'task_weights': task_weights,
                'attention_entropy': {},
                'hierarchical_mode': False
            }

    def _interpret_hierarchical(
        self,
        sentences: List[str],
        top_k: int,
        task_weights: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Interpret attention for hierarchical (token-level) mode."""
        S = len(sentences)
        K = self._num_confounders

        # 1. Encode all sentences with BERT (full token embeddings)
        token_embeddings, attention_mask = self._encode_sentence_tokens_batch(sentences)
        # token_embeddings: (S, L, D)
        # attention_mask: (S, L) boolean

        # 2. Apply token-level gated pooling with attention returned
        confounder_sentence_embeddings, token_attention = self._token_level_pooling.forward_batch(
            token_embeddings, attention_mask, return_attention=True
        )  # confounder_sentence_embeddings: (S, K, D), token_attention: (S, K, L)

        # 3. Get sentence-level attention for each confounder view
        confounder_views = confounder_sentence_embeddings.permute(1, 0, 2)  # (K, S, D)

        sentence_attention_per_confounder = []
        for k in range(K):
            view_k = confounder_views[k]  # (S, D)
            _, sent_attn = self._gated_mil_attention(view_k, return_attention=True)
            sentence_attention_per_confounder.append(sent_attn.mean(dim=0).cpu())  # (S,)

        # Build interpretation
        confounder_attention = {}
        top_sentences_per_confounder = {}
        attention_entropy = {}
        token_attention_info = {}

        # Get tokenized sentences for token-level interpretation
        encoded = self._tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=self._max_sentence_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids']

        for k in range(K):
            # Sentence-level attention
            sent_attn = sentence_attention_per_confounder[k]  # (S,)
            confounder_attention[f'confounder_{k}'] = sent_attn.tolist()

            # Compute sentence-level entropy
            entropy = self._compute_attention_entropy(sent_attn)
            attention_entropy[f'confounder_{k}'] = {
                'sentence_entropy': entropy,
                'token_entropy_mean': 0.0  # Will compute below
            }

            # Get top-k sentences
            k_actual = min(top_k, S)
            top_vals, top_indices = torch.topk(sent_attn, k_actual)

            top_sentences_data = []
            token_entropies = []

            for val, sent_idx in zip(top_vals, top_indices):
                sent_idx = int(sent_idx)

                # Get token attention for this sentence and confounder
                if token_attention is not None:
                    tok_attn = token_attention[sent_idx, k, :].cpu()  # (L,)
                    valid_mask = attention_mask[sent_idx].cpu()

                    # Get top tokens
                    valid_tok_attn = tok_attn.clone()
                    valid_tok_attn[~valid_mask] = 0.0

                    # Decode tokens to get actual words
                    tokens = self._tokenizer.convert_ids_to_tokens(
                        input_ids[sent_idx].tolist()
                    )

                    # Get top attended tokens (excluding padding)
                    top_token_vals, top_token_indices = torch.topk(
                        valid_tok_attn, min(5, valid_mask.sum().item())
                    )

                    top_tokens = [
                        {
                            'token': tokens[int(idx)],
                            'attention': float(val_t),
                            'position': int(idx)
                        }
                        for val_t, idx in zip(top_token_vals, top_token_indices)
                        if tokens[int(idx)] not in ['[PAD]', '[CLS]', '[SEP]']
                    ]

                    # Compute token-level entropy for valid tokens
                    valid_attn = tok_attn[valid_mask]
                    if len(valid_attn) > 0:
                        tok_entropy = self._compute_attention_entropy(valid_attn)
                        token_entropies.append(tok_entropy)
                else:
                    top_tokens = []

                top_sentences_data.append({
                    'sentence': sentences[sent_idx],
                    'attention': float(val),
                    'idx': sent_idx,
                    'top_tokens': top_tokens
                })

            top_sentences_per_confounder[f'confounder_{k}'] = top_sentences_data

            # Update token entropy mean
            if token_entropies:
                attention_entropy[f'confounder_{k}']['token_entropy_mean'] = float(np.mean(token_entropies))

            # Store full token attention for analysis
            if token_attention is not None:
                token_attention_info[f'confounder_{k}'] = {
                    'shape': list(token_attention[:, k, :].shape),
                    'mean_attention': float(token_attention[:, k, :].mean()),
                    'max_attention': float(token_attention[:, k, :].max())
                }

        return {
            'sentences': sentences,
            'confounder_attention': confounder_attention,
            'top_sentences_per_confounder': top_sentences_per_confounder,
            'task_weights': task_weights,
            'attention_entropy': attention_entropy,
            'token_attention_info': token_attention_info,
            'hierarchical_mode': True
        }

    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """
        Compute normalized entropy of attention distribution.

        Lower entropy = more focused attention (good for needle-in-haystack)
        Higher entropy = more diffuse attention (bad, spreading attention everywhere)

        Args:
            attention_weights: 1D tensor of attention weights (should sum to ~1)

        Returns:
            Normalized entropy in [0, 1] where 0 = completely focused, 1 = uniform
        """
        eps = 1e-8
        attn = attention_weights.clamp(min=eps)

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(attn * torch.log(attn))

        # Normalize by max entropy (uniform distribution)
        n = len(attention_weights)
        if n > 1:
            max_entropy = math.log(n)
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = torch.tensor(0.0)

        return float(normalized_entropy)

    def get_attention_weights(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get raw attention weights for visualization.

        Args:
            texts: List of document texts

        Returns:
            Dictionary with interpretations and model metadata
        """
        interpretations = self.interpret_attention(texts, top_k=self._max_sentences)
        return {
            'interpretations': interpretations,
            'num_confounders': self._num_confounders,
            'model_type': self._model_type,
            'sentence_model': self._sentence_encoder_model
        }

    def get_task_weights(self) -> Dict[str, List[float]]:
        """
        Get the task-specific confounder weights for interpretability.

        Returns:
            Dictionary with normalized weights per task:
            - 'propensity': [w1, w2, ..., wK] weights for propensity prediction
            - 'tau' or 'y0': [w1, w2, ..., wK] weights for tau/y0 prediction
            - 'outcome' or 'y1': [w1, w2, ..., wK] weights for outcome/y1 prediction
        """
        self._ensure_initialized()
        return self._task_weighting.get_weights()

    def forward_with_attention(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features and return attention weights for regularization.

        This is used during training when attention_entropy_weight > 0 to compute
        entropy regularization loss that penalizes diffuse attention.

        Args:
            texts: List of document texts

        Returns:
            features: Feature tensor of shape (batch_size, projection_dim)
            attention_info: Dict with attention weights for entropy computation:
                - 'sentence_attention': (batch, K, S) sentence-level attention
                - 'token_attention': (batch, K, S, L) token-level attention (hierarchical only)
                - 'attention_entropy': Mean normalized entropy across batch (scalar)
        """
        self._ensure_initialized()
        batch_outputs = []
        all_sentence_attention = []
        all_token_attention = []
        all_entropies = []

        for text in texts:
            # 1. Split into sentences
            sentences = split_into_sentences(text, self._max_sentences)
            if not sentences:
                sentences = [text[:500]]

            if self._hierarchical:
                # Hierarchical (token-level) mode
                combined, sent_attn, tok_attn, entropy = self._forward_hierarchical_with_attention(sentences)
                all_token_attention.append(tok_attn)
            else:
                # Standard sentence-level mode
                combined, sent_attn, entropy = self._forward_sentence_level_with_attention(sentences)

            batch_outputs.append(combined)
            all_sentence_attention.append(sent_attn)
            all_entropies.append(entropy)

        # Stack batch
        batch_outputs = torch.stack(batch_outputs)  # (B, 3 * sentence_dim)

        # Project to output dimension
        features = self._output_projection(batch_outputs)  # (B, projection_dim)

        # Compute mean entropy
        mean_entropy = torch.stack(all_entropies).mean()

        attention_info = {
            'attention_entropy': mean_entropy,
            'sentence_attention': all_sentence_attention,  # List of (K, S) tensors
        }

        if self._hierarchical and all_token_attention:
            attention_info['token_attention'] = all_token_attention

        return features, attention_info

    def _forward_sentence_level_with_attention(
        self,
        sentences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sentence-level forward pass returning attention for regularization.

        Returns:
            combined: (3 * sentence_dim,) combined task representations
            attention: (K, S) attention weights
            entropy: Scalar mean entropy across confounders
        """
        # Encode sentences
        sentence_embeddings = self._encode_sentences_batch(sentences)  # (S, D)

        # Get attention weights
        confounders, attention_weights = self._gated_mil_attention(
            sentence_embeddings, return_attention=True
        )  # confounders: (K, D), attention: (K, S)

        # Compute entropy for regularization (lower = more focused)
        K = self._num_confounders
        entropies = []
        for k in range(K):
            attn = attention_weights[k]
            # Entropy: -sum(p * log(p))
            eps = 1e-8
            entropy_k = -torch.sum(attn * torch.log(attn + eps))
            # Normalize by max entropy
            max_entropy = math.log(len(sentences)) if len(sentences) > 1 else 1.0
            normalized_entropy_k = entropy_k / max_entropy
            entropies.append(normalized_entropy_k)

        mean_entropy = torch.stack(entropies).mean()

        # Apply task-specific weighting
        prop_repr, task2_repr, task3_repr = self._task_weighting(confounders)

        combined = torch.cat([prop_repr, task2_repr, task3_repr], dim=0)

        return combined, attention_weights, mean_entropy

    def _forward_hierarchical_with_attention(
        self,
        sentences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hierarchical forward pass returning attention for regularization.

        Returns:
            combined: (3 * sentence_dim,) combined task representations
            sentence_attention: (K, S) sentence-level attention
            token_attention: (S, K, L) token-level attention
            entropy: Scalar mean entropy across confounders
        """
        S = len(sentences)
        K = self._num_confounders

        # 1. Encode all sentences with BERT (full token embeddings)
        token_embeddings, attention_mask = self._encode_sentence_tokens_batch(sentences)

        # 2. Apply token-level gated pooling
        confounder_sentence_embeddings, token_attention = self._token_level_pooling.forward_batch(
            token_embeddings, attention_mask, return_attention=True
        )

        # 3. For each confounder, apply sentence-level attention
        confounder_views = confounder_sentence_embeddings.permute(1, 0, 2)  # (K, S, D)

        all_confounders = []
        all_sentence_attentions = []
        entropies = []

        for k in range(K):
            view_k = confounder_views[k]  # (S, D)
            conf_k, sent_attn_k = self._gated_mil_attention(view_k, return_attention=True)
            all_confounders.append(conf_k.mean(dim=0))  # (D,)

            # Aggregate sentence attention across queries
            sent_attn_mean = sent_attn_k.mean(dim=0)  # (S,)
            all_sentence_attentions.append(sent_attn_mean)

            # Compute entropy
            eps = 1e-8
            entropy_k = -torch.sum(sent_attn_mean * torch.log(sent_attn_mean + eps))
            max_entropy = math.log(S) if S > 1 else 1.0
            normalized_entropy_k = entropy_k / max_entropy
            entropies.append(normalized_entropy_k)

            # Also add token-level entropy
            if token_attention is not None:
                for s in range(S):
                    tok_attn = token_attention[s, k, :]
                    valid_len = attention_mask[s].sum().item()
                    if valid_len > 1:
                        valid_attn = tok_attn[:valid_len]
                        tok_entropy = -torch.sum(valid_attn * torch.log(valid_attn + eps))
                        tok_max_entropy = math.log(valid_len)
                        entropies.append(tok_entropy / tok_max_entropy)

        confounders = torch.stack(all_confounders, dim=0)  # (K, D)
        sentence_attention = torch.stack(all_sentence_attentions, dim=0)  # (K, S)
        mean_entropy = torch.stack(entropies).mean()

        # Apply task-specific weighting
        prop_repr, task2_repr, task3_repr = self._task_weighting(confounders)
        combined = torch.cat([prop_repr, task2_repr, task3_repr], dim=0)

        return combined, sentence_attention, token_attention, mean_entropy

    def compute_attention_entropy_loss(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """
        Compute attention entropy loss for regularization.

        This loss penalizes high-entropy (diffuse) attention distributions,
        encouraging the model to focus on specific sentences rather than
        spreading attention uniformly.

        Lower entropy = more focused attention = lower loss

        Args:
            texts: List of document texts

        Returns:
            entropy_loss: Mean normalized entropy across batch (scalar in [0, 1])
        """
        _, attention_info = self.forward_with_attention(texts)
        return attention_info['attention_entropy']
