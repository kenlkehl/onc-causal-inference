# cdt/models/gru_extractor.py
"""Feature extractor using bidirectional GRU with attention pooling."""

import logging
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_extractor import WordTokenizer  # Reuse word-level tokenizer

logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """
    Attention pooling layer that produces a fixed-size representation from variable-length sequences.

    Uses a learned query vector to compute attention weights over sequence positions,
    then returns a weighted sum of hidden states. This is O(N) in sequence length,
    unlike full self-attention which is O(N^2).

    Attention formula:
        scores = tanh(W_h * h + b) @ v  # (batch, seq_len)
        weights = softmax(scores)       # (batch, seq_len)
        output = sum(weights * h)       # (batch, hidden_dim)
    """

    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None):
        """
        Initialize attention pooling.

        Args:
            hidden_dim: Dimension of input hidden states
            attention_dim: Dimension of attention hidden layer (default: hidden_dim)
        """
        super().__init__()

        attention_dim = attention_dim or hidden_dim

        # Project hidden states to attention space
        self.W = nn.Linear(hidden_dim, attention_dim, bias=True)

        # Learned query vector
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) - sequence of hidden states
            attention_mask: (batch, seq_len) - 1 for valid positions, 0 for padding

        Returns:
            pooled: (batch, hidden_dim) - attention-weighted representation
        """
        # Compute attention scores: (batch, seq_len, attention_dim) -> (batch, seq_len, 1)
        scores = self.v(torch.tanh(self.W(hidden_states)))
        scores = scores.squeeze(-1)  # (batch, seq_len)

        # Apply mask if provided
        if attention_mask is not None:
            # Set padding positions to large negative value before softmax
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Compute attention weights
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum of hidden states
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_dim)

        return pooled


class GRUFeatureExtractor(nn.Module):
    """
    Feature extractor using bidirectional GRU with attention pooling.

    Architecture:
    1. Word embeddings (can be initialized from BERT)
    2. Bidirectional GRU (O(N) in sequence length)
    3. Attention pooling with learned query (O(N) in sequence length)
    4. Optional projection layer

    Total complexity: O(N) where N is sequence length.
    This is much more efficient than BERT's O(N^2) self-attention for long sequences.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        attention_dim: Optional[int] = None,
        projection_dim: Optional[int] = 128,
        max_length: int = 8192,
        min_word_freq: int = 2,
        max_vocab_size: Optional[int] = 50000,
        device: torch.device = None
    ):
        """
        Initialize GRU feature extractor.

        Args:
            embedding_dim: Dimension of word embeddings
            hidden_dim: GRU hidden state dimension (per direction)
            num_layers: Number of stacked GRU layers
            dropout: Dropout rate
            bidirectional: Use bidirectional GRU
            attention_dim: Dimension of attention hidden layer (default: 2*hidden_dim if bidirectional)
            projection_dim: Output projection dimension (None to skip projection)
            max_length: Maximum sequence length in tokens
            min_word_freq: Minimum word frequency for vocabulary inclusion
            max_vocab_size: Maximum vocabulary size
            device: Device for computation
        """
        super().__init__()

        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.projection_dim = projection_dim

        # Word tokenizer (same as CNN extractor)
        self.tokenizer = WordTokenizer(
            max_length=max_length,
            min_freq=min_word_freq,
            max_vocab_size=max_vocab_size
        )

        # Embedding layer (initialized after tokenizer is fitted)
        self.embedding = None

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention pooling over GRU outputs
        gru_output_dim = hidden_dim * self.num_directions
        self.attention = AttentionPooling(
            hidden_dim=gru_output_dim,
            attention_dim=attention_dim
        )

        # Optional projection layer
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(gru_output_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self._output_dim = projection_dim
        else:
            self.projection = None
            self._output_dim = gru_output_dim

        # Layer norm for embeddings
        self.embed_layer_norm = nn.LayerNorm(embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)

        logger.info(f"GRUFeatureExtractor initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  GRU hidden dim: {hidden_dim} x {self.num_directions} directions")
        logger.info(f"  GRU layers: {num_layers}")
        logger.info(f"  Output dim: {self._output_dim}")
        logger.info(f"  Max sequence length: {max_length}")

    @property
    def output_dim(self) -> int:
        """Return output feature dimension."""
        return self._output_dim

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size

    def fit_tokenizer(self, texts: List[str]) -> 'GRUFeatureExtractor':
        """
        Fit tokenizer on training texts and initialize embedding layer.

        MUST be called before using the model for training or inference.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        self.tokenizer.fit(texts)

        # Initialize embedding layer with vocabulary size
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.tokenizer.pad_token
        )
        self.embedding.to(self._device)

        logger.info(f"Tokenizer fitted: vocab size = {self.tokenizer.vocab_size}")

        return self

    def init_embeddings_from_bert(
        self,
        bert_model_name: str = "bert-base-uncased",
        freeze: bool = False
    ):
        """
        Initialize word embeddings from a BERT model.

        Maps vocabulary words to BERT subword tokens and averages their embeddings.

        Args:
            bert_model_name: HuggingFace BERT model name
            freeze: Whether to freeze embeddings after initialization
        """
        if self.embedding is None:
            raise RuntimeError("Must call fit_tokenizer() before init_embeddings_from_bert()")

        from transformers import AutoTokenizer, AutoModel

        logger.info(f"Initializing embeddings from {bert_model_name}...")

        bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert_model = AutoModel.from_pretrained(bert_model_name)
        bert_embeddings = bert_model.get_input_embeddings().weight.data

        # Get BERT embedding dimension
        bert_dim = bert_embeddings.shape[1]

        # If dimensions don't match, we'll need a projection
        if bert_dim != self.embedding_dim:
            logger.warning(f"BERT embedding dim ({bert_dim}) != model embedding dim ({self.embedding_dim})")
            logger.warning("Using random projection to match dimensions")
            projection = torch.randn(bert_dim, self.embedding_dim) / (bert_dim ** 0.5)
        else:
            projection = None

        # Map each word in vocabulary to BERT embeddings
        with torch.no_grad():
            for word, idx in self.tokenizer.word_to_id.items():
                if word in ["<PAD>", "<UNK>"]:
                    continue

                # Tokenize word with BERT
                bert_tokens = bert_tokenizer.encode(word, add_special_tokens=False)

                if bert_tokens:
                    # Average BERT subword embeddings
                    word_embedding = bert_embeddings[bert_tokens].mean(dim=0)

                    if projection is not None:
                        word_embedding = word_embedding @ projection

                    self.embedding.weight.data[idx] = word_embedding

        if freeze:
            self.embedding.weight.requires_grad = False
            logger.info("Embeddings frozen")

        logger.info(f"Initialized {self.tokenizer.vocab_size - 2} word embeddings from BERT")

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from texts.

        Args:
            texts: List of text strings

        Returns:
            features: (batch, output_dim) tensor of features
        """
        if self.embedding is None:
            raise RuntimeError("Must call fit_tokenizer() before forward()")

        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        # Embed tokens
        embeddings = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        embeddings = self.embed_layer_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)

        # Pack padded sequence for efficient GRU processing
        lengths = attention_mask.sum(dim=1).cpu()

        # GRU forward pass
        # Note: We don't use pack_padded_sequence because it requires sorting by length
        # which complicates the attention mask handling. GRU handles padding reasonably well.
        gru_output, _ = self.gru(embeddings)  # (batch, seq_len, hidden_dim * num_directions)

        # Attention pooling
        pooled = self.attention(gru_output, attention_mask)  # (batch, hidden_dim * num_directions)

        # Optional projection
        if self.projection is not None:
            pooled = self.projection(pooled)

        return pooled

    def get_attention_weights(self, texts: List[str]) -> torch.Tensor:
        """
        Get attention weights for interpretability.

        Args:
            texts: List of text strings

        Returns:
            weights: (batch, seq_len) attention weights
        """
        if self.embedding is None:
            raise RuntimeError("Must call fit_tokenizer() before get_attention_weights()")

        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        # Embed and process
        embeddings = self.embedding(input_ids)
        embeddings = self.embed_layer_norm(embeddings)

        gru_output, _ = self.gru(embeddings)

        # Get attention scores before softmax
        scores = self.attention.v(torch.tanh(self.attention.W(gru_output))).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)

        return weights

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 10
    ) -> List[List[tuple]]:
        """
        Get top-k attended tokens for each text.

        Args:
            texts: List of text strings
            top_k: Number of top tokens to return

        Returns:
            List of lists of (token, attention_weight) tuples
        """
        weights = self.get_attention_weights(texts)  # (batch, seq_len)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids']

        results = []
        for i in range(len(texts)):
            # Get top-k positions
            seq_weights = weights[i]
            seq_ids = input_ids[i]

            # Get non-padding positions
            valid_mask = seq_ids != self.tokenizer.pad_token
            valid_weights = seq_weights[valid_mask]
            valid_ids = seq_ids[valid_mask]

            # Sort by attention weight
            sorted_indices = torch.argsort(valid_weights, descending=True)[:top_k]

            top_tokens = []
            for idx in sorted_indices:
                token_id = valid_ids[idx].item()
                weight = valid_weights[idx].item()
                token = self.tokenizer.id_to_word.get(token_id, "<UNK>")
                top_tokens.append((token, weight))

            results.append(top_tokens)

        return results
