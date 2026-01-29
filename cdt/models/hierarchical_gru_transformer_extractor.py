"""Hierarchical GRU Transformer feature extractor using overlapping token chunks + GRU + transformer pooling.

This module implements a hierarchical approach for extracting features from long
clinical text using learned attention instead of pretrained sentence embeddings:

1. Tokenize text into words (using WordTokenizer from cnn_extractor)
2. Split into overlapping chunks (default: 128 tokens, 32 overlap)
3. Encode each chunk with BiGRU + attention pooling
4. Apply transformer layer(s) on top to pool chunk embeddings into a final representation

Key difference from HierarchicalTransformerExtractor:
- Uses overlapping fixed-size chunks instead of sentence boundaries
- BiGRU with learned attention instead of frozen bert-tiny [CLS]
- Guarantees confounder text appears fully in at least one chunk
- Requires fit_tokenizer() to build vocabulary

Architecture:
    Long Clinical Text
            |
    Tokenize to words (WordTokenizer)
            |
    Split into overlapping chunks (K chunks x chunk_size tokens)
            |
    BiGRU + Attention per chunk -> chunk embeddings (K x chunk_dim)
            |
    Transformer Layer(s) with learnable [POOL] token
            |
    Final Representation (D,) -> DragonNet/RLearner
"""

import logging
import math
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_extractor import WordTokenizer


logger = logging.getLogger(__name__)


class GRUChunkEncoder(nn.Module):
    """
    Encodes a chunk of tokens using BiGRU with attention pooling.

    Architecture:
        Token IDs -> Word Embedding -> LayerNorm -> Dropout
        -> BiGRU (bidirectional) -> Attention Pooling -> Projection

    Args:
        vocab_size: Size of vocabulary (set after fit_tokenizer)
        embedding_dim: Dimension of word embeddings
        hidden_dim: Hidden dimension for GRU (output will be 2*hidden_dim for BiGRU)
        num_layers: Number of GRU layers
        chunk_dim: Output dimension after projection
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 100,  # Placeholder, will be updated after fit_tokenizer
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        chunk_dim: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._chunk_dim = chunk_dim
        self._dropout = dropout
        self._padding_idx = padding_idx

        # Word embedding layer (will be rebuilt after fit_tokenizer)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        # Pre-GRU normalization and dropout
        self.embed_norm = nn.LayerNorm(embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention pooling: learned query vector
        gru_output_dim = hidden_dim * 2  # Bidirectional
        self.attention_query = nn.Parameter(torch.randn(gru_output_dim) * 0.02)
        self.attention_key = nn.Linear(gru_output_dim, gru_output_dim)

        # Output projection: BiGRU output -> chunk_dim
        self.projection = nn.Sequential(
            nn.Linear(gru_output_dim, chunk_dim),
            nn.LayerNorm(chunk_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        logger.info(f"GRUChunkEncoder initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  GRU hidden dim: {hidden_dim} x 2 (bidirectional)")
        logger.info(f"  Num GRU layers: {num_layers}")
        logger.info(f"  Chunk dim output: {chunk_dim}")

    def rebuild_embedding(self, vocab_size: int) -> None:
        """Rebuild embedding layer with new vocabulary size."""
        device = self.embedding.weight.device
        self._vocab_size = vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self._embedding_dim,
            padding_idx=self._padding_idx
        ).to(device)
        logger.info(f"GRUChunkEncoder embedding rebuilt with vocab_size={vocab_size}")

    def forward(
        self,
        chunk_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode chunks with BiGRU and attention pooling.

        Args:
            chunk_ids: Token IDs of shape (batch, chunk_size)
            attention_mask: Mask of shape (batch, chunk_size), 1 for valid tokens
            return_attention: Whether to return attention weights

        Returns:
            chunk_embeddings: Shape (batch, chunk_dim)
            attention_weights: Shape (batch, chunk_size) if return_attention, else None
        """
        # Embed tokens: (batch, chunk_size, embedding_dim)
        x = self.embedding(chunk_ids)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        # Pack for GRU efficiency (handles variable length)
        lengths = attention_mask.sum(dim=1).cpu()
        lengths = lengths.clamp(min=1)  # Ensure at least 1 for empty sequences

        # GRU forward: (batch, chunk_size, hidden_dim * 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        gru_output, _ = self.gru(packed)
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(
            gru_output, batch_first=True, total_length=chunk_ids.size(1)
        )

        # Attention pooling
        # Keys: (batch, chunk_size, hidden_dim * 2)
        keys = self.attention_key(gru_output)

        # Attention scores: (batch, chunk_size)
        scores = torch.matmul(keys, self.attention_query)  # (batch, chunk_size)

        # Mask padding positions
        mask = attention_mask.bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, chunk_size)

        # Handle all-padding case
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )

        # Weighted sum: (batch, hidden_dim * 2)
        pooled = torch.bmm(attention_weights.unsqueeze(1), gru_output).squeeze(1)

        # Project to chunk_dim
        chunk_embedding = self.projection(pooled)  # (batch, chunk_dim)

        if return_attention:
            return chunk_embedding, attention_weights
        return chunk_embedding, None


class HierarchicalGRUTransformerExtractor(nn.Module):
    """
    Hierarchical GRU Transformer feature extractor.

    Architecture:
    1. Tokenize text into words
    2. Split into overlapping chunks (chunk_size tokens, chunk_overlap overlap)
    3. Encode each chunk with BiGRU + attention pooling
    4. Apply transformer layer(s) with learnable [POOL] token
    5. Output [POOL] representation for causal head

    Key design choices:
    - Overlapping chunks guarantee confounder text appears in at least one chunk
    - BiGRU learns task-specific attention over tokens
    - Requires fit_tokenizer() before training (builds vocabulary from training text)

    Args:
        chunk_size: Number of tokens per chunk (default: 128)
        chunk_overlap: Overlap between consecutive chunks (default: 32)
        max_chunks: Maximum number of chunks to process per document
        embedding_dim: Dimension of word embeddings
        gru_hidden_dim: Hidden dimension for BiGRU
        gru_num_layers: Number of GRU layers
        chunk_dim: Output dimension of chunk encoder (input to transformer)
        num_transformer_layers: Number of transformer layers for pooling
        num_attention_heads: Number of attention heads in transformer layers
        transformer_dropout: Dropout rate for transformer layers
        projection_dim: Final output dimension
        max_vocab_size: Maximum vocabulary size
        min_word_freq: Minimum word frequency to include in vocabulary
        device: PyTorch device
    """

    def __init__(
        self,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        max_chunks: int = 100,
        embedding_dim: int = 128,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        chunk_dim: int = 256,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        transformer_dropout: float = 0.1,
        projection_dim: int = 128,
        max_vocab_size: int = 50000,
        min_word_freq: int = 2,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self._device = device or torch.device('cpu')
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunk_step = chunk_size - chunk_overlap
        self._max_chunks = max_chunks
        self._embedding_dim = embedding_dim
        self._gru_hidden_dim = gru_hidden_dim
        self._gru_num_layers = gru_num_layers
        self._chunk_dim = chunk_dim
        self._num_transformer_layers = num_transformer_layers
        self._num_attention_heads = num_attention_heads
        self._transformer_dropout = transformer_dropout
        self._projection_dim = projection_dim

        # Word tokenizer (must be fitted before use)
        self.tokenizer = WordTokenizer(
            max_length=chunk_size * max_chunks,  # Max total tokens
            min_freq=min_word_freq,
            max_vocab_size=max_vocab_size
        )

        # GRU chunk encoder
        self.chunk_encoder = GRUChunkEncoder(
            vocab_size=100,  # Placeholder
            embedding_dim=embedding_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            chunk_dim=chunk_dim,
            dropout=transformer_dropout
        )

        # Positional encoding for chunks
        self._register_positional_encoding()

        # Learnable [POOL] token
        self._pool_token = nn.Parameter(
            torch.randn(1, chunk_dim, device=self._device) * 0.02
        )

        # Transformer layers for pooling chunk embeddings
        from .hierarchical_transformer_extractor import InterpretableTransformerLayer
        self._transformer_layers = nn.ModuleList([
            InterpretableTransformerLayer(
                d_model=chunk_dim,
                nhead=num_attention_heads,
                dim_feedforward=chunk_dim * 4,
                dropout=transformer_dropout
            )
            for _ in range(num_transformer_layers)
        ])

        # Output projection
        self._output_projection = nn.Sequential(
            nn.Linear(chunk_dim, chunk_dim),
            nn.LayerNorm(chunk_dim),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(chunk_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        self._initialized = False

        logger.info(f"HierarchicalGRUTransformerExtractor initialized:")
        logger.info(f"  Chunk size: {chunk_size}, overlap: {chunk_overlap}, step: {self._chunk_step}")
        logger.info(f"  Max chunks: {max_chunks}")
        logger.info(f"  GRU: embedding_dim={embedding_dim}, hidden_dim={gru_hidden_dim}, layers={gru_num_layers}")
        logger.info(f"  Chunk dim: {chunk_dim}")
        logger.info(f"  Transformer: layers={num_transformer_layers}, heads={num_attention_heads}")
        logger.info(f"  Projection dim: {projection_dim}")
        logger.info(f"  NOTE: Call fit_tokenizer() before training")

    def _register_positional_encoding(self):
        """Create sinusoidal positional encoding for chunks."""
        max_len = self._max_chunks + 1  # +1 for pool token
        d_model = self._chunk_dim

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('_positional_encoding', pe)

    @property
    def output_dim(self) -> int:
        """Return the output dimension of this feature extractor."""
        return self._projection_dim

    def fit_tokenizer(self, texts: List[str]) -> 'HierarchicalGRUTransformerExtractor':
        """
        Fit the tokenizer on training texts and rebuild the embedding layer.

        This MUST be called before using the model for training or inference.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        logger.info(f"Fitting tokenizer on {len(texts)} texts...")

        # Fit tokenizer to build vocabulary
        self.tokenizer.fit(texts)

        # Rebuild chunk encoder embedding with correct vocabulary size
        self.chunk_encoder.rebuild_embedding(self.tokenizer.vocab_size)

        self._initialized = True
        logger.info(f"Tokenizer fitted: vocabulary size = {self.tokenizer.vocab_size}")

        return self

    def init_extractor(self, texts: List[str]) -> 'HierarchicalGRUTransformerExtractor':
        """Alias for fit_tokenizer() for API compatibility."""
        return self.fit_tokenizer(texts)

    def _create_chunks(
        self,
        token_ids: List[int],
        attention_mask: List[int]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Split token sequence into overlapping chunks.

        Args:
            token_ids: List of token IDs for a document
            attention_mask: List of 1s (valid) and 0s (padding)

        Returns:
            chunk_ids: List of chunks, each is a list of token IDs
            chunk_masks: List of chunks, each is a list of mask values
        """
        chunk_ids = []
        chunk_masks = []

        n_tokens = len(token_ids)
        if n_tokens == 0:
            # Handle empty document - create one chunk of padding
            chunk_ids.append([self.tokenizer.PAD_ID] * self._chunk_size)
            chunk_masks.append([0] * self._chunk_size)
            return chunk_ids, chunk_masks

        # Generate chunks with overlap
        start = 0
        while start < n_tokens and len(chunk_ids) < self._max_chunks:
            end = min(start + self._chunk_size, n_tokens)

            chunk = token_ids[start:end]
            mask = attention_mask[start:end]

            # Pad if chunk is shorter than chunk_size
            if len(chunk) < self._chunk_size:
                pad_len = self._chunk_size - len(chunk)
                chunk = chunk + [self.tokenizer.PAD_ID] * pad_len
                mask = mask + [0] * pad_len

            chunk_ids.append(chunk)
            chunk_masks.append(mask)

            start += self._chunk_step

            # Stop if we've reached the end
            if end >= n_tokens:
                break

        return chunk_ids, chunk_masks

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from texts.

        Args:
            texts: List of document texts

        Returns:
            Feature tensor of shape (batch_size, projection_dim)
        """
        if not self._initialized:
            raise RuntimeError(
                "Tokenizer not fitted. Call fit_tokenizer(texts) before using the model."
            )

        batch_size = len(texts)
        batch_outputs = []

        for text in texts:
            # 1. Tokenize entire document
            encoded = self.tokenizer(
                [text],
                padding=False,
                truncation=True,
                max_length=self._chunk_size * self._max_chunks,
                return_tensors='pt'
            )
            token_ids = encoded['input_ids'][0].tolist()
            attention_mask = encoded['attention_mask'][0].tolist()

            # 2. Create overlapping chunks
            chunk_ids_list, chunk_masks_list = self._create_chunks(token_ids, attention_mask)

            # 3. Encode chunks with GRU
            n_chunks = len(chunk_ids_list)
            chunk_ids_tensor = torch.tensor(chunk_ids_list, dtype=torch.long, device=self._device)
            chunk_masks_tensor = torch.tensor(chunk_masks_list, dtype=torch.long, device=self._device)

            chunk_embeddings, _ = self.chunk_encoder(
                chunk_ids_tensor, chunk_masks_tensor, return_attention=False
            )  # (n_chunks, chunk_dim)

            # 4. Prepend [POOL] token
            sequence = torch.cat([self._pool_token, chunk_embeddings], dim=0)  # (n_chunks+1, chunk_dim)

            # 5. Add positional encoding
            seq_len = sequence.size(0)
            sequence = sequence + self._positional_encoding[:seq_len].to(self._device)

            # 6. Run through transformer layers
            sequence = sequence.unsqueeze(0)  # (1, n_chunks+1, chunk_dim)
            for layer in self._transformer_layers:
                sequence, _ = layer(sequence, return_attention=False)

            # 7. Extract [POOL] output (position 0)
            pool_output = sequence[0, 0, :]  # (chunk_dim,)
            batch_outputs.append(pool_output)

        # Stack batch
        batch_outputs = torch.stack(batch_outputs)  # (B, chunk_dim)

        # 8. Output projection
        features = self._output_projection(batch_outputs)  # (B, projection_dim)

        return features

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of chunk and token attention.

        This extracts:
        1. Attention weights from [POOL] token to each chunk
        2. Attention weights within each chunk (which tokens are attended to)

        Args:
            texts: List of document texts
            top_k: Number of top-attended chunks to show

        Returns:
            List of dicts per document with attention interpretations:
            - 'chunks': List of chunk info with text and attention weights
            - 'chunk_attention': Attention weights from [POOL] to each chunk
            - 'top_chunks': Top-k chunks by attention weight with token details
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer not fitted.")

        interpretations = []

        with torch.no_grad():
            for text in texts:
                # Tokenize
                tokens = self.tokenizer._tokenize(text)
                encoded = self.tokenizer(
                    [text],
                    padding=False,
                    truncation=True,
                    max_length=self._chunk_size * self._max_chunks,
                    return_tensors='pt'
                )
                token_ids = encoded['input_ids'][0].tolist()
                attention_mask = encoded['attention_mask'][0].tolist()

                # Create chunks
                chunk_ids_list, chunk_masks_list = self._create_chunks(token_ids, attention_mask)
                n_chunks = len(chunk_ids_list)

                # Get chunk token texts
                chunk_tokens = []
                for chunk_ids in chunk_ids_list:
                    chunk_text = []
                    for tid in chunk_ids:
                        if tid == self.tokenizer.PAD_ID:
                            break
                        word = self.tokenizer.id_to_word.get(tid, '<UNK>')
                        chunk_text.append(word)
                    chunk_tokens.append(chunk_text)

                # Encode chunks with attention
                chunk_ids_tensor = torch.tensor(chunk_ids_list, dtype=torch.long, device=self._device)
                chunk_masks_tensor = torch.tensor(chunk_masks_list, dtype=torch.long, device=self._device)

                chunk_embeddings, chunk_token_attention = self.chunk_encoder(
                    chunk_ids_tensor, chunk_masks_tensor, return_attention=True
                )

                # Get transformer attention (from [POOL] to chunks)
                sequence = torch.cat([self._pool_token, chunk_embeddings], dim=0)
                seq_len = sequence.size(0)
                sequence = sequence + self._positional_encoding[:seq_len].to(self._device)
                sequence = sequence.unsqueeze(0)

                transformer_attention = None
                for layer in self._transformer_layers:
                    sequence, transformer_attention = layer(sequence, return_attention=True)

                # Extract pool-to-chunk attention (position 0 attending to 1:)
                if transformer_attention is not None:
                    pool_attention = transformer_attention[0, 0, 1:].cpu()  # (n_chunks,)
                    pool_attention = pool_attention / (pool_attention.sum() + 1e-9)
                else:
                    pool_attention = torch.ones(n_chunks) / n_chunks

                # Get top-k chunks
                k_actual = min(top_k, n_chunks)
                top_vals, top_indices = torch.topk(pool_attention, k_actual)

                top_chunks = []
                for val, idx in zip(top_vals.tolist(), top_indices.tolist()):
                    chunk_info = {
                        'chunk_idx': idx,
                        'chunk_attention': val,
                        'tokens': chunk_tokens[idx],
                        'chunk_text': ' '.join(chunk_tokens[idx]),
                    }

                    # Add token-level attention within chunk
                    if chunk_token_attention is not None:
                        token_attn = chunk_token_attention[idx].cpu().tolist()
                        # Truncate to actual tokens
                        n_tokens_in_chunk = len(chunk_tokens[idx])
                        token_attn = token_attn[:n_tokens_in_chunk]

                        # Get top attended tokens
                        token_attn_pairs = list(zip(chunk_tokens[idx], token_attn))
                        token_attn_pairs.sort(key=lambda x: x[1], reverse=True)
                        chunk_info['top_tokens'] = token_attn_pairs[:10]
                        chunk_info['token_attention'] = token_attn

                    top_chunks.append(chunk_info)

                interpretations.append({
                    'n_chunks': n_chunks,
                    'chunk_attention': pool_attention.tolist(),
                    'top_chunks': top_chunks
                })

        return interpretations

    def get_state(self) -> Dict[str, Any]:
        """Get extractor state for checkpoint saving."""
        return {
            'chunk_size': self._chunk_size,
            'chunk_overlap': self._chunk_overlap,
            'max_chunks': self._max_chunks,
            'embedding_dim': self._embedding_dim,
            'gru_hidden_dim': self._gru_hidden_dim,
            'gru_num_layers': self._gru_num_layers,
            'chunk_dim': self._chunk_dim,
            'num_transformer_layers': self._num_transformer_layers,
            'num_attention_heads': self._num_attention_heads,
            'transformer_dropout': self._transformer_dropout,
            'projection_dim': self._projection_dim,
            'tokenizer_state': self.tokenizer.get_state() if self._initialized else None,
        }

    def load_state(self, state: Dict[str, Any]) -> 'HierarchicalGRUTransformerExtractor':
        """Load extractor state from checkpoint."""
        if state.get('tokenizer_state'):
            self.tokenizer.load_state(state['tokenizer_state'])
            self.chunk_encoder.rebuild_embedding(self.tokenizer.vocab_size)
            self._initialized = True
        return self

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if hasattr(self, '_positional_encoding') and self._positional_encoding is not None:
            self._positional_encoding = self._positional_encoding.to(self._device)

        return super().to(device)
