# cdt/models/bert_cross_chunk_extractor.py
"""BERT Cross-Chunk feature extractor with token-level cross-chunk attention.

This module implements a hierarchical approach for extracting features from long
clinical text where individual tokens gain global document context:

1. Split text into overlapping token chunks
2. Pass 1: Encode each chunk with BERT -> [CLS] embeddings + token hidden states
3. Pass 2: Cross-chunk transformer where each chunk's tokens attend to both
   local tokens AND condensed [CLS] embeddings from ALL other chunks
4. Intra-chunk attention pooling collapses enriched tokens into chunk vectors
5. Gated attention pooling aggregates chunk vectors into a single document vector

Key advantage over HierarchicalTransformerExtractor: tokens see context from
other chunks (via the global [CLS] embeddings), not just their local chunk.

Architecture:
    Long Clinical Text
            |
    Split into Overlapping Token Chunks (C chunks)
            |
    Pass 1: BERT per chunk -> [CLS] (C x bert_dim) + tokens (C x T x bert_dim)
            |
    Project to cross_chunk_dim
            |
    Pass 2: Cross-Chunk Transformer (batched across chunks)
      For chunk i: [global_1, ..., global_C, local_token_1, ..., local_token_T]
      Standard self-attention -> tokens attend to globals AND local tokens
            |
    AttentionPooling over local token outputs -> chunk embedding (C x cross_chunk_dim)
            |
    GatedAttentionPooling -> single document vector (cross_chunk_dim)
            |
    Output Projection -> Final Representation (projection_dim)
"""

import logging
import math
import threading
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chunking import split_into_chunks_hf
from .gru_extractor import AttentionPooling
from .gated_attention_pooling import GatedAttentionPooling
from .hierarchical_transformer_extractor import InterpretableTransformerLayer
from .numeric_features import NumericFeatureVector


logger = logging.getLogger(__name__)

# Module-level lock for thread-safe HuggingFace model loading.
# AutoModel.from_pretrained with accelerate is not thread-safe (can leave
# meta tensors when called concurrently), so we serialize model loading.
_model_load_lock = threading.Lock()


class BertCrossChunkExtractor(nn.Module):
    """
    BERT Cross-Chunk feature extractor with token-level cross-chunk attention.

    Pass 1 encodes each chunk independently with BERT, producing both [CLS]
    embeddings and per-token hidden states. Pass 2 builds a sequence for each
    chunk consisting of all chunks' [CLS] embeddings (global context) followed
    by that chunk's token hidden states (local context), and runs transformer
    layers so tokens can attend to global summaries from other chunks.

    After Pass 2, intra-chunk AttentionPooling collapses each chunk's enriched
    tokens into a single vector, and GatedAttentionPooling aggregates across
    chunks into a document representation.

    No fit_tokenizer() required - uses pretrained HF tokenizer.

    Args:
        sentence_encoder_model: HuggingFace model name for chunk encoding
        freeze_sentence_encoder: Whether to freeze BERT weights
        max_chunks: Maximum number of chunks per document
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        num_cross_layers: Number of cross-chunk transformer layers
        num_attention_heads: Attention heads in cross-chunk layers
        cross_chunk_dim: Hidden dim for cross-chunk transformer
        cross_chunk_dropout: Dropout in cross-chunk layers
        gated_attention_dim: Hidden dim for gated attention pooling
        projection_dim: Final output dimension
        device: PyTorch device
        numeric_features_enabled: Enable numeric features
    """

    def __init__(
        self,
        sentence_encoder_model: str = "prajjwal1/bert-tiny",
        freeze_sentence_encoder: bool = False,
        max_chunks: int = 100,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        num_cross_layers: int = 2,
        num_attention_heads: int = 4,
        cross_chunk_dim: int = 256,
        cross_chunk_dropout: float = 0.1,
        gated_attention_dim: int = 128,
        projection_dim: int = 128,
        device: Optional[torch.device] = None,
        numeric_features_enabled: bool = False,
        numeric_embedding_dim: int = 32,
        numeric_magnitude_bins: int = 8,
        numeric_type_categories: int = 10
    ):
        super().__init__()

        self._device = device or torch.device('cpu')
        self._numeric_features_enabled = numeric_features_enabled
        self._numeric_embedding_dim = numeric_embedding_dim
        self._numeric_magnitude_bins = numeric_magnitude_bins
        self._numeric_type_categories = numeric_type_categories
        self._sentence_encoder_model = sentence_encoder_model
        self._freeze = freeze_sentence_encoder
        self._max_chunks = max_chunks
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._num_cross_layers = num_cross_layers
        self._num_heads = num_attention_heads
        self._cross_chunk_dim = cross_chunk_dim
        self._dropout = cross_chunk_dropout
        self._gated_attention_dim = gated_attention_dim
        self._projection_dim = projection_dim

        # Lazy initialization
        self._sentence_encoder = None
        self._tokenizer = None
        self._sentence_dim = None
        self._global_projection = None
        self._token_projection = None
        self._cross_chunk_layers = None
        self._intra_chunk_pooling = None
        self._gated_pooling = None
        self._output_projection = None
        self._initialized = False

        logger.info(f"BertCrossChunkExtractor initialized:")
        logger.info(f"  Chunk encoder: {sentence_encoder_model}")
        logger.info(f"  Freeze encoder: {freeze_sentence_encoder}")
        logger.info(f"  Max chunks: {max_chunks}, chunk_size: {chunk_size}, overlap: {chunk_overlap}")
        logger.info(f"  Cross-chunk layers: {num_cross_layers}")
        logger.info(f"  Cross-chunk dim: {cross_chunk_dim}")
        logger.info(f"  Attention heads: {num_attention_heads}")
        logger.info(f"  Gated attention dim: {gated_attention_dim}")
        logger.info(f"  Projection dim: {projection_dim}")

    def _ensure_initialized(self):
        """Lazily initialize components."""
        if self._initialized:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading sentence encoder: {self._sentence_encoder_model}")
        # Serialize model loading across threads — accelerate's from_pretrained
        # uses init_empty_weights (meta tensors) internally and is not thread-safe.
        with _model_load_lock:
            self._tokenizer = AutoTokenizer.from_pretrained(self._sentence_encoder_model)
            self._sentence_encoder = AutoModel.from_pretrained(self._sentence_encoder_model)
        self._sentence_encoder = self._sentence_encoder.to(self._device)
        self._sentence_dim = self._sentence_encoder.config.hidden_size
        logger.info(f"  Sentence encoder dim: {self._sentence_dim}")

        if self._freeze:
            for param in self._sentence_encoder.parameters():
                param.requires_grad = False
            logger.info("  Sentence encoder frozen")

        # Projection: [CLS] -> cross_chunk_dim (for global tokens)
        self._global_projection = nn.Linear(self._sentence_dim, self._cross_chunk_dim).to(self._device)

        # Projection: token hidden states -> cross_chunk_dim (for local tokens)
        self._token_projection = nn.Linear(self._sentence_dim, self._cross_chunk_dim).to(self._device)

        # Positional encodings: separate for globals and locals
        self._register_positional_encodings()

        # Cross-chunk transformer layers
        self._cross_chunk_layers = nn.ModuleList([
            InterpretableTransformerLayer(
                d_model=self._cross_chunk_dim,
                nhead=self._num_heads,
                dim_feedforward=self._cross_chunk_dim * 4,
                dropout=self._dropout
            )
            for _ in range(self._num_cross_layers)
        ]).to(self._device)

        # Intra-chunk attention pooling (collapses tokens within a chunk)
        self._intra_chunk_pooling = AttentionPooling(
            hidden_dim=self._cross_chunk_dim,
            attention_dim=self._cross_chunk_dim
        ).to(self._device)

        # Gated attention pooling for final document aggregation
        self._gated_pooling = GatedAttentionPooling(
            hidden_dim=self._cross_chunk_dim,
            attention_dim=self._gated_attention_dim
        ).to(self._device)

        # Output projection
        self._output_projection = nn.Sequential(
            nn.Linear(self._cross_chunk_dim, self._cross_chunk_dim),
            nn.LayerNorm(self._cross_chunk_dim),
            nn.GELU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._cross_chunk_dim, self._projection_dim),
            nn.LayerNorm(self._projection_dim)
        ).to(self._device)

        # Numeric feature vector
        self._numeric_feature_vector = None
        self._numeric_merge = None
        if self._numeric_features_enabled:
            self._numeric_feature_vector = NumericFeatureVector(
                num_magnitude_bins=self._numeric_magnitude_bins,
                num_type_categories=self._numeric_type_categories,
                output_dim=self._numeric_embedding_dim
            ).to(self._device)
            self._numeric_merge = nn.Sequential(
                nn.Linear(self._cross_chunk_dim + self._numeric_embedding_dim, self._cross_chunk_dim),
                nn.LayerNorm(self._cross_chunk_dim),
                nn.ReLU(),
            ).to(self._device)

        self._initialized = True
        logger.info("BertCrossChunkExtractor initialization complete")

    def _register_positional_encodings(self):
        """Create sinusoidal positional encodings for globals and locals."""
        d_model = self._cross_chunk_dim

        # Global positional encoding (for chunk positions 0..max_chunks-1)
        max_global = self._max_chunks
        pe_global = torch.zeros(max_global, d_model)
        position = torch.arange(0, max_global, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_global[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe_global[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe_global[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe_global', pe_global)

        # Local positional encoding (for token positions 0..chunk_size-1)
        max_local = self._chunk_size
        pe_local = torch.zeros(max_local, d_model)
        position = torch.arange(0, max_local, dtype=torch.float).unsqueeze(1)
        # Use a different base frequency to distinguish from globals
        div_term_local = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(5000.0) / d_model))
        pe_local[:, 0::2] = torch.sin(position * div_term_local)
        if d_model % 2 == 1:
            pe_local[:, 1::2] = torch.cos(position * div_term_local[:-1])
        else:
            pe_local[:, 1::2] = torch.cos(position * div_term_local)
        self.register_buffer('_pe_local', pe_local)

    @property
    def output_dim(self) -> int:
        """Return the output dimension of this feature extractor."""
        return self._projection_dim

    def _encode_chunks_batch(
        self,
        chunks: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode chunks with BERT, returning [CLS] tokens AND per-token hidden states.

        Args:
            chunks: List of chunk strings

        Returns:
            cls_embeddings: (C, sentence_dim) - [CLS] token embeddings
            token_hidden_states: (C, T, sentence_dim) - per-token hidden states (padded)
            token_attention_mask: (C, T) - attention mask for tokens (1=valid, 0=pad)
        """
        if not chunks:
            self._ensure_initialized()
            return (
                torch.zeros(0, self._sentence_dim, device=self._device),
                torch.zeros(0, 0, self._sentence_dim, device=self._device),
                torch.zeros(0, 0, device=self._device)
            )

        encoded = self._tokenizer(
            chunks,
            padding=True,
            truncation=True,
            max_length=self._chunk_size,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        with torch.set_grad_enabled(not self._freeze):
            outputs = self._sentence_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # [CLS] at position 0
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (C, sentence_dim)

        # All token hidden states (including [CLS] and [SEP])
        token_hidden_states = outputs.last_hidden_state  # (C, T_padded, sentence_dim)

        return cls_embeddings, token_hidden_states, attention_mask

    def _encode_pass1_subbatched(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_bert_batch: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sub-batch all pre-tokenized chunks through BERT (Pass 1).

        Processes chunks in sub-batches to avoid OOM when total_chunks is large.
        Returns raw BERT outputs needed for per-document Pass 2 processing.

        Args:
            input_ids: (total_chunks, seq_len) pre-tokenized input IDs
            attention_mask: (total_chunks, seq_len) attention mask (1=valid, 0=pad)
            max_bert_batch: Maximum chunks per BERT forward pass

        Returns:
            cls_embeddings: (total_chunks, sentence_dim) [CLS] token embeddings
            token_hidden_states: (total_chunks, seq_len, sentence_dim) per-token hidden states
            attention_mask: (total_chunks, seq_len) attention mask passed through
        """
        total = input_ids.size(0)
        all_cls = []
        all_tokens = []

        for start in range(0, total, max_bert_batch):
            end = min(start + max_bert_batch, total)
            batch_ids = input_ids[start:end].to(self._device)
            batch_mask = attention_mask[start:end].to(self._device)

            with torch.set_grad_enabled(not self._freeze):
                outputs = self._sentence_encoder(
                    input_ids=batch_ids,
                    attention_mask=batch_mask
                )

            all_cls.append(outputs.last_hidden_state[:, 0, :])  # (sub_B, sentence_dim)
            all_tokens.append(outputs.last_hidden_state)  # (sub_B, seq_len, sentence_dim)

        cls_embeddings = torch.cat(all_cls, dim=0)  # (total_chunks, sentence_dim)
        token_hidden_states = torch.cat(all_tokens, dim=0)  # (total_chunks, seq_len, sentence_dim)
        # Ensure attention_mask is on the right device
        attention_mask = attention_mask.to(self._device)

        return cls_embeddings, token_hidden_states, attention_mask

    def _process_single_document_from_pass1(
        self,
        cls_embs: torch.Tensor,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run Pass 2 (cross-chunk transformer + pooling) for a single document
        given pre-computed Pass 1 BERT outputs.

        Args:
            cls_embs: (C, sentence_dim) - [CLS] embeddings from Pass 1
            token_states: (C, T_padded, sentence_dim) - token hidden states from Pass 1
            token_mask: (C, T_padded) - attention mask for tokens
            return_attention: Whether to return attention weights

        Returns:
            pooled: (cross_chunk_dim,) - document vector
            chunk_embeddings: (C, cross_chunk_dim) - per-chunk embeddings (if return_attention)
            gated_weights: (C,) - gated attention weights (if return_attention)
        """
        num_chunks = cls_embs.size(0)
        if num_chunks == 0:
            pooled = torch.zeros(self._cross_chunk_dim, device=self._device)
            if return_attention:
                return pooled, torch.zeros(0, self._cross_chunk_dim, device=self._device), torch.zeros(0, device=self._device)
            return pooled, None, None

        T_padded = token_states.size(1)

        # 1. Project to cross_chunk_dim
        global_tokens = self._global_projection(cls_embs)  # (C, cross_chunk_dim)
        local_tokens = self._token_projection(token_states)  # (C, T_padded, cross_chunk_dim)

        # 2. Add positional encodings
        global_tokens = global_tokens + self._pe_global[:num_chunks].to(self._device)
        local_tokens = local_tokens + self._pe_local[:T_padded].to(self._device).unsqueeze(0)

        # 3. Build cross-chunk sequences: for each chunk i, sequence = [globals, local_tokens_i]
        # Expand globals for all chunks: (C, C, cross_chunk_dim)
        globals_expanded = global_tokens.unsqueeze(0).expand(num_chunks, -1, -1)

        # Concatenate: (C, C + T_padded, cross_chunk_dim)
        cross_sequences = torch.cat([globals_expanded, local_tokens], dim=1)

        # Build attention mask: globals are always valid, locals use token_mask
        global_mask = torch.ones(num_chunks, num_chunks, device=self._device)  # (C, C)
        # cross_mask: (C, C + T_padded)
        cross_mask = torch.cat([global_mask, token_mask], dim=1)

        # 4. Run through cross-chunk transformer layers
        attn_weights = None
        for layer in self._cross_chunk_layers:
            cross_sequences, attn_weights = layer(
                cross_sequences,
                return_attention=return_attention
            )

        # 5. Extract local token outputs (positions C: onward)
        local_outputs = cross_sequences[:, num_chunks:, :]  # (C, T_padded, cross_chunk_dim)
        local_mask = token_mask  # (C, T_padded)

        # 6. Intra-chunk attention pooling: collapse tokens within each chunk
        chunk_embeddings = self._intra_chunk_pooling(
            local_outputs,
            attention_mask=local_mask
        )  # (C, cross_chunk_dim)

        # 7. Gated attention pooling over chunks
        pooled, gated_weights = self._gated_pooling(chunk_embeddings)  # (cross_chunk_dim,), (C,)

        if return_attention:
            return pooled, chunk_embeddings, gated_weights
        return pooled, None, None

    def _process_single_document(
        self,
        text: str,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Process a single document through both passes (chunk, tokenize, encode, cross-chunk).

        Args:
            text: Document text
            return_attention: Whether to return attention weights

        Returns:
            pooled: (cross_chunk_dim,) - document vector
            chunk_embeddings: (C, cross_chunk_dim) - per-chunk embeddings (if return_attention)
            gated_weights: (C,) - gated attention weights (if return_attention)
        """
        # 1. Split into overlapping token chunks
        chunks = split_into_chunks_hf(
            text,
            self._tokenizer,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            max_chunks=self._max_chunks
        )
        if not chunks:
            chunks = [text[:500]]

        # 2. Pass 1: Encode all chunks with BERT
        cls_embs, token_states, token_mask = self._encode_chunks_batch(chunks)

        # 3. Pass 2: Cross-chunk transformer + pooling
        return self._process_single_document_from_pass1(
            cls_embs, token_states, token_mask, return_attention=return_attention
        )

    def _forward_preprocessed(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        GPU-only forward pass on pre-tokenized batch from HFChunkCollator.

        Pass 1 (BERT) is batched across all chunks from all documents via sub-batching.
        Pass 2 (cross-chunk transformer) stays per-document because different documents
        have different chunk counts and cross-chunk attention patterns.

        Args:
            batch: Dict with 'chunk_input_ids', 'chunk_attention_mask', 'doc_chunk_counts', 'texts'

        Returns:
            Feature tensor of shape (B, projection_dim)
        """
        input_ids = batch['chunk_input_ids']          # (total_chunks, seq_len)
        attn_mask = batch['chunk_attention_mask']      # (total_chunks, seq_len)
        doc_chunk_counts = batch['doc_chunk_counts']   # List[int], len B
        texts = batch['texts']

        # 1. Batched Pass 1: Encode ALL chunks through BERT (sub-batched)
        all_cls, all_tokens, all_mask = self._encode_pass1_subbatched(input_ids, attn_mask)
        # all_cls: (total_chunks, sentence_dim)
        # all_tokens: (total_chunks, seq_len, sentence_dim)
        # all_mask: (total_chunks, seq_len)

        # 2. Split per-document and run Pass 2 per-document
        batch_outputs = []
        offset = 0
        for i, count in enumerate(doc_chunk_counts):
            doc_cls = all_cls[offset:offset + count]        # (C_i, sentence_dim)
            doc_tokens = all_tokens[offset:offset + count]  # (C_i, seq_len, sentence_dim)
            doc_mask = all_mask[offset:offset + count]      # (C_i, seq_len)
            offset += count

            # Pass 2: cross-chunk transformer + pooling (per-document)
            pooled, _, _ = self._process_single_document_from_pass1(
                doc_cls, doc_tokens, doc_mask, return_attention=False
            )

            # Merge numeric features if enabled
            if self._numeric_features_enabled and self._numeric_feature_vector is not None:
                numeric_feats = self._numeric_feature_vector([texts[i]])  # (1, numeric_dim)
                pooled = self._numeric_merge(
                    torch.cat([pooled.unsqueeze(0), numeric_feats], dim=1)
                ).squeeze(0)

            batch_outputs.append(pooled)

        batch_outputs = torch.stack(batch_outputs)  # (B, cross_chunk_dim)
        features = self._output_projection(batch_outputs)  # (B, projection_dim)
        return features

    def _forward_from_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Legacy forward path: chunk + tokenize + encode per document.

        Args:
            texts: List of document texts

        Returns:
            Feature tensor of shape (batch_size, projection_dim)
        """
        batch_outputs = []

        for text in texts:
            pooled, _, _ = self._process_single_document(text, return_attention=False)

            # Merge numeric features if enabled
            if self._numeric_features_enabled and self._numeric_feature_vector is not None:
                numeric_feats = self._numeric_feature_vector([text])  # (1, numeric_dim)
                pooled = self._numeric_merge(
                    torch.cat([pooled.unsqueeze(0), numeric_feats], dim=1)
                ).squeeze(0)

            batch_outputs.append(pooled)

        batch_outputs = torch.stack(batch_outputs)  # (B, cross_chunk_dim)
        features = self._output_projection(batch_outputs)  # (B, projection_dim)
        return features

    def forward(self, texts_or_batch) -> torch.Tensor:
        """
        Extract features from texts or preprocessed batch.

        Accepts either:
        - List[str]: Raw text strings (legacy path, chunks + tokenizes internally)
        - Dict with 'chunk_input_ids': Preprocessed batch from HFChunkCollator

        Args:
            texts_or_batch: List of document texts or preprocessed batch dict

        Returns:
            Feature tensor of shape (batch_size, projection_dim)
        """
        self._ensure_initialized()

        if isinstance(texts_or_batch, dict) and 'chunk_input_ids' in texts_or_batch:
            return self._forward_preprocessed(texts_or_batch)
        return self._forward_from_texts(texts_or_batch)

    def _forward_with_instances_preprocessed(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Batched forward pass returning chunk-level info for CLAM-style instance loss.

        Pass 1 (BERT) is batched across all chunks. Pass 2 stays per-document.

        Args:
            batch: Preprocessed batch dict from HFChunkCollator

        Returns:
            doc_features: (B, projection_dim) - document-level features
            chunk_embeddings_list: List of (C_i, cross_chunk_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - gated attention weights per doc
        """
        input_ids = batch['chunk_input_ids']
        attn_mask = batch['chunk_attention_mask']
        doc_chunk_counts = batch['doc_chunk_counts']
        texts = batch['texts']

        # 1. Batched Pass 1: Encode ALL chunks through BERT (sub-batched)
        all_cls, all_tokens, all_mask = self._encode_pass1_subbatched(input_ids, attn_mask)

        # 2. Split per-document and run Pass 2 per-document (with attention for CLAM)
        batch_outputs = []
        chunk_embeddings_list = []
        attention_weights_list = []

        offset = 0
        for i, count in enumerate(doc_chunk_counts):
            doc_cls = all_cls[offset:offset + count]
            doc_tokens = all_tokens[offset:offset + count]
            doc_mask = all_mask[offset:offset + count]
            offset += count

            # Pass 2: cross-chunk transformer + pooling (per-document, with attention)
            pooled, chunk_embs, gated_weights = self._process_single_document_from_pass1(
                doc_cls, doc_tokens, doc_mask, return_attention=True
            )

            if chunk_embs is None or chunk_embs.size(0) == 0:
                batch_outputs.append(
                    torch.zeros(self._projection_dim, device=self._device)
                )
                chunk_embeddings_list.append(
                    torch.zeros(0, self._cross_chunk_dim, device=self._device)
                )
                attention_weights_list.append(
                    torch.zeros(0, device=self._device)
                )
                continue

            # Merge numeric features if enabled
            if self._numeric_features_enabled and self._numeric_feature_vector is not None:
                numeric_feats = self._numeric_feature_vector([texts[i]])
                pooled = self._numeric_merge(
                    torch.cat([pooled.unsqueeze(0), numeric_feats], dim=1)
                ).squeeze(0)

            output = self._output_projection(pooled.unsqueeze(0)).squeeze(0)
            batch_outputs.append(output)
            chunk_embeddings_list.append(chunk_embs)
            attention_weights_list.append(gated_weights)

        doc_features = torch.stack(batch_outputs)
        return doc_features, chunk_embeddings_list, attention_weights_list

    def _forward_with_instances_from_texts(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Legacy forward_with_instances path from raw text strings.

        Args:
            texts: List of document texts

        Returns:
            doc_features: (B, projection_dim) - document-level features
            chunk_embeddings_list: List of (C_i, cross_chunk_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - gated attention weights per doc
        """
        batch_outputs = []
        chunk_embeddings_list = []
        attention_weights_list = []

        for text in texts:
            pooled, chunk_embs, gated_weights = self._process_single_document(
                text, return_attention=True
            )

            if chunk_embs is None or chunk_embs.size(0) == 0:
                batch_outputs.append(
                    torch.zeros(self._projection_dim, device=self._device)
                )
                chunk_embeddings_list.append(
                    torch.zeros(0, self._cross_chunk_dim, device=self._device)
                )
                attention_weights_list.append(
                    torch.zeros(0, device=self._device)
                )
                continue

            # Merge numeric features if enabled
            if self._numeric_features_enabled and self._numeric_feature_vector is not None:
                numeric_feats = self._numeric_feature_vector([text])
                pooled = self._numeric_merge(
                    torch.cat([pooled.unsqueeze(0), numeric_feats], dim=1)
                ).squeeze(0)

            output = self._output_projection(pooled.unsqueeze(0)).squeeze(0)
            batch_outputs.append(output)
            chunk_embeddings_list.append(chunk_embs)
            attention_weights_list.append(gated_weights)

        doc_features = torch.stack(batch_outputs)  # (B, projection_dim)
        return doc_features, chunk_embeddings_list, attention_weights_list

    def forward_with_instances(
        self,
        texts_or_batch
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass returning document features AND chunk-level info for CLAM-style instance loss.

        Accepts either List[str] or preprocessed batch dict.

        Args:
            texts_or_batch: List of document texts or preprocessed batch dict

        Returns:
            doc_features: (B, projection_dim) - document-level features
            chunk_embeddings_list: List of (C_i, cross_chunk_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - gated attention weights per doc
        """
        self._ensure_initialized()

        if isinstance(texts_or_batch, dict) and 'chunk_input_ids' in texts_or_batch:
            return self._forward_with_instances_preprocessed(texts_or_batch)
        return self._forward_with_instances_from_texts(texts_or_batch)

    def init_extractor(self, texts: List[str]) -> 'BertCrossChunkExtractor':
        """
        Initialize the feature extractor (triggers lazy initialization).

        The texts argument is not used since we use pretrained tokenizers.

        Args:
            texts: List of training text strings (not used, kept for API compatibility)

        Returns:
            self for method chaining
        """
        self._ensure_initialized()
        return self

    def fit_tokenizer(self, texts: List[str]) -> 'BertCrossChunkExtractor':
        """Alias for init_extractor() for backward compatibility."""
        return self.init_extractor(texts)

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if self._sentence_encoder is not None:
            self._sentence_encoder = self._sentence_encoder.to(self._device)
        if self._global_projection is not None:
            self._global_projection = self._global_projection.to(self._device)
        if self._token_projection is not None:
            self._token_projection = self._token_projection.to(self._device)
        if self._cross_chunk_layers is not None:
            self._cross_chunk_layers = self._cross_chunk_layers.to(self._device)
        if self._intra_chunk_pooling is not None:
            self._intra_chunk_pooling = self._intra_chunk_pooling.to(self._device)
        if self._gated_pooling is not None:
            self._gated_pooling = self._gated_pooling.to(self._device)
        if self._output_projection is not None:
            self._output_projection = self._output_projection.to(self._device)
        if hasattr(self, '_pe_global') and self._pe_global is not None:
            self._pe_global = self._pe_global.to(self._device)
        if hasattr(self, '_pe_local') and self._pe_local is not None:
            self._pe_local = self._pe_local.to(self._device)

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
            'max_chunks': self._max_chunks,
            'chunk_size': self._chunk_size,
            'chunk_overlap': self._chunk_overlap,
            'num_cross_layers': self._num_cross_layers,
            'num_attention_heads': self._num_heads,
            'cross_chunk_dim': self._cross_chunk_dim,
            'cross_chunk_dropout': self._dropout,
            'gated_attention_dim': self._gated_attention_dim,
            'projection_dim': self._projection_dim,
        }

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of chunk and token attention.

        Extracts gated attention weights over chunks and intra-chunk token
        attention for the top-attended chunks.

        Args:
            texts: List of document texts
            top_k: Number of top-attended chunks to show

        Returns:
            List of dicts per document with attention interpretations
        """
        self._ensure_initialized()
        interpretations = []

        with torch.no_grad():
            for text in texts:
                chunks = split_into_chunks_hf(
                    text,
                    self._tokenizer,
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                    max_chunks=self._max_chunks
                )
                if not chunks:
                    chunks = [text[:500]]

                _, chunk_embs, gated_weights = self._process_single_document(
                    text, return_attention=True
                )

                if chunk_embs is None or chunk_embs.size(0) == 0:
                    interpretations.append({
                        'num_chunks': 0,
                        'chunks': chunks,
                        'chunk_attention': [],
                        'top_chunks': []
                    })
                    continue

                gated_weights_cpu = gated_weights.cpu()
                num_chunks = chunk_embs.size(0)

                # Get top-k chunks
                k_actual = min(top_k, num_chunks)
                top_vals, top_indices = torch.topk(gated_weights_cpu, k_actual)

                top_chunks = [
                    {
                        'chunk_idx': int(idx),
                        'attention': float(val),
                        'text_preview': chunks[idx][:200] + '...' if len(chunks[idx]) > 200 else chunks[idx]
                    }
                    for val, idx in zip(top_vals.tolist(), top_indices.tolist())
                ]

                interpretations.append({
                    'num_chunks': num_chunks,
                    'chunks': chunks,
                    'chunk_attention': gated_weights_cpu.tolist(),
                    'top_chunks': top_chunks
                })

        return interpretations

    def get_attention_weights(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get raw attention weights for visualization.

        Args:
            texts: List of document texts

        Returns:
            Dictionary with interpretations and model metadata
        """
        interpretations = self.interpret_attention(texts, top_k=self._max_chunks)
        return {
            'interpretations': interpretations,
            'num_cross_layers': self._num_cross_layers,
            'num_heads': self._num_heads,
            'cross_chunk_dim': self._cross_chunk_dim,
            'model': self._sentence_encoder_model
        }
