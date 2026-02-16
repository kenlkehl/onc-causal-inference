# cdt/data/collators.py
"""Collator classes that move chunking + tokenization into DataLoader workers.

Two collator families matching the two tokenizer types:
- HFChunkCollator: For HuggingFace tokenizer extractors (bert_pool, hierarchical_transformer,
  gated_mil_hierarchical, bert_cross_chunk)
- VocabChunkCollator: For learned vocabulary extractors (gru_pool, conv_pool, gru_transformer_mil)

Usage:
    collator = create_collator(model.feature_extractor, base_collate_fn=collate_batch)
    train_loader = DataLoader(dataset, collate_fn=collator, num_workers=4)
"""

import logging
import re
from typing import List, Dict, Any, Optional, Callable

import torch

from .dataset import collate_batch
from ..models.chunking import split_into_chunks_hf, split_into_chunks_vocab

logger = logging.getLogger(__name__)


class HFChunkCollator:
    """Collate function that chunks + tokenizes text using an HF tokenizer.

    Moves CPU-heavy work (chunking + HF tokenization) into DataLoader workers.
    Produces pre-tokenized tensors for GPU-only forward passes.

    Args:
        tokenizer: HuggingFace PreTrainedTokenizer
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        max_chunks: Maximum number of chunks per document
    """

    def __init__(
        self,
        tokenizer,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        max_chunks: int = 100
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks

    def __call__(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. Start with standard collation for non-text fields
        result = collate_batch(batch_items)

        # 2. Chunk all documents
        all_chunks = []
        doc_chunk_counts = []
        for item in batch_items:
            text = item['text']
            chunks = split_into_chunks_hf(
                text,
                self.tokenizer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                max_chunks=self.max_chunks
            )
            if not chunks:
                chunks = [text[:500]]
            doc_chunk_counts.append(len(chunks))
            all_chunks.extend(chunks)

        # 3. Single HF tokenizer call for ALL chunks across all documents
        encoded = self.tokenizer(
            all_chunks,
            padding=True,
            truncation=True,
            max_length=self.chunk_size,
            return_tensors='pt'
        )

        # 4. Add preprocessed fields to batch
        result['chunk_input_ids'] = encoded['input_ids']           # (total_chunks, seq_len)
        result['chunk_attention_mask'] = encoded['attention_mask']  # (total_chunks, seq_len)
        result['doc_chunk_counts'] = doc_chunk_counts              # List[int], len B

        return result


class VocabChunkCollator:
    """Collate function that chunks + tokenizes using a learned vocabulary.

    For vocab-based extractors (GRU, Conv) that use split_into_chunks_vocab.

    Args:
        word_to_idx: Dictionary mapping words to token indices
        pad_token: Padding token index
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        max_chunks: Maximum number of chunks per document
    """

    def __init__(
        self,
        word_to_idx: dict,
        pad_token: int = 0,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        max_chunks: int = 100
    ):
        self.word_to_idx = word_to_idx
        self.pad_token = pad_token
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks

    @staticmethod
    def _tokenize_fn(text: str) -> List[str]:
        """Tokenize text into words (matches extractor tokenization)."""
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def __call__(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. Start with standard collation for non-text fields
        result = collate_batch(batch_items)

        # 2. Chunk all documents -> token ID lists
        all_chunk_ids = []  # List[List[int]]
        doc_chunk_counts = []
        for item in batch_items:
            text = item['text']
            chunks = split_into_chunks_vocab(
                text,
                word_to_idx=self.word_to_idx,
                tokenize_fn=self._tokenize_fn,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                max_chunks=self.max_chunks
            )
            if not chunks:
                unk_idx = self.word_to_idx.get('<unk>', 0)
                chunks = [[unk_idx]]
            doc_chunk_counts.append(len(chunks))
            all_chunk_ids.extend(chunks)

        # 3. Pad all chunks to same length
        max_len = max(len(c) for c in all_chunk_ids)
        padded = torch.full((len(all_chunk_ids), max_len), self.pad_token, dtype=torch.long)
        chunk_lengths = torch.zeros(len(all_chunk_ids), dtype=torch.long)
        for i, chunk in enumerate(all_chunk_ids):
            padded[i, :len(chunk)] = torch.tensor(chunk, dtype=torch.long)
            chunk_lengths[i] = len(chunk)

        # 4. Add preprocessed fields to batch
        result['chunk_token_ids'] = padded           # (total_chunks, max_len)
        result['chunk_lengths'] = chunk_lengths      # (total_chunks,)
        result['doc_chunk_counts'] = doc_chunk_counts  # List[int], len B

        return result


def create_collator(
    feature_extractor,
    effect_feature_extractor=None
) -> Optional[Callable]:
    """Factory that inspects extractor type and returns the right collator.

    Args:
        feature_extractor: The model's feature extractor (already initialized)
        effect_feature_extractor: Optional second extractor for dual mode

    Returns:
        Collator function, or None if extractor doesn't support preprocessing
    """
    from ..models.bert_pool_extractor import BertPoolExtractor
    from ..models.hierarchical_transformer_extractor import HierarchicalTransformerExtractor
    from ..models.gated_mil_hierarchical_extractor import GatedMILHierarchicalExtractor
    from ..models.bert_cross_chunk_extractor import BertCrossChunkExtractor
    from ..models.gru_pool_extractor import GRUPoolExtractor
    from ..models.conv_pool_extractor import DilatedConvPoolExtractor
    from ..models.gru_transformer_mil_extractor import GRUTransformerMILExtractor
    from ..models.transformer_pool_extractor import TransformerPoolExtractor

    ext = feature_extractor

    # HF-based extractors
    if isinstance(ext, (BertPoolExtractor, HierarchicalTransformerExtractor,
                        GatedMILHierarchicalExtractor, BertCrossChunkExtractor)):
        ext._ensure_initialized()
        return HFChunkCollator(
            tokenizer=ext._tokenizer,
            chunk_size=ext._chunk_size,
            chunk_overlap=ext._chunk_overlap,
            max_chunks=ext._max_chunks
        )

    # Vocab-based extractors
    if isinstance(ext, (GRUPoolExtractor, DilatedConvPoolExtractor, GRUTransformerMILExtractor, TransformerPoolExtractor)):
        if not ext._initialized or ext._embedding is None:
            logger.warning("Vocab-based extractor not initialized, cannot create collator")
            return None
        return VocabChunkCollator(
            word_to_idx=ext._tokenizer.word_to_id,
            pad_token=ext._tokenizer.pad_token,
            chunk_size=ext._chunk_size,
            chunk_overlap=ext._chunk_overlap,
            max_chunks=ext._max_chunks
        )

    # Extractors that don't need special collation (bert, cnn, llm, conv1d_transformer_hybrid)
    return None
