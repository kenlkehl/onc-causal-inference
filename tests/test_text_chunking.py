"""Tests for oci/models/text_chunking.py"""

import torch
import pytest

from oci.models.text_chunking import chunk_token_ids, pad_and_batch_chunks


class TestChunkTokenIds:
    def test_basic_chunking(self):
        ids = list(range(20))
        chunks = chunk_token_ids(ids, chunk_size=8, chunk_overlap=2, max_chunks=10)
        assert len(chunks) > 1
        assert all(len(c) <= 8 for c in chunks)

    def test_overlap(self):
        ids = list(range(20))
        chunks = chunk_token_ids(ids, chunk_size=10, chunk_overlap=4, max_chunks=10)
        # Stride = 10 - 4 = 6, so chunk 0 = [0..9], chunk 1 = [6..15], chunk 2 = [12..19]
        assert chunks[0][-4:] == chunks[1][:4]  # overlap region

    def test_max_chunks_truncation(self):
        ids = list(range(100))
        chunks = chunk_token_ids(ids, chunk_size=10, chunk_overlap=2, max_chunks=3)
        assert len(chunks) == 3

    def test_short_text(self):
        ids = [1, 2, 3]
        chunks = chunk_token_ids(ids, chunk_size=10, chunk_overlap=2, max_chunks=5)
        assert len(chunks) == 1
        assert chunks[0] == [1, 2, 3]

    def test_empty_text(self):
        chunks = chunk_token_ids([], chunk_size=10, chunk_overlap=2, max_chunks=5)
        assert len(chunks) == 1
        assert chunks[0] == []

    def test_exact_chunk_size(self):
        ids = list(range(10))
        chunks = chunk_token_ids(ids, chunk_size=10, chunk_overlap=0, max_chunks=5)
        assert len(chunks) == 1
        assert chunks[0] == ids

    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(ValueError):
            chunk_token_ids([1, 2, 3], chunk_size=5, chunk_overlap=5, max_chunks=3)

    def test_no_overlap(self):
        ids = list(range(20))
        chunks = chunk_token_ids(ids, chunk_size=10, chunk_overlap=0, max_chunks=10)
        assert len(chunks) == 2
        assert chunks[0] == list(range(10))
        assert chunks[1] == list(range(10, 20))


class TestPadAndBatchChunks:
    def test_basic_padding(self):
        batch = [
            [[1, 2, 3], [4, 5]],      # 2 chunks, lengths 3 and 2
            [[6, 7, 8, 9]],             # 1 chunk, length 4
        ]
        input_ids, attn_mask, chunk_mask = pad_and_batch_chunks(batch, pad_token_id=0)

        assert input_ids.shape == (2, 2, 4)  # B=2, max_chunks=2, max_len=4
        assert attn_mask.shape == (2, 2, 4)
        assert chunk_mask.shape == (2, 2)

        # Check chunk mask
        assert chunk_mask[0].tolist() == [1.0, 1.0]
        assert chunk_mask[1].tolist() == [1.0, 0.0]

        # Check padding in attention mask
        assert attn_mask[0, 0].tolist() == [1.0, 1.0, 1.0, 0.0]
        assert attn_mask[0, 1].tolist() == [1.0, 1.0, 0.0, 0.0]
        assert attn_mask[1, 1].tolist() == [0.0, 0.0, 0.0, 0.0]  # padding chunk

    def test_single_sample(self):
        batch = [[[1, 2, 3, 4, 5]]]
        input_ids, attn_mask, chunk_mask = pad_and_batch_chunks(batch, pad_token_id=0)
        assert input_ids.shape == (1, 1, 5)
        assert chunk_mask[0, 0] == 1.0
