"""Unit tests for feature extractors.

Tests each extractor's forward shape, fit_tokenizer, get_state, get_num_parameters.
LLM-based extractors are tested separately with @pytest.mark.slow.
"""

import pytest
import torch

# Sample texts for testing
SAMPLE_TEXTS = [
    "Patient is a 65 year old male with stage IV NSCLC diagnosed in January 2024.",
    "ECOG performance status 1. Started pembrolizumab 200mg IV every 3 weeks.",
    "CT scan shows partial response after 4 cycles of chemotherapy.",
    "Lab results: WBC 5.2, hemoglobin 12.1, platelets 180.",
]


class TestSimpleCNN:
    def test_forward_shape(self):
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor
        ext = SimpleCNNExtractor(
            embedding_dim=32, conv_dim=32, kernel_size=3, num_conv_blocks=2,
            max_length=100, vocab_size=500, gated_attention_dim=16,
            projection_dim=24, dropout=0.0,
        )
        ext.fit_tokenizer(SAMPLE_TEXTS)
        out = ext(SAMPLE_TEXTS)
        assert out.shape == (4, 24)

    def test_fit_tokenizer_required(self):
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor
        ext = SimpleCNNExtractor(vocab_size=500, projection_dim=24)
        with pytest.raises(RuntimeError, match="not fitted"):
            ext(SAMPLE_TEXTS)

    def test_get_state(self):
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor
        ext = SimpleCNNExtractor(vocab_size=500, projection_dim=24)
        ext.fit_tokenizer(SAMPLE_TEXTS)
        state = ext.get_state()
        assert state['extractor_type'] == 'simple_cnn'
        assert state['output_dim'] == 24
        assert state['tokenizer_state'] is not None

    def test_get_num_parameters(self):
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor
        ext = SimpleCNNExtractor(vocab_size=500, projection_dim=24)
        params = ext.get_num_parameters()
        assert params['trainable'] > 0
        assert params['frozen'] == 0

    def test_dict_input(self):
        from oci.models.simple_cnn_extractor import SimpleCNNExtractor
        ext = SimpleCNNExtractor(
            embedding_dim=32, conv_dim=32, kernel_size=3, num_conv_blocks=2,
            max_length=100, vocab_size=500, projection_dim=24,
        )
        ext.fit_tokenizer(SAMPLE_TEXTS)
        out = ext({'texts': SAMPLE_TEXTS})
        assert out.shape == (4, 24)


class TestHierarchicalCNN:
    def test_forward_shape(self):
        from oci.models.hierarchical_cnn_extractor import HierarchicalCNNExtractor
        ext = HierarchicalCNNExtractor(
            embedding_dim=32, conv_dim=32, kernel_size=3, num_conv_blocks=2,
            chunk_size=20, chunk_overlap=4, max_chunks=8,
            vocab_size=500, gated_attention_dim=16, projection_dim=24,
        )
        ext.fit_tokenizer(SAMPLE_TEXTS)
        out = ext(SAMPLE_TEXTS)
        assert out.shape == (4, 24)

    def test_fit_tokenizer_required(self):
        from oci.models.hierarchical_cnn_extractor import HierarchicalCNNExtractor
        ext = HierarchicalCNNExtractor(vocab_size=500, projection_dim=24)
        with pytest.raises(RuntimeError, match="not fitted"):
            ext(SAMPLE_TEXTS)

    def test_get_state(self):
        from oci.models.hierarchical_cnn_extractor import HierarchicalCNNExtractor
        ext = HierarchicalCNNExtractor(vocab_size=500, projection_dim=24)
        ext.fit_tokenizer(SAMPLE_TEXTS)
        state = ext.get_state()
        assert state['extractor_type'] == 'hierarchical_cnn'
        assert 'chunk_size' in state


class TestHierarchicalGRU:
    def test_forward_shape(self):
        from oci.models.hierarchical_gru_extractor import HierarchicalGRUExtractor
        ext = HierarchicalGRUExtractor(
            embedding_dim=32, gru_hidden_dim=24, num_gru_layers=1,
            chunk_size=20, chunk_overlap=4, max_chunks=8,
            vocab_size=500, gated_attention_dim=16, projection_dim=24,
        )
        ext.fit_tokenizer(SAMPLE_TEXTS)
        out = ext(SAMPLE_TEXTS)
        assert out.shape == (4, 24)

    def test_fit_tokenizer_required(self):
        from oci.models.hierarchical_gru_extractor import HierarchicalGRUExtractor
        ext = HierarchicalGRUExtractor(vocab_size=500, projection_dim=24)
        with pytest.raises(RuntimeError, match="not fitted"):
            ext(SAMPLE_TEXTS)

    def test_get_state(self):
        from oci.models.hierarchical_gru_extractor import HierarchicalGRUExtractor
        ext = HierarchicalGRUExtractor(vocab_size=500, projection_dim=24)
        ext.fit_tokenizer(SAMPLE_TEXTS)
        state = ext.get_state()
        assert state['extractor_type'] == 'hierarchical_gru'
        assert 'gru_hidden_dim' in state


class TestLearnedTokenizer:
    def test_fit_and_encode(self):
        from oci.models.learned_tokenizer import LearnedTokenizer
        tok = LearnedTokenizer()
        tok.fit(SAMPLE_TEXTS, vocab_size=200, min_freq=1)
        assert tok.vocab_size > 2  # at least PAD and UNK
        ids = tok.encode("patient is stage IV", max_length=10)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_batch(self):
        from oci.models.learned_tokenizer import LearnedTokenizer
        tok = LearnedTokenizer()
        tok.fit(SAMPLE_TEXTS, vocab_size=200, min_freq=1)
        input_ids, mask = tok.encode_batch(SAMPLE_TEXTS[:2], max_length=20)
        assert input_ids.shape[0] == 2
        assert mask.shape == input_ids.shape

    def test_state_roundtrip(self):
        from oci.models.learned_tokenizer import LearnedTokenizer
        tok = LearnedTokenizer()
        tok.fit(SAMPLE_TEXTS, vocab_size=200, min_freq=1)
        state = tok.get_state()

        tok2 = LearnedTokenizer()
        tok2.load_state(state)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.encode("test", 5) == tok.encode("test", 5)

    def test_unk_token(self):
        from oci.models.learned_tokenizer import LearnedTokenizer
        tok = LearnedTokenizer()
        tok.fit(["hello world"], vocab_size=100, min_freq=1)
        ids = tok.encode("xyzzy_unknown_word", max_length=5)
        assert tok.unk_token_id in ids
