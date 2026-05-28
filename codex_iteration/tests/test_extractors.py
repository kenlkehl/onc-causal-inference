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


class TestTextMarkerExtractor:
    def test_forward_and_role_shapes(self):
        from oci.models.text_marker_extractor import TextMarkerExtractor

        texts = [
            "The patient is a 64-year-old woman. PD-L1 TPS 70%.",
            "At age 59, pre-treatment evaluation. PD-L1 tumor proportion score was <1%.",
        ]
        ext = TextMarkerExtractor()
        out = ext(texts)
        assert out.shape == (2, 6)

        role_features = ext.extract_role_features(texts)
        assert role_features["confounder"].shape == (2, 2)
        assert role_features["effect_modifier"].shape == (2, 4)
        assert role_features["effect_modifier"][0, 2] == 1.0
        assert role_features["effect_modifier"][1, 0] == 1.0

    def test_dict_input(self):
        from oci.models.text_marker_extractor import TextMarkerExtractor

        ext = TextMarkerExtractor()
        out = ext({"texts": ["65 year old male with PD-L1 expression of 30%."]})
        assert out.shape == (1, 6)
        assert out[0, 3] == 1.0


class TestByteCNNExtractor:
    def test_forward_shape_without_tokenizer_fit(self):
        from oci.models.byte_cnn_extractor import ByteCNNExtractor

        ext = ByteCNNExtractor(
            embedding_dim=8,
            conv_dim=12,
            kernel_size=3,
            num_conv_blocks=2,
            chunk_size=32,
            chunk_overlap=4,
            max_chunks=4,
            gated_attention_dim=8,
            projection_dim=16,
            dropout=0.0,
        )
        out = ext(SAMPLE_TEXTS[:2])
        assert out.shape == (2, 16)

    def test_dict_input_and_shared_forest_features(self):
        from oci.models.byte_cnn_extractor import ByteCNNExtractor

        ext = ByteCNNExtractor(
            embedding_dim=8,
            conv_dim=12,
            chunk_size=32,
            chunk_overlap=4,
            max_chunks=4,
            projection_dim=16,
        )
        out = ext({"texts": SAMPLE_TEXTS[:1]})
        shared = ext.extract_shared_forest_features({"texts": SAMPLE_TEXTS[:1]}, out)
        assert shared.shape == (1, 16)
        assert shared is out


class TestFrozenLLMMultiSlotPooling:
    def test_cached_multislot_shape(self):
        from oci.models.frozen_llm_pooler_extractor import FrozenLLMPoolerExtractor

        ext = FrozenLLMPoolerExtractor(
            skip_llm=True,
            cached_hidden_size=12,
            gated_attention_dim=8,
            projection_dim=16,
            attention_slots=3,
            dropout=0.0,
        )
        hidden = torch.randn(2, 5, 12)
        mask = torch.ones(2, 5)
        out = ext({"cached_hidden_states": hidden, "cached_attention_mask": mask})
        assert out.shape == (2, 16)
        shared = ext.extract_shared_forest_features({}, out)
        assert shared is out


class TestFrozenLLMTokenCNNExtractor:
    def test_cached_token_cnn_shape(self):
        from oci.models.frozen_llm_pooler_extractor import FrozenLLMTokenCNNExtractor

        ext = FrozenLLMTokenCNNExtractor(
            skip_llm=True,
            cached_hidden_size=12,
            gated_attention_dim=8,
            projection_dim=16,
            dropout=0.0,
        )
        hidden = torch.randn(2, 7, 12)
        mask = torch.ones(2, 7)
        out = ext({"cached_hidden_states": hidden, "cached_attention_mask": mask})
        assert out.shape == (2, 16)
        assert ext.get_state()["pooler_type"] == "token_cnn"


class TestFrozenLLMStatPoolerExtractor:
    def test_cached_stat_pooler_shared_features(self):
        from oci.models.frozen_llm_pooler_extractor import FrozenLLMStatPoolerExtractor

        ext = FrozenLLMStatPoolerExtractor(
            skip_llm=True,
            cached_hidden_size=12,
            gated_attention_dim=8,
            projection_dim=16,
            dropout=0.0,
        )
        hidden = torch.randn(2, 7, 12)
        mask = torch.ones(2, 7)
        out = ext({"cached_hidden_states": hidden, "cached_attention_mask": mask})
        shared = ext.extract_shared_forest_features(
            {"cached_hidden_states": hidden, "cached_attention_mask": mask},
            out,
        )
        assert out.shape == (2, 16)
        assert shared.shape == (2, 120)
        assert ext.get_state()["pooler_type"] == "stat_pooler"


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
