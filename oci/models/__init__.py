# oci/models/__init__.py
"""Model components for causal inference from text."""

from .dragonnet import DragonNet
from .rlearner import RLearnerNet
from .frozen_llm_pooler_extractor import FrozenLLMPoolerExtractor
from .hierarchical_llm_extractor import HierarchicalLLMExtractor
from .hierarchical_cnn_extractor import HierarchicalCNNExtractor
from .hierarchical_gru_extractor import HierarchicalGRUExtractor
from .simple_cnn_extractor import SimpleCNNExtractor
from .gated_attention_pooling import GatedAttentionPooling
from .learned_tokenizer import LearnedTokenizer
from .text_chunking import chunk_token_ids, pad_and_batch_chunks
from .explicit_feature_featurizer import (
    ExplicitFeatureFeaturizer,
    filter_specs_by_role,
    get_raw_explicit_features,
    get_raw_explicit_feature_matrices,
)
ExplicitConfounderFeaturizer = ExplicitFeatureFeaturizer
get_raw_confounder_features = get_raw_explicit_features
from .hidden_state_cache import HiddenStateCache
from .gpu_hidden_state_store import GPUHiddenStateStore
from .causal_text import CausalText
from .propensity_model import PropensityOnlyModel, PropensityNet, create_propensity_model_from_config
from .extractor_factory import create_feature_extractor, create_feature_extractor_from_config
from .causal_forest_head import CausalForestHead, ECONML_AVAILABLE
from .causal_text_forest import CausalTextForest
from .contrastive_causal_text_forest import (
    ContrastiveCausalTextForest,
    MatchedContrastiveEffectHead,
    grad_reverse,
)

__all__ = [
    'DragonNet',
    'RLearnerNet',
    'FrozenLLMPoolerExtractor',
    'HierarchicalLLMExtractor',
    'HierarchicalCNNExtractor',
    'HierarchicalGRUExtractor',
    'SimpleCNNExtractor',
    'GatedAttentionPooling',
    'LearnedTokenizer',
    'chunk_token_ids',
    'pad_and_batch_chunks',
    'CausalText',
    'PropensityOnlyModel',
    'PropensityNet',
    'create_propensity_model_from_config',
    'CausalForestHead',
    'ExplicitFeatureFeaturizer',
    'ExplicitConfounderFeaturizer',
    'filter_specs_by_role',
    'get_raw_explicit_features',
    'get_raw_confounder_features',
    'get_raw_explicit_feature_matrices',
    'HiddenStateCache',
    'GPUHiddenStateStore',
    'CausalTextForest',
    'ContrastiveCausalTextForest',
    'MatchedContrastiveEffectHead',
    'grad_reverse',
    'ECONML_AVAILABLE',
    'create_feature_extractor',
    'create_feature_extractor_from_config',
]
