# cdt/models/__init__.py
"""Model components for CDT - causal inference from text."""

from .components import CrossAttentionAggregator
from .dragonnet import DragonNet
from .uplift import UpliftNet
from .rlearner import RLearnerNet
from .traditional_logreg import TraditionalLogRegNet
from .dr_moce import DRMoCENet, NuisancePredictionBuffer, compute_dr_pseudo_outcome
from .cnn_extractor import CNNFeatureExtractor, WordTokenizer
from .bert_extractor import BertFeatureExtractor
from .gru_extractor import GRUFeatureExtractor, AttentionPooling
from .confounder_extractor import ConfounderExtractor, HierarchicalConfounderExtractor, GRUHierarchicalConfounderExtractor
from .hierarchical_transformer_extractor import HierarchicalTransformerExtractor
from .bert_pool_extractor import BertPoolExtractor
from .gated_mil_hierarchical_extractor import GatedMILHierarchicalExtractor
from .gru_transformer_mil_extractor import GRUTransformerMILExtractor
from .gru_pool_extractor import GRUPoolExtractor
from .gated_attention_pooling import GatedAttentionPooling
from .conv_pool_extractor import DilatedConvPoolExtractor
from .conv1d_transformer_hybrid_extractor import Conv1dTransformerHybridExtractor
from .transformer_pool_extractor import TransformerPoolExtractor
from .bert_cross_chunk_extractor import BertCrossChunkExtractor
from .llm_extractor import LLMFeatureExtractor
from .frozen_llm_pooler_extractor import FrozenLLMPoolerExtractor
from .gated_mil_attention import GatedMILAttention, TaskSpecificConfounderWeighting, TokenLevelGatedPooling
from .sparse_attention import (
    sparse_softmax,
    top_k_attention,
    adaptive_top_k,
    SparseCrossAttention,
)
from .numeric_features import NumericEmbedding, NumericFeatureVector, extract_numeric_patterns
from .explicit_confounder_featurizer import ExplicitConfounderFeaturizer, get_raw_confounder_features
from .intra_batch_contrastive import IntraBatchContrastiveLoss
from .hidden_state_cache import HiddenStateCache
from .gpu_hidden_state_store import GPUHiddenStateStore
from .causal_text import CausalText
from .propensity_model import PropensityOnlyModel, PropensityNet, create_propensity_model_from_config
from .extractor_factory import create_feature_extractor, create_feature_extractor_from_config
from .causal_forest_head import CausalForestHead, ECONML_AVAILABLE
from .causal_text_forest import CausalTextForest

__all__ = [
    'CrossAttentionAggregator',
    'DragonNet',
    'UpliftNet',
    'RLearnerNet',
    'TraditionalLogRegNet',
    'DRMoCENet',
    'NuisancePredictionBuffer',
    'compute_dr_pseudo_outcome',
    'CNNFeatureExtractor',
    'WordTokenizer',
    'BertFeatureExtractor',
    'GRUFeatureExtractor',
    'AttentionPooling',
    'ConfounderExtractor',
    'HierarchicalConfounderExtractor',
    'GRUHierarchicalConfounderExtractor',
    'HierarchicalTransformerExtractor',
    'BertPoolExtractor',
    'GatedMILHierarchicalExtractor',
    'GRUTransformerMILExtractor',
    'GRUPoolExtractor',
    'GatedAttentionPooling',
    'DilatedConvPoolExtractor',
    'Conv1dTransformerHybridExtractor',
    'TransformerPoolExtractor',
    'BertCrossChunkExtractor',
    'LLMFeatureExtractor',
    'FrozenLLMPoolerExtractor',
    'GatedMILAttention',
    'TaskSpecificConfounderWeighting',
    'TokenLevelGatedPooling',
    'sparse_softmax',
    'top_k_attention',
    'adaptive_top_k',
    'SparseCrossAttention',
    'IntraBatchContrastiveLoss',
    'CausalText',
    'PropensityOnlyModel',
    'PropensityNet',
    'create_propensity_model_from_config',
    'CausalForestHead',
    'NumericEmbedding',
    'NumericFeatureVector',
    'extract_numeric_patterns',
    'ExplicitConfounderFeaturizer',
    'get_raw_confounder_features',
    'HiddenStateCache',
    'GPUHiddenStateStore',
    'CausalTextForest',
    'ECONML_AVAILABLE',
    'create_feature_extractor',
    'create_feature_extractor_from_config',
]