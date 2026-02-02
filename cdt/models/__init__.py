# cdt/models/__init__.py
"""Model components for CDT - causal inference from text."""

from .components import CrossAttentionAggregator
from .dragonnet import DragonNet
from .uplift import UpliftNet
from .rlearner import RLearnerNet
from .traditional_logreg import TraditionalLogRegNet
from .outcome_heads import OutcomeHeadsOnly, UpliftHeadsOnly
from .cnn_extractor import CNNFeatureExtractor, WordTokenizer
from .bert_extractor import BertFeatureExtractor
from .gru_extractor import GRUFeatureExtractor, AttentionPooling
from .confounder_extractor import ConfounderExtractor, HierarchicalConfounderExtractor, GRUHierarchicalConfounderExtractor
from .hierarchical_transformer_extractor import HierarchicalTransformerExtractor
from .gated_mil_hierarchical_extractor import GatedMILHierarchicalExtractor
from .gru_transformer_mil_extractor import GRUTransformerMILExtractor
from .gru_pool_extractor import GRUPoolExtractor, GatedAttentionPooling
from .llm_extractor import LLMFeatureExtractor
from .gated_mil_attention import GatedMILAttention, TaskSpecificConfounderWeighting, TokenLevelGatedPooling
from .sparse_attention import (
    sparse_softmax,
    top_k_attention,
    adaptive_top_k,
    SparseCrossAttention,
)
from .numeric_features import NumericEmbedding, NumericFeatureVector, extract_numeric_patterns
from .causal_text import CausalText, CausalCNNText  # CausalCNNText is deprecated alias
from .propensity_model import PropensityOnlyModel, PropensityNet, create_propensity_model_from_config
from .causal_forest_head import CausalForestHead, ECONML_AVAILABLE
from .causal_text_forest import CausalTextForest

__all__ = [
    'CrossAttentionAggregator',
    'DragonNet',
    'UpliftNet',
    'RLearnerNet',
    'TraditionalLogRegNet',
    'OutcomeHeadsOnly',
    'UpliftHeadsOnly',
    'CNNFeatureExtractor',
    'WordTokenizer',
    'BertFeatureExtractor',
    'GRUFeatureExtractor',
    'AttentionPooling',
    'ConfounderExtractor',
    'HierarchicalConfounderExtractor',
    'GRUHierarchicalConfounderExtractor',
    'HierarchicalTransformerExtractor',
    'GatedMILHierarchicalExtractor',
    'GRUTransformerMILExtractor',
    'GRUPoolExtractor',
    'GatedAttentionPooling',
    'LLMFeatureExtractor',
    'GatedMILAttention',
    'TaskSpecificConfounderWeighting',
    'TokenLevelGatedPooling',
    'sparse_softmax',
    'top_k_attention',
    'adaptive_top_k',
    'SparseCrossAttention',
    'CausalText',
    'CausalCNNText',  # Deprecated alias for CausalText
    'PropensityOnlyModel',
    'PropensityNet',
    'create_propensity_model_from_config',
    'CausalForestHead',
    'NumericEmbedding',
    'NumericFeatureVector',
    'extract_numeric_patterns',
    'CausalTextForest',
    'ECONML_AVAILABLE',
]