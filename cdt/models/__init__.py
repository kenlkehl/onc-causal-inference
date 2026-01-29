# cdt/models/__init__.py
"""Model components for CDT - causal inference from text."""

from .components import CrossAttentionAggregator
from .dragonnet import DragonNet
from .uplift import UpliftNet
from .rlearner import RLearnerNet
from .outcome_heads import OutcomeHeadsOnly, UpliftHeadsOnly
from .cnn_extractor import CNNFeatureExtractor, WordTokenizer
from .bert_extractor import BertFeatureExtractor
from .gru_extractor import GRUFeatureExtractor, AttentionPooling
from .confounder_extractor import ConfounderExtractor, HierarchicalConfounderExtractor, GRUHierarchicalConfounderExtractor
from .hierarchical_transformer_extractor import HierarchicalTransformerExtractor
from .hierarchical_gru_transformer_extractor import HierarchicalGRUTransformerExtractor
from .gated_mil_hierarchical_extractor import GatedMILHierarchicalExtractor
from .gated_mil_attention import GatedMILAttention, TaskSpecificConfounderWeighting, TokenLevelGatedPooling
from .sparse_attention import (
    sparse_softmax,
    top_k_attention,
    adaptive_top_k,
    SparseCrossAttention,
)
from .causal_text import CausalText, CausalCNNText  # CausalCNNText is deprecated alias
from .propensity_model import PropensityOnlyModel, PropensityNet, create_propensity_model_from_config
from .matched_pair_ite import (
    PropensityMatchingModel,
    MatchedPairOutcomeModel,
    EnhancedMatchedPairOutcomeModel,
    CombinedMatchedPairModel,
    EndToEndMatchedPairModel,
    EndToEndMatchedPairModelGRU
)
from .residual_cross_encoder import ResidualCrossEncoder

__all__ = [
    'CrossAttentionAggregator',
    'DragonNet',
    'UpliftNet',
    'RLearnerNet',
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
    'HierarchicalGRUTransformerExtractor',
    'GatedMILHierarchicalExtractor',
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
    'PropensityMatchingModel',
    'MatchedPairOutcomeModel',
    'EnhancedMatchedPairOutcomeModel',
    'CombinedMatchedPairModel',
    'EndToEndMatchedPairModel',
    'EndToEndMatchedPairModelGRU',
    'ResidualCrossEncoder',
]