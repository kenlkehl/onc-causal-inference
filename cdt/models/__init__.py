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
from .confounder_extractor import ConfounderExtractor
from .sparse_attention import (
    sparse_softmax,
    top_k_attention,
    adaptive_top_k,
    SparseCrossAttention,
)
from .causal_cnn import CausalCNNText
from .propensity_model import PropensityOnlyModel, PropensityNet, create_propensity_model_from_config

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
    'sparse_softmax',
    'top_k_attention',
    'adaptive_top_k',
    'SparseCrossAttention',
    'CausalCNNText',
    'PropensityOnlyModel',
    'PropensityNet',
    'create_propensity_model_from_config',
]