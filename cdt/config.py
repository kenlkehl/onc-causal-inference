# cdt/config.py
"""Configuration classes for CDT experiments - CNN-based approach."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import hashlib


# =============================================================================
# EXPLICIT CONFOUNDER EXTRACTION CONFIGURATION
# =============================================================================

@dataclass
class ExplicitConfounderSpec:
    """Specification for a single explicit confounder to extract from clinical text."""
    name: str  # e.g., "performance_status"
    type: str  # "categorical" or "continuous"
    categories: Optional[List[str]] = None  # For categorical only (e.g., ["0", "1", "2", "3", "4"])
    description: Optional[str] = None  # Used in LLM prompt (e.g., "ECOG performance status")

    def __post_init__(self):
        if self.type not in ("categorical", "continuous"):
            raise ValueError(f"type must be 'categorical' or 'continuous', got '{self.type}'")
        if self.type == "categorical" and not self.categories:
            raise ValueError(f"categories required for categorical confounder '{self.name}'")


@dataclass
class ExplicitConfounderExtractionConfig:
    """Configuration for LLM-based confounder extraction from clinical text.

    This enables extraction of explicit confounder variables (e.g., performance status,
    disease stage) from unstructured clinical text using an LLM. The extracted values
    are then featurized and concatenated to text embeddings before the causal head.
    """
    enabled: bool = False
    confounders: List[ExplicitConfounderSpec] = field(default_factory=list)

    # vLLM mode: "server", "start_server", or "python_api"
    # - "server": Connect to running vLLM OpenAI-compatible server
    # - "start_server": Start vLLM server subprocess for the job, then connect
    # - "python_api": Use vLLM Python API directly (no server, in-process)
    vllm_mode: str = "server"
    vllm_server_url: Optional[str] = "http://localhost:8000/v1"
    vllm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_download_dir: Optional[str] = None  # Model download directory

    # Extraction settings
    extraction_batch_size: int = 32
    extraction_max_retries: int = 3  # Retries per patient before marking as missing
    extraction_temperature: float = 0.0  # LLM temperature (0 for deterministic)
    extraction_max_tokens: int = 1024  # Max tokens for LLM response

    # Caching
    cache_enabled: bool = True  # Cache extraction results to disk
    cache_dir: Optional[str] = None  # Directory for cache files (default: alongside dataset)

    # Featurizer settings (for neural models only)
    featurizer_output_dim: int = 64
    featurizer_hidden_dim: int = 128
    featurizer_dropout: float = 0.1


# =============================================================================
# MATCHING ANALYSIS CONFIGURATION (used as post-hoc analysis with DragonNet)
# =============================================================================

@dataclass
class MatchingAnalysisConfig:
    """Configuration for propensity score matching analysis (post-hoc)."""

    # Whether to run PSM analysis using DragonNet's propensity scores
    enabled: bool = True

    # Matching method: 'nearest', 'optimal', 'caliper'
    method: str = "nearest"

    # Caliper (maximum allowed distance for a match)
    # None = no caliper
    caliper: Optional[float] = 0.2

    # Scale for caliper: 'propensity', 'logit', 'std'
    # 'std' means caliper is in standard deviations of logit propensity
    caliper_scale: str = "std"

    # Matching ratio (1:k matching)
    ratio: int = 1

    # Whether to match with replacement
    replacement: bool = False

    # Number of bootstrap iterations for confidence intervals
    n_bootstrap: int = 1000

    # Confidence level for intervals
    ci_level: float = 0.95


# =============================================================================
# CAUSAL FOREST CONFIGURATION
# =============================================================================

@dataclass
class CausalForestConfig:
    """Configuration for causal forest head (used with model_type="causal_forest").

    Note: Nuisance functions (propensity, outcome) are estimated using sklearn
    random forests on the neural network's learned features. The neural network's
    key contribution is the learned text representation that captures confounders.
    """

    # Number of trees in the causal forest (must be divisible by 4 for econml)
    n_estimators: int = 100

    # Maximum depth of trees (None = unlimited)
    max_depth: Optional[int] = None

    # Minimum samples per leaf
    min_samples_leaf: int = 5

    # Feature subset strategy for splitting
    max_features: str = "sqrt"

    # Use honest estimation (sample splitting within trees)
    honest: bool = True

    # Enable inference for confidence intervals
    inference: bool = True

    # R-learner representation training: adds a τ head and R-loss to Stage 1
    # When True, Stage 1 trains with propensity + outcome + R-learner losses
    # to encourage embeddings to capture treatment effect heterogeneity
    use_rlearner_representation: bool = False

    # Weight for R-learner loss during representation training
    gamma_rlearner: float = 1.0

    # Dual extractor mode for R-learner representation training
    # When enabled with use_rlearner_representation=True, uses two independent feature extractors:
    # - Nuisance extractor: for propensity e(X) and marginal outcome m(X)
    # - Effect extractor: for treatment effect τ(X)
    # In dual mode, Stage 2 uses the effect extractor's features (optimized for τ)
    # Memory note: approximately doubles feature extraction compute
    rlearner_dual_extractors: bool = False


def normalize_feature_extractor_type(feature_type: str) -> str:
    """
    Normalize feature extractor type to one of: "cnn", "bert", "gru", "confounder",
    "hierarchical_transformer", "gated_mil_hierarchical", "gru_transformer_mil",
    "gru_pool", "bert_pool", "bert_cross_chunk", or "llm".

    This handles variants like "modernbert" which should be treated as "bert".

    Args:
        feature_type: The raw feature extractor type string

    Returns:
        Normalized type: "cnn", "bert", "gru", "confounder", "hierarchical_transformer",
        "gated_mil_hierarchical", "gru_transformer_mil", "gru_pool", "bert_pool",
        "bert_cross_chunk", or "llm"
    """
    if feature_type is None:
        return "cnn"

    feature_type_lower = feature_type.lower()

    # Check for LLM extractor (decoder-only with last token embedding)
    if feature_type_lower in ("llm", "gpt", "qwen", "llama", "decoder"):
        return "llm"

    # Check for BERT Pool extractor (BERT [CLS] per chunk + transformer + gated attention pooling)
    if feature_type_lower in ("bert_pool", "bert_pool_transformer"):
        return "bert_pool"

    # Check for BERT Cross-Chunk extractor (token-level cross-chunk attention)
    if feature_type_lower in ("bert_cross_chunk", "cross_chunk", "cross_chunk_bert"):
        return "bert_cross_chunk"

    # Check for GRU-Pool extractor (BiGRU + transformer + gated attention pooling)
    if feature_type_lower in ("gru_pool", "gru_pool_transformer"):
        return "gru_pool"

    # Check for GRU-Transformer-MIL extractor
    if feature_type_lower in ("gru_mil", "gru_transformer_mil", "gru_mil_hierarchical"):
        return "gru_transformer_mil"

    # Check for gated MIL hierarchical extractor
    if feature_type_lower in ("gated_mil", "gated_mil_hierarchical", "mil", "gated_mil_hier"):
        return "gated_mil_hierarchical"

    # Check for hierarchical transformer extractor
    if feature_type_lower in ("hierarchical_transformer", "hier_transformer", "hierarchical"):
        return "hierarchical_transformer"

    # Check for confounder extractor
    if feature_type_lower in ("confounder", "perceiver", "sentence_perceiver"):
        return "confounder"

    # Check for GRU variants
    if feature_type_lower == "gru":
        return "gru"

    # Check for BERT variants (bert, modernbert, clinicalbert, etc.)
    if "bert" in feature_type_lower:
        return "bert"

    # Check for CNN
    if feature_type_lower == "cnn":
        return "cnn"

    # Default to CNN for backward compatibility
    return "cnn"


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture."""
    model_type: str = "dragonnet"  # "dragonnet", "uplift", "rlearner", "traditional_logreg", "causal_forest", or "dr_moce"

    # Feature extractor type: "cnn", "bert", or "gru"
    feature_extractor_type: str = "cnn"

    # CNN architecture (used when feature_extractor_type="cnn")
    cnn_embedding_dim: int = 128  # Word embedding dimension
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 4, 5, 7])
    cnn_dropout: float = 0.1
    cnn_max_length: int = 2048  # Max sequence length in tokens (words)
    cnn_min_word_freq: int = 2  # Minimum word frequency for vocabulary
    cnn_max_vocab_size: int = 50000  # Maximum vocabulary size

    # CNN embedding initialization
    cnn_use_random_embedding_init: bool = False  # If True, use random init (ignore cnn_init_embeddings_from)
    cnn_init_embeddings_from: Optional[str] = None  # e.g., "emilyalsentzer/Bio_ClinicalBERT"
    cnn_freeze_embeddings: bool = False  # Whether to freeze BERT-initialized embeddings

    # CNN filter initialization: specify each filter type separately
    # Total filters per kernel = max(explicit_count_per_kernel) + kmeans + random
    # Dict mapping kernel_size (as string in JSON) to list of concept phrases
    cnn_explicit_filter_concepts: Optional[Dict[str, List[str]]] = None
    cnn_num_kmeans_filters: int = 64  # Number of k-means derived filters per kernel size
    cnn_num_random_filters: int = 0  # Number of randomly initialized filters per kernel size
    cnn_freeze_filters: bool = False  # Whether to freeze CNN filters after initialization

    # BERT architecture (used when feature_extractor_type="bert")
    bert_model_name: str = "bert-base-uncased"  # HuggingFace model name or path
    bert_max_length: int = 512  # Max sequence length in subword tokens
    bert_projection_dim: Optional[int] = 128  # Projection dim after CLS; None = use raw hidden size
    bert_dropout: float = 0.1  # Dropout rate for projection layer
    bert_freeze_encoder: bool = False  # If True, freeze transformer (only train projection + DragonNet)
    bert_gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory efficiency

    # GRU architecture (used when feature_extractor_type="gru")
    # BiGRU + attention is O(N) vs BERT's O(N^2), efficient for long sequences
    gru_embedding_dim: int = 256  # Word embedding dimension (can be initialized from BERT)
    gru_hidden_dim: int = 256  # Hidden state dimension per direction
    gru_num_layers: int = 2  # Number of stacked GRU layers
    gru_dropout: float = 0.1  # Dropout rate
    gru_bidirectional: bool = True  # Use bidirectional GRU
    gru_attention_dim: Optional[int] = None  # Attention hidden dimension (default: 2*hidden_dim if bidirectional)
    gru_projection_dim: Optional[int] = 128  # Output projection dimension
    gru_max_length: int = 8192  # Max sequence length in tokens (words)
    gru_min_word_freq: int = 2  # Minimum word frequency for vocabulary
    gru_max_vocab_size: int = 50000  # Maximum vocabulary size
    gru_init_embeddings_from: Optional[str] = None  # e.g., "emilyalsentzer/Bio_ClinicalBERT"
    gru_freeze_embeddings: bool = False  # Whether to freeze initialized embeddings

    # Confounder extractor architecture (used when feature_extractor_type="confounder")
    # Perceiver-style cross-attention with sparse attention for long document understanding
    confounder_num_latents: int = 4  # Number of learnable latent confounder vectors
    confounder_explicit_texts: Optional[List[str]] = None  # Explicit confounder phrases (e.g., ["metastatic sites", "performance status"])
    confounder_value_dim: int = 128  # Dimension per confounder (and output dimension)
    confounder_sentence_model: str = "all-MiniLM-L6-v2"  # Sentence transformer model for chunk encoding
    confounder_freeze_encoder: bool = True  # Whether to freeze encoder
    confounder_max_chunks: int = 100  # Maximum chunks per document
    confounder_chunk_size: int = 128  # Tokens per chunk
    confounder_chunk_overlap: int = 32  # Overlapping tokens between chunks
    confounder_num_heads: int = 4  # Number of attention heads
    confounder_num_iterations: int = 2  # Number of iterative refinement passes
    confounder_use_self_attention: bool = True  # Whether latents attend to each other
    confounder_sparse_attention: bool = True  # Use sparse attention (entmax/top-k)
    confounder_sparse_method: str = "entmax"  # Sparsity method: "entmax", "topk", "softmax"
    confounder_sparse_alpha: float = 1.5  # Alpha for entmax (1.5=entmax15, 2.0=sparsemax)
    confounder_top_k: int = 5  # K for top-k attention method
    confounder_dropout: float = 0.1  # Dropout rate

    # Hierarchical confounder extractor (token-level attention)
    # When enabled, uses per-chunk BERT encoding + token-level cross-attention
    # This preserves fine-grained token signal that chunk embeddings may lose
    confounder_hierarchical: bool = False  # Enable hierarchical token-level attention
    confounder_token_encoder: str = "distilbert-base-uncased"  # Token encoder for hierarchical mode
    confounder_freeze_token_encoder: bool = True  # Whether to freeze token encoder

    # GRU-based hierarchical confounder extractor (learns from scratch)
    # When enabled, uses BiGRU instead of pretrained BERT for chunk encoding
    # All parameters (embeddings, GRU, attention) learn together via causal loss
    confounder_use_gru: bool = False  # Use GRU instead of BERT for chunk encoding
    confounder_gru_embedding_dim: int = 128  # Word embedding dimension
    confounder_gru_hidden_dim: int = 128  # GRU hidden state dimension per direction
    confounder_gru_num_layers: int = 1  # Number of stacked GRU layers
    confounder_gru_bidirectional: bool = True  # Use bidirectional GRU
    confounder_gru_dropout: float = 0.1  # Dropout rate
    confounder_gru_max_vocab: int = 50000  # Maximum vocabulary size
    confounder_gru_min_word_freq: int = 2  # Minimum word frequency for vocabulary
    confounder_gru_chunk_size: int = 128  # Tokens per chunk for GRU mode
    confounder_gru_chunk_overlap: int = 32  # Overlapping tokens between chunks

    # Hierarchical Transformer extractor (used when feature_extractor_type="hierarchical_transformer")
    # Simple hierarchical approach: chunk-level BERT encoding + transformer pooling
    hier_transformer_sentence_model: str = "prajjwal1/bert-tiny"  # Chunk encoder model
    hier_transformer_freeze_sentence_encoder: bool = True  # Whether to freeze encoder
    hier_transformer_max_chunks: int = 100  # Maximum chunks per document
    hier_transformer_chunk_size: int = 128  # Tokens per chunk
    hier_transformer_chunk_overlap: int = 32  # Overlapping tokens between chunks
    hier_transformer_num_layers: int = 2  # Number of transformer layers for pooling
    hier_transformer_num_heads: int = 4  # Number of attention heads
    hier_transformer_dim: int = 256  # Hidden dimension for transformer layers
    hier_transformer_dropout: float = 0.1  # Dropout rate
    hier_transformer_projection_dim: int = 128  # Final output dimension

    # BERT Pool extractor (used when feature_extractor_type="bert_pool")
    # BERT [CLS] per chunk + transformer cross-chunk context + gated attention pooling
    # Like hierarchical_transformer but with gated pooling instead of [POOL] token,
    # BERT unfrozen by default, and optional random weight initialization
    bert_pool_sentence_model: str = "prajjwal1/bert-tiny"  # Chunk encoder model
    bert_pool_freeze_sentence_encoder: bool = False  # Unfrozen by default for end-to-end fine-tuning
    bert_pool_use_pretrained: bool = True  # False = random init (architecture + tokenizer only)
    bert_pool_max_chunks: int = 100  # Maximum chunks per document
    bert_pool_chunk_size: int = 128  # Tokens per chunk
    bert_pool_chunk_overlap: int = 32  # Overlapping tokens between chunks
    bert_pool_transformer_layers: int = 2  # Number of transformer layers for cross-chunk processing
    bert_pool_transformer_heads: int = 4  # Number of attention heads in transformer
    bert_pool_transformer_dim: int = 256  # Hidden dimension for transformer layers
    bert_pool_transformer_dropout: float = 0.1  # Dropout rate for transformer layers
    bert_pool_gated_attention_dim: int = 128  # Hidden dimension for gated attention pooling
    bert_pool_projection_dim: int = 128  # Final output dimension

    # BERT Cross-Chunk extractor (used when feature_extractor_type="bert_cross_chunk")
    # Token-level cross-chunk attention: each chunk's tokens attend to global [CLS] embeddings
    # from all other chunks, giving tokens document-wide context
    bcc_sentence_model: str = "prajjwal1/bert-tiny"  # Chunk encoder model
    bcc_freeze_sentence_encoder: bool = False  # Whether to freeze BERT (unfrozen for end-to-end fine-tuning)
    bcc_max_chunks: int = 100  # Maximum chunks per document
    bcc_chunk_size: int = 128  # Tokens per chunk
    bcc_chunk_overlap: int = 32  # Overlapping tokens between chunks
    bcc_num_cross_layers: int = 2  # Number of cross-chunk transformer layers
    bcc_num_attention_heads: int = 4  # Number of attention heads in cross-chunk layers
    bcc_cross_chunk_dim: int = 256  # Hidden dimension for cross-chunk transformer
    bcc_cross_chunk_dropout: float = 0.1  # Dropout in cross-chunk layers
    bcc_gated_attention_dim: int = 128  # Hidden dimension for gated attention pooling
    bcc_projection_dim: int = 128  # Final output dimension

    # Gated MIL Hierarchical extractor (used when feature_extractor_type="gated_mil_hierarchical")
    # Uses gated MIL attention (tanh * sigmoid gating) with K confounder queries
    # Task-specific weighting of shared confounders for propensity, tau, outcome
    gated_mil_sentence_model: str = "prajjwal1/bert-tiny"  # Chunk encoder model
    gated_mil_freeze_sentence_encoder: bool = True  # Whether to freeze encoder
    gated_mil_max_chunks: int = 100  # Maximum chunks per document
    gated_mil_chunk_size: int = 128  # Tokens per chunk
    gated_mil_chunk_overlap: int = 32  # Overlapping tokens between chunks
    gated_mil_hidden_dim: int = 128  # Hidden dimension for gated attention
    gated_mil_num_confounders: int = 4  # Number of confounder queries (K)
    gated_mil_dropout: float = 0.1  # Dropout rate
    gated_mil_projection_dim: int = 128  # Final output dimension
    # Token-level hierarchical mode (preserves fine-grained signal like "PS 0" vs "PS 2")
    gated_mil_hierarchical: bool = False  # Enable token-level gated pooling
    gated_mil_token_hidden_dim: int = 64  # Hidden dimension for token-level gating
    # Mean pooling option (use mean over tokens instead of [CLS] token)
    gated_mil_use_mean_pooling: bool = False  # Use mean pooling instead of [CLS] for chunk embeddings

    # GRU-Transformer-MIL extractor (used when feature_extractor_type="gru_transformer_mil")
    # Combines learned BiGRU chunk encoding + transformer cross-chunk + gated MIL attention
    # Learns entirely from scratch (no pretrained encoder) - requires fit_tokenizer()
    gru_mil_embedding_dim: int = 128  # Word embedding dimension
    gru_mil_gru_hidden_dim: int = 128  # GRU hidden state dimension per direction
    gru_mil_gru_num_layers: int = 1  # Number of stacked GRU layers
    gru_mil_gru_bidirectional: bool = True  # Use bidirectional GRU
    gru_mil_gru_dropout: float = 0.1  # Dropout rate for GRU
    gru_mil_max_chunks: int = 100  # Maximum chunks per document
    gru_mil_chunk_size: int = 128  # Tokens per chunk
    gru_mil_chunk_overlap: int = 32  # Overlapping tokens between chunks
    gru_mil_transformer_layers: int = 2  # Number of transformer layers for cross-chunk processing
    gru_mil_transformer_heads: int = 4  # Number of attention heads in transformer
    gru_mil_transformer_dim: int = 256  # Hidden dimension for transformer layers
    gru_mil_num_confounders: int = 4  # Number of confounder queries (K)
    gru_mil_mil_hidden_dim: int = 128  # Hidden dimension for gated MIL attention
    gru_mil_projection_dim: int = 128  # Final output dimension
    gru_mil_max_vocab: int = 50000  # Maximum vocabulary size
    gru_mil_min_word_freq: int = 2  # Minimum word frequency for vocabulary

    # GRU-Pool extractor (used when feature_extractor_type="gru_pool")
    # Combines BiGRU chunk encoding + transformer cross-chunk + gated attention pooling
    # Learns entirely from scratch (no pretrained encoder) - requires fit_tokenizer()
    # Produces single feature vector (no task-specific weighting)
    gru_pool_embedding_dim: int = 128  # Word embedding dimension
    gru_pool_gru_hidden_dim: int = 128  # GRU hidden state dimension per direction
    gru_pool_gru_num_layers: int = 1  # Number of stacked GRU layers
    gru_pool_gru_bidirectional: bool = True  # Use bidirectional GRU
    gru_pool_gru_dropout: float = 0.1  # Dropout rate for GRU
    gru_pool_max_chunks: int = 100  # Maximum chunks per document
    gru_pool_chunk_size: int = 128  # Tokens per chunk
    gru_pool_chunk_overlap: int = 32  # Overlapping tokens between chunks
    gru_pool_transformer_layers: int = 2  # Number of transformer layers for cross-chunk processing
    gru_pool_transformer_heads: int = 4  # Number of attention heads in transformer
    gru_pool_transformer_dim: int = 256  # Hidden dimension for transformer layers
    gru_pool_gated_attention_dim: int = 128  # Hidden dimension for gated attention pooling
    gru_pool_projection_dim: int = 128  # Final output dimension
    gru_pool_max_vocab: int = 50000  # Maximum vocabulary size
    gru_pool_min_word_freq: int = 2  # Minimum word frequency for vocabulary

    # CLAM-style instance-level loss config (for GRU-Pool extractor)
    # CLAM (Lu et al., Nature BME 2021) supervises top-attended chunks with document labels
    # This forces top-attended chunks to be predictive of treatment/outcome
    clam_enabled: bool = False  # Master switch for CLAM instance-level loss
    clam_num_instances: int = 5  # B: number of top-attended chunks to supervise
    clam_instance_hidden_dim: int = 64  # Hidden dimension for instance causal head (smaller than doc head)

    # R-Learner dual extractor mode
    # When enabled with model_type="rlearner", uses two independent feature extractors:
    # - Nuisance extractor: shared for propensity e(X) and marginal outcome m(X)
    # - Effect extractor: dedicated to treatment effect τ(X)
    # This prevents gradient interference between confounder learning (nuisance) and
    # effect modifier learning (τ). Memory note: approximately doubles feature extraction compute.
    rlearner_dual_extractors: bool = False

    # Uplift dual extractor mode
    # When enabled with model_type="uplift", uses two independent feature extractors:
    # - Nuisance extractor: shared for propensity e(X) and baseline outcome Y0(X)
    # - Effect extractor: dedicated to treatment effect τ(X)
    # Y1 is computed as Y0 + τ. This separation prevents gradient interference between
    # confounder learning (nuisance) and effect modifier learning (τ).
    # Memory note: approximately doubles feature extraction compute.
    uplift_dual_extractors: bool = False

    # DR-MoCE (Doubly-Robust Mixture of Causal Experts) config
    # Used when model_type="dr_moce"
    dr_moce_num_experts: int = 8  # Number of effect expert heads (K)
    dr_moce_router_temperature: float = 1.0  # Softmax temperature (lower = sharper routing)
    dr_moce_propensity_clip: float = 0.01  # Clip e(X) to [clip, 1-clip] in pseudo-outcome
    dr_moce_het_weight: float = 0.1  # Weight for heterogeneity regularization (expert specialization)
    dr_moce_balance_weight: float = 0.01  # Weight for load balancing loss
    dr_moce_crossfit_buffer_size: int = 1024  # Nuisance prediction buffer size (0 = disabled)

    # LLM Feature Extractor (decoder-only with last token embedding)
    # Pretrained tokenizer is always used
    llm_model_name: str = "Qwen/Qwen3-0.6B-Base"  # HuggingFace model name for architecture/tokenizer
    llm_max_length: int = 8192  # Max sequence length (up to 32768 for Qwen3)
    llm_projection_dim: Optional[int] = 128  # Output projection dim (None = use raw hidden size)
    llm_dropout: float = 0.1  # Dropout rate for projection layers
    llm_gradient_checkpointing: bool = True  # Enable gradient checkpointing for memory efficiency
    llm_use_pretrained: bool = False  # If True, load pretrained weights; if False, random init

    # Numeric feature extraction (magnitude-aware number encoding)
    numeric_features_enabled: bool = False  # Enable numeric feature extraction
    numeric_embedding_dim: int = 32  # Dimension of numeric embeddings
    numeric_magnitude_bins: int = 8  # Number of log-scale magnitude bins
    numeric_type_categories: int = 10  # Number of numeric type categories

    # Intra-batch contrastive learning
    # Clusters samples by representation similarity, then applies SupCon loss within clusters
    # to improve confounder detection among similar patients
    contrastive_enabled: bool = False  # Master switch for contrastive loss
    contrastive_num_clusters: int = 4  # Number of K-means clusters (K)
    contrastive_temperature: float = 0.1  # SupCon temperature (lower = sharper)
    contrastive_label_mode: str = "joint"  # "treatment", "outcome", or "joint" (T*2+Y)
    contrastive_projection_dim: int = 64  # Projection head output dimension
    contrastive_min_cluster_size: int = 2  # Minimum samples per cluster
    contrastive_clustering_method: str = "kmeans"  # "kmeans" or "random"

    # Causal head dimensions (applies to all causal heads: DragonNet, RLearner, UpliftNet, etc.)
    causal_head_representation_dim: int = 128
    causal_head_hidden_outcome_dim: int = 64
    causal_head_dropout: float = 0.2  # Dropout in causal head representation and outcome layers

    # Causal Forest config (used when model_type="causal_forest")
    causal_forest: CausalForestConfig = field(default_factory=CausalForestConfig)

    def get_num_filters_per_kernel(self) -> int:
        """
        Compute total number of filters per kernel size.

        Total = max(explicit concepts per kernel) + kmeans + random
        This ensures all kernel sizes have the same number of output channels.
        """
        max_explicit = 0
        if self.cnn_explicit_filter_concepts:
            for concepts in self.cnn_explicit_filter_concepts.values():
                max_explicit = max(max_explicit, len(concepts))
        return max_explicit + self.cnn_num_kmeans_filters + self.cnn_num_random_filters


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    lr_schedule: str = "linear"
    epochs: int = 50
    batch_size: int = 8
    alpha_propensity: float = 1.0
    beta_targreg: float = 0.1
    gamma_rlearner: float = 1.0  # Weight for R-learner loss (when model_type="rlearner")
    gamma_dr: float = 1.0  # Weight for DR effect loss (when model_type="dr_moce")
    # Regularization options
    weight_decay: float = 0.01  # L2 regularization (AdamW decoupled weight decay)
    gradient_clip_norm: float = 1.0  # Max gradient norm (0 to disable)
    label_smoothing: float = 0.0  # Label smoothing for BCE (0 to disable)
    # Advanced training options for improving tau learning
    stop_grad_propensity: bool = False  # Detach features before propensity loss (prevents propensity from dominating representation)
    attention_entropy_weight: float = 0.0  # Weight for attention entropy regularization (encourages focused attention)
    # CLAM instance-level loss weight (only used when clam_enabled=True in architecture config)
    clam_instance_weight: float = 0.5  # Weight for CLAM instance-level loss
    # Intra-batch contrastive loss weight (only used when contrastive_enabled=True)
    contrastive_weight: float = 0.1  # Weight for contrastive loss term


@dataclass
class PropensityTrimmingConfig:
    """Configuration for propensity score trimming before causal inference.

    When enabled, trains a propensity-only model using k-fold cross-validation
    to generate out-of-sample propensity scores, then trims the dataset by
    removing patients with propensity scores outside the specified bounds.
    This helps enforce positivity assumption for causal inference.
    """
    enabled: bool = False  # Whether to trim by propensity before DragonNet training
    min_propensity: float = 0.1  # Remove patients with P(T=1|X) below this
    max_propensity: float = 0.9  # Remove patients with P(T=1|X) above this
    cv_folds: int = 5  # Number of CV folds for propensity model training
    propensity_epochs: int = 20  # Training epochs for propensity model
    propensity_learning_rate: float = 1e-4  # Learning rate for propensity model
    propensity_batch_size: int = 8  # Batch size for propensity model


@dataclass
class OutcomeModelConfig:
    """Configuration for standalone outcome model training.

    When enabled, trains an outcome-only model using k-fold cross-validation
    to generate out-of-sample outcome predictions. This helps assess the
    prognostic signal in the data before DragonNet training.
    Unlike propensity trimming, this does NOT trim the dataset.
    """
    enabled: bool = False  # Whether to train outcome model before DragonNet
    cv_folds: int = 5  # Number of CV folds for outcome model training
    outcome_epochs: int = 20  # Training epochs for outcome model
    outcome_learning_rate: float = 1e-4  # Learning rate for outcome model
    outcome_batch_size: int = 8  # Batch size for outcome model


@dataclass
class PlasmodeConfig:
    """Configuration for plasmode simulation."""
    generation_mode: str = "phi_linear"
    preserve_observed_treatments: bool = True
    baseline_control_outcome_rate: float = 0.20
    target_ate_prob: float = 0.10  # Target ATE on probability scale (e.g., 0.10 = 10% increase in outcome probability)
    outcome_heterogeneity_scale: float = 1.0
    ite_heterogeneity_scale: float = 1.0
    deep_nonlinear_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    deep_nonlinear_dropout: float = 0.0
    uplift_hidden_dims: List[int] = field(default_factory=list)
    uplift_activation: str = "relu"
    uplift_dropout: float = 0.0


@dataclass
class AppliedInferenceConfig:
    """Configuration for applied inference on real data."""
    dataset_path: str = ""
    text_column: str = "clinical_text"
    outcome_column: str = "outcome_indicator"
    treatment_column: str = "treatment_indicator"
    split_column: str = "split"
    cv_folds: int = 5  # Number of CV folds (0 or 1 = fixed split)
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    propensity_trimming: PropensityTrimmingConfig = field(default_factory=PropensityTrimmingConfig)
    outcome_model: OutcomeModelConfig = field(default_factory=OutcomeModelConfig)
    use_pretrained_weights: bool = False  # Not used for CNN, kept for API compatibility
    skip: bool = False  # Skip applied inference, go straight to plasmode

    # PSM analysis configuration (uses DragonNet's propensity scores)
    matching_analysis: MatchingAnalysisConfig = field(default_factory=MatchingAnalysisConfig)

    # Explicit confounder extraction configuration (LLM-based)
    explicit_confounders: ExplicitConfounderExtractionConfig = field(default_factory=ExplicitConfounderExtractionConfig)


@dataclass
class PlasmodeExperimentConfig:
    """Configuration for plasmode sensitivity experiments."""
    enabled: bool = False
    num_repeats: int = 1
    save_datasets: bool = False
    train_fraction: float = 0.8  # Fraction of data for training (rest is eval)
    generator_architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    generator_training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluator_architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    evaluator_training: TrainingConfig = field(default_factory=TrainingConfig)
    plasmode_scenarios: List[PlasmodeConfig] = field(default_factory=list)
    propensity_trimming: PropensityTrimmingConfig = field(default_factory=PropensityTrimmingConfig)
    oracle_mode: bool = False  # If True, evaluator sees generator's exact features

    # Explicit confounder extraction configuration (LLM-based)
    explicit_confounders: ExplicitConfounderExtractionConfig = field(default_factory=ExplicitConfounderExtractionConfig)


@dataclass
class ExperimentConfig:
    """Main configuration for CDT experiments."""
    output_dir: str = "./cdt_results"
    seed: int = 42
    device: Optional[str] = None
    num_workers: int = 1
    gpu_ids: Optional[List[int]] = None

    # Filter interpretation settings
    save_filter_interpretations: bool = False  # Save post-hoc filter interpretations after training
    filter_interpretation_top_k: int = 10  # Number of top n-grams per filter to save

    # Confounder interpretation settings (for confounder extractor)
    save_confounder_interpretations: bool = False  # Save confounder attention interpretations after training
    confounder_interpretation_top_k: int = 5  # Number of top-attended sentences per confounder to save

    applied_inference: AppliedInferenceConfig = field(default_factory=AppliedInferenceConfig)
    plasmode_experiments: PlasmodeExperimentConfig = field(default_factory=PlasmodeExperimentConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""

        def parse_architecture_config(arch_data: Dict[str, Any]) -> ModelArchitectureConfig:
            """Parse architecture config, handling nested causal_forest."""
            if 'causal_forest' in arch_data and isinstance(arch_data['causal_forest'], dict):
                arch_data = arch_data.copy()
                arch_data['causal_forest'] = CausalForestConfig(**arch_data['causal_forest'])
            return ModelArchitectureConfig(**arch_data)

        def parse_explicit_confounders_config(conf_data: Dict[str, Any]) -> ExplicitConfounderExtractionConfig:
            """Parse explicit confounders config, handling nested confounder specs."""
            if not conf_data:
                return ExplicitConfounderExtractionConfig()
            conf_data = conf_data.copy()
            if 'confounders' in conf_data and isinstance(conf_data['confounders'], list):
                conf_data['confounders'] = [
                    ExplicitConfounderSpec(**c) if isinstance(c, dict) else c
                    for c in conf_data['confounders']
                ]
            return ExplicitConfounderExtractionConfig(**conf_data)

        applied = AppliedInferenceConfig(
            **{k: parse_architecture_config(v) if k == 'architecture'
               else TrainingConfig(**v) if k == 'training'
               else PropensityTrimmingConfig(**v) if k == 'propensity_trimming'
               else OutcomeModelConfig(**v) if k == 'outcome_model'
               else MatchingAnalysisConfig(**v) if k == 'matching_analysis'
               else parse_explicit_confounders_config(v) if k == 'explicit_confounders'
               else v
               for k, v in data.get('applied_inference', {}).items()}
        )

        plasmode_data = data.get('plasmode_experiments', {})
        plasmode = PlasmodeExperimentConfig(
            enabled=plasmode_data.get('enabled', False),
            num_repeats=plasmode_data.get('num_repeats', 1),
            save_datasets=plasmode_data.get('save_datasets', False),
            train_fraction=plasmode_data.get('train_fraction', 0.8),
            generator_architecture=parse_architecture_config(plasmode_data.get('generator_architecture', {})),
            generator_training=TrainingConfig(**plasmode_data.get('generator_training', {})),
            evaluator_architecture=parse_architecture_config(plasmode_data.get('evaluator_architecture', {})),
            evaluator_training=TrainingConfig(**plasmode_data.get('evaluator_training', {})),
            plasmode_scenarios=[PlasmodeConfig(**s) for s in plasmode_data.get('plasmode_scenarios', [])],
            propensity_trimming=PropensityTrimmingConfig(**plasmode_data.get('propensity_trimming', {})),
            oracle_mode=plasmode_data.get('oracle_mode', False),
            explicit_confounders=parse_explicit_confounders_config(plasmode_data.get('explicit_confounders', {}))
        )

        return cls(
            output_dir=data.get('output_dir', './cdt_results'),
            seed=data.get('seed', 42),
            device=data.get('device'),
            num_workers=data.get('num_workers', 1),
            gpu_ids=data.get('gpu_ids'),
            save_filter_interpretations=data.get('save_filter_interpretations', False),
            filter_interpretation_top_k=data.get('filter_interpretation_top_k', 10),
            save_confounder_interpretations=data.get('save_confounder_interpretations', False),
            confounder_interpretation_top_k=data.get('confounder_interpretation_top_k', 5),
            applied_inference=applied,
            plasmode_experiments=plasmode
        )

    def get_hash(self) -> str:
        """Get hash of config for caching."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def validate(self) -> None:
        """Validate configuration."""
        if not self.applied_inference.dataset_path:
            raise ValueError("applied_inference.dataset_path is required")

        if not Path(self.applied_inference.dataset_path).exists():
            raise ValueError(f"Dataset not found: {self.applied_inference.dataset_path}")

        if self.plasmode_experiments.enabled and not self.plasmode_experiments.plasmode_scenarios:
            raise ValueError("plasmode_experiments.plasmode_scenarios cannot be empty when enabled=True")

        # Validate matching config
        if self.applied_inference.matching_analysis.enabled:
            valid_methods = {'nearest', 'optimal', 'caliper'}
            if self.applied_inference.matching_analysis.method not in valid_methods:
                raise ValueError(f"matching_analysis.method must be one of {valid_methods}")


def create_default_config(output_path: str) -> None:
    """Create a default configuration file."""
    config = ExperimentConfig(
        output_dir="./cdt_results",
        seed=42,
        device="cuda:0",
        num_workers=1,
        gpu_ids=[0, 1],

        applied_inference=AppliedInferenceConfig(
            dataset_path="./dataset.parquet",
            cv_folds=5,
            architecture=ModelArchitectureConfig(
                cnn_init_embeddings_from="emilyalsentzer/Bio_ClinicalBERT",
                cnn_num_kmeans_filters=64,
                cnn_num_random_filters=0
            ),
            training=TrainingConfig(
                epochs=50,
                batch_size=8
            )
        ),

        plasmode_experiments=PlasmodeExperimentConfig(
            enabled=False,
            num_repeats=3,
            save_datasets=False,
            plasmode_scenarios=[
                PlasmodeConfig(
                    generation_mode="phi_linear",
                    target_ate_prob=0.10
                )
            ]
        )
    )

    config.to_json(output_path)
    print(f"Default configuration saved to: {output_path}")
