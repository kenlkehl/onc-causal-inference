# oci/config.py
"""Configuration classes for OCI experiments."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import hashlib


# =============================================================================
# EXPLICIT FEATURE EXTRACTION CONFIGURATION
# =============================================================================

@dataclass
class ExplicitFeatureSpec:
    """Specification for a single explicit feature to extract from clinical text.

    Roles are causal roles, not mutually exclusive classes. A variable can be a
    confounder, an effect modifier, or both.
    """
    name: str  # e.g., "performance_status"
    type: str  # "categorical" or "continuous"
    categories: Optional[List[str]] = None  # For categorical only (e.g., ["0", "1", "2", "3", "4"])
    description: Optional[str] = None  # Used in LLM prompt (e.g., "ECOG performance status")
    roles: List[str] = field(default_factory=list)  # "confounder", "effect_modifier", or both

    def __post_init__(self):
        if self.type not in ("categorical", "continuous"):
            raise ValueError(f"type must be 'categorical' or 'continuous', got '{self.type}'")
        if self.type == "categorical" and not self.categories:
            raise ValueError(f"categories required for categorical explicit feature '{self.name}'")
        valid_roles = {"confounder", "effect_modifier"}
        if not self.roles:
            raise ValueError(
                f"roles required for explicit feature '{self.name}'; "
                "use one or both of ['confounder', 'effect_modifier']"
            )
        invalid_roles = set(self.roles) - valid_roles
        if invalid_roles:
            raise ValueError(
                f"invalid roles for explicit feature '{self.name}': {sorted(invalid_roles)}. "
                f"Valid roles: {sorted(valid_roles)}"
            )
        # Preserve order while deduplicating roles.
        self.roles = list(dict.fromkeys(self.roles))


@dataclass
class ExplicitFeatureExtractionConfig:
    """Configuration for LLM-based explicit feature extraction from clinical text.

    Extracted features are role-tagged as confounders, effect modifiers, or both.
    """
    enabled: bool = False
    features: List[ExplicitFeatureSpec] = field(default_factory=list)

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
    vllm_max_model_len: Optional[int] = None  # Max context length for start_server/python_api

    # Extraction settings
    extraction_batch_size: int = 32
    extraction_max_retries: int = 3  # Retries per patient before marking as missing
    extraction_temperature: float = 0.0  # LLM temperature (0 for deterministic)
    extraction_max_tokens: int = 1024  # Max tokens for LLM response
    extraction_max_text_length: int = 8000  # Max clinical text chars in extraction prompt

    # Caching
    cache_enabled: bool = True  # Cache extraction results to disk
    cache_dir: Optional[str] = None  # Directory for cache files (default: alongside dataset)

    # Featurizer settings (for neural models only)
    featurizer_output_dim: int = 64
    featurizer_hidden_dim: int = 128
    featurizer_dropout: float = 0.1


# Backward-compatible symbol aliases for older internal imports. Config files
# using the old explicit_confounders key are rejected in ExperimentConfig.from_dict.
ExplicitConfounderSpec = ExplicitFeatureSpec
ExplicitConfounderExtractionConfig = ExplicitFeatureExtractionConfig


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
class ContrastiveEffectConfig:
    """Configuration for matched contrastive effect-modifier representation learning.

    This stage uses cross-fitted nuisance predictions from W to create
    propensity-neighborhood treatment/control contrasts, then trains the X
    representation to explain within-neighborhood outcome differences.
    """
    enabled: bool = False

    # X representation bottleneck
    bottleneck_dim: int = 8
    hidden_dim: int = 64

    # Propensity-neighborhood batching
    batch_size: int = 16
    n_propensity_bins: int = 10
    overlap_min: float = 0.05
    overlap_max: float = 0.95
    min_arm_per_bin: int = 2

    # Loss weights
    lambda_factual: float = 1.0
    lambda_contrast: float = 2.0
    lambda_adversary: float = 0.05
    lambda_z_l2: float = 1e-4

    # Residual contrast target stabilization
    target_clip: float = 1.0

    # Causal forest X feature export mode:
    # "bottleneck", "tau", or "bottleneck_plus_tau"
    forest_x_mode: str = "bottleneck_plus_tau"

    def __post_init__(self):
        valid_modes = {"bottleneck", "tau", "bottleneck_plus_tau"}
        if self.forest_x_mode not in valid_modes:
            raise ValueError(
                f"forest_x_mode must be one of {sorted(valid_modes)}, "
                f"got '{self.forest_x_mode}'"
            )
        if self.bottleneck_dim < 1:
            raise ValueError("bottleneck_dim must be >= 1")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        if self.n_propensity_bins < 1:
            raise ValueError("n_propensity_bins must be >= 1")
        if not (0.0 <= self.overlap_min < self.overlap_max <= 1.0):
            raise ValueError("overlap_min/overlap_max must satisfy 0 <= min < max <= 1")
        if self.min_arm_per_bin < 1:
            raise ValueError("min_arm_per_bin must be >= 1")


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

    # R-learner representation training for causal forest. When True, staged
    # training learns nuisance W features and effect-modifier X features.
    use_rlearner_representation: bool = False

    # Weight for R-learner loss during representation training
    gamma_rlearner: float = 1.0

    # Inner folds used for out-of-fold nuisance predictions in staged R-learning.
    rlearner_nuisance_folds: int = 5

    # Matched contrastive X-stage alternative to per-patient R-loss training.
    contrastive_effect: ContrastiveEffectConfig = field(default_factory=ContrastiveEffectConfig)

    def __post_init__(self):
        if isinstance(self.contrastive_effect, dict):
            self.contrastive_effect = ContrastiveEffectConfig(**self.contrastive_effect)


# =============================================================================
# TF-IDF + CAUSAL FOREST CONFIGURATION
# =============================================================================

@dataclass
class TfidfForestConfig:
    """Configuration for TF-IDF + Causal Forest baseline (model_type="tfidf_forest").

    A non-neural baseline that uses TF-IDF features directly with CausalForestDML.
    No GPU, no training epochs, no neural network.
    """

    # TF-IDF vectorizer parameters
    max_features: int = 10000       # Maximum number of TF-IDF features
    ngram_range_min: int = 1        # Minimum n-gram size
    ngram_range_max: int = 2        # Maximum n-gram size
    min_df: int = 5                 # Minimum document frequency (absolute count)
    max_df: float = 0.95            # Maximum document frequency (proportion)
    sublinear_tf: bool = True       # Use sublinear TF scaling (1 + log(tf))

    # Causal forest parameters
    n_estimators: int = 200         # Number of trees (must be divisible by 4 for econml)
    max_depth: Optional[int] = None # Maximum tree depth (None = unlimited)
    min_samples_leaf: int = 10      # Minimum samples per leaf
    max_features_forest: str = "sqrt"  # Feature subset strategy for splitting
    honest: bool = True             # Honest estimation (sample splitting within trees)
    inference: bool = True          # Enable confidence intervals


# =============================================================================
# EXPLICIT-FEATURE-ONLY CAUSAL FOREST CONFIGURATION
# =============================================================================

@dataclass
class ExplicitFeatureForestConfig:
    """Configuration for Explicit-Feature-Only Causal Forest.

    A non-neural pathway that uses only LLM-extracted explicit features with
    CausalForestDML. Confounder-role features are passed as W, and
    effect-modifier-role features are passed as X.
    """
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_leaf: int = 10
    max_features: str = "sqrt"
    honest: bool = True
    inference: bool = True


ConfounderForestConfig = ExplicitFeatureForestConfig


# =============================================================================
# AGENTIC EXPLICIT FEATURE SEARCH CONFIGURATION
# =============================================================================

@dataclass
class AgenticFeatureSearchConfig:
    """Configuration for adaptive explicit-feature causal forest search.

    This pathway treats the whole LLM-guided variable-selection loop as the
    object being evaluated. The outer CV folds report performance, while the
    inner folds decide whether a proposed add/remove/re-role action is accepted.
    """
    outer_folds: int = 5
    inner_folds: int = 3
    max_iterations: int = 3
    max_additions_per_iter: int = 6
    max_removals_per_iter: int = 3
    min_feature_coverage: float = 0.70

    # Acceptance thresholds for inner-CV candidate feature sets.
    min_r_loss_improvement: float = 0.01
    max_outcome_auroc_drop: float = 0.002
    max_treatment_auroc_drop: float = 0.002
    min_improvement_fold_fraction: float = 2.0 / 3.0

    # LLM proposal agent settings. The endpoint is OpenAI-compatible so it can
    # point to vLLM, OpenAI, or another compatible local server.
    agent_server_url: Optional[str] = "http://localhost:8000/v1"
    agent_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    agent_api_key: str = "EMPTY"
    agent_temperature: float = 0.0
    agent_max_tokens: int = 2048

    # Prompt/context controls. Clinical text examples are sent to the proposal
    # agent to ground variable suggestions, but are not written to artifacts by
    # default because they may contain sensitive patient text.
    clinical_text_examples_per_prompt: int = 3
    clinical_text_example_chars: int = 1600
    save_agent_context: bool = False

    random_state: int = 42
    stop_after_rejected_iteration: bool = True

    def __post_init__(self):
        if self.outer_folds < 2:
            raise ValueError("agentic_feature_search.outer_folds must be >= 2")
        if self.inner_folds < 2:
            raise ValueError("agentic_feature_search.inner_folds must be >= 2")
        if self.max_iterations < 1:
            raise ValueError("agentic_feature_search.max_iterations must be >= 1")
        if self.max_additions_per_iter < 0:
            raise ValueError("agentic_feature_search.max_additions_per_iter must be >= 0")
        if self.max_removals_per_iter < 0:
            raise ValueError("agentic_feature_search.max_removals_per_iter must be >= 0")
        if not 0.0 <= self.min_feature_coverage <= 1.0:
            raise ValueError("agentic_feature_search.min_feature_coverage must be in [0, 1]")
        if not 0.0 <= self.min_improvement_fold_fraction <= 1.0:
            raise ValueError(
                "agentic_feature_search.min_improvement_fold_fraction must be in [0, 1]"
            )
        if self.clinical_text_examples_per_prompt < 0:
            raise ValueError(
                "agentic_feature_search.clinical_text_examples_per_prompt must be >= 0"
            )
        if self.clinical_text_example_chars < 0:
            raise ValueError(
                "agentic_feature_search.clinical_text_example_chars must be >= 0"
            )


EXTRACTOR_ALIASES = {
    "frozen_llm_pooler": {"frozen_llm_pooler", "frozen_llm", "llm_pooler", "llm_pool", "flp"},
    "hierarchical_llm": {"hierarchical_llm", "hier_llm", "hlm"},
    "hierarchical_cnn": {"hierarchical_cnn", "hier_cnn", "hcnn"},
    "hierarchical_gru": {"hierarchical_gru", "hier_gru", "hgru"},
    "simple_cnn": {"simple_cnn", "scnn"},
}

VALID_EXTRACTOR_TYPES = set(EXTRACTOR_ALIASES.keys())

# Extractors that require fit_tokenizer() before training
TRAINABLE_EXTRACTOR_TYPES = {"hierarchical_cnn", "hierarchical_gru", "simple_cnn"}

# Extractors that support hidden state caching
CACHEABLE_EXTRACTOR_TYPES = {"frozen_llm_pooler", "hierarchical_llm"}


def normalize_feature_extractor_type(feature_type: str) -> str:
    """
    Normalize feature extractor type string to its canonical name.

    Args:
        feature_type: The raw feature extractor type string

    Returns:
        Normalized type string

    Raises:
        ValueError: If the feature extractor type is not recognized
    """
    if feature_type is None:
        return "frozen_llm_pooler"

    feature_type_lower = feature_type.lower().strip()

    for canonical, aliases in EXTRACTOR_ALIASES.items():
        if feature_type_lower in aliases:
            return canonical

    raise ValueError(
        f"Unsupported feature_extractor_type: '{feature_type}'. "
        f"Supported types: {sorted(VALID_EXTRACTOR_TYPES)}"
    )


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture."""
    model_type: str = "dragonnet"  # "dragonnet", "rlearner", "causal_forest", "tfidf_forest", "explicit_feature_forest", or "agentic_explicit_feature_forest"

    # Feature extractor type: "frozen_llm_pooler"
    feature_extractor_type: str = "frozen_llm_pooler"

    # Frozen LLM Pooler extractor (pretrained LLM + gated attention pooling)
    # Uses all token hidden states + GatedAttentionPooling instead of last-token embedding
    # Always loads pretrained weights; frozen by default for efficient training
    flp_model_name: str = "Qwen/Qwen3-0.6B-Base"  # HuggingFace model name
    flp_max_length: int = 8192  # Max sequence length
    flp_freeze_llm: bool = True  # Freeze LLM backbone (only train pooling + projection)
    flp_gated_attention_dim: int = 128  # Hidden dim for gated attention pooling
    flp_projection_dim: int = 128  # Final output dimension
    flp_dropout: float = 0.1  # Dropout rate for projection layers
    flp_gradient_checkpointing: bool = True  # Gradient checkpointing (when not frozen)
    flp_downprojection_dim: Optional[int] = None  # Trainable linear downprojection dim applied to LLM hidden states before pooling (None = no downprojection, pool on full hidden_size)
    flp_cache_hidden_states: bool = False  # Pre-compute and cache LLM hidden states to disk (when frozen). Default False = live LLM forward per batch.
    flp_gpu_cache: bool = False  # Keep hidden states on GPU VRAM instead of disk (auto-fallback to disk if insufficient VRAM)
    flp_random_projection_dim: Optional[int] = None  # Random linear projection dimension for cached hidden states (None = no projection, keeps original hidden_size)
    flp_chat_template_prompt: Optional[str] = None  # Chat template prompt for instruct models. When set, wraps each text in the model's chat template with this prompt preceding the clinical text. None = disabled (raw text). Recommended for instruct models: "You are an expert clinical cancer researcher. Read this patient history, and then extract a set of features that will predict the patient's next treatment and their outcome on that treatment. The history is: "

    # Hierarchical LLM extractor (frozen LLM on overlapping chunks + two-level pooling)
    hlm_model_name: str = "Qwen/Qwen3-0.6B-Base"
    hlm_chunk_size: int = 2048          # tokens per chunk
    hlm_chunk_overlap: int = 256        # overlapping tokens between chunks
    hlm_max_chunks: int = 16            # maximum chunks per document
    hlm_freeze_llm: bool = True
    hlm_gated_attention_dim: int = 128
    hlm_projection_dim: int = 128
    hlm_dropout: float = 0.1
    hlm_gradient_checkpointing: bool = True
    hlm_downprojection_dim: Optional[int] = None
    hlm_cache_hidden_states: bool = False
    hlm_gpu_cache: bool = False
    hlm_chat_template_prompt: Optional[str] = None

    # Hierarchical CNN extractor (dilated CNN on chunks + two-level pooling, trains from scratch)
    hcnn_embedding_dim: int = 256
    hcnn_conv_dim: int = 256
    hcnn_kernel_size: int = 5
    hcnn_num_conv_blocks: int = 4
    hcnn_chunk_size: int = 512
    hcnn_chunk_overlap: int = 64
    hcnn_max_chunks: int = 32
    hcnn_vocab_size: int = 50000
    hcnn_gated_attention_dim: int = 128
    hcnn_projection_dim: int = 128
    hcnn_dropout: float = 0.1

    # Hierarchical GRU extractor (BiGRU on chunks + two-level pooling, trains from scratch)
    hgru_embedding_dim: int = 256
    hgru_gru_hidden_dim: int = 256
    hgru_num_gru_layers: int = 2
    hgru_chunk_size: int = 512
    hgru_chunk_overlap: int = 64
    hgru_max_chunks: int = 32
    hgru_vocab_size: int = 50000
    hgru_gated_attention_dim: int = 128
    hgru_projection_dim: int = 128
    hgru_dropout: float = 0.1

    # Simple CNN extractor (dilated CNN on whole text, trains from scratch)
    scnn_embedding_dim: int = 256
    scnn_conv_dim: int = 256
    scnn_kernel_size: int = 5
    scnn_num_conv_blocks: int = 4
    scnn_max_length: int = 10000
    scnn_vocab_size: int = 50000
    scnn_gated_attention_dim: int = 128
    scnn_projection_dim: int = 128
    scnn_dropout: float = 0.1

    # Causal head dimensions (applies to all causal heads: DragonNet, RLearner, etc.)
    causal_head_representation_dim: int = 128
    causal_head_hidden_outcome_dim: int = 64
    causal_head_dropout: float = 0.2  # Dropout in causal head representation and outcome layers

    # Causal Forest config (used when model_type="causal_forest")
    causal_forest: CausalForestConfig = field(default_factory=CausalForestConfig)

    # TF-IDF + Causal Forest config (used when model_type="tfidf_forest")
    tfidf_forest: TfidfForestConfig = field(default_factory=TfidfForestConfig)

    # Explicit-Feature-Only Causal Forest config (used when model_type="explicit_feature_forest")
    explicit_feature_forest: ExplicitFeatureForestConfig = field(default_factory=ExplicitFeatureForestConfig)

    # Agentic explicit feature search config (used when model_type="agentic_explicit_feature_forest")
    agentic_feature_search: AgenticFeatureSearchConfig = field(default_factory=AgenticFeatureSearchConfig)


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
    # Regularization options
    weight_decay: float = 0.01  # L2 regularization (AdamW decoupled weight decay)
    gradient_clip_norm: float = 1.0  # Max gradient norm (0 to disable)
    label_smoothing: float = 0.0  # Label smoothing for BCE (0 to disable)
    # Advanced training options for improving tau learning
    stop_grad_propensity: bool = False  # Detach features before propensity loss (prevents propensity from dominating representation)
    attention_entropy_weight: float = 0.0  # Weight for attention entropy regularization (encourages focused attention)


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
class AppliedInferenceConfig:
    """Configuration for applied inference on real data."""
    outcome_type: str = "binary"  # "binary" or "continuous"
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
    # PSM analysis configuration (uses DragonNet's propensity scores)
    matching_analysis: MatchingAnalysisConfig = field(default_factory=MatchingAnalysisConfig)

    # Explicit feature extraction configuration (LLM-based)
    explicit_features: ExplicitFeatureExtractionConfig = field(default_factory=ExplicitFeatureExtractionConfig)



@dataclass
class ExperimentConfig:
    """Main configuration for OCI experiments."""
    output_dir: str = "./oci_results"
    seed: int = 42
    device: Optional[str] = None
    num_workers: int = 1
    gpu_ids: Optional[List[int]] = None

    # Confounder interpretation settings
    save_confounder_interpretations: bool = False  # Save confounder attention interpretations after training
    confounder_interpretation_top_k: int = 5  # Number of top-attended sentences per confounder to save

    applied_inference: AppliedInferenceConfig = field(default_factory=AppliedInferenceConfig)

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
        applied_data = data.get('applied_inference', {})
        if 'explicit_confounders' in applied_data:
            raise ValueError(
                "Configuration key applied_inference.explicit_confounders has been removed. "
                "Use applied_inference.explicit_features.features with role-tagged "
                "ExplicitFeatureSpec entries instead."
            )

        def parse_architecture_config(arch_data: Dict[str, Any]) -> ModelArchitectureConfig:
            """Parse architecture config, handling nested causal_forest and tfidf_forest."""
            arch_data = arch_data.copy()
            if arch_data.get('model_type') == 'confounder_forest':
                raise ValueError(
                    "model_type='confounder_forest' has been removed. "
                    "Use model_type='explicit_feature_forest' with role-tagged explicit_features."
                )
            if 'causal_forest' in arch_data and isinstance(arch_data['causal_forest'], dict):
                cf_data = arch_data['causal_forest'].copy()
                if 'contrastive_effect' in cf_data and isinstance(cf_data['contrastive_effect'], dict):
                    cf_data['contrastive_effect'] = ContrastiveEffectConfig(**cf_data['contrastive_effect'])
                arch_data['causal_forest'] = CausalForestConfig(**cf_data)
            if 'tfidf_forest' in arch_data and isinstance(arch_data['tfidf_forest'], dict):
                arch_data['tfidf_forest'] = TfidfForestConfig(**arch_data['tfidf_forest'])
            if 'confounder_forest' in arch_data:
                raise ValueError(
                    "architecture.confounder_forest has been removed. "
                    "Use architecture.explicit_feature_forest."
                )
            if 'explicit_feature_forest' in arch_data and isinstance(arch_data['explicit_feature_forest'], dict):
                arch_data['explicit_feature_forest'] = ExplicitFeatureForestConfig(**arch_data['explicit_feature_forest'])
            if (
                'agentic_feature_search' in arch_data
                and isinstance(arch_data['agentic_feature_search'], dict)
            ):
                arch_data['agentic_feature_search'] = AgenticFeatureSearchConfig(
                    **arch_data['agentic_feature_search']
                )
            return ModelArchitectureConfig(**arch_data)

        def parse_explicit_features_config(feat_data: Dict[str, Any]) -> ExplicitFeatureExtractionConfig:
            """Parse explicit features config, handling nested feature specs."""
            if not feat_data:
                return ExplicitFeatureExtractionConfig()
            feat_data = feat_data.copy()
            if 'confounders' in feat_data:
                raise ValueError(
                    "explicit_features.confounders is not supported. "
                    "Use explicit_features.features and set roles on each feature."
                )
            if 'features' in feat_data and isinstance(feat_data['features'], list):
                feat_data['features'] = [
                    ExplicitFeatureSpec(**f) if isinstance(f, dict) else f
                    for f in feat_data['features']
                ]
            return ExplicitFeatureExtractionConfig(**feat_data)

        applied = AppliedInferenceConfig(
            **{k: parse_architecture_config(v) if k == 'architecture'
               else TrainingConfig(**v) if k == 'training'
               else PropensityTrimmingConfig(**v) if k == 'propensity_trimming'
               else OutcomeModelConfig(**v) if k == 'outcome_model'
               else MatchingAnalysisConfig(**v) if k == 'matching_analysis'
               else parse_explicit_features_config(v) if k == 'explicit_features'
               else v
               for k, v in applied_data.items()}
        )

        return cls(
            output_dir=data.get('output_dir', './oci_results'),
            seed=data.get('seed', 42),
            device=data.get('device'),
            num_workers=data.get('num_workers', 1),
            gpu_ids=data.get('gpu_ids'),
            save_confounder_interpretations=data.get('save_confounder_interpretations', False),
            confounder_interpretation_top_k=data.get('confounder_interpretation_top_k', 5),
            applied_inference=applied,
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

        # Validate outcome_type
        valid_outcome_types = {"binary", "continuous"}
        if self.applied_inference.outcome_type not in valid_outcome_types:
            raise ValueError(f"applied_inference.outcome_type must be one of {valid_outcome_types}, "
                           f"got '{self.applied_inference.outcome_type}'")

        if self.applied_inference.architecture.model_type == "confounder_forest":
            raise ValueError(
                "model_type='confounder_forest' has been removed. "
                "Use model_type='explicit_feature_forest'."
            )

        if (
            self.applied_inference.explicit_features.enabled
            and not self.applied_inference.explicit_features.features
            and self.applied_inference.architecture.model_type != "agentic_explicit_feature_forest"
        ):
            raise ValueError(
                "applied_inference.explicit_features.enabled=True requires at least one "
                "role-tagged explicit feature in explicit_features.features"
            )

        # Validate matching config
        if self.applied_inference.matching_analysis.enabled:
            valid_methods = {'nearest', 'optimal', 'caliper'}
            if self.applied_inference.matching_analysis.method not in valid_methods:
                raise ValueError(f"matching_analysis.method must be one of {valid_methods}")


def create_default_config(output_path: str) -> None:
    """Create a default configuration file."""
    config = ExperimentConfig(
        output_dir="./oci_results",
        seed=42,
        device="cuda:0",
        num_workers=1,
        gpu_ids=[0, 1],

        applied_inference=AppliedInferenceConfig(
            dataset_path="./dataset.parquet",
            cv_folds=5,
            architecture=ModelArchitectureConfig(
                feature_extractor_type="frozen_llm_pooler",
            ),
            training=TrainingConfig(
                epochs=50,
                batch_size=8
            )
        ),

    )

    config.to_json(output_path)
    print(f"Default configuration saved to: {output_path}")
