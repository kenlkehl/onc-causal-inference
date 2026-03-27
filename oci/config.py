# oci/config.py
"""Configuration classes for OCI experiments."""

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
# CONFOUNDERS-ONLY CAUSAL FOREST CONFIGURATION
# =============================================================================

@dataclass
class ConfounderForestConfig:
    """Configuration for Confounders-Only Causal Forest (model_type="confounder_forest").

    A non-neural pathway that uses only LLM-extracted confounder features with
    CausalForestDML. No text processing, no GPU, no training epochs.
    """
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_leaf: int = 10
    max_features: str = "sqrt"
    honest: bool = True
    inference: bool = True


def normalize_feature_extractor_type(feature_type: str) -> str:
    """
    Normalize feature extractor type to "frozen_llm_pooler".

    Args:
        feature_type: The raw feature extractor type string

    Returns:
        Normalized type: "frozen_llm_pooler"

    Raises:
        ValueError: If the feature extractor type is not a frozen_llm_pooler variant
    """
    if feature_type is None:
        return "frozen_llm_pooler"

    feature_type_lower = feature_type.lower()

    # Check for Frozen LLM Pooler
    if feature_type_lower in ("frozen_llm_pooler", "frozen_llm", "llm_pooler", "llm_pool"):
        return "frozen_llm_pooler"

    raise ValueError(
        f"Unsupported feature_extractor_type: '{feature_type}'. "
        f"Only 'frozen_llm_pooler' is supported."
    )


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture."""
    model_type: str = "dragonnet"  # "dragonnet", "rlearner", "causal_forest", or "tfidf_forest"

    # Feature extractor type: "frozen_llm_pooler"
    feature_extractor_type: str = "frozen_llm_pooler"

    # R-Learner dual extractor mode
    # When enabled with model_type="rlearner", uses two independent feature extractors:
    # - Nuisance extractor: shared for propensity e(X) and marginal outcome m(X)
    # - Effect extractor: dedicated to treatment effect τ(X)
    # This prevents gradient interference between confounder learning (nuisance) and
    # effect modifier learning (τ). Memory note: approximately doubles feature extraction compute.
    rlearner_dual_extractors: bool = False

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

    # Causal head dimensions (applies to all causal heads: DragonNet, RLearner, etc.)
    causal_head_representation_dim: int = 128
    causal_head_hidden_outcome_dim: int = 64
    causal_head_dropout: float = 0.2  # Dropout in causal head representation and outcome layers

    # Causal Forest config (used when model_type="causal_forest")
    causal_forest: CausalForestConfig = field(default_factory=CausalForestConfig)

    # TF-IDF + Causal Forest config (used when model_type="tfidf_forest")
    tfidf_forest: TfidfForestConfig = field(default_factory=TfidfForestConfig)

    # Confounders-Only Causal Forest config (used when model_type="confounder_forest")
    confounder_forest: ConfounderForestConfig = field(default_factory=ConfounderForestConfig)


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

    # Explicit confounder extraction configuration (LLM-based)
    explicit_confounders: ExplicitConfounderExtractionConfig = field(default_factory=ExplicitConfounderExtractionConfig)



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

        def parse_architecture_config(arch_data: Dict[str, Any]) -> ModelArchitectureConfig:
            """Parse architecture config, handling nested causal_forest and tfidf_forest."""
            arch_data = arch_data.copy()
            if 'causal_forest' in arch_data and isinstance(arch_data['causal_forest'], dict):
                arch_data['causal_forest'] = CausalForestConfig(**arch_data['causal_forest'])
            if 'tfidf_forest' in arch_data and isinstance(arch_data['tfidf_forest'], dict):
                arch_data['tfidf_forest'] = TfidfForestConfig(**arch_data['tfidf_forest'])
            if 'confounder_forest' in arch_data and isinstance(arch_data['confounder_forest'], dict):
                arch_data['confounder_forest'] = ConfounderForestConfig(**arch_data['confounder_forest'])
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
