# cdt/config.py
"""Configuration classes for propensity score matching experiments."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import hashlib


# =============================================================================
# NEW PROPENSITY MATCHING CONFIGURATION
# =============================================================================

@dataclass
class PropensityModelConfig:
    """Configuration for propensity score model architecture."""

    # Encoder architecture: 'cnn', 'transformer', 'gru'
    encoder_type: str = "gru"

    # Model dimensions
    hidden_dim: int = 256
    dropout: float = 0.1

    # Text embedding
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 128
    chunk_overlap: int = 32

    # Confounder feature extraction
    use_confounder_features: bool = True
    num_latent_confounders: int = 20
    features_per_confounder: int = 1
    explicit_confounder_texts: Optional[List[str]] = None
    aggregator_mode: str = "attn"
    arctanh_transform: bool = False

    # Joint outcome prediction (to encourage learning true confounders)
    joint_outcome_prediction: bool = False
    outcome_weight: float = 0.5  # Weight for outcome loss (0-1)

    # Data columns
    text_column: str = "clinical_text"
    outcome_column: str = "outcome_indicator"
    treatment_column: str = "treatment_indicator"
    split_column: str = "split"


@dataclass
class PropensityTrainingConfig:
    """Configuration for propensity model training."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_schedule: str = "linear"  # 'linear', 'cosine', or 'none'
    epochs: int = 50
    batch_size: int = 8
    early_stopping_patience: int = 10
    init_latents_from_kmeans: bool = True


@dataclass
class MatchingConfig:
    """Configuration for propensity score matching algorithm."""

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


@dataclass
class PropensityExperimentConfig:
    """Main configuration for propensity score matching experiments."""

    output_dir: str = "./psm_results"
    seed: int = 42
    device: Optional[str] = None
    num_workers: int = 1
    cache_dir: Optional[str] = None

    # Dataset
    dataset_path: str = ""

    # Cross-validation folds (1 = no CV, use fixed split)
    cv_folds: int = 5

    # Model and training configuration
    model: PropensityModelConfig = field(default_factory=PropensityModelConfig)
    training: PropensityTrainingConfig = field(default_factory=PropensityTrainingConfig)

    # Matching configuration
    matching: MatchingConfig = field(default_factory=MatchingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'PropensityExperimentConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PropensityExperimentConfig':
        """Create config from dictionary."""
        model_config = PropensityModelConfig(**data.get('model', {}))
        training_config = PropensityTrainingConfig(**data.get('training', {}))
        matching_config = MatchingConfig(**data.get('matching', {}))

        return cls(
            output_dir=data.get('output_dir', './psm_results'),
            seed=data.get('seed', 42),
            device=data.get('device'),
            num_workers=data.get('num_workers', 1),
            cache_dir=data.get('cache_dir'),
            dataset_path=data.get('dataset_path', ''),
            cv_folds=data.get('cv_folds', 5),
            model=model_config,
            training=training_config,
            matching=matching_config
        )

    def get_hash(self) -> str:
        """Get hash of config for caching."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def validate(self) -> None:
        """Validate configuration."""
        if not self.dataset_path:
            raise ValueError("dataset_path is required")

        if not Path(self.dataset_path).exists():
            raise ValueError(f"Dataset not found: {self.dataset_path}")

        valid_encoders = {'cnn', 'transformer', 'gru'}
        if self.model.encoder_type not in valid_encoders:
            raise ValueError(f"encoder_type must be one of {valid_encoders}")

        valid_methods = {'nearest', 'optimal', 'caliper'}
        if self.matching.method not in valid_methods:
            raise ValueError(f"matching.method must be one of {valid_methods}")


def create_propensity_config(output_path: str) -> None:
    """Create a default propensity matching configuration file."""
    config = PropensityExperimentConfig(
        output_dir="./psm_results",
        seed=42,
        device="cuda:0",
        num_workers=1,
        dataset_path="./dataset.parquet",
        cv_folds=5,

        model=PropensityModelConfig(
            encoder_type="gru",
            hidden_dim=256,
            dropout=0.1,
            num_latent_confounders=20,
            features_per_confounder=1,
            joint_outcome_prediction=True,
            outcome_weight=0.3
        ),

        training=PropensityTrainingConfig(
            learning_rate=1e-4,
            epochs=50,
            batch_size=8,
            early_stopping_patience=10
        ),

        matching=MatchingConfig(
            method="nearest",
            caliper=0.2,
            caliper_scale="std",
            ratio=1,
            replacement=False
        )
    )

    config.to_json(output_path)
    print(f"Default propensity matching configuration saved to: {output_path}")


# =============================================================================
# LEGACY DRAGONNET CONFIGURATION (DEPRECATED)
# =============================================================================

@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture. DEPRECATED: Use PropensityModelConfig instead."""
    model_type: str = "dragonnet"  # "dragonnet" or "uplift"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    dragonnet_representation_dim: int = 128
    dragonnet_hidden_outcome_dim: int = 64
    aggregator_mode: str = "attn"
    num_latent_confounders: int = 20
    features_per_confounder: int = 1
    explicit_confounder_texts: Optional[List[str]] = None
    chunk_size: int = 128
    chunk_overlap: int = 32
    arctanh_transform: bool = False


@dataclass
class TrainingConfig:
    """Configuration for model training. DEPRECATED: Use PropensityTrainingConfig instead."""
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    lr_schedule: str = "linear"
    epochs: int = 50
    batch_size: int = 8
    alpha_propensity: float = 1.0
    beta_targreg: float = 0.1
    init_latents_from_kmeans: bool = True


@dataclass
class PretrainingConfig:
    """Configuration for multi-treatment pretraining. DEPRECATED."""
    enabled: bool = False
    dataset_path: Optional[str] = None
    treatment_column: str = "treatment"
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class PlasmodeConfig:
    """Configuration for plasmode simulation. DEPRECATED."""
    generation_mode: str = "phi_linear"
    preserve_observed_treatments: bool = True
    baseline_control_outcome_rate: float = 0.20
    target_ate_logit: float = 0.50
    outcome_heterogeneity_scale: float = 1.0
    ite_heterogeneity_scale: float = 1.0
    deep_nonlinear_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    deep_nonlinear_dropout: float = 0.0
    uplift_hidden_dims: List[int] = field(default_factory=list)
    uplift_activation: str = "relu"
    uplift_dropout: float = 0.0


@dataclass
class AppliedInferenceConfig:
    """Configuration for applied inference on real data. DEPRECATED."""
    dataset_path: str = ""
    text_column: str = "clinical_text"
    outcome_column: str = "outcome_indicator"
    treatment_column: str = "treatment_indicator"
    split_column: str = "split"
    cv_folds: int = 5
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    use_pretrained_weights: bool = True
    skip: bool = False


@dataclass
class PlasmodeExperimentConfig:
    """Configuration for plasmode sensitivity experiments. DEPRECATED."""
    enabled: bool = False
    num_repeats: int = 1
    save_datasets: bool = False
    train_fraction: float = 0.8
    generator_architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    generator_training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluator_architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    evaluator_training: TrainingConfig = field(default_factory=TrainingConfig)
    plasmode_scenarios: List[PlasmodeConfig] = field(default_factory=list)
    oracle_mode: bool = False


@dataclass
class ExperimentConfig:
    """Main configuration for CDT experiments. DEPRECATED: Use PropensityExperimentConfig instead."""
    output_dir: str = "./cdt_results"
    seed: int = 42
    device: Optional[str] = None
    num_workers: int = 1
    gpu_ids: Optional[List[int]] = None
    cache_dir: Optional[str] = None

    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
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
        pretraining = PretrainingConfig(
            **{k: ModelArchitectureConfig(**v) if k == 'architecture'
               else TrainingConfig(**v) if k == 'training'
               else v
               for k, v in data.get('pretraining', {}).items()}
        )

        applied = AppliedInferenceConfig(
            **{k: ModelArchitectureConfig(**v) if k == 'architecture'
               else TrainingConfig(**v) if k == 'training'
               else v
               for k, v in data.get('applied_inference', {}).items()}
        )

        plasmode_data = data.get('plasmode_experiments', {})
        plasmode = PlasmodeExperimentConfig(
            enabled=plasmode_data.get('enabled', False),
            num_repeats=plasmode_data.get('num_repeats', 1),
            save_datasets=plasmode_data.get('save_datasets', False),
            train_fraction=plasmode_data.get('train_fraction', 0.8),
            generator_architecture=ModelArchitectureConfig(**plasmode_data.get('generator_architecture', {})),
            generator_training=TrainingConfig(**plasmode_data.get('generator_training', {})),
            evaluator_architecture=ModelArchitectureConfig(**plasmode_data.get('evaluator_architecture', {})),
            evaluator_training=TrainingConfig(**plasmode_data.get('evaluator_training', {})),
            plasmode_scenarios=[PlasmodeConfig(**s) for s in plasmode_data.get('plasmode_scenarios', [])],
            oracle_mode=plasmode_data.get('oracle_mode', plasmode_data.get('evaluator_use_generator_confounders', False))
        )

        return cls(
            output_dir=data.get('output_dir', './cdt_results'),
            seed=data.get('seed', 42),
            device=data.get('device'),
            num_workers=data.get('num_workers', 1),
            gpu_ids=data.get('gpu_ids'),
            cache_dir=data.get('cache_dir'),
            pretraining=pretraining,
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

        if self.pretraining.enabled:
            if not self.pretraining.dataset_path:
                raise ValueError("pretraining.dataset_path required when pretraining.enabled=True")
            if not Path(self.pretraining.dataset_path).exists():
                raise ValueError(f"Pretraining dataset not found: {self.pretraining.dataset_path}")

        if self.plasmode_experiments.enabled and not self.plasmode_experiments.plasmode_scenarios:
            raise ValueError("plasmode_experiments.plasmode_scenarios cannot be empty when enabled=True")


def create_default_config(output_path: str) -> None:
    """Create a default configuration file. DEPRECATED: Use create_propensity_config instead."""
    config = ExperimentConfig(
        output_dir="./cdt_results",
        seed=42,
        device="cuda:0",
        num_workers=1,
        gpu_ids=[0, 1],

        pretraining=PretrainingConfig(
            enabled=False,
            dataset_path="./pretrain_dataset.parquet",
            treatment_column="treatment",
            architecture=ModelArchitectureConfig(
                num_latent_confounders=50,
                features_per_confounder=1
            ),
            training=TrainingConfig(
                epochs=10,
                batch_size=8
            )
        ),

        applied_inference=AppliedInferenceConfig(
            dataset_path="./dataset.parquet",
            cv_folds=5,
            architecture=ModelArchitectureConfig(
                num_latent_confounders=20,
                features_per_confounder=4
            ),
            training=TrainingConfig(
                epochs=50,
                batch_size=8
            ),
            use_pretrained_weights=True
        ),

        plasmode_experiments=PlasmodeExperimentConfig(
            enabled=False,
            num_repeats=3,
            save_datasets=False,
            plasmode_scenarios=[
                PlasmodeConfig(
                    generation_mode="phi_linear",
                    target_ate_logit=0.5
                ),
                PlasmodeConfig(
                    generation_mode="deep_nonlinear",
                    target_ate_logit=0.5
                )
            ]
        )
    )

    config.to_json(output_path)
    print(f"Default configuration saved to: {output_path}")
