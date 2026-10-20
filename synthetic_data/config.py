# synthetic_data/config.py
"""Configuration classes for synthetic data generation."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
import json


DEFAULT_CLINICAL_QUESTION = (
    "Compare the effectiveness of letrozole+palbociclib with letrozole+ribociclib "
    "for the first line treatment of metastatic ER positive breast cancer."
)


@dataclass
class StructuredDataConfig:
    """Configuration for structured clinical data event generation.

    When enabled, the synthetic data pipeline generates additional structured
    clinical data events (encounter records, lab results, hospitalizations,
    patient-reported outcomes) alongside narrative clinical notes. These are
    converted to text using deterministic templates and interleaved
    chronologically with narrative notes in the final clinical_text.
    """
    enabled: bool = False
    include_encounters: bool = True       # ICD-10 + HCPCS/CPT encounter records
    include_hospitalizations: bool = True  # Hospital admission/discharge records
    include_labs: bool = True              # CBC, CMP, tumor markers
    include_pros: bool = True              # Patient-reported outcomes
    pro_instruments: List[str] = field(default_factory=lambda: [
        "EORTC_QLQ_C30", "PRO_CTCAE"
    ])


@dataclass
class LLMConfig:
    """Configuration for OpenAI-compatible LLM API."""
    api_base_url: str = "http://localhost:8000/v1"  # vLLM default
    api_key: str = ""  # Can be blank for local models
    model_name: str = "openai/gpt-oss-120b"  # Model name to use
    # Fallback for LLMClient calls that pass no temperature. Every generation
    # step sets its own (0.7 for variables and timelines, 0.5 for equations,
    # distributions and documents), so this does not change generated data;
    # the field remains so saved generation configs keep loading.
    temperature: float = 0.7
    max_tokens: int = 12000
    timeout: float = 300.0  # Request timeout in seconds
    api_base_urls: List[str] = field(default_factory=list)  # Optional replica pool
    extra_body: Dict[str, Any] = field(default_factory=dict)
    reject_incomplete: bool = False


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic dataset generation."""
    # Clinical research question
    clinical_question: str = DEFAULT_CLINICAL_QUESTION

    # Dataset parameters
    dataset_size: int = 500
    # Treatment main-effect coefficient added to the outcome linear predictor:
    # log-odds units for binary outcomes, outcome units for continuous outcomes.
    # It is NOT a probability-scale ATE: the realized ATE also depends on the
    # randomly drawn treatment x effect-modifier interactions (see the
    # ``true_ite`` column and ``dataset_statistics.mean_ite``). The field name
    # is kept because saved generation configs and saved-DGP checks use it.
    treatment_effect_prob: float = 0.20

    # Target rates (intercepts will be calibrated to achieve these)
    target_treatment_rate: float = 0.5  # Proportion of patients receiving treatment=1
    target_control_outcome_rate: float = 0.5  # Outcome rate in control group (treatment=0)

    # Positivity enforcement - ensures adequate treatment/control overlap for causal inference
    enforce_positivity: bool = False  # If True, ensures minimum treatment rate per confounder stratum
    min_treatment_rate_per_stratum: float = 0.1  # Minimum P(T=1|X) for each stratum (requires enforce_positivity=True)
    max_treatment_rate_per_stratum: float = 0.9  # Maximum P(T=1|X) for each stratum (requires enforce_positivity=True)

    # Internal coefficient scaling (advanced users only)
    main_coefficient_scale: float = 0.3  # Scale for main effect coefficients
    interaction_coefficient_scale: float = 0.1  # Scale for interaction coefficients
    target_logit_std: float = 2.0  # Target std of logits; lower values compress propensities toward 0.5
    # How continuous covariates are drawn from their summary mean/SD.
    # "positive_lognormal": moment-matched log-normal when the mean is positive
    # (so values such as NLR or creatinine clearance cannot be negative),
    # Normal otherwise. "normal": the original untruncated Normal draw.
    # Saved configs that lack this field predate it and load as "normal"
    # (see ``from_dict``), which is how their cohorts were generated.
    continuous_covariate_distribution: str = "positive_lognormal"
    
    # Number of role-tagged explicit features (None = use LLM default of 8-12).
    # num_confounders / num_effect_modifiers are role counts, not necessarily
    # mutually exclusive feature counts: a feature with both roles counts in both.
    num_features: Optional[int] = None
    num_confounders: Optional[int] = None
    num_effect_modifiers: Optional[int] = None
    
    # Outcome type: "binary" or "continuous"
    outcome_type: str = "binary"
    outcome_noise_std: float = 1.0  # Noise std for continuous outcomes

    # Additional causal roles and equation shape (see ``dgp_terms``). The LLM
    # still tags variables only as confounders or effect modifiers: Python
    # turns ``num_instruments`` + ``num_prognostic`` of the confounder-only
    # variables into instruments (treatment equation only) and prognostic
    # variables (outcome equation only) and draws their coefficients.
    num_instruments: int = 0
    num_prognostic: int = 0
    # Give effect-modifier-only variables an outcome main effect as well.
    modifier_main_effects: bool = False
    # Shape of continuous terms in the outcome main terms and the treatment
    # interactions: "none" (linear z), "threshold" (1[z > 0]), "quadratic" (z^2).
    nonlinearity: str = "none"
    # Separate logit SDs; None uses ``target_logit_std``.
    treatment_logit_std: Optional[float] = None
    outcome_logit_std: Optional[float] = None
    # Multiplier for the treatment x effect-modifier interaction coefficients.
    modifier_strength: float = 2.0

    # Generation-time factors (see ``generation_factors``); all off by default.
    # Exchangeable correlation of the latent normals of a Gaussian copula.
    covariate_correlation: float = 0.0
    # Probability that a variable's value is given to the note writer.
    documentation_rate: float = 1.0
    documentation_mechanism: str = "mcar"  # "mcar" or "mnar"
    documentation_mnar_slope: float = 1.0
    documented_variables_subset: Optional[List[str]] = None
    # {"rate": P, "events": [...], "regimen_mention_prob": P}; None disables.
    post_treatment_notes: Optional[Dict[str, Any]] = None
    # Noisy copies of confounders that enter no equation.
    num_proxy_variables: int = 0
    proxy_correlation: float = 0.9

    # Generation mode. Only "two_stage" (event timeline, then per-event note
    # expansion) is supported; the field is kept because saved generation
    # configs record it and saved-DGP generation compares it.
    generation_mode: str = "two_stage"

    # Two-stage generation settings
    min_events_per_patient: int = 15
    max_events_per_patient: int = 30
    note_types_to_expand: List[str] = field(default_factory=lambda: [
        "clinical_note", "imaging_report", "pathology_report", "ngs_report"
    ])
    note_separator: str = "\n\n<new_note>\n\n"
    drug_perturbation_prob: float = 0.3

    # Output (kept outside the tracked synthetic_data/example_synthetic_datasets/)
    output_dir: str = "./synthetic_output"

    # LLM settings
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Structured clinical data events
    structured_data: StructuredDataConfig = field(default_factory=StructuredDataConfig)

    # Reproducibility
    seed: int = 42

    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_public_dict(self) -> Dict[str, Any]:
        """Dictionary for files written to disk: ``llm.api_key`` is blanked."""
        values = self.to_dict()
        values["llm"]["api_key"] = ""
        return values

    def to_json(self, path: str) -> None:
        """Save config to JSON file (without the API key)."""
        with open(path, 'w') as f:
            json.dump(self.to_public_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> 'SyntheticDataConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyntheticDataConfig':
        """Create config from dictionary."""
        data = data.copy()
        # Saved configs written before continuous_covariate_distribution
        # existed drew continuous covariates from a Normal distribution.
        data.setdefault('continuous_covariate_distribution', 'normal')
        llm_data = data.pop('llm', {})
        llm_config = LLMConfig(**llm_data) if llm_data else LLMConfig()
        structured_data_data = data.pop('structured_data', {})
        structured_data_config = StructuredDataConfig(**structured_data_data) if structured_data_data else StructuredDataConfig()
        return cls(llm=llm_config, structured_data=structured_data_config, **data)
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.dataset_size < 1:
            raise ValueError("dataset_size must be at least 1")

        if not self.clinical_question.strip():
            raise ValueError("clinical_question cannot be empty")

        if not self.llm.api_base_url:
            raise ValueError("llm.api_base_url is required")

        if self.continuous_covariate_distribution not in {"positive_lognormal", "normal"}:
            raise ValueError(
                "continuous_covariate_distribution must be 'positive_lognormal' or "
                f"'normal', got {self.continuous_covariate_distribution!r}"
            )

        if self.generation_mode != "two_stage":
            raise ValueError(f"generation_mode must be 'two_stage', got '{self.generation_mode}'")

        if self.min_events_per_patient < 5:
            raise ValueError("min_events_per_patient must be at least 5")
        if self.max_events_per_patient < self.min_events_per_patient:
            raise ValueError("max_events_per_patient must be >= min_events_per_patient")
        if not (0 <= self.drug_perturbation_prob <= 1):
            raise ValueError("drug_perturbation_prob must be between 0 and 1")

        # Validate structured data config
        if self.structured_data.enabled:
            valid_instruments = {"EORTC_QLQ_C30", "PRO_CTCAE"}
            for inst in self.structured_data.pro_instruments:
                if inst not in valid_instruments:
                    raise ValueError(f"Unknown PRO instrument: '{inst}'. Valid: {valid_instruments}")

        # Validate positivity enforcement parameters
        if self.enforce_positivity:
            if not (0 < self.min_treatment_rate_per_stratum < 1):
                raise ValueError("min_treatment_rate_per_stratum must be between 0 and 1 (exclusive)")
            if not (0 < self.max_treatment_rate_per_stratum < 1):
                raise ValueError("max_treatment_rate_per_stratum must be between 0 and 1 (exclusive)")
            if self.min_treatment_rate_per_stratum >= self.max_treatment_rate_per_stratum:
                raise ValueError("min_treatment_rate_per_stratum must be less than max_treatment_rate_per_stratum")

        for field_name in ("num_features", "num_confounders", "num_effect_modifiers"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.num_features is not None and self.num_features < 1:
            raise ValueError("num_features must be at least 1")
        if self.num_confounders is not None and self.num_features is not None:
            if self.num_confounders > self.num_features:
                raise ValueError("num_confounders cannot exceed num_features")
        if self.num_effect_modifiers is not None and self.num_features is not None:
            if self.num_effect_modifiers > self.num_features:
                raise ValueError("num_effect_modifiers cannot exceed num_features")
        if (
            self.num_features is not None
            and self.num_confounders is not None
            and self.num_effect_modifiers is not None
            and self.num_features > self.num_confounders + self.num_effect_modifiers
        ):
            raise ValueError(
                "num_features cannot exceed num_confounders + num_effect_modifiers "
                "when both role counts are specified"
            )
        self.validate_dgp_factors()

    def validate_dgp_factors(self) -> None:
        """Validate the role, equation-shape, and generation-time factor fields."""
        from .dgp_terms import NONLINEARITIES
        from .generation_factors import validate_post_treatment_notes

        for field_name in ("num_instruments", "num_prognostic", "num_proxy_variables"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not isinstance(self.modifier_main_effects, bool):
            raise ValueError("modifier_main_effects must be true or false")
        if self.nonlinearity not in NONLINEARITIES:
            raise ValueError(
                f"nonlinearity must be one of {list(NONLINEARITIES)}, got {self.nonlinearity!r}"
            )
        for field_name in ("treatment_logit_std", "outcome_logit_std"):
            value = getattr(self, field_name)
            if value is not None and not value > 0:
                raise ValueError(f"{field_name} must be positive")
        if not self.target_logit_std > 0:
            raise ValueError("target_logit_std must be positive")
        if not self.modifier_strength >= 0:
            raise ValueError("modifier_strength must be non-negative")
        if not (0.0 <= self.covariate_correlation < 1.0):
            raise ValueError("covariate_correlation must be in [0, 1)")
        if not (0.0 <= self.documentation_rate <= 1.0):
            raise ValueError("documentation_rate must be between 0 and 1")
        if self.documentation_mechanism not in {"mcar", "mnar"}:
            raise ValueError("documentation_mechanism must be 'mcar' or 'mnar'")
        if self.documented_variables_subset is not None:
            subset = self.documented_variables_subset
            if (
                not isinstance(subset, list)
                or not subset
                or not all(isinstance(name, str) and name.strip() for name in subset)
                or len(set(subset)) != len(subset)
            ):
                raise ValueError(
                    "documented_variables_subset must be a non-empty list of distinct "
                    "variable names"
                )
        if not (0.0 < self.proxy_correlation < 1.0):
            raise ValueError("proxy_correlation must be between 0 and 1 (exclusive)")
        if self.post_treatment_notes is not None:
            self.post_treatment_notes = validate_post_treatment_notes(self.post_treatment_notes)

    def logit_stds(self) -> Tuple[float, float]:
        """Target SDs of the treatment and outcome linear predictors."""
        treatment = (
            self.treatment_logit_std if self.treatment_logit_std is not None
            else self.target_logit_std
        )
        outcome = (
            self.outcome_logit_std if self.outcome_logit_std is not None
            else self.target_logit_std
        )
        return float(treatment), float(outcome)

    def documentation_enabled(self) -> bool:
        """Whether some variable values are withheld from the note writer."""
        return self.documentation_rate < 1.0

    def post_treatment_enabled(self) -> bool:
        """Whether post-treatment dated notes are generated."""
        return self.post_treatment_notes is not None
