# oci/config.py
"""Configuration classes for OCI experiments."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import json
import hashlib
import math
from .colbert_config import ColBERTConfig, colbert_config_from_mapping


def _validate_parallelism_setting(value: Any, name: str) -> None:
    if str(value).strip().lower() == "auto":
        return
    try:
        if int(value) < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be 'auto' or a positive integer") from exc


MULTI_MODEL_FEATURE_DISCOVERY_METHODS = (
    "bow",
    "htr",
    "embedding_contrast",
)

_MULTI_MODEL_FEATURE_DISCOVERY_METHOD_ALIASES = {
    "bow": {
        "bow",
        "bag_of_words",
        "bag-of-words",
        "bagofwords",
        "tfidf",
        "tf-idf",
        "bow_modeling",
        "bow-modeling",
        "bow-modelling",
        "bow_modelling",
    },
    "htr": {
        "htr",
        "htr_modeling",
        "htr-modeling",
        "htr-modelling",
        "htr_modelling",
        "htr_evidence",
        "hierarchical_transformer",
        "hierarchical-transformer",
        "attention",
    },
    "embedding_contrast": {
        "embedding",
        "embeddings",
        "embedding_contrast",
        "embedding_contrasts",
        "embedding-contrast",
        "embedding-contrasts",
        "embedding_delta",
        "embedding-delta",
        "contrast",
        "contrasts",
    },
}


def normalize_multi_model_feature_discovery_methods(
    methods: Any,
    *,
    source: str = "feature_discovery_methods",
) -> Optional[List[str]]:
    """Normalize a multi-model discovery-method selector.

    Accepted canonical methods are "bow", "htr", and "embedding_contrast".
    Strings may be comma-separated; "all" expands to all methods.
    """
    if methods is None:
        return None

    raw_values: List[Any]
    if isinstance(methods, str):
        raw_values = [methods]
    elif isinstance(methods, (list, tuple, set)):
        raw_values = list(methods)
    else:
        raw_values = [methods]

    tokens: List[str] = []
    for raw in raw_values:
        for part in str(raw).replace(";", ",").split(","):
            token = part.strip().lower()
            if token:
                tokens.append(token)

    if not tokens:
        raise ValueError(
            f"{source} must include at least one of "
            f"{list(MULTI_MODEL_FEATURE_DISCOVERY_METHODS)}"
        )

    if any(token == "all" for token in tokens):
        return list(MULTI_MODEL_FEATURE_DISCOVERY_METHODS)

    normalized: List[str] = []
    for token in tokens:
        canonical = None
        for method, aliases in _MULTI_MODEL_FEATURE_DISCOVERY_METHOD_ALIASES.items():
            if token in aliases:
                canonical = method
                break
        if canonical is None:
            raise ValueError(
                f"Unknown {source} entry {token!r}; expected one or more of "
                f"{list(MULTI_MODEL_FEATURE_DISCOVERY_METHODS)}"
            )
        if canonical not in normalized:
            normalized.append(canonical)

    if not normalized:
        raise ValueError(
            f"{source} must include at least one of "
            f"{list(MULTI_MODEL_FEATURE_DISCOVERY_METHODS)}"
        )
    return normalized


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
    value_aliases: Optional[Dict[str, List[str]]] = None  # canonical category -> accepted aliases
    temporal_rule: str = "use_only_complete_prepared_decision_time_text"
    aggregation_rule: str = "reconcile_all_pages_without_loss"

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
        if self.type != "categorical":
            self.value_aliases = None
        elif self.value_aliases:
            normalized_aliases: Dict[str, List[str]] = {}
            for category, aliases in self.value_aliases.items():
                category_text = str(category).strip()
                if not category_text:
                    continue
                alias_values = aliases if isinstance(aliases, list) else [aliases]
                normalized_aliases[category_text] = [
                    str(alias).strip() for alias in alias_values if str(alias).strip()
                ]
            self.value_aliases = {
                category: values for category, values in normalized_aliases.items() if values
            } or None
        if not isinstance(self.temporal_rule, str) or not self.temporal_rule.strip():
            raise ValueError(f"temporal_rule required for explicit feature '{self.name}'")
        if not isinstance(self.aggregation_rule, str) or not self.aggregation_rule.strip():
            raise ValueError(f"aggregation_rule required for explicit feature '{self.name}'")


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
    # Set to "auto" in server mode to discover each configured endpoint's
    # first /v1/models id; heterogeneous endpoint pools retain per-URL ids.
    vllm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_api_key: str = "EMPTY"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_download_dir: Optional[str] = None  # Model download directory
    vllm_max_model_len: Optional[int] = None  # Max context length for start_server/python_api
    vllm_reasoning_parser: Optional[str] = "auto"  # vLLM reasoning parser, or auto/none
    # None leaves the endpoint's chat-template reasoning behavior unchanged.
    vllm_enable_thinking: Optional[bool] = None

    # Extraction settings
    extraction_batch_size: int = 32
    # Maximum variables in one extraction schema. This is independent of
    # extraction_batch_size, which controls patient/request concurrency.
    max_variables_per_extraction_request: int = 10
    extraction_max_retries: int = 3  # Retries per patient before marking as missing
    extraction_retry_initial_delay: float = 1.0
    extraction_retry_max_delay: float = 30.0
    extraction_retry_backoff_factor: float = 2.0
    extraction_request_timeout: Optional[float] = 900.0
    extraction_temperature: float = 0.0  # LLM temperature (0 for deterministic)
    extraction_max_tokens: int = 25000  # Max tokens for LLM response
    # None means no implicit character truncation. Bounded context strategies
    # must receive their limit explicitly from configuration.
    extraction_max_text_length: Optional[int] = None
    # Required for complete_paged_v1. These are scientific settings with no
    # implicit defaults; they bound one request but never truncate a
    # note because page cores must cover the full prepared text exactly once.
    complete_page_core_chars: Optional[int] = None
    complete_page_context_chars: Optional[int] = None
    complete_page_max_chars: Optional[int] = None
    complete_reconciliation_fan_in: Optional[int] = None
    # Request packing and retrieval from cached patient token vectors.
    extraction_grouping_strategy: str = "clinical_domain"
    extraction_context_strategy: str = "colbert"
    colbert: ColBERTConfig = field(default_factory=ColBERTConfig)
    extraction_provider: str = "openai"
    # Opt-in for datasets whose text has already been made temporally valid by
    # construction. The legacy default preserves treatment-time extraction
    # eligibility instructions for every existing caller.
    source_text_temporally_valid_by_design: bool = False
    codex_cli_executable: str = "codex"
    codex_cli_model_name: Optional[str] = "gpt-5.4-mini"
    codex_cli_reasoning_effort: Optional[str] = "medium"
    codex_cli_extra_args: List[str] = field(default_factory=list)
    codex_cli_parallelism: int = 4

    # Caching
    cache_enabled: bool = True  # Cache extraction results to disk
    cache_dir: Optional[str] = None  # Directory for cache files (default: alongside dataset)

    # Featurizer settings (for neural models only)
    featurizer_output_dim: int = 64
    featurizer_hidden_dim: int = 128
    featurizer_dropout: float = 0.1

    def __post_init__(self):
        self.colbert = colbert_config_from_mapping(self.colbert)
        if not isinstance(self.source_text_temporally_valid_by_design, bool):
            raise ValueError(
                "explicit_features.source_text_temporally_valid_by_design must be boolean"
            )
        if not 1 <= int(self.max_variables_per_extraction_request) <= 10:
            raise ValueError(
                "explicit_features.max_variables_per_extraction_request must be in [1, 10]"
            )
        self.max_variables_per_extraction_request = int(self.max_variables_per_extraction_request)
        grouping = str(self.extraction_grouping_strategy).strip().lower().replace("-", "_")
        if grouping not in {"clinical_domain", "packed"}:
            raise ValueError(
                "explicit_features.extraction_grouping_strategy must be "
                "'clinical_domain' or 'packed'"
            )
        self.extraction_grouping_strategy = grouping
        context = str(self.extraction_context_strategy).strip().lower().replace("-", "_")
        if context not in {"colbert", "tail", "contract_lexical_rag", "complete_paged_v1"}:
            raise ValueError(
                "explicit_features.extraction_context_strategy must be "
                "'colbert', 'tail', 'contract_lexical_rag', or 'complete_paged_v1'"
            )
        self.extraction_context_strategy = context
        if self.extraction_max_text_length is not None:
            self.extraction_max_text_length = int(self.extraction_max_text_length)
            if self.extraction_max_text_length < 1:
                raise ValueError("explicit_features.extraction_max_text_length must be positive")
        if context == "contract_lexical_rag" and (
            self.extraction_max_text_length is None or self.extraction_max_text_length < 256
        ):
            raise ValueError(
                "explicit_features.extraction_max_text_length must be at least 256 "
                "for contract_lexical_rag"
            )
        complete_geometry = (
            self.complete_page_core_chars,
            self.complete_page_context_chars,
            self.complete_page_max_chars,
            self.complete_reconciliation_fan_in,
        )
        if context == "complete_paged_v1":
            if any(value is None for value in complete_geometry):
                raise ValueError(
                    "complete_paged_v1 requires configured complete-page core, "
                    "context, and maximum character counts"
                )
            from .extraction.complete_paged import CompletePagingGeometry

            CompletePagingGeometry(
                core_chars=int(self.complete_page_core_chars),
                context_chars=int(self.complete_page_context_chars),
                max_page_chars=int(self.complete_page_max_chars),
            )
            self.complete_page_core_chars = int(self.complete_page_core_chars)
            self.complete_page_context_chars = int(self.complete_page_context_chars)
            self.complete_page_max_chars = int(self.complete_page_max_chars)
            self.complete_reconciliation_fan_in = int(
                self.complete_reconciliation_fan_in
            )
            if self.complete_reconciliation_fan_in < 2:
                raise ValueError(
                    "complete_reconciliation_fan_in must be at least two"
                )
            if (
                self.extraction_max_text_length is not None
                and self.extraction_max_text_length != self.complete_page_max_chars
            ):
                raise ValueError(
                    "complete_paged_v1 extraction_max_text_length must equal the "
                    "configured complete_page_max_chars request bound"
                )
        elif any(value is not None for value in complete_geometry):
            raise ValueError(
                "complete-page geometry is only valid with complete_paged_v1"
            )


def parse_explicit_feature_spec_entries(
    entries: Any,
    *,
    default_roles: Optional[List[str]] = None,
    source: str = "explicit feature specs",
) -> List[ExplicitFeatureSpec]:
    """Parse feature-spec entries, optionally applying roles from their container."""
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"{source} must be a list of explicit feature spec objects")

    specs: List[ExplicitFeatureSpec] = []
    for idx, entry in enumerate(entries):
        specs.append(
            _parse_explicit_feature_spec_entry(
                entry,
                default_roles=default_roles,
                source=f"{source}[{idx}]",
            )
        )
    return specs


def load_explicit_feature_specs_json(path: str) -> List[ExplicitFeatureSpec]:
    """Load role-tagged feature specs from JSON.

    Accepted shapes:
      - a list of full feature specs with roles
      - {"features": [...]} with full role-tagged specs
      - {"confounders": [...], "effect_modifiers": [...]} where section roles
        are applied automatically and overlapping names can later be merged
    """
    with open(path, "r") as f:
        data = json.load(f)
    return parse_explicit_feature_specs_payload(
        data,
        source=f"explicit feature specs JSON {path}",
    )


def parse_explicit_feature_specs_payload(
    data: Any,
    *,
    source: str = "explicit feature specs payload",
) -> List[ExplicitFeatureSpec]:
    """Parse a feature-spec JSON payload."""
    if isinstance(data, list):
        return parse_explicit_feature_spec_entries(data, source=source)
    if not isinstance(data, dict):
        raise ValueError(f"{source} must be a list or object")

    specs: List[ExplicitFeatureSpec] = []
    if "features" in data:
        specs.extend(
            parse_explicit_feature_spec_entries(
                data.get("features"),
                source=f"{source}.features",
            )
        )
    if "confounders" in data:
        specs.extend(
            parse_explicit_feature_spec_entries(
                data.get("confounders"),
                default_roles=["confounder"],
                source=f"{source}.confounders",
            )
        )
    if "effect_modifiers" in data:
        specs.extend(
            parse_explicit_feature_spec_entries(
                data.get("effect_modifiers"),
                default_roles=["effect_modifier"],
                source=f"{source}.effect_modifiers",
            )
        )
    if not specs:
        raise ValueError(f"{source} must contain 'features', 'confounders', or 'effect_modifiers'")
    return specs


def _parse_explicit_feature_spec_entry(
    entry: Any,
    *,
    default_roles: Optional[List[str]],
    source: str,
) -> ExplicitFeatureSpec:
    if isinstance(entry, ExplicitFeatureSpec):
        if not default_roles:
            return entry
        return ExplicitFeatureSpec(
            name=entry.name,
            type=entry.type,
            categories=entry.categories,
            description=entry.description,
            roles=_merge_spec_roles(default_roles, entry.roles),
            value_aliases=entry.value_aliases,
            temporal_rule=entry.temporal_rule,
            aggregation_rule=entry.aggregation_rule,
        )

    if isinstance(entry, str):
        try:
            entry = json.loads(entry)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{source} must be a JSON object with at least name, type, and roles; "
                "plain variable names are not enough to define extraction."
            ) from exc

    if not isinstance(entry, dict):
        raise ValueError(f"{source} must be an explicit feature spec object")

    spec_data = entry.copy()
    if default_roles:
        spec_data["roles"] = _merge_spec_roles(default_roles, spec_data.get("roles"))
    return ExplicitFeatureSpec(**spec_data)


def _merge_spec_roles(left: Any, right: Any) -> List[str]:
    roles: List[str] = []
    for value in _as_list(left) + _as_list(right):
        role = str(value).strip()
        if role and role not in roles:
            roles.append(role)
    return roles


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


# Backward-compatible symbol aliases for older internal imports. Config files
# using the old explicit_confounders key are rejected in ExperimentConfig.from_dict.
ExplicitConfounderSpec = ExplicitFeatureSpec
ExplicitConfounderExtractionConfig = ExplicitFeatureExtractionConfig


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
    # Outcome-regression interaction head used by the integrated evidence
    # pathway.  The regularization value is selected by fit-sample inner CV on
    # observed outcome loss; no ITE labels are available to this selection.
    interaction_regularization_grid: List[float] = field(
        default_factory=lambda: [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    )
    interaction_inner_folds: int = 3
    interaction_interact_all_features: bool = True
    interaction_max_iter: int = 3000


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
    min_feature_coverage: float = 0.0
    search_mode: str = "broad_screen"
    broad_candidate_count: int = 80
    broad_screen_top_k: int = 20

    # Acceptance thresholds for inner-CV candidate feature sets.
    min_r_loss_improvement: float = 0.01
    max_outcome_auroc_drop: float = 0.002
    max_treatment_auroc_drop: float = 0.002
    min_improvement_fold_fraction: float = 2.0 / 3.0

    # Train-fold role diagnostics for proposed variables. These regressions are
    # advisory: they are recorded for the proposal agent and artifacts, while
    # final acceptance still depends on nested-CV candidate performance.
    role_diagnostics_enabled: bool = True
    role_diagnostic_min_n: int = 20
    role_diagnostic_min_non_missing: int = 10
    role_diagnostic_score_delta_threshold: float = 0.001

    # LLM proposal agent settings. The endpoint is OpenAI-compatible so it can
    # point to vLLM, OpenAI, or another compatible local server.
    agent_server_url: Optional[str] = "http://localhost:8000/v1"
    # Set to "auto" to discover each configured endpoint's first /v1/models id.
    agent_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    agent_api_key: str = "EMPTY"
    agent_temperature: float = 0.0
    agent_max_tokens: int = 25000
    # None leaves the endpoint's chat-template reasoning behavior unchanged.
    agent_enable_thinking: Optional[bool] = None
    # Optional server-side reasoning budget.  This is sent only when thinking
    # is explicitly enabled, so a configured budget cannot silently turn
    # reasoning on for endpoints or workflows that leave it disabled.
    agent_thinking_token_budget: Optional[int] = None
    agent_schema_repair_attempts: int = 1
    agent_request_max_retries: int = 3
    agent_retry_initial_delay: float = 1.0
    agent_retry_max_delay: float = 30.0
    agent_retry_backoff_factor: float = 2.0
    agent_request_timeout: Optional[float] = 900.0
    agent_provider: str = "openai"
    codex_cli_executable: str = "codex"
    codex_cli_model_name: Optional[str] = "gpt-5.4-mini"
    codex_cli_reasoning_effort: Optional[str] = "medium"
    codex_cli_extra_args: List[str] = field(default_factory=list)

    # Prompt/context controls. Clinical text examples are sent to the proposal
    # agent to ground variable suggestions, but are not written to artifacts by
    # default because they may contain sensitive patient text.
    # A finite example count is an explicit scientific sampling choice. The
    # legacy character-count field is accepted for configuration compatibility
    # but is nonbinding: every selected note is passed in full.
    clinical_text_examples_per_prompt: Optional[int] = None
    clinical_text_example_chars: Optional[int] = None
    save_agent_context: bool = False
    # Raw agent output may include quoted prompt snippets or reasoning text.
    save_agent_raw_output: bool = False

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
        if self.agent_request_timeout is not None and self.agent_request_timeout <= 0:
            raise ValueError("agentic_feature_search.agent_request_timeout must be > 0 or None")
        if self.agent_enable_thinking is not None and not isinstance(
            self.agent_enable_thinking, bool
        ):
            raise ValueError(
                "agentic_feature_search.agent_enable_thinking must be a boolean or None"
            )
        if (
            isinstance(self.agent_max_tokens, bool)
            or not isinstance(self.agent_max_tokens, int)
            or self.agent_max_tokens <= 0
        ):
            raise ValueError("agentic_feature_search.agent_max_tokens must be a positive integer")
        if self.agent_thinking_token_budget is not None:
            if (
                isinstance(self.agent_thinking_token_budget, bool)
                or not isinstance(self.agent_thinking_token_budget, int)
                or self.agent_thinking_token_budget <= 0
            ):
                raise ValueError(
                    "agentic_feature_search.agent_thinking_token_budget must be a "
                    "positive integer or None"
                )
            if self.agent_thinking_token_budget >= self.agent_max_tokens:
                raise ValueError(
                    "agentic_feature_search.agent_thinking_token_budget must be "
                    "strictly less than agent_max_tokens"
                )
        if not 0.0 <= self.min_feature_coverage <= 1.0:
            raise ValueError("agentic_feature_search.min_feature_coverage must be in [0, 1]")
        if self.search_mode not in {"iterative", "broad_screen"}:
            raise ValueError(
                "agentic_feature_search.search_mode must be one of " "['iterative', 'broad_screen']"
            )
        if self.broad_candidate_count < 1:
            raise ValueError("agentic_feature_search.broad_candidate_count must be >= 1")
        if self.broad_screen_top_k < 1:
            raise ValueError("agentic_feature_search.broad_screen_top_k must be >= 1")
        if not 0.0 <= self.min_improvement_fold_fraction <= 1.0:
            raise ValueError(
                "agentic_feature_search.min_improvement_fold_fraction must be in [0, 1]"
            )
        if self.role_diagnostic_min_n < 2:
            raise ValueError("agentic_feature_search.role_diagnostic_min_n must be >= 2")
        if self.role_diagnostic_min_non_missing < 1:
            raise ValueError("agentic_feature_search.role_diagnostic_min_non_missing must be >= 1")
        if self.role_diagnostic_score_delta_threshold < 0.0:
            raise ValueError(
                "agentic_feature_search.role_diagnostic_score_delta_threshold must be >= 0"
            )
        if (
            self.clinical_text_examples_per_prompt is not None
            and self.clinical_text_examples_per_prompt < 0
        ):
            raise ValueError(
                "agentic_feature_search.clinical_text_examples_per_prompt must be "
                ">= 0 or None"
            )
        if (
            self.clinical_text_example_chars is not None
            and self.clinical_text_example_chars < 1
        ):
            raise ValueError(
                "agentic_feature_search.clinical_text_example_chars must be "
                "positive or None"
            )
        if self.agent_schema_repair_attempts < 0:
            raise ValueError("agentic_feature_search.agent_schema_repair_attempts must be >= 0")
        if self.agent_request_max_retries < 0:
            raise ValueError("agentic_feature_search.agent_request_max_retries must be >= 0")
        if self.agent_retry_initial_delay < 0:
            raise ValueError("agentic_feature_search.agent_retry_initial_delay must be >= 0")
        if self.agent_retry_max_delay < 0:
            raise ValueError("agentic_feature_search.agent_retry_max_delay must be >= 0")
        if self.agent_retry_backoff_factor < 1:
            raise ValueError("agentic_feature_search.agent_retry_backoff_factor must be >= 1")


@dataclass(frozen=True)
class ClusterLocalEmbeddingScientificConfig:
    """Closed scientific controls for native cluster-local embedding evidence.

    This type intentionally has no defaults so an omitted sklearn, pooling,
    SVD, rank, or replay setting cannot silently inherit a library default. The
    KMeans ``random_state`` is derived from the context's ordered rows and is
    therefore represented by a seed policy rather than a free-standing integer.
    """

    requested_cluster_count: int
    cluster_count_policy: str
    maximum_components_per_family: int
    loading_evidence_capacity: Optional[int]
    loading_evidence_overflow_policy: str
    minimum_cluster_size: int
    minimum_group_size: int
    minimum_cell_size: int
    minimum_distinct_local_clusters_per_family: int
    minimum_numerical_rank_per_family: int
    patient_pooling_policy: str
    computation_dtype: str
    normalize_patient_embeddings: bool
    normalization_epsilon: float
    zero_vector_policy: str
    local_direction_weighting_policy: str
    kmeans_init: str
    kmeans_max_iter: int
    kmeans_batch_size_policy: str
    kmeans_batch_size_lower_bound: int
    kmeans_batch_size_upper_bound: int
    kmeans_verbose: int
    kmeans_compute_labels: bool
    kmeans_seed_derivation_policy: str
    kmeans_tol: float
    kmeans_max_no_improvement: Optional[int]
    kmeans_init_size: Optional[int]
    kmeans_n_init: int | str
    kmeans_reassignment_ratio: float
    svd_full_matrices: bool
    svd_compute_uv: bool
    svd_hermitian: bool
    svd_sign_canonicalization_policy: str
    svd_rank_tolerance_policy: str
    svd_rank_tolerance_dtype: str
    svd_rank_tolerance_multiplier: float
    replay_comparison_policy: str
    replay_relative_tolerance: float
    replay_absolute_tolerance: float
    exception_policy: str

    def __post_init__(self) -> None:
        integer_minimums = {
            "requested_cluster_count": 2,
            "maximum_components_per_family": 1,
            "minimum_cluster_size": 2,
            "minimum_group_size": 2,
            "minimum_cell_size": 1,
            "minimum_distinct_local_clusters_per_family": 2,
            "minimum_numerical_rank_per_family": 2,
            "kmeans_max_iter": 1,
            "kmeans_batch_size_lower_bound": 1,
            "kmeans_batch_size_upper_bound": 1,
            "kmeans_verbose": 0,
        }
        for name, minimum in integer_minimums.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or int(value) < minimum
            ):
                raise ValueError(
                    f"cluster_local_scientific.{name} must be an integer >= {minimum}"
                )
        if (
            int(self.kmeans_batch_size_upper_bound)
            < int(self.kmeans_batch_size_lower_bound)
        ):
            raise ValueError(
                "cluster_local_scientific KMeans batch-size bounds are reversed"
            )
        if self.loading_evidence_capacity is not None and (
            isinstance(self.loading_evidence_capacity, bool)
            or not isinstance(self.loading_evidence_capacity, int)
            or self.loading_evidence_capacity < 1
        ):
            raise ValueError(
                "cluster_local_scientific.loading_evidence_capacity must be "
                "null or a positive integer"
            )
        if self.kmeans_max_no_improvement is not None and (
            isinstance(self.kmeans_max_no_improvement, bool)
            or not isinstance(self.kmeans_max_no_improvement, int)
            or self.kmeans_max_no_improvement < 1
        ):
            raise ValueError(
                "cluster_local_scientific.kmeans_max_no_improvement must be "
                "null or a positive integer"
            )
        if self.kmeans_init_size is not None and (
            isinstance(self.kmeans_init_size, bool)
            or not isinstance(self.kmeans_init_size, int)
            or self.kmeans_init_size < 1
        ):
            raise ValueError(
                "cluster_local_scientific.kmeans_init_size must be null or positive"
            )
        if isinstance(self.kmeans_n_init, bool) or not (
            (
                isinstance(self.kmeans_n_init, int)
                and int(self.kmeans_n_init) >= 1
            )
            or self.kmeans_n_init == "auto"
        ):
            raise ValueError(
                "cluster_local_scientific.kmeans_n_init must be a positive "
                "integer or 'auto'"
            )
        for name in (
            "normalize_patient_embeddings",
            "kmeans_compute_labels",
            "svd_full_matrices",
            "svd_compute_uv",
            "svd_hermitian",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"cluster_local_scientific.{name} must be Boolean")
        if self.svd_compute_uv is not True:
            raise ValueError(
                "cluster-local component evidence requires svd_compute_uv=true"
            )
        finite_nonnegative = (
            "kmeans_tol",
            "kmeans_reassignment_ratio",
            "replay_relative_tolerance",
            "replay_absolute_tolerance",
        )
        for name in finite_nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"cluster_local_scientific.{name} must be finite and nonnegative"
                )
        for name in ("normalization_epsilon", "svd_rank_tolerance_multiplier"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"cluster_local_scientific.{name} must be finite and positive"
                )
        if self.cluster_count_policy != "require_exact_configured_count_v1":
            raise ValueError("cluster-local cluster_count_policy is unsupported")
        if (
            self.loading_evidence_overflow_policy
            != "fail_closed_no_truncation_v1"
        ):
            raise ValueError(
                "cluster-local loading evidence must fail closed instead of truncating"
            )
        if (
            self.patient_pooling_policy
            != "arithmetic_mean_all_chunks_v1"
        ):
            raise ValueError("cluster-local patient pooling policy is unsupported")
        if self.computation_dtype not in {"float32", "float64"}:
            raise ValueError(
                "cluster_local_scientific.computation_dtype must be float32 or float64"
            )
        if self.zero_vector_policy != "reject":
            raise ValueError("cluster-local zero-vector policy must be reject")
        if (
            self.local_direction_weighting_policy
            != "sqrt_cluster_size_times_unit_direction_v1"
        ):
            raise ValueError(
                "cluster-local direction weighting policy is unsupported"
            )
        if self.kmeans_init not in {"k-means++", "random"}:
            raise ValueError("cluster-local KMeans init must be k-means++ or random")
        if (
            self.kmeans_batch_size_policy
            != "clamp_usable_rows_to_configured_bounds_v1"
        ):
            raise ValueError("cluster-local KMeans batch-size policy is unsupported")
        if (
            self.kmeans_seed_derivation_policy
            != "canonical_ordered_fit_rows_group_seed_v1"
        ):
            raise ValueError("cluster-local KMeans seed policy is unsupported")
        if (
            self.svd_sign_canonicalization_policy
            != "largest_absolute_coordinate_positive_v1"
        ):
            raise ValueError("cluster-local SVD sign policy is unsupported")
        if (
            self.svd_rank_tolerance_policy
            != "dtype_epsilon_times_max_shape_times_largest_singular_v1"
        ):
            raise ValueError("cluster-local SVD rank policy is unsupported")
        if self.svd_rank_tolerance_dtype not in {"float32", "float64"}:
            raise ValueError("cluster-local SVD rank dtype is unsupported")
        if self.replay_comparison_policy != "allclose_and_exact_discrete_state_v1":
            raise ValueError("cluster-local replay comparison policy is unsupported")
        if self.exception_policy != "abort_scope_no_skip_or_fallback_v1":
            raise ValueError("cluster-local exceptions must abort the scope")
        if (
            self.minimum_numerical_rank_per_family
            > self.minimum_distinct_local_clusters_per_family
            or self.minimum_numerical_rank_per_family
            > self.maximum_components_per_family
        ):
            raise ValueError(
                "cluster-local rank requirement exceeds configured support/components"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ClusterLocalEmbeddingScientificConfig":
        if not isinstance(value, Mapping):
            raise TypeError("cluster_local_scientific must be one object")
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        if missing or extra:
            raise ValueError(
                "cluster_local_scientific must be explicitly and exactly "
                f"configured; missing={missing}, extra={extra}"
            )
        return cls(**dict(value))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmbeddingContrastDiscoveryConfig:
    """Configuration for embedding-contrast evidence in multi-model agentic search."""

    enabled: bool = True
    disable_reason: Optional[str] = None
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    batch_size: int = 16
    max_seq_length: Optional[int] = 1024
    chunk_size_words: int = 256
    chunk_overlap_words: int = 64
    # Complete-note encoder capacity. This must not select a prefix/tail; 512
    # is nonbinding for the bundled long-note cohorts.
    max_chunks: int = 512
    chunk_selection: str = "last"
    normalize_embeddings: bool = True
    top_k_chunks_per_tail: int = 12
    max_chunks_per_patient: int = 2
    min_probe_auc: float = 0.0
    pseudo_target_quantile: float = 0.20
    pseudo_target_weighted: bool = True
    include_cell_contrasts: bool = True
    include_confounder_vector_contrast: bool = True
    include_residualized_interaction_contrast: bool = True
    include_orthogonal_r_score_contrasts: bool = True
    include_cluster_contrast_vectors: bool = True
    cluster_contrast_n_clusters: int = 10
    cluster_contrast_max_components: int = 5
    cluster_contrast_min_cluster_size: int = 24
    cluster_contrast_min_group_size: int = 8
    cluster_contrast_min_cell_size: int = 4
    cluster_contrast_top_loadings: int = 5
    cluster_contrast_random_state: int = 42
    cluster_contrast_kmeans_n_init: int = 20
    cluster_local_scientific: Optional[
        ClusterLocalEmbeddingScientificConfig
    ] = None
    external_corpus_cache_dirs: List[str] = field(default_factory=list)
    external_top_k_chunks_per_tail: int = 12
    # Optional lexical summaries of embedding contrasts. Contextual embedding
    # witnesses remain enabled when this additional evidence lane is disabled.
    retrieval_tfidf_enabled: bool = False
    retrieval_tfidf_ngram_range_min: int = 1
    retrieval_tfidf_ngram_range_max: int = 3
    retrieval_tfidf_max_features: int = 20_000
    retrieval_tfidf_top_terms_per_side: int = 30
    residualize_columns: List[str] = field(default_factory=list)
    concept_phrases: List[str] = field(default_factory=list)
    include_bow_phrases_as_concepts: bool = True
    max_concept_phrases: int = 80
    concept_probe_top_k: int = 20

    def __post_init__(self):
        if not bool(self.enabled) and not str(self.disable_reason or "").strip():
            raise ValueError(
                "embedding_contrast.enabled=False requires disable_reason because "
                "embedding contrast is a required multi-model evidence source"
            )
        if self.batch_size < 1:
            raise ValueError("embedding_contrast.batch_size must be >= 1")
        if self.max_seq_length is not None:
            self.max_seq_length = int(self.max_seq_length)
            if self.max_seq_length < 1:
                raise ValueError("embedding_contrast.max_seq_length must be >= 1")
        if self.chunk_size_words < 1:
            raise ValueError("embedding_contrast.chunk_size_words must be >= 1")
        if self.chunk_overlap_words < 0:
            raise ValueError("embedding_contrast.chunk_overlap_words must be >= 0")
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError(
                "embedding_contrast.chunk_overlap_words must be smaller than " "chunk_size_words"
            )
        if self.max_chunks < 1:
            raise ValueError("embedding_contrast.max_chunks must be >= 1")
        chunk_selection = str(self.chunk_selection).strip().lower()
        if chunk_selection not in {"first", "last"}:
            raise ValueError("embedding_contrast.chunk_selection must be 'first' or 'last'")
        self.chunk_selection = chunk_selection
        if self.top_k_chunks_per_tail < 1:
            raise ValueError("embedding_contrast.top_k_chunks_per_tail must be >= 1")
        if self.max_chunks_per_patient < 1:
            raise ValueError("embedding_contrast.max_chunks_per_patient must be >= 1")
        if self.external_top_k_chunks_per_tail < 1:
            raise ValueError("embedding_contrast.external_top_k_chunks_per_tail must be >= 1")
        if self.retrieval_tfidf_ngram_range_min < 1:
            raise ValueError("embedding_contrast.retrieval_tfidf_ngram_range_min must be >= 1")
        if self.retrieval_tfidf_ngram_range_max < self.retrieval_tfidf_ngram_range_min:
            raise ValueError(
                "embedding_contrast.retrieval_tfidf_ngram_range_max must be at least "
                "retrieval_tfidf_ngram_range_min"
            )
        if self.retrieval_tfidf_max_features < 1:
            raise ValueError("embedding_contrast.retrieval_tfidf_max_features must be >= 1")
        if self.retrieval_tfidf_top_terms_per_side < 1:
            raise ValueError(
                "embedding_contrast.retrieval_tfidf_top_terms_per_side must be >= 1"
            )
        if not 0.0 <= self.min_probe_auc <= 1.0:
            raise ValueError("embedding_contrast.min_probe_auc must be in [0, 1]")
        if not 0.0 < self.pseudo_target_quantile < 0.5:
            raise ValueError("embedding_contrast.pseudo_target_quantile must be in (0, 0.5)")
        if self.max_concept_phrases < 0:
            raise ValueError("embedding_contrast.max_concept_phrases must be >= 0")
        if self.concept_probe_top_k < 1:
            raise ValueError("embedding_contrast.concept_probe_top_k must be >= 1")
        if self.cluster_contrast_n_clusters < 2:
            raise ValueError("embedding_contrast.cluster_contrast_n_clusters must be >= 2")
        if self.cluster_contrast_max_components < 1:
            raise ValueError("embedding_contrast.cluster_contrast_max_components must be >= 1")
        if self.cluster_contrast_min_cluster_size < 2:
            raise ValueError("embedding_contrast.cluster_contrast_min_cluster_size must be >= 2")
        if self.cluster_contrast_min_group_size < 2:
            raise ValueError("embedding_contrast.cluster_contrast_min_group_size must be >= 2")
        if self.cluster_contrast_min_cell_size < 1:
            raise ValueError("embedding_contrast.cluster_contrast_min_cell_size must be >= 1")
        if self.cluster_contrast_top_loadings < 1:
            raise ValueError("embedding_contrast.cluster_contrast_top_loadings must be >= 1")
        if self.cluster_contrast_kmeans_n_init < 1:
            raise ValueError("embedding_contrast.cluster_contrast_kmeans_n_init must be >= 1")
        if isinstance(self.cluster_local_scientific, dict):
            self.cluster_local_scientific = (
                ClusterLocalEmbeddingScientificConfig.from_mapping(
                    self.cluster_local_scientific
                )
            )
        if self.cluster_local_scientific is not None and (
            type(self.cluster_local_scientific)
            is not ClusterLocalEmbeddingScientificConfig
        ):
            raise TypeError(
                "embedding_contrast.cluster_local_scientific must be the "
                "closed ClusterLocalEmbeddingScientificConfig"
            )
        self.external_corpus_cache_dirs = [
            str(path).strip() for path in self.external_corpus_cache_dirs if str(path).strip()
        ]
        self.residualize_columns = [str(col) for col in self.residualize_columns]
        self.concept_phrases = [
            str(phrase).strip() for phrase in self.concept_phrases if str(phrase).strip()
        ]


@dataclass
class TfidfVectorizerScientificConfig:
    """Closed, portable scientific parameters for one TF-IDF vectorizer.

    Operational I/O is deliberately absent.  Text is always supplied as
    complete in-memory content; case handling, tokenization, vocabulary
    learning, weighting, and numerical precision are scientific behavior.
    """

    input_text_case_policy: str = "vectorizer_controls_complete_text_case_v1"
    input: str = "content"
    encoding: str = "utf-8"
    decode_error: str = "strict"
    strip_accents: Optional[str] = None
    lowercase: bool = True
    preprocessor_policy: str = "none"
    tokenizer_policy: str = "token_pattern"
    analyzer: str = "word"
    stop_words: Optional[Any] = None
    token_pattern: Optional[str] = r"(?u)[a-z0-9%<>+=-]+"
    ngram_range_min: int = 1
    ngram_range_max: int = 3
    max_df: float = 0.95
    min_df: int = 5
    max_features: Optional[int] = 30000
    vocabulary_policy: str = "fit_scope_only"
    binary: bool = False
    dtype: str = "float32"
    norm: Optional[str] = "l2"
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = True
    feature_selection_rule: str = "sklearn_term_frequency_rank_v1"

    def __post_init__(self):
        if self.input_text_case_policy != "vectorizer_controls_complete_text_case_v1":
            raise ValueError(
                "TF-IDF input_text_case_policy must consume complete text and "
                "delegate case handling to the configured vectorizer"
            )
        if self.input != "content":
            raise ValueError("TF-IDF vectorizer input must be 'content'")
        if not isinstance(self.encoding, str) or not self.encoding:
            raise ValueError("TF-IDF vectorizer encoding must be nonempty")
        if self.decode_error not in {"strict", "ignore", "replace"}:
            raise ValueError("TF-IDF vectorizer decode_error is invalid")
        if self.strip_accents not in {None, "ascii", "unicode"}:
            raise ValueError("TF-IDF vectorizer strip_accents is invalid")
        if self.preprocessor_policy != "none":
            raise ValueError("TF-IDF vectorizer supports only preprocessor_policy='none'")
        if self.tokenizer_policy != "token_pattern":
            raise ValueError(
                "TF-IDF vectorizer supports only tokenizer_policy='token_pattern'"
            )
        if self.analyzer not in {"word", "char", "char_wb"}:
            raise ValueError("TF-IDF vectorizer analyzer is invalid")
        if self.stop_words is not None:
            if isinstance(self.stop_words, str):
                if self.stop_words != "english":
                    raise ValueError(
                        "TF-IDF vectorizer stop_words string must be 'english'"
                    )
            elif isinstance(self.stop_words, (list, tuple)):
                normalized = [str(value) for value in self.stop_words]
                if not normalized or any(not value for value in normalized):
                    raise ValueError(
                        "TF-IDF vectorizer explicit stop_words must be nonempty strings"
                    )
                self.stop_words = normalized
            else:
                raise ValueError(
                    "TF-IDF vectorizer stop_words must be null, 'english', or a list"
                )
        if self.analyzer == "word":
            if not isinstance(self.token_pattern, str) or not self.token_pattern:
                raise ValueError(
                    "word TF-IDF vectorizer requires a nonempty token_pattern"
                )
        elif self.token_pattern is not None:
            raise ValueError(
                "character TF-IDF vectorizers must configure token_pattern=null"
            )
        if (
            isinstance(self.ngram_range_min, bool)
            or isinstance(self.ngram_range_max, bool)
            or int(self.ngram_range_min) < 1
            or int(self.ngram_range_max) < int(self.ngram_range_min)
        ):
            raise ValueError("TF-IDF vectorizer ngram range is invalid")
        self.ngram_range_min = int(self.ngram_range_min)
        self.ngram_range_max = int(self.ngram_range_max)
        if isinstance(self.min_df, bool) or int(self.min_df) < 1:
            raise ValueError("TF-IDF vectorizer min_df must be a positive count")
        self.min_df = int(self.min_df)
        if isinstance(self.max_df, bool) or not 0.0 < float(self.max_df) <= 1.0:
            raise ValueError("TF-IDF vectorizer max_df must be in (0, 1]")
        self.max_df = float(self.max_df)
        if self.max_features is not None:
            if isinstance(self.max_features, bool) or int(self.max_features) < 1:
                raise ValueError(
                    "TF-IDF vectorizer max_features must be null or positive"
                )
            self.max_features = int(self.max_features)
        if self.vocabulary_policy != "fit_scope_only":
            raise ValueError(
                "TF-IDF vectorizer vocabulary_policy must be 'fit_scope_only'"
            )
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("TF-IDF vectorizer dtype must be float32 or float64")
        if self.norm not in {None, "l1", "l2"}:
            raise ValueError("TF-IDF vectorizer norm is invalid")
        if self.feature_selection_rule != "sklearn_term_frequency_rank_v1":
            raise ValueError("TF-IDF vectorizer feature_selection_rule is invalid")


@dataclass
class LogisticRegressionScientificConfig:
    """Result-changing LogisticRegression settings other than C/iterations."""

    penalty: Optional[str] = "l2"
    dual: bool = False
    tol: float = 1e-4
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: Optional[Any] = None
    solver: str = "liblinear"
    multi_class: str = "auto"
    warm_start: bool = False
    l1_ratio: Optional[float] = None

    def __post_init__(self):
        if self.penalty not in {None, "l1", "l2", "elasticnet"}:
            raise ValueError("logistic penalty is invalid")
        if not float(self.tol) > 0.0:
            raise ValueError("logistic tol must be > 0")
        self.tol = float(self.tol)
        if not float(self.intercept_scaling) > 0.0:
            raise ValueError("logistic intercept_scaling must be > 0")
        self.intercept_scaling = float(self.intercept_scaling)
        if self.class_weight is not None and not (
            self.class_weight == "balanced" or isinstance(self.class_weight, dict)
        ):
            raise ValueError("logistic class_weight must be null, balanced, or a mapping")
        if self.solver not in {
            "lbfgs",
            "liblinear",
            "newton-cg",
            "newton-cholesky",
            "sag",
            "saga",
        }:
            raise ValueError("logistic solver is invalid")
        if self.multi_class not in {"auto", "ovr", "multinomial"}:
            raise ValueError("logistic multi_class is invalid")
        if self.l1_ratio is not None and not 0.0 <= float(self.l1_ratio) <= 1.0:
            raise ValueError("logistic l1_ratio must be null or in [0, 1]")
        if self.l1_ratio is not None:
            self.l1_ratio = float(self.l1_ratio)


@dataclass
class RidgeScientificConfig:
    """Closed result-changing Ridge settings other than alpha."""

    fit_intercept: bool = True
    max_iter: Optional[int] = None
    tol: float = 1e-4
    solver: str = "auto"
    positive: bool = False

    def __post_init__(self):
        if self.max_iter is not None:
            if isinstance(self.max_iter, bool) or int(self.max_iter) < 1:
                raise ValueError("ridge max_iter must be null or positive")
            self.max_iter = int(self.max_iter)
        if not float(self.tol) > 0.0:
            raise ValueError("ridge tol must be > 0")
        self.tol = float(self.tol)
        if self.solver not in {
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
            "lbfgs",
        }:
            raise ValueError("ridge solver is invalid")


@dataclass
class ForestScientificConfig:
    """Closed ExtraTrees/RandomForest classifier and regressor settings."""

    n_estimators: int = 300
    classifier_criterion: str = "gini"
    regressor_criterion: str = "squared_error"
    max_depth: Optional[int] = None
    min_samples_split: Any = 2
    min_samples_leaf: Any = 2
    min_weight_fraction_leaf: float = 0.0
    max_features: Any = "sqrt"
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    extra_trees_bootstrap: bool = False
    random_forest_bootstrap: bool = True
    oob_score: bool = False
    warm_start: bool = False
    class_weight: Optional[Any] = None
    ccp_alpha: float = 0.0
    max_samples: Optional[Any] = None
    monotonic_cst: Optional[List[int]] = None

    def __post_init__(self):
        if isinstance(self.n_estimators, bool) or int(self.n_estimators) < 1:
            raise ValueError("forest n_estimators must be positive")
        self.n_estimators = int(self.n_estimators)
        if self.classifier_criterion not in {"gini", "entropy", "log_loss"}:
            raise ValueError("forest classifier_criterion is invalid")
        if self.regressor_criterion not in {
            "squared_error",
            "absolute_error",
            "friedman_mse",
            "poisson",
        }:
            raise ValueError("forest regressor_criterion is invalid")
        if self.max_depth is not None:
            if isinstance(self.max_depth, bool) or int(self.max_depth) < 1:
                raise ValueError("forest max_depth must be null or positive")
            self.max_depth = int(self.max_depth)
        for name in ("min_samples_split", "min_samples_leaf"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"forest {name} must be an integer count or fraction")
            if isinstance(value, int):
                lower = 2 if name == "min_samples_split" else 1
                if value < lower:
                    raise ValueError(f"forest {name} count is too small")
            elif not 0.0 < float(value) <= 1.0:
                raise ValueError(f"forest {name} fraction must be in (0, 1]")
        if not 0.0 <= float(self.min_weight_fraction_leaf) <= 0.5:
            raise ValueError("forest min_weight_fraction_leaf must be in [0, 0.5]")
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        if self.max_features is not None and not isinstance(
            self.max_features, (str, int, float)
        ):
            raise ValueError("forest max_features has an unsupported type")
        if isinstance(self.max_features, str) and self.max_features not in {
            "sqrt",
            "log2",
        }:
            raise ValueError("forest max_features string is invalid")
        if self.max_leaf_nodes is not None:
            if isinstance(self.max_leaf_nodes, bool) or int(self.max_leaf_nodes) < 2:
                raise ValueError("forest max_leaf_nodes must be null or at least two")
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        if float(self.min_impurity_decrease) < 0.0:
            raise ValueError("forest min_impurity_decrease must be >= 0")
        self.min_impurity_decrease = float(self.min_impurity_decrease)
        if self.class_weight is not None and not (
            isinstance(self.class_weight, (dict, list))
            or self.class_weight in {"balanced", "balanced_subsample"}
        ):
            raise ValueError("forest class_weight is invalid")
        if float(self.ccp_alpha) < 0.0:
            raise ValueError("forest ccp_alpha must be >= 0")
        self.ccp_alpha = float(self.ccp_alpha)
        if self.max_samples is not None:
            if isinstance(self.max_samples, bool) or not isinstance(
                self.max_samples, (int, float)
            ):
                raise ValueError("forest max_samples must be null, a count, or a fraction")
            if isinstance(self.max_samples, int):
                if self.max_samples < 1:
                    raise ValueError("forest max_samples count must be positive")
            elif not 0.0 < float(self.max_samples) <= 1.0:
                raise ValueError("forest max_samples fraction must be in (0, 1]")
        if self.monotonic_cst is not None:
            self.monotonic_cst = [int(value) for value in self.monotonic_cst]
            if any(value not in {-1, 0, 1} for value in self.monotonic_cst):
                raise ValueError("forest monotonic_cst values must be -1, 0, or 1")


@dataclass
class XGBoostScientificConfig:
    """Closed XGBoost tree-booster science; device/worker count is operational."""

    n_estimators: int = 300
    max_depth: int = 3
    max_leaves: int = 0
    max_bin: int = 256
    grow_policy: str = "depthwise"
    learning_rate: float = 0.05
    booster: str = "gbtree"
    tree_method: str = "hist"
    gamma: float = 0.0
    min_child_weight: float = 1.0
    max_delta_step: float = 0.0
    subsample: float = 0.9
    sampling_method: str = "uniform"
    colsample_bytree: float = 0.6
    colsample_bylevel: float = 1.0
    colsample_bynode: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0
    base_score: float = 0.5
    missing_value_policy: str = "nan"
    num_parallel_tree: int = 1
    monotone_constraints: Optional[Any] = None
    interaction_constraints: Optional[Any] = None
    enable_categorical: bool = False
    max_cat_to_onehot: int = 4
    max_cat_threshold: int = 64
    multi_strategy: str = "one_output_per_tree"
    classifier_objective: str = "binary:logistic"
    classifier_eval_metric: str = "logloss"
    regressor_objective: str = "reg:squarederror"
    regressor_eval_metric: str = "rmse"

    def __post_init__(self):
        for name, minimum in (
            ("n_estimators", 1),
            ("max_depth", 0),
            ("max_leaves", 0),
            ("max_bin", 2),
            ("num_parallel_tree", 1),
            ("max_cat_to_onehot", 1),
            ("max_cat_threshold", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < minimum:
                raise ValueError(f"xgboost {name} is invalid")
            setattr(self, name, int(value))
        if self.grow_policy not in {"depthwise", "lossguide"}:
            raise ValueError("xgboost grow_policy is invalid")
        if self.booster not in {"gbtree", "dart"}:
            raise ValueError("xgboost booster must be gbtree or dart")
        if self.tree_method not in {"auto", "exact", "approx", "hist"}:
            raise ValueError("xgboost tree_method is invalid")
        if self.sampling_method not in {"uniform", "gradient_based"}:
            raise ValueError("xgboost sampling_method is invalid")
        for name in (
            "learning_rate",
            "min_child_weight",
            "reg_lambda",
            "scale_pos_weight",
            "base_score",
        ):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"xgboost {name} must be > 0")
            setattr(self, name, float(getattr(self, name)))
        for name in ("gamma", "max_delta_step", "reg_alpha"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"xgboost {name} must be >= 0")
            setattr(self, name, float(getattr(self, name)))
        for name in (
            "subsample",
            "colsample_bytree",
            "colsample_bylevel",
            "colsample_bynode",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"xgboost {name} must be in (0, 1]")
            setattr(self, name, value)
        if self.missing_value_policy != "nan":
            raise ValueError("xgboost missing_value_policy must be 'nan'")
        if self.multi_strategy not in {"one_output_per_tree", "multi_output_tree"}:
            raise ValueError("xgboost multi_strategy is invalid")
        for name in (
            "classifier_objective",
            "classifier_eval_metric",
            "regressor_objective",
            "regressor_eval_metric",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"xgboost {name} must be nonempty")


@dataclass
class NuisanceCalibrationScientificConfig:
    """Closed probability calibration diagnostics for nuisance models."""

    probability_clip_epsilon: float = 1e-6
    logistic_c: float = 1e6
    logistic_max_iter: int = 1000
    logistic: LogisticRegressionScientificConfig = field(
        default_factory=lambda: LogisticRegressionScientificConfig(
            solver="lbfgs",
        )
    )
    ece_bins: int = 10
    ece_strategy: str = "uniform_width"

    def __post_init__(self):
        if isinstance(self.logistic, dict):
            self.logistic = LogisticRegressionScientificConfig(**self.logistic)
        if type(self.logistic) is not LogisticRegressionScientificConfig:
            raise ValueError("nuisance calibration logistic config must be typed")
        if not 0.0 < float(self.probability_clip_epsilon) < 0.5:
            raise ValueError("calibration probability_clip_epsilon must be in (0, 0.5)")
        self.probability_clip_epsilon = float(self.probability_clip_epsilon)
        if not float(self.logistic_c) > 0.0:
            raise ValueError("calibration logistic_c must be > 0")
        self.logistic_c = float(self.logistic_c)
        if isinstance(self.logistic_max_iter, bool) or int(self.logistic_max_iter) < 1:
            raise ValueError("calibration logistic_max_iter must be positive")
        self.logistic_max_iter = int(self.logistic_max_iter)
        if isinstance(self.ece_bins, bool) or int(self.ece_bins) < 2:
            raise ValueError("calibration ece_bins must be at least two")
        self.ece_bins = int(self.ece_bins)
        if self.ece_strategy != "uniform_width":
            raise ValueError("calibration ece_strategy must be uniform_width")


@dataclass
class TfidfNuisanceStackScientificConfig:
    """Exact nested cross-fit, meta-model, and degeneracy policies."""

    meta_logistic_c: float = 1.0
    meta_logistic_max_iter: int = 1000
    meta_logistic: LogisticRegressionScientificConfig = field(
        default_factory=lambda: LogisticRegressionScientificConfig(
            solver="lbfgs",
        )
    )
    meta_ridge_alpha: float = 1.0
    meta_ridge: RidgeScientificConfig = field(default_factory=RidgeScientificConfig)
    calibration: NuisanceCalibrationScientificConfig = field(
        default_factory=NuisanceCalibrationScientificConfig
    )
    split_policy: str = "stratified_if_feasible_else_kfold_v1"
    split_shuffle: bool = True
    seed_derivation_policy: str = "scope_seed_offset_v1"
    single_class_policy: str = "empirical_mean_constant"
    empty_vocabulary_policy: str = "fail_closed"
    meta_degenerate_policy: str = "empirical_mean_constant"

    def __post_init__(self):
        if isinstance(self.meta_logistic, dict):
            self.meta_logistic = LogisticRegressionScientificConfig(
                **self.meta_logistic
            )
        if isinstance(self.meta_ridge, dict):
            self.meta_ridge = RidgeScientificConfig(**self.meta_ridge)
        if isinstance(self.calibration, dict):
            self.calibration = NuisanceCalibrationScientificConfig(
                **self.calibration
            )
        if type(self.meta_logistic) is not LogisticRegressionScientificConfig:
            raise ValueError("TF-IDF meta logistic config must be typed")
        if type(self.meta_ridge) is not RidgeScientificConfig:
            raise ValueError("TF-IDF meta ridge config must be typed")
        if type(self.calibration) is not NuisanceCalibrationScientificConfig:
            raise ValueError("TF-IDF calibration config must be typed")
        if not float(self.meta_logistic_c) > 0.0:
            raise ValueError("TF-IDF meta_logistic_c must be > 0")
        self.meta_logistic_c = float(self.meta_logistic_c)
        if (
            isinstance(self.meta_logistic_max_iter, bool)
            or int(self.meta_logistic_max_iter) < 1
        ):
            raise ValueError("TF-IDF meta_logistic_max_iter must be positive")
        self.meta_logistic_max_iter = int(self.meta_logistic_max_iter)
        if float(self.meta_ridge_alpha) < 0.0:
            raise ValueError("TF-IDF meta_ridge_alpha must be >= 0")
        self.meta_ridge_alpha = float(self.meta_ridge_alpha)
        if self.split_policy != "stratified_if_feasible_else_kfold_v1":
            raise ValueError("TF-IDF nuisance split_policy is invalid")
        if self.seed_derivation_policy != "scope_seed_offset_v1":
            raise ValueError("TF-IDF nuisance seed_derivation_policy is invalid")
        if self.single_class_policy != "empirical_mean_constant":
            raise ValueError("TF-IDF nuisance single_class_policy is invalid")
        if self.empty_vocabulary_policy not in {
            "fail_closed",
            "empirical_mean_constant",
        }:
            raise ValueError("TF-IDF nuisance empty_vocabulary_policy is invalid")
        if self.meta_degenerate_policy != "empirical_mean_constant":
            raise ValueError("TF-IDF nuisance meta_degenerate_policy is invalid")


@dataclass
class NMFScientificConfig:
    """Remaining result-changing NMF constructor and numerical settings."""

    alpha_w: float = 0.0
    alpha_h: Any = "same"
    l1_ratio: float = 0.0
    shuffle: bool = False
    importance_scale_epsilon: float = 1e-12
    component_capacity_policy: str = "min_requested_rows_minus_one_selected_terms"

    def __post_init__(self):
        if float(self.alpha_w) < 0.0:
            raise ValueError("NMF alpha_w must be >= 0")
        self.alpha_w = float(self.alpha_w)
        if self.alpha_h != "same":
            if isinstance(self.alpha_h, bool) or float(self.alpha_h) < 0.0:
                raise ValueError("NMF alpha_h must be 'same' or >= 0")
            self.alpha_h = float(self.alpha_h)
        if not 0.0 <= float(self.l1_ratio) <= 1.0:
            raise ValueError("NMF l1_ratio must be in [0, 1]")
        self.l1_ratio = float(self.l1_ratio)
        if not float(self.importance_scale_epsilon) > 0.0:
            raise ValueError("NMF importance_scale_epsilon must be > 0")
        self.importance_scale_epsilon = float(self.importance_scale_epsilon)
        if (
            self.component_capacity_policy
            != "min_requested_rows_minus_one_selected_terms"
        ):
            raise ValueError("NMF component_capacity_policy is invalid")


@dataclass
class TfidfScreeningScientificConfig:
    """Closed source selection and aggregation weights for linear screens."""

    model_source_view_name: str = "linear_1_3"
    linear_base_weight: float = 0.5
    linear_selection_stability_weight: float = 0.25
    linear_rank_stability_weight: float = 0.25
    effect_base_weight: float = 0.4
    effect_nuisance_agreement_weight: float = 0.2
    effect_selection_stability_weight: float = 0.2
    effect_sign_stability_weight: float = 0.2
    seed_derivation_policy: str = "scope_seed_offset_v1"
    topic_term_evidence_policy: str = "fail_closed_exact_configured_capacity"

    def __post_init__(self):
        self.model_source_view_name = str(self.model_source_view_name).strip()
        if not self.model_source_view_name:
            raise ValueError("TF-IDF screening model_source_view_name is required")
        linear = (
            float(self.linear_base_weight),
            float(self.linear_selection_stability_weight),
            float(self.linear_rank_stability_weight),
        )
        effect = (
            float(self.effect_base_weight),
            float(self.effect_nuisance_agreement_weight),
            float(self.effect_selection_stability_weight),
            float(self.effect_sign_stability_weight),
        )
        if any(value < 0.0 for value in (*linear, *effect)):
            raise ValueError("TF-IDF screening weights must be nonnegative")
        if abs(sum(linear) - 1.0) > 1e-12 or abs(sum(effect) - 1.0) > 1e-12:
            raise ValueError("TF-IDF screening weight groups must each sum to one")
        (
            self.linear_base_weight,
            self.linear_selection_stability_weight,
            self.linear_rank_stability_weight,
        ) = linear
        (
            self.effect_base_weight,
            self.effect_nuisance_agreement_weight,
            self.effect_selection_stability_weight,
            self.effect_sign_stability_weight,
        ) = effect
        if self.seed_derivation_policy != "scope_seed_offset_v1":
            raise ValueError("TF-IDF screening seed_derivation_policy is invalid")
        if self.topic_term_evidence_policy != "fail_closed_exact_configured_capacity":
            raise ValueError("TF-IDF topic term evidence policy is invalid")


@dataclass
class OrphanSemanticClusteringScientificConfig:
    """Exact sparse-semantic clustering settings for residual n-gram groups."""

    alias_jaccard_threshold: float = 0.8
    word_vectorizer: TfidfVectorizerScientificConfig = field(
        default_factory=lambda: TfidfVectorizerScientificConfig(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range_min=1,
            ngram_range_max=2,
            min_df=1,
            max_df=1.0,
            max_features=None,
            sublinear_tf=False,
        )
    )
    char_vectorizer: TfidfVectorizerScientificConfig = field(
        default_factory=lambda: TfidfVectorizerScientificConfig(
            lowercase=True,
            analyzer="char_wb",
            token_pattern=None,
            ngram_range_min=3,
            ngram_range_max=5,
            min_df=1,
            max_df=1.0,
            max_features=None,
            sublinear_tf=False,
        )
    )
    word_similarity_weight: float = 0.35
    char_similarity_weight: float = 0.35
    occurrence_similarity_weight: float = 0.30
    row_normalization_norm: str = "l2"
    neighbor_metric: str = "cosine"
    neighbor_algorithm: str = "brute"

    def __post_init__(self):
        if isinstance(self.word_vectorizer, dict):
            self.word_vectorizer = TfidfVectorizerScientificConfig(
                **self.word_vectorizer
            )
        if isinstance(self.char_vectorizer, dict):
            self.char_vectorizer = TfidfVectorizerScientificConfig(
                **self.char_vectorizer
            )
        if type(self.word_vectorizer) is not TfidfVectorizerScientificConfig:
            raise ValueError("orphan word_vectorizer must be typed")
        if type(self.char_vectorizer) is not TfidfVectorizerScientificConfig:
            raise ValueError("orphan char_vectorizer must be typed")
        if not 0.0 <= float(self.alias_jaccard_threshold) <= 1.0:
            raise ValueError("orphan alias_jaccard_threshold must be in [0, 1]")
        self.alias_jaccard_threshold = float(self.alias_jaccard_threshold)
        weights = (
            float(self.word_similarity_weight),
            float(self.char_similarity_weight),
            float(self.occurrence_similarity_weight),
        )
        if any(value < 0.0 for value in weights) or abs(sum(weights) - 1.0) > 1e-12:
            raise ValueError("orphan semantic similarity weights must sum to one")
        (
            self.word_similarity_weight,
            self.char_similarity_weight,
            self.occurrence_similarity_weight,
        ) = weights
        if self.row_normalization_norm != "l2":
            raise ValueError("orphan row_normalization_norm must be l2")
        if self.neighbor_metric != "cosine":
            raise ValueError("orphan neighbor_metric must be cosine")
        if self.neighbor_algorithm != "brute":
            raise ValueError("orphan neighbor_algorithm must be brute")


@dataclass
class BoWViewConfig:
    """One sparse text-model view used by multi-model agentic discovery."""

    name: str = ""
    max_features: int = 30000
    min_df: int = 5
    max_df: float = 0.95
    ngram_range_min: int = 1
    ngram_range_max: int = 3
    sublinear_tf: bool = True
    # Learner family for BoW nuisance and pseudo-target models:
    # "linear", "extratrees", "random_forest", or "xgboost".
    bow_model: str = "linear"
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    ridge_alpha: float = 10.0
    vectorizer_scientific: Optional[TfidfVectorizerScientificConfig] = None
    logistic_scientific: LogisticRegressionScientificConfig = field(
        default_factory=LogisticRegressionScientificConfig
    )
    ridge_scientific: RidgeScientificConfig = field(
        default_factory=RidgeScientificConfig
    )
    forest_scientific: ForestScientificConfig = field(
        default_factory=ForestScientificConfig
    )
    xgboost_scientific: XGBoostScientificConfig = field(
        default_factory=XGBoostScientificConfig
    )
    single_class_policy: str = "empirical_mean_constant"
    empty_vocabulary_policy: str = "fail_closed"
    unsupported_sample_weight_policy: str = "fail_closed"

    def __post_init__(self):
        self.name = str(self.name or "").strip()
        if self.max_features < 1:
            raise ValueError("bow view max_features must be >= 1")
        if self.min_df < 1:
            raise ValueError("bow view min_df must be >= 1")
        if not 0.0 < self.max_df <= 1.0:
            raise ValueError("bow view max_df must be in (0, 1]")
        if self.ngram_range_min < 1 or self.ngram_range_max < self.ngram_range_min:
            raise ValueError(
                "bow view ngram range must satisfy " "1 <= ngram_range_min <= ngram_range_max"
            )
        bow_model = str(self.bow_model).strip().lower()
        if bow_model not in {"linear", "extratrees", "random_forest", "xgboost"}:
            raise ValueError(
                "bow view bow_model must be one of "
                "'linear', 'extratrees', 'random_forest', or 'xgboost'"
            )
        self.bow_model = bow_model
        if self.logistic_c <= 0:
            raise ValueError("bow view logistic_c must be > 0")
        if self.logistic_max_iter < 1:
            raise ValueError("bow view logistic_max_iter must be >= 1")
        if self.ridge_alpha < 0:
            raise ValueError("bow view ridge_alpha must be >= 0")
        if self.vectorizer_scientific is None:
            self.vectorizer_scientific = TfidfVectorizerScientificConfig(
                ngram_range_min=int(self.ngram_range_min),
                ngram_range_max=int(self.ngram_range_max),
                min_df=int(self.min_df),
                max_df=float(self.max_df),
                max_features=int(self.max_features),
                sublinear_tf=bool(self.sublinear_tf),
            )
        elif isinstance(self.vectorizer_scientific, dict):
            self.vectorizer_scientific = TfidfVectorizerScientificConfig(
                **self.vectorizer_scientific
            )
        if type(self.vectorizer_scientific) is not TfidfVectorizerScientificConfig:
            raise ValueError("bow view vectorizer_scientific must be typed")
        vectorizer = self.vectorizer_scientific
        if (
            vectorizer.ngram_range_min != int(self.ngram_range_min)
            or vectorizer.ngram_range_max != int(self.ngram_range_max)
            or vectorizer.min_df != int(self.min_df)
            or vectorizer.max_df != float(self.max_df)
            or vectorizer.max_features != int(self.max_features)
            or vectorizer.sublinear_tf != bool(self.sublinear_tf)
        ):
            raise ValueError(
                "bow view compatibility fields differ from vectorizer_scientific"
            )
        if isinstance(self.logistic_scientific, dict):
            self.logistic_scientific = LogisticRegressionScientificConfig(
                **self.logistic_scientific
            )
        if type(self.logistic_scientific) is not LogisticRegressionScientificConfig:
            raise ValueError("bow view logistic_scientific must be typed")
        if isinstance(self.ridge_scientific, dict):
            self.ridge_scientific = RidgeScientificConfig(
                **self.ridge_scientific
            )
        if type(self.ridge_scientific) is not RidgeScientificConfig:
            raise ValueError("bow view ridge_scientific must be typed")
        if isinstance(self.forest_scientific, dict):
            self.forest_scientific = ForestScientificConfig(
                **self.forest_scientific
            )
        if type(self.forest_scientific) is not ForestScientificConfig:
            raise ValueError("bow view forest_scientific must be typed")
        if isinstance(self.xgboost_scientific, dict):
            self.xgboost_scientific = XGBoostScientificConfig(
                **self.xgboost_scientific
            )
        if type(self.xgboost_scientific) is not XGBoostScientificConfig:
            raise ValueError("bow view xgboost_scientific must be typed")
        if self.empty_vocabulary_policy not in {
            "fail_closed",
            "empirical_mean_constant",
        }:
            raise ValueError("bow view empty_vocabulary_policy is invalid")
        if self.single_class_policy != "empirical_mean_constant":
            raise ValueError("bow view single_class_policy is invalid")
        if self.unsupported_sample_weight_policy not in {
            "fail_closed",
            "unweighted_legacy_compatibility",
        }:
            raise ValueError(
                "bow view unsupported_sample_weight_policy is invalid"
            )
        if self.bow_model == "xgboost":
            import importlib.util

            if importlib.util.find_spec("xgboost") is None:
                raise ValueError(
                    "bow_model='xgboost' is configured but xgboost is unavailable; "
                    "refusing learner substitution"
                )


def legacy_default_bow_views_v1() -> List[BoWViewConfig]:
    """Historical implicit grid for non-production compatibility callers only."""
    return [
        BoWViewConfig(
            name="linear_unigram_c0p5",
            bow_model="linear",
            ngram_range_min=1,
            ngram_range_max=1,
            logistic_c=0.5,
            ridge_alpha=20.0,
        ),
        BoWViewConfig(
            name="linear_1_2",
            bow_model="linear",
            ngram_range_min=1,
            ngram_range_max=2,
            logistic_c=1.0,
            ridge_alpha=10.0,
        ),
        BoWViewConfig(
            name="linear_1_3",
            bow_model="linear",
            ngram_range_min=1,
            ngram_range_max=3,
            logistic_c=1.0,
            ridge_alpha=10.0,
        ),
        BoWViewConfig(
            name="linear_2_4_min_df3",
            bow_model="linear",
            ngram_range_min=2,
            ngram_range_max=4,
            min_df=3,
            logistic_c=1.0,
            ridge_alpha=10.0,
        ),
        BoWViewConfig(
            name="extratrees_1_3",
            bow_model="extratrees",
            ngram_range_min=1,
            ngram_range_max=3,
        ),
        BoWViewConfig(
            name="random_forest_1_2",
            bow_model="random_forest",
            ngram_range_min=1,
            ngram_range_max=2,
        ),
    ]


def default_multi_model_bow_views() -> List[BoWViewConfig]:
    """Deprecated compatibility alias; portable production never calls this."""

    return legacy_default_bow_views_v1()


@dataclass
class TfidfTopicDiscoveryConfig:
    """Deterministic nuisance, contrast, and consensus-NMF settings."""

    max_features: int = 30000
    ngram_range_min: int = 1
    ngram_range_max: int = 3
    min_df: int = 5
    max_df: float = 0.98
    sublinear_tf: bool = True
    vectorizer_scientific: Optional[TfidfVectorizerScientificConfig] = None
    nuisance_stack_scientific: TfidfNuisanceStackScientificConfig = field(
        default_factory=TfidfNuisanceStackScientificConfig
    )
    top_fraction: float = 0.10
    topic_count: int = 100
    topic_seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    # Complete, ordered term evidence supplied for each fitted topic. This is
    # a scientific capacity selected by the deployment, not a source-code
    # prompt constant. Downstream consumers must account for exactly this many
    # terms or fail closed; they may not slice the configured evidence.
    terms_per_topic: int = 15
    nmf_init: str = "nndsvdar"
    nmf_solver: str = "cd"
    nmf_beta_loss: str = "frobenius"
    nmf_max_iter: int = 400
    nmf_tol: float = 1e-4
    nmf_scientific: NMFScientificConfig = field(
        default_factory=NMFScientificConfig
    )
    screening_scientific: TfidfScreeningScientificConfig = field(
        default_factory=TfidfScreeningScientificConfig
    )
    importance_weight_min: float = 0.5
    importance_weight_max: float = 2.0
    stability_repeats: int = 30
    stability_fraction: float = 0.75
    minimum_arm_document_support: int = 2
    minimum_nuisance_source_agreement: float = 0.50
    minimum_subsample_selection_fraction: float = 0.20
    minimum_tail_sign_agreement: float = 0.50
    topic_label_parallelism: int = 8
    initial_effect_coverage_target: float = 0.80
    topic_reconstruction_tolerance: float = 0.03
    contrast_coverage_tolerance: float = 0.05
    # Honest inner-held-out group score tests used to filter topics before
    # agent labeling.  Every bank retains a bounded evidence-ranked set; the
    # minimum is a power safeguard rather than a significance claim.
    score_test_enabled: bool = True
    # Historical experiments scored on the registered context holdout. New
    # production bundles split the registered fit partition again, freeze
    # selection there, and only then transform the label-free holdout.
    score_selection_label_policy: str = "registered_context_heldout"
    score_test_bootstrap_repeats: int = 500
    # Zero means every fitted topic.  Production defaults to the complete
    # family so a topic is never chosen for bootstrap calibration after its
    # held-out statistic has already been inspected.  A positive limit is an
    # explicitly approximate/debug mode; selection then falls back to the
    # asymptotic p-values shared by the complete family.
    score_test_bootstrap_top_topics: int = 0
    score_test_bootstrap_chunk_size: int = 100
    score_test_fdr_level: float = 0.20
    score_test_p_threshold: float = 0.10
    score_test_min_topics_per_bank: int = 5
    score_test_max_topics_per_bank: int = 20
    score_test_full_topic_min_inner_folds: int = 1
    # Opt-in sparse skip connection around NMF. Candidate groups are built only from
    # stable fit-side effect n-grams that are absent from every fitted topic's
    # configured term summary, then tested once on the exact inner-held-out rows.
    orphan_ngram_enabled: bool = False
    orphan_ngram_min_abs_fit_score: float = 2.0
    orphan_ngram_cluster_similarity_threshold: float = 0.25
    orphan_ngram_cluster_max_terms: int = 15
    orphan_ngram_cluster_neighbors: int = 20
    orphan_ngram_fdr_level: float = 0.20
    orphan_ngram_p_threshold: float = 0.10
    orphan_ngram_min_selected_clusters: int = 5
    orphan_ngram_max_selected_clusters: int = 20
    orphan_ngram_full_min_inner_folds: int = 1
    orphan_semantic_clustering_scientific: OrphanSemanticClusteringScientificConfig = field(
        default_factory=OrphanSemanticClusteringScientificConfig
    )
    prompt_version: str = "tfidf_topic_label_v2"
    random_state: int = 42

    def __post_init__(self):
        if self.max_features < 1:
            raise ValueError("tfidf_topic.max_features must be >= 1")
        if self.ngram_range_min < 1 or self.ngram_range_max < self.ngram_range_min:
            raise ValueError("tfidf_topic ngram range is invalid")
        if self.min_df < 1:
            raise ValueError("tfidf_topic.min_df must be >= 1")
        if not 0.0 < self.max_df <= 1.0:
            raise ValueError("tfidf_topic.max_df must be in (0, 1]")
        if self.vectorizer_scientific is None:
            self.vectorizer_scientific = TfidfVectorizerScientificConfig(
                ngram_range_min=int(self.ngram_range_min),
                ngram_range_max=int(self.ngram_range_max),
                min_df=int(self.min_df),
                max_df=float(self.max_df),
                max_features=int(self.max_features),
                sublinear_tf=bool(self.sublinear_tf),
            )
        elif isinstance(self.vectorizer_scientific, dict):
            self.vectorizer_scientific = TfidfVectorizerScientificConfig(
                **self.vectorizer_scientific
            )
        if type(self.vectorizer_scientific) is not TfidfVectorizerScientificConfig:
            raise ValueError("tfidf_topic.vectorizer_scientific must be typed")
        vectorizer = self.vectorizer_scientific
        if (
            vectorizer.ngram_range_min != int(self.ngram_range_min)
            or vectorizer.ngram_range_max != int(self.ngram_range_max)
            or vectorizer.min_df != int(self.min_df)
            or vectorizer.max_df != float(self.max_df)
            or vectorizer.max_features != int(self.max_features)
            or vectorizer.sublinear_tf != bool(self.sublinear_tf)
        ):
            raise ValueError(
                "tfidf_topic compatibility fields differ from vectorizer_scientific"
            )
        if isinstance(self.nuisance_stack_scientific, dict):
            self.nuisance_stack_scientific = TfidfNuisanceStackScientificConfig(
                **self.nuisance_stack_scientific
            )
        if (
            type(self.nuisance_stack_scientific)
            is not TfidfNuisanceStackScientificConfig
        ):
            raise ValueError("tfidf_topic.nuisance_stack_scientific must be typed")
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("tfidf_topic.top_fraction must be in (0, 1]")
        if self.topic_count < 1:
            raise ValueError("tfidf_topic.topic_count must be >= 1")
        self.topic_seeds = [int(seed) for seed in self.topic_seeds]
        if not self.topic_seeds:
            raise ValueError("tfidf_topic.topic_seeds must not be empty")
        if int(self.terms_per_topic) < 1:
            raise ValueError("tfidf_topic.terms_per_topic must be >= 1")
        self.terms_per_topic = int(self.terms_per_topic)
        if self.nmf_init != "nndsvdar" or self.nmf_solver != "cd":
            raise ValueError("tfidf_topic v2 requires nndsvdar coordinate-descent NMF")
        if self.nmf_beta_loss != "frobenius":
            raise ValueError("tfidf_topic v2 requires the Frobenius objective")
        if self.nmf_max_iter < 1 or self.nmf_tol <= 0.0:
            raise ValueError("tfidf_topic NMF convergence settings are invalid")
        if isinstance(self.nmf_scientific, dict):
            self.nmf_scientific = NMFScientificConfig(**self.nmf_scientific)
        if type(self.nmf_scientific) is not NMFScientificConfig:
            raise ValueError("tfidf_topic.nmf_scientific must be typed")
        if isinstance(self.screening_scientific, dict):
            self.screening_scientific = TfidfScreeningScientificConfig(
                **self.screening_scientific
            )
        if type(self.screening_scientific) is not TfidfScreeningScientificConfig:
            raise ValueError("tfidf_topic.screening_scientific must be typed")
        if not (
            float(self.importance_weight_min) > 0.0
            and float(self.importance_weight_max)
            >= float(self.importance_weight_min)
        ):
            raise ValueError(
                "tfidf_topic importance weights must satisfy "
                "0 < importance_weight_min <= importance_weight_max"
            )
        self.importance_weight_min = float(self.importance_weight_min)
        self.importance_weight_max = float(self.importance_weight_max)
        if int(self.minimum_arm_document_support) < 1:
            raise ValueError(
                "tfidf_topic.minimum_arm_document_support must be >= 1"
            )
        self.minimum_arm_document_support = int(
            self.minimum_arm_document_support
        )
        if self.stability_repeats < 0:
            raise ValueError("tfidf_topic.stability_repeats must be >= 0")
        if not 0.0 < self.stability_fraction <= 1.0:
            raise ValueError("tfidf_topic.stability_fraction must be in (0, 1]")
        if self.topic_label_parallelism < 1:
            raise ValueError("tfidf_topic.topic_label_parallelism must be >= 1")
        if self.score_test_bootstrap_repeats < 0:
            raise ValueError("tfidf_topic.score_test_bootstrap_repeats must be >= 0")
        if self.score_selection_label_policy not in {
            "registered_context_heldout",
            "nested_fit_calibration",
        }:
            raise ValueError(
                "tfidf_topic.score_selection_label_policy must be "
                "registered_context_heldout or nested_fit_calibration"
            )
        if self.score_test_bootstrap_top_topics < 0:
            raise ValueError("tfidf_topic.score_test_bootstrap_top_topics must be >= 0")
        if self.score_test_bootstrap_chunk_size < 1:
            raise ValueError("tfidf_topic.score_test_bootstrap_chunk_size must be >= 1")
        if self.score_test_min_topics_per_bank < 0:
            raise ValueError("tfidf_topic.score_test_min_topics_per_bank must be >= 0")
        if self.score_test_max_topics_per_bank < self.score_test_min_topics_per_bank:
            raise ValueError(
                "tfidf_topic.score_test_max_topics_per_bank must be >= "
                "score_test_min_topics_per_bank"
            )
        if self.score_test_full_topic_min_inner_folds < 1:
            raise ValueError("tfidf_topic.score_test_full_topic_min_inner_folds must be >= 1")
        if float(self.orphan_ngram_min_abs_fit_score) < 0.0:
            raise ValueError("tfidf_topic.orphan_ngram_min_abs_fit_score must be >= 0")
        if int(self.orphan_ngram_cluster_max_terms) < 1:
            raise ValueError(
                "tfidf_topic.orphan_ngram_cluster_max_terms must be >= 1"
            )
        if int(self.orphan_ngram_cluster_neighbors) < 1:
            raise ValueError("tfidf_topic.orphan_ngram_cluster_neighbors must be >= 1")
        if int(self.orphan_ngram_max_selected_clusters) < 0:
            raise ValueError("tfidf_topic.orphan_ngram_max_selected_clusters must be >= 0")
        if (
            not 0
            <= int(self.orphan_ngram_min_selected_clusters)
            <= int(self.orphan_ngram_max_selected_clusters)
        ):
            raise ValueError(
                "tfidf_topic.orphan_ngram_min_selected_clusters must be in "
                "[0, orphan_ngram_max_selected_clusters]"
            )
        if int(self.orphan_ngram_full_min_inner_folds) < 1:
            raise ValueError("tfidf_topic.orphan_ngram_full_min_inner_folds must be >= 1")
        if isinstance(self.orphan_semantic_clustering_scientific, dict):
            self.orphan_semantic_clustering_scientific = (
                OrphanSemanticClusteringScientificConfig(
                    **self.orphan_semantic_clustering_scientific
                )
            )
        if (
            type(self.orphan_semantic_clustering_scientific)
            is not OrphanSemanticClusteringScientificConfig
        ):
            raise ValueError(
                "tfidf_topic.orphan_semantic_clustering_scientific must be typed"
            )
        if self.orphan_ngram_enabled and not self.score_test_enabled:
            raise ValueError("tfidf_topic.orphan_ngram_enabled requires score_test_enabled")
        for field_name in (
            "minimum_nuisance_source_agreement",
            "minimum_subsample_selection_fraction",
            "minimum_tail_sign_agreement",
            "initial_effect_coverage_target",
            "topic_reconstruction_tolerance",
            "contrast_coverage_tolerance",
            "score_test_fdr_level",
            "score_test_p_threshold",
            "orphan_ngram_cluster_similarity_threshold",
            "orphan_ngram_fdr_level",
            "orphan_ngram_p_threshold",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"tfidf_topic.{field_name} must be in [0, 1]")


@dataclass
class MultiModelAgenticForestConfig:
    """Configuration for BoW-guided agentic variable discovery.

    This pathway uses multiple cross-fitted sparse text-model views,
    embedding-contrast retrieval, and HTR attention/span evidence to produce
    agent-facing evidence.
    The proposal agent is used as a candidate-generation tool. Downstream
    extraction review, parsimony review, and honest causal-forest fitting decide
    which explicit patient-level variables are retained.
    """

    nuisance_folds: int = 5
    effect_folds: int = 5
    # Explicit selector for the evidence/discovery methods used in this run.
    # Accepted entries: "bow", "htr", and "embedding_contrast".
    # When omitted, the selector is derived from the per-method toggles below.
    feature_discovery_methods: Optional[List[str]] = None
    bow_discovery_enabled: bool = True
    bow_discovery_disable_reason: Optional[str] = None
    bow_views: List[BoWViewConfig] = field(default_factory=list)
    # Optional researcher-provided variables to extract before BoW discovery.
    # Items must be ExplicitFeatureSpec-shaped dicts. confounder/effect-modifier
    # section fields apply the corresponding role automatically; duplicate names
    # are merged downstream, so the same variable can be supplied in both lists.
    prespecified_features: List[ExplicitFeatureSpec] = field(default_factory=list)
    prespecified_confounders: List[ExplicitFeatureSpec] = field(default_factory=list)
    prespecified_effect_modifiers: List[ExplicitFeatureSpec] = field(default_factory=list)
    prespecified_features_json: Optional[str] = None
    e_clip: float = 0.01
    top_n_features: int = 100
    candidate_proposals_per_fold: int = 60
    # "evidence_digest" gives agents compact role-specific evidence blurbs by
    # default. "rich_context" preserves the older full compacted evidence object.
    agent_context_mode: str = "evidence_digest"
    concept_inventory_enabled: bool = True
    concept_inventory_max_concepts: int = 60
    candidate_consistency_enabled: bool = True
    candidate_consistency_inner_folds: int = 3
    # Fit-only calibration inside a registered TF-IDF scope. This is distinct
    # from hierarchy partitions and downstream interaction cross-fitting.
    tfidf_nested_calibration_folds: int = 3
    candidate_consistency_min_folds: int = 2
    candidate_consistency_min_fold_fraction: float = 0.5
    candidate_consistency_recovery_max_candidates: int = 12
    candidate_consistency_parallelism: str = "1"
    extracted_feature_review_enabled: bool = True
    extracted_feature_review_max_rounds: int = 3
    extracted_feature_review_auc_margin: float = 0.02
    extracted_feature_review_loss_relative_margin: float = 0.05
    extracted_feature_review_min_benchmark_auc: float = 0.55
    # Value-driven cluster-to-factor parsimony before final forest fitting.
    # The clustering pass uses the actual outer-training-fold extracted values
    # together with feature-contract semantics.  An agent may then propose up
    # to a small number of operationalized latent factors for each coherent
    # cluster; replacements are retained only when every applicable diagnostic
    # metric is preserved or improved.
    parsimony_review_enabled: bool = False
    parsimony_cluster_semantic_weight: float = 0.5
    parsimony_cluster_neighbors: int = 20
    parsimony_cluster_combined_threshold: float = 0.60
    parsimony_cluster_empirical_min_similarity: float = 0.30
    parsimony_cluster_strong_empirical_threshold: float = 0.80
    parsimony_cluster_missingness_weight: float = 0.15
    parsimony_cluster_min_size: int = 2
    parsimony_cluster_max_size: int = 12
    parsimony_cluster_sketch_dim: int = 32
    parsimony_max_factors_per_cluster: int = 2
    parsimony_factor_min_coverage: float = 0.10
    parsimony_parallelism: str = "auto"
    parsimony_metric_epsilon: float = 1e-6
    # Deprecated compatibility fields from the legacy single-feature ablation
    # implementation.  They remain parseable but no longer govern parsimony.
    parsimony_review_auc_tolerance: float = 0.01
    parsimony_review_loss_relative_tolerance: float = 0.03
    parsimony_review_corr_threshold: float = 0.75
    parsimony_review_max_single_feature_ablations: int = 30
    # Compatibility default: allow legacy full-data runs, but label them as
    # non-honest. Set true to require CV or an explicit held-out test split.
    require_honest_outer_split: bool = False
    # Final accepted values must come from complete-document reading. This guard
    # prevents the current LLM extractor from silently using only the note tail.
    fail_on_extraction_truncation: bool = True
    outer_parallelism: str = "1"
    bow_parallel_backend: str = "processes"
    # "auto" uses the runner num_workers setting; set a positive integer to
    # parallelize BoW nuisance/effect cross-fit folds explicitly.
    fold_parallelism: str = "auto"
    embedding_contrast: EmbeddingContrastDiscoveryConfig = field(
        default_factory=EmbeddingContrastDiscoveryConfig
    )
    htr_evidence_enabled: bool = True
    htr_evidence_disable_reason: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.embedding_contrast, dict):
            self.embedding_contrast = EmbeddingContrastDiscoveryConfig(**self.embedding_contrast)
        if self.feature_discovery_methods is not None:
            self.set_feature_discovery_methods(
                self.feature_discovery_methods,
                source="multi_model_agentic_forest.feature_discovery_methods",
            )
        else:
            self.feature_discovery_methods = self._feature_discovery_methods_from_flags()
        if not self.feature_discovery_methods:
            raise ValueError(
                "multi_model_agentic_forest must enable at least one feature "
                "discovery method: bow, htr, or embedding_contrast"
            )
        if (
            not bool(self.bow_discovery_enabled)
            and not str(self.bow_discovery_disable_reason or "").strip()
        ):
            self.bow_discovery_disable_reason = (
                "disabled by multi_model_agentic_forest.feature_discovery_methods"
            )
        if (
            not bool(self.htr_evidence_enabled)
            and not str(self.htr_evidence_disable_reason or "").strip()
        ):
            self.htr_evidence_disable_reason = (
                "disabled by multi_model_agentic_forest.feature_discovery_methods"
            )
        if self.bow_views:
            self.bow_views = [
                view if isinstance(view, BoWViewConfig) else BoWViewConfig(**view)
                for view in self.bow_views
            ]
        else:
            # Generic experiment loading retains the historical grid. The
            # portable production profile/factory rejects an empty configured
            # list before this compatibility branch can run.
            self.bow_views = legacy_default_bow_views_v1()
        seen_view_names = set()
        for idx, view in enumerate(self.bow_views, start=1):
            if not view.name:
                view.name = (
                    f"{view.bow_model}_{view.ngram_range_min}_" f"{view.ngram_range_max}_{idx}"
                )
            if view.name in seen_view_names:
                raise ValueError("multi_model_agentic_forest.bow_views names must be unique")
            seen_view_names.add(view.name)
        self.prespecified_features = parse_explicit_feature_spec_entries(
            self.prespecified_features,
            source="multi_model_agentic_forest.prespecified_features",
        )
        self.prespecified_confounders = parse_explicit_feature_spec_entries(
            self.prespecified_confounders,
            default_roles=["confounder"],
            source="multi_model_agentic_forest.prespecified_confounders",
        )
        self.prespecified_effect_modifiers = parse_explicit_feature_spec_entries(
            self.prespecified_effect_modifiers,
            default_roles=["effect_modifier"],
            source="multi_model_agentic_forest.prespecified_effect_modifiers",
        )
        if self.prespecified_features_json:
            if not Path(str(self.prespecified_features_json)).exists():
                raise ValueError(
                    "multi_model_agentic_forest.prespecified_features_json "
                    f"does not exist: {self.prespecified_features_json}"
                )
        if self.nuisance_folds < 2:
            raise ValueError("multi_model_agentic_forest.nuisance_folds must be >= 2")
        if self.effect_folds < 2:
            raise ValueError("multi_model_agentic_forest.effect_folds must be >= 2")
        if not 0.0 < self.e_clip < 0.5:
            raise ValueError("multi_model_agentic_forest.e_clip must be in (0, 0.5)")
        if self.top_n_features < 1:
            raise ValueError("multi_model_agentic_forest.top_n_features must be >= 1")
        if self.candidate_proposals_per_fold < 1:
            raise ValueError("multi_model_agentic_forest.candidate_proposals_per_fold must be >= 1")
        agent_context_mode = str(self.agent_context_mode or "").strip().lower()
        if agent_context_mode not in {"evidence_digest", "rich_context"}:
            raise ValueError(
                "multi_model_agentic_forest.agent_context_mode must be "
                "'evidence_digest' or 'rich_context'"
            )
        self.agent_context_mode = agent_context_mode
        if self.concept_inventory_max_concepts < 1:
            raise ValueError(
                "multi_model_agentic_forest.concept_inventory_max_concepts must be >= 1"
            )
        if self.candidate_consistency_inner_folds < 2:
            raise ValueError(
                "multi_model_agentic_forest.candidate_consistency_inner_folds must be >= 2"
            )
        if self.tfidf_nested_calibration_folds < 2:
            raise ValueError(
                "multi_model_agentic_forest.tfidf_nested_calibration_folds must be >= 2"
            )
        if self.candidate_consistency_min_folds < 1:
            raise ValueError(
                "multi_model_agentic_forest.candidate_consistency_min_folds must be >= 1"
            )
        if not 0.0 < self.candidate_consistency_min_fold_fraction <= 1.0:
            raise ValueError(
                "multi_model_agentic_forest.candidate_consistency_min_fold_fraction "
                "must be in (0, 1]"
            )
        if self.candidate_consistency_recovery_max_candidates < 0:
            raise ValueError(
                "multi_model_agentic_forest.candidate_consistency_recovery_max_candidates "
                "must be >= 0"
            )
        if self.extracted_feature_review_max_rounds < 0:
            raise ValueError(
                "multi_model_agentic_forest.extracted_feature_review_max_rounds " "must be >= 0"
            )
        if self.extracted_feature_review_auc_margin < 0.0:
            raise ValueError(
                "multi_model_agentic_forest.extracted_feature_review_auc_margin " "must be >= 0"
            )
        if self.extracted_feature_review_loss_relative_margin < 0.0:
            raise ValueError(
                "multi_model_agentic_forest.extracted_feature_review_loss_relative_margin "
                "must be >= 0"
            )
        if not 0.0 <= self.extracted_feature_review_min_benchmark_auc <= 1.0:
            raise ValueError(
                "multi_model_agentic_forest.extracted_feature_review_min_benchmark_auc "
                "must be in [0, 1]"
            )
        if self.parsimony_review_auc_tolerance < 0.0:
            raise ValueError(
                "multi_model_agentic_forest.parsimony_review_auc_tolerance " "must be >= 0"
            )
        if self.parsimony_review_loss_relative_tolerance < 0.0:
            raise ValueError(
                "multi_model_agentic_forest.parsimony_review_loss_relative_tolerance "
                "must be >= 0"
            )
        if not 0.0 <= self.parsimony_review_corr_threshold <= 1.0:
            raise ValueError(
                "multi_model_agentic_forest.parsimony_review_corr_threshold " "must be in [0, 1]"
            )
        if self.parsimony_review_max_single_feature_ablations < 0:
            raise ValueError(
                "multi_model_agentic_forest.parsimony_review_max_single_feature_ablations "
                "must be >= 0"
            )
        if not 0.0 <= self.parsimony_cluster_semantic_weight <= 1.0:
            raise ValueError(
                "multi_model_agentic_forest.parsimony_cluster_semantic_weight " "must be in [0, 1]"
            )
        if self.parsimony_cluster_neighbors < 1:
            raise ValueError("multi_model_agentic_forest.parsimony_cluster_neighbors must be >= 1")
        for field_name in [
            "parsimony_cluster_combined_threshold",
            "parsimony_cluster_empirical_min_similarity",
            "parsimony_cluster_strong_empirical_threshold",
            "parsimony_cluster_missingness_weight",
            "parsimony_factor_min_coverage",
        ]:
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"multi_model_agentic_forest.{field_name} must be in [0, 1]")
        if self.parsimony_cluster_min_size < 2:
            raise ValueError("multi_model_agentic_forest.parsimony_cluster_min_size must be >= 2")
        if self.parsimony_cluster_max_size < self.parsimony_cluster_min_size:
            raise ValueError(
                "multi_model_agentic_forest.parsimony_cluster_max_size must be >= "
                "parsimony_cluster_min_size"
            )
        if self.parsimony_cluster_sketch_dim < 1:
            raise ValueError("multi_model_agentic_forest.parsimony_cluster_sketch_dim must be >= 1")
        if not 1 <= self.parsimony_max_factors_per_cluster <= 2:
            raise ValueError(
                "multi_model_agentic_forest.parsimony_max_factors_per_cluster " "must be 1 or 2"
            )
        if self.parsimony_metric_epsilon < 0.0:
            raise ValueError("multi_model_agentic_forest.parsimony_metric_epsilon must be >= 0")
        _validate_parallelism_setting(
            self.candidate_consistency_parallelism,
            "multi_model_agentic_forest.candidate_consistency_parallelism",
        )
        _validate_parallelism_setting(
            self.parsimony_parallelism,
            "multi_model_agentic_forest.parsimony_parallelism",
        )
        _validate_parallelism_setting(
            self.outer_parallelism,
            "multi_model_agentic_forest.outer_parallelism",
        )
        bow_backend = str(self.bow_parallel_backend).strip().lower()
        if bow_backend not in {"processes", "threads", "loky"}:
            raise ValueError(
                "multi_model_agentic_forest.bow_parallel_backend must be "
                "'processes', 'loky', or 'threads'"
            )
        self.bow_parallel_backend = "processes" if bow_backend == "loky" else bow_backend
        if self.fold_parallelism != "auto":
            try:
                if int(self.fold_parallelism) < 1:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    "multi_model_agentic_forest.fold_parallelism must be 'auto' "
                    "or a positive integer"
                ) from exc

    def set_feature_discovery_methods(
        self,
        methods: Any,
        *,
        source: str = "feature_discovery_methods",
    ) -> None:
        normalized = normalize_multi_model_feature_discovery_methods(
            methods,
            source=source,
        )
        assert normalized is not None
        self.feature_discovery_methods = normalized
        self.bow_discovery_enabled = "bow" in normalized
        self.htr_evidence_enabled = "htr" in normalized
        self.embedding_contrast.enabled = "embedding_contrast" in normalized

        if self.bow_discovery_enabled:
            self.bow_discovery_disable_reason = None
        elif (
            not self.bow_discovery_enabled
            and not str(self.bow_discovery_disable_reason or "").strip()
        ):
            self.bow_discovery_disable_reason = f"disabled by {source}"

        if self.htr_evidence_enabled:
            self.htr_evidence_disable_reason = None
        elif (
            not self.htr_evidence_enabled
            and not str(self.htr_evidence_disable_reason or "").strip()
        ):
            self.htr_evidence_disable_reason = f"disabled by {source}"

        if self.embedding_contrast.enabled:
            self.embedding_contrast.disable_reason = None
        elif (
            not self.embedding_contrast.enabled
            and not str(self.embedding_contrast.disable_reason or "").strip()
        ):
            self.embedding_contrast.disable_reason = f"disabled by {source}"

    def _feature_discovery_methods_from_flags(self) -> List[str]:
        methods: List[str] = []
        if bool(self.bow_discovery_enabled):
            methods.append("bow")
        if bool(self.htr_evidence_enabled):
            methods.append("htr")
        if bool(getattr(self.embedding_contrast, "enabled", False)):
            methods.append("embedding_contrast")
        return methods


@dataclass
class MultiModelForestConfig(MultiModelAgenticForestConfig):
    """Stage 1 scientific configuration for the all-evidence workflow."""

    tfidf_topic: TfidfTopicDiscoveryConfig = field(default_factory=TfidfTopicDiscoveryConfig)
    # Optional audited outer/inner fold registry for exact-context Stage 1
    # artifact reproduction. The registry content, rather than its path, is
    # incorporated into the Stage 1 cache identity.
    split_registry_path: Optional[str] = None
    # Outer-fold execution backend for CPU-only TF-IDF/NMF contexts.
    outer_parallel_backend: str = "processes"
    # Optional overrides for the two nested fold families. When unset, the legacy
    # fold_parallelism setting is used for both.
    bow_fold_parallelism: Optional[str] = None
    htr_fold_parallelism: Optional[str] = None
    # Public scheduler controls for exact Stage 1 context fits.
    cpus_total: Optional[int] = None
    htr_jobs_per_gpu: int = 1
    # Matched-pair uplift evidence. Inner folds fit pair-level models on
    # observed treated/control matches from the outer-train fold; outer-test
    # patients are scored as candidate treated patients against similar
    # untreated outer-train controls. The model output is a delta logit added
    # to the matched untreated patient's outcome logit.
    matched_pair_uplift_enabled: bool = True
    matched_pair_bow_enabled: bool = True
    matched_pair_htr_enabled: bool = True
    matched_pair_propensity_caliper: float = 0.05
    matched_pair_outcome_caliper: float = 0.05
    matched_pair_max_controls_per_candidate: int = 3
    matched_pair_nearest_fallback_controls: int = 1
    matched_pair_bow_l2_alpha: float = 1.0
    matched_pair_bow_max_iter: int = 100
    matched_pair_bow_optimizer_method: str = "L-BFGS-B"
    matched_pair_bow_optimizer_ftol: float = 1e-8
    matched_pair_bow_optimizer_gtol: float = 1e-5
    matched_pair_bow_optimizer_maxls: int = 30
    matched_pair_bow_optimizer_maxcor: int = 10
    matched_pair_bow_optimizer_maxfun: int = 15_000
    matched_pair_bow_optimizer_tol: Optional[float] = None
    matched_pair_bow_optimizer_initialization: str = "zeros"
    matched_pair_bow_require_optimizer_success: bool = False
    matched_pair_htr_optimizer_name: str = "adamw"
    matched_pair_htr_adamw_beta1: float = 0.9
    matched_pair_htr_adamw_beta2: float = 0.999
    matched_pair_htr_adamw_eps: float = 1e-8
    matched_pair_htr_adamw_amsgrad: bool = False
    matched_pair_htr_adamw_maximize: bool = False
    matched_pair_htr_adamw_foreach: bool = False
    matched_pair_htr_adamw_capturable: bool = False
    matched_pair_htr_adamw_differentiable: bool = False
    matched_pair_htr_adamw_fused: bool = False
    matched_pair_htr_optimizer_zero_grad_set_to_none: bool = True
    matched_pair_htr_gradient_clip_norm: float = 0.0
    matched_pair_htr_gradient_clip_norm_type: float = 2.0
    matched_pair_htr_gradient_clip_error_if_nonfinite: bool = False
    matched_pair_htr_gradient_clip_foreach: bool = False
    matched_pair_htr_head_depth: int = 2
    matched_pair_htr_head_activation: str = "relu"
    matched_pair_htr_head_layer_norm: bool = False
    matched_pair_htr_head_bias: bool = True
    matched_pair_htr_attention_pairs_per_fold: int = 16
    # Final structured CATE head.  Keep the honest causal-forest path as the
    # production default; the interaction S-learner remains an explicit
    # diagnostic/ablation option.
    structured_effect_estimator: str = "causal_forest"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.tfidf_topic, dict):
            self.tfidf_topic = TfidfTopicDiscoveryConfig(**self.tfidf_topic)
        if self.split_registry_path is not None:
            registry_path = str(self.split_registry_path).strip()
            self.split_registry_path = registry_path or None
        backend = str(self.outer_parallel_backend).strip().lower()
        if backend not in {"threads", "processes", "loky", "multiprocessing", "fork"}:
            raise ValueError(
                "multi_model_forest.outer_parallel_backend must be "
                "'threads', 'processes', 'loky', 'multiprocessing', or 'fork'"
            )
        backend_aliases = {
            "loky": "processes",
            "fork": "multiprocessing",
        }
        self.outer_parallel_backend = backend_aliases.get(backend, backend)
        for field_name in ("bow_fold_parallelism", "htr_fold_parallelism"):
            value = getattr(self, field_name)
            if value is None:
                continue
            value_text = str(value).strip().lower()
            if not value_text:
                setattr(self, field_name, None)
                continue
            _validate_parallelism_setting(
                value_text,
                f"multi_model_forest.{field_name}",
            )
            setattr(self, field_name, value_text)
        if self.cpus_total is not None and int(self.cpus_total) < 1:
            raise ValueError("multi_model_forest.cpus_total must be >= 1")
        if int(self.htr_jobs_per_gpu) < 1:
            raise ValueError("multi_model_forest.htr_jobs_per_gpu must be >= 1")
        self.cpus_total = None if self.cpus_total is None else int(self.cpus_total)
        self.htr_jobs_per_gpu = int(self.htr_jobs_per_gpu)
        if not 0.0 < float(self.matched_pair_propensity_caliper) <= 1.0:
            raise ValueError("multi_model_forest.matched_pair_propensity_caliper must be in (0, 1]")
        if not 0.0 < float(self.matched_pair_outcome_caliper) <= 1.0:
            raise ValueError("multi_model_forest.matched_pair_outcome_caliper must be in (0, 1]")
        if int(self.matched_pair_max_controls_per_candidate) < 1:
            raise ValueError(
                "multi_model_forest.matched_pair_max_controls_per_candidate must be >= 1"
            )
        if int(self.matched_pair_nearest_fallback_controls) < 0:
            raise ValueError(
                "multi_model_forest.matched_pair_nearest_fallback_controls must be >= 0"
            )
        if float(self.matched_pair_bow_l2_alpha) < 0.0:
            raise ValueError("multi_model_forest.matched_pair_bow_l2_alpha must be >= 0")
        if int(self.matched_pair_bow_max_iter) < 1:
            raise ValueError("multi_model_forest.matched_pair_bow_max_iter must be >= 1")
        if self.matched_pair_bow_optimizer_method != "L-BFGS-B":
            raise ValueError(
                "multi_model_forest.matched_pair_bow_optimizer_method "
                "must be 'L-BFGS-B'"
            )
        if self.matched_pair_bow_optimizer_initialization != "zeros":
            raise ValueError(
                "multi_model_forest.matched_pair_bow_optimizer_initialization "
                "must be 'zeros'"
            )
        for field_name in (
            "matched_pair_bow_optimizer_ftol",
            "matched_pair_bow_optimizer_gtol",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"multi_model_forest.{field_name} must be finite and >= 0"
                )
            setattr(self, field_name, value)
        if self.matched_pair_bow_optimizer_tol is not None:
            tolerance = float(self.matched_pair_bow_optimizer_tol)
            if not math.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError(
                    "multi_model_forest.matched_pair_bow_optimizer_tol "
                    "must be null or finite and >= 0"
                )
            self.matched_pair_bow_optimizer_tol = tolerance
        for field_name in (
            "matched_pair_bow_optimizer_maxls",
            "matched_pair_bow_optimizer_maxcor",
            "matched_pair_bow_optimizer_maxfun",
            "matched_pair_htr_head_depth",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(
                    f"multi_model_forest.{field_name} must be a positive integer"
                )
            setattr(self, field_name, int(value))
        if self.matched_pair_htr_optimizer_name != "adamw":
            raise ValueError(
                "multi_model_forest.matched_pair_htr_optimizer_name "
                "must be 'adamw'"
            )
        for field_name in (
            "matched_pair_htr_adamw_beta1",
            "matched_pair_htr_adamw_beta2",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(
                    f"multi_model_forest.{field_name} must be in [0, 1)"
                )
            setattr(self, field_name, value)
        self.matched_pair_htr_adamw_eps = float(
            self.matched_pair_htr_adamw_eps
        )
        if (
            not math.isfinite(self.matched_pair_htr_adamw_eps)
            or self.matched_pair_htr_adamw_eps <= 0.0
        ):
            raise ValueError(
                "multi_model_forest.matched_pair_htr_adamw_eps "
                "must be finite and positive"
            )
        self.matched_pair_htr_gradient_clip_norm = float(
            self.matched_pair_htr_gradient_clip_norm
        )
        self.matched_pair_htr_gradient_clip_norm_type = float(
            self.matched_pair_htr_gradient_clip_norm_type
        )
        if (
            not math.isfinite(self.matched_pair_htr_gradient_clip_norm)
            or self.matched_pair_htr_gradient_clip_norm < 0.0
            or not math.isfinite(self.matched_pair_htr_gradient_clip_norm_type)
            or self.matched_pair_htr_gradient_clip_norm_type <= 0.0
        ):
            raise ValueError(
                "multi_model_forest matched-pair HTR gradient-clip "
                "configuration is invalid"
            )
        if self.matched_pair_htr_head_activation not in {
            "gelu_exact",
            "gelu_tanh",
            "relu",
            "silu",
            "tanh",
        }:
            raise ValueError(
                "multi_model_forest.matched_pair_htr_head_activation "
                "is unsupported"
            )
        for field_name in (
            "matched_pair_bow_require_optimizer_success",
            "matched_pair_htr_adamw_amsgrad",
            "matched_pair_htr_adamw_maximize",
            "matched_pair_htr_adamw_foreach",
            "matched_pair_htr_adamw_capturable",
            "matched_pair_htr_adamw_differentiable",
            "matched_pair_htr_adamw_fused",
            "matched_pair_htr_optimizer_zero_grad_set_to_none",
            "matched_pair_htr_gradient_clip_error_if_nonfinite",
            "matched_pair_htr_gradient_clip_foreach",
            "matched_pair_htr_head_layer_norm",
            "matched_pair_htr_head_bias",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(
                    f"multi_model_forest.{field_name} must be an exact boolean"
                )
        if int(self.matched_pair_htr_attention_pairs_per_fold) < 0:
            raise ValueError(
                "multi_model_forest.matched_pair_htr_attention_pairs_per_fold must be >= 0"
            )
        self.matched_pair_propensity_caliper = float(self.matched_pair_propensity_caliper)
        self.matched_pair_outcome_caliper = float(self.matched_pair_outcome_caliper)
        self.matched_pair_max_controls_per_candidate = int(
            self.matched_pair_max_controls_per_candidate
        )
        self.matched_pair_nearest_fallback_controls = int(
            self.matched_pair_nearest_fallback_controls
        )
        self.matched_pair_bow_l2_alpha = float(self.matched_pair_bow_l2_alpha)
        self.matched_pair_bow_max_iter = int(self.matched_pair_bow_max_iter)
        self.matched_pair_htr_attention_pairs_per_fold = int(
            self.matched_pair_htr_attention_pairs_per_fold
        )
        estimator = str(self.structured_effect_estimator).strip().lower().replace("-", "_")
        estimator_aliases = {
            "interaction": "interaction_s_learner",
            "s_learner": "interaction_s_learner",
            "interaction_s_learner": "interaction_s_learner",
            "causal_forest": "causal_forest",
            "forest": "causal_forest",
        }
        if estimator not in estimator_aliases:
            raise ValueError(
                "multi_model_forest.structured_effect_estimator must be "
                "'interaction_s_learner' or 'causal_forest'"
            )
        self.structured_effect_estimator = estimator_aliases[estimator]


@dataclass
class AgenticAttentionVariableForestConfig:
    """Configuration for attention-evidence agentic variable discovery.

    This model discovers explicit variables from cross-fitted neural attention
    evidence, extracts their values, then fits a programmatic causal forest.
    Agent and extraction endpoints are configured through
    ``architecture.agentic_feature_search`` and ``explicit_features``.
    """

    nuisance_folds: int = 5
    nuisance_epochs: Optional[int] = 30
    nuisance_weight_decay: Optional[float] = 0.01
    nuisance_label_smoothing: float = 0.02
    nuisance_calibration: str = "temperature_isotonic"
    effect_folds: int = 5
    effect_epochs: Optional[int] = 30
    # "auto" parallelizes folds on CPU via num_workers and across configured
    # CUDA devices when more than one device is supplied.
    fold_parallelism: str = "auto"
    # Outer analysis-fold parallelism. "auto" uses num_workers, capped to the
    # configured CUDA device count when more than one device is supplied.
    outer_parallelism: str = "1"
    attention_top_k_chunks: int = 5
    candidate_proposals_per_fold: int = 3
    # Parallelism for per-inner-fold agent candidate proposal calls. Defaults
    # to serial to avoid surprising endpoint concurrency.
    candidate_proposal_parallelism: str = "1"
    coverage_retry_attempts: int = 1
    signal_retry_attempts: int = 1
    association_alpha: float = 0.05
    association_min_n: int = 20
    association_min_non_missing: int = 10
    signal_cv_folds: int = 3
    min_signal_treatment_auroc: float = 0.55
    min_signal_outcome_auroc: float = 0.55
    consensus_min_folds: Optional[int] = 2
    consensus_min_fold_fraction: float = 2.0 / 3.0
    consensus_recovery_enabled: bool = True
    consensus_recovery_max_candidates: int = 12
    min_extraction_coverage: float = 0.10
    e_clip: float = 0.01
    r_stage_min_propensity: float = 0.0
    r_stage_max_propensity: float = 1.0
    effect_objective: str = "pseudo_outcome_mse"
    neural_stage_mode: str = "staged"
    joint_rlearner_gamma: float = 1.0
    interaction_l2_weight: float = 1e-3
    tarnet_offset_batch_size: Optional[int] = 128
    tarnet_offset_heterogeneity_weight: float = 0.1
    tarnet_offset_min_logit_std: float = 0.5
    residual_contrastive_enabled: bool = False
    residual_contrastive_use_for_effect_discovery: bool = True
    residual_contrastive_score: str = "r_score"
    residual_contrastive_high_quantile: float = 0.80
    residual_contrastive_low_quantile: float = 0.20
    residual_contrastive_neutral_abs_quantile: float = 0.40
    residual_contrastive_min_class_count: int = 10
    manual_features_locked: bool = True
    neural_only: bool = False

    def __post_init__(self):
        if self.nuisance_folds < 2:
            raise ValueError("agentic_attention_variable_forest.nuisance_folds must be >= 2")
        if self.nuisance_epochs is not None and self.nuisance_epochs < 1:
            raise ValueError(
                "agentic_attention_variable_forest.nuisance_epochs must be >= 1 when set"
            )
        if self.nuisance_weight_decay is not None and self.nuisance_weight_decay < 0:
            raise ValueError(
                "agentic_attention_variable_forest.nuisance_weight_decay must be >= 0 when set"
            )
        if not 0.0 <= float(self.nuisance_label_smoothing) < 1.0:
            raise ValueError(
                "agentic_attention_variable_forest.nuisance_label_smoothing must be in [0, 1)"
            )
        nuisance_calibration = str(self.nuisance_calibration).strip().lower()
        if nuisance_calibration not in {"none", "temperature", "isotonic", "temperature_isotonic"}:
            raise ValueError(
                "agentic_attention_variable_forest.nuisance_calibration must be one of "
                "'none', 'temperature', 'isotonic', or 'temperature_isotonic'"
            )
        self.nuisance_calibration = nuisance_calibration
        if self.effect_folds < 2:
            raise ValueError("agentic_attention_variable_forest.effect_folds must be >= 2")
        if self.effect_epochs is not None and self.effect_epochs < 1:
            raise ValueError(
                "agentic_attention_variable_forest.effect_epochs must be >= 1 when set"
            )
        _validate_parallelism_setting(
            self.candidate_proposal_parallelism,
            "agentic_attention_variable_forest.candidate_proposal_parallelism",
        )
        if self.fold_parallelism != "auto":
            try:
                if int(self.fold_parallelism) < 1:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    "agentic_attention_variable_forest.fold_parallelism must be 'auto' "
                    "or a positive integer"
                ) from exc
        if self.outer_parallelism != "auto":
            try:
                if int(self.outer_parallelism) < 1:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    "agentic_attention_variable_forest.outer_parallelism must be 'auto' "
                    "or a positive integer"
                ) from exc
        if self.attention_top_k_chunks < 1:
            raise ValueError(
                "agentic_attention_variable_forest.attention_top_k_chunks must be >= 1"
            )
        effect_objective = str(self.effect_objective).strip().lower()
        if effect_objective not in {
            "dina",
            "squared_r_loss",
            "logistic_r_loss",
            "pseudo_outcome_mse",
        }:
            raise ValueError(
                "agentic_attention_variable_forest.effect_objective must be one "
                "of 'squared_r_loss', 'logistic_r_loss', or 'pseudo_outcome_mse'"
            )
        self.effect_objective = effect_objective
        neural_stage_mode = str(self.neural_stage_mode).strip().lower()
        if neural_stage_mode not in {
            "staged",
            "joint_rlearner",
            "interaction_outcome",
            "tarnet_offset",
        }:
            raise ValueError(
                "agentic_attention_variable_forest.neural_stage_mode must be "
                "one of 'staged', 'joint_rlearner', 'interaction_outcome', "
                "or 'tarnet_offset'"
            )
        self.neural_stage_mode = neural_stage_mode
        self.joint_rlearner_gamma = float(self.joint_rlearner_gamma)
        if self.joint_rlearner_gamma < 0:
            raise ValueError("agentic_attention_variable_forest.joint_rlearner_gamma must be >= 0")
        self.interaction_l2_weight = float(self.interaction_l2_weight)
        if self.interaction_l2_weight < 0:
            raise ValueError("agentic_attention_variable_forest.interaction_l2_weight must be >= 0")
        if self.tarnet_offset_batch_size is not None:
            self.tarnet_offset_batch_size = int(self.tarnet_offset_batch_size)
            if self.tarnet_offset_batch_size < 1:
                raise ValueError(
                    "agentic_attention_variable_forest.tarnet_offset_batch_size "
                    "must be >= 1 when set"
                )
        self.tarnet_offset_heterogeneity_weight = float(self.tarnet_offset_heterogeneity_weight)
        if self.tarnet_offset_heterogeneity_weight < 0:
            raise ValueError(
                "agentic_attention_variable_forest."
                "tarnet_offset_heterogeneity_weight must be >= 0"
            )
        self.tarnet_offset_min_logit_std = float(self.tarnet_offset_min_logit_std)
        if self.tarnet_offset_min_logit_std < 0:
            raise ValueError(
                "agentic_attention_variable_forest.tarnet_offset_min_logit_std " "must be >= 0"
            )
        if self.candidate_proposals_per_fold < 1:
            raise ValueError(
                "agentic_attention_variable_forest.candidate_proposals_per_fold " "must be >= 1"
            )
        if self.coverage_retry_attempts < 0:
            raise ValueError(
                "agentic_attention_variable_forest.coverage_retry_attempts must be >= 0"
            )
        if self.signal_retry_attempts < 0:
            raise ValueError("agentic_attention_variable_forest.signal_retry_attempts must be >= 0")
        if not 0.0 < self.association_alpha < 1.0:
            raise ValueError(
                "agentic_attention_variable_forest.association_alpha must be in (0, 1)"
            )
        if self.association_min_n < 1:
            raise ValueError("agentic_attention_variable_forest.association_min_n must be >= 1")
        if self.association_min_non_missing < 1:
            raise ValueError(
                "agentic_attention_variable_forest.association_min_non_missing must be >= 1"
            )
        if self.signal_cv_folds < 2:
            raise ValueError("agentic_attention_variable_forest.signal_cv_folds must be >= 2")
        if not 0.5 <= self.min_signal_treatment_auroc <= 1.0:
            raise ValueError(
                "agentic_attention_variable_forest.min_signal_treatment_auroc "
                "must be in [0.5, 1]"
            )
        if not 0.5 <= self.min_signal_outcome_auroc <= 1.0:
            raise ValueError(
                "agentic_attention_variable_forest.min_signal_outcome_auroc " "must be in [0.5, 1]"
            )
        if self.consensus_min_folds is not None and self.consensus_min_folds < 1:
            raise ValueError(
                "agentic_attention_variable_forest.consensus_min_folds " "must be >= 1 when set"
            )
        if not 0.0 < self.consensus_min_fold_fraction <= 1.0:
            raise ValueError(
                "agentic_attention_variable_forest.consensus_min_fold_fraction " "must be in (0, 1]"
            )
        if self.consensus_recovery_max_candidates < 0:
            raise ValueError(
                "agentic_attention_variable_forest.consensus_recovery_max_candidates "
                "must be >= 0"
            )
        if not 0.0 <= self.min_extraction_coverage <= 1.0:
            raise ValueError(
                "agentic_attention_variable_forest.min_extraction_coverage must be in [0, 1]"
            )
        if not 0.0 < self.e_clip < 0.5:
            raise ValueError("agentic_attention_variable_forest.e_clip must be in (0, 0.5)")
        if not 0.0 <= self.r_stage_min_propensity < self.r_stage_max_propensity <= 1.0:
            raise ValueError(
                "agentic_attention_variable_forest r-stage propensity bounds "
                "must satisfy 0 <= min < max <= 1"
            )
        valid_scores = {"r_score", "r_score_normalized"}
        if self.residual_contrastive_score not in valid_scores:
            raise ValueError(
                "agentic_attention_variable_forest.residual_contrastive_score "
                f"must be one of {sorted(valid_scores)}"
            )
        if not (
            0.0
            < self.residual_contrastive_low_quantile
            < self.residual_contrastive_high_quantile
            < 1.0
        ):
            raise ValueError(
                "agentic_attention_variable_forest residual contrastive low/high "
                "quantiles must satisfy 0 < low < high < 1"
            )
        if not 0.0 < self.residual_contrastive_neutral_abs_quantile < 1.0:
            raise ValueError(
                "agentic_attention_variable_forest.residual_contrastive_neutral_abs_quantile "
                "must be in (0, 1)"
            )
        if self.residual_contrastive_min_class_count < 1:
            raise ValueError(
                "agentic_attention_variable_forest.residual_contrastive_min_class_count "
                "must be >= 1"
            )


@dataclass
class ModelArchitectureConfig:
    """Configuration shared by retained Stage 1 and explicit-feature workflows."""

    model_type: str = "multi_model_forest"
    feature_extractor_type: str = "hierarchical_transformer"

    # Hierarchical short-chunk transformer used by the HTR Stage 1 lane.
    htr_sentence_model: str = "prajjwal1/bert-tiny"
    htr_freeze_sentence_encoder: bool = False
    htr_chunk_size_words: int = 96
    htr_chunk_overlap_words: int = 24
    htr_max_chunks: int = 512
    htr_max_chunk_length: int = 128
    htr_num_layers: int = 2
    htr_num_heads: int = 4
    htr_transformer_dim: int = 256
    htr_dropout: float = 0.05
    htr_projection_dim: int = 128
    htr_hash_embedding_dim: int = 256
    htr_sentence_encoder_batch_size: int = 128
    htr_sentence_encoder_backend: str = "auto"
    htr_sentence_pooling: str = "auto"
    htr_normalize_sentence_embeddings: bool = True
    htr_trainable_sentence_encoder_layers: int = 0
    # Require a real trainable encoder whenever the sentence encoder is
    # configured as unfrozen. Lightweight hash-backed tests leave this false.
    htr_require_live_unfrozen_encoder: bool = False
    htr_role_attention: bool = False
    htr_w_attention_heads: int = 1
    htr_x_attention_heads: int = 1
    # Explicit HTR transformer/output topology used by Stage 1.
    htr_transformer_feedforward_dim: int = 1024
    htr_transformer_activation: str = "gelu_exact"
    htr_transformer_norm_style: str = "post_norm"
    htr_transformer_layer_norm_eps: float = 1e-5
    htr_transformer_layer_norm_elementwise_affine: bool = True
    htr_transformer_layer_norm_bias: bool = True
    htr_transformer_attention_dropout: float = 0.05
    htr_transformer_residual_dropout: float = 0.05
    htr_transformer_feedforward_dropout: float = 0.05
    htr_transformer_attention_bias: bool = True
    htr_transformer_feedforward_bias: bool = True
    htr_output_projection_depth: int = 1
    htr_output_projection_hidden_dim: int = 256
    htr_output_projection_activation: str = "gelu_exact"
    htr_output_projection_dropout: float = 0.05
    htr_output_projection_hidden_layer_norm: bool = True
    htr_output_projection_final_layer_norm: bool = True
    htr_output_projection_bias: bool = True
    htr_pool_token_init_std: float = 0.02
    htr_positional_encoding_base: float = 10_000.0
    htr_environment_override_policy: str = "legacy_allow"
    # Closed causal nuisance/effect heads used by the role-neutral producer.
    htr_nuisance_head_depth: int = 1
    htr_nuisance_head_activation: str = "relu"
    htr_nuisance_head_dropout: float = 0.1
    htr_nuisance_head_layer_norm: bool = False
    htr_nuisance_head_bias: bool = True
    htr_effect_head_depth: int = 1
    htr_effect_head_activation: str = "relu"
    htr_effect_head_dropout: float = 0.1
    htr_effect_head_layer_norm: bool = False
    htr_effect_head_bias: bool = True
    htr_prediction_head_hidden_dim: int = 64

    # Explicit-Feature-Only Causal Forest config (used when model_type="explicit_feature_forest")
    explicit_feature_forest: ExplicitFeatureForestConfig = field(
        default_factory=ExplicitFeatureForestConfig
    )

    # Agentic explicit feature search config (used when model_type="agentic_explicit_feature_forest")
    agentic_feature_search: AgenticFeatureSearchConfig = field(
        default_factory=AgenticFeatureSearchConfig
    )

    # Agentic attention-evidence variable discovery + explicit-feature causal forest
    agentic_attention_variable_forest: AgenticAttentionVariableForestConfig = field(
        default_factory=AgenticAttentionVariableForestConfig
    )

    # Multi-model BoW-guided variable discovery + explicit-feature causal forest
    multi_model_agentic_forest: MultiModelAgenticForestConfig = field(
        default_factory=MultiModelAgenticForestConfig
    )

    # All-evidence Stage 1 scientific configuration
    multi_model_forest: MultiModelForestConfig = field(default_factory=MultiModelForestConfig)


@dataclass
class TrainingConfig:
    """Shared optimization settings used by the retained HTR workflows."""

    learning_rate: float = 1e-4
    lr_schedule: str = "linear"
    epochs: int = 50
    batch_size: int = 8
    effect_batch_size: Optional[int] = 32
    dataloader_workers: Optional[int] = None  # None = 0 on CPU, 2 on accelerator devices
    alpha_propensity: float = 1.0
    weight_decay: float = 0.01  # L2 regularization (AdamW decoupled weight decay)
    gradient_clip_norm: float = 1.0  # Max gradient norm (0 to disable)


@dataclass
class AppliedInferenceConfig:
    """Configuration for applied inference on real data."""

    clinical_question: Optional[str] = None
    # The outer workflow/experiment must provide the scientific seed whenever
    # a stochastic applied-inference architecture is executed.  ``None`` is a
    # fail-closed sentinel, not a hidden fallback.
    seed: Optional[int] = None
    outcome_type: str = "binary"  # "binary" or "continuous"
    dataset_path: str = ""
    text_column: str = "clinical_text"
    outcome_column: str = "outcome_indicator"
    treatment_column: str = "treatment_indicator"
    split_column: str = "split"
    cv_folds: int = 5  # Number of CV folds (0 or 1 = fixed split)
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Explicit feature extraction configuration (LLM-based)
    explicit_features: ExplicitFeatureExtractionConfig = field(
        default_factory=ExplicitFeatureExtractionConfig
    )


@dataclass
class ExperimentConfig:
    """Main configuration for OCI experiments."""

    output_dir: str = "./oci_results"
    seed: int = 42
    device: Optional[str] = None
    num_workers: int = 1
    gpu_ids: Optional[List[int]] = None

    applied_inference: AppliedInferenceConfig = field(default_factory=AppliedInferenceConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        applied_data = data.get("applied_inference", {})
        if "explicit_confounders" in applied_data:
            raise ValueError(
                "Configuration key applied_inference.explicit_confounders has been removed. "
                "Use applied_inference.explicit_features.features with role-tagged "
                "ExplicitFeatureSpec entries instead."
            )

        def parse_architecture_config(arch_data: Dict[str, Any]) -> ModelArchitectureConfig:
            """Parse retained architectures while tolerating inert legacy fields."""
            arch_data = arch_data.copy()
            retired_model_types = {
                "causal_forest",
                "confounder_forest",
                "dragonnet",
                "dragonnet_drlearner",
                "non_neural_agentic_forest",
                "rlearner",
                "tfidf_forest",
            }
            if arch_data.get("model_type") in retired_model_types:
                raise ValueError(
                    f"model_type={arch_data.get('model_type')!r} has been removed; "
                    "use the all-evidence workflow or a retained explicit-feature model"
                )
            retired_keys = {
                "causal_forest",
                "dragonnet_drlearner",
                "tfidf_forest",
            }
            retired_prefixes = (
                "causal_head_",
                "ctcnn_",
                "flp_",
                "hcnn_",
                "hgru_",
                "hlm_",
                "scnn_",
                "svx_",
            )
            legacy_head_hidden_dim = arch_data.pop(
                "causal_head_hidden_outcome_dim",
                None,
            )
            if (
                legacy_head_hidden_dim is not None
                and "htr_prediction_head_hidden_dim" not in arch_data
            ):
                arch_data["htr_prediction_head_hidden_dim"] = legacy_head_hidden_dim
            for key in tuple(arch_data):
                if key in retired_keys or key.startswith(retired_prefixes):
                    arch_data.pop(key)
            if "explicit_feature_forest" in arch_data and isinstance(
                arch_data["explicit_feature_forest"], dict
            ):
                arch_data["explicit_feature_forest"] = ExplicitFeatureForestConfig(
                    **arch_data["explicit_feature_forest"]
                )
            if "agentic_feature_search" in arch_data and isinstance(
                arch_data["agentic_feature_search"], dict
            ):
                arch_data["agentic_feature_search"] = AgenticFeatureSearchConfig(
                    **arch_data["agentic_feature_search"]
                )
            if "agentic_attention_variable_forest" in arch_data and isinstance(
                arch_data["agentic_attention_variable_forest"], dict
            ):
                avf_data = arch_data["agentic_attention_variable_forest"].copy()
                if "inner_fold_parallelism" in avf_data and "fold_parallelism" not in avf_data:
                    avf_data["fold_parallelism"] = avf_data.pop("inner_fold_parallelism")
                else:
                    avf_data.pop("inner_fold_parallelism", None)
                arch_data["agentic_attention_variable_forest"] = (
                    AgenticAttentionVariableForestConfig(**avf_data)
                )
            if "multi_model_agentic_forest" in arch_data and isinstance(
                arch_data["multi_model_agentic_forest"], dict
            ):
                arch_data["multi_model_agentic_forest"] = MultiModelAgenticForestConfig(
                    **arch_data["multi_model_agentic_forest"]
                )
            if "multi_model_forest" in arch_data and isinstance(
                arch_data["multi_model_forest"], dict
            ):
                arch_data["multi_model_forest"] = MultiModelForestConfig(
                    **arch_data["multi_model_forest"]
                )
            return ModelArchitectureConfig(**arch_data)

        def parse_training_config(training_data: Dict[str, Any]) -> TrainingConfig:
            """Drop retired neural-model settings while preserving active HTR knobs."""

            training_data = training_data.copy()
            retired_training_keys = {
                "adamw_amsgrad",
                "adamw_beta1",
                "adamw_beta2",
                "adamw_capturable",
                "adamw_differentiable",
                "adamw_eps",
                "adamw_foreach",
                "adamw_fused",
                "adamw_maximize",
                "attention_entropy_weight",
                "beta_targreg",
                "gamma_rlearner",
                "gradient_clip_error_if_nonfinite",
                "gradient_clip_foreach",
                "gradient_clip_norm_type",
                "label_smoothing",
                "optimizer_zero_grad_set_to_none",
                "optimizer",
                "stop_grad_propensity",
            }
            for key in retired_training_keys:
                training_data.pop(key, None)
            return TrainingConfig(**training_data)

        def parse_explicit_features_config(
            feat_data: Dict[str, Any],
        ) -> ExplicitFeatureExtractionConfig:
            """Parse explicit features config, handling nested feature specs."""
            if not feat_data:
                return ExplicitFeatureExtractionConfig()
            feat_data = feat_data.copy()
            if "confounders" in feat_data:
                raise ValueError(
                    "explicit_features.confounders is not supported. "
                    "Use explicit_features.features and set roles on each feature."
                )
            if "features" in feat_data and isinstance(feat_data["features"], list):
                feat_data["features"] = [
                    ExplicitFeatureSpec(**f) if isinstance(f, dict) else f
                    for f in feat_data["features"]
                ]
            return ExplicitFeatureExtractionConfig(**feat_data)

        parsed_applied: Dict[str, Any] = {}
        for key, value in applied_data.items():
            if key in {"matching_analysis", "outcome_model", "propensity_trimming"}:
                continue
            if key == "architecture":
                parsed_applied[key] = parse_architecture_config(value)
            elif key == "training":
                parsed_applied[key] = parse_training_config(value)
            elif key == "explicit_features":
                parsed_applied[key] = parse_explicit_features_config(value)
            else:
                parsed_applied[key] = value
        applied = AppliedInferenceConfig(**parsed_applied)

        return cls(
            output_dir=data.get("output_dir", "./oci_results"),
            seed=data.get("seed", 42),
            device=data.get("device"),
            num_workers=data.get("num_workers", 1),
            gpu_ids=data.get("gpu_ids"),
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
            raise ValueError(
                f"applied_inference.outcome_type must be one of {valid_outcome_types}, "
                f"got '{self.applied_inference.outcome_type}'"
            )

        model_type = self.applied_inference.architecture.model_type
        retained_model_types = {
            "agentic_attention_variable_forest",
            "agentic_explicit_feature_forest",
            "explicit_feature_forest",
            "multi_model_agentic_forest",
            "multi_model_forest",
        }
        if model_type not in retained_model_types:
            raise ValueError(
                f"Unsupported or retired model_type={model_type!r}; retained types are "
                + ", ".join(sorted(retained_model_types))
            )
        if model_type == "multi_model_forest":
            raise ValueError(
                "model_type='multi_model_forest' is a Stage 1 all-evidence producer and "
                "cannot run through ExperimentRunner; use "
                "scripts/run_all_evidence.py (ResearchAllEvidenceWorkflow)"
            )
        if model_type == "agentic_attention_variable_forest":
            avf_config = self.applied_inference.architecture.agentic_attention_variable_forest
            if not (
                0.0 <= avf_config.r_stage_min_propensity < avf_config.r_stage_max_propensity <= 1.0
            ):
                raise ValueError(
                    "agentic_attention_variable_forest R-stage propensity bounds "
                    "must satisfy 0 <= min < max <= 1"
                )
            if str(avf_config.fold_parallelism).strip().lower() != "auto":
                try:
                    if int(avf_config.fold_parallelism) < 1:
                        raise ValueError
                except ValueError as exc:
                    raise ValueError(
                        "agentic_attention_variable_forest.fold_parallelism must "
                        "be 'auto' or a positive integer"
                    ) from exc
        if model_type == "multi_model_agentic_forest":
            mm_config = self.applied_inference.architecture.multi_model_agentic_forest
            methods = normalize_multi_model_feature_discovery_methods(
                getattr(mm_config, "feature_discovery_methods", None),
                source="multi_model_agentic_forest.feature_discovery_methods",
            )
            if methods is None:
                methods = mm_config._feature_discovery_methods_from_flags()
            if not methods:
                raise ValueError(
                    "multi_model_agentic_forest must enable at least one feature "
                    "discovery method: bow, htr, or embedding_contrast"
                )
            embedding_config = mm_config.embedding_contrast
            if (
                not bool(getattr(embedding_config, "enabled", False))
                and not str(getattr(embedding_config, "disable_reason", "") or "").strip()
            ):
                raise ValueError(
                    "multi_model_agentic_forest.embedding_contrast.enabled=False "
                    "requires embedding_contrast.disable_reason"
                )
            if (
                not bool(getattr(mm_config, "htr_evidence_enabled", True))
                and not str(getattr(mm_config, "htr_evidence_disable_reason", "") or "").strip()
            ):
                raise ValueError(
                    "multi_model_agentic_forest.htr_evidence_enabled=False "
                    "requires htr_evidence_disable_reason"
                )
        if (
            self.applied_inference.explicit_features.enabled
            and not self.applied_inference.explicit_features.features
            and self.applied_inference.architecture.model_type
            not in {
                "agentic_explicit_feature_forest",
                "agentic_attention_variable_forest",
                "multi_model_agentic_forest",
                "multi_model_forest",
            }
        ):
            raise ValueError(
                "applied_inference.explicit_features.enabled=True requires at least one "
                "role-tagged explicit feature in explicit_features.features"
            )

def create_default_config(output_path: str) -> None:
    """Create a default configuration file."""
    config = ExperimentConfig(
        output_dir="./oci_results",
        seed=42,
        # Device placement is deployment state.  Leave it unresolved so this
        # generated configuration is valid on CPU-only, single-GPU, and
        # arbitrary multi-GPU hosts without prescribing a physical device ID.
        device=None,
        num_workers=1,
        gpu_ids=None,
        applied_inference=AppliedInferenceConfig(
            dataset_path="./dataset.parquet",
            cv_folds=5,
            architecture=ModelArchitectureConfig(
                model_type="agentic_explicit_feature_forest",
                feature_extractor_type="hierarchical_transformer",
            ),
            training=TrainingConfig(epochs=50, batch_size=8),
        ),
    )

    config.to_json(output_path)
    print(f"Default configuration saved to: {output_path}")
