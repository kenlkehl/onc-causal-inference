"""Plain, resumable Stage 2 analysis of the researcher Stage 1 handoff.

This module intentionally treats a directory as the checkpoint.  It reads the
ordinary JSONL handoff, defines and extracts patient-level variables, reviews
small-model extraction aggregates with a separate primary model, selects roles
with fold-local group elastic nets and an R-learner, and produces causal-forest
estimates. It
has no bundle format, artifact authentication, immutable
request, content hashes, or checkpoint adoption.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .nuisance_diagnostics import (
    calibration_diagnostics,
    overlap_diagnostics,
    validate_propensity_bounds,
)

from .plain_handoff_stage2_evidence import (
    EVIDENCE_COMPILER_VERSION,
    SUPPORTED_STAGE2_ARCHITECTURES,
    compile_stage2_handoff_evidence,
    stage1_embedding_cache_dependency_identity,
)
from .stage1_architectures import DEFAULT_STAGE1_ARCHITECTURES
from .plain_handoff_stage2_analysis import (
    EXTRACTION_FIELD_REPAIR_LIMIT,
    _ExtractionCategoryError,
    _ExtractionFieldError,
    _ExtractionValueError,
    _PageObservationValidationError,
    Stage2RequestExhaustedError,
    Stage2ResponseValidationError,
    _token_id_count,
    prompt_token_count,
    run_fold_analysis,
)
from .stage2_elastic_net_selection import (
    Stage2ElasticNetSelectionConfig,
    statistical_selection_config_from_mapping,
)
from .stage2_role_adjudication import (
    Stage2RoleAdjudicationConfig,
    role_adjudication_config_from_mapping,
)
from .stage2_sampling import (
    SAMPLING_FIELDS, recommended_sampling, default_reasoning,
    is_qwen_flash_next, sampling_provenance,
)
from .stage2_endpoint_pool import ExtractionEndpoint, EndpointPool, endpoints_from_mapping
from .stage2_discovery_prompt import DISCOVERY_PROMPT_VERSION, DISCOVERY_SYSTEM_PROMPT
from .stage2_candidate_consolidation import (
    MIXED_SCHEMA, CandidateConsolidationPolicy, CandidateEmbeddingCache,
    policy_from_mapping as consolidation_policy_from_mapping,
    mixed_batches, diminishing_returns,
)
from . import stage2_clinical_prompts as clinical_prompts
from . import stage2_request_audit as request_audit
from .stage2_sequential_consolidation import (
    Stage2SequentialConsolidationConfig,
    sequential_consolidation_config_from_mapping,
)
from .stage2_estimand_ontology_config import (
    EstimandOntologyConfig,
    estimand_ontology_config_from_mapping,
)
from .vllm_server_pool import (
    ManagedVLLMConfig,
    launch_managed_vllm_servers,
    managed_argument,
    managed_vllm_config_from_mapping,
    validate_managed_vllm_pool_isolation,
)

LOGGER = logging.getLogger(__name__)
# Checkpoint reads share a smaller filesystem budget across all outer folds.
# LLM request concurrency is controlled independently.
_CHECKPOINT_READ_SLOTS = threading.BoundedSemaphore(16)

ALLOWED_VALUE_TYPES = {"binary", "categorical", "continuous", "ordinal", "ambiguous"}
ALLOWED_EVIDENCE_AXES = {
    "treatment",
    "outcome",
    "residual_effect",
    "matched_pair",
    "semantic",
    "unclear",
}
ALLOWED_ROLES = {"confounder", "effect_modifier"}
DEFAULT_MAX_RESPONSE_REPAIRS = 15
DEFAULT_THINKING_AFTER_RESPONSE_REPAIRS = 5
DEFAULT_REQUEST_TIMEOUT = 240 * 60.0
DEFAULT_REQUEST_ATTEMPT_TIMEOUT = 60 * 60.0
DEFAULT_TRANSPORT_MAX_ATTEMPTS = 6
THINKING_RESPONSE_REPAIR_EFFORT = "high"
# This is an output ceiling, not a requested output length. Models still stop
# normally at EOS as soon as the validated JSON object is complete.
MINIMUM_MAX_TOKENS = 100_000
DEFAULT_MAX_TOKENS = MINIMUM_MAX_TOKENS
# Extraction responses are bounded JSON records, so the output ceiling may be
# lowered independently when a model fails to emit EOS.  Keep the default
# generous, but permit a 4K safety cap without invalidating completed
# scientific checkpoints (this transport-only option is excluded from their
# fingerprints below).
MINIMUM_EXTRACTION_MAX_TOKENS = 4_096
DEFAULT_EXTRACTION_MAX_TOKENS = 75_000
DEFAULT_INTERPRETATION_REASONING_EFFORT = "auto"
DEFAULT_EXTRACTION_REASONING_EFFORT = "none"
STAGE2_REQUEST_KINDS = frozenset({"interpretation", "extraction"})
MANAGED_MODEL_PHASE_SCHEMA_VERSION = "stage2_managed_model_phase_v1"
DEFAULT_VLLM_RAPID_SWITCH_SECONDS = 15 * 60.0
MANAGED_VLLM_ALLOCATION_MODES = frozenset({"all_gpus", "configured_split"})
SUPPORTED_REASONING_EFFORTS = frozenset(
    {"auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
DEFAULT_EXTRACTION_MAX_PROMPT_CHARS = 640_000
DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE = 10
DEFAULT_EXTRACTION_CHUNK_SIZE_TOKENS = 50_000
DEFAULT_EXTRACTION_CONTEXT_WINDOW_TOKENS = 131_072
DEFAULT_EXTRACTION_CONTEXT_MARGIN_TOKENS = 1_024
DEFAULT_CONSOLIDATION_MAX_PROMPT_CHARS = 640_000
DEFAULT_OPERATIONALIZATION_MAX_PROMPT_CHARS = 640_000
DEFAULT_CONSOLIDATION_BATCH_SIZE = 20
DEFAULT_CONSOLIDATION_MAX_ROUNDS = 55
DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS = 3
DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS = 2
DEFAULT_EXTRACTION_VLLM_BASE_PORT = 8110
DEFAULT_EXTRACTION_VLLM_INTERNAL_PORT_BASE = 40_000
MODEL_IDENTITY_SCHEMA_VERSION = "stage2_endpoint_model_identity_v1"
FEATURE_DEFINITION_INPUT_SCHEMA_VERSION = (
    "stage2_feature_definition_inputs_v2_primary_model_only"
)
_CANDIDATE_SELECTION_ENCODING_LOCK = threading.Lock()
CONSOLIDATION_SCHEMA_VERSION = "global_candidate_pool_v16_exhaustive_cards"
EXTRACTION_ONTOLOGY_FEEDBACK_SCHEMA_VERSION = (
    "training_failure_ontology_refinement_v1_explicit_feature_invariants"
)
OPERATIONALIZATION_SCHEMA_VERSION = (
    "feature_name_bounded_supporting_text_v7_conflict_resolution"
)
INTERPRETATION_SCHEMA_VERSION = DISCOVERY_PROMPT_VERSION
INTERPRETATION_AUDIT_SCHEMA_VERSION = (
    "rejected_packet_text_only_ordinals_v7_exhaustive_decomposition"
)
CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE = "configured_explicit_feature"

def _canonical_evidence_axes(value: Any) -> list[str]:
    raw_axes = [value] if isinstance(value, str) else list(value or [])
    aliases = {
        "assignment": ("treatment",),
        "confounder": ("treatment", "outcome"),
        "effect": ("residual_effect",),
        "effect_modifier": ("residual_effect",),
        "heterogeneity": ("residual_effect",),
        "interaction": ("residual_effect",),
        "matched": ("matched_pair",),
        "propensity": ("treatment",),
        "prognostic": ("outcome",),
        "r_loss": ("residual_effect",),
        "unknown": ("unclear",),
    }
    canonical: set[str] = set()
    for raw_axis in raw_axes:
        tokens = re.split(r"[,;|/]", str(raw_axis))
        for token in tokens:
            normalized = re.sub(r"[^a-z0-9]+", "_", token.strip().lower()).strip("_")
            if normalized in ALLOWED_EVIDENCE_AXES:
                canonical.add(normalized)
            else:
                canonical.update(aliases.get(normalized, ()))
    return sorted(canonical or {"unclear"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _value_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_ite_correlation(
    truth: Sequence[float],
    estimate: Sequence[float],
    *,
    rank: bool,
) -> float | None:
    left = np.asarray(truth, dtype=float)
    right = np.asarray(estimate, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 2:
        return None
    left = left[finite]
    right = right[finite]
    if rank:
        left = pd.Series(left).rank(method="average").to_numpy(dtype=float)
        right = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if float(np.std(left)) <= 0.0 or float(np.std(right)) <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _stage2_oracle_ite_column(columns: Iterable[Any], outcome_type: str) -> str | None:
    """Pick the oracle ITE column on the analysis outcome scale.

    ``true_ite`` is on the outcome scale for both outcome types; legacy
    ``true_ite_prob`` is the outcome scale only for binary outcomes.  This is
    the same contract as ``applied.resolve_oracle_ite_column``.
    """

    names = {str(column) for column in columns}
    if "true_ite" in names:
        return "true_ite"
    if "true_ite_prob" in names and str(outcome_type).lower() == "binary":
        return "true_ite_prob"
    return None


def _evaluate_stage2_oracle_ite(
    *,
    prediction_path: Path,
    dataset: pd.DataFrame,
    output_dir: Path,
    outcome_type: str = "binary",
    oracle_ite_column: str | None = None,
) -> dict[str, Any]:
    """Evaluate frozen cross-fitted ITEs without exposing truth to modeling."""

    unavailable_reason: str | None = None
    if oracle_ite_column is None:
        oracle_ite_column = _stage2_oracle_ite_column(dataset.columns, outcome_type)
        if oracle_ite_column is None and "true_ite_prob" in map(str, dataset.columns):
            unavailable_reason = (
                "continuous-outcome dataset lacks outcome-scale true_ite; "
                "probability-scale true_ite_prob is not comparable"
            )
    prediction_path = Path(prediction_path)
    output_dir = Path(output_dir)
    metrics_path = output_dir / "posthoc_oracle_ite_metrics.json"
    frozen_sha256 = _file_sha256(prediction_path)
    base: dict[str, Any] = {
        "schema_version": "stage2_posthoc_oracle_ite_v1",
        "available": False,
        "evaluation_is_post_hoc": True,
        "all_outer_predictions_frozen_before_oracle_join": True,
        "oracle_columns_consumed_by_modeling": False,
        "oracle_ite_column": oracle_ite_column,
        "estimated_ite_column": "estimated_cate",
        "frozen_prediction_path": str(prediction_path),
        "frozen_prediction_sha256": frozen_sha256,
        "metrics_path": str(metrics_path),
    }
    if oracle_ite_column is None or oracle_ite_column not in dataset.columns:
        payload = {
            **base,
            "reason": unavailable_reason or (
                f"dataset does not contain {oracle_ite_column or 'true_ite'!r}"
            ),
        }
        _write_json(metrics_path, payload)
        return payload

    predictions = pd.read_csv(prediction_path)
    if any(str(column).startswith("true_") for column in predictions.columns):
        raise RuntimeError("frozen Stage 2 predictions contain an oracle column")
    required = {"_oci_row_id", "outer_fold", "estimated_cate"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"frozen Stage 2 predictions lack required columns: {missing}")

    oracle_values = pd.to_numeric(dataset[oracle_ite_column], errors="coerce")
    if oracle_values.isna().any() or not np.isfinite(oracle_values.to_numpy(dtype=float)).all():
        payload = {
            **base,
            "reason": f"dataset column {oracle_ite_column!r} is not complete and finite",
        }
        _write_json(metrics_path, payload)
        return payload
    oracle = pd.DataFrame(
        {
            "_oci_row_id": np.arange(len(dataset), dtype=int),
            oracle_ite_column: oracle_values.to_numpy(dtype=float),
        }
    )
    evaluated = predictions.merge(
        oracle,
        on="_oci_row_id",
        how="left",
        validate="one_to_one",
    )
    if len(evaluated) != len(dataset) or evaluated[oracle_ite_column].isna().any():
        raise ValueError("oracle ITE join did not cover every frozen Stage 2 prediction")

    def metrics_for(frame: pd.DataFrame) -> dict[str, Any]:
        truth = frame[oracle_ite_column].to_numpy(dtype=float)
        estimate = pd.to_numeric(frame["estimated_cate"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(truth) & np.isfinite(estimate)
        truth = truth[finite]
        estimate = estimate[finite]
        if not len(truth):
            return {
                "n": int(len(frame)),
                "finite_pairs": 0,
                "pearson_correlation": None,
                "spearman_correlation": None,
                "mae": None,
                "rmse": None,
                "mean_error": None,
                "mean_estimated_ite": None,
                "oracle_ate": None,
                "ate_bias": None,
                "estimated_ite_standard_deviation": None,
                "oracle_ite_standard_deviation": None,
            }
        error = estimate - truth
        return {
            "n": int(len(frame)),
            "finite_pairs": int(len(truth)),
            "pearson_correlation": _safe_ite_correlation(truth, estimate, rank=False),
            "spearman_correlation": _safe_ite_correlation(truth, estimate, rank=True),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mean_error": float(np.mean(error)),
            "mean_estimated_ite": float(np.mean(estimate)),
            "oracle_ate": float(np.mean(truth)),
            "ate_bias": float(np.mean(estimate) - np.mean(truth)),
            "estimated_ite_standard_deviation": float(np.std(estimate)),
            "oracle_ite_standard_deviation": float(np.std(truth)),
        }

    evaluated_path = output_dir / "posthoc_predictions_with_oracle_ite.csv"
    temporary = evaluated_path.with_name(f".{evaluated_path.name}.{os.getpid()}.tmp")
    evaluated.to_csv(temporary, index=False)
    os.replace(temporary, evaluated_path)
    payload = {
        **base,
        "available": True,
        "predictions_with_oracle_path": str(evaluated_path),
        "overall": metrics_for(evaluated),
        "per_fold": [
            {"outer_fold": int(fold), **metrics_for(frame)}
            for fold, frame in evaluated.groupby("outer_fold", sort=True)
        ],
    }
    _write_json(metrics_path, payload)
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            yield dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _read_checkpoint_jsons(paths: Sequence[Path]) -> list[Any] | None:
    """Read a cache entry with bounded retries for transient mount denials.

    Missing or malformed files make an entry incomplete. Access failures must
    propagate after retries, rather than causing expensive inference to repeat.
    """
    for attempt in range(3):
        try:
            with _CHECKPOINT_READ_SLOTS:
                return [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        except PermissionError:
            if attempt == 2:
                raise
            delay = 0.25 * (2 ** attempt)
            LOGGER.warning("Checkpoint read denied; retrying attempt %s/3 after %.2fs: %s",
                           attempt + 2, delay, paths[0].parent)
            time.sleep(delay)
        except (FileNotFoundError, json.JSONDecodeError):
            return None


@dataclass(frozen=True)
class Stage2ExplicitFeature:
    """Investigator-specified feature and complete extraction ontology."""

    name: str
    description: str
    value_type: str
    categories_or_unit: tuple[str, ...]
    measurement_definition: str
    missing_value_rule: str
    roles: tuple[str, ...]
    stability_summary: str = ""
    caveats: str = ""
    conflict_resolution: Mapping[str, Any] | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("stage2 explicit feature name must be a string")
        name = re.sub(r"[^a-z0-9]+", "_", str(self.name).strip().lower()).strip("_")
        if not name:
            raise ValueError("stage2 explicit feature name must contain letters or numbers")
        description = str(self.description or "").strip()
        if not description:
            raise ValueError(f"stage2 explicit feature {name!r} requires a nonempty description")
        value_type = str(self.value_type or "").strip().lower()
        value_type = {
            "bool": "binary",
            "boolean": "binary",
            "category": "categorical",
            "numeric": "continuous",
            "number": "continuous",
            "unknown": "ambiguous",
        }.get(value_type, value_type)
        if value_type not in ALLOWED_VALUE_TYPES:
            raise ValueError(
                f"stage2 explicit feature {name!r} value_type must be one of "
                f"{sorted(ALLOWED_VALUE_TYPES)}"
            )
        raw_categories: Any = self.categories_or_unit
        if isinstance(raw_categories, str):
            raw_categories = [raw_categories]
        if not isinstance(raw_categories, Sequence) or isinstance(
            raw_categories, (bytes, bytearray)
        ):
            raise ValueError(f"stage2 explicit feature {name!r} categories_or_unit must be a list")
        categories = [str(item).strip() for item in raw_categories if str(item).strip()]
        if value_type in {"binary", "categorical", "ordinal"}:
            from .plain_handoff_stage2_analysis import _validated_closed_category_values

            categories = _validated_closed_category_values(
                value_type=value_type,
                values=categories,
                source=f"stage2 explicit feature {name!r}",
            )
        measurement_definition = str(self.measurement_definition or "").strip()
        if not measurement_definition:
            raise ValueError(f"stage2 explicit feature {name!r} requires measurement_definition")
        missing_value_rule = str(self.missing_value_rule or "").strip()
        if not missing_value_rule:
            raise ValueError(f"stage2 explicit feature {name!r} requires missing_value_rule")
        raw_roles: Any = self.roles
        if isinstance(raw_roles, str):
            raw_roles = [raw_roles]
        if not isinstance(raw_roles, Sequence) or isinstance(raw_roles, (bytes, bytearray)):
            raise ValueError(f"stage2 explicit feature {name!r} roles must be a list")
        roles: list[str] = []
        for raw_role in raw_roles:
            role = str(raw_role).strip().lower()
            if role == "both":
                roles.extend(["confounder", "effect_modifier"])
            elif role:
                roles.append(role)
        roles = list(dict.fromkeys(roles))
        if not roles:
            raise ValueError(f"stage2 explicit feature {name!r} requires at least one causal role")
        unsupported_roles = sorted(set(roles) - ALLOWED_ROLES)
        if unsupported_roles:
            raise ValueError(
                f"stage2 explicit feature {name!r} contains unsupported roles: "
                f"{unsupported_roles}; allowed roles are {sorted(ALLOWED_ROLES)}"
            )
        from .plain_handoff_stage2_analysis import _resolved_conflict_resolution

        resolved_conflict = _resolved_conflict_resolution(
            {
                "name": name,
                "description": description,
                "value_type": value_type,
                "categories_or_unit": categories,
                "measurement_definition": measurement_definition,
                "missing_value_rule": missing_value_rule,
                "conflict_resolution": self.conflict_resolution,
            }
        )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "categories_or_unit", tuple(categories))
        object.__setattr__(self, "measurement_definition", measurement_definition)
        object.__setattr__(self, "missing_value_rule", missing_value_rule)
        object.__setattr__(self, "roles", tuple(roles))
        object.__setattr__(
            self,
            "conflict_resolution",
            {
                "strategy": resolved_conflict["strategy"],
                "positive_category": resolved_conflict["positive_category"],
            },
        )
        object.__setattr__(self, "stability_summary", str(self.stability_summary or "").strip())
        object.__setattr__(self, "caveats", str(self.caveats or "").strip())

    def as_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "value_type": self.value_type,
            "categories_or_unit": list(self.categories_or_unit),
            "measurement_definition": self.measurement_definition,
            "missing_value_rule": self.missing_value_rule,
            "conflict_resolution": dict(self.conflict_resolution or {}),
            "roles": list(self.roles),
            "stability_summary": self.stability_summary,
            "caveats": self.caveats,
        }


def _stage2_explicit_feature_from_mapping(
    raw: Any,
    *,
    source: str,
) -> Stage2ExplicitFeature:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source} must be an object containing a feature ontology")
    entry = dict(raw)
    nested = entry.pop("ontology", None)
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError(f"{source}.ontology must be an object")
        combined = dict(nested)
        combined.update(entry)
    else:
        combined = entry
    _reject_unknown_stage2_keys(
        combined, Stage2ExplicitFeature, section=source, extra=("type", "categories", "unit"),
    )

    def required_value(key: str, *aliases: str) -> Any:
        for candidate in (key, *aliases):
            if candidate in combined:
                return combined[candidate]
        alias_text = f" (or {', '.join(aliases)})" if aliases else ""
        raise ValueError(f"{source} requires ontology field {key}{alias_text}")

    raw_categories = required_value("categories_or_unit", "categories", "unit")
    if raw_categories is None:
        raw_categories = []
    elif isinstance(raw_categories, (str, int, float, bool)):
        raw_categories = [raw_categories]
    elif not isinstance(raw_categories, Sequence):
        raise ValueError(f"{source}.categories_or_unit must be a list or unit string")
    raw_roles = required_value("roles")
    if isinstance(raw_roles, str):
        raw_roles = [raw_roles]
    elif not isinstance(raw_roles, Sequence):
        raise ValueError(f"{source}.roles must be a list")

    return Stage2ExplicitFeature(
        name=required_value("name"),
        description=required_value("description"),
        value_type=required_value("value_type", "type"),
        categories_or_unit=tuple(item for item in raw_categories if item is not None),
        measurement_definition=required_value("measurement_definition"),
        missing_value_rule=required_value("missing_value_rule"),
        roles=tuple(role for role in raw_roles if role is not None),
        conflict_resolution=combined.get("conflict_resolution"),
        stability_summary=str(combined.get("stability_summary") or ""),
        caveats=str(combined.get("caveats") or ""),
    )


def _stage2_explicit_features_from_value(raw: Any) -> tuple[Stage2ExplicitFeature, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        unknown = sorted(set(map(str, raw)) - {"enabled", "features"})
        if unknown:
            raise ValueError(
                f"stage2.explicit_features contains unsupported fields: {unknown}. "
                "Use {\"enabled\": ..., \"features\": [...]} or a list of features."
            )
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("stage2.explicit_features.enabled must be true or false")
        entries = raw.get("features")
        if entries is None:
            raise ValueError("stage2.explicit_features must contain a features list")
        if not enabled:
            if entries:
                raise ValueError(
                    "stage2.explicit_features cannot contain features when enabled=false"
                )
            return ()
    else:
        entries = raw
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise ValueError("stage2.explicit_features must be a list of feature ontologies")
    features = tuple(
        _stage2_explicit_feature_from_mapping(
            entry,
            source=f"stage2.explicit_features[{index}]",
        )
        for index, entry in enumerate(entries)
    )
    names = [feature.name for feature in features]
    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate_names:
        raise ValueError(
            "stage2.explicit_features contains duplicate normalized feature names: "
            f"{duplicate_names}"
        )
    return features


@dataclass(frozen=True)
class Stage2ExtractionLLMConfig:
    """Small-model transport used only for patient value extraction."""

    endpoint: str = ""
    model: str = ""
    api_key: str = "EMPTY"
    workers: int = 4
    vllm: ManagedVLLMConfig | None = None
    # External servers already started by the user. All serve the same model.
    endpoints: tuple[ExtractionEndpoint, ...] = ()
    # Populated only while an extractor pool owned by this pipeline is alive.
    runtime_endpoints: tuple[str, ...] = ()
    # Optional continuation route for extracting cache misses with an explicitly
    # authorized replacement model.  The configured model above remains the
    # scientific identity of frozen extraction checkpoints; newly generated
    # request artifacts record this runtime model instead.
    runtime_endpoint: str = ""
    runtime_model: str = ""
    runtime_api_key: str = "EMPTY"

    def validate(self, *, require_model: bool = True) -> None:
        if self.endpoints:
            if self.endpoint or self.vllm is not None or self.runtime_endpoints or self.runtime_endpoint:
                raise ValueError(
                    "configure stage2.extraction_llm.endpoints independently of endpoint, "
                    "vllm, and runtime continuation routes"
                )
            for server in self.endpoints:
                if not isinstance(server, ExtractionEndpoint):
                    raise ValueError("stage2.extraction_llm.endpoints requires ExtractionEndpoint objects")
                server.validate()
            if len({server.endpoint for server in self.endpoints}) != len(self.endpoints):
                raise ValueError("stage2.extraction_llm.endpoints contains duplicate URLs")
        if self.endpoint and self.vllm is not None and not self.runtime_endpoints:
            raise ValueError(
                "configure either stage2.extraction_llm.endpoint or "
                "stage2.extraction_llm.vllm, not both"
            )
        if self.endpoint:
            parsed = urlparse(self.endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(
                    "stage2.extraction_llm.endpoint must be one HTTP(S) "
                    "OpenAI-compatible base URL"
                )
        elif self.vllm is None and not self.endpoints:
            raise ValueError(
                "stage2.extraction_llm requires endpoint, endpoints, or vllm"
            )
        if self.vllm is not None:
            self.vllm.validate()
            if not self.model.strip():
                raise ValueError(
                    "stage2.extraction_llm.model is required when its vllm pool is configured"
                )
        for runtime_endpoint in self.runtime_endpoints:
            parsed = urlparse(runtime_endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(
                    "stage2.extraction_llm.runtime_endpoints must contain HTTP(S) "
                    "OpenAI-compatible base URLs"
                )
        continuation_endpoint = str(self.runtime_endpoint).strip()
        continuation_model = str(self.runtime_model).strip()
        if bool(continuation_endpoint) != bool(continuation_model):
            raise ValueError(
                "stage2.extraction_llm.runtime_endpoint and runtime_model must be "
                "configured together"
            )
        if continuation_endpoint:
            parsed = urlparse(continuation_endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(
                    "stage2.extraction_llm.runtime_endpoint must be one HTTP(S) "
                    "OpenAI-compatible base URL"
                )
        if require_model and not self.model.strip():
            raise ValueError("stage2.extraction_llm.model must be nonempty")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("stage2.extraction_llm.workers must be a positive integer")

    def public_dict(self) -> dict[str, Any]:
        values = {
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key": "<redacted>",
            "workers": self.workers,
            "vllm": self.vllm.public_dict() if self.vllm is not None else None,
            "runtime_endpoint": self.runtime_endpoint,
            "runtime_model": self.runtime_model,
            "runtime_api_key": "<redacted>",
        }
        if self.endpoints:
            values["endpoints"] = [server.public_dict() for server in self.endpoints]
        return values


@dataclass(frozen=True)
class PlainHandoffStage2Config:
    endpoint: str
    model: str = ""
    # Reuse completed definitions produced by this model while subsequent
    # extraction/interpretation uses model. Missing definitions must not be
    # regenerated under a different model and attributed to this provenance.
    frozen_feature_definition_model: str = ""
    api_key: str = "EMPTY"
    # Total wall-clock budget for one logical JSON request, including transport
    # retries and validator-guided repair turns.
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    # Each individual HTTP attempt gets a smaller timeout so a straggling
    # endpoint can be abandoned and retried within the logical request budget.
    request_attempt_timeout: float = DEFAULT_REQUEST_ATTEMPT_TIMEOUT
    transport_max_attempts: int = DEFAULT_TRANSPORT_MAX_ATTEMPTS
    transport_retry_backoff: float = 2.0
    # Resume a fold's checkpointed work after exhausting a logical request's
    # transport/deadline budget. These operational controls do not change
    # scientific cache identities. Limits apply per invocation of the runner.
    outer_fold_recovery_attempts: int = 2
    outer_fold_recovery_backoff: float = 60.0
    # Invalid completed responses receive validator-guided repair requests.
    # Later repairs turn reasoning on without changing the initial request's
    # request-kind policy.
    max_response_repairs: int = DEFAULT_MAX_RESPONSE_REPAIRS
    thinking_after_response_repairs: int = DEFAULT_THINKING_AFTER_RESPONSE_REPAIRS
    # Primary-model interpretation requests receive this generous output
    # ceiling. It never forces the model to generate this many tokens; normal
    # EOS/stop behavior is unchanged.
    max_tokens: int = DEFAULT_MAX_TOKENS
    # Patient extraction uses a smaller model and an independent ceiling so a
    # long patient prompt plus the allowed completion fits smaller contexts.
    # This transport-only limit is deliberately absent from feature-definition
    # fingerprints, allowing completed discovery to resume under a safer cap.
    extraction_max_tokens: int = DEFAULT_EXTRACTION_MAX_TOKENS
    # Separate total output allowance (reasoning plus final JSON) when thinking
    # is enabled. None preserves the historical shared extraction ceiling.
    extraction_reasoning_max_tokens: int | None = None
    # Opt in for OpenAI-compatible servers that implement streaming and usage.
    # Streaming separates active generation from a stalled HTTP response.
    extraction_stream: bool = False
    extraction_deferred_retry_passes: int = 1
    interpretation_reasoning_effort: str = DEFAULT_INTERPRETATION_REASONING_EFFORT
    extraction_reasoning_effort: str = DEFAULT_EXTRACTION_REASONING_EFFORT
    max_prompt_chars: int = 100_000
    # Candidate consolidation has its own prompt allowance even though each
    # request sees only one bounded deterministic batch.
    consolidation_max_prompt_chars: int = DEFAULT_CONSOLIDATION_MAX_PROMPT_CHARS
    # A merged alias family can cite substantially more evidence than one
    # interpretation batch. Pack that evidence under an independent allowance.
    operationalization_max_prompt_chars: int = DEFAULT_OPERATIONALIZATION_MAX_PROMPT_CHARS
    consolidation_batch_size: int = DEFAULT_CONSOLIDATION_BATCH_SIZE
    consolidation_max_rounds: int = DEFAULT_CONSOLIDATION_MAX_ROUNDS
    consolidation_policy: CandidateConsolidationPolicy = field(default_factory=CandidateConsolidationPolicy)
    # Extraction keeps patients isolated and slices the frozen ontology across
    # independently checkpointed prompts. Keep its context allowance separate
    # so discovery batching and its evidence-compilation fingerprints remain stable.
    extraction_max_prompt_chars: int = DEFAULT_EXTRACTION_MAX_PROMPT_CHARS
    extraction_feature_batch_size: int = DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE
    # Long records are processed in ordered, lossless source chunks. This is a
    # token cap rather than a target: the planner shrinks a chunk when feature
    # definitions and carried-forward state need more of the context window.
    extraction_chunk_size_tokens: int = DEFAULT_EXTRACTION_CHUNK_SIZE_TOKENS
    extraction_context_window_tokens: int = DEFAULT_EXTRACTION_CONTEXT_WINDOW_TOKENS
    extraction_context_margin_tokens: int = DEFAULT_EXTRACTION_CONTEXT_MARGIN_TOKENS
    # Require the default lanes while retaining support for explicit selections
    # and saved handoffs containing either optional lexical architecture.
    required_architectures: tuple[str, ...] = DEFAULT_STAGE1_ARCHITECTURES
    included_architectures: tuple[str, ...] | None = None
    evidence_max_cards_per_fold: int = 400
    evidence_max_exemplars_per_card: int = 4
    evidence_max_exemplar_chars: int = 2_400
    workers: int = 4
    extraction_llm: Stage2ExtractionLLMConfig | None = None
    # Global CPU fit budget, independent of LLM request concurrency and GPUs.
    modeling_workers: int = 1
    # Limits aggregate ontology-supervisor reviews. These reviews cannot add,
    # drop, rename, or assign roles to candidate features.
    max_review_rounds: int = 2
    ontology_refinement_min_failure_patients: int = DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS
    max_ontology_refinement_rounds: int = DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS
    estimand_ontology: EstimandOntologyConfig = field(default_factory=EstimandOntologyConfig)
    # Optional outer-training-only semantic/association consolidation runs
    # immediately before supervised role selection.
    selection_consolidation: Stage2SequentialConsolidationConfig = field(
        default_factory=Stage2SequentialConsolidationConfig
    )
    # Deterministic post-extraction supervised selection.
    statistical_selection: Stage2ElasticNetSelectionConfig = field(
        default_factory=Stage2ElasticNetSelectionConfig
    )
    # Final role assignment reconciles all fold-honest statistical views. The
    # prompt builder has no dataset interface and accepts only an allowlisted,
    # aggregate evidence bundle.
    role_adjudication: Stage2RoleAdjudicationConfig = field(
        default_factory=Stage2RoleAdjudicationConfig
    )
    estimation_trees: int = 200
    min_propensity: float | None = None
    max_propensity: float | None = None
    propensity_clip: float = 0.02
    min_nonmissing_fraction: float = 0.05
    max_dominant_fraction: float = 0.98
    # None selects the publisher profile after endpoint/model resolution.
    # Explicit overrides apply to both primary and extraction requests.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    explicit_features: tuple[Stage2ExplicitFeature, ...] = ()
    vllm: ManagedVLLMConfig | None = None
    # When two pipeline-managed models request each other again within this
    # interval, keep both resident on their configured GPU splits for the rest
    # of the run. Zero retains all-GPU alternation unconditionally.
    vllm_rapid_switch_seconds: float = DEFAULT_VLLM_RAPID_SWITCH_SECONDS
    # Populated only while pipeline-owned servers are alive. It is deliberately
    # excluded from the persisted scientific configuration; the server-pool
    # manifest records the concrete endpoints and process/GPU assignments.
    runtime_endpoints: tuple[str, ...] = ()
    # Set only on the immutable config copy passed to one completion. It is not
    # persisted as scientific configuration because both request policies are.
    runtime_request_kind: str = "interpretation"
    # A bounded repair may temporarily strengthen the request-level reasoning
    # policy. This override is transport state, not scientific configuration.
    runtime_reasoning_effort: str | None = None
    # One task's response contract, supplied by its validator. It never enters
    # discovery/measurement identities or the persisted scientific config.
    runtime_response_format: Mapping[str, Any] | None = None
    # Absolute monotonic deadline and remaining HTTP-call allowance for one
    # transport-retry turn. These are populated only by the request runner so
    # compatibility negotiation cannot create unbounded hidden attempts.
    runtime_request_deadline: float | None = None
    runtime_transport_attempt_budget: int | None = None
    # Detected from the live endpoint's model record (including its backing
    # root when available) so served aliases still receive the right controls.
    runtime_model_family: str = ""
    runtime_sampling_model: str = ""
    # Live server context limits are transport state, not checkpoint identity.
    runtime_context_window_tokens: int | None = None
    # Operational guard for post-extraction reselection. Preserve the configured
    # extractor identity in checkpoints, but neither launch nor call it. Any
    # unexpected extraction request fails closed instead of occupying a GPU.
    runtime_disable_extraction: bool = False

    def validate(
        self,
        *,
        require_model: bool = True,
        require_endpoint: bool = True,
    ) -> None:
        if self.endpoint:
            parsed = urlparse(self.endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("stage2.endpoint must be one HTTP(S) OpenAI-compatible base URL")
        elif require_endpoint:
            raise ValueError("stage2.endpoint must be one HTTP(S) OpenAI-compatible base URL")
        if require_model and not self.model.strip():
            raise ValueError("stage2.model must be nonempty")
        if not isinstance(self.frozen_feature_definition_model, str) or (
            self.frozen_feature_definition_model
            and not self.frozen_feature_definition_model.strip()
        ):
            raise ValueError("stage2.frozen_feature_definition_model must be a model ID or empty string")
        if self.vllm is not None:
            self.vllm.validate()
        if (
            isinstance(self.vllm_rapid_switch_seconds, bool)
            or not isinstance(self.vllm_rapid_switch_seconds, (int, float))
            or not math.isfinite(float(self.vllm_rapid_switch_seconds))
            or self.vllm_rapid_switch_seconds < 0
        ):
            raise ValueError(
                "stage2.vllm_rapid_switch_seconds must be a finite nonnegative number"
            )
        for field_name, value in (
            ("request_timeout", self.request_timeout),
            ("request_attempt_timeout", self.request_attempt_timeout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"stage2.{field_name} must be a finite positive number")
        if self.transport_max_attempts < 1:
            raise ValueError("stage2.transport_max_attempts must be positive")
        if self.transport_retry_backoff < 0:
            raise ValueError("stage2.transport_retry_backoff must be nonnegative")
        if (isinstance(self.outer_fold_recovery_attempts, bool)
                or not isinstance(self.outer_fold_recovery_attempts, int)
                or self.outer_fold_recovery_attempts < 0):
            raise ValueError("stage2.outer_fold_recovery_attempts must be a nonnegative integer")
        if (isinstance(self.outer_fold_recovery_backoff, bool)
                or not isinstance(self.outer_fold_recovery_backoff, (int, float))
                or not math.isfinite(self.outer_fold_recovery_backoff)
                or self.outer_fold_recovery_backoff < 0):
            raise ValueError("stage2.outer_fold_recovery_backoff must be finite and nonnegative")
        if not isinstance(self.extraction_stream, bool):
            raise ValueError("stage2.extraction_stream must be a boolean")
        if (isinstance(self.extraction_deferred_retry_passes, bool)
                or not isinstance(self.extraction_deferred_retry_passes, int)
                or self.extraction_deferred_retry_passes < 0):
            raise ValueError("stage2.extraction_deferred_retry_passes must be a nonnegative integer")
        if (
            isinstance(self.max_response_repairs, bool)
            or not isinstance(self.max_response_repairs, int)
            or self.max_response_repairs < 0
        ):
            raise ValueError("stage2.max_response_repairs must be a nonnegative integer")
        if (
            isinstance(self.thinking_after_response_repairs, bool)
            or not isinstance(self.thinking_after_response_repairs, int)
            or self.thinking_after_response_repairs < 0
        ):
            raise ValueError(
                "stage2.thinking_after_response_repairs must be a nonnegative integer"
            )
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < MINIMUM_MAX_TOKENS
        ):
            raise ValueError(
                "stage2.max_tokens must be an integer of at least "
                f"{MINIMUM_MAX_TOKENS}; it is an output ceiling, not a minimum length"
            )
        if (
            isinstance(self.extraction_max_tokens, bool)
            or not isinstance(self.extraction_max_tokens, int)
            or self.extraction_max_tokens < MINIMUM_EXTRACTION_MAX_TOKENS
        ):
            raise ValueError(
                "stage2.extraction_max_tokens must be an integer of at least "
                f"{MINIMUM_EXTRACTION_MAX_TOKENS}; it is an output ceiling, not a "
                "minimum length"
            )
        if self.extraction_reasoning_max_tokens is not None and (
            isinstance(self.extraction_reasoning_max_tokens, bool)
            or not isinstance(self.extraction_reasoning_max_tokens, int)
            or self.extraction_reasoning_max_tokens < MINIMUM_EXTRACTION_MAX_TOKENS
        ):
            raise ValueError(
                "stage2.extraction_reasoning_max_tokens must be null or an integer "
                f"of at least {MINIMUM_EXTRACTION_MAX_TOKENS}"
            )
        for field_name, effort in (
            ("interpretation_reasoning_effort", self.interpretation_reasoning_effort),
            ("extraction_reasoning_effort", self.extraction_reasoning_effort),
        ):
            if not isinstance(effort, str) or effort not in SUPPORTED_REASONING_EFFORTS:
                raise ValueError(
                    f"stage2.{field_name} must be one of " f"{sorted(SUPPORTED_REASONING_EFFORTS)}"
                )
        if self.runtime_request_kind not in STAGE2_REQUEST_KINDS:
            raise ValueError("stage2.runtime_request_kind must be interpretation or extraction")
        if (
            self.runtime_reasoning_effort is not None
            and (
                not isinstance(self.runtime_reasoning_effort, str)
                or self.runtime_reasoning_effort not in SUPPORTED_REASONING_EFFORTS
            )
        ):
            raise ValueError(
                "stage2.runtime_reasoning_effort must be null or one of "
                f"{sorted(SUPPORTED_REASONING_EFFORTS)}"
            )
        if self.runtime_request_deadline is not None and (
            isinstance(self.runtime_request_deadline, bool)
            or not isinstance(self.runtime_request_deadline, (int, float))
            or not math.isfinite(float(self.runtime_request_deadline))
        ):
            raise ValueError("stage2.runtime_request_deadline must be null or finite")
        if self.runtime_transport_attempt_budget is not None and (
            isinstance(self.runtime_transport_attempt_budget, bool)
            or not isinstance(self.runtime_transport_attempt_budget, int)
            or self.runtime_transport_attempt_budget < 1
        ):
            raise ValueError(
                "stage2.runtime_transport_attempt_budget must be null or a positive integer"
            )
        if self.runtime_model_family not in {"", "qwen3", "gemma4", "lfm2.5", "other"}:
            raise ValueError("stage2.runtime_model_family is not recognized")
        if self.runtime_context_window_tokens is not None and (
            isinstance(self.runtime_context_window_tokens, bool)
            or not isinstance(self.runtime_context_window_tokens, int)
            or self.runtime_context_window_tokens < 1
        ):
            raise ValueError("stage2.runtime_context_window_tokens must be a positive integer or None")
        if not isinstance(self.runtime_disable_extraction, bool):
            raise ValueError("stage2.runtime_disable_extraction must be true or false")
        if self.max_prompt_chars < 4_000:
            raise ValueError("stage2.max_prompt_chars must be at least 4000")
        if self.consolidation_max_prompt_chars < 4_000:
            raise ValueError("stage2.consolidation_max_prompt_chars must be at least 4000")
        if self.operationalization_max_prompt_chars < 4_000:
            raise ValueError("stage2.operationalization_max_prompt_chars must be at least 4000")
        if self.consolidation_batch_size < 2:
            raise ValueError("stage2.consolidation_batch_size must be at least 2")
        if self.consolidation_max_rounds < 1:
            raise ValueError("stage2.consolidation_max_rounds must be positive")
        if not isinstance(self.consolidation_policy, CandidateConsolidationPolicy):
            raise ValueError("stage2.consolidation_policy must be a CandidateConsolidationPolicy")
        if self.extraction_max_prompt_chars < 4_000:
            raise ValueError("stage2.extraction_max_prompt_chars must be at least 4000")
        if (
            isinstance(self.extraction_feature_batch_size, bool)
            or not isinstance(self.extraction_feature_batch_size, int)
            or self.extraction_feature_batch_size < 1
        ):
            raise ValueError("stage2.extraction_feature_batch_size must be a positive integer")
        for field_name, value in (
            ("extraction_chunk_size_tokens", self.extraction_chunk_size_tokens),
            ("extraction_context_window_tokens", self.extraction_context_window_tokens),
            ("extraction_context_margin_tokens", self.extraction_context_margin_tokens),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if field_name == "extraction_context_margin_tokens" else 1)
            ):
                qualifier = (
                    "a nonnegative integer"
                    if field_name == "extraction_context_margin_tokens"
                    else "a positive integer"
                )
                raise ValueError(f"stage2.{field_name} must be {qualifier}")
        if (
            max(self.extraction_max_tokens, self.extraction_reasoning_max_tokens or 0)
            + self.extraction_context_margin_tokens
            >= self.extraction_context_window_tokens
        ):
            raise ValueError(
                "stage2 extraction context window must exceed extraction_max_tokens "
                "and extraction_reasoning_max_tokens "
                "plus extraction_context_margin_tokens"
            )
        required = tuple(self.required_architectures)
        if len(required) != len(set(required)):
            raise ValueError("stage2.required_architectures must not contain duplicates")
        unsupported = sorted(set(required) - set(SUPPORTED_STAGE2_ARCHITECTURES))
        if unsupported:
            raise ValueError(
                f"stage2.required_architectures contains unsupported values: {unsupported}"
            )
        included = (
            tuple(self.included_architectures) if self.included_architectures is not None else None
        )
        if included is not None:
            if len(included) != len(set(included)):
                raise ValueError("stage2.included_architectures must not contain duplicates")
            unsupported_included = sorted(set(included) - set(SUPPORTED_STAGE2_ARCHITECTURES))
            if unsupported_included:
                raise ValueError(
                    "stage2.included_architectures contains unsupported values: "
                    f"{unsupported_included}"
                )
            if not set(required).issubset(included):
                raise ValueError(
                    "stage2.required_architectures must be a subset of "
                    "stage2.included_architectures"
                )
        if self.evidence_max_cards_per_fold < 16:
            raise ValueError("stage2.evidence_max_cards_per_fold must be at least 16")
        if self.evidence_max_exemplars_per_card < 1:
            raise ValueError("stage2.evidence_max_exemplars_per_card must be positive")
        if self.evidence_max_exemplar_chars < 256:
            raise ValueError("stage2.evidence_max_exemplar_chars must be at least 256")
        if self.workers < 1:
            raise ValueError("stage2.workers must be positive")
        if type(self.modeling_workers) is not int or self.modeling_workers < 1:
            raise ValueError("stage2.modeling_workers must be a positive integer")
        if self.extraction_llm is not None:
            if not isinstance(self.extraction_llm, Stage2ExtractionLLMConfig):
                raise ValueError(
                    "stage2.extraction_llm must be a Stage2ExtractionLLMConfig object"
                )
            self.extraction_llm.validate(require_model=require_model)
        if self.max_review_rounds < 1:
            raise ValueError("stage2.max_review_rounds must be positive")
        if self.ontology_refinement_min_failure_patients < 2:
            raise ValueError("stage2.ontology_refinement_min_failure_patients must be at least 2")
        if self.max_ontology_refinement_rounds < 0:
            raise ValueError("stage2.max_ontology_refinement_rounds must be nonnegative")
        if not isinstance(
            self.selection_consolidation, Stage2SequentialConsolidationConfig
        ):
            raise ValueError(
                "stage2.selection_consolidation must be a "
                "Stage2SequentialConsolidationConfig object"
            )
        self.selection_consolidation.validate()
        if not isinstance(self.estimand_ontology, EstimandOntologyConfig):
            raise ValueError("stage2.estimand_ontology must be an EstimandOntologyConfig")
        self.estimand_ontology.validate()
        if self.estimand_ontology.enabled and self.runtime_disable_extraction:
            raise ValueError("estimand ontology search requires extraction of alternative definitions")
        if not isinstance(
            self.statistical_selection, Stage2ElasticNetSelectionConfig
        ):
            raise ValueError(
                "stage2.statistical_selection must be a "
                "Stage2ElasticNetSelectionConfig object"
            )
        self.statistical_selection.validate()
        if not isinstance(self.role_adjudication, Stage2RoleAdjudicationConfig):
            raise ValueError(
                "stage2.role_adjudication must be a "
                "Stage2RoleAdjudicationConfig object"
            )
        self.role_adjudication.validate()
        if self.estimation_trees < 10:
            raise ValueError("stage2.estimation_trees must be at least 10")
        validate_propensity_bounds(self.min_propensity, self.max_propensity)
        if (
            self.statistical_selection.min_propensity is not None
            or self.statistical_selection.max_propensity is not None
        ):
            raise ValueError(
                "configure min_propensity/max_propensity at stage2 level, not statistical_selection"
            )
        if not 0.0 < self.propensity_clip < 0.5:
            raise ValueError("stage2.propensity_clip must be between 0 and 0.5")
        if not 0.0 <= self.min_nonmissing_fraction <= 1.0:
            raise ValueError("stage2.min_nonmissing_fraction must be between 0 and 1")
        if not 0.0 <= self.max_dominant_fraction <= 1.0:
            raise ValueError("stage2.max_dominant_fraction must be between 0 and 1")
        for name in SAMPLING_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"stage2.{name} must be a finite number or null")
            if name == "top_k":
                valid = isinstance(value, int) and value >= -1
            elif name == "repetition_penalty":
                valid = value > 0
            elif name == "top_p":
                valid = 0 < value <= 1
            elif name == "min_p":
                valid = 0 <= value <= 1
            elif name == "temperature":
                valid = 0 <= value <= 2
            else:
                valid = -2 <= value <= 2
            if not valid:
                raise ValueError(f"stage2.{name} is outside its supported sampling range")
        names: list[str] = []
        for index, feature in enumerate(self.explicit_features):
            if not isinstance(feature, Stage2ExplicitFeature):
                raise ValueError(
                    "stage2.explicit_features entries must be Stage2ExplicitFeature "
                    f"objects; invalid entry at index {index}"
                )
            names.append(feature.name)
        duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicate_names:
            raise ValueError(
                "stage2.explicit_features contains duplicate normalized feature names: "
                f"{duplicate_names}"
            )

    def public_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("runtime_response_format", None)
        values["api_key"] = "<redacted>"
        values["extraction_llm"] = (
            self.extraction_llm.public_dict() if self.extraction_llm is not None else None
        )
        values.pop("runtime_endpoints", None)
        values.pop("runtime_request_kind", None)
        values.pop("runtime_reasoning_effort", None)
        values.pop("runtime_request_deadline", None)
        values.pop("runtime_transport_attempt_budget", None)
        values.pop("runtime_model_family", None)
        values.pop("runtime_sampling_model", None)
        values.pop("runtime_context_window_tokens", None)
        values.pop("runtime_disable_extraction", None)
        values["explicit_features"] = [
            feature.as_definition() for feature in self.explicit_features
        ]
        values["selection_consolidation"] = (
            self.selection_consolidation.public_dict()
        )
        values["estimand_ontology"] = self.estimand_ontology.public_dict()
        values["statistical_selection"] = self.statistical_selection.public_dict()
        values["role_adjudication"] = self.role_adjudication.public_dict()
        return values


def _reject_unknown_stage2_keys(
    raw: Mapping[str, Any],
    schema: type,
    *,
    section: str,
    extra: Iterable[str] = (),
) -> None:
    unknown = sorted(set(map(str, raw)) - set(schema.__dataclass_fields__) - set(extra))
    if unknown:
        raise ValueError(
            f"{section} contains unsupported fields: {unknown}. Fix the spelling or "
            "remove them; Stage 2 does not ignore unknown settings."
        )


def plain_stage2_config_from_mapping(
    raw: Mapping[str, Any],
    *,
    default_workers: int,
) -> PlainHandoffStage2Config | None:
    if raw.get("command"):
        raise ValueError(
            "stage2.command is not used by the plain workflow; configure stage2.endpoint"
        )
    # A misspelled scientific key would otherwise fall back to its default.
    # Every dataclass field is accepted so a saved run_config.json snapshot
    # round-trips; its runtime_* fields are re-derived at run time.
    _reject_unknown_stage2_keys(raw, PlainHandoffStage2Config, section="stage2")
    endpoint = str(raw.get("endpoint") or "").strip()
    model = str(raw.get("model") or "").strip()
    managed_vllm = managed_vllm_config_from_mapping(raw.get("vllm"), model=model)
    explicit_features = _stage2_explicit_features_from_value(raw.get("explicit_features"))
    if not endpoint and not model and managed_vllm is None:
        return None
    if endpoint and managed_vllm is not None:
        raise ValueError("configure either stage2.endpoint or stage2.vllm, not both")
    if managed_vllm is not None and not model:
        raise ValueError("stage2.model is required when stage2.vllm is configured")
    if not endpoint and managed_vllm is None:
        raise ValueError("stage2.endpoint is required when stage2.model is specified")
    api_key = str(raw.get("api_key") or os.environ.get("OCI_STAGE2_API_KEY") or "EMPTY")

    def architecture_names(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            if value.strip().lower() == "all":
                return SUPPORTED_STAGE2_ARCHITECTURES
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return tuple(value)

    interpretation_reasoning_effort = (
        str(
            raw.get(
                "interpretation_reasoning_effort",
                DEFAULT_INTERPRETATION_REASONING_EFFORT,
            )
        )
        .strip()
        .lower()
    )
    extraction_reasoning_effort = (
        str(
            raw.get(
                "extraction_reasoning_effort",
                DEFAULT_EXTRACTION_REASONING_EFFORT,
            )
        )
        .strip()
        .lower()
    )

    raw_extraction_llm = raw.get("extraction_llm")
    if raw_extraction_llm is None:
        extraction_llm = None
    elif not isinstance(raw_extraction_llm, Mapping):
        raise ValueError("stage2.extraction_llm must be an object")
    else:
        _reject_unknown_stage2_keys(
            raw_extraction_llm, Stage2ExtractionLLMConfig, section="stage2.extraction_llm"
        )
        extraction_endpoint = str(raw_extraction_llm.get("endpoint") or "").strip().rstrip("/")
        extraction_model = str(raw_extraction_llm.get("model") or "").strip()
        extraction_workers = max(1, int(raw_extraction_llm.get(
            "workers", min(4, max(1, default_workers)))))
        external_endpoints = endpoints_from_mapping(
            raw_extraction_llm.get("endpoints"), default_concurrency=extraction_workers,
        )
        if external_endpoints and "workers" not in raw_extraction_llm:
            extraction_workers = sum(server.max_concurrency for server in external_endpoints)
        extraction_vllm = managed_vllm_config_from_mapping(
            raw_extraction_llm.get("vllm"),
            model=extraction_model,
            default_base_port=DEFAULT_EXTRACTION_VLLM_BASE_PORT,
            default_internal_port_base=DEFAULT_EXTRACTION_VLLM_INTERNAL_PORT_BASE,
            section="stage2.extraction_llm.vllm",
        )
        if extraction_endpoint and extraction_vllm is not None:
            raise ValueError(
                "configure either stage2.extraction_llm.endpoint or "
                "stage2.extraction_llm.vllm, not both"
            )
        if not extraction_endpoint and not external_endpoints and extraction_vllm is None:
            raise ValueError(
                "stage2.extraction_llm requires endpoint, endpoints, or vllm"
            )
        if extraction_vllm is not None and not extraction_model:
            raise ValueError(
                "stage2.extraction_llm.model is required when its vllm pool is configured"
            )
        extraction_llm = Stage2ExtractionLLMConfig(
            endpoint=extraction_endpoint,
            model=extraction_model,
            api_key=str(
                raw_extraction_llm.get("api_key")
                or os.environ.get("OCI_STAGE2_EXTRACTION_API_KEY")
                or "EMPTY"
            ),
            workers=extraction_workers,
            vllm=extraction_vllm,
            endpoints=external_endpoints,
            runtime_endpoint=str(
                raw_extraction_llm.get("runtime_endpoint") or ""
            ).strip().rstrip("/"),
            runtime_model=str(
                raw_extraction_llm.get("runtime_model") or ""
            ).strip(),
            runtime_api_key=str(
                raw_extraction_llm.get("runtime_api_key")
                or os.environ.get("OCI_STAGE2_EXTRACTION_RUNTIME_API_KEY")
                or "EMPTY"
            ),
        )
        extraction_llm.validate(require_model=False)

    configured_workers = max(
        1,
        int(raw.get("workers", min(4, max(1, default_workers)))),
    )
    raw_statistical_selection = raw.get("statistical_selection")
    if raw_statistical_selection is None:
        statistical_selection_value: dict[str, Any] = {}
    elif isinstance(raw_statistical_selection, Mapping):
        statistical_selection_value = dict(raw_statistical_selection)
    else:
        raise ValueError("stage2.statistical_selection must be a configuration object")
    if any(
        statistical_selection_value.get(name) is not None
        for name in ("min_propensity", "max_propensity")
    ):
        raise ValueError(
            "configure min_propensity/max_propensity at stage2 level, not statistical_selection"
        )
    config = PlainHandoffStage2Config(
        endpoint=endpoint.rstrip("/"),
        model=model,
        frozen_feature_definition_model=raw.get("frozen_feature_definition_model", ""),
        api_key=api_key,
        request_timeout=float(raw.get("request_timeout", DEFAULT_REQUEST_TIMEOUT)),
        request_attempt_timeout=float(
            raw.get("request_attempt_timeout", DEFAULT_REQUEST_ATTEMPT_TIMEOUT)
        ),
        transport_max_attempts=int(
            raw.get("transport_max_attempts", DEFAULT_TRANSPORT_MAX_ATTEMPTS)
        ),
        transport_retry_backoff=float(raw.get("transport_retry_backoff", 2.0)),
        outer_fold_recovery_attempts=raw.get("outer_fold_recovery_attempts", 2),
        outer_fold_recovery_backoff=raw.get("outer_fold_recovery_backoff", 60.0),
        max_response_repairs=int(raw.get("max_response_repairs", DEFAULT_MAX_RESPONSE_REPAIRS)),
        thinking_after_response_repairs=int(
            raw.get(
                "thinking_after_response_repairs",
                DEFAULT_THINKING_AFTER_RESPONSE_REPAIRS,
            )
        ),
        max_tokens=int(raw.get("max_tokens", DEFAULT_MAX_TOKENS)),
        extraction_max_tokens=int(raw.get("extraction_max_tokens", DEFAULT_EXTRACTION_MAX_TOKENS)),
        extraction_reasoning_max_tokens=raw.get("extraction_reasoning_max_tokens"),
        extraction_stream=raw.get("extraction_stream", False),
        extraction_deferred_retry_passes=raw.get("extraction_deferred_retry_passes", 1),
        interpretation_reasoning_effort=interpretation_reasoning_effort,
        extraction_reasoning_effort=extraction_reasoning_effort,
        max_prompt_chars=int(raw.get("max_prompt_chars", 100_000)),
        consolidation_max_prompt_chars=int(
            raw.get(
                "consolidation_max_prompt_chars",
                DEFAULT_CONSOLIDATION_MAX_PROMPT_CHARS,
            )
        ),
        operationalization_max_prompt_chars=int(
            raw.get(
                "operationalization_max_prompt_chars",
                DEFAULT_OPERATIONALIZATION_MAX_PROMPT_CHARS,
            )
        ),
        consolidation_batch_size=int(
            raw.get("consolidation_batch_size", DEFAULT_CONSOLIDATION_BATCH_SIZE)
        ),
        consolidation_max_rounds=int(
            raw.get("consolidation_max_rounds", DEFAULT_CONSOLIDATION_MAX_ROUNDS)
        ),
        consolidation_policy=consolidation_policy_from_mapping(raw.get("consolidation_policy")),
        extraction_max_prompt_chars=int(
            raw.get(
                "extraction_max_prompt_chars",
                DEFAULT_EXTRACTION_MAX_PROMPT_CHARS,
            )
        ),
        extraction_feature_batch_size=int(
            raw.get(
                "extraction_feature_batch_size",
                DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE,
            )
        ),
        extraction_chunk_size_tokens=int(
            raw.get(
                "extraction_chunk_size_tokens",
                DEFAULT_EXTRACTION_CHUNK_SIZE_TOKENS,
            )
        ),
        extraction_context_window_tokens=int(
            raw.get(
                "extraction_context_window_tokens",
                DEFAULT_EXTRACTION_CONTEXT_WINDOW_TOKENS,
            )
        ),
        extraction_context_margin_tokens=int(
            raw.get(
                "extraction_context_margin_tokens",
                DEFAULT_EXTRACTION_CONTEXT_MARGIN_TOKENS,
            )
        ),
        required_architectures=architecture_names(
            raw.get("required_architectures", DEFAULT_STAGE1_ARCHITECTURES)
        ),
        included_architectures=(
            architecture_names(raw["included_architectures"])
            if raw.get("included_architectures") is not None
            else None
        ),
        evidence_max_cards_per_fold=int(raw.get("evidence_max_cards_per_fold", 400)),
        evidence_max_exemplars_per_card=int(raw.get("evidence_max_exemplars_per_card", 4)),
        evidence_max_exemplar_chars=int(raw.get("evidence_max_exemplar_chars", 2_400)),
        workers=configured_workers,
        modeling_workers=raw.get("modeling_workers", 1),
        extraction_llm=extraction_llm,
        max_review_rounds=int(raw.get("max_review_rounds", 2)),
        ontology_refinement_min_failure_patients=int(
            raw.get(
                "ontology_refinement_min_failure_patients",
                DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS,
            )
        ),
        max_ontology_refinement_rounds=int(
            raw.get(
                "max_ontology_refinement_rounds",
                DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS,
            )
        ),
        selection_consolidation=sequential_consolidation_config_from_mapping(
            raw.get("selection_consolidation")
        ),
        estimand_ontology=estimand_ontology_config_from_mapping(raw.get("estimand_ontology")),
        statistical_selection=statistical_selection_config_from_mapping(
            statistical_selection_value
        ),
        role_adjudication=role_adjudication_config_from_mapping(raw.get("role_adjudication")),
        estimation_trees=int(raw.get("estimation_trees", 200)),
        min_propensity=raw.get("min_propensity"),
        max_propensity=raw.get("max_propensity"),
        propensity_clip=float(raw.get("propensity_clip", 0.02)),
        min_nonmissing_fraction=float(raw.get("min_nonmissing_fraction", 0.05)),
        max_dominant_fraction=float(raw.get("max_dominant_fraction", 0.98)),
        **{name: raw.get(name) for name in SAMPLING_FIELDS},
        explicit_features=explicit_features,
        vllm=managed_vllm,
        vllm_rapid_switch_seconds=float(
            raw.get(
                "vllm_rapid_switch_seconds",
                DEFAULT_VLLM_RAPID_SWITCH_SECONDS,
            )
        ),
        runtime_disable_extraction=raw.get(
            "runtime_disable_extraction",
            False,
        ),
    )
    config.validate(
        require_model=False,
        require_endpoint=managed_vllm is None,
    )
    return config


class _ServedModelIds(list[str]):
    """Model IDs plus stable optional metadata returned by richer servers."""

    def __init__(self, values: Iterable[str], *, records: Sequence[Mapping[str, Any]]):
        super().__init__(values)
        self.records = tuple(dict(record) for record in records)


def _served_model_ids(config: PlainHandoffStage2Config) -> list[str]:
    """Return the distinct model IDs advertised by an OpenAI-compatible server."""

    from openai import OpenAI

    client = OpenAI(
        base_url=config.endpoint,
        api_key=config.api_key,
        timeout=config.request_timeout,
        max_retries=2,
    )
    try:
        response = client.models.list()
    except Exception as exc:
        raise RuntimeError(
            "Stage 2 could not auto-discover a model from "
            f"{config.endpoint}/models: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        client.close()
    records_by_id: dict[str, dict[str, Any]] = {}
    for model in response.data:
        model_id = str(getattr(model, "id", "") or "").strip()
        if not model_id:
            continue
        extras = getattr(model, "model_extra", None)
        extras = extras if isinstance(extras, Mapping) else {}

        def stable_field(name: str) -> str | None:
            value = getattr(model, name, None)
            if value is None:
                value = extras.get(name)
            rendered = str(value or "").strip()
            return rendered or None

        def positive_int_field(name: str) -> int | None:
            value = getattr(model, name, None)
            if value is None:
                value = extras.get(name)
            if isinstance(value, bool):
                return None
            if isinstance(value, str) and value.strip().isdigit():
                value = int(value.strip())
            return value if isinstance(value, int) and value > 0 else None

        records_by_id[model_id] = {
            "id": model_id,
            "root": stable_field("root"),
            "parent": stable_field("parent"),
            "revision": stable_field("revision") or stable_field("model_revision"),
            # vLLM advertises the served context window here; external
            # endpoints have no managed --max-model-len to fall back on.
            "max_model_len": positive_int_field("max_model_len"),
        }
    records = [records_by_id[model_id] for model_id in sorted(records_by_id)]
    return _ServedModelIds(
        (record["id"] for record in records),
        records=records,
    )


def _resolve_stage2_model(config: PlainHandoffStage2Config) -> PlainHandoffStage2Config:
    """Use the sole model advertised by the endpoint when none was configured."""

    if config.model.strip():
        return config
    model_ids = _served_model_ids(config)
    if not model_ids:
        models_url = f"{config.endpoint}/models"
        raise RuntimeError(f"Stage 2 model auto-discovery found no models at {models_url}")
    if len(model_ids) != 1:
        raise RuntimeError(
            "Stage 2 model auto-discovery requires exactly one served model; "
            f"{config.endpoint}/models advertised {model_ids}. Set stage2.model explicitly."
        )
    resolved = replace(config, model=model_ids[0])
    LOGGER.info(
        "auto-discovered Stage 2 model=%s from %s/models",
        resolved.model,
        config.endpoint,
    )
    return resolved


def _resolve_extraction_llm_model(
    config: PlainHandoffStage2Config,
) -> PlainHandoffStage2Config:
    """Resolve and persist the model identity for the independent extractor."""

    extraction = config.extraction_llm
    if extraction is None or extraction.model.strip():
        return config
    if extraction.vllm is not None:
        raise ValueError(
            "stage2.extraction_llm.model is required when its vllm pool is configured"
        )
    transport_config = replace(
        config,
        endpoint=extraction.endpoints[0].endpoint if extraction.endpoints else extraction.endpoint,
        model="",
        api_key=extraction.api_key,
        workers=extraction.workers,
        extraction_llm=None,
        vllm=None,
        runtime_endpoints=(),
    )
    resolved = _resolve_stage2_model(transport_config)
    LOGGER.info(
        "auto-discovered Stage 2 extraction model=%s from %s/models",
        resolved.model,
        transport_config.endpoint,
    )
    return replace(config, extraction_llm=replace(extraction, model=resolved.model))


def _stage2_model_family(model: str) -> str:
    """Classify model IDs only where Stage 2 has a concrete reasoning control."""

    compact = re.sub(r"[^a-z0-9]+", "", str(model).strip().lower())
    if "qwen3" in compact:
        # Covers hybrid Qwen 3 releases as well as the 3.5/3.6/3.8 naming
        # convention. Dedicated -Instruct/-Thinking variants are harmlessly
        # handled by the same output parser even if their switch is fixed.
        return "qwen3"
    if "gemma4" in compact:
        return "gemma4"
    if "lfm25" in compact:
        return "lfm2.5"
    return "other"


def _endpoint_model_identity(
    config: PlainHandoffStage2Config,
    *,
    endpoints: Sequence[str],
    role: str,
    verify_live_endpoint: bool,
) -> dict[str, Any]:
    """Record the selected model and, for live transports, verify every replica."""

    selected_model = str(config.model).strip()
    if not selected_model:
        raise RuntimeError(f"Stage 2 {role} model identity is unresolved")
    observations: list[dict[str, Any]] = []
    actual_identities: list[dict[str, Any]] = []
    normalized_endpoints = tuple(
        dict.fromkeys(str(endpoint).strip().rstrip("/") for endpoint in endpoints)
    )
    for endpoint in normalized_endpoints:
        if verify_live_endpoint:
            advertised = _served_model_ids(replace(config, endpoint=endpoint))
            if selected_model not in advertised:
                raise RuntimeError(
                    f"Stage 2 {role} endpoint {endpoint}/models advertises {advertised}, "
                    f"not the selected model {selected_model!r}. Refusing to run because "
                    "the actual served model does not match the configured/resolved identity."
                )
            advertised_records = [
                dict(record)
                for record in getattr(advertised, "records", ())
                if str(record.get("id") or "") == selected_model
            ]
            selected_record = (
                advertised_records[0]
                if advertised_records
                else {
                    "id": selected_model,
                    "root": None,
                    "parent": None,
                    "revision": None,
                    "max_model_len": None,
                }
            )
        else:
            # Custom completion callbacks have no standard endpoint to probe.
            advertised = []
            selected_record = None
        if selected_record is not None:
            actual_identities.append(
                {
                    "served_id": selected_model,
                    "root": str(selected_record.get("root") or selected_model),
                    "parent": selected_record.get("parent"),
                    "revision": selected_record.get("revision"),
                }
            )
        observations.append(
            {
                "endpoint": endpoint,
                "advertised_model_ids": list(advertised),
                "selected_model_advertised": (
                    selected_model in advertised if verify_live_endpoint else None
                ),
                "selected_model_record": selected_record,
            }
        )
    distinct_actual = {
        json.dumps(identity, sort_keys=True, separators=(",", ":"))
        for identity in actual_identities
    }
    if len(distinct_actual) > 1:
        raise RuntimeError(
            f"Stage 2 {role} replicas advertise inconsistent backing model "
            f"identities: {actual_identities}"
        )
    family_source = (
        str(actual_identities[0].get("root") or selected_model)
        if actual_identities
        else selected_model
    )
    return {
        "role": role,
        "selected_model": selected_model,
        "model_family": _stage2_model_family(family_source),
        "sampling_model": family_source,
        "actual_model_identity": actual_identities[0] if actual_identities else None,
        "live_endpoint_verified": bool(verify_live_endpoint),
        "endpoint_observations": observations,
    }


def _scientific_model_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project a model manifest onto identities that may affect saved science."""

    def role_identity(role: str) -> dict[str, Any] | None:
        raw = value.get(role)
        if not isinstance(raw, Mapping):
            return None
        selected = str(raw.get("selected_model") or "").strip()
        if not selected:
            return None
        identity: dict[str, Any] = {
            "selected_model": selected,
            "model_family": str(raw.get("model_family") or _stage2_model_family(selected)),
        }
        actual = raw.get("actual_model_identity")
        if isinstance(actual, Mapping):
            identity["actual_model_identity"] = {
                "served_id": str(actual.get("served_id") or selected),
                "root": str(actual.get("root") or selected),
                "parent": actual.get("parent"),
                "revision": actual.get("revision"),
            }
        return identity

    return {
        "primary": role_identity("primary"),
        "extraction": role_identity("extraction"),
    }


def _model_identities_compatible(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    """Compare served model names; backing metadata is retained only for audit."""

    for role in ("primary", "extraction"):
        old = previous.get(role)
        new = current.get(role)
        if old is None or new is None:
            if old is not new:
                return False
            continue
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            return False
        if str(old.get("selected_model") or "") != str(new.get("selected_model") or ""):
            return False
    return True


def _model_role_identity_compatible(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    role: str,
) -> bool:
    """Compare one persisted model role without coupling independent roles."""

    return _model_identities_compatible(
        {"primary": previous.get(role), "extraction": None},
        {"primary": current.get(role), "extraction": None},
    )


def _has_extraction_dependent_checkpoints(output_dir: Path) -> bool:
    """Return whether saved science depends on the configured extractor model."""

    output_dir = Path(output_dir)
    root_outputs = (
        "causal_estimate.json",
        "cross_fitted_predictions.csv",
        "posthoc_oracle_ite_metrics.json",
        "posthoc_predictions_with_oracle_ite.csv",
    )
    if any((output_dir / name).exists() for name in root_outputs):
        return True
    extraction_science_names = (
        "extraction",
        "ontology_supervision",
        "review",
        "selection",
        "estimation",
        "final_definitions.json",
    )
    for outer_dir in output_dir.glob("outer_*"):
        if not outer_dir.is_dir():
            continue
        if any((outer_dir / name).exists() for name in extraction_science_names):
            return True
        complete_path = outer_dir / "complete.json"
        if complete_path.is_file():
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return True
            if completion.get("phase") == "causal_estimation":
                return True
    return False


CompletionFunction = Callable[[Sequence[Mapping[str, str]], PlainHandoffStage2Config], str]


class _RetryableStage2ResponseError(RuntimeError):
    """A completed transport that did not yield any response content."""


class _Stage2OutputLengthError(ValueError):
    """The server exhausted the available completion length."""


class _Stage2OutputWhitespaceError(_Stage2OutputLengthError):
    """Abort a streamed JSON formatting loop and request a fresh response."""


class _Stage2ReasoningOnlyResponseError(ValueError):
    """The server returned reasoning text but no final message content.

    Reasoning can contain draft JSON objects, so it is never parsed as the
    answer; the request is repaired instead.
    """


class _Stage2TransportFailure(RuntimeError):
    """Carry the number of real HTTP calls consumed by one completion call."""

    def __init__(self, cause: Exception, *, attempts_used: int) -> None:
        self.cause = cause
        self.attempts_used = max(1, int(attempts_used))
        super().__init__(str(cause))


class _ManagedStage2ModelSwitch(RuntimeError):
    """Signal that a checkpointed managed run needs the other model loaded."""

    def __init__(self, required_role: str) -> None:
        role = str(required_role).strip().lower()
        if role not in STAGE2_REQUEST_KINDS:
            raise ValueError(f"unknown managed Stage 2 model role: {required_role!r}")
        self.required_role = role
        super().__init__(f"managed Stage 2 requires the {role} model")


@dataclass
class _ManagedStage2SwitchTracker:
    """Track model-role switches with a monotonic in-process clock."""

    rapid_switch_seconds: float
    last_switch_monotonic: float | None = None
    last_switch_at: str | None = None
    previous_switch_elapsed_seconds: float | None = None

    def mark_switch(self) -> float | None:
        switched_at = time.monotonic()
        elapsed = (
            None
            if self.last_switch_monotonic is None
            else max(0.0, switched_at - self.last_switch_monotonic)
        )
        self.last_switch_monotonic = switched_at
        self.last_switch_at = _now()
        self.previous_switch_elapsed_seconds = elapsed
        return elapsed

    def is_rapid(self, elapsed: float | None) -> bool:
        return bool(
            self.rapid_switch_seconds > 0
            and elapsed is not None
            and elapsed < self.rapid_switch_seconds
        )


class _LazyStage2ExtractionTokenizer:
    """Load an exact model tokenizer only when token-bounded work begins."""

    def __init__(
        self,
        *,
        model: str,
        cache_dir: str = "",
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = str(model)
        self.cache_dir = str(cache_dir)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._tokenizer: Any | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._lock:
            if self._tokenizer is None:
                try:
                    from transformers import AutoTokenizer

                    kwargs: dict[str, Any] = {
                        "trust_remote_code": True,
                        "use_fast": True,
                    }
                    if self.cache_dir:
                        kwargs["cache_dir"] = self.cache_dir
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model,
                        **kwargs,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Stage 2 cannot enforce prompt token limits because "
                        f"the tokenizer {self.model!r} could not be loaded. "
                        "Make the tokenizer available in the vLLM download "
                        "directory or the Hugging Face cache."
                    ) from exc
        return self._tokenizer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._load()(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        template_kwargs = {**self.chat_template_kwargs, **kwargs}
        return self._load().apply_chat_template(*args, **template_kwargs)


def _primary_context_window_tokens(
    config: PlainHandoffStage2Config,
    identity: Mapping[str, Any] | None = None,
) -> int | None:
    limits: list[int] = []
    if config.runtime_context_window_tokens is not None:
        limits.append(int(config.runtime_context_window_tokens))
    for observation in (identity or {}).get("endpoint_observations", []):
        record = observation.get("selected_model_record") or {}
        value = record.get("max_model_len")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            limits.append(value)
    args = list(config.vllm.extra_args) if config.vllm is not None else []
    for index, arg in enumerate(args):
        if arg == "--max-model-len" and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith("--max-model-len="):
            value = arg.split("=", 1)[1]
        else:
            continue
        # vLLM also accepts "auto"; in that case use its advertised limit.
        try:
            limit = int(value)
        except (TypeError, ValueError):
            continue
        if limit > 0:
            limits.append(limit)
    return min(limits) if limits else None


@lru_cache(maxsize=8)
def _primary_prompt_tokenizer(
    model: str, cache_dir: str, template_kwargs_json: str,
) -> _LazyStage2ExtractionTokenizer:
    return _LazyStage2ExtractionTokenizer(
        model=model, cache_dir=cache_dir,
        chat_template_kwargs=json.loads(template_kwargs_json),
    )


@lru_cache(maxsize=8)
def _external_primary_tokenizer_available(model: str) -> bool:
    """Budget external-endpoint prompts only when the served tokenizer loads locally.

    Without client-side budgeting an over-long prompt is rejected by the server,
    so no semantic content is silently truncated.
    """
    tokenizer = _primary_prompt_tokenizer(model, "", json.dumps({}, sort_keys=True))
    try:
        if hasattr(tokenizer, "_load"):
            tokenizer._load()
    except RuntimeError as exc:
        LOGGER.warning(
            "Prompt token budgeting is disabled for external endpoint model %r because "
            "its tokenizer could not be loaded locally (%s). The server will reject "
            "prompts that exceed its context window.",
            model,
            exc.__cause__ or exc,
        )
        return False
    return True


def _primary_prompt_token_counter(
    config: PlainHandoffStage2Config,
) -> Callable[[Sequence[Mapping[str, str]]], int]:
    template_kwargs = dict(config.vllm.default_chat_template_kwargs or {}) if config.vllm else {}
    tokenizer = _primary_prompt_tokenizer(
        config.runtime_sampling_model or config.model,
        str(config.vllm.download_dir) if config.vllm else "",
        json.dumps(template_kwargs, sort_keys=True),
    )
    policy = _stage2_request_policy(config, "interpretation")
    family = config.runtime_model_family or _stage2_model_family(config.model)

    def count(messages: Sequence[Mapping[str, str]]) -> int:
        # Repairs can enable thinking. Reserve the larger framing in either
        # mode, including the same prompt controls used by the transport.
        counts = []
        for thinking in (False, True):
            controlled = (
                list(messages) if policy.get("preserve_thinking") else
                _reasoning_controlled_messages(
                    messages, model_family=family, enable_thinking=thinking,
                    max_prompt_chars=int(config.max_prompt_chars),
                )
            )
            encoded = tokenizer.apply_chat_template(
                controlled, tokenize=True, add_generation_prompt=True,
                enable_thinking=thinking,
                **({"preserve_thinking": True} if policy.get("preserve_thinking") else {}),
            )
            counts.append(_token_id_count(encoded))
        return max(counts)

    return count


def _stage2_request_policy(
    config: PlainHandoffStage2Config,
    request_kind: str | None = None,
) -> dict[str, Any]:
    """Resolve one request's reasoning mode and role-specific output ceiling."""

    kind = str(request_kind or config.runtime_request_kind).strip().lower()
    if kind not in STAGE2_REQUEST_KINDS:
        raise ValueError("Stage 2 request_kind must be interpretation or extraction")
    configured_reasoning_effort = (
        config.extraction_reasoning_effort
        if kind == "extraction"
        else config.interpretation_reasoning_effort
    )
    model = config.runtime_sampling_model or config.model
    reasoning_effort = config.runtime_reasoning_effort or configured_reasoning_effort
    if reasoning_effort == "auto":
        reasoning_effort = default_reasoning(model, kind)
    extraction_ceiling = config.extraction_max_tokens
    if _reasoning_enabled(reasoning_effort) and config.extraction_reasoning_max_tokens is not None:
        extraction_ceiling = config.extraction_reasoning_max_tokens
    return {
        "request_kind": kind,
        "reasoning_effort": reasoning_effort,
        "sampling_provenance": sampling_provenance(
            model, config.runtime_model_family or _stage2_model_family(model)
        ),
        "preserve_thinking": is_qwen_flash_next(model),
        "max_tokens": int(
            extraction_ceiling if kind == "extraction" else config.max_tokens
        ),
        **recommended_sampling(
            config.runtime_sampling_model or config.model,
            config.runtime_model_family or _stage2_model_family(model),
            _reasoning_enabled(reasoning_effort),
        ),
        **{
            name: getattr(config, name)
            for name in SAMPLING_FIELDS
            if getattr(config, name) is not None
        },
    }


def _feature_definition_input_value(
    *,
    config: PlainHandoffStage2Config,
    clinical_question: str,
    outer_fold: int,
    discovery_packets: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Fingerprint only inputs that can affect discovery and operationalization."""

    if config.frozen_feature_definition_model:
        config = replace(
            config,
            model=config.frozen_feature_definition_model,
            runtime_sampling_model=config.frozen_feature_definition_model,
            runtime_model_family=_stage2_model_family(config.frozen_feature_definition_model),
            runtime_reasoning_effort=None,
        )
    return {
        "feature_definition_input_schema": FEATURE_DEFINITION_INPUT_SCHEMA_VERSION,
        "prompt_catalog_version": clinical_prompts.PROMPT_VERSION,
        "alias_merge_prompt_version": clinical_prompts.ALIAS_MERGE_PROMPT_VERSION,
        "outer_fold": int(outer_fold),
        "compiler": EVIDENCE_COMPILER_VERSION,
        "candidate_discovery_source": "all_semantic_evidence_cards",
        "interpretation_schema": INTERPRETATION_SCHEMA_VERSION,
        "primary_llm": {
            "model": config.model,
        },
        "interpretation_request_policy": _stage2_request_policy(
            config,
            "interpretation",
        ),
        "consolidation_schema": CONSOLIDATION_SCHEMA_VERSION,
        "global_candidate_pool_schema": MIXED_SCHEMA,
        "consolidation_batch_size": int(config.consolidation_batch_size),
        "consolidation_max_rounds": int(config.consolidation_max_rounds),
        "consolidation_policy": config.consolidation_policy.public_dict(),
        "consolidation_seed": int(seed),
        "extraction_ontology_feedback_schema": EXTRACTION_ONTOLOGY_FEEDBACK_SCHEMA_VERSION,
        "ontology_refinement_min_failure_patients": int(
            config.ontology_refinement_min_failure_patients
        ),
        "max_ontology_refinement_rounds": int(config.max_ontology_refinement_rounds),
        "clinical_question": str(clinical_question),
        "explicit_features": [
            feature.as_definition() for feature in config.explicit_features
        ],
        "discovery_packets": list(discovery_packets),
    }


def _thinking_response_repair_effort(configured_effort: str) -> str:
    """Turn thinking on for a late repair without weakening stronger policies."""

    if configured_effort in {"high", "xhigh", "max"}:
        return configured_effort
    return THINKING_RESPONSE_REPAIR_EFFORT


def _reasoning_enabled(reasoning_effort: str) -> bool:
    return str(reasoning_effort).strip().lower() not in {"none", "minimal"}


def _reasoning_controlled_messages(
    messages: Sequence[Mapping[str, str]],
    *,
    model_family: str,
    enable_thinking: bool,
    max_prompt_chars: int,
) -> list[dict[str, str]]:
    """Add a portable prompt fallback without violating the prompt budget."""

    controlled = [dict(message) for message in messages]
    if model_family not in {"qwen3", "gemma4", "lfm2.5"}:
        return controlled
    if model_family == "qwen3":
        directive = "/think" if enable_thinking else "/no_think"
    elif enable_thinking:
        directive = (
            "Enable the model's thinking mode for this request, but return only the "
            "final JSON object in response content."
        )
    else:
        directive = "Disable thinking for this request and return only the final JSON object."
    target_index = next(
        (
            index
            for index in range(len(controlled) - 1, -1, -1)
            if str(controlled[index].get("role") or "")
            == ("user" if model_family == "qwen3" else "system")
        ),
        None,
    )
    if target_index is None:
        target_index = len(controlled) - 1 if controlled else None
    candidate = [dict(message) for message in controlled]
    if target_index is None:
        candidate.append({"role": "system", "content": directive})
    else:
        prior = str(candidate[target_index].get("content") or "")
        candidate[target_index]["content"] = f"{prior}\n\n{directive}".strip()
    candidate_chars = sum(
        len(str(message.get("content") or "")) for message in candidate
    )
    return candidate if candidate_chars <= int(max_prompt_chars) else controlled


def _openai_optional_parameter_error(exc: Exception) -> bool:
    """Recognize a server rejecting nonstandard OpenAI-compatible controls."""

    status_code = getattr(exc, "status_code", None)
    if status_code not in {400, 422}:
        return False
    text = str(exc).lower()
    normalized_text = re.sub(r"[_-]+", " ", text)
    parameter_names = (
        "reasoning effort",
        "enable thinking",
        "enablethinking",
        "chat template kwargs",
        "response format",
        "stream options",
        "include usage",
        "repetition penalty",
        "top k",
        "min p",
        "extra body",
    )
    rejection_words = (
        "unknown",
        "unsupported",
        "unrecognized",
        "not permitted",
        "not allowed",
        "extra input",
        "unexpected",
        "invalid parameter",
    )
    return any(name in normalized_text for name in parameter_names) and any(
        word in normalized_text for word in rejection_words
    )


def _wire_reasoning_effort(
    *,
    configured_effort: str,
    model_family: str,
) -> str | None:
    """Translate the pipeline policy to one endpoint's accepted wire values."""

    effort = str(configured_effort).strip().lower()
    enabled = _reasoning_enabled(effort)
    if model_family in {"qwen3", "gemma4", "lfm2.5"} and not enabled:
        # These families have an explicit hard switch in extra_body (and a
        # prompt fallback). Sending reasoning_effort="none" as well breaks
        # endpoints whose enum contains only enabled reasoning levels.
        return None
    if model_family == "qwen3":
        # Qwen 3.8 exposes low/medium/xhigh. Keep the public pipeline policy
        # model-agnostic while translating high-strength requests to its wire
        # vocabulary. Older Qwen 3 servers can reject this optional field and
        # fall through to the enable_thinking switch below.
        if effort in {"high", "xhigh", "max"}:
            return "xhigh"
        if effort in {"low", "medium"}:
            return effort
    return effort


def _openai_request_variants(
    *,
    base_kwargs: Mapping[str, Any],
    request_policy: Mapping[str, Any],
    model_family: str,
) -> list[dict[str, Any]]:
    """Prefer hard thinking controls, then degrade across compatible APIs."""

    enabled = _reasoning_enabled(str(request_policy["reasoning_effort"]))
    wire_reasoning_effort = _wire_reasoning_effort(
        configured_effort=str(request_policy["reasoning_effort"]),
        model_family=model_family,
    )
    sampling_extra = {
        name: request_policy[name] for name in ("repetition_penalty", "top_k", "min_p")
        if name in request_policy
    }
    preferred_format = base_kwargs.get("response_format") or {"type": "json_object"}
    response_formats = [preferred_format]
    if preferred_format.get("type") == "json_schema":
        # External compatible servers can reject JSON schemas. Retain the task
        # reminder and semantic validator when negotiating JSON-only serving.
        response_formats.append({"type": "json_object"})
    family_bodies: list[dict[str, Any]] = []
    if model_family in {"qwen3", "gemma4", "lfm2.5"}:
        if model_family == "lfm2.5":
            family_bodies.append(
                {
                    **sampling_extra,
                    "enableThinking": enabled,
                }
            )
        family_bodies.extend(
            [
                {
                    **sampling_extra,
                    "chat_template_kwargs": {
                        "enable_thinking": enabled,
                        **({"preserve_thinking": True} if request_policy.get("preserve_thinking") else {}),
                    },
                },
                {
                    **sampling_extra,
                    "enable_thinking": enabled,
                },
            ]
        )
    family_bodies.extend([sampling_extra, {}])

    if request_policy.get("preserve_thinking"):
        # The Flash Next publisher documents these controls. Never silently
        # lower reasoning effort or discard its sampling profile on a retry.
        strict = {**dict(base_kwargs), "extra_body": family_bodies[0]}
        if wire_reasoning_effort is not None:
            strict["reasoning_effort"] = wire_reasoning_effort
        unformatted = {k: v for k, v in strict.items() if k != "response_format"}
        return [{**strict, "response_format": fmt} for fmt in response_formats] + [unformatted]

    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    reasoning_variants = (True, False) if wire_reasoning_effort is not None else (False,)
    for extra_body in family_bodies:
        for include_reasoning in reasoning_variants:
            for response_format in response_formats:
                candidate = {**dict(base_kwargs), "response_format": response_format}
                if include_reasoning:
                    candidate["reasoning_effort"] = wire_reasoning_effort
                if extra_body:
                    candidate["extra_body"] = extra_body
                key = json.dumps(candidate, sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key)
                    variants.append(candidate)
    # Last-resort OpenAI-compatible endpoints may support neither structured
    # output nor any reasoning/sampling extension. The prompt still requires JSON.
    bare = dict(base_kwargs)
    bare.pop("response_format", None)
    key = json.dumps(bare, sort_keys=True, default=str)
    if key not in seen:
        variants.append(bare)
    return variants


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for part in value:
            if isinstance(part, Mapping):
                text = part.get("text") or part.get("content")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _response_message_text(message: Any) -> str:
    """Read final message content; inline reasoning is removed by the JSON parser.

    Separately parsed reasoning is never used as the answer (see
    ``_response_reasoning_text``).
    """

    content = _message_text(
        message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    )
    return content if content.strip() else ""


def _response_reasoning_text(message: Any) -> str:
    """Return reasoning text that a server parsed out of the final content."""

    containers: list[Mapping[str, Any]] = []
    if isinstance(message, Mapping):
        containers.append(message)
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, Mapping):
        containers.append(model_extra)
    for field_name in ("reasoning_content", "reasoningContent", "reasoning"):
        direct = getattr(message, field_name, None)
        text = _message_text(direct)
        if text.strip():
            return text
        for container in containers:
            text = _message_text(container.get(field_name))
            if text.strip():
                return text
    return ""


def _openai_completion(
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
) -> str:
    from openai import OpenAI

    request_policy = _stage2_request_policy(config)
    model_family = config.runtime_model_family or _stage2_model_family(config.model)
    wire_reasoning_effort = _wire_reasoning_effort(
        configured_effort=str(request_policy["reasoning_effort"]),
        model_family=model_family,
    )
    controlled_messages = (
        [dict(message) for message in messages]
        if request_policy.get("preserve_thinking") else _reasoning_controlled_messages(
            messages, model_family=model_family,
            enable_thinking=_reasoning_enabled(str(request_policy["reasoning_effort"])),
            max_prompt_chars=int(config.max_prompt_chars),
        )
    )
    base_kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": controlled_messages,
        **{
            name: request_policy[name]
            for name in ("temperature", "top_p", "presence_penalty", "frequency_penalty")
            if name in request_policy
        },
        "max_tokens": request_policy["max_tokens"],
    }
    if config.runtime_response_format is not None:
        base_kwargs["response_format"] = dict(config.runtime_response_format)
    prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
    if prompt_chars > int(config.max_prompt_chars):
        raise ValueError(
            "Stage 2 rendered prompt exceeds max_prompt_chars; the caller must "
            f"partition it losslessly before transport ({prompt_chars} > "
            f"{config.max_prompt_chars})"
        )
    LOGGER.info(
        "Stage 2 request kind=%s endpoint=%s model=%s prompt_chars=%s "
        "family=%s reasoning_effort=%s wire_reasoning_effort=%s "
        "max_tokens=%s sampling=%s request_id=%s",
        request_policy["request_kind"],
        config.endpoint,
        config.model,
        prompt_chars,
        model_family,
        request_policy["reasoning_effort"],
        wire_reasoning_effort,
        request_policy["max_tokens"],
        {name: request_policy[name] for name in SAMPLING_FIELDS if name in request_policy},
        request_audit.request_id(),
    )
    variants = _openai_request_variants(
        base_kwargs=base_kwargs,
        request_policy=request_policy,
        model_family=model_family,
    )
    streaming = request_policy["request_kind"] == "extraction" and config.extraction_stream
    if streaming:
        variants = [dict(variant, stream=True, stream_options={"include_usage": True})
                    for variant in variants]
    logical_deadline = (
        float(config.runtime_request_deadline)
        if config.runtime_request_deadline is not None
        else time.monotonic() + float(config.request_timeout)
    )
    attempt_budget = int(
        config.runtime_transport_attempt_budget
        if config.runtime_transport_attempt_budget is not None
        else config.transport_max_attempts
    )
    response = None
    attempts_used = 0
    for variant_index, kwargs in enumerate(variants, start=1):
        remaining = logical_deadline - time.monotonic()
        if remaining <= 0:
            raise Stage2RequestExhaustedError(
                "Stage 2 logical request deadline expired during compatibility negotiation"
            )
        if attempts_used >= attempt_budget:
            raise Stage2RequestExhaustedError(
                "Stage 2 transport attempt budget expired during compatibility negotiation"
            )
        client = OpenAI(
            base_url=config.endpoint,
            api_key=config.api_key,
            timeout=min(float(config.request_timeout), remaining),
            # Stage 2 owns completion retries so they are logged, bounded, and do
            # not multiply invisibly with SDK-level retries.
            max_retries=0,
        )
        attempts_used += 1
        attempt_started = time.monotonic()
        request_audit.event(
            "http_attempt_started", http_variant=variant_index,
            reasoning_effort=request_policy["reasoning_effort"],
            max_tokens=request_policy["max_tokens"], streaming=streaming,
            wire_reasoning_effort=kwargs.get("reasoning_effort"),
            sampling={name: kwargs.get(name, (kwargs.get("extra_body") or {}).get(name))
                      for name in SAMPLING_FIELDS},
            chat_template_kwargs=(kwargs.get("extra_body") or {}).get("chat_template_kwargs"),
            response_format=(kwargs.get("response_format") or {}).get("type"),
            sampling_provenance=request_policy["sampling_provenance"],
            read_timeout_seconds=min(float(config.request_timeout), remaining),
        )
        try:
            response = client.chat.completions.create(**kwargs)
            if streaming:
                response = request_audit.collect_stream(
                    response, deadline=logical_deadline,
                    deadline_error=Stage2RequestExhaustedError,
                    incomplete_error=_RetryableStage2ResponseError,
                    whitespace_error=_Stage2OutputWhitespaceError,
                )
            request_audit.event(
                "http_attempt_completed", provider_request_id=getattr(response, "id", None),
                duration_seconds=round(time.monotonic() - attempt_started, 3),
                finish_reason=getattr(response.choices[0], "finish_reason", None),
                **request_audit.usage_fields(getattr(response, "usage", None)),
            )
        except Exception as exc:
            request_audit.event("http_attempt_failed", error_type=type(exc).__name__,
                                error=str(exc)[:2000],
                                duration_seconds=round(time.monotonic() - attempt_started, 3))
            if isinstance(exc, _Stage2OutputWhitespaceError):
                # The server responded, but generation made no JSON progress.
                # Send this through response repair, not network retry/backoff.
                raise
            if (
                variant_index == len(variants)
                or not _openai_optional_parameter_error(exc)
            ):
                raise _Stage2TransportFailure(
                    exc,
                    attempts_used=attempts_used,
                ) from exc
            if attempts_used >= attempt_budget:
                raise Stage2RequestExhaustedError(
                    "Stage 2 transport exhausted its HTTP attempt budget while "
                    "negotiating optional request controls"
                ) from exc
            # Do not spend the entire attempt budget resending an explicitly
            # rejected sampling extension across thinking-control variants.
            normalized_error = re.sub(r"[_-]+", " ", str(exc).lower())
            if "stream options" in normalized_error or "include usage" in normalized_error:
                for remaining_variant in variants[variant_index:]:
                    remaining_variant.pop("stream_options", None)
            rejected_sampling = {
                name for name in ("top_k", "min_p", "repetition_penalty")
                if name.replace("_", " ") in normalized_error
            }
            if request_policy.get("preserve_thinking") and (rejected_sampling
                    or "reasoning effort" in normalized_error or "thinking" in normalized_error):
                raise _Stage2TransportFailure(exc, attempts_used=attempts_used) from exc
            for remaining_variant in variants[variant_index:]:
                body = remaining_variant.get("extra_body")
                if body is not None:
                    remaining_variant["extra_body"] = {
                        key: value for key, value in body.items()
                        if key not in rejected_sampling
                    }
            LOGGER.warning(
                "Stage 2 endpoint rejected optional request controls; trying "
                "compatibility variant %s/%s (%s: %s)",
                variant_index + 1,
                len(variants),
                type(exc).__name__,
                exc,
            )
            continue
        finally:
            client.close()
        if time.monotonic() > logical_deadline:
            raise Stage2RequestExhaustedError(
                "Stage 2 logical request deadline expired during a transport attempt"
            )
        break
    if response is None:  # pragma: no cover - loop either returns or raises
        raise RuntimeError("Stage 2 request compatibility negotiation failed")
    choice = response.choices[0]
    finish_reason = str(getattr(choice, "finish_reason", "") or "")
    if finish_reason in {"length", "max_tokens"}:
        raise _Stage2OutputLengthError(
            f"Stage 2 server stopped the response with finish_reason={finish_reason}"
        )
    content = _response_message_text(choice.message)
    if not content and _response_reasoning_text(choice.message):
        raise _Stage2ReasoningOnlyResponseError(
            "Stage 2 model returned reasoning but no final message content"
        )
    if not content:
        raise _Stage2TransportFailure(
            _RetryableStage2ResponseError("Stage 2 model returned an empty response"),
            attempts_used=attempts_used,
        )
    return content


class _RoundRobinOpenAICompletion:
    """Thread-safe request distribution over equivalent vLLM replicas."""

    def __init__(self, endpoints: Sequence[str]) -> None:
        self.endpoints = tuple(str(endpoint).rstrip("/") for endpoint in endpoints)
        if not self.endpoints:
            raise ValueError("Stage 2 completion routing requires at least one endpoint")
        self._next_index = 0
        self._lock = threading.Lock()

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        config: PlainHandoffStage2Config,
    ) -> str:
        with self._lock:
            endpoint = self.endpoints[self._next_index % len(self.endpoints)]
            self._next_index += 1
        return _openai_completion(messages, replace(config, endpoint=endpoint))


class _LoadAwareOpenAICompletion:
    """Route HTTP attempts to an available equivalent model server."""

    def __init__(self, endpoints: Sequence[ExtractionEndpoint], *, api_key: str,
                 latency_weighted: bool = True) -> None:
        self.pool = EndpointPool(endpoints, api_key=api_key, latency_weighted=latency_weighted)

    def __call__(self, messages: Sequence[Mapping[str, str]], config: PlainHandoffStage2Config) -> str:
        try:
            index = self.pool.reserve(deadline=config.runtime_request_deadline)
        except TimeoutError as error:
            raise Stage2RequestExhaustedError(str(error)) from error
        server = self.pool.server(index)
        started = time.monotonic()
        succeeded, transport_failed = False, False
        try:
            request_audit.update(endpoint=server.endpoint)
            request_audit.event("endpoint_selected", **self.pool.snapshot()[index])
            response = _openai_completion(messages, replace(config, endpoint=server.endpoint))
            succeeded = True
            return response
        except Exception as error:
            transport_failed = _is_retryable_transport_error(error)
            if transport_failed:
                request_audit.event("endpoint_transport_failed", endpoint=server.endpoint,
                                    error_type=type(error).__name__)
            raise
        finally:
            self.pool.release(index, duration_seconds=time.monotonic()-started,
                              succeeded=succeeded, transport_failed=transport_failed)


class _ConcurrencyLimitedCompletion:
    """Apply one global request ceiling across nested Stage 2 executors."""

    def __init__(self, completion: CompletionFunction, max_concurrency: int) -> None:
        self.completion = completion
        self._semaphore = threading.BoundedSemaphore(max(1, int(max_concurrency)))
        # Optional live admission policy, installed before starting the run.
        self.admission_gate: Callable[[], Any] | None = None

    @contextmanager
    def request_slot(self) -> Iterator[CompletionFunction]:
        queued_at = time.monotonic()
        gate = self.admission_gate() if self.admission_gate is not None else nullcontext()
        with gate, self._semaphore:
            waited = time.monotonic() - queued_at
            if waited >= 1.0:
                LOGGER.info(
                    "Stage 2 request admitted after %.1fs waiting for a local slot; "
                    "queue time is excluded from the logical request deadline",
                    waited,
                )
            yield self.completion

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        config: PlainHandoffStage2Config,
    ) -> str:
        with self.request_slot() as completion:
            return completion(messages, config)


class _DisabledExtractionCompletion:
    """Fail closed when a post-extraction run unexpectedly requests extraction."""

    def __call__(
        self,
        _messages: Sequence[Mapping[str, str]],
        _config: PlainHandoffStage2Config,
    ) -> str:
        raise RuntimeError(
            "Stage 2 attempted a new patient extraction while "
            "stage2.runtime_disable_extraction=true; the post-extraction "
            "reselection run must use its frozen measurement caches"
        )


class _InterruptibleCompletion:
    """Cooperatively stop same-role calls after another role is requested."""

    def __init__(
        self,
        completion: CompletionFunction,
        *,
        required_role: str,
        switch_event: threading.Event,
    ) -> None:
        self.completion = completion
        self.required_role = str(required_role)
        self.switch_event = switch_event
        self.uses_default_transport = isinstance(
            completion,
            (_RoundRobinOpenAICompletion, _LoadAwareOpenAICompletion),
        )

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        config: PlainHandoffStage2Config,
    ) -> str:
        if self.switch_event.is_set():
            raise _ManagedStage2ModelSwitch(self.required_role)
        return self.completion(messages, config)


def _uses_default_openai_transport(
    completion: CompletionFunction | None,
) -> bool:
    return bool(
        completion is None
        or isinstance(completion, (_RoundRobinOpenAICompletion, _LoadAwareOpenAICompletion))
        or (
            isinstance(completion, _InterruptibleCompletion)
            and completion.uses_default_transport
        )
    )


def _is_retryable_transport_error(exc: Exception) -> bool:
    """Return whether a failed OpenAI-compatible request is safe to retry."""

    if isinstance(exc, _Stage2TransportFailure):
        exc = exc.cause
    if isinstance(exc, _RetryableStage2ResponseError):
        return True
    try:
        import httpx
        from openai import APIConnectionError, APIStatusError
    except ImportError:  # pragma: no cover - OpenAI is required for live requests
        return False
    if isinstance(exc, APIConnectionError):
        return True
    # The SDK translates errors while opening a response, but iteration over
    # an already-open SSE response can expose the underlying HTTPX exception.
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = int(exc.status_code)
        return status_code in {408, 409, 429} or status_code >= 500
    return False


def _completion_with_transport_retries(
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
    completion: CompletionFunction,
    *,
    deadline: float | None = None,
    prompt_token_counter: Callable[[Sequence[Mapping[str, str]]], int] | None = None,
    context_window_tokens: int | None = None,
    context_margin_tokens: int = 0,
) -> str:
    logical_deadline = (
        float(deadline)
        if deadline is not None
        else time.monotonic() + float(config.request_timeout)
    )
    max_attempts = max(1, int(config.transport_max_attempts))
    remaining_attempts = max_attempts
    # Every transport attempt resends the caller's messages unchanged.  A
    # checkpointed measurement is fingerprinted by those messages, so failure
    # text must never be appended here (it is logged and audited instead).
    retry_messages = [dict(message) for message in messages]

    def exhausted(reason: str, cause: Exception | None = None) -> Stage2RequestExhaustedError:
        detail = (
            f"{reason}; kind={config.runtime_request_kind} endpoint={config.endpoint} "
            f"model={config.model} attempts_used={max_attempts - remaining_attempts} "
            f"attempt_timeout={config.request_attempt_timeout:g}s "
            f"request_timeout={config.request_timeout:g}s"
        )
        if cause is not None:
            detail += f"; last_error={type(cause).__name__}: {cause}"
        return Stage2RequestExhaustedError(detail)

    while remaining_attempts > 0:
        remaining = logical_deadline - time.monotonic()
        if remaining <= 0:
            raise exhausted(
                "Stage 2 logical request deadline expired before a transport attempt"
            )
        attempt_config = replace(
            config,
            request_timeout=min(float(config.request_attempt_timeout), remaining),
            runtime_request_deadline=logical_deadline,
            runtime_transport_attempt_budget=remaining_attempts,
        )
        request_audit.update(transport_attempt=max_attempts - remaining_attempts + 1)
        if prompt_token_counter is not None and context_window_tokens is not None:
            available = (
                int(context_window_tokens)
                - int(prompt_token_counter(retry_messages))
                - int(context_margin_tokens)
            )
            if available < 1:
                raise ValueError("Stage 2 transport retry prompt leaves no model output context")
            attempt_config = replace(
                attempt_config,
                max_tokens=min(attempt_config.max_tokens, available),
                extraction_max_tokens=min(attempt_config.extraction_max_tokens, available),
                extraction_reasoning_max_tokens=(
                    min(attempt_config.extraction_reasoning_max_tokens, available)
                    if attempt_config.extraction_reasoning_max_tokens is not None else None
                ),
            )
        try:
            return completion(retry_messages, attempt_config)
        except Exception as exc:
            cause = exc.cause if isinstance(exc, _Stage2TransportFailure) else exc
            attempts_used = (
                exc.attempts_used if isinstance(exc, _Stage2TransportFailure) else 1
            )
            remaining_attempts -= min(remaining_attempts, attempts_used)
            if not _is_retryable_transport_error(cause):
                raise cause
            if remaining_attempts <= 0:
                raise exhausted(
                    f"Stage 2 transport exhausted {max_attempts} attempt(s)", cause,
                ) from cause
            remaining = logical_deadline - time.monotonic()
            if remaining <= 0:
                raise exhausted(
                    "Stage 2 logical request deadline expired after a transport failure", cause,
                ) from cause
            consumed_attempts = max_attempts - remaining_attempts
            delay = float(config.transport_retry_backoff) * (
                2 ** max(0, consumed_attempts - 1)
            )
            if delay >= remaining:
                raise exhausted(
                    "Stage 2 logical request deadline would expire during retry backoff", cause,
                ) from cause
            LOGGER.warning(
                "Stage 2 transport failed; retrying request attempt %s/%s "
                "after %.1fs kind=%s endpoint=%s model=%s "
                "attempt_timeout=%.1fs remaining=%.1fs (%s: %s)",
                consumed_attempts + 1,
                max_attempts,
                delay,
                config.runtime_request_kind,
                config.endpoint,
                config.model,
                attempt_config.request_timeout,
                remaining,
                type(cause).__name__,
                cause,
            )
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError("unreachable Stage 2 transport retry state")


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        direct = json.loads(stripped)
    except json.JSONDecodeError:
        direct = None
    else:
        if isinstance(direct, dict):
            return direct
        raise ValueError("Stage 2 response must be one JSON object, not an array or scalar")

    # Inline Qwen/LFM <think> blocks and Gemma <|channel>thought channels may
    # surround the final object. Scanning with JSONDecoder is safe around braces
    # inside valid JSON strings and also handles an unclosed reasoning prefix.
    decoder = json.JSONDecoder()
    decoded: list[tuple[int, int, dict[str, Any]]] = []
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            decoded.append((start + consumed, start, value))
    if decoded:
        # Prefer the object ending latest in the response; when nested objects
        # share an end position, prefer the outermost (earliest-starting) one.
        _end, _start, value = max(decoded, key=lambda item: (item[0], -item[1]))
        return value
    raise ValueError(
        "Stage 2 response must contain one final JSON object after any reasoning content"
    )


def _compact_json_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Losslessly reclaim prompt space from JSON message bodies for repairs."""

    compacted: list[dict[str, Any]] = []
    for message in messages:
        row = dict(message)
        content = row.get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                rendered = json.dumps(
                    parsed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            else:
                if len(rendered) < len(content):
                    row["content"] = rendered
        compacted.append(row)
    return compacted


def _repair_message(exc: Exception, *, repair_context: Mapping[str, Any] | None = None, repeated_error_count: int = 1) -> dict[str, str]:
    del repeated_error_count
    if isinstance(exc, _Stage2OutputWhitespaceError):
        content = (
            f"Your preceding response was interrupted: {exc}. "
            "Return the complete JSON object without repeated formatting whitespace. "
            "Numerical category labels may be JSON numbers or quoted strings; "
            "use the declared values and null only as required by the measurement definition."
        )
    elif isinstance(exc, _PageObservationValidationError):
        content = (
            f"Your preceding observation list needs this correction: {exc}. "
            "Return the complete observations array with each supported, nonmissing "
            "occurrence and its supporting quote. Preserve valid repeated occurrences. "
            "Omit null or unavailable placeholders instead of recording them as observations. "
            'If no supported values occur in the section, return {"observations": []}.'
        )
    elif isinstance(exc, _Stage2ReasoningOnlyResponseError):
        content = (
            "Your preceding response contained reasoning but no final answer. "
            "Return the complete JSON object as the final message content, after any "
            "reasoning, using the task and information above."
        )
    elif isinstance(exc, _Stage2OutputLengthError):
        content = (
            "Your preceding answer ended before the JSON object was complete. "
            "Return a complete JSON object using the task and information above. "
            "Include every requested entry and field. Keep explanations concise."
        )
    else:
        content = (
            "Your preceding answer needs this correction: "
            + type(exc).__name__ + ": " + str(exc) + "\n"
            "Use the task and information above to return the complete corrected JSON object. "
            "Preserve supported values and use null when the definition requires it."
        )
    instructions = (repair_context or {}).get("response_instructions")
    if isinstance(instructions, str) and instructions:
        content += "\n\n" + instructions
    return {"role": "user", "content": content}



def _bounded_repair_directive(
    exc: Exception,
    *,
    max_chars: int,
    repair_context: Mapping[str, Any] | None = None,
    repeated_error_count: int = 1,
) -> str:
    """Keep the validation failure visible even in a packed repair prompt."""

    if max_chars < 16:
        raise ValueError("Stage 2 repair prompt has no room for its validation error") from exc
    message = _repair_message(
        exc,
        repair_context=repair_context,
        repeated_error_count=repeated_error_count,
    )["content"]
    if len(message) <= max_chars:
        return message
    detail = f"{type(exc).__name__}: {exc}"
    if len(detail) <= max_chars:
        return detail
    return detail[: max_chars - 3].rstrip() + "..."


@request_audit.audited
def _request_json(
    *,
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
    completion: CompletionFunction,
    validate: Callable[[Mapping[str, Any]], dict[str, Any]],
    request_kind: str = "interpretation",
    initial_reasoning_effort: str | None = None,
    prompt_token_counter: Callable[[Sequence[Mapping[str, str]]], int] | None = None,
    context_window_tokens: int | None = None,
    context_margin_tokens: int = 0,
    repair_context: Mapping[str, Any] | None = None,
    validation_event_observer: Callable[[Mapping[str, Any]], None] | None = None,
    conservative_validation_fallback: Mapping[str, Any] | None = None,
    fallback_after_same_error: int = 3,
) -> dict[str, Any]:
    if isinstance(completion, _ConcurrencyLimitedCompletion):
        # Admit the entire logical request before starting its deadline. Keep
        # the slot through transport retries and semantic repairs so neither
        # can be starved by other folds while its deadline keeps running.
        with completion.request_slot() as admitted_completion:
            return _request_json(
                messages=messages,
                config=config,
                completion=admitted_completion,
                validate=validate,
                request_kind=request_kind,
                initial_reasoning_effort=initial_reasoning_effort,
                prompt_token_counter=prompt_token_counter,
                context_window_tokens=context_window_tokens,
                context_margin_tokens=context_margin_tokens,
                repair_context=repair_context,
                validation_event_observer=validation_event_observer,
                conservative_validation_fallback=conservative_validation_fallback,
                fallback_after_same_error=fallback_after_same_error,
            )
    if (
        request_kind == "interpretation"
        and prompt_token_counter is None
        and _uses_default_openai_transport(completion)
    ):
        primary_window = _primary_context_window_tokens(config)
        if primary_window is not None and (
            config.vllm is not None
            or _external_primary_tokenizer_available(config.runtime_sampling_model or config.model)
        ):
            prompt_token_counter = _primary_prompt_token_counter(config)
            context_window_tokens = primary_window
            context_margin_tokens = max(context_margin_tokens, 4096)
    if initial_reasoning_effort is not None and (
        not isinstance(initial_reasoning_effort, str)
        or initial_reasoning_effort not in SUPPORTED_REASONING_EFFORTS
    ):
        raise ValueError("unsupported initial_reasoning_effort")
    request_policy = _stage2_request_policy(
        replace(config, runtime_reasoning_effort=initial_reasoning_effort)
        if initial_reasoning_effort is not None else config,
        request_kind,
    )
    request_config = replace(
        config,
        runtime_request_kind=str(request_policy["request_kind"]),
        runtime_reasoning_effort=initial_reasoning_effort,
        runtime_response_format=getattr(validate, "response_format", None),
    )
    response_instructions = getattr(validate, "repair_instructions", None)
    if isinstance(response_instructions, str) and response_instructions:
        repair_context = {**(repair_context or {}), "response_instructions": response_instructions}
    base_conversation = [dict(message) for message in messages]
    conversation = [dict(message) for message in base_conversation]
    first_error: Exception | None = None
    max_repairs = int(config.max_response_repairs)
    max_attempts = 1 + max_repairs
    logical_deadline = time.monotonic() + float(config.request_timeout)
    request_audit.event("request_admitted", logical_budget_seconds=config.request_timeout)
    validation_error_counts: Counter[str] = Counter()
    field_error_counts: Counter[str] = Counter()
    validation_failure_count = 0
    if fallback_after_same_error < 1:
        raise ValueError("fallback_after_same_error must be a positive integer")

    def token_window_fits(candidate: Sequence[Mapping[str, str]]) -> bool:
        if prompt_token_counter is None or context_window_tokens is None:
            return True
        return (
            int(prompt_token_counter(candidate)) + int(context_margin_tokens)
            < int(context_window_tokens)
        )

    for attempt in range(max_attempts):
        request_audit.update(response_attempt=attempt + 1, transport_attempt=0)
        if time.monotonic() >= logical_deadline:
            raise Stage2RequestExhaustedError(
                "Stage 2 logical request deadline expired across response repairs"
            )
        response: str | None = None
        parsed_response: dict[str, Any] | None = None
        prompt_chars = sum(len(str(message.get("content") or "")) for message in conversation)
        if prompt_chars > int(config.max_prompt_chars):
            raise ValueError(
                "Stage 2 rendered prompt exceeds max_prompt_chars before transport "
                f"({prompt_chars} > {config.max_prompt_chars})"
            )
        try:
            attempt_config = request_config
            if attempt > int(config.thinking_after_response_repairs):
                attempt_config = replace(
                    request_config,
                    runtime_reasoning_effort=_thinking_response_repair_effort(
                        str(request_policy["reasoning_effort"])
                    ),
                )
                if attempt == int(config.thinking_after_response_repairs) + 1:
                    LOGGER.warning(
                        "Stage 2 response repair attempt %s/%s enables thinking "
                        "with reasoning_effort=%s",
                        attempt,
                        max_repairs,
                        attempt_config.runtime_reasoning_effort,
                    )
            if prompt_token_counter is not None and context_window_tokens is not None:
                prompt_tokens = int(prompt_token_counter(conversation))
                available_output_tokens = (
                    int(context_window_tokens)
                    - prompt_tokens
                    - int(context_margin_tokens)
                )
                if available_output_tokens < 1:
                    raise ValueError(
                        "Stage 2 repair prompt leaves no model output context: "
                        f"prompt_tokens={prompt_tokens}, "
                        f"context_window_tokens={context_window_tokens}, "
                        f"context_margin_tokens={context_margin_tokens}"
                    )
                dynamic_output_ceiling = min(
                    int(_stage2_request_policy(attempt_config)["max_tokens"]),
                    available_output_tokens,
                )
                request_audit.event(
                    "context_budget", prompt_tokens=prompt_tokens,
                    context_window_tokens=context_window_tokens,
                    context_margin_tokens=context_margin_tokens,
                    max_tokens=dynamic_output_ceiling,
                )
                if dynamic_output_ceiling < int(_stage2_request_policy(attempt_config)["max_tokens"]):
                    LOGGER.info(
                        "Stage 2 context budget kind=%s prompt_tokens=%s max_tokens=%s "
                        "context_window_tokens=%s margin_tokens=%s",
                        request_kind, prompt_tokens, dynamic_output_ceiling,
                        context_window_tokens, context_margin_tokens,
                    )
                if request_policy["request_kind"] == "extraction":
                    reasoning_ceiling = (
                        _reasoning_enabled(_stage2_request_policy(attempt_config)["reasoning_effort"])
                        and attempt_config.extraction_reasoning_max_tokens is not None
                    )
                    attempt_config = replace(
                        attempt_config,
                        **{("extraction_reasoning_max_tokens" if reasoning_ceiling else
                            "extraction_max_tokens"): dynamic_output_ceiling},
                    )
                else:
                    attempt_config = replace(
                        attempt_config,
                        max_tokens=dynamic_output_ceiling,
                    )
            response = _completion_with_transport_retries(
                conversation,
                attempt_config,
                completion,
                deadline=logical_deadline,
                prompt_token_counter=prompt_token_counter,
                context_window_tokens=context_window_tokens,
                context_margin_tokens=context_margin_tokens,
            )
            parsed_response = _parse_json_object(response)
            return validate(parsed_response)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            # Only a returned (or length-truncated) model response is eligible
            # for semantic repair/fallback. Local prompt-budget, configuration,
            # and transport exceptions must abort instead of becoming evidence.
            if response is None and not isinstance(
                exc, (_Stage2OutputLengthError, _Stage2ReasoningOnlyResponseError)
            ):
                raise
            validation_failure_count += 1
            request_audit.event("response_validation_failed", error_type=type(exc).__name__,
                                error=str(exc)[:2000])
            error_signature = f"{type(exc).__name__}: {exc}"
            validation_error_counts[error_signature] += 1
            repeated_error_count = validation_error_counts[error_signature]
            if validation_event_observer is not None:
                validation_event_observer(
                    {
                        "event": "invalid_response",
                        "response_attempt": attempt + 1,
                        "validation_failure_count": validation_failure_count,
                        "same_error_occurrence": repeated_error_count,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "raw_response": response,
                        "parsed_response": parsed_response,
                    }
                )
            if request_kind == "extraction" and isinstance(exc, _ExtractionFieldError):
                field_error_counts.update(exc.feature_names)
                exhausted_fields = [name for name in exc.feature_names
                    if field_error_counts[name] > EXTRACTION_FIELD_REPAIR_LIMIT]
                if exhausted_fields:
                    # Leave the logical request and release its admission slot.
                    # The extraction caller will rebuild a smaller fresh prompt.
                    error = _ExtractionFieldError(str(exc), feature_names=exhausted_fields)
                    raise Stage2ResponseValidationError(
                        f"Stage 2 field validation remained invalid after "
                        f"{EXTRACTION_FIELD_REPAIR_LIMIT} repairs: {exc}"
                    ) from error
            if (request_kind == "extraction"
                    and isinstance(exc, (_ExtractionCategoryError, _ExtractionValueError,
                                         _PageObservationValidationError))
                    and attempt > int(config.thinking_after_response_repairs)
                    and (repeated_error_count >= 2 or isinstance(exc, _PageObservationValidationError))):
                # Five non-thinking repairs and one thinking repair have already
                # been tried under the default policy. Preserve the returned
                # good fields/observations and let the caller audit the failures,
                # rather than rereading the whole record for an unchanged error.
                request_audit.event("extraction_semantic_repair_handoff",
                                    error_type=type(exc).__name__, error=str(exc)[:2000])
                raise Stage2ResponseValidationError(
                    f"Repeated extraction contract failure after thinking repair: {exc}"
                ) from exc
            fallback_trigger: str | None = None
            if conservative_validation_fallback is not None:
                if repeated_error_count >= int(fallback_after_same_error):
                    fallback_trigger = "repeated_identical_validation_error"
                elif attempt == max_attempts - 1:
                    fallback_trigger = "validation_repairs_exhausted"
            if fallback_trigger is not None:
                fallback_value = validate(dict(conservative_validation_fallback))
                LOGGER.warning(
                    "Stage 2 uses conservative validation fallback after %s failed "
                    "response(s), trigger=%s last_error=%s",
                    validation_failure_count,
                    fallback_trigger,
                    error_signature,
                )
                if validation_event_observer is not None:
                    validation_event_observer(
                        {
                            "event": "conservative_fallback",
                            "response_attempt": attempt + 1,
                            "validation_failure_count": validation_failure_count,
                            "same_error_occurrence": repeated_error_count,
                            "trigger": fallback_trigger,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "fallback_response": fallback_value,
                        }
                    )
                return fallback_value
            if attempt == max_attempts - 1:
                raise Stage2ResponseValidationError(
                    f"Stage 2 response remained invalid after {max_attempts - 1} repairs: {exc}"
                ) from exc
            first_error = exc
            LOGGER.warning(
                "Stage 2 response failed validation; repair attempt %s/%s (%s: %s)",
                attempt + 1,
                max_repairs,
                type(exc).__name__,
                exc,
            )
            repair_message = _repair_message(
                exc,
                repair_context=repair_context,
                repeated_error_count=repeated_error_count,
            )
            response_context = [dict(message) for message in base_conversation]
            if response is not None:
                response_context.append({"role": "assistant", "content": str(response)})
            for candidate_context in (response_context, base_conversation):
                repaired = [*candidate_context, repair_message]
                repaired_chars = sum(len(str(message.get("content") or "")) for message in repaired)
                if (
                    repaired_chars <= int(config.max_prompt_chars)
                    and token_window_fits(repaired)
                ):
                    conversation = repaired
                    break
            else:
                repaired = []
            if repaired:
                continue

            # A fully packed initial prompt may leave no room for another turn.
            # Minify JSON bodies without changing their content, then retry with
            # the same explicit validation error.
            compact_context = _compact_json_messages(response_context)
            compact_repaired = [*compact_context, repair_message]
            if (
                sum(len(str(message.get("content") or "")) for message in compact_repaired)
                <= int(config.max_prompt_chars)
                and token_window_fits(compact_repaired)
            ):
                conversation = compact_repaired
                continue
            compact_base = _compact_json_messages(base_conversation)
            compact_repaired = [*compact_base, repair_message]
            if (
                sum(len(str(message.get("content") or "")) for message in compact_repaired)
                <= int(config.max_prompt_chars)
                and token_window_fits(compact_repaired)
            ):
                conversation = compact_repaired
                continue

            # If lossless compaction is still insufficient, spend the system
            # message's full character budget on the concrete validation error.
            # The complete original user payload remains present.
            system_index = next(
                (
                    index
                    for index, message in enumerate(compact_base)
                    if str(message.get("role") or "") == "system"
                ),
                None,
            )
            if system_index is None:
                raise ValueError(
                    "Stage 2 repair prompt cannot fit max_prompt_chars and has no "
                    "system instruction available to replace"
                ) from exc
            conversation = [dict(message) for message in compact_base]
            non_system_chars = sum(
                len(str(message.get("content") or ""))
                for index, message in enumerate(conversation)
                if index != system_index
            )
            available = int(config.max_prompt_chars) - non_system_chars
            conversation[system_index]["content"] = _bounded_repair_directive(
                exc,
                max_chars=available,
                repair_context=repair_context,
                repeated_error_count=repeated_error_count,
            )
    raise RuntimeError(f"unreachable Stage 2 response state: {first_error}")


def _checkpointed_request_json(
    *,
    output_dir: Path | None,
    input_value: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
    config: PlainHandoffStage2Config,
    completion: CompletionFunction,
    validate: Callable[[Mapping[str, Any]], dict[str, Any]],
    validation_fallback: (
        Callable[[Stage2ResponseValidationError], Mapping[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Cache one validated LLM leaf by its complete deterministic input."""

    def request_with_fallback() -> tuple[dict[str, Any], Stage2ResponseValidationError | None]:
        try:
            return (
                _request_json(
                    messages=messages,
                    config=config,
                    completion=completion,
                    validate=validate,
                ),
                None,
            )
        except Stage2ResponseValidationError as exc:
            if validation_fallback is None:
                raise
            result = validate(validation_fallback(exc))
            LOGGER.warning(
                "Stage 2 response remained invalid; using conservative validated fallback (%s)",
                exc,
            )
            return result, exc

    if output_dir is None:
        result, _fallback_error = request_with_fallback()
        return result
    output_dir = Path(output_dir)
    checkpoint_input = {
        **dict(input_value),
        "consolidation_schema": CONSOLIDATION_SCHEMA_VERSION,
        "messages": list(messages),
        "prompt_catalog_version": clinical_prompts.PROMPT_VERSION,
        "llm_identity": {
            "model": config.model,
        },
        "request_policy": _stage2_request_policy(config, "interpretation"),
    }
    input_fingerprint = _value_fingerprint(checkpoint_input)
    input_path = output_dir / "input.json"
    result_path = output_dir / "result.json"
    complete_path = output_dir / "complete.json"
    cached_files = _read_checkpoint_jsons([input_path, complete_path, result_path])
    if cached_files is not None:
        try:
            previous_input, completion_state, cached_result = cached_files
            if (
                previous_input.get("input_fingerprint") == input_fingerprint
                and completion_state.get("input_fingerprint") == input_fingerprint
            ):
                validated = validate(cached_result)
                LOGGER.info("skip completed Stage 2 consolidation request: %s", output_dir)
                return validated
        except (KeyError, TypeError, ValueError):
            pass
        LOGGER.info("rerun stale or inconsistent Stage 2 consolidation request: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        input_path,
        {
            **checkpoint_input,
            "input_fingerprint": input_fingerprint,
        },
    )
    result, fallback_error = request_with_fallback()
    _write_json(result_path, result)
    if fallback_error is not None:
        _write_json(
            output_dir / "fallback.json",
            {
                "status": "conservative_validation_fallback",
                "completed_at": _now(),
                "validation_error": str(fallback_error),
            },
        )
    _write_json(
        complete_path,
        {
            "status": (
                "complete_with_validation_fallback" if fallback_error is not None else "complete"
            ),
            "completed_at": _now(),
            "input_fingerprint": input_fingerprint,
            "validation_fallback": fallback_error is not None,
        },
    )
    return result


def _interpretation_prompt(
    *,
    architecture: str,
    packets: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Render one card without asking the model to track source identifiers."""

    del architecture
    if len(packets) != 1:
        raise ValueError("discovery requires exactly one evidence card per request")
    texts = _readable_supporting_text(packets)
    if not texts:
        raise ValueError("discovery evidence card has no readable text")
    excerpts = "\n".join(f"<excerpt>\n{text}\n</excerpt>" for text in texts)
    return [
        {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Identify the clinical variables supported by these excerpts using the "
            "instructions above.\n\n<clinical_excerpts>\n"
            f"{excerpts}\n</clinical_excerpts>"
        )},
    ]


def _validate_discovery(
    value: Mapping[str, Any],
    *,
    packet_id: str,
    evidence_axes: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate clinical content; attach trusted call provenance in Python."""

    if not isinstance(value, Mapping) or set(value) != {"candidates"}:
        raise ValueError("discovery requires one object containing only candidates")
    rows = value["candidates"]
    if not isinstance(rows, list):
        raise ValueError("discovery candidates must be an array")
    fields = {"name", "description", "basis", "uncertainty"}
    cleaned = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError("each discovery candidate requires name, description, basis, uncertainty only")
        if any(not isinstance(row[field], str) for field in fields):
            raise ValueError("discovery candidate fields must be text")
        candidate = {field: row[field].strip() for field in fields}
        if any(not candidate[field] for field in ("name", "description", "basis")):
            raise ValueError("discovery name, description, and basis must be nonempty")
        base = _snake_case_name(candidate["name"], fallback="")
        if not base:
            raise ValueError("discovery name must contain a usable clinical label")
        cleaned.append((base, candidate))

    # Avoid silently merging different measurements whose labels normalize to
    # the same spelling. Stable ordering also makes suffixes independent of the
    # order in which the model returned those candidates.
    cleaned.sort(key=lambda pair: (pair[0], *(pair[1][field] for field in sorted(fields))))
    names: set[str] = set()
    concepts = []
    for base, candidate in cleaned:
        name, suffix = base, 2
        while name in names:
            name, suffix = f"{base}_{suffix}", suffix + 1
        names.add(name)
        concepts.append({
            "name": name,
            "description": candidate["description"],
            "supporting_packet_ids": [str(packet_id)],
            "evidence_axes": _canonical_evidence_axes(evidence_axes),
            "evidence_rationale": candidate["basis"],
            "caveats": candidate["uncertainty"],
        })
    return {
        "concepts": concepts,
        "packet_dispositions": {
            str(packet_id): {
                "status": "supports_concept" if concepts else "reviewed_no_specific_concept",
                "concept_names": sorted(names),
                "reason": "Derived from the single-card discovery response; provenance attached by Python.",
            }
        },
    }


def _rejected_packet_audit_prompt(*, architecture: str, packets: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    del architecture
    if len(packets) != 1:
        raise ValueError("recall review requires one evidence card per call")
    text = "\n\n".join(_readable_supporting_text(packets))
    return clinical_prompts.messages("02_audit_unmapped", "Clinical excerpts\n\n" + text)



def _cached_interpretation_matches_packets(
    value: Any,
    *,
    packet_ids: set[str],
) -> bool:
    """Return whether a normalized checkpoint cites exactly its current inputs."""

    if not isinstance(value, Mapping):
        return False
    concepts = value.get("concepts")
    dispositions = value.get("packet_dispositions")
    if not isinstance(concepts, list) or not isinstance(dispositions, Mapping):
        return False
    if {str(packet_id) for packet_id in dispositions} != packet_ids:
        return False
    for concept in concepts:
        if not isinstance(concept, Mapping):
            return False
        supports = concept.get("supporting_packet_ids")
        if not isinstance(supports, list) or not supports:
            return False
        if not {str(packet_id) for packet_id in supports} <= packet_ids:
            return False
        if not str(concept.get("evidence_rationale") or "").strip():
            return False
    return True


def _partition_interpretation_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    architecture: str,
    max_prompt_chars: int,
) -> list[list[Mapping[str, Any]]]:
    """Keep each card in its own call and check its complete prompt budget."""

    batches = []
    for packet in packets:
        messages = _interpretation_prompt(architecture=architecture, packets=[packet])
        if sum(len(message["content"]) for message in messages) > int(max_prompt_chars):
            raise ValueError("one Stage 2 evidence card cannot fit the rendered prompt budget")
        batches.append([packet])
    return batches


def _merge_interpretation_audit(
    *,
    packet_ids: set[str],
    initial: Mapping[str, Any],
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge grounded audit recoveries into an initial interpretation result."""

    concepts = [
        dict(concept)
        for result in [initial, *audits]
        for concept in list(result.get("concepts") or [])
    ]
    initial_dispositions = dict(initial.get("packet_dispositions") or {})
    latest_dispositions = dict(initial_dispositions)
    for audit in audits:
        latest_dispositions.update(dict(audit.get("packet_dispositions") or {}))

    dispositions: dict[str, Any] = {}
    for packet_id in sorted(packet_ids):
        names = sorted(
            str(concept["name"])
            for concept in concepts
            if packet_id in set(map(str, concept.get("supporting_packet_ids") or []))
        )
        source = latest_dispositions.get(packet_id)
        reason = str(source.get("reason") or "") if isinstance(source, Mapping) else ""
        dispositions[packet_id] = {
            "status": "supports_concept" if names else "reviewed_no_specific_concept",
            "concept_names": names,
            "reason": reason
            or (
                "Recovered by the rejected-packet audit."
                if names
                else "No interpretation pass recovered a defensible clinical feature."
            ),
        }

    initially_rejected = {
        str(packet_id)
        for packet_id, disposition in initial_dispositions.items()
        if isinstance(disposition, Mapping)
        and disposition.get("status") == "reviewed_no_specific_concept"
    }
    recovered = sorted(
        packet_id
        for packet_id in initially_rejected
        if dispositions.get(packet_id, {}).get("status") == "supports_concept"
    )
    remaining = sorted(initially_rejected - set(recovered))
    return {
        "concepts": concepts,
        "packet_dispositions": dispositions,
        "rejected_packet_audit": {
            "schema_version": INTERPRETATION_AUDIT_SCHEMA_VERSION,
            "initially_rejected_packet_ids": sorted(initially_rejected),
            "recovered_packet_ids": recovered,
            "remaining_rejected_packet_ids": remaining,
            "audit_batches": len(audits),
        },
    }


def _string_values(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, int, float, bool)):
        return [str(raw)]
    if not isinstance(raw, Sequence):
        return []
    return list(dict.fromkeys(str(item) for item in raw if item is not None))


def _short_text(value: Any, *, max_chars: int) -> str:
    rendered = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 3].rstrip() + "..."


def _snake_case_name(value: Any, *, fallback: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return name or fallback


def _candidate_architectures(candidate: Mapping[str, Any]) -> list[str]:
    primary = str(candidate.get("architecture") or "").strip()
    inherited = _string_values(candidate.get("supporting_architectures"))
    if primary in {
        "deterministic_candidate_group",
        "bounded_multi_architecture_consolidation",
    }:
        primary = ""
    if primary == CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE:
        primary = ""
    return list(
        dict.fromkeys(
            architecture
            for architecture in [
                primary,
                *inherited,
            ]
            if architecture
        )
    )


def _configured_feature_definitions(
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw = candidate.get("configured_feature_definitions") or []
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    definitions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, Mapping):
            continue
        definition = dict(value)
        name = _snake_case_name(definition.get("name"), fallback="")
        if not name or name in seen:
            continue
        definition["name"] = name
        definitions.append(definition)
        seen.add(name)
    return definitions


def _flatten_member_measurements(
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Preserve original candidate views through repeated consolidation rounds."""

    flattened: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for member in members:
        inherited = member.get("member_measurements")
        raw_views = (
            list(inherited)
            if isinstance(inherited, Sequence)
            and not isinstance(inherited, (str, bytes, bytearray))
            and inherited
            else [member]
        )
        for raw_view in raw_views:
            if not isinstance(raw_view, Mapping):
                continue
            view = {
                "name": str(raw_view.get("name") or "").strip(),
                "description": _short_text(raw_view.get("description"), max_chars=500),
                "evidence_rationale": _short_text(
                    raw_view.get("evidence_rationale"),
                    max_chars=700,
                ),
                "value_type": str(raw_view.get("value_type") or "ambiguous"),
            }
            identity = (
                view["name"],
                view["description"],
                view["evidence_rationale"],
                view["value_type"],
            )
            if identity in seen:
                continue
            flattened.append(view)
            seen.add(identity)
    return flattened


def _materialize_candidate_group(
    *,
    candidate_id: str,
    members: Sequence[Mapping[str, Any]],
    canonical_name: str,
    canonical_description: str,
    ontology_packet_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    packets = list(
        dict.fromkeys(
            packet_id
            for member in members
            for packet_id in _string_values(member.get("supporting_packet_ids"))
        )
    )
    architectures = list(
        dict.fromkeys(
            architecture for member in members for architecture in _candidate_architectures(member)
        )
    )
    evidence_axes = sorted(
        {
            axis
            for member in members
            for axis in _canonical_evidence_axes(member.get("evidence_axes"))
        }
    )
    packet_evidence_axes: dict[str, list[str]] = {}
    for member in members:
        inherited = member.get("packet_evidence_axes") or {}
        member_axes = _canonical_evidence_axes(member.get("evidence_axes"))
        for packet_id in _string_values(member.get("supporting_packet_ids")):
            packet_evidence_axes[packet_id] = sorted(
                {
                    *packet_evidence_axes.get(packet_id, []),
                    *_canonical_evidence_axes(inherited.get(packet_id) or member_axes),
                }
            )
    origins = list(
        dict.fromkeys(
            origin
            for member in members
            for origin in (
                _string_values(member.get("origin_candidate_ids")) or [str(member["candidate_id"])]
            )
        )
    )
    value_types = list(
        dict.fromkeys(str(member.get("value_type") or "ambiguous") for member in members)
    )
    descriptions = [
        str(member.get("description") or "").strip()
        for member in members
        if str(member.get("description") or "").strip()
    ]
    caveats = list(
        dict.fromkeys(
            str(member.get("caveats") or "").strip()
            for member in members
            if str(member.get("caveats") or "").strip()
        )
    )
    configured_definitions: list[dict[str, Any]] = []
    configured_names: set[str] = set()
    for member in members:
        for definition in _configured_feature_definitions(member):
            configured_name = str(definition["name"])
            if configured_name in configured_names:
                continue
            configured_definitions.append(definition)
            configured_names.add(configured_name)
    description = canonical_description or (descriptions[0] if descriptions else canonical_name)
    if ontology_packet_ids is None:
        ontology_packet_ids = [
            packet_id
            for member in members
            for packet_id in _string_values(member.get("supporting_packet_ids"))
        ]
    return {
        "candidate_id": candidate_id,
        "architecture": "deterministic_candidate_group",
        "supporting_architectures": architectures,
        "name": canonical_name,
        "description": description,
        "value_type": value_types[0] if len(value_types) == 1 else "ambiguous",
        "supporting_packet_ids": packets,
        "evidence_axes": evidence_axes,
        "packet_evidence_axes": packet_evidence_axes,
        "caveats": " ".join(caveats),
        "origin_candidate_ids": origins,
        "configured_feature_definitions": configured_definitions,
        # Internal routing only. Python resolves these to readable supporting
        # text; the ontology model never receives packet structure or IDs.
        "ontology_packet_ids": list(dict.fromkeys(map(str, ontology_packet_ids))),
        "member_measurements": _flatten_member_measurements(members),
    }


def _canonical_cluster_member(
    members: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    name_counts = Counter(_snake_case_name(member.get("name"), fallback="") for member in members)
    return max(
        enumerate(members),
        key=lambda item: (
            bool(_configured_feature_definitions(item[1])),
            name_counts[_snake_case_name(item[1].get("name"), fallback="")],
            len(_string_values(item[1].get("origin_candidate_ids"))),
            len(_string_values(item[1].get("supporting_packet_ids"))),
            bool(str(item[1].get("description") or "").strip()),
            -item[0],
        ),
    )[1]


def _materialize_exact_name_groups(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Coalesce exact normalized names before iterative semantic consolidation.

    This is identity bookkeeping, not semantic consolidation: distinct names
    are never compared or merged here. Combining exact names keeps every batch
    response contract unambiguous while preserving candidate evidence and
    provenance.
    """

    members_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for index, candidate in enumerate(candidates, start=1):
        name = _snake_case_name(candidate.get("name"), fallback=f"measurement_{index:03d}")
        members_by_name.setdefault(name, []).append(candidate)

    materialized: list[dict[str, Any]] = []
    for group_index, (name, members) in enumerate(members_by_name.items(), start=1):
        canonical = _canonical_cluster_member(members)
        materialized.append(
            _materialize_candidate_group(
                candidate_id=f"candidate_pool_group_{group_index:04d}",
                members=members,
                canonical_name=name,
                canonical_description=_short_text(
                    canonical.get("description") or name,
                    max_chars=2_000,
                ),
            )
        )
    return materialized


def _candidate_group_sort_key(group: Mapping[str, Any]) -> tuple[str, str, str]:
    name = str(group.get("name") or "")
    return (
        _snake_case_name(name, fallback=""),
        name.casefold(),
        str(group.get("candidate_id") or ""),
    )


def _coalesce_exact_candidate_group_names(
    groups: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Coalesce identical canonical outputs produced by independent batches."""

    members_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for index, group in enumerate(groups, start=1):
        name = _snake_case_name(group.get("name"), fallback=f"measurement_{index:03d}")
        members_by_name.setdefault(name, []).append(group)

    coalesced: list[dict[str, Any]] = []
    exact_merges = 0
    for name, members in members_by_name.items():
        if len(members) == 1:
            coalesced.append(dict(members[0]))
            continue
        exact_merges += len(members) - 1
        canonical = _canonical_cluster_member(members)
        coalesced.append(
            _materialize_candidate_group(
                candidate_id=str(members[0]["candidate_id"]),
                members=members,
                canonical_name=name,
                canonical_description=_short_text(
                    canonical.get("description") or name,
                    max_chars=2_000,
                ),
            )
        )
    return sorted(coalesced, key=_candidate_group_sort_key), exact_merges


def _candidate_pool_feature_view(group: Mapping[str, Any]) -> dict[str, Any]:
    """Expose all distinct candidate descriptions without internal provenance."""

    descriptions = list(
        dict.fromkeys(
            description
            for description in [
                _short_text(group.get("description"), max_chars=400),
                *(
                    _short_text(member.get("description"), max_chars=400)
                    for member in list(group.get("member_measurements") or [])
                ),
            ]
            if description
        )
    )
    return {
        "name": str(group.get("name") or ""),
        "descriptions": descriptions,
    }


def _global_candidate_pool_prompt(*, groups: Sequence[Mapping[str, Any]], configured_feature_names: Sequence[str] = (), batch_ordering: str = "alphabetical_shift") -> list[dict[str, str]]:
    del batch_ordering
    views = [_candidate_pool_feature_view(group) for group in sorted(groups, key=_candidate_group_sort_key)]
    text = "Variables\n" + "\n".join(
        "- " + clinical_prompts.label(view["name"]) + ": " + "; ".join(view.get("descriptions") or [view.get("description") or ""])
        for view in views
    )
    text += "\n\nProtected names: " + ("; ".join(clinical_prompts.label(n) for n in configured_feature_names) or "none")
    return clinical_prompts.messages("03_merge_aliases", text)



def _validate_global_candidate_pool_directives(
    value: Mapping[str, Any],
    *,
    group_names: Sequence[str],
    configured_feature_names: Sequence[str] = (),
    group_descriptions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Validate directives using only names and descriptions supplied in the batch."""

    if set(value) == {"merges"}:
        names = {clinical_prompts.label(n): str(n) for n in group_names}
        if not isinstance(value["merges"], list):
            raise ValueError("merges must be an array")
        directives = []
        for row in value["merges"]:
            if not isinstance(row, Mapping) or set(row) != {"members", "canonical_label"}:
                raise ValueError("each merge requires members and canonical_label")
            if not isinstance(row["members"], list) or not isinstance(row["canonical_label"], str):
                raise ValueError("members must be an array and canonical_label must be text")
            inputs = [clinical_prompts.resolve_label(n, names) for n in row["members"]]
            output = clinical_prompts.resolve_label(row["canonical_label"], names)
            if output not in inputs:
                raise ValueError("canonical_label must be one of the merge's supplied member names")
            directives.append({"inputs": inputs, "output": output})
        value = {"merge_directives": directives}
    available = [str(name) for name in group_names]
    if len(available) != len(set(available)):
        raise ValueError("global candidate pool requires unique supplied feature names")
    available_set = set(available)
    exact_alias_owners: dict[str, set[str]] = defaultdict(set)
    normalized_alias_owners: dict[str, set[str]] = defaultdict(set)

    def register_alias(alias: Any, owner: str) -> None:
        rendered = str(alias or "").strip()
        if not rendered:
            return
        exact_alias_owners[rendered.casefold()].add(owner)
        normalized = _snake_case_name(rendered, fallback="")
        if normalized:
            normalized_alias_owners[normalized].add(owner)

    supplied_descriptions = group_descriptions or {}
    unknown_description_names = sorted(set(map(str, supplied_descriptions)) - available_set)
    if unknown_description_names:
        raise ValueError(
            "global candidate pool received descriptions for unknown feature names: "
            f"{unknown_description_names}"
        )
    for name in available:
        register_alias(name, name)
        for description in supplied_descriptions.get(name, ()):
            register_alias(description, name)
    configured_names = set(map(str, configured_feature_names))
    unknown_configured_names = sorted(configured_names - available_set)
    if unknown_configured_names:
        raise ValueError(
            "global candidate pool received unknown configured feature names: "
            f"{unknown_configured_names}"
        )

    def resolve_input_name(raw_name: Any) -> str:
        rendered = str(raw_name or "").strip()
        if rendered in available_set:
            return rendered
        matches = exact_alias_owners.get(rendered.casefold(), set())
        if len(matches) == 1:
            return next(iter(matches))
        normalized = _snake_case_name(rendered, fallback="")
        matches = normalized_alias_owners.get(normalized, set())
        if len(matches) == 1:
            return next(iter(matches))
        raise ValueError(f"global candidate pool named unknown or ambiguous feature {rendered!r}")

    def resolve_output_name(raw_name: Any) -> tuple[str, str | None]:
        rendered = str(raw_name or "").strip()
        if not rendered:
            return "", None
        try:
            known_name = resolve_input_name(rendered)
        except ValueError:
            return _snake_case_name(rendered, fallback=""), None
        return known_name, known_name

    payload = value
    if not isinstance(payload.get("merge_directives"), list):
        for key in ("result", "response", "consolidation"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if "exclude_feature_names" in payload:
        raise ValueError(
            "iterative candidate consolidation is merge-only; omit exclude_feature_names"
        )
    raw_directives = payload.get("merge_directives")
    if raw_directives is None:
        raw_directives = payload.get("merges")
    if not isinstance(raw_directives, list):
        raise ValueError("global candidate pool requires a merge_directives array")

    directives: list[dict[str, Any]] = []
    used_inputs: set[str] = set()
    input_directive_by_name: dict[str, int] = {}
    for index, raw_directive in enumerate(raw_directives, start=1):
        if not isinstance(raw_directive, Mapping):
            raise ValueError(f"global merge directive {index} must be an object")
        raw_inputs = raw_directive.get("inputs")
        if not isinstance(raw_inputs, list):
            raise ValueError(f"global merge directive {index} requires an inputs array")
        inputs = list(dict.fromkeys(resolve_input_name(name) for name in raw_inputs))
        output, known_output_name = resolve_output_name(raw_directive.get("output"))
        if not output:
            raise ValueError(f"global merge directive {index} requires an output name")
        # Models sometimes omit a reused canonical feature from ``inputs`` even
        # though they select it as ``output``. The intended complete family is
        # unambiguous when that output resolves to one supplied feature.
        if known_output_name is not None and known_output_name not in inputs:
            inputs.append(known_output_name)
        if len(inputs) < 2:
            # A name and its own supplied prose description can be emitted as
            # two apparent aliases. Once both resolve to the same feature this
            # is a harmless no-op, not a reason to reject the whole batch.
            continue
        repeated = sorted(set(inputs).intersection(used_inputs))
        if repeated:
            owners = {name: input_directive_by_name[name] for name in repeated}
            raise ValueError(
                "global merge input names may appear in only one directive; "
                f"directive {index} repeats inputs already used by earlier directives: "
                f"{dict(list(owners.items())[:8])}. Combine the complete alias family "
                "into one directive instead of chaining or splitting directives"
            )
        configured_inputs = [name for name in inputs if name in configured_names]
        if len(configured_inputs) > 1:
            raise ValueError(
                "global merge directives must not combine distinct investigator-configured "
                f"features: {configured_inputs}"
            )
        if configured_inputs and output != configured_inputs[0]:
            raise ValueError(
                "a global merge containing an investigator-configured feature must use "
                f"that exact configured name as output: {configured_inputs[0]!r}"
            )
        for name in inputs:
            input_directive_by_name[name] = index
        used_inputs.update(inputs)
        directives.append({"inputs": inputs, "output": output})

    output_directive_by_name: dict[str, int] = {}
    pass_through_names = set(available) - used_inputs
    for index, directive in enumerate(directives, start=1):
        output = str(directive["output"])
        previous_output_index = output_directive_by_name.get(output)
        if previous_output_index is not None:
            raise ValueError(
                f"global merge directive {index} duplicates output name {output!r} from "
                f"directive {previous_output_index}; each directive requires a unique output"
            )
        own_inputs = set(map(str, directive["inputs"]))
        if output in available and output not in own_inputs:
            input_owner = input_directive_by_name.get(output)
            if input_owner is not None:
                raise ValueError(
                    f"global merge directive {index} output name {output!r} is an input "
                    f"of global merge directive {input_owner}; do not chain directives. "
                    "Combine the complete alias family into one directive, or choose an "
                    "output name that is not a supplied feature"
                )
            if output in pass_through_names:
                raise ValueError(
                    f"global merge directive {index} output name {output!r} names an "
                    "unchanged supplied feature; include it in this directive's inputs, "
                    "or choose an output name that is not a supplied feature"
                )
            raise RuntimeError(  # pragma: no cover - exhaustive partition invariant
                f"unclassified global merge output collision for {output!r}"
            )
        output_directive_by_name[output] = index
    return {"merge_directives": directives}


def _apply_global_candidate_pool_directives(
    groups: Sequence[Mapping[str, Any]],
    directives: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply name directives while passing every unmentioned group through unchanged."""

    group_by_name = {str(group["name"]): group for group in groups}
    position_by_name = {str(group["name"]): index for index, group in enumerate(groups)}
    directive_by_first_position: dict[int, Mapping[str, Any]] = {}
    consumed: set[str] = set()
    for directive in directives:
        inputs = [str(name) for name in directive["inputs"]]
        missing = [name for name in inputs if name not in group_by_name]
        if missing:  # pragma: no cover - validator invariant
            raise ValueError(f"global merge directive contains unknown input(s): {missing[:8]}")
        first_position = min(position_by_name[name] for name in inputs)
        directive_by_first_position[first_position] = directive
        consumed.update(inputs)

    merged: list[dict[str, Any]] = []
    for position, raw_group in enumerate(groups):
        group_name = str(raw_group["name"])
        position_directive = directive_by_first_position.get(position)
        if position_directive is not None:
            inputs = [str(name) for name in position_directive["inputs"]]
            members = [group_by_name[name] for name in inputs]
            output_name = str(position_directive["output"])
            canonical = next(
                (member for member in members if str(member["name"]) == output_name),
                _canonical_cluster_member(members),
            )
            first = groups[position]
            merged.append(
                _materialize_candidate_group(
                    candidate_id=str(first["candidate_id"]),
                    members=members,
                    canonical_name=output_name,
                    canonical_description=_short_text(
                        canonical.get("description") or output_name,
                        max_chars=2_000,
                    ),
                )
            )
            continue
        if group_name not in consumed:
            merged.append(dict(raw_group))
    return merged


def _operationalization_prompt(*, feature_name: str, supporting_evidence: Sequence[str]) -> list[dict[str, str]]:
    evidence = list(dict.fromkeys(str(item).strip() for item in supporting_evidence if str(item).strip()))
    if not evidence:
        raise ValueError("operationalization requires readable supporting evidence")
    return clinical_prompts.messages("04_define_ontology", "Clinical variable: " + clinical_prompts.label(feature_name) + "\n\nExample clinical excerpts\n\n" + "\n\n".join(evidence))



def _readable_supporting_text(
    packets: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Project compiled evidence packets to unique readable text only."""

    texts: list[str] = []
    for packet in packets:
        content = packet.get("content")
        if not isinstance(content, Mapping):
            continue
        representative_evidence = content.get("representative_evidence") or []
        if isinstance(representative_evidence, (str, Mapping)):
            representative_evidence = [representative_evidence]
        if not isinstance(representative_evidence, Sequence):
            continue
        for item in representative_evidence:
            raw_text = item.get("text") if isinstance(item, Mapping) else item
            text = str(raw_text or "").strip()
            if text:
                texts.append(text)
    return list(dict.fromkeys(texts))


def _natural_language_feature_name(value: Any) -> str:
    """Render an identifier as a short natural-language embedding query."""

    raw = str(value or "").strip()
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw)
    words = re.findall(r"[A-Za-z0-9]+", spaced.replace("_", " ").replace("-", " "))
    rendered: list[str] = []
    for word in words:
        if word.isupper():
            rendered.append(word)
        elif any(character.isdigit() for character in word) and len(word) <= 8:
            rendered.append(word.upper())
        else:
            rendered.append(word[:1].upper() + word[1:].lower())
    return " ".join(rendered) or raw


def _packet_observable_axes(packet: Mapping[str, Any]) -> list[str]:
    raw_axes = packet.get("observable_axes") or packet.get("evidence_axes")
    content = packet.get("content")
    if not raw_axes and isinstance(content, Mapping):
        raw_axes = content.get("evidence_axes") or content.get("observable_axes")
    return _canonical_evidence_axes(raw_axes)


def _encode_candidate_selection_texts(
    texts: Sequence[str],
    model_name: str,
    device: str,
) -> np.ndarray:
    """Encode and L2-normalize selector text with one shared local model."""

    if not texts:
        raise ValueError("candidate selection requires at least one text to embed")
    from ..models.concept_embedding_cache import load_sentence_transformer

    resolved_device = None if str(device).strip().lower() == "auto" else str(device).strip()
    # Outer folds execute concurrently. SentenceTransformer modules are shared
    # to avoid loading one 0.6B model per fold, so serialize their forward calls.
    with _CANDIDATE_SELECTION_ENCODING_LOCK:
        encoder = load_sentence_transformer(
            str(model_name).strip(),
            device=resolved_device,
        )
        embeddings = encoder.encode(
            list(texts),
            batch_size=min(32, len(texts)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] < 1:
        raise RuntimeError(
            "candidate selection sentence transformer returned unexpected shape "
            f"{matrix.shape} for {len(texts)} texts"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("candidate selection sentence transformer returned non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise RuntimeError("candidate selection sentence transformer returned a zero vector")
    return matrix / norms


def _candidate_registry_text(candidate: Mapping[str, Any]) -> str:
    name = _natural_language_feature_name(candidate.get("name"))
    descriptions = list(
        dict.fromkeys(
            text
            for text in [
                _short_text(candidate.get("description"), max_chars=600),
                *(
                    _short_text(member.get("description"), max_chars=600)
                    for member in list(candidate.get("member_measurements") or [])
                ),
            ]
            if text
        )
    )
    # Exact-name groups can carry thousands of distinct interpretations. The
    # name remains the primary identity signal; a small deterministic sample of
    # descriptions supplies enough context for conservative alias matching.
    return ". ".join([name, *descriptions[:4]])


def _packet_support_architectures(packet: Mapping[str, Any]) -> list[str]:
    content = packet.get("content")
    architectures = content.get("source_architectures") if isinstance(content, Mapping) else None
    if not architectures and packet.get("architecture"):
        architectures = [packet["architecture"]]
    return list(dict.fromkeys(_string_values(architectures)))


def _pack_operationalization_supporting_evidence(
    *,
    feature_name: str,
    supporting_evidence: Sequence[str],
    max_prompt_chars: int,
) -> tuple[list[str], dict[str, Any]]:
    """Pack whole evidence excerpts under a prompt limit with repair headroom."""

    evidence = list(
        dict.fromkeys(str(item).strip() for item in supporting_evidence if str(item).strip())
    )
    if not evidence:
        raise ValueError("operationalization requires readable supporting evidence")
    prompt_limit = int(max_prompt_chars)
    repair_headroom = min(16_000, max(512, prompt_limit // 20))
    initial_prompt_budget = prompt_limit - repair_headroom

    # Count exactly the prose the model receives, including Unicode text.
    template = _operationalization_prompt(feature_name=feature_name, supporting_evidence=[evidence[0]])
    fixed_chars_without_list = sum(len(message["content"]) for message in template) - len(evidence[0])
    list_chars = 0
    packed: list[str] = []
    for text_value in evidence:
        candidate_list_chars = list_chars + (2 if packed else 0) + len(text_value)
        if fixed_chars_without_list + candidate_list_chars > initial_prompt_budget:
            continue
        packed.append(text_value)
        list_chars = candidate_list_chars

    truncated_items = 0
    if not packed:
        available_encoded_chars = initial_prompt_budget - fixed_chars_without_list
        first = evidence[0]
        best = ""
        low, high = 1, len(first)
        while low <= high:
            midpoint = (low + high) // 2
            candidate = first[:midpoint].rstrip()
            if midpoint < len(first):
                candidate = candidate.rstrip(" .") + "..."
            if candidate and len(candidate) <= available_encoded_chars:
                best = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1
        if not best:
            raise ValueError(
                "stage2.operationalization_max_prompt_chars is too small for the "
                "operationalization instructions and one evidence excerpt"
            )
        packed = [best]
        truncated_items = 1

    messages = _operationalization_prompt(
        feature_name=feature_name,
        supporting_evidence=packed,
    )
    prompt_chars = sum(len(message["content"]) for message in messages)
    if prompt_chars > initial_prompt_budget:  # pragma: no cover - exact accounting invariant
        raise RuntimeError(
            "operationalization evidence packing exceeded its calculated prompt budget"
        )
    metadata = {
        "available_evidence_items": len(evidence),
        "included_evidence_items": len(packed),
        "omitted_evidence_items": len(evidence) - len(packed),
        "truncated_evidence_items": truncated_items,
        "available_evidence_chars": sum(len(item) for item in evidence),
        "included_evidence_chars": sum(len(item) for item in packed),
        "available_evidence_fingerprint": _value_fingerprint(evidence),
        "prompt_chars": prompt_chars,
        "initial_prompt_budget_chars": initial_prompt_budget,
        "repair_headroom_chars": repair_headroom,
        "request_prompt_limit_chars": prompt_limit,
    }
    return packed, metadata


def _ambiguous_operationalization_fallback(
    *,
    group: Mapping[str, Any],
    validation_error: Stage2ResponseValidationError,
) -> dict[str, Any]:
    """Return a conservative extraction-ready ontology after exhausted repairs."""

    name = str(group.get("name") or "measurement")
    readable_name = name.replace("_", " ")
    description = str(group.get("description") or readable_name).strip()
    existing_caveats = str(group.get("caveats") or "").strip()
    fallback_caveat = (
        "The ontology response remained structurally invalid after bounded repairs; "
        "the value type is conservatively marked ambiguous for training-fold extraction "
        "and review."
    )
    return {
        "description": description,
        "value_type": "ambiguous",
        "categories_or_unit": [],
        "measurement_definition": (
            f"Extract one explicitly documented pretreatment scalar for {readable_name}; "
            "preserve the documented scalar representation without inventing categories."
        ),
        "missing_value_rule": (
            "Return null when the pretreatment record does not explicitly document one "
            "unambiguous scalar value."
        ),
        "conflict_resolution": {
            "strategy": "single_or_null",
            "positive_category": None,
        },
        "stability_summary": "Model-authored ontology required a conservative fallback.",
        "caveats": " ".join(value for value in (existing_caveats, fallback_caveat) if value),
        "validation_fallback_error": str(validation_error),
    }


def _validate_operationalization(
    value: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
) -> dict[str, Any]:
    # Some instruction-following models wrap the requested fields in a named
    # object or helpfully repeat the feature name/provenance. Those extras are
    # harmless here: Python never reads them when assembling the final feature.
    # Prefer the first recognized nested object while preserving usable scalar
    # fields returned at the top level.
    normalized = dict(value)
    if "unit" in normalized or "categories" in normalized:
        unit, categories = normalized.pop("unit", None), normalized.pop("categories", [])
        if unit is not None and not isinstance(unit, str):
            raise ValueError("unit must be a string or null")
        if not isinstance(categories, list):
            raise ValueError("categories must be an array")
        if normalized.get("value_type") == "continuous":
            if categories:
                raise ValueError("a continuous measurement requires an empty categories array")
            normalized["categories_or_unit"] = [unit] if unit else []
        else:
            if unit is not None:
                raise ValueError("categorical and ambiguous variables require a null unit")
            normalized["categories_or_unit"] = categories
        if isinstance(normalized.get("caveats"), list):
            normalized["caveats"] = " ".join(map(str, normalized["caveats"]))
    for key in ("operationalization", "feature", "definition", "variable", "result"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            normalized.update(nested)
            break
    raw_features = value.get("features")
    if isinstance(raw_features, Sequence) and not isinstance(raw_features, str):
        if len(raw_features) == 1 and isinstance(raw_features[0], Mapping):
            normalized.update(raw_features[0])

    raw_categories = normalized.get("categories_or_unit")
    if isinstance(raw_categories, Mapping):
        raw_categories = (
            raw_categories.get("categories")
            or raw_categories.get("values")
            or raw_categories.get("unit")
        )
    if raw_categories is None:
        raw_categories = (
            normalized.get("categories")
            or normalized.get("allowed_values")
            or normalized.get("levels")
            or normalized.get("unit")
        )
    categories = _string_values(raw_categories)
    raw_value_type = (
        normalized.get("value_type") or normalized.get("data_type") or normalized.get("type")
    )
    if not str(raw_value_type or "").strip():
        raise ValueError(
            "operationalization requires the model to choose value_type from "
            "binary, categorical, continuous, ordinal, or ambiguous"
        )
    value_type = str(raw_value_type)
    value_type = value_type.strip().lower()
    value_type = {
        "bool": "binary",
        "boolean": "binary",
        "category": "categorical",
        "numeric": "continuous",
        "number": "continuous",
        "unknown": "ambiguous",
    }.get(value_type, value_type)
    if value_type not in ALLOWED_VALUE_TYPES:
        raise ValueError(
            "operationalization value_type must be binary, categorical, continuous, "
            f"ordinal, or ambiguous; received {value_type!r}"
        )
    if value_type in {"binary", "categorical", "ordinal"}:
        from .plain_handoff_stage2_analysis import _validated_closed_category_values

        categories = _validated_closed_category_values(
            value_type=value_type,
            values=categories,
            source="operationalization",
        )
    description = str(
        normalized.get("description")
        or normalized.get("clinical_definition")
        or normalized.get("summary")
        or group.get("description")
        or ""
    ).strip()
    measurement_definition = str(
        normalized.get("measurement_definition")
        or normalized.get("operational_definition")
        or normalized.get("extraction_definition")
        or normalized.get("extraction_instruction")
        or normalized.get("measurement_rule")
        or normalized.get("how_to_measure")
        or (
            normalized.get("definition")
            if not isinstance(normalized.get("definition"), Mapping)
            else ""
        )
        or ""
    ).strip()
    if not measurement_definition:
        canonical_name = str(group.get("name") or "measurement").replace("_", " ")
        canonical_description = description or canonical_name
        measurement_definition = (
            f"Extract one pretreatment scalar for {canonical_name} according to this "
            f"canonical definition: {canonical_description}"
        )
    missing_value_rule = str(
        normalized.get("missing_value_rule")
        or normalized.get("missingness_rule")
        or normalized.get("missing_data_rule")
        or normalized.get("missing_value_handling")
        or ""
    ).strip()
    if not missing_value_rule:
        missing_value_rule = (
            "Return null when the pretreatment record does not explicitly document "
            "a single unambiguous value."
        )
    from .plain_handoff_stage2_analysis import _resolved_conflict_resolution

    conflict_resolution = _resolved_conflict_resolution(
        {
            "name": str(group.get("name") or ""),
            "description": description,
            "value_type": value_type,
            "categories_or_unit": categories,
            "measurement_definition": measurement_definition,
            "missing_value_rule": missing_value_rule,
            "conflict_resolution": (
                normalized.get("conflict_resolution")
                or normalized.get("longitudinal_resolution")
                or normalized.get("conflict_rule")
            ),
        }
    )
    architecture_count = len(_candidate_architectures(group))
    packet_count = len(_string_values(group.get("supporting_packet_ids")))
    stability_summary = str(
        normalized.get("stability_summary")
        or normalized.get("support_summary")
        or normalized.get("evidence_summary")
        or ""
    ).strip()
    if not stability_summary:
        stability_summary = (
            f"Supported by {packet_count} evidence packet(s) across "
            f"{architecture_count} Stage 1 architecture(s)."
        )
    return {
        "description": description or str(group.get("name") or ""),
        "value_type": value_type,
        "categories_or_unit": categories,
        "measurement_definition": measurement_definition,
        "missing_value_rule": missing_value_rule,
        "conflict_resolution": {
            "strategy": conflict_resolution["strategy"],
            "positive_category": conflict_resolution["positive_category"],
        },
        "stability_summary": stability_summary,
        "caveats": str(
            normalized.get("caveats") or normalized.get("limitations") or group.get("caveats") or ""
        ).strip(),
    }


_SPLIT_PROVENANCE_REMEDY = (
    "Restore components/tfidf/split_provenance.jsonl from the original Stage 1 "
    "run, or rerun Stage 1 (scripts/run_all_evidence.py with the same output "
    "directory) so the handoff and split provenance are regenerated together."
)


def _load_stage2_splits(
    *,
    provenance_path: Path | None,
    dataset_rows: int,
    outer_fold_ids: Sequence[int],
    inner_folds: int | None = None,
    seed: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Read the frozen Stage 1 split file; never recreate outer folds.

    A resampled split can place rows that Stage 1 fitted on into Stage 2
    held-out sets, so a missing or empty file fails closed.  ``inner_folds``
    and ``seed`` are accepted for caller compatibility and are not used.
    """

    del inner_folds, seed
    rows: list[dict[str, Any]] = []
    if provenance_path is not None and Path(provenance_path).is_file():
        rows = [
            json.loads(line)
            for line in Path(provenance_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not rows:
        location = (
            str(provenance_path)
            if provenance_path is not None
            else "components/tfidf/split_provenance.jsonl (no path was supplied)"
        )
        raise FileNotFoundError(
            "Stage 2 requires the frozen Stage 1 split provenance, but "
            f"{location} is missing or empty. Stage 2 will not recreate outer "
            "folds because a different split could put rows that Stage 1 "
            "trained on into Stage 2 held-out sets. " + _SPLIT_PROVENANCE_REMEDY
        )
    by_fold: dict[int, dict[str, Any]] = {}
    for raw in rows:
        outer_fold = int(raw["outer_fold"])
        fit_ids = [int(value) for value in raw["fit_row_ids"]]
        heldout_ids = [int(value) for value in raw["heldout_row_ids"]]
        if outer_fold in by_fold:
            raise ValueError(f"duplicate split provenance for outer fold {outer_fold}")
        if not fit_ids or not heldout_ids:
            raise ValueError(f"outer fold {outer_fold} requires nonempty fit and heldout rows")
        if set(fit_ids).intersection(heldout_ids):
            raise ValueError(f"outer fold {outer_fold} has overlapping fit and heldout rows")
        if any(value < 0 or value >= dataset_rows for value in [*fit_ids, *heldout_ids]):
            raise ValueError(f"outer fold {outer_fold} contains an out-of-range row id")
        inner_splits = []
        for inner_index, inner in enumerate(raw.get("inner_splits") or [], start=1):
            inner_splits.append(
                {
                    "inner_fold": int(inner.get("inner_fold", inner_index)),
                    "fit_row_ids": [int(value) for value in inner["fit_row_ids"]],
                    "heldout_row_ids": [int(value) for value in inner["heldout_row_ids"]],
                }
            )
        by_fold[outer_fold] = {
            "outer_fold": outer_fold,
            "fit_row_ids": fit_ids,
            "heldout_row_ids": heldout_ids,
            "inner_splits": inner_splits,
        }
    missing = sorted(set(map(int, outer_fold_ids)) - set(by_fold))
    if missing:
        raise ValueError(
            f"split provenance is missing Stage 2 outer folds: {missing}. "
            + _SPLIT_PROVENANCE_REMEDY
        )
    return by_fold


def _stage1_context_row_provenance(handoff_path: Path) -> list[dict[str, Any]]:
    """Return the fit/held-out rows each Stage 1 context recorded.

    Raw (omitted-selector) handoff rows embed their producer row.  Materialized
    architecture handoffs carry compact occurrences only, so their producer
    rows are read from the component files referenced by ``index.json``.
    TF-IDF and neural-query contexts record row IDs; text-model contexts
    record fit/held-out row counts.  Rows are streamed, never retained.
    """

    def context_of(row: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any] | None:
        outer_fold = row.get("outer_fold")
        outer_fold = payload.get("outer_fold") if outer_fold is None else outer_fold
        if outer_fold is None:
            return None
        inner_fold = row.get("inner_fold")
        inner_fold = payload.get("inner_fold") if inner_fold is None else inner_fold
        scope = str(row.get("scope") or payload.get("scope") or "")
        if scope == "full_outer_train":
            inner_fold = None
        fit_ids = payload.get("fit_row_ids")
        heldout_ids = payload.get("heldout_row_ids")
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
        context: dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "inner_fold": None if inner_fold is None else int(inner_fold),
            "scope": scope,
        }
        if isinstance(fit_ids, list) and isinstance(heldout_ids, list):
            context["fit_row_ids"] = [int(value) for value in fit_ids]
            context["heldout_row_ids"] = [int(value) for value in heldout_ids]
        elif (
            metrics.get("handoff_fit_rows") is not None
            and metrics.get("handoff_heldout_rows") is not None
        ):
            context["fit_rows"] = int(metrics["handoff_fit_rows"])
            context["heldout_rows"] = int(metrics["handoff_heldout_rows"])
        else:
            return None
        return context

    handoff_path = Path(handoff_path)
    contexts: list[dict[str, Any]] = []
    materialized = False
    for row in _iter_jsonl(handoff_path):
        if str(row.get("source") or "") == "stage1_architecture":
            materialized = True
            continue
        payload = row.get("evidence")
        context = context_of(row, payload if isinstance(payload, Mapping) else row)
        if context is not None:
            contexts.append(context)
    if materialized:
        index_path = handoff_path.parent / "index.json"
        index = (
            json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
        )
        sources = index.get("sources") if isinstance(index, Mapping) else None
        if not isinstance(sources, Mapping) or not sources:
            raise RuntimeError(
                f"Stage 2 cannot verify outer folds: {index_path} does not list "
                "the Stage 1 component evidence behind the architecture handoff. "
                + _SPLIT_PROVENANCE_REMEDY
            )
        for _name, relative in sorted(sources.items()):
            source_path = handoff_path.parent / str(relative)
            if not source_path.is_file():
                raise FileNotFoundError(
                    "Stage 2 cannot verify outer folds: Stage 1 component evidence "
                    f"{source_path} referenced by {index_path} is missing. "
                    + _SPLIT_PROVENANCE_REMEDY
                )
            for row in _iter_jsonl(source_path):
                context = context_of(row, row)
                if context is not None:
                    contexts.append(context)
    return contexts


def _verify_stage2_splits_against_handoff(
    splits: Mapping[int, Mapping[str, Any]],
    *,
    handoff_path: Path,
    outer_fold_ids: Sequence[int],
) -> None:
    """Require the split file to reproduce every Stage 1 context's rows."""

    contexts = _stage1_context_row_provenance(handoff_path)
    verified_outer: set[int] = set()
    for context in contexts:
        outer_fold = int(context["outer_fold"])
        inner_fold = context["inner_fold"]
        label = (
            f"outer fold {outer_fold}"
            if inner_fold is None
            else f"outer fold {outer_fold} inner fold {inner_fold}"
        )
        split = splits.get(outer_fold)
        if split is not None and inner_fold is not None:
            split = next(
                (
                    inner
                    for inner in split.get("inner_splits") or []
                    if int(inner["inner_fold"]) == int(inner_fold)
                ),
                None,
            )
        if split is None:
            raise RuntimeError(
                f"Stage 1 handoff context {label} ({context['scope']}) has no "
                "matching entry in the Stage 2 split provenance. "
                + _SPLIT_PROVENANCE_REMEDY
            )
        for side in ("fit", "heldout"):
            expected = [int(value) for value in split[f"{side}_row_ids"]]
            if f"{side}_row_ids" in context:
                actual = context[f"{side}_row_ids"]
                matches = len(actual) == len(expected) and set(actual) == set(expected)
            else:
                matches = int(context[f"{side}_rows"]) == len(expected)
            if not matches:
                raise RuntimeError(
                    f"Stage 2 split provenance disagrees with the Stage 1 handoff "
                    f"for {label} ({context['scope']}): the {side} rows Stage 1 "
                    "used differ from the split file, so Stage 2 held-out rows "
                    "could overlap Stage 1 training rows. " + _SPLIT_PROVENANCE_REMEDY
                )
        if inner_fold is None:
            verified_outer.add(outer_fold)
    unverified = sorted(set(map(int, outer_fold_ids)) - verified_outer)
    if unverified:
        raise RuntimeError(
            "Stage 2 cannot verify outer folds "
            f"{unverified}: the Stage 1 handoff records no full_outer_train "
            "fit/held-out row provenance for them. " + _SPLIT_PROVENANCE_REMEDY
        )


class PlainHandoffStage2:
    def __init__(
        self,
        *,
        config: PlainHandoffStage2Config,
        clinical_question: str,
        completion: CompletionFunction | None = None,
        extraction_completion: CompletionFunction | None = None,
        extraction_tokenizer: Any | None = None,
    ) -> None:
        config = _resolve_stage2_model(config)
        config = _resolve_extraction_llm_model(config)
        config.validate()
        endpoints = config.runtime_endpoints or (config.endpoint,)
        primary_identity = _endpoint_model_identity(
            config,
            endpoints=endpoints,
            role="primary",
            verify_live_endpoint=_uses_default_openai_transport(completion),
        )
        config = replace(
            config,
            runtime_model_family=str(primary_identity["model_family"]),
            runtime_sampling_model=str(primary_identity.get("sampling_model") or ""),
            runtime_context_window_tokens=_primary_context_window_tokens(config, primary_identity),
        )
        config.validate()
        self.config = config
        self.clinical_question = str(clinical_question)
        routed_completion = completion
        if routed_completion is None:
            if config.vllm is not None:
                server_limit = int(managed_argument(config.vllm, "--max-num-seqs", config.workers))
                server_limit = min(server_limit, math.ceil(config.workers / len(endpoints)))
                routed_completion = _LoadAwareOpenAICompletion(
                    tuple(ExtractionEndpoint(url, server_limit) for url in endpoints),
                    api_key=config.api_key, latency_weighted=False,
                )
            else:
                routed_completion = _RoundRobinOpenAICompletion(endpoints)
        self.completion = _ConcurrencyLimitedCompletion(
            routed_completion,
            max_concurrency=config.workers,
        )
        self.selection_consolidation_request_config: PlainHandoffStage2Config | None = None
        self.selection_consolidation_completion: CompletionFunction | None = None
        consolidation_runtime = config.selection_consolidation
        if str(consolidation_runtime.runtime_llm_endpoint).strip():
            consolidation_request_config = replace(
                config,
                endpoint=str(consolidation_runtime.runtime_llm_endpoint).rstrip("/"),
                model=str(consolidation_runtime.runtime_llm_model).strip(),
                api_key=str(consolidation_runtime.runtime_llm_api_key),
                max_prompt_chars=max(
                    int(config.max_prompt_chars),
                    int(consolidation_runtime.max_prompt_chars),
                ),
                extraction_llm=None,
                vllm=None,
                runtime_endpoints=(),
                runtime_model_family="",
                runtime_sampling_model="",
                runtime_context_window_tokens=None,
            )
            consolidation_identity = _endpoint_model_identity(
                consolidation_request_config,
                endpoints=(consolidation_request_config.endpoint,),
                role="selection_consolidation_runtime",
                verify_live_endpoint=True,
            )
            self.selection_consolidation_request_config = replace(
                consolidation_request_config,
                runtime_model_family=str(consolidation_identity["model_family"]),
                runtime_sampling_model=str(consolidation_identity.get("sampling_model") or ""),
                runtime_context_window_tokens=_primary_context_window_tokens(
                    consolidation_request_config, consolidation_identity,
                ),
            )
            self.selection_consolidation_completion = _ConcurrencyLimitedCompletion(
                _RoundRobinOpenAICompletion((consolidation_request_config.endpoint,)),
                max_concurrency=config.workers,
            )
        else:
            consolidation_identity = None
        self.extraction_request_config: PlainHandoffStage2Config | None = None
        self.extraction_completion: CompletionFunction | None = None
        self.extraction_tokenizer: Any | None = None
        if config.extraction_llm is not None:
            extraction = config.extraction_llm
            continuation_route = bool(str(extraction.runtime_endpoint).strip())
            extraction_endpoints = (
                (str(extraction.runtime_endpoint).rstrip("/"),)
                if continuation_route
                else tuple(server.endpoint for server in extraction.endpoints)
                or extraction.runtime_endpoints or (extraction.endpoint,)
            )
            request_model = (
                str(extraction.runtime_model).strip()
                if continuation_route
                else extraction.model
            )
            request_api_key = (
                extraction.runtime_api_key
                if continuation_route
                else extraction.api_key
            )
            extraction_request_config = replace(
                config,
                endpoint=extraction_endpoints[0],
                model=request_model,
                api_key=request_api_key,
                workers=extraction.workers,
                extraction_llm=None,
                vllm=None,
                runtime_endpoints=(),
                runtime_model_family="",
                runtime_sampling_model="",
            )
            routed_extraction = extraction_completion or completion
            uses_default_extraction_transport = _uses_default_openai_transport(
                routed_extraction
            )
            live_extraction_identity = _endpoint_model_identity(
                extraction_request_config,
                endpoints=extraction_endpoints,
                role=(
                    "extraction_runtime_continuation"
                    if continuation_route
                    else "extraction"
                ),
                verify_live_endpoint=_uses_default_openai_transport(routed_extraction),
            )
            if continuation_route:
                extraction_identity = {
                    "role": "extraction",
                    "selected_model": extraction.model,
                    "model_family": _stage2_model_family(extraction.model),
                    "actual_model_identity": None,
                    "live_endpoint_verified": False,
                    "endpoint_observations": [],
                    "runtime_continuation_is_separately_audited": True,
                }
                extraction_runtime_identity = live_extraction_identity
            else:
                extraction_identity = live_extraction_identity
                extraction_runtime_identity = None
            self.extraction_request_config = replace(
                extraction_request_config,
                runtime_model_family=str(live_extraction_identity["model_family"]),
                runtime_sampling_model=str(live_extraction_identity.get("sampling_model") or ""),
            )
            if routed_extraction is None:
                routed_extraction = (
                    _LoadAwareOpenAICompletion(extraction.endpoints, api_key=request_api_key)
                    if extraction.endpoints
                    else _RoundRobinOpenAICompletion(extraction_endpoints)
                )
            self.extraction_completion = _ConcurrencyLimitedCompletion(
                routed_extraction,
                max_concurrency=(min(extraction.workers, sum(server.max_concurrency
                                                             for server in extraction.endpoints))
                                 if extraction.endpoints else extraction.workers),
            )
            if extraction_tokenizer is not None:
                self.extraction_tokenizer = extraction_tokenizer
            elif uses_default_extraction_transport:
                extraction_policy = _stage2_request_policy(self.extraction_request_config, "extraction")
                template_kwargs = dict(extraction.vllm.default_chat_template_kwargs or {}) if extraction.vllm else {}
                template_kwargs["enable_thinking"] = _reasoning_enabled(extraction_policy["reasoning_effort"])
                if extraction_policy.get("preserve_thinking"):
                    template_kwargs.update(preserve_thinking=True, reasoning_effort=extraction_policy["reasoning_effort"])
                self.extraction_tokenizer = _LazyStage2ExtractionTokenizer(
                    model=str(live_extraction_identity.get("sampling_model") or request_model),
                    cache_dir=str(extraction.vllm.download_dir) if extraction.vllm else "",
                    chat_template_kwargs=template_kwargs,
                )
        else:
            extraction_identity = None
            extraction_runtime_identity = None
        self.model_identity = {
            "schema_version": MODEL_IDENTITY_SCHEMA_VERSION,
            "feature_definition_model": config.frozen_feature_definition_model or config.model,
            "primary": primary_identity,
            "extraction": extraction_identity,
            "extraction_runtime_continuation": extraction_runtime_identity,
            "selection_consolidation_runtime": consolidation_identity,
            "endpoint_urls_are_transport_only": True,
            "effective_request_policies": {
                "interpretation": _stage2_request_policy(self.config, "interpretation"),
                "extraction": _stage2_request_policy(
                    self.extraction_request_config or self.config, "extraction"
                ),
            },
        }

    def _check_and_record_model_identity(self, output_dir: Path) -> None:
        """Check served names while allowing endpoint and backing model changes."""

        identity_path = output_dir / "model_identity.json"
        current_scientific = _scientific_model_identity(self.model_identity)
        previous_scientific: dict[str, Any] | None = None
        previous_identity: Mapping[str, Any] | None = None
        if identity_path.is_file():
            previous_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            previous_scientific = _scientific_model_identity(previous_identity)
        if previous_scientific is not None:
            if not _model_role_identity_compatible(
                previous_scientific,
                current_scientific,
                role="primary",
            ):
                raise RuntimeError(
                    "Stage 2 cannot resume because the actual running model identity "
                    "changed for the primary role. "
                    f"Previous={previous_scientific}; current={current_scientific}. "
                    "Preserve the existing output for audit and use a fresh Stage 2 "
                    "output directory."
                )
            if not _model_role_identity_compatible(
                previous_scientific,
                current_scientific,
                role="extraction",
            ):
                if _has_extraction_dependent_checkpoints(output_dir):
                    raise RuntimeError(
                        "Stage 2 cannot resume because the actual running model identity "
                        "changed for the extraction role while extraction-dependent checkpoints "
                        "remain. Remove the outer-fold extraction-and-later artifacts "
                        "before resuming with a new extractor. "
                        f"Previous={previous_scientific}; current={current_scientific}."
                    )
                LOGGER.info(
                    "Stage 2 extraction model changed before any extraction-dependent "
                    "checkpoint; preserving completed interpretation and feature definitions"
                )
        recorded_identity = dict(self.model_identity)
        if previous_identity is not None:
            # Alternating managed phases intentionally cannot probe the inactive
            # model. Preserve its last verified backing identity so the audit
            # manifest retains verification for both scientific roles.
            for role in ("primary", "extraction"):
                current_role = recorded_identity.get(role)
                previous_role = previous_identity.get(role)
                if not isinstance(current_role, Mapping) or not isinstance(
                    previous_role, Mapping
                ):
                    continue
                if str(current_role.get("selected_model") or "") != str(
                    previous_role.get("selected_model") or ""
                ):
                    continue
                if isinstance(current_role.get("actual_model_identity"), Mapping):
                    continue
                prior_actual = previous_role.get("actual_model_identity")
                if not isinstance(prior_actual, Mapping):
                    continue
                merged_role = dict(current_role)
                merged_role["actual_model_identity"] = dict(prior_actual)
                merged_role["previous_live_endpoint_verification"] = {
                    "live_endpoint_verified": bool(
                        previous_role.get("live_endpoint_verified")
                    ),
                    "endpoint_observations": list(
                        previous_role.get("endpoint_observations") or []
                    ),
                }
                recorded_identity[role] = merged_role
        self.model_identity = recorded_identity
        _write_json(
            identity_path,
            {
                **recorded_identity,
                "checked_at": _now(),
                "scientific_identity": _scientific_model_identity(recorded_identity),
            },
        )

    def _load_or_compile_evidence(
        self,
        *,
        handoff_path: Path,
        output_dir: Path,
        seed: int,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        """Load a valid compiled plan or build it once from the raw handoff."""

        compilation_dir = output_dir / "evidence_compilation"
        packets_path = compilation_dir / "packets.jsonl"
        summary_path = compilation_dir / "summary.json"
        complete_path = compilation_dir / "compile_complete.json"
        max_packet_chars = max(2_000, self.config.max_prompt_chars // 4)
        embedding_cache_dependency = stage1_embedding_cache_dependency_identity(
            handoff_path
        )
        signature = {
            "compiler": EVIDENCE_COMPILER_VERSION,
            "required_architectures": list(self.config.required_architectures),
            "included_architectures": (
                None
                if self.config.included_architectures is None
                else list(self.config.included_architectures)
            ),
            "max_cards_per_outer_fold": self.config.evidence_max_cards_per_fold,
            "max_exemplars_per_card": self.config.evidence_max_exemplars_per_card,
            "max_exemplar_chars": self.config.evidence_max_exemplar_chars,
            "max_packet_chars": max_packet_chars,
            "seed": int(seed),
            "stage1_embedding_cache_dependency": embedding_cache_dependency,
        }
        signature_fingerprint = _value_fingerprint(signature)
        handoff_size = handoff_path.stat().st_size
        LOGGER.info(
            "fingerprinting Stage 1 handoff bytes=%s path=%s",
            handoff_size,
            handoff_path,
        )
        hash_started = time.monotonic()
        handoff_sha256 = _file_sha256(handoff_path)
        LOGGER.info(
            "fingerprinted Stage 1 handoff bytes=%s seconds=%.2f path=%s",
            handoff_size,
            time.monotonic() - hash_started,
            handoff_path,
        )
        if complete_path.is_file() and packets_path.is_file() and summary_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                complete.get("handoff_sha256") == handoff_sha256
                and complete.get("compiler_signature_sha256") == signature_fingerprint
            ):
                packets = _read_jsonl(packets_path)
                outer_folds = sorted({int(packet["outer_fold"]) for packet in packets})
                manifest_paths = [
                    compilation_dir / f"outer_{outer_fold:03d}" / filename
                    for outer_fold in outer_folds
                    for filename in ("cards.jsonl", "members.jsonl", "lineage.jsonl")
                ]
                if all(path.is_file() for path in manifest_paths):
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    LOGGER.info(
                        "loaded cached Stage 2 evidence compilation packets=%s path=%s",
                        len(packets),
                        compilation_dir,
                    )
                    return packets, summary
                LOGGER.warning(
                    "rebuild incomplete Stage 2 evidence compilation; missing manifests=%s",
                    [str(path) for path in manifest_paths if not path.is_file()][:8],
                )
        compile_started = time.monotonic()
        LOGGER.info(
            "compiling Stage 2 evidence cards from raw handoff bytes=%s path=%s",
            handoff_size,
            handoff_path,
        )
        compiled = compile_stage2_handoff_evidence(
            _iter_jsonl(handoff_path),
            handoff_path=handoff_path,
            max_cards_per_outer_fold=self.config.evidence_max_cards_per_fold,
            max_exemplars_per_card=self.config.evidence_max_exemplars_per_card,
            max_exemplar_chars=self.config.evidence_max_exemplar_chars,
            max_packet_chars=max_packet_chars,
            seed=seed,
            required_architectures=self.config.required_architectures,
            included_architectures=self.config.included_architectures,
        )
        packets = [dict(packet) for packet in compiled.packets]
        summary = dict(compiled.summary)
        for outer_fold, cards in compiled.cards_by_outer_fold.items():
            fold_dir = compilation_dir / f"outer_{int(outer_fold):03d}"
            _write_jsonl(fold_dir / "cards.jsonl", cards)
            _write_jsonl(
                fold_dir / "members.jsonl",
                compiled.members_by_outer_fold[outer_fold],
            )
            _write_jsonl(
                fold_dir / "lineage.jsonl",
                compiled.lineage_by_outer_fold[outer_fold],
            )
        elapsed = time.monotonic() - compile_started
        summary = {
            **summary,
            "handoff_path": str(handoff_path),
            "handoff_bytes": handoff_size,
            "handoff_sha256": handoff_sha256,
            "compiler_signature": signature,
            "compiler_signature_sha256": signature_fingerprint,
            "compilation_seconds": elapsed,
        }
        _write_jsonl(packets_path, packets)
        _write_json(summary_path, summary)
        _write_json(
            complete_path,
            {
                "status": "complete",
                "completed_at": _now(),
                "handoff_sha256": handoff_sha256,
                "compiler_signature_sha256": signature_fingerprint,
                "packets": len(packets),
            },
        )
        LOGGER.info(
            "compiled Stage 1 evidence compiler=%s packets=%s seconds=%.2f path=%s",
            EVIDENCE_COMPILER_VERSION,
            len(packets),
            elapsed,
            compilation_dir,
        )
        return packets, summary

    def _interpret_batch(
        self,
        *,
        architecture: str,
        packets: Sequence[Mapping[str, Any]],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        if len(packets) != 1:
            raise ValueError("discovery requires exactly one evidence card per request")
        input_value = {
            "interpretation_schema": INTERPRETATION_SCHEMA_VERSION,
            "llm_identity": {
                "model": self.config.model,
            },
            "architecture": architecture,
            "packets": list(packets),
        }
        input_fingerprint = _value_fingerprint(input_value)
        packet = packets[0]
        packet_id = str(packet["packet_id"])
        complete_path = output_dir / "complete.json"
        result_path = output_dir / "result.json"
        input_path = output_dir / "input.json"
        cached_files = _read_checkpoint_jsons([input_path, complete_path, result_path])
        if cached_files is not None:
            previous, completion_state, cached_result = cached_files
            if (
                previous.get("input_fingerprint") == input_fingerprint
                and completion_state.get("input_fingerprint") == input_fingerprint
            ):
                if _cached_interpretation_matches_packets(
                    cached_result,
                    packet_ids={packet_id},
                ) and (
                    cached_result.get("rejected_packet_audit", {}).get("schema_version")
                    == INTERPRETATION_AUDIT_SCHEMA_VERSION
                ):
                    LOGGER.info("skip completed Stage 2 interpretation: %s", output_dir)
                    return cached_result
            LOGGER.info("rerun stale or inconsistent Stage 2 interpretation: %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(input_path, {**input_value, "input_fingerprint": input_fingerprint})
        evidence_axes = _packet_observable_axes(packet)
        initial = _checkpointed_request_json(
            output_dir=output_dir / "initial",
            input_value={
                "phase": "initial_interpretation",
                **input_value,
            },
            messages=_interpretation_prompt(
                architecture=architecture,
                packets=packets,
            ),
            config=self.config,
            completion=self.completion,
            validate=lambda value: _validate_discovery(
                value,
                packet_id=packet_id,
                evidence_axes=evidence_axes,
            ),
        )

        rejected = (
            initial["packet_dispositions"][packet_id].get("status")
            == "reviewed_no_specific_concept"
        )
        audit_results: list[Mapping[str, Any]] = []
        if rejected:
            LOGGER.info(
                "Stage 2 rejected-packet audit architecture=%s packet=%s",
                architecture,
                packet_id,
            )
            # The audit prompt returns the same candidate contract as discovery.
            audit_results.append(
                _checkpointed_request_json(
                    output_dir=output_dir / "rejected_packet_audit" / "batch_001",
                    input_value={
                        "phase": "rejected_packet_audit",
                        "audit_schema": INTERPRETATION_AUDIT_SCHEMA_VERSION,
                        "interpretation_schema": INTERPRETATION_SCHEMA_VERSION,
                        "architecture": architecture,
                        "packets": list(packets),
                    },
                    messages=_rejected_packet_audit_prompt(
                        architecture=architecture,
                        packets=packets,
                    ),
                    config=self.config,
                    completion=self.completion,
                    validate=lambda value: _validate_discovery(
                        value,
                        packet_id=packet_id,
                        evidence_axes=evidence_axes,
                    ),
                )
            )

        result = _merge_interpretation_audit(
            packet_ids={packet_id},
            initial=initial,
            audits=audit_results,
        )
        _write_json(result_path, result)
        _write_json(
            complete_path,
            {
                "status": "complete",
                "completed_at": _now(),
                "input_fingerprint": input_fingerprint,
                "initially_rejected_packets": int(rejected),
                "recovered_packets": len(result["rejected_packet_audit"]["recovered_packet_ids"]),
                "audit_batches": len(audit_results),
            },
        )
        return result

    def _operationalize_candidate_group(
        self,
        *,
        group: Mapping[str, Any],
        packet_by_id: Mapping[str, Mapping[str, Any]],
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        configured = _configured_feature_definitions(group)
        if configured:
            if len(configured) != 1:
                raise ValueError(
                    "Stage 2 consolidation attempted to assign more than one supplied "
                    "ontology to one feature group"
                )
            definition = dict(configured[0])
            configured_name = str(definition.pop("name"))
            group_name = _snake_case_name(group.get("name"), fallback="")
            if group_name != configured_name:
                raise ValueError(
                    "Stage 2 consolidation renamed an investigator-configured feature; "
                    f"expected {configured_name!r}, received {group_name!r}"
                )
            roles = [
                role
                for role in _string_values(definition.pop("roles", []))
                if role in ALLOWED_ROLES
            ]
            if not roles:  # pragma: no cover - config validation invariant
                raise ValueError(
                    f"Stage 2 configured feature {configured_name!r} has no causal role"
                )
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    output_dir / "provided_ontology.json",
                    {
                        "status": "used_without_model_operationalization",
                        "source": "stage2.explicit_features",
                        "feature": {
                            "name": configured_name,
                            **definition,
                            "roles": roles,
                        },
                    },
                )
            return {
                "name": configured_name,
                **definition,
                "roles": roles,
                "supporting_packet_ids": _string_values(group.get("supporting_packet_ids")),
                "supporting_architectures": _string_values(group.get("supporting_architectures")),
                "configured_explicit_feature": True,
            }

        ontology_packet_ids = _string_values(group.get("ontology_packet_ids"))
        if not ontology_packet_ids:
            raise ValueError(
                f"Stage 2 group {group.get('name')!r} has no original evidence packet "
                "for ontology definition"
            )
        missing_packet_ids = [
            packet_id for packet_id in ontology_packet_ids if packet_id not in packet_by_id
        ]
        if missing_packet_ids:
            raise ValueError(
                f"Stage 2 group {group.get('name')!r} cites unknown ontology evidence "
                f"packet(s): {missing_packet_ids[:8]}"
            )
        evidence_packets = [packet_by_id[packet_id] for packet_id in ontology_packet_ids]
        available_supporting_evidence = _readable_supporting_text(evidence_packets)
        if not available_supporting_evidence:
            raise ValueError(
                f"Stage 2 group {group.get('name')!r} has no readable supporting "
                "evidence for ontology definition"
            )
        feature_name = str(group.get("name") or "")
        operationalization_prompt_limit = int(self.config.operationalization_max_prompt_chars)
        supporting_evidence, evidence_packing = _pack_operationalization_supporting_evidence(
            feature_name=feature_name,
            supporting_evidence=available_supporting_evidence,
            max_prompt_chars=operationalization_prompt_limit,
        )
        if (
            evidence_packing["omitted_evidence_items"]
            or evidence_packing["truncated_evidence_items"]
        ):
            LOGGER.warning(
                "Stage 2 operationalization packed supporting evidence feature=%s "
                "included=%s available=%s prompt_chars=%s prompt_limit=%s",
                feature_name,
                evidence_packing["included_evidence_items"],
                evidence_packing["available_evidence_items"],
                evidence_packing["prompt_chars"],
                operationalization_prompt_limit,
            )
        messages = _operationalization_prompt(
            feature_name=feature_name,
            supporting_evidence=supporting_evidence,
        )
        request_config = replace(
            self.config,
            max_prompt_chars=operationalization_prompt_limit,
        )
        operational = _checkpointed_request_json(
            output_dir=output_dir,
            input_value={
                "phase": "group_operationalization",
                "operationalization_schema": OPERATIONALIZATION_SCHEMA_VERSION,
                "candidate_feature_name": feature_name,
                "ontology_packet_ids": ontology_packet_ids,
                "supporting_evidence": supporting_evidence,
                "evidence_packing": evidence_packing,
            },
            messages=messages,
            config=request_config,
            completion=self.completion,
            validate=lambda value: _validate_operationalization(value, group=group),
            validation_fallback=lambda exc: _ambiguous_operationalization_fallback(
                group=group,
                validation_error=exc,
            ),
        )
        return {
            "name": str(group["name"]),
            **operational,
            # Discovered candidates receive causal roles only from the
            # fold-local statistical screens. Evidence never pre-selects a role.
            "roles": [],
            "supporting_packet_ids": _string_values(group.get("supporting_packet_ids")),
            "supporting_architectures": _string_values(group.get("supporting_architectures")),
        }

    def _consolidate_candidate_pool(
        self,
        *,
        outer_fold: int,
        groups: Sequence[Mapping[str, Any]],
        output_dir: Path | None = None,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Review bounded candidate groups, preserving all unmerged candidates."""

        current = [dict(group) for group in sorted(groups, key=_candidate_group_sort_key)]
        if not current or all(_configured_feature_definitions(group) for group in current):
            return current
        prompt_limit = int(self.config.consolidation_max_prompt_chars)
        request_config = replace(self.config, max_prompt_chars=prompt_limit)
        round_summaries: list[dict[str, Any]] = []
        stopped_reason = "maximum_rounds_reached"
        policy = self.config.consolidation_policy
        pool_schema = MIXED_SCHEMA
        embedding_cache = None
        process_input = {
            "phase": "iterative_candidate_pool_merge_only_consolidation",
            "alias_merge_prompt_version": clinical_prompts.ALIAS_MERGE_PROMPT_VERSION,
            "consolidation_schema": CONSOLIDATION_SCHEMA_VERSION,
            "global_candidate_pool_schema": pool_schema,
            "consolidation_policy": policy.public_dict(),
            "outer_fold": int(outer_fold),
            "consolidation_batch_size": int(self.config.consolidation_batch_size),
            "consolidation_max_rounds": int(self.config.consolidation_max_rounds),
            "consolidation_seed": int(seed),
            "features": [_candidate_pool_feature_view(group) for group in current],
            "configured_feature_names": [
                str(group["name"]) for group in current if _configured_feature_definitions(group)
            ],
        }
        process_fingerprint = _value_fingerprint(process_input)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                output_dir / "input.json",
                {**process_input, "input_fingerprint": process_fingerprint},
            )

        for round_number in range(1, int(self.config.consolidation_max_rounds) + 1):
            round_dir = output_dir / f"round_{round_number:03d}" if output_dir is not None else None
            if not current:
                stopped_reason = "candidate_pool_empty"
                break
            if all(_configured_feature_definitions(group) for group in current):
                stopped_reason = "only_explicit_features_remain"
                break
            matrix = None
            if len(current) > self.config.consolidation_batch_size and policy.semantic_fraction:
                if embedding_cache is None:
                    embedding_cache = CandidateEmbeddingCache(
                        output_dir / "semantic_embeddings.npz" if output_dir is not None else None,
                        policy, _encode_candidate_selection_texts,
                    )
                LOGGER.info("Stage 2 semantic grouping fold=%s round=%s candidates=%s", outer_fold, round_number, len(current))
                matrix = embedding_cache.matrix([_candidate_registry_text(group) for group in current])
            batches, batch_orderings = mixed_batches(
                current, batch_size=self.config.consolidation_batch_size,
                round_number=round_number, seed=int(seed) + 1_000_003 * int(outer_fold),
                policy=policy, embeddings=matrix,
            )

            batch_responses: dict[int, dict[str, Any]] = {}
            jobs: list[dict[str, Any]] = []
            for batch_number, batch in enumerate(batches, start=1):
                group_names = [str(group["name"]) for group in batch]
                configured_feature_names = [
                    str(group["name"]) for group in batch if _configured_feature_definitions(group)
                ]
                if len(configured_feature_names) == len(batch):
                    batch_responses[batch_number] = {"merge_directives": []}
                    continue
                messages = _global_candidate_pool_prompt(
                    groups=batch,
                    configured_feature_names=configured_feature_names,
                    batch_ordering=batch_orderings[batch_number - 1],
                )
                prompt_chars = sum(len(message["content"]) for message in messages)
                if prompt_chars > prompt_limit:
                    raise ValueError(
                        "one Stage 2 candidate consolidation batch cannot fit the prompt "
                        f"budget in round {round_number}, batch {batch_number} "
                        f"({prompt_chars} > {prompt_limit}); reduce "
                        "stage2.consolidation_batch_size or increase "
                        "stage2.consolidation_max_prompt_chars"
                    )
                feature_views = [_candidate_pool_feature_view(group) for group in batch]
                group_descriptions = {
                    str(feature["name"]): tuple(_string_values(feature.get("descriptions")))
                    for feature in feature_views
                }
                jobs.append(
                    {
                        "batch_number": batch_number,
                        "output_dir": (
                            round_dir / f"batch_{batch_number:03d}"
                            if round_dir is not None
                            else None
                        ),
                        "input_value": {
                            "phase": "iterative_candidate_pool_batch_consolidation",
                            "global_candidate_pool_schema": pool_schema,
                            "consolidation_policy": policy.public_dict(),
                            "outer_fold": int(outer_fold),
                            "round": round_number,
                            "batch": batch_number,
                            "ordering": batch_orderings[batch_number - 1],
                            "consolidation_batch_size": int(self.config.consolidation_batch_size),
                            "consolidation_max_rounds": int(self.config.consolidation_max_rounds),
                            "consolidation_seed": int(seed),
                            "features": feature_views,
                            "configured_feature_names": configured_feature_names,
                        },
                        "messages": messages,
                        "validate": (
                            lambda value, names=tuple(group_names), configured=tuple(
                                configured_feature_names
                            ), descriptions=group_descriptions: (
                                _validate_global_candidate_pool_directives(
                                    value,
                                    group_names=names,
                                    configured_feature_names=configured,
                                    group_descriptions=descriptions,
                                )
                            )
                        ),
                    }
                )

            validation_fallbacks: dict[int, str] = {}
            if jobs:
                job_by_batch = {int(job["batch_number"]): job for job in jobs}
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, min(self.config.workers, len(jobs)))
                ) as executor:
                    futures = {
                        executor.submit(
                            _checkpointed_request_json,
                            output_dir=job["output_dir"],
                            input_value=job["input_value"],
                            messages=job["messages"],
                            config=request_config,
                            completion=self.completion,
                            validate=job["validate"],
                        ): int(job["batch_number"])
                        for job in jobs
                    }
                    for future in concurrent.futures.as_completed(futures):
                        batch_number = futures[future]
                        try:
                            batch_responses[batch_number] = future.result()
                        except Stage2ResponseValidationError as exc:
                            # Consolidation is an optional semantic reduction.
                            # Retaining every member of an invalid batch is the
                            # conservative, lossless fallback and necessarily
                            # preserves investigator-configured features.
                            validation_fallbacks[batch_number] = str(exc)
                            batch_responses[batch_number] = {"merge_directives": []}
                            job = job_by_batch[batch_number]
                            batch_output_dir = job["output_dir"]
                            if batch_output_dir is not None:
                                _write_json(
                                    Path(batch_output_dir) / "fallback.json",
                                    {
                                        "status": "conservative_passthrough",
                                        "completed_at": _now(),
                                        "round": round_number,
                                        "batch": batch_number,
                                        "retained_feature_names": [
                                            str(feature["name"])
                                            for feature in job["input_value"]["features"]
                                        ],
                                        "validation_error": str(exc),
                                    },
                                )
                            LOGGER.warning(
                                "Stage 2 consolidation round=%s batch=%s remained invalid; "
                                "retaining all %s supplied features unchanged (%s)",
                                round_number,
                                batch_number,
                                len(job["input_value"]["features"]),
                                exc,
                            )

            next_groups: list[dict[str, Any]] = []
            directive_count = 0
            merged_input_count = 0
            for batch_number, batch in enumerate(batches, start=1):
                response = batch_responses[batch_number]
                directives = list(response["merge_directives"])
                directive_count += len(directives)
                merged_input_count += sum(
                    max(0, len(_string_values(directive.get("inputs"))) - 1)
                    for directive in directives
                )
                next_groups.extend(
                    _apply_global_candidate_pool_directives(
                        batch,
                        directives,
                    )
                )

            next_groups, cross_batch_exact_merges = _coalesce_exact_candidate_group_names(
                next_groups
            )
            for group in next_groups:
                configured = _configured_feature_definitions(group)
                if len(configured) > 1:
                    raise ValueError(
                        "iterative Stage 2 consolidation attempted to merge distinct "
                        "investigator-configured features: "
                        f"{[feature['name'] for feature in configured]}"
                    )
                if configured and str(group["name"]) != str(configured[0]["name"]):
                    raise ValueError(
                        "iterative Stage 2 consolidation renamed an investigator-configured "
                        f"feature; expected {configured[0]['name']!r}, received "
                        f"{group['name']!r}"
                    )

            changed = bool(merged_input_count or cross_batch_exact_merges)
            round_summary = {
                "round": round_number,
                "input_groups": len(current),
                "batches": len(batches),
                "model_requested_batches": len(jobs),
                "explicit_only_batches": len(batches) - len(jobs),
                "merge_directives": directive_count,
                "merged_inputs_removed": merged_input_count,
                "cross_batch_exact_name_merges": cross_batch_exact_merges,
                "validation_fallback_batches": len(validation_fallbacks),
                "validation_fallback_batch_numbers": sorted(validation_fallbacks),
                "output_groups": len(next_groups),
                "changed": changed,
                "grouping_batches": dict(Counter(batch_orderings)),
                "relative_reduction": (len(current) - len(next_groups)) / len(current),
            }
            round_summaries.append(round_summary)
            if round_dir is not None:
                _write_json(
                    round_dir / "complete.json",
                    {
                        "status": "complete",
                        "completed_at": _now(),
                        "global_candidate_pool_schema": pool_schema,
                        **round_summary,
                    },
                )
            LOGGER.info(
                "Stage 2 iterative consolidation round=%s input_groups=%s "
                "grouping=%s batches=%s merge_directives=%s "
                "cross_batch_exact_merges=%s output_groups=%s changed=%s",
                round_number,
                len(current),
                dict(Counter(batch_orderings)),
                len(batches),
                directive_count,
                cross_batch_exact_merges,
                len(next_groups),
                changed,
            )
            current = next_groups
            if diminishing_returns(round_summaries, policy):
                stopped_reason = "diminishing_returns"
                LOGGER.info("Stage 2 mixed consolidation stopped fold=%s round=%s: last %s successful rounds each removed <%.3f%%",
                            outer_fold, round_number, policy.early_stop_patience, 100 * policy.early_stop_min_reduction)
                break
            if not current:
                stopped_reason = "candidate_pool_empty"
                break
            if all(_configured_feature_definitions(group) for group in current):
                stopped_reason = "only_explicit_features_remain"
                break
            if not changed and len(batches) == 1:
                stopped_reason = (
                    "single_batch_validation_fallback"
                    if validation_fallbacks
                    else "single_batch_no_change"
                )
                break

        if output_dir is not None:
            _write_json(
                output_dir / "result.json",
                {
                    "groups": current,
                    "rounds": round_summaries,
                },
            )
            _write_json(
                output_dir / "complete.json",
                {
                    "status": "complete",
                    "completed_at": _now(),
                    "input_fingerprint": process_fingerprint,
                    "global_candidate_pool_schema": pool_schema,
                    "consolidation_policy": policy.public_dict(),
                    "rounds_executed": len(round_summaries),
                    "stopped_reason": stopped_reason,
                    "output_groups": len(current),
                    "validation_fallback_batches": sum(
                        int(summary["validation_fallback_batches"]) for summary in round_summaries
                    ),
                },
            )
        return current

    def _consolidate_candidates(
        self,
        *,
        outer_fold: int,
        candidates: Sequence[Mapping[str, Any]],
        evidence_packets: Sequence[Mapping[str, Any]],
        output_dir: Path | None = None,
        seed: int = 42,
    ) -> Mapping[str, Any]:
        """Iteratively consolidate candidate batches, then operationalize routed groups."""

        configured_candidates = [
            {
                "candidate_id": f"configured_explicit_feature_{index:04d}",
                "architecture": CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE,
                "name": feature.name,
                "description": feature.description,
                "value_type": feature.value_type,
                "supporting_packet_ids": [],
                "evidence_axes": [],
                "evidence_rationale": "Investigator-specified feature for this analysis.",
                "caveats": feature.caveats,
                "configured_feature_definitions": [feature.as_definition()],
            }
            for index, feature in enumerate(self.config.explicit_features, start=1)
        ]
        all_candidates = [*configured_candidates, *(dict(candidate) for candidate in candidates)]
        input_candidate_ids = [str(candidate["candidate_id"]) for candidate in all_candidates]
        if len(input_candidate_ids) != len(set(input_candidate_ids)):
            raise ValueError("Stage 2 consolidation received duplicate candidate IDs")
        original_ids = [
            origin
            for candidate in all_candidates
            for origin in (
                _string_values(candidate.get("origin_candidate_ids"))
                or [str(candidate["candidate_id"])]
            )
        ]
        if len(original_ids) != len(set(original_ids)):
            raise ValueError("Stage 2 consolidation received duplicate origin candidate IDs")
        packet_by_id = {str(packet["packet_id"]): packet for packet in evidence_packets}
        if len(packet_by_id) != len(evidence_packets):
            raise ValueError("Stage 2 ontology evidence contains duplicate packet IDs")
        cited_packet_ids = {
            packet_id
            for candidate in all_candidates
            for packet_id in _string_values(candidate.get("supporting_packet_ids"))
        }
        unknown_packet_ids = sorted(cited_packet_ids - set(packet_by_id))
        if unknown_packet_ids:
            raise ValueError(
                "Stage 2 candidates cite ontology evidence outside the supplied packet "
                f"set: {unknown_packet_ids[:8]}"
            )
        groups = _materialize_exact_name_groups(all_candidates)
        LOGGER.info(
            "Stage 2 candidate pool candidates=%s distinct_names=%s",
            len(all_candidates),
            len(groups),
        )
        retained_groups = self._consolidate_candidate_pool(
            outer_fold=outer_fold,
            groups=groups,
            output_dir=(
                output_dir / "candidate_pool_consolidation_mixed"
                if output_dir is not None else None
            ),
            seed=seed,
        )
        features_by_group_id: dict[str, dict[str, Any]] = {}
        if retained_groups:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(self.config.workers, len(retained_groups)))
            ) as executor:
                futures = {
                    executor.submit(
                        self._operationalize_candidate_group,
                        group=group,
                        packet_by_id=packet_by_id,
                        output_dir=(
                            output_dir / "operationalization" / f"group_{group_index:03d}"
                            if output_dir is not None
                            else None
                        ),
                    ): str(group["candidate_id"])
                    for group_index, group in enumerate(retained_groups, start=1)
                }
                for future in concurrent.futures.as_completed(futures):
                    features_by_group_id[futures[future]] = future.result()
        features = [features_by_group_id[str(group["candidate_id"])] for group in retained_groups]

        dispositions: dict[str, dict[str, str]] = {}
        original_id_set = set(original_ids)
        for group, feature in zip(retained_groups, features):
            origins = [
                origin
                for origin in _string_values(group.get("origin_candidate_ids"))
                if origin in original_id_set
            ]
            for origin_index, origin in enumerate(origins):
                dispositions[origin] = {
                    "status": "retained" if origin_index == 0 else "merged",
                    "feature_name": str(feature["name"]),
                    "reason": (
                        "Retained as the canonical candidate for this scalar measurement."
                        if origin_index == 0
                        else "Merged with candidates describing the same scalar measurement."
                    ),
                }
        missing_origins = sorted(original_id_set - set(dispositions))
        if missing_origins:
            raise RuntimeError(
                "merge-only Stage 2 consolidation lost candidate origin IDs: "
                f"{missing_origins[:8]}"
            )
        return {"features": features, "candidate_dispositions": dispositions}

    def _run_outer_fold_with_recovery(self, **kwargs: Any) -> Mapping[str, Any]:
        """Retry exhausted requests by resuming the same fold's checkpoints.

        Each call to _run_outer_fold finishes shutting down its own executors
        before raising, so the retry cannot overlap writers in that fold.
        Other outer folds keep running during the bounded backoff.
        """

        outer_fold = int(kwargs["outer_fold"])
        output_dir = Path(kwargs["output_dir"])
        maximum = self.config.outer_fold_recovery_attempts
        started_at = _now()

        def record(status: str, attempt: int, **fields: Any) -> None:
            value = {
                "schema_version": "stage2_outer_fold_recovery_v1",
                "outer_fold": outer_fold,
                "invocation_started_at": started_at,
                "updated_at": _now(),
                "status": status,
                "attempt": attempt,
                "recoveries_used": attempt - 1,
                "maximum_recoveries": maximum,
                **fields,
            }
            _write_json(output_dir / "recovery_status.json", value)
            with (output_dir / "recovery_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, allow_nan=False) + "\n")

        for attempt in range(1, maximum + 2):
            record("running", attempt)
            try:
                result = self._run_outer_fold(**kwargs)
            except Stage2RequestExhaustedError as exc:
                error = {"error_type": type(exc).__name__, "error": str(exc)}
                if attempt > maximum:
                    record("failed", attempt, **error)
                    raise
                delay = self.config.outer_fold_recovery_backoff * attempt
                record("retry_wait", attempt, retry_after_seconds=delay, **error)
                LOGGER.warning(
                    "Stage 2 recovering outer_fold=%s retry=%s/%s after %.1fs; "
                    "resuming completed checkpoints (%s: %s)",
                    outer_fold, attempt, maximum, delay, type(exc).__name__, exc,
                )
                time.sleep(delay)
            except Exception as exc:
                record("failed", attempt, error_type=type(exc).__name__, error=str(exc))
                raise
            else:
                record("complete", attempt)
                if attempt > 1:
                    LOGGER.info("Stage 2 recovered outer_fold=%s retries=%s", outer_fold, attempt - 1)
                return result
        raise RuntimeError("unreachable Stage 2 fold recovery state")

    def _run_outer_fold(
        self,
        *,
        outer_fold: int,
        packets: Sequence[Mapping[str, Any]],
        output_dir: Path,
        dataset: pd.DataFrame | None = None,
        split: Mapping[str, Any] | None = None,
        unit_id_column: str = "patient_id",
        text_column: str = "clinical_text",
        treatment_column: str = "treatment_indicator",
        outcome_column: str = "outcome_indicator",
        outcome_type: str = "binary",
        inner_folds: int = 5,
        seed: int = 42,
    ) -> Mapping[str, Any]:
        discovery_packets = list(packets)
        complete_path = output_dir / "complete.json"
        features_path = output_dir / "feature_definitions.json"
        definitions_complete_path = output_dir / "definitions_complete.json"
        if self.config.frozen_feature_definition_model and not (
            features_path.is_file() and definitions_complete_path.is_file()
        ):
            raise RuntimeError(
                "frozen_feature_definition_model requires completed feature definitions "
                f"for outer fold {outer_fold}; refusing to regenerate them with another model"
            )
        interpreted_candidates_path = output_dir / "interpreted_candidates.json"
        definition_inputs = _feature_definition_input_value(
            config=self.config,
            clinical_question=self.clinical_question,
            outer_fold=outer_fold,
            discovery_packets=discovery_packets,
            seed=seed,
        )
        evidence_input_fingerprint = _value_fingerprint(definition_inputs)
        definitions_state = (
            json.loads(definitions_complete_path.read_text(encoding="utf-8"))
            if definitions_complete_path.is_file()
            else {}
        )
        if complete_path.is_file() and dataset is not None:
            # Do not trust the coarse outer marker as a science checkpoint.
            # The analysis layer reconstructs the current selection and
            # estimation fingerprints before reusing those artifacts.
            LOGGER.info(
                "revalidate completed Stage 2 outer-fold semantic checkpoints fold=%s",
                outer_fold,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "input_packets.jsonl", discovery_packets)

        by_architecture: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for packet in discovery_packets:
            by_architecture[str(packet["architecture"])].append(packet)
        if features_path.is_file():
            if definitions_state.get("evidence_input_fingerprint") == evidence_input_fingerprint:
                final = json.loads(features_path.read_text(encoding="utf-8"))
            else:
                raise RuntimeError(
                    f"Stage 2 outer fold {outer_fold} has feature definitions from a "
                    "different evidence plan, feature-definition policy, or explicit-feature "
                    "configuration. Preserve the old output for audit and use a fresh "
                    "Stage 2 output directory."
                )
        else:
            jobs: list[tuple[str, int, list[Mapping[str, Any]], Path]] = []
            for architecture_index, architecture in enumerate(sorted(by_architecture), start=1):
                batches = _partition_interpretation_packets(
                    by_architecture[architecture],
                    architecture=architecture,
                    max_prompt_chars=self.config.max_prompt_chars,
                )
                for batch_index, batch in enumerate(batches, start=1):
                    jobs.append(
                        (
                            architecture,
                            batch_index,
                            batch,
                            output_dir
                            / "interpretations"
                            / f"architecture_{architecture_index:02d}"
                            / f"batch_{batch_index:03d}",
                        )
                    )
            LOGGER.info(
                "Stage 2 outer_fold=%s architectures=%s interpretation_batches=%s workers=%s",
                outer_fold,
                len(by_architecture),
                len(jobs),
                min(self.config.workers, len(jobs)),
            )
            results: list[tuple[str, int, Mapping[str, Any]]] = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(self.config.workers, len(jobs)))
            ) as executor:
                futures = {
                    executor.submit(
                        self._interpret_batch,
                        architecture=architecture,
                        packets=batch,
                        output_dir=job_dir,
                    ): (architecture, batch_index)
                    for architecture, batch_index, batch, job_dir in jobs
                }
                for future in concurrent.futures.as_completed(futures):
                    architecture, batch_index = futures[future]
                    results.append((architecture, batch_index, future.result()))

            packet_by_id = {
                str(packet["packet_id"]): packet for packet in discovery_packets
            }
            if len(packet_by_id) != len(discovery_packets):
                raise ValueError("Stage 2 feature discovery received duplicate packet IDs")
            candidates: list[dict[str, Any]] = []
            for architecture, _batch_index, result in sorted(
                results,
                key=lambda item: (item[0], item[1]),
            ):
                for concept in result["concepts"]:
                    supporting_packet_ids = [
                        str(packet_id) for packet_id in concept["supporting_packet_ids"]
                    ]
                    unknown_packet_ids = sorted(set(supporting_packet_ids) - set(packet_by_id))
                    if unknown_packet_ids:
                        raise RuntimeError(
                            "Stage 2 interpretation checkpoint cites packets outside "
                            f"the current evidence plan: {unknown_packet_ids[:8]}"
                        )
                    evidence_axes = sorted(
                        {
                            str(axis)
                            for packet_id in supporting_packet_ids
                            for axis in packet_by_id[packet_id]["observable_axes"]
                        }
                    )
                    supporting_architectures = sorted(
                        {
                            source_architecture
                            for packet_id in supporting_packet_ids
                            for source_architecture in _packet_support_architectures(
                                packet_by_id[packet_id]
                            )
                        }
                    )
                    candidates.append(
                        {
                            "candidate_id": f"candidate_{len(candidates) + 1:04d}",
                            "architecture": architecture,
                            **concept,
                            "supporting_packet_ids": supporting_packet_ids,
                            "supporting_architectures": supporting_architectures,
                            "evidence_axes": evidence_axes,
                        }
                    )
            _write_json(interpreted_candidates_path, candidates)
            LOGGER.info(
                "Stage 2 exhaustive semantic-card discovery outer_fold=%s candidates=%s",
                outer_fold,
                len(candidates),
            )

            if not candidates and not self.config.explicit_features:
                final = {
                    "outer_fold": outer_fold,
                    "features": [],
                    "candidate_dispositions": {},
                }
            else:
                consolidated = self._consolidate_candidates(
                    outer_fold=outer_fold,
                    candidates=candidates,
                    evidence_packets=discovery_packets,
                    output_dir=output_dir / "consolidation",
                    seed=seed,
                )
                features = []
                for index, feature in enumerate(consolidated["features"], start=1):
                    features.append(
                        {
                            "feature_id": f"outer_{outer_fold:03d}_feature_{index:03d}",
                            **feature,
                        }
                    )
                final = {
                    "outer_fold": outer_fold,
                    "features": features,
                    "candidate_dispositions": consolidated["candidate_dispositions"],
                }
            _write_json(features_path, final)
            _write_json(
                definitions_complete_path,
                {
                    "status": "complete",
                    "completed_at": _now(),
                    "evidence_input_fingerprint": evidence_input_fingerprint,
                    "feature_definition_input_schema": (
                        FEATURE_DEFINITION_INPUT_SCHEMA_VERSION
                    ),
                    "consolidation_schema": CONSOLIDATION_SCHEMA_VERSION,
                    "global_candidate_pool_schema": MIXED_SCHEMA,
                    "candidate_selection": "none_exhaustive_discovery",
                    "consolidation_batch_size": int(self.config.consolidation_batch_size),
                    "consolidation_max_rounds": int(self.config.consolidation_max_rounds),
                    "consolidation_policy": self.config.consolidation_policy.public_dict(),
                    "extraction_ontology_feedback_schema": (
                        EXTRACTION_ONTOLOGY_FEEDBACK_SCHEMA_VERSION
                    ),
                    "ontology_refinement_min_failure_patients": int(
                        self.config.ontology_refinement_min_failure_patients
                    ),
                    "max_ontology_refinement_rounds": int(
                        self.config.max_ontology_refinement_rounds
                    ),
                    "architectures": len(by_architecture),
                    "packets": len(discovery_packets),
                    "discovery_packets": len(discovery_packets),
                    "features": len(final["features"]),
                },
            )

        if dataset is None:
            _write_json(
                complete_path,
                {
                    "status": "complete",
                    "phase": "feature_definitions",
                    "completed_at": _now(),
                    "evidence_input_fingerprint": evidence_input_fingerprint,
                    "features": len(final["features"]),
                },
            )
            return final
        if split is None:
            raise ValueError(f"Stage 2 outer fold {outer_fold} has no row split")

        # Analysis contains the high-context, one-patient extraction calls. Its
        # review planner still uses max_prompt_chars, but transport must accept
        # extraction prompts up to their independent limit.
        primary_analysis_request_config = replace(
            self.config,
            max_prompt_chars=max(
                self.config.max_prompt_chars,
                self.config.extraction_max_prompt_chars,
            ),
        )
        extraction_analysis_request_config = (
            replace(
                self.extraction_request_config,
                max_prompt_chars=self.config.extraction_max_prompt_chars,
            )
            if self.extraction_request_config is not None
            else None
        )

        def request_analysis_json(
            messages: Sequence[Mapping[str, str]],
            validate: Callable[[Mapping[str, Any]], dict[str, Any]],
            *,
            request_kind: str = "interpretation",
            initial_reasoning_effort: str | None = None,
            repair_context: Mapping[str, Any] | None = None,
            validation_event_observer: (
                Callable[[Mapping[str, Any]], None] | None
            ) = None,
            conservative_validation_fallback: Mapping[str, Any] | None = None,
            fallback_after_same_error: int = 3,
        ) -> dict[str, Any]:
            if request_kind == "extraction":
                if (
                    extraction_analysis_request_config is None
                    or self.extraction_completion is None
                ):
                    raise RuntimeError(
                        "Stage 2 extraction request attempted without stage2.extraction_llm"
                    )
                request_config = extraction_analysis_request_config
                completion = self.extraction_completion
            else:
                request_config = primary_analysis_request_config
                completion = self.completion
            return _request_json(
                messages=messages,
                config=request_config,
                completion=completion,
                validate=validate,
                request_kind=request_kind,
                initial_reasoning_effort=initial_reasoning_effort,
                prompt_token_counter=(
                    (
                        lambda candidate: prompt_token_count(
                            self.extraction_tokenizer,
                            candidate,
                        )
                    )
                    if request_kind == "extraction"
                    and self.extraction_tokenizer is not None
                    else None
                ),
                context_window_tokens=(
                    self.config.extraction_context_window_tokens
                    if request_kind == "extraction"
                    else None
                ),
                context_margin_tokens=(
                    self.config.extraction_context_margin_tokens
                    if request_kind == "extraction"
                    else 0
                ),
                repair_context=repair_context,
                validation_event_observer=validation_event_observer,
                conservative_validation_fallback=conservative_validation_fallback,
                fallback_after_same_error=fallback_after_same_error,
            )

        def request_selection_consolidation_json(
            messages: Sequence[Mapping[str, str]],
            validate: Callable[[Mapping[str, Any]], dict[str, Any]],
            *,
            request_kind: str = "interpretation",
            repair_context: Mapping[str, Any] | None = None,
            validation_event_observer: (
                Callable[[Mapping[str, Any]], None] | None
            ) = None,
            conservative_validation_fallback: Mapping[str, Any] | None = None,
            fallback_after_same_error: int = 3,
        ) -> dict[str, Any]:
            if (
                self.selection_consolidation_request_config is None
                or self.selection_consolidation_completion is None
            ):
                return request_analysis_json(
                    messages,
                    validate,
                    request_kind=request_kind,
                    repair_context=repair_context,
                    validation_event_observer=validation_event_observer,
                    conservative_validation_fallback=(
                        conservative_validation_fallback
                    ),
                    fallback_after_same_error=fallback_after_same_error,
                )
            if request_kind != "interpretation":
                raise ValueError(
                    "the selection-consolidation runtime route accepts only "
                    "interpretation requests"
                )
            return _request_json(
                messages=messages,
                config=self.selection_consolidation_request_config,
                completion=self.selection_consolidation_completion,
                validate=validate,
                request_kind="interpretation",
                repair_context=repair_context,
                validation_event_observer=validation_event_observer,
                conservative_validation_fallback=conservative_validation_fallback,
                fallback_after_same_error=fallback_after_same_error,
            )

        analysis = run_fold_analysis(
            dataset=dataset,
            definitions=final["features"],
            split=split,
            clinical_question=self.clinical_question,
            unit_id_column=unit_id_column,
            text_column=text_column,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=outcome_type,
            inner_folds=inner_folds,
            seed=seed + 100_000 * outer_fold,
            output_dir=output_dir,
            request_json=request_analysis_json,
            selection_consolidation_request_json=(
                request_selection_consolidation_json
            ),
            config=self.config,
            extraction_tokenizer=self.extraction_tokenizer,
            stage1_packets=discovery_packets,
        )
        completed = {
            "outer_fold": outer_fold,
            "features": analysis["features"],
            "candidate_dispositions": final.get("candidate_dispositions", {}),
            "review_rounds": analysis["review_rounds"],
            "evaluation_rounds": analysis["evaluation_rounds"],
            "review_converged": analysis["review_converged"],
            "review_convergence": analysis["review_convergence"],
            "harmonization_validation_fallbacks": analysis[
                "harmonization_validation_fallbacks"
            ],
            "ontology_refinement_rounds": analysis["ontology_refinement_rounds"],
            "estimation": analysis["estimation"],
        }
        _write_json(
            complete_path,
            {
                "status": "complete",
                "phase": "causal_estimation",
                "completed_at": _now(),
                "evidence_input_fingerprint": evidence_input_fingerprint,
                "features": len(completed["features"]),
                "review_rounds": completed["review_rounds"],
                "evaluation_rounds": completed["evaluation_rounds"],
                "review_converged": completed["review_converged"],
                "review_convergence": completed["review_convergence"],
                "harmonization_validation_fallbacks": completed[
                    "harmonization_validation_fallbacks"
                ],
                "ontology_refinement_rounds": completed["ontology_refinement_rounds"],
                "estimation": completed["estimation"],
            },
        )
        return completed

    def run(
        self,
        *,
        handoff_path: Path,
        output_dir: Path,
        dataset: pd.DataFrame | None = None,
        split_provenance_path: Path | None = None,
        unit_id_column: str = "patient_id",
        text_column: str = "clinical_text",
        treatment_column: str = "treatment_indicator",
        outcome_column: str = "outcome_indicator",
        outcome_type: str = "binary",
        inner_folds: int = 5,
        seed: int = 42,
    ) -> Mapping[str, Any]:
        handoff_path = Path(handoff_path)
        output_dir = Path(output_dir)
        if dataset is not None and (
            self.extraction_request_config is None or self.extraction_completion is None
        ):
            raise ValueError(
                "dataset-backed Stage 2 requires stage2.extraction_llm so patient "
                "value extraction has its own request policy"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        self._check_and_record_model_identity(output_dir)
        _write_json(output_dir / "config.json", self.config.public_dict())
        compiled_packets, compilation_summary = self._load_or_compile_evidence(
            handoff_path=handoff_path,
            output_dir=output_dir,
            seed=seed,
        )
        discovery_packets = compiled_packets

        packets_by_outer: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for packet in discovery_packets:
            packets_by_outer[int(packet["outer_fold"])].append(packet)
        splits: dict[int, dict[str, Any]] = {}
        if dataset is not None:
            dataset = dataset.reset_index(drop=True)
            required_columns = {
                unit_id_column,
                text_column,
                treatment_column,
                outcome_column,
            }
            missing = sorted(required_columns - set(dataset.columns))
            if missing:
                raise ValueError(f"Stage 2 dataset is missing configured columns: {missing}")
            treatment_numeric = pd.to_numeric(dataset[treatment_column], errors="coerce")
            treatment_values = set(treatment_numeric.dropna().unique())
            if treatment_numeric.isna().any() or not treatment_values <= {0.0, 1.0}:
                raise ValueError("Stage 2 treatment must be a complete binary 0/1 column")
            if outcome_type not in {"binary", "continuous"}:
                raise ValueError("Stage 2 outcome_type must be binary or continuous")
            outcome_numeric = pd.to_numeric(dataset[outcome_column], errors="coerce")
            if outcome_numeric.isna().any():
                raise ValueError("Stage 2 outcome column must be complete and numeric")
            if outcome_type == "binary" and not set(outcome_numeric.unique()) <= {0.0, 1.0}:
                raise ValueError("binary Stage 2 outcome must contain only 0 and 1")
            splits = _load_stage2_splits(
                provenance_path=split_provenance_path,
                dataset_rows=len(dataset),
                outer_fold_ids=sorted(packets_by_outer),
            )
            _verify_stage2_splits_against_handoff(
                splits,
                handoff_path=handoff_path,
                outer_fold_ids=sorted(packets_by_outer),
            )
        outer_fold_ids = sorted(packets_by_outer)
        fold_results_by_id: dict[int, Mapping[str, Any]] = {}
        if outer_fold_ids:
            fold_workers = min(len(outer_fold_ids), self.config.workers)
            LOGGER.info(
                "Stage 2 outer-fold execution folds=%s fold_workers=%s "
                "global_request_workers=%s "
                "role_selection=all_evidence_llm_adjudication",
                len(outer_fold_ids),
                fold_workers,
                self.config.workers,
            )
            from .stage2_modeling_pool import modeling_session

            with modeling_session(
                self.config.modeling_workers, parallel_fits=fold_workers
            ), concurrent.futures.ThreadPoolExecutor(
                max_workers=fold_workers,
                thread_name_prefix="stage2-outer-fold",
            ) as outer_executor:
                futures = {
                    outer_executor.submit(
                        self._run_outer_fold_with_recovery,
                        outer_fold=outer_fold,
                        packets=packets_by_outer[outer_fold],
                        output_dir=output_dir / f"outer_{outer_fold:03d}",
                        dataset=dataset,
                        split=splits.get(outer_fold),
                        unit_id_column=unit_id_column,
                        text_column=text_column,
                        treatment_column=treatment_column,
                        outcome_column=outcome_column,
                        outcome_type=outcome_type,
                        inner_folds=inner_folds,
                        seed=seed,
                    ): outer_fold
                    for outer_fold in outer_fold_ids
                }
                for future in concurrent.futures.as_completed(futures):
                    outer_fold = futures[future]
                    try:
                        fold_results_by_id[outer_fold] = future.result()
                    except Exception:
                        # Report now: executor shutdown waits for sibling folds,
                        # whose successful fits can otherwise hide this failure.
                        LOGGER.exception(
                            "Stage 2 failed outer_fold=%s; waiting for sibling folds to finish",
                            outer_fold,
                        )
                        raise
                    LOGGER.info("Stage 2 completed outer_fold=%s", outer_fold)
        fold_results = [fold_results_by_id[outer_fold] for outer_fold in outer_fold_ids]
        _write_jsonl(output_dir / "features_by_outer_fold.jsonl", fold_results)
        name_counts: Counter[str] = Counter()
        for result in fold_results:
            names_in_fold = {
                re.sub(r"[^a-z0-9]+", "_", str(feature["name"]).lower()).strip("_")
                for feature in result["features"]
            }
            name_counts.update(names_in_fold)
        summary = {
            "outer_folds": len(fold_results),
            "candidate_discovery_source": "all_semantic_evidence_cards",
            "evidence_packets": len(discovery_packets),
            "candidate_discovery_packets": len(discovery_packets),
            "compiled_evidence_packets": len(compiled_packets),
            "evidence_compiler": EVIDENCE_COMPILER_VERSION,
            "evidence_compilation_path": str(output_dir / "evidence_compilation"),
            "evidence_compilation": compilation_summary,
            "features_by_fold": {
                str(result["outer_fold"]): len(result["features"]) for result in fold_results
            },
            "feature_name_fold_counts": dict(sorted(name_counts.items())),
            "features_path": str(output_dir / "features_by_outer_fold.jsonl"),
        }
        convergence_results = [
            result for result in fold_results if "review_converged" in result
        ]
        if convergence_results:
            summary["review_convergence_by_fold"] = {
                str(result["outer_fold"]): result.get("review_convergence")
                for result in convergence_results
            }
            summary["nonconverged_outer_folds"] = [
                int(result["outer_fold"])
                for result in convergence_results
                if result.get("review_converged") is False
            ]
        summary["harmonization_validation_fallbacks_by_fold"] = {
            str(result["outer_fold"]): list(
                result.get("harmonization_validation_fallbacks") or []
            )
            for result in fold_results
        }
        summary["outer_folds_with_harmonization_validation_fallbacks"] = [
            int(result["outer_fold"])
            for result in fold_results
            if result.get("harmonization_validation_fallbacks")
        ]
        artifacts = [
            str(output_dir / "features_by_outer_fold.jsonl"),
            str(output_dir / "summary.json"),
        ]
        if dataset is not None:
            prediction_frames = []
            for result in fold_results:
                outer_fold = int(result["outer_fold"])
                frame = pd.read_csv(
                    output_dir / f"outer_{outer_fold:03d}" / "estimation" / "predictions.csv"
                )
                frame.insert(1, "outer_fold", outer_fold)
                prediction_frames.append(frame)
            predictions = pd.concat(prediction_frames, ignore_index=True)
            row_ids = predictions["_oci_row_id"].astype(int).tolist()
            if len(row_ids) != len(dataset) or set(row_ids) != set(range(len(dataset))):
                raise ValueError(
                    "outer heldout predictions must cover every dataset row exactly once"
                )
            if len(set(row_ids)) != len(row_ids):
                raise ValueError("a dataset row appears in more than one outer heldout fold")
            predictions = predictions.sort_values("_oci_row_id").reset_index(drop=True)
            predictions_path = output_dir / "cross_fitted_predictions.csv"
            temporary = predictions_path.with_name(f".{predictions_path.name}.{os.getpid()}.tmp")
            predictions.to_csv(temporary, index=False)
            os.replace(temporary, predictions_path)
            oracle_ite_evaluation = _evaluate_stage2_oracle_ite(
                prediction_path=predictions_path,
                dataset=dataset,
                output_dir=output_dir,
                outcome_type=outcome_type,
            )
            oracle_overall = dict(oracle_ite_evaluation.get("overall") or {})
            if "effect_eligible" in predictions:
                eligible = predictions["effect_eligible"].to_numpy(dtype=bool)
                if predictions.loc[~eligible, ["aipw_score", "estimated_cate"]].notna().any().any():
                    raise ValueError("propensity-excluded patients have effect estimates")
                if predictions.loc[eligible, "aipw_score"].isna().any():
                    raise ValueError("eligible patients have missing AIPW scores")
            elif self.config.min_propensity is not None or self.config.max_propensity is not None:
                raise ValueError("bounded effect estimation requires patient eligibility flags")
            scores = predictions["aipw_score"].to_numpy(dtype=float)
            scores = scores[np.isfinite(scores)]
            if not len(scores):
                raise ValueError("cross-fitted Stage 2 estimation produced no finite AIPW scores")
            ate = float(np.mean(scores))
            standard_error = (
                float(np.std(scores, ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else None
            )
            causal_estimate = {
                "estimator": "cross-fitted_aipw_with_fold_trained_nuisance_models",
                "estimand": (
                    "average_treatment_effect_in_propensity_eligible_population"
                    if self.config.min_propensity is not None
                    or self.config.max_propensity is not None
                    else "average_treatment_effect"
                ),
                "effect_estimation_rows": len(scores),
                "excluded_rows": len(predictions) - len(scores),
                "propensity_overlap": overlap_diagnostics(
                    predictions["propensity"],
                    self.config.min_propensity,
                    self.config.max_propensity,
                ),
                "nuisance_calibration": {
                    "treatment": calibration_diagnostics(
                        predictions["treatment"], predictions["propensity"], binary=True
                    ),
                    "outcome_factual": calibration_diagnostics(
                        predictions["outcome"],
                        np.where(
                            predictions["treatment"] == 1, predictions["mu1"], predictions["mu0"]
                        ),
                        binary=outcome_type == "binary",
                    ),
                },
                "rows": len(predictions),
                "ate": ate,
                "standard_error": standard_error,
                "confidence_interval_95": (
                    [ate - 1.96 * standard_error, ate + 1.96 * standard_error]
                    if standard_error is not None
                    else None
                ),
                "mean_estimated_cate": float(predictions["estimated_cate"].mean()),
                "oracle_ite_pearson_correlation": oracle_overall.get("pearson_correlation"),
                "oracle_ite_spearman_correlation": oracle_overall.get("spearman_correlation"),
                "oracle_ite_evaluation": oracle_ite_evaluation,
                "predictions_path": str(predictions_path),
            }
            causal_path = output_dir / "causal_estimate.json"
            _write_json(causal_path, causal_estimate)
            summary.update(
                {
                    "phase": "causal_estimation",
                    "causal_estimate": causal_estimate,
                    "cross_fitted_predictions_path": str(predictions_path),
                }
            )
            artifacts.extend(
                [
                    str(predictions_path),
                    str(causal_path),
                    str(output_dir / "posthoc_oracle_ite_metrics.json"),
                ]
            )
            if oracle_ite_evaluation.get("available"):
                artifacts.append(str(output_dir / "posthoc_predictions_with_oracle_ite.csv"))
        else:
            summary["phase"] = "feature_definitions"
        _write_json(output_dir / "summary.json", summary)
        return {
            "artifacts": artifacts,
            **summary,
        }


def _all_gpu_managed_vllm_config(
    active: ManagedVLLMConfig,
    other: ManagedVLLMConfig,
    *,
    role: str,
) -> ManagedVLLMConfig:
    """Expand one alternately loaded model across both allowed managed GPU sets."""

    active.validate()
    other.validate()
    all_gpus = tuple(dict.fromkeys((*active.gpus, *other.gpus)))
    if all_gpus == active.gpus:
        return active

    active_width = active.effective_gpus_per_server()
    if len(all_gpus) % active_width == 0:
        gpus_per_server = active_width
        server_count = len(all_gpus) // gpus_per_server
    else:
        # Equal-width replicas cannot consume the complete union in this case.
        # A single tensor-parallel server is the only homogeneous layout that
        # both uses every GPU and remains representable by ManagedVLLMConfig.
        gpus_per_server = len(all_gpus)
        server_count = 1
        LOGGER.info(
            "managed Stage 2 %s GPU count=%s is not divisible by its "
            "configured tensor-parallel width=%s; using one all-GPU server",
            role,
            len(all_gpus),
            active_width,
        )

    existing_ports = list(active.effective_ports())
    ports = existing_ports[:server_count]
    next_port = max(existing_ports) + 1
    while len(ports) < server_count:
        if next_port > 65_535:
            raise ValueError(
                f"Stage 2 cannot allocate enough all-GPU {role} ports below 65536"
            )
        if next_port not in ports:
            ports.append(next_port)
        next_port += 1

    combined = replace(
        active,
        server_count=server_count,
        gpus=all_gpus,
        gpus_per_server=gpus_per_server,
        ports=tuple(ports),
    )
    combined.validate()
    return combined


def _all_gpu_interpretation_vllm_config(
    primary: ManagedVLLMConfig,
    extraction: ManagedVLLMConfig,
) -> ManagedVLLMConfig:
    """Expand the alternately loaded primary model across all allowed GPUs."""

    return _all_gpu_managed_vllm_config(
        primary,
        extraction,
        role="interpretation",
    )


def _all_gpu_extraction_vllm_config(
    primary: ManagedVLLMConfig,
    extraction: ManagedVLLMConfig,
) -> ManagedVLLMConfig:
    """Expand the alternately loaded extraction model across all allowed GPUs."""

    return _all_gpu_managed_vllm_config(
        extraction,
        primary,
        role="extraction",
    )


def run_plain_handoff_stage2(
    *,
    handoff_path: Path,
    output_dir: Path,
    clinical_question: str,
    config: PlainHandoffStage2Config,
    completion: CompletionFunction | None = None,
    extraction_completion: CompletionFunction | None = None,
    extraction_tokenizer: Any | None = None,
    dataset: pd.DataFrame | None = None,
    split_provenance_path: Path | None = None,
    unit_id_column: str = "patient_id",
    text_column: str = "clinical_text",
    treatment_column: str = "treatment_indicator",
    outcome_column: str = "outcome_indicator",
    outcome_type: str = "binary",
    inner_folds: int = 5,
    seed: int = 42,
) -> Mapping[str, Any]:
    def run_with_config(
        runtime_config: PlainHandoffStage2Config,
        *,
        runtime_dataset: pd.DataFrame | None,
        runtime_primary_completion: CompletionFunction | None,
        runtime_extraction_completion: CompletionFunction | None,
    ) -> Mapping[str, Any]:
        runner = PlainHandoffStage2(
            config=runtime_config,
            clinical_question=clinical_question,
            completion=runtime_primary_completion,
            extraction_completion=runtime_extraction_completion,
            extraction_tokenizer=extraction_tokenizer,
        )
        return runner.run(
            handoff_path=handoff_path,
            output_dir=output_dir,
            dataset=runtime_dataset,
            split_provenance_path=split_provenance_path,
            unit_id_column=unit_id_column,
            text_column=text_column,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=outcome_type,
            inner_folds=inner_folds,
            seed=seed,
        )

    extraction = config.extraction_llm
    extraction_vllm = extraction.vllm if extraction is not None else None
    if extraction is not None and str(extraction.runtime_endpoint).strip():
        # An explicit continuation route replaces only the live transport. The
        # configured pool remains in the persisted checkpoint identity and must
        # not be launched.
        extraction_vllm = None
    if config.runtime_disable_extraction:
        if extraction_completion is not None:
            raise ValueError(
                "stage2.runtime_disable_extraction cannot be combined with a custom "
                "extraction completion"
            )
        extraction_completion = _DisabledExtractionCompletion()
        extraction_vllm = None
        LOGGER.info(
            "Stage 2 extraction is runtime-disabled; preserving extractor model "
            "identity without launching its managed vLLM pool"
        )
    if config.vllm is None and extraction_vllm is None:
        return run_with_config(
            config,
            runtime_dataset=dataset,
            runtime_primary_completion=completion,
            runtime_extraction_completion=extraction_completion,
        )

    managed_root = Path(output_dir) / "vllm_servers"

    def run_configured_managed_pools() -> Mapping[str, Any]:
        """Run with each managed model on its exact configured allocation."""

        if config.vllm is not None and extraction_vllm is not None:
            validate_managed_vllm_pool_isolation(
                config.vllm,
                extraction_vllm,
            )
        with ExitStack() as stack:
            runtime_config = config
            if config.vllm is not None:
                primary_endpoints = stack.enter_context(
                    launch_managed_vllm_servers(
                        config=config.vllm,
                        model=config.model,
                        api_key=config.api_key,
                        output_dir=managed_root / "orchestrator",
                    )
                )
                runtime_config = replace(
                    runtime_config,
                    endpoint=primary_endpoints[0],
                    runtime_endpoints=tuple(primary_endpoints),
                )
            if extraction is not None and extraction_vllm is not None:
                extraction_endpoints = stack.enter_context(
                    launch_managed_vllm_servers(
                        config=extraction_vllm,
                        model=extraction.model,
                        api_key=extraction.api_key,
                        output_dir=managed_root / "extractor",
                    )
                )
                runtime_extraction = replace(
                    extraction,
                    endpoint=extraction_endpoints[0],
                    runtime_endpoints=tuple(extraction_endpoints),
                )
                runtime_config = replace(
                    runtime_config,
                    extraction_llm=runtime_extraction,
                )
            return run_with_config(
                runtime_config,
                runtime_dataset=dataset,
                runtime_primary_completion=completion,
                runtime_extraction_completion=extraction_completion,
            )

    primary_vllm = config.vllm
    if (
        primary_vllm is not None
        and extraction_vllm is not None
        and config.model.strip() == extraction.model.strip()
    ):
        # A single model can handle both roles through per-request thinking
        # controls. Launch one pool over the GPU union, share its router, and
        # never enter the alternating-model/configured-split state machine.
        shared_vllm = _all_gpu_extraction_vllm_config(primary_vllm, extraction_vllm)
        phase_path = managed_root / "model_phase.json"

        def record_shared_phase(status: str) -> None:
            _write_json(phase_path, {
                "schema_version": MANAGED_MODEL_PHASE_SCHEMA_VERSION,
                "status": status,
                "active_role": "shared",
                "allocation_mode": "shared_model",
                "model": extraction.model,
                "roles": sorted(STAGE2_REQUEST_KINDS),
                "gpus": list(shared_vllm.gpus),
                "transition": 0,
                "recorded_at": _now(),
            })

        record_shared_phase("starting")
        LOGGER.info("start shared managed Stage 2 model=%s gpus=%s replicas=%s",
                    extraction.model, list(shared_vllm.gpus), shared_vllm.server_count)
        with launch_managed_vllm_servers(
            config=shared_vllm, model=extraction.model, api_key=extraction.api_key,
            output_dir=managed_root / "shared",
        ) as endpoints:
            shared_completion = _RoundRobinOpenAICompletion(tuple(endpoints))
            runtime_config = replace(
                config,
                endpoint=endpoints[0],
                api_key=extraction.api_key,
                vllm=shared_vllm,
                runtime_endpoints=tuple(endpoints),
                extraction_llm=replace(
                    extraction, endpoint=endpoints[0], vllm=shared_vllm,
                    runtime_endpoints=tuple(endpoints),
                ),
            )
            record_shared_phase("running")
            result = run_with_config(
                runtime_config,
                runtime_dataset=dataset,
                runtime_primary_completion=completion or shared_completion,
                runtime_extraction_completion=extraction_completion or completion or shared_completion,
            )
            record_shared_phase("complete")
            return result
    if primary_vllm is not None and extraction_vllm is not None:
        all_gpu_interpretation = _all_gpu_interpretation_vllm_config(
            primary_vllm,
            extraction_vllm,
        )
        all_gpu_extraction = _all_gpu_extraction_vllm_config(
            primary_vllm,
            extraction_vllm,
        )
        phase_path = managed_root / "model_phase.json"
        switch_tracker = _ManagedStage2SwitchTracker(
            rapid_switch_seconds=float(config.vllm_rapid_switch_seconds)
        )

        def unavailable_model(
            required_role: str,
            switch_event: threading.Event,
        ) -> CompletionFunction:
            def request_switch(
                _messages: Sequence[Mapping[str, str]],
                _request_config: PlainHandoffStage2Config,
            ) -> str:
                switch_event.set()
                raise _ManagedStage2ModelSwitch(required_role)

            return request_switch

        def record_phase(
            *,
            status: str,
            active_role: str,
            transition: int,
            allocation_mode: str = "all_gpus",
            rapid_switch_trigger_elapsed_seconds: float | None = None,
        ) -> None:
            if active_role not in STAGE2_REQUEST_KINDS:
                raise ValueError(f"unknown managed Stage 2 role: {active_role!r}")
            if allocation_mode not in MANAGED_VLLM_ALLOCATION_MODES:
                raise ValueError(
                    f"unknown managed Stage 2 allocation mode: {allocation_mode!r}"
                )
            if allocation_mode == "all_gpus":
                role_config = (
                    all_gpu_interpretation
                    if active_role == "interpretation"
                    else all_gpu_extraction
                )
            else:
                role_config = (
                    primary_vllm
                    if active_role == "interpretation"
                    else extraction_vllm
                )
            role_model = config.model if active_role == "interpretation" else extraction.model
            _write_json(
                phase_path,
                {
                    "schema_version": MANAGED_MODEL_PHASE_SCHEMA_VERSION,
                    "status": status,
                    "active_role": active_role,
                    "allocation_mode": allocation_mode,
                    "model": role_model,
                    "gpus": list(role_config.gpus),
                    "configured_gpu_allocations": {
                        "interpretation": list(primary_vllm.gpus),
                        "extraction": list(extraction_vllm.gpus),
                    },
                    "transition": int(transition),
                    "rapid_switch_seconds": float(config.vllm_rapid_switch_seconds),
                    "last_switch_at": switch_tracker.last_switch_at,
                    "previous_switch_elapsed_seconds": (
                        switch_tracker.previous_switch_elapsed_seconds
                    ),
                    "rapid_switch_trigger_elapsed_seconds": (
                        rapid_switch_trigger_elapsed_seconds
                    ),
                    "recorded_at": _now(),
                },
            )

        def saved_phase() -> Mapping[str, Any] | None:
            if not phase_path.is_file():
                return None
            try:
                state = json.loads(phase_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return None
            if not isinstance(state, Mapping):
                return None
            if state.get("schema_version") != MANAGED_MODEL_PHASE_SCHEMA_VERSION:
                return None
            return state

        def phase_role(state: Mapping[str, Any] | None) -> str | None:
            if state is None:
                return None
            role = str(state.get("active_role") or "").strip().lower()
            return role if role in STAGE2_REQUEST_KINDS else None

        def configured_split_result(
            *,
            active_role: str,
            transition: int,
            trigger_elapsed_seconds: float | None,
            resumed: bool,
        ) -> Mapping[str, Any]:
            status = "running_configured_split"
            record_phase(
                status=status,
                active_role=active_role,
                transition=transition,
                allocation_mode="configured_split",
                rapid_switch_trigger_elapsed_seconds=trigger_elapsed_seconds,
            )
            LOGGER.info(
                "%s managed Stage 2 configured-split mode; keeping interpretation "
                "gpus=%s and extraction gpus=%s resident concurrently",
                "resume" if resumed else "enter",
                list(primary_vllm.gpus),
                list(extraction_vllm.gpus),
            )
            result = run_configured_managed_pools()
            record_phase(
                status="complete",
                active_role=active_role,
                transition=transition,
                allocation_mode="configured_split",
                rapid_switch_trigger_elapsed_seconds=trigger_elapsed_seconds,
            )
            return result

        def run_managed_role(
            active_role: str,
            *,
            runtime_dataset: pd.DataFrame | None,
            transition: int,
        ) -> Mapping[str, Any]:
            switch_event = threading.Event()
            record_phase(
                status="running",
                active_role=active_role,
                transition=transition,
            )
            if active_role == "interpretation":
                role_config = all_gpu_interpretation
                role_model = config.model
                role_api_key = config.api_key
                role_output_dir = managed_root / "orchestrator_all_gpus"
            else:
                role_config = all_gpu_extraction
                role_model = extraction.model
                role_api_key = extraction.api_key
                role_output_dir = managed_root / "extractor_all_gpus"
            LOGGER.info(
                "start managed Stage 2 all-GPU %s phase transition=%s gpus=%s "
                "replicas=%s tensor_parallel_size=%s",
                active_role,
                transition,
                list(role_config.gpus),
                role_config.server_count,
                role_config.effective_gpus_per_server(),
            )
            with launch_managed_vllm_servers(
                config=role_config,
                model=role_model,
                api_key=role_api_key,
                output_dir=role_output_dir,
            ) as active_endpoints:
                if active_role == "interpretation":
                    runtime_config = replace(
                        config,
                        endpoint=active_endpoints[0],
                        runtime_endpoints=tuple(active_endpoints),
                    )
                    routed_primary = completion or _RoundRobinOpenAICompletion(
                        tuple(active_endpoints)
                    )
                    primary_completion = _InterruptibleCompletion(
                        routed_primary,
                        required_role="extraction",
                        switch_event=switch_event,
                    )
                    runtime_extraction_completion = unavailable_model(
                        "extraction",
                        switch_event,
                    )
                else:
                    runtime_extraction = replace(
                        extraction,
                        endpoint=active_endpoints[0],
                        runtime_endpoints=tuple(active_endpoints),
                    )
                    runtime_config = replace(
                        config,
                        # The primary completion is a role-switch sentinel in
                        # this phase, but the top-level config still requires a
                        # syntactically valid transport URL.
                        endpoint=active_endpoints[0],
                        runtime_endpoints=tuple(active_endpoints),
                        extraction_llm=runtime_extraction,
                    )
                    primary_completion = unavailable_model(
                        "interpretation",
                        switch_event,
                    )
                    routed_extraction = (
                        extraction_completion
                        or _RoundRobinOpenAICompletion(tuple(active_endpoints))
                    )
                    runtime_extraction_completion = _InterruptibleCompletion(
                        routed_extraction,
                        required_role="interpretation",
                        switch_event=switch_event,
                    )
                return run_with_config(
                    runtime_config,
                    runtime_dataset=runtime_dataset,
                    runtime_primary_completion=primary_completion,
                    runtime_extraction_completion=runtime_extraction_completion,
                )

        saved_state = saved_phase()
        if saved_state is not None:
            saved_last_switch_at = str(
                saved_state.get("last_switch_at") or ""
            ).strip()
            if saved_last_switch_at:
                switch_tracker.last_switch_at = saved_last_switch_at
            raw_previous_elapsed = saved_state.get(
                "previous_switch_elapsed_seconds"
            )
            if isinstance(raw_previous_elapsed, (int, float)) and not isinstance(
                raw_previous_elapsed, bool
            ):
                switch_tracker.previous_switch_elapsed_seconds = max(
                    0.0, float(raw_previous_elapsed)
                )
        saved_allocation_mode = (
            str(saved_state.get("allocation_mode") or "all_gpus").strip().lower()
            if saved_state is not None
            else "all_gpus"
        )
        saved_transition = 0
        if saved_state is not None:
            try:
                saved_transition = max(0, int(saved_state.get("transition", 0)))
            except (TypeError, ValueError):
                saved_transition = 0
        saved_trigger_elapsed: float | None = None
        if saved_state is not None:
            raw_saved_elapsed = saved_state.get(
                "rapid_switch_trigger_elapsed_seconds"
            )
            if isinstance(raw_saved_elapsed, (int, float)) and not isinstance(
                raw_saved_elapsed, bool
            ):
                saved_trigger_elapsed = max(0.0, float(raw_saved_elapsed))

        # Once rapid alternation has selected the configured split, retain it
        # across process restarts. Setting the cutoff to zero explicitly opts
        # back into unconditional all-GPU alternation.
        if (
            dataset is not None
            and config.vllm_rapid_switch_seconds > 0
            and saved_allocation_mode == "configured_split"
        ):
            return configured_split_result(
                active_role=phase_role(saved_state) or "extraction",
                transition=saved_transition,
                trigger_elapsed_seconds=saved_trigger_elapsed,
                resumed=True,
            )

        # Feature discovery and operationalization are an interpretation-only
        # phase. Finish them first, then use ordinary request checkpoints to
        # alternate the two models while phases remain long enough to justify
        # reloading them over the complete GPU union.
        extraction_has_started = bool(
            dataset is not None and _has_extraction_dependent_checkpoints(output_dir)
        )
        if not extraction_has_started:
            interpretation_result = run_managed_role(
                "interpretation",
                runtime_dataset=None,
                transition=0,
            )
            if dataset is None:
                record_phase(
                    status="complete",
                    active_role="interpretation",
                    transition=0,
                )
                return interpretation_result
            active_role = "extraction"
            transition = 1
            switch_tracker.mark_switch()
            record_phase(
                status="switch_required",
                active_role=active_role,
                transition=transition,
            )
        else:
            active_role = phase_role(saved_state) or "extraction"
            transition = saved_transition
            # A resumed all-GPU role load is the start of a new residency
            # interval even though the previous process's monotonic clock is
            # intentionally not reused.
            switch_tracker.mark_switch()
            LOGGER.info(
                "resume alternating managed Stage 2 role=%s because "
                "extraction-dependent checkpoints already exist",
                active_role,
            )

        while True:
            try:
                result = run_managed_role(
                    active_role,
                    runtime_dataset=dataset,
                    transition=transition,
                )
            except _ManagedStage2ModelSwitch as switch:
                if switch.required_role == active_role:
                    raise RuntimeError(
                        "managed Stage 2 requested a switch to the model that is "
                        f"already active: {active_role}"
                    ) from switch
                elapsed = switch_tracker.mark_switch()
                transition += 1
                if switch_tracker.is_rapid(elapsed):
                    LOGGER.info(
                        "checkpointed managed Stage 2 %s phase after %.1f seconds "
                        "(< %.1f); keeping both models resident on their configured "
                        "GPU allocations",
                        active_role,
                        elapsed,
                        config.vllm_rapid_switch_seconds,
                    )
                    active_role = switch.required_role
                    record_phase(
                        status="rapid_switch_fallback",
                        active_role=active_role,
                        transition=transition,
                        allocation_mode="configured_split",
                        rapid_switch_trigger_elapsed_seconds=elapsed,
                    )
                    return configured_split_result(
                        active_role=active_role,
                        transition=transition,
                        trigger_elapsed_seconds=elapsed,
                        resumed=False,
                    )
                LOGGER.info(
                    "checkpointed managed Stage 2 %s phase after %s seconds; "
                    "switching all GPUs to %s",
                    active_role,
                    f"{elapsed:.1f}" if elapsed is not None else "an untracked interval",
                    switch.required_role,
                )
                active_role = switch.required_role
                record_phase(
                    status="switch_required",
                    active_role=active_role,
                    transition=transition,
                )
                continue
            record_phase(
                status="complete",
                active_role=active_role,
                transition=transition,
            )
            return result

    return run_configured_managed_pools()


__all__ = [
    "PlainHandoffStage2",
    "PlainHandoffStage2Config",
    "ManagedVLLMConfig",
    "ExtractionEndpoint",
    "Stage2ExtractionLLMConfig",
    "Stage2ExplicitFeature",
    "plain_stage2_config_from_mapping",
    "run_plain_handoff_stage2",
]
