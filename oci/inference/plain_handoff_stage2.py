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
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
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
from .plain_handoff_stage2_analysis import (
    Stage2RequestExhaustedError,
    Stage2ResponseValidationError,
    infrastructure_failure_audit_paths,
    prompt_token_count,
    run_fold_analysis,
)
from .stage2_agentic_selection import (
    DEFAULT_PAIRWISE_CHUNK_SIZE,
    Stage2AgenticSelectionConfig,
    agentic_selection_config_from_mapping,
)
from .stage2_elastic_net_selection import (
    TEMPORAL_SCOPE,
    Stage2ElasticNetSelectionConfig,
    statistical_selection_config_from_mapping,
)
from .stage2_role_adjudication import (
    Stage2RoleAdjudicationConfig,
    role_adjudication_config_from_mapping,
)
from .stage2_sampling import SAMPLING_FIELDS, recommended_sampling
from . import stage2_request_audit as request_audit
from .stage2_sequential_consolidation import (
    Stage2SequentialConsolidationConfig,
    sequential_consolidation_config_from_mapping,
)
from .vllm_server_pool import (
    ManagedVLLMConfig,
    launch_managed_vllm_servers,
    managed_vllm_config_from_mapping,
    validate_managed_vllm_pool_isolation,
)

LOGGER = logging.getLogger(__name__)

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
DEFAULT_REQUEST_TIMEOUT = 120 * 60.0
DEFAULT_REQUEST_ATTEMPT_TIMEOUT = 15 * 60.0
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
DEFAULT_INTERPRETATION_REASONING_EFFORT = "high"
DEFAULT_EXTRACTION_REASONING_EFFORT = "none"
STAGE2_REQUEST_KINDS = frozenset({"interpretation", "extraction"})
MANAGED_MODEL_PHASE_SCHEMA_VERSION = "stage2_managed_model_phase_v1"
DEFAULT_VLLM_RAPID_SWITCH_SECONDS = 15 * 60.0
MANAGED_VLLM_ALLOCATION_MODES = frozenset({"all_gpus", "configured_split"})
SUPPORTED_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
DEFAULT_EXTRACTION_MAX_PROMPT_CHARS = 640_000
DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE = 10
DEFAULT_EXTRACTION_CHUNK_SIZE_TOKENS = 50_000
DEFAULT_EXTRACTION_CONTEXT_WINDOW_TOKENS = 131_072
DEFAULT_EXTRACTION_CONTEXT_MARGIN_TOKENS = 1_024
DEFAULT_CONSOLIDATION_MAX_PROMPT_CHARS = 640_000
DEFAULT_OPERATIONALIZATION_MAX_PROMPT_CHARS = 640_000
DEFAULT_CONSOLIDATION_BATCH_SIZE = 20
DEFAULT_CONSOLIDATION_ALPHABETICAL_ROUNDS = 5
DEFAULT_CONSOLIDATION_SHUFFLE_ROUNDS = 50
DEFAULT_CONSOLIDATION_MAX_ROUNDS = (
    DEFAULT_CONSOLIDATION_ALPHABETICAL_ROUNDS + DEFAULT_CONSOLIDATION_SHUFFLE_ROUNDS
)
DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS = 3
DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS = 2
DEFAULT_EXTRACTION_VLLM_BASE_PORT = 8110
DEFAULT_EXTRACTION_VLLM_INTERNAL_PORT_BASE = 40_000
MODEL_IDENTITY_SCHEMA_VERSION = "stage2_endpoint_model_identity_v1"
FEATURE_DEFINITION_INPUT_SCHEMA_VERSION = (
    "stage2_feature_definition_inputs_v2_primary_model_only"
)
# Kept only so historical, non-exported candidate-funnel helpers remain
# importable while old checkpoints can be inspected. The Stage 2 execution
# path never calls them.
DEFAULT_CANDIDATE_SELECTION_HIERARCHY_TOP_COMMUNITIES = 3
CANDIDATE_SELECTION_SCHEMA_VERSION = "retired_colbert_candidate_selection"
_CANDIDATE_SELECTION_ENCODING_LOCK = threading.Lock()
CONSOLIDATION_SCHEMA_VERSION = "global_candidate_pool_v16_exhaustive_cards"
GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION = (
    "alphabetical_then_seeded_shuffle_candidate_batches_v10_no_prefilter"
)
EXTRACTION_ONTOLOGY_FEEDBACK_SCHEMA_VERSION = (
    "training_failure_ontology_refinement_v1_explicit_feature_invariants"
)
OPERATIONALIZATION_SCHEMA_VERSION = (
    "feature_name_bounded_supporting_text_v7_conflict_resolution"
)
INTERPRETATION_SCHEMA_VERSION = "semantic_cards_exhaustive_feature_discovery_v12"
INTERPRETATION_AUDIT_SCHEMA_VERSION = (
    "rejected_packet_text_only_ordinals_v7_exhaustive_decomposition"
)
CONFIGURED_EXPLICIT_FEATURE_ARCHITECTURE = "configured_explicit_feature"

_CONCEPT_IDENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "assessment",
    "at",
    "baseline",
    "be",
    "biomarker",
    "by",
    "clinical",
    "confounder",
    "diagnosis",
    "disease",
    "documentation",
    "effect",
    "evidence",
    "expression",
    "feature",
    "for",
    "from",
    "has",
    "have",
    "history",
    "in",
    "information",
    "is",
    "level",
    "measurement",
    "metastasis",
    "modifier",
    "mutation",
    "outcome",
    "of",
    "on",
    "or",
    "patient",
    "presence",
    "pretreatment",
    "prognostic",
    "record",
    "residual",
    "score",
    "specific",
    "status",
    "that",
    "the",
    "this",
    "to",
    "treatment",
    "use",
    "value",
    "variable",
    "was",
    "were",
    "with",
    "without",
}


def _concept_identity_tokens(*values: Any) -> set[str]:
    """Return conservative lexical anchors for one patient-level measurement."""

    text = " ".join(str(value) for value in values if value is not None).lower()
    tokens: set[str] = set()
    for raw_token in re.findall(r"[a-z]+[a-z0-9]*", text):
        if raw_token in _CONCEPT_IDENTITY_STOPWORDS:
            continue
        token = raw_token
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        if len(token) > 1 and token not in _CONCEPT_IDENTITY_STOPWORDS:
            tokens.add(token)

    # Add compact variants for separator-delimited identifier fragments without
    # knowing anything about the clinical domain. This makes forms such as
    # ``ab_c1`` and ``ab-c1`` comparable while remaining conservative for words.
    for compound in re.findall(r"[a-z0-9]+(?:[_-][a-z0-9]+)+", text):
        parts = re.split(r"[_-]", compound)
        for start in range(len(parts) - 1):
            for stop in range(start + 2, len(parts) + 1):
                segment = parts[start:stop]
                has_digit = any(any(char.isdigit() for char in part) for part in segment)
                all_short = all(len(part) <= 3 for part in segment)
                if not (has_digit or all_short):
                    continue
                compact = "".join(segment)
                if len(compact) > 2:
                    tokens.add(compact)
    return tokens


def _consolidation_route_is_semantically_compatible(
    candidate: Mapping[str, Any],
    feature: Mapping[str, Any],
) -> bool:
    """Require a shared measurement anchor before two concepts may be routed together."""

    candidate_tokens = _concept_identity_tokens(
        candidate.get("name"),
        candidate.get("description"),
    )
    feature_tokens = _concept_identity_tokens(
        feature.get("name"),
        feature.get("description"),
        feature.get("measurement_definition"),
    )
    # Old/custom candidate producers may not supply concept text. In that case
    # packet grounding remains the only available check.
    if not candidate_tokens or not feature_tokens:
        return True
    return bool(candidate_tokens.intersection(feature_tokens))


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


def _evaluate_stage2_oracle_ite(
    *,
    prediction_path: Path,
    dataset: pd.DataFrame,
    output_dir: Path,
    oracle_ite_column: str = "true_ite_prob",
) -> dict[str, Any]:
    """Evaluate frozen cross-fitted ITEs without exposing truth to modeling."""

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
    if oracle_ite_column not in dataset.columns:
        payload = {
            **base,
            "reason": f"dataset does not contain {oracle_ite_column!r}",
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
        elif self.vllm is None:
            raise ValueError(
                "stage2.extraction_llm requires either endpoint or vllm"
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
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key": "<redacted>",
            "workers": self.workers,
            "vllm": self.vllm.public_dict() if self.vllm is not None else None,
            "runtime_endpoint": self.runtime_endpoint,
            "runtime_model": self.runtime_model,
            "runtime_api_key": "<redacted>",
        }


@dataclass(frozen=True)
class PlainHandoffStage2Config:
    endpoint: str
    model: str = ""
    api_key: str = "EMPTY"
    # Total wall-clock budget for one logical JSON request, including transport
    # retries and validator-guided repair turns.
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    # Each individual HTTP attempt gets a smaller timeout so a straggling
    # endpoint can be abandoned and retried within the logical request budget.
    request_attempt_timeout: float = DEFAULT_REQUEST_ATTEMPT_TIMEOUT
    transport_max_attempts: int = DEFAULT_TRANSPORT_MAX_ATTEMPTS
    transport_retry_backoff: float = 2.0
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
    consolidation_alphabetical_rounds: int = DEFAULT_CONSOLIDATION_ALPHABETICAL_ROUNDS
    consolidation_max_rounds: int = DEFAULT_CONSOLIDATION_MAX_ROUNDS
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
    evidence_compiler: str = EVIDENCE_COMPILER_VERSION
    required_architectures: tuple[str, ...] = SUPPORTED_STAGE2_ARCHITECTURES
    included_architectures: tuple[str, ...] | None = None
    evidence_max_cards_per_fold: int = 400
    evidence_max_exemplars_per_card: int = 4
    evidence_max_exemplar_chars: int = 2_400
    workers: int = 4
    extraction_llm: Stage2ExtractionLLMConfig | None = None
    # Limits aggregate ontology-supervisor reviews. These reviews cannot add,
    # drop, rename, or assign roles to candidate features.
    max_review_rounds: int = 2
    ontology_refinement_min_failure_patients: int = DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS
    max_ontology_refinement_rounds: int = DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS
    # This is a hard upstream invariant. Historical treatments remain valid;
    # Stage 2 never guesses timepoints from feature semantics.
    input_temporal_scope: str = TEMPORAL_SCOPE
    agentic_selection: Stage2AgenticSelectionConfig = field(
        default_factory=Stage2AgenticSelectionConfig
    )
    # Operational tuning only: deterministic pair chunks are checkpointed and
    # produce identical evidence regardless of this size.
    agentic_evidence_pair_chunk_size: int = DEFAULT_PAIRWISE_CHUNK_SIZE
    # Optional outer-training-only semantic/association consolidation runs
    # immediately before supervised role selection.  The older global agentic
    # policy remains parseable only for historical checkpoint inspection.
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
    # Absolute monotonic deadline and remaining HTTP-call allowance for one
    # transport-retry turn. These are populated only by the request runner so
    # compatibility negotiation cannot create unbounded hidden attempts.
    runtime_request_deadline: float | None = None
    runtime_transport_attempt_budget: int | None = None
    # Detected from the live endpoint's model record (including its backing
    # root when available) so served aliases still receive the right controls.
    runtime_model_family: str = ""
    runtime_sampling_model: str = ""
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
        if self.consolidation_alphabetical_rounds < 0:
            raise ValueError("stage2.consolidation_alphabetical_rounds must be nonnegative")
        if self.consolidation_max_rounds < 1:
            raise ValueError("stage2.consolidation_max_rounds must be positive")
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
        if self.evidence_compiler != EVIDENCE_COMPILER_VERSION:
            raise ValueError(
                f"stage2.evidence_compiler must be {EVIDENCE_COMPILER_VERSION}; "
                "raw_packets_v1 was retired because it merged distinct scientific "
                "architectures"
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
        if self.input_temporal_scope != TEMPORAL_SCOPE:
            raise ValueError(
                f"stage2.input_temporal_scope must be {TEMPORAL_SCOPE!r}; Stage 2 "
                "does not perform semantic temporal filtering"
            )
        if not isinstance(self.agentic_selection, Stage2AgenticSelectionConfig):
            raise ValueError(
                "stage2.agentic_selection must be a Stage2AgenticSelectionConfig object"
            )
        self.agentic_selection.validate()
        if (
            isinstance(self.agentic_evidence_pair_chunk_size, bool)
            or not isinstance(self.agentic_evidence_pair_chunk_size, int)
            or self.agentic_evidence_pair_chunk_size < 1
        ):
            raise ValueError(
                "stage2.agentic_evidence_pair_chunk_size must be a positive integer"
            )
        if not isinstance(
            self.selection_consolidation, Stage2SequentialConsolidationConfig
        ):
            raise ValueError(
                "stage2.selection_consolidation must be a "
                "Stage2SequentialConsolidationConfig object"
            )
        self.selection_consolidation.validate()
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
        if (
            self.statistical_selection.selection_mode == "multi_model"
            and not self.role_adjudication.enabled
        ):
            raise ValueError("multi_model selection requires role_adjudication.enabled=true")
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
        values.pop("runtime_disable_extraction", None)
        values["explicit_features"] = [
            feature.as_definition() for feature in self.explicit_features
        ]
        values["agentic_selection"] = self.agentic_selection.public_dict()
        values["selection_consolidation"] = (
            self.selection_consolidation.public_dict()
        )
        values["statistical_selection"] = self.statistical_selection.public_dict()
        values["role_adjudication"] = self.role_adjudication.public_dict()
        return values


def plain_stage2_config_from_mapping(
    raw: Mapping[str, Any],
    *,
    default_workers: int,
) -> PlainHandoffStage2Config | None:
    legacy_screen_keys = sorted(
        set(raw).intersection(
            {
                "selection_workers",
                "confounder_p_value_threshold",
                "confounder_min_inner_fold_fraction",
                "effect_modifier_p_value_threshold",
                "effect_modifier_min_inner_fold_fraction",
            }
        )
    )
    if legacy_screen_keys:
        LOGGER.warning(
            "migrating legacy Stage 2 screen settings to the group-elastic-net selector "
            "where possible; raw p-value thresholds and selection_workers are ignored: %s",
            ", ".join(f"stage2.{key}" for key in legacy_screen_keys),
        )
    if raw.get("command"):
        raise ValueError(
            "stage2.command is not used by the plain workflow; configure stage2.endpoint"
        )
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
    legacy_extraction_batch_size = raw.get("extraction_batch_size")
    if legacy_extraction_batch_size is not None and int(legacy_extraction_batch_size) != 1:
        LOGGER.warning(
            "stage2.extraction_batch_size=%s is ignored; Stage 2 extraction is permanently "
            "isolated to one patient per prompt",
            legacy_extraction_batch_size,
        )

    def architecture_names(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            if value.strip().lower() == "all":
                return SUPPORTED_STAGE2_ARCHITECTURES
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return tuple(value)

    if "interpretation_reasoning_effort" in raw:
        interpretation_reasoning_effort = (
            str(raw["interpretation_reasoning_effort"]).strip().lower()
        )
    elif "enable_thinking" in raw:
        legacy_enable_thinking = raw["enable_thinking"]
        if not isinstance(legacy_enable_thinking, bool):
            raise ValueError("stage2.enable_thinking must be true or false")
        interpretation_reasoning_effort = "high" if legacy_enable_thinking else "none"
        LOGGER.warning(
            "stage2.enable_thinking is deprecated; use "
            "stage2.interpretation_reasoning_effort. Extraction reasoning is "
            "controlled independently."
        )
    else:
        interpretation_reasoning_effort = DEFAULT_INTERPRETATION_REASONING_EFFORT
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

    retired_keys = sorted(
        key
        for key in raw
        if key == "candidate_discovery_source"
        or key == "max_candidates_per_fold"
        or key == "consolidation_oversample_factor"
        or key == "screening_trees"
        or key == "max_evaluation_rounds"
        or key.startswith("evidence_community_")
        or key.startswith("candidate_registry_")
        or key.startswith("candidate_selection_")
        or key.startswith("stability_selection_")
        or key.startswith("effect_modifier_negative_")
    )
    if retired_keys:
        LOGGER.warning(
            "ignoring retired Stage 2 ColBERT/forest-screening settings: %s",
            ", ".join(f"stage2.{key}" for key in retired_keys),
        )

    raw_extraction_llm = raw.get("extraction_llm")
    if raw_extraction_llm is None:
        extraction_llm = None
    elif not isinstance(raw_extraction_llm, Mapping):
        raise ValueError("stage2.extraction_llm must be an object")
    else:
        extraction_endpoint = str(raw_extraction_llm.get("endpoint") or "").strip().rstrip("/")
        extraction_model = str(raw_extraction_llm.get("model") or "").strip()
        extraction_vllm = managed_vllm_config_from_mapping(
            raw_extraction_llm.get("vllm"),
            model=extraction_model,
            default_base_port=DEFAULT_EXTRACTION_VLLM_BASE_PORT,
            default_internal_port_base=DEFAULT_EXTRACTION_VLLM_INTERNAL_PORT_BASE,
        )
        if extraction_endpoint and extraction_vllm is not None:
            raise ValueError(
                "configure either stage2.extraction_llm.endpoint or "
                "stage2.extraction_llm.vllm, not both"
            )
        if not extraction_endpoint and extraction_vllm is None:
            raise ValueError(
                "stage2.extraction_llm requires either endpoint or vllm"
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
            workers=max(
                1,
                int(
                    raw_extraction_llm.get(
                        "workers",
                        min(4, max(1, default_workers)),
                    )
                ),
            ),
            vllm=extraction_vllm,
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
    raw_pair_chunk_size = raw.get(
        "agentic_evidence_pair_chunk_size",
        DEFAULT_PAIRWISE_CHUNK_SIZE,
    )
    if isinstance(raw_pair_chunk_size, bool):
        raise ValueError(
            "stage2.agentic_evidence_pair_chunk_size must be a positive integer"
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
        api_key=api_key,
        request_timeout=float(raw.get("request_timeout", DEFAULT_REQUEST_TIMEOUT)),
        request_attempt_timeout=float(
            raw.get("request_attempt_timeout", DEFAULT_REQUEST_ATTEMPT_TIMEOUT)
        ),
        transport_max_attempts=int(
            raw.get("transport_max_attempts", DEFAULT_TRANSPORT_MAX_ATTEMPTS)
        ),
        transport_retry_backoff=float(raw.get("transport_retry_backoff", 2.0)),
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
        consolidation_alphabetical_rounds=int(
            raw.get(
                "consolidation_alphabetical_rounds",
                DEFAULT_CONSOLIDATION_ALPHABETICAL_ROUNDS,
            )
        ),
        consolidation_max_rounds=int(
            raw.get("consolidation_max_rounds", DEFAULT_CONSOLIDATION_MAX_ROUNDS)
        ),
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
        evidence_compiler=str(raw.get("evidence_compiler", EVIDENCE_COMPILER_VERSION)).strip(),
        required_architectures=architecture_names(
            raw.get("required_architectures", SUPPORTED_STAGE2_ARCHITECTURES)
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
        input_temporal_scope=str(raw.get("input_temporal_scope", TEMPORAL_SCOPE)).strip(),
        agentic_selection=agentic_selection_config_from_mapping(raw.get("agentic_selection")),
        agentic_evidence_pair_chunk_size=int(raw_pair_chunk_size),
        selection_consolidation=sequential_consolidation_config_from_mapping(
            raw.get("selection_consolidation")
        ),
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

        records_by_id[model_id] = {
            "id": model_id,
            "root": stable_field("root"),
            "parent": stable_field("parent"),
            "revision": stable_field("revision") or stable_field("model_revision"),
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
        endpoint=extraction.endpoint,
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
        extraction.endpoint,
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


_DROP_KEYS = {
    "artifacts",
    "artifact_inventory",
    "common_vocabulary",
    "config",
    "fit_row_ids",
    "heldout_row_ids",
    "metrics",
    "model_diagnostics",
    "predictions",
    "run_config",
    "schema_version",
    "train_activations",
}


def _is_operational_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in _DROP_KEYS
        or lowered.endswith(("_sha256", "_hash", "_fingerprint", "_path"))
        or lowered.startswith(("authenticated_", "attestation_", "immutable_"))
    )


def _scientific_projection(value: Any) -> Any:
    """Remove old control-plane fields while retaining readable evidence."""

    if isinstance(value, Mapping):
        return {
            str(key): _scientific_projection(child)
            for key, child in value.items()
            if not _is_operational_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_scientific_projection(child) for child in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _infer_evidence_axes(value: Any) -> list[str]:
    """Infer statistical axes from ordinary Stage 1 field names and labels."""

    tokens: list[str] = []
    axis_value_keys = {
        "architecture",
        "axis",
        "axes",
        "bank",
        "contrast_family",
        "evidence_type",
        "json_path",
        "meaning",
        "mechanical_role",
        "objective",
        "observable_axes",
        "role",
        "signal",
        "target",
        "target_source",
    }

    def visit(child: Any) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                lowered = str(key).lower()
                tokens.append(lowered)
                if lowered in axis_value_keys and isinstance(nested, (str, list, tuple)):
                    if isinstance(nested, str):
                        tokens.append(nested.lower())
                    else:
                        tokens.extend(str(item).lower() for item in nested)
                visit(nested)
        elif isinstance(child, (list, tuple)):
            for nested in child:
                visit(nested)

    visit(value)
    joined = " ".join(tokens)
    axes: set[str] = {"semantic"}
    if "treatment" in joined or "propensity" in joined:
        axes.add("treatment")
    if "outcome" in joined or "prognostic" in joined:
        axes.add("outcome")
    if any(
        token in joined
        for token in (
            "residual_effect",
            "residual effect",
            "pseudo_target",
            "r_loss",
            "r-loss",
            "heterogeneity",
            "interaction",
            "uplift",
            "effect_modifier",
            "effect modifier",
        )
    ) or any(token in {"effect", "effect_bank", "r"} for token in tokens):
        axes.add("residual_effect")
    if "matched_pair" in joined or "matched pair" in joined:
        axes.add("matched_pair")
    return sorted(axes)


def _row_sections(row: Mapping[str, Any]) -> list[tuple[str, Any, str]]:
    """Expose natural scientific sections without requiring one input schema."""

    source = str(row.get("source") or "unknown")
    payload = row.get("evidence")
    if not isinstance(payload, Mapping):
        return [(source, payload, "evidence")]
    architecture = payload.get("architecture")
    if isinstance(architecture, str) and architecture.strip():
        return [(architecture.strip(), payload, "evidence")]

    sections: list[tuple[str, Any, str]] = []
    if source == "text_models":
        importance = payload.get("importance")
        if importance:
            sections.append(("sparse_and_matched_pair_models", importance, "importance"))
        embedding = payload.get("embedding_contrast_evidence")
        if isinstance(embedding, Mapping) and embedding.get("contrasts"):
            for index, contrast in enumerate(embedding["contrasts"], start=1):
                sections.append(
                    (
                        "embedding_contrasts_and_retrieval_terms",
                        contrast,
                        f"embedding_contrast_evidence.contrasts[{index - 1}]",
                    )
                )
        elif embedding:
            sections.append(
                (
                    "embedding_contrasts_and_retrieval_terms",
                    embedding,
                    "embedding_contrast_evidence",
                )
            )
        htr = payload.get("htr_evidence")
        if isinstance(htr, Mapping):
            for key, value in htr.items():
                if value:
                    sections.append(("hierarchical_neural_text", value, f"htr_evidence.{key}"))
        elif htr:
            sections.append(("hierarchical_neural_text", htr, "htr_evidence"))
    elif source == "tfidf":
        discovery = payload.get("discovery")
        if isinstance(discovery, Mapping):
            topic_banks = discovery.get("topic_banks")
            if isinstance(topic_banks, Mapping):
                for bank, value in topic_banks.items():
                    if value:
                        sections.append(("tfidf_topics", value, f"discovery.topic_banks.{bank}"))
            score_tests = discovery.get("topic_score_tests")
            if isinstance(score_tests, Mapping) and score_tests.get("effect_orphan_ngram_branch"):
                sections.append(
                    (
                        "tfidf_orphan_ngrams",
                        score_tests["effect_orphan_ngram_branch"],
                        "discovery.topic_score_tests.effect_orphan_ngram_branch",
                    )
                )
            elif score_tests:
                sections.append(("tfidf_orphan_ngrams", score_tests, "discovery.topic_score_tests"))
    elif source == "neural_queries":
        query_evidence = payload.get("evidence")
        if isinstance(query_evidence, list):
            by_bank: dict[str, list[Any]] = defaultdict(list)
            for row_value in query_evidence:
                bank = (
                    str(row_value.get("bank") or "unspecified")
                    if isinstance(row_value, Mapping)
                    else "unspecified"
                )
                by_bank[bank].append(row_value)
            for bank, values in by_bank.items():
                sections.append(("neural_query_moments", values, f"evidence.{bank}"))
        elif query_evidence:
            sections.append(("neural_query_moments", query_evidence, "evidence"))
    return sections or [(source, payload, "evidence")]


def _json_chars(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), sort_keys=True))


def _split_value(
    value: Any,
    *,
    max_chars: int,
    path: str,
    path_budget: Callable[[str], int] | None = None,
) -> list[tuple[str, Any]]:
    def fragment_budget(fragment_path: str) -> int:
        if path_budget is None:
            return int(max_chars)
        return min(int(max_chars), int(path_budget(fragment_path)))

    def fragment_fits(fragment_path: str, fragment: Any) -> bool:
        return _json_chars(fragment) <= fragment_budget(fragment_path)

    if fragment_fits(path, value):
        return [(path, value)]
    if isinstance(value, Mapping):
        fragments: list[tuple[str, Any]] = []
        scalars: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if isinstance(child, (Mapping, list, tuple)):
                fragments.extend(
                    _split_value(
                        child,
                        max_chars=max_chars,
                        path=child_path,
                        path_budget=path_budget,
                    )
                )
            else:
                scalars[str(key)] = child
        if scalars:
            if fragment_fits(path, scalars):
                fragments.insert(0, (path, scalars))
            else:
                for key, child in scalars.items():
                    fragments.extend(
                        _split_value(
                            child,
                            max_chars=max_chars,
                            path=f"{path}.{key}",
                            path_budget=path_budget,
                        )
                    )
        return fragments
    if isinstance(value, (list, tuple)):
        output: list[tuple[str, Any]] = []
        batch: list[Any] = []
        batch_chars = 2  # Opening and closing brackets in compact JSON.
        batch_start = 0
        for index, child in enumerate(value):
            child_chars = _json_chars(child)
            if batch:
                candidate_path = f"{path}[{batch_start}:{index + 1}]"
                candidate_chars = batch_chars + 1 + child_chars
                if candidate_chars <= fragment_budget(candidate_path):
                    batch.append(child)
                    batch_chars = candidate_chars
                    continue
                output.append((f"{path}[{batch_start}:{index}]", batch))
                batch = []
                batch_chars = 2
            singleton_path = f"{path}[{index}:{index + 1}]"
            singleton_chars = 2 + child_chars
            if singleton_chars <= fragment_budget(singleton_path):
                batch = [child]
                batch_chars = singleton_chars
                batch_start = index
            else:
                output.extend(
                    _split_value(
                        child,
                        max_chars=max_chars,
                        path=f"{path}[{index}]",
                        path_budget=path_budget,
                    )
                )
                batch_start = index + 1
        if batch:
            output.append((f"{path}[{batch_start}:{len(value)}]", batch))
        return output
    text = str(value)
    segments: list[tuple[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        segment_path = f"{path}.text_segment_{len(segments) + 1:03d}"
        low, high = cursor + 1, len(text)
        best = cursor
        while low <= high:
            end = (low + high) // 2
            if fragment_fits(segment_path, text[cursor:end]):
                best = end
                low = end + 1
            else:
                high = end - 1
        if best == cursor:
            raise ValueError(
                f"max_packet_chars cannot encode one source character at {segment_path}"
            )
        segments.append((segment_path, text[cursor:best]))
        cursor = best
    if "".join(str(segment) for _path, segment in segments) != text:
        raise RuntimeError("Stage 2 scalar packetization changed source text")
    return segments


def packetize_handoff(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_packet_chars: int,
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        try:
            outer_fold = int(row["outer_fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"handoff row {row_index} has no integer outer_fold") from exc
        for section_index, (architecture, raw_section, section_path) in enumerate(
            _row_sections(row), start=1
        ):
            section = _scientific_projection(raw_section)
            prototype = {
                "packet_id": (
                    f"outer_{outer_fold:03d}_row_{row_index:04d}_"
                    f"section_{section_index:02d}_part_000000000000"
                ),
                "source": str(row.get("source") or "unknown"),
                "architecture": str(architecture),
                "outer_fold": outer_fold,
                "inner_fold": row.get("inner_fold"),
                "scope": str(row.get("scope") or "unspecified"),
                "json_path": section_path,
                "observable_axes": [
                    "matched_pair",
                    "outcome",
                    "residual_effect",
                    "semantic",
                    "treatment",
                    "unclear",
                ],
                "content": "",
            }

            def packet_content_budget(json_path: str) -> int:
                envelope = {**prototype, "json_path": json_path}
                envelope_chars = _json_chars(envelope) - _json_chars("")
                return int(max_packet_chars) - envelope_chars

            if packet_content_budget(section_path) < 1:
                raise ValueError("max_packet_chars is too small for the Stage 2 packet envelope")
            fragments = _split_value(
                section,
                max_chars=int(max_packet_chars),
                path=section_path,
                path_budget=packet_content_budget,
            )
            for fragment_index, (json_path, content) in enumerate(fragments, start=1):
                packet = {
                    "packet_id": (
                        f"outer_{outer_fold:03d}_row_{row_index:04d}_"
                        f"section_{section_index:02d}_part_{fragment_index:03d}"
                    ),
                    "source": str(row.get("source") or "unknown"),
                    "architecture": str(architecture),
                    "outer_fold": outer_fold,
                    "inner_fold": row.get("inner_fold"),
                    "scope": str(row.get("scope") or "unspecified"),
                    "json_path": json_path,
                    "observable_axes": _infer_evidence_axes(
                        {
                            "architecture": architecture,
                            "json_path": json_path,
                            "content": content,
                        }
                    ),
                    "content": content,
                }
                if _json_chars(packet) > int(max_packet_chars):
                    raise RuntimeError("Stage 2 packet planner emitted an oversized packet")
                packets.append(packet)
    if not packets:
        raise ValueError("the Stage 1 handoff contains no evidence packets")
    return packets


def _partition_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    max_chars: int,
) -> list[list[Mapping[str, Any]]]:
    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for packet in packets:
        candidate = [*current, packet]
        if current and _json_chars(candidate) > max_chars:
            batches.append(current)
            current = []
        current.append(packet)
    if current:
        batches.append(current)
    return batches


CompletionFunction = Callable[[Sequence[Mapping[str, str]], PlainHandoffStage2Config], str]


class _RetryableStage2ResponseError(RuntimeError):
    """A completed transport that did not yield any response content."""


class _Stage2OutputLengthError(ValueError):
    """The server exhausted the available completion length."""


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
    """Load the exact extraction-model tokenizer only when patient work begins."""

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
                        "Stage 2 cannot enforce serial extraction token limits because "
                        f"the extraction tokenizer {self.model!r} could not be loaded. "
                        "Make the tokenizer available in the extraction vLLM download "
                        "directory or the Hugging Face cache."
                    ) from exc
        return self._tokenizer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._load()(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        template_kwargs = {**self.chat_template_kwargs, **kwargs}
        return self._load().apply_chat_template(*args, **template_kwargs)


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
    reasoning_effort = config.runtime_reasoning_effort or configured_reasoning_effort
    extraction_ceiling = config.extraction_max_tokens
    if _reasoning_enabled(reasoning_effort) and config.extraction_reasoning_max_tokens is not None:
        extraction_ceiling = config.extraction_reasoning_max_tokens
    return {
        "request_kind": kind,
        "reasoning_effort": reasoning_effort,
        "max_tokens": int(
            extraction_ceiling if kind == "extraction" else config.max_tokens
        ),
        **recommended_sampling(
            config.runtime_sampling_model or config.model,
            config.runtime_model_family or _stage2_model_family(config.model),
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

    return {
        "feature_definition_input_schema": FEATURE_DEFINITION_INPUT_SCHEMA_VERSION,
        "outer_fold": int(outer_fold),
        "compiler": config.evidence_compiler,
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
        "global_candidate_pool_schema": GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION,
        "consolidation_batch_size": int(config.consolidation_batch_size),
        "consolidation_alphabetical_rounds": int(config.consolidation_alphabetical_rounds),
        "consolidation_max_rounds": int(config.consolidation_max_rounds),
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
                    "chat_template_kwargs": {"enable_thinking": enabled},
                },
                {
                    **sampling_extra,
                    "enable_thinking": enabled,
                },
            ]
        )
    family_bodies.extend([sampling_extra, {}])

    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    reasoning_variants = (True, False) if wire_reasoning_effort is not None else (False,)
    for extra_body in family_bodies:
        for include_reasoning in reasoning_variants:
            candidate = {
                **dict(base_kwargs),
                "response_format": {"type": "json_object"},
            }
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
    """Read final content whether reasoning was parsed separately or left inline."""

    content = _message_text(
        message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    )
    if content.strip():
        return content
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
    controlled_messages = _reasoning_controlled_messages(
        messages,
        model_family=model_family,
        enable_thinking=_reasoning_enabled(str(request_policy["reasoning_effort"])),
        max_prompt_chars=int(config.max_prompt_chars),
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
            read_timeout_seconds=min(float(config.request_timeout), remaining),
        )
        try:
            response = client.chat.completions.create(**kwargs)
            if streaming:
                response = request_audit.collect_stream(
                    response, deadline=logical_deadline,
                    deadline_error=Stage2RequestExhaustedError,
                    incomplete_error=_RetryableStage2ResponseError,
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


class _ConcurrencyLimitedCompletion:
    """Apply one global request ceiling across nested Stage 2 executors."""

    def __init__(self, completion: CompletionFunction, max_concurrency: int) -> None:
        self.completion = completion
        self._semaphore = threading.BoundedSemaphore(max(1, int(max_concurrency)))

    @contextmanager
    def request_slot(self) -> Iterator[CompletionFunction]:
        queued_at = time.monotonic()
        with self._semaphore:
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
        with self._semaphore:
            return self.completion(messages, config)


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
            _RoundRobinOpenAICompletion,
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
        or isinstance(completion, _RoundRobinOpenAICompletion)
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
            retry_messages = _transport_retry_messages(
                messages,
                cause,
                max_prompt_chars=int(config.max_prompt_chars),
                prompt_token_counter=prompt_token_counter,
                context_window_tokens=context_window_tokens,
                context_margin_tokens=context_margin_tokens,
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


def _transport_retry_messages(
    messages: Sequence[Mapping[str, str]],
    exc: Exception,
    *,
    max_prompt_chars: int,
    prompt_token_counter: Callable[[Sequence[Mapping[str, str]]], int] | None,
    context_window_tokens: int | None,
    context_margin_tokens: int,
) -> list[dict[str, Any]]:
    """Carry the latest call failure forward without accumulating retry turns."""

    detail = f"{type(exc).__name__}: {exc}"
    directive = (
        f"The previous request failed: {detail}. "
        "Retry the original task and return one complete JSON object using the required "
        "schema. Keep the response concise. This is a request failure, not evidence "
        "about the patient or the scientific result."
    )

    def fits(candidate: Sequence[Mapping[str, str]]) -> bool:
        return (
            sum(len(str(row.get("content") or "")) for row in candidate) <= max_prompt_chars
            and (
                prompt_token_counter is None
                or context_window_tokens is None
                or int(prompt_token_counter(candidate)) + context_margin_tokens
                < context_window_tokens
            )
        )

    # Preserve any semantic repair feedback already in the input. Compact JSON
    # losslessly when needed; never truncate patient records to make room.
    for base in ([dict(row) for row in messages], _compact_json_messages(messages)):
        for content in (directive, detail):
            candidate = [*base, {"role": "user", "content": content}]
            if fits(candidate):
                return candidate
    raise ValueError(
        "Stage 2 transport retry prompt cannot fit the previous error within its "
        "prompt budget"
    ) from exc


def _repair_message(
    exc: Exception,
    *,
    repair_context: Mapping[str, Any] | None = None,
    repeated_error_count: int = 1,
) -> dict[str, str]:
    if isinstance(exc, _Stage2OutputLengthError):
        content = (
            "The previous JSON exceeded the available response length. "
            f"{type(exc).__name__}: {exc}. Return one materially "
            "shorter corrected JSON object using the same required schema. Remove redundancy, "
            "merge duplicate entries, and keep descriptions and rationales concise. Do not omit "
            "required records or fields. Return JSON only."
        )
    else:
        content = (
            "The previous JSON failed validation. Correct this exact error: "
            f"{type(exc).__name__}: {exc}. Return one corrected JSON object only."
        )
    context = dict(repair_context or {})
    allowed_feature_ids = context.get("allowed_feature_ids")
    if isinstance(allowed_feature_ids, list) and allowed_feature_ids:
        content += (
            " The only feature_ids allowed in this response are: "
            + json.dumps(
                list(map(str, allowed_feature_ids)),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "."
        )
    expression_examples = context.get("valid_expression_examples")
    if isinstance(expression_examples, Mapping) and expression_examples:
        content += (
            " Valid categorical-rule expression examples are: "
            + json.dumps(
                dict(expression_examples),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "."
        )
    conservative_response = context.get("conservative_response")
    if repeated_error_count >= 2 and isinstance(conservative_response, Mapping):
        content += (
            f" This exact validation error has now occurred {repeated_error_count} times. "
            "Abandon the latent proposal and return exactly this conservative response: "
            + json.dumps(
                dict(conservative_response),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "."
        )
    return {
        "role": "user",
        "content": content,
    }


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
                prompt_token_counter=prompt_token_counter,
                context_window_tokens=context_window_tokens,
                context_margin_tokens=context_margin_tokens,
                repair_context=repair_context,
                validation_event_observer=validation_event_observer,
                conservative_validation_fallback=conservative_validation_fallback,
                fallback_after_same_error=fallback_after_same_error,
            )
    request_policy = _stage2_request_policy(config, request_kind)
    request_config = replace(
        config,
        runtime_request_kind=str(request_policy["request_kind"]),
        runtime_reasoning_effort=None,
    )
    base_conversation = [dict(message) for message in messages]
    conversation = [dict(message) for message in base_conversation]
    first_error: Exception | None = None
    max_repairs = int(config.max_response_repairs)
    max_attempts = 1 + max_repairs
    logical_deadline = time.monotonic() + float(config.request_timeout)
    request_audit.event("request_admitted", logical_budget_seconds=config.request_timeout)
    validation_error_counts: Counter[str] = Counter()
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
                        "Stage 2 extraction repair prompt leaves no model output context: "
                        f"prompt_tokens={prompt_tokens}, "
                        f"context_window_tokens={context_window_tokens}, "
                        f"context_margin_tokens={context_margin_tokens}"
                    )
                dynamic_output_ceiling = min(
                    int(_stage2_request_policy(attempt_config)["max_tokens"]),
                    available_output_tokens,
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
                else:  # pragma: no cover - token counting is extraction-only today
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
            if response is None and not isinstance(exc, _Stage2OutputLengthError):
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
        "llm_identity": {
            "model": config.model,
        },
        "request_policy": _stage2_request_policy(config, "interpretation"),
    }
    input_fingerprint = _value_fingerprint(checkpoint_input)
    input_path = output_dir / "input.json"
    result_path = output_dir / "result.json"
    complete_path = output_dir / "complete.json"
    if input_path.is_file() and result_path.is_file() and complete_path.is_file():
        try:
            previous_input = json.loads(input_path.read_text(encoding="utf-8"))
            completion_state = json.loads(complete_path.read_text(encoding="utf-8"))
            cached_result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                previous_input.get("input_fingerprint") == input_fingerprint
                and completion_state.get("input_fingerprint") == input_fingerprint
            ):
                validated = validate(cached_result)
                LOGGER.info("skip completed Stage 2 consolidation request: %s", output_dir)
                return validated
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
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


def _interpretation_response_contract() -> dict[str, Any]:
    """Return the shared response contract for interpretation passes."""

    return {
        "candidates": [
            {
                "name": "snake_case_clinical_feature_name",
                "description": (
                    "exactly one atomic, reusable patient-level clinical measurement or "
                    "attribute with one coherent value domain"
                ),
                "supporting_items": [1],
                "evidence_rationale": (
                    "how the cited words, phrases, or clinical context could arise from this "
                    "feature, including whether the feature is explicit or inferred"
                ),
                "caveats": "limitations, ambiguity, or competing clinical explanations",
            }
        ],
    }


def _atomic_feature_interpretation_rules() -> list[str]:
    """Return shared rules that keep discovered variables specific and scalar."""

    return [
        "Exhaustively enumerate every distinct atomic patient-level clinical feature explicitly stated or unambiguously encoded anywhere in each evidence item's text. Do not restrict candidates to the item's apparent topic, dominant concept, or consensus theme.",
        "Read every string in an evidence item's text array. Treat both consensus phrases and every representative excerpt as evidence; consensus phrases are not an exhaustive label for the variables present in the excerpts.",
        "Do not stop after finding the most salient feature, and do not limit an evidence item to one candidate. Inspect demographic clauses, headers, timepoint labels, laboratory lines, biomarker statements, diagnoses, symptoms, and other embedded fields separately.",
        "When one clause, header, or line explicitly states multiple independently varying patient attributes, return a separate atomic candidate for each attribute.",
        "Attribute each feature to the correct subject. Attributes belonging to relatives, specimens, clinicians, or other people do not support the corresponding patient feature.",
        "When an item contains multiple exemplar patients with different observed values of the same field, treat that as support for one reusable patient-level feature. Do not encode exemplar values or patient identities in the candidate name.",
        "Prefer atomic clinical variables.",
        "A candidate is atomic only when a downstream extractor could assign exactly one patient-level value under one coherent ontology. It must not require returning a list, set, tuple, mapping, concatenated code, profile, inventory, or ad hoc aggregation of independently varying values. Collapsing whether any member of an open-ended family is present into one indicator does not make that family atomic.",
        "Use the narrowest stable and reusable clinical construct directly supported by the cited evidence. Do not use a parent domain, umbrella label, or catch-all concept when the evidence supports separately measurable attributes.",
        "When evidence explicitly states or unambiguously encodes multiple independently meaningful components, return those components as separate candidates. Do not also return their umbrella or composite representation.",
        "Do not split a variable merely because its evidence contains different values, categories, thresholds, units, synonyms, or reporting formats. Those may be representations of one underlying measurement rather than distinct variables.",
        "An established construct that is conventionally reported as one scalar or category under one ontology remains one candidate even if that value is derived from multiple inputs. This exception does not apply to concatenated or multi-field encodings that preserve separately varying component values.",
        "Specificity concerns the measured clinical dimension, not a particular patient, observed value, document, or wording. Do not encode instance-specific details in a candidate name.",
        "Do not invent components that are not directly stated or unambiguously encoded in the evidence. If an atomic reusable variable cannot be identified, return no candidate rather than a vague catch-all.",
        "Each candidate name must identify its exact extraction target. A broad name cannot be repaired by placing a more specific target only in its description.",
    ]


def _interpretation_evidence_items(
    packets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose only readable text under prompt-local ordinal labels."""

    items: list[dict[str, Any]] = []
    for item_number, packet in enumerate(packets, start=1):
        texts = _readable_supporting_text([packet])
        if not texts:
            raise ValueError(f"interpretation evidence item {item_number} has no readable text")
        items.append({"item": item_number, "text": texts})
    return items


def _interpretation_prompt(
    *,
    architecture: str,
    packets: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    del architecture
    body = {
        "job": "infer_clinical_features_from_text_evidence",
        "task": (
            "Identify all patient-level clinical features supported anywhere in each supplied "
            "item's consensus text and representative excerpts. Enumerate every distinct "
            "feature atomically; some items may support multiple features or no valid feature."
        ),
        "rules": [
            "Each candidate must represent one patient-level clinical variable with one value per patient. It must be assignable by examining one patient's record without comparing or aggregating across patients.",
            "Return explicitly documented clinical features and narrower latent clinical features reasonably implied by the text.",
            *_atomic_feature_interpretation_rules(),
            "Use longitudinal information as clinical context when it appears in the evidence. Do not perform temporal eligibility filtering.",
            "Do not return patient names, administrative identifiers, documentation artifacts, descriptions of the input collection, multiple-patient heterogeneity, grouping methods, or analysis methods as clinical features.",
            "If no valid feature is supported, return an empty candidates list. Never turn the absence of a common feature into a candidate.",
            "Every returned candidate must have a nonempty snake_case name. Omit any candidate you cannot name; never return a blank or null name.",
            "For each candidate, cite one or more supplied item numbers in supporting_items and explain how its text supports the feature.",
            "Do not choose a value type, unit, categories, or extraction ontology in this step.",
        ],
        "evidence_items": _interpretation_evidence_items(packets),
        "response": _interpretation_response_contract(),
    }
    return [
        {
            "role": "system",
            "content": (
                "Exhaustively decompose every evidence item into all explicitly stated or "
                "unambiguously encoded atomic patient-level clinical features. Do not stop at "
                "the community's apparent topic or most salient feature. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _rejected_packet_audit_prompt(
    *,
    architecture: str,
    packets: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build a recall-oriented second pass over initially rejected packets."""

    del architecture
    body = {
        "job": "audit_unmapped_text_evidence_for_missed_clinical_features",
        "task": (
            "The supplied text was not cited by an initial review. Re-examine every string in "
            "every item and enumerate all atomic patient-level clinical features that may have "
            "been missed, including features embedded outside the apparent community topic."
        ),
        "rules": [
            "Review every evidence item independently. One clear item is sufficient to support a candidate; a clue does not need to recur.",
            "Each candidate must represent one patient-level clinical variable with one value per patient. It must be assignable by examining one patient's record without comparing or aggregating across patients.",
            "Return explicitly documented clinical features and narrower latent clinical features reasonably implied by the text.",
            *_atomic_feature_interpretation_rules(),
            "Use longitudinal information as clinical context when it appears in the evidence. Do not perform temporal eligibility filtering.",
            "Do not return patient names, administrative identifiers, documentation artifacts, descriptions of the input collection, multiple-patient heterogeneity, grouping methods, or analysis methods as clinical features.",
            "If no valid feature is supported, return an empty candidates list. Never turn the absence of a common feature into a candidate.",
            "Every returned candidate must have a nonempty snake_case name. Omit any candidate you cannot name; never return a blank or null name.",
            "For each candidate, cite one or more supplied item numbers in supporting_items and explain how its text supports the feature.",
            "Do not choose a value type, unit, categories, or extraction ontology in this step.",
        ],
        "evidence_items": _interpretation_evidence_items(packets),
        "response": _interpretation_response_contract(),
    }
    return [
        {
            "role": "system",
            "content": (
                "Re-examine every supplied text string for missed patient-level clinical "
                "features. Favor recall by exhaustively identifying all supported atomic "
                "variables, including secondary features outside the dominant topic, not by "
                "creating umbrella, inventory, or composite candidates. Return no candidate "
                "for input or analysis artifacts. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _validate_interpretation(
    value: Mapping[str, Any],
    *,
    packet_ids: Sequence[str],
    packet_evidence_axes: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    ordered_packet_ids = list(map(str, packet_ids))
    if len(ordered_packet_ids) != len(set(ordered_packet_ids)):
        raise ValueError("interpretation input contains duplicate packet IDs")
    packet_id_by_item = {
        item_number: packet_id for item_number, packet_id in enumerate(ordered_packet_ids, start=1)
    }
    payload = value
    if not isinstance(payload.get("concepts"), list):
        for key in ("result", "response", "interpretation"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    concepts = next(
        (
            payload.get(key)
            for key in ("concepts", "features", "variables", "candidates")
            if isinstance(payload.get(key), list)
        ),
        None,
    )
    if not isinstance(concepts, list):
        raise ValueError("interpretation requires a concepts list")
    clean_concepts: list[dict[str, Any]] = []
    for concept_index, concept in enumerate(concepts, start=1):
        if not isinstance(concept, Mapping):
            raise ValueError("each interpreted concept must be an object")
        name = str(concept.get("name") or concept.get("feature_name") or "").strip()
        if not name:
            LOGGER.warning(
                "Stage 2 interpretation ignored unnamed candidate at position=%s; "
                "its citations were not used",
                concept_index,
            )
            continue
        raw_supports = concept.get("supporting_items") or []
        if isinstance(raw_supports, (str, int)):
            raw_supports = [raw_supports]
        elif not isinstance(raw_supports, Sequence):
            raw_supports = []
        cited_items: list[int] = []
        invalid_items: list[Any] = []
        for raw_item in raw_supports:
            if isinstance(raw_item, bool):
                invalid_items.append(raw_item)
                continue
            try:
                item_number = int(raw_item)
            except (TypeError, ValueError):
                invalid_items.append(raw_item)
                continue
            if str(raw_item).strip() not in {str(item_number), f"{item_number}.0"}:
                invalid_items.append(raw_item)
                continue
            if item_number not in packet_id_by_item:
                invalid_items.append(raw_item)
                continue
            cited_items.append(item_number)
        cited_items = list(dict.fromkeys(cited_items))
        supports = [packet_id_by_item[item_number] for item_number in cited_items]
        if invalid_items:
            LOGGER.warning(
                "Stage 2 interpretation concept=%s ignored %s invalid supporting item(s): %s",
                name,
                len(invalid_items),
                invalid_items[:8],
            )
        if not supports:
            LOGGER.warning(
                "Stage 2 interpretation dropped ungrounded concept=%s; no supplied "
                "evidence item cited it",
                name,
            )
            continue
        if packet_evidence_axes is not None:
            axes = sorted(
                {
                    axis
                    for packet_id in supports
                    for axis in _canonical_evidence_axes(packet_evidence_axes.get(packet_id))
                }
            )
        else:
            axes = []
        evidence_rationale = str(
            concept.get("evidence_rationale")
            or concept.get("pattern_rationale")
            or concept.get("rationale")
            or ""
        ).strip()
        if not evidence_rationale:
            raise ValueError(
                f"interpreted candidate {name!r} has no evidence_rationale explaining "
                "how the cited text evidence could arise from it"
            )
        clean_concepts.append(
            {
                "name": name,
                "description": str(concept.get("description") or name),
                "supporting_packet_ids": supports,
                "evidence_axes": axes,
                "evidence_rationale": evidence_rationale,
                "caveats": str(concept.get("caveats") or ""),
            }
        )
    clean_dispositions: dict[str, Any] = {}
    for packet_id in ordered_packet_ids:
        names = sorted(
            concept["name"]
            for concept in clean_concepts
            if packet_id in concept["supporting_packet_ids"]
        )
        clean_dispositions[packet_id] = {
            "status": "supports_concept" if names else "reviewed_no_specific_concept",
            "concept_names": names,
            "reason": (
                "Derived from the candidates' evidence-item citations."
                if names
                else "No returned candidate cited this evidence item."
            ),
        }
    return {"concepts": clean_concepts, "packet_dispositions": clean_dispositions}


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


def _partition_packets_for_prompt(
    packets: Sequence[Mapping[str, Any]],
    *,
    render_prompt: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, str]]],
    max_prompt_chars: int,
) -> list[list[Mapping[str, Any]]]:
    """Pack evidence using the exact rendered prompt supplied by the caller."""

    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for packet in packets:
        candidate = [*current, packet]
        messages = render_prompt(candidate)
        prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
        if not current and prompt_chars > int(max_prompt_chars):
            raise ValueError("one Stage 2 evidence packet cannot fit the rendered prompt budget")
        if current and prompt_chars > int(max_prompt_chars):
            batches.append(current)
            current = [packet]
            singleton = render_prompt(current)
            if sum(len(message["content"]) for message in singleton) > int(max_prompt_chars):
                raise ValueError(
                    "one Stage 2 evidence packet cannot fit the rendered prompt budget"
                )
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _partition_interpretation_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    architecture: str,
    max_prompt_chars: int,
) -> list[list[Mapping[str, Any]]]:
    """Pack evidence using the exact fully rendered interpretation prompt."""

    return _partition_packets_for_prompt(
        packets,
        render_prompt=lambda batch: _interpretation_prompt(
            architecture=architecture,
            packets=batch,
        ),
        max_prompt_chars=max_prompt_chars,
    )


def _partition_rejected_packet_audit(
    packets: Sequence[Mapping[str, Any]],
    *,
    architecture: str,
    max_prompt_chars: int,
) -> list[list[Mapping[str, Any]]]:
    """Pack rejected packets using the exact rendered audit prompt."""

    return _partition_packets_for_prompt(
        packets,
        render_prompt=lambda batch: _rejected_packet_audit_prompt(
            architecture=architecture,
            packets=batch,
        ),
        max_prompt_chars=max_prompt_chars,
    )


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


def _consolidation_prompt(
    *,
    clinical_question: str,
    outer_fold: int,
    candidates: Sequence[Mapping[str, Any]],
    max_candidates: int,
) -> list[dict[str, str]]:
    del clinical_question, outer_fold, candidates, max_candidates
    raise RuntimeError(
        "the monolithic Stage 2 consolidation prompt is retired; use candidate-ID "
        "alias grouping and per-group operationalization"
    )


def _validate_consolidation(
    value: Mapping[str, Any],
    *,
    candidates: Sequence[Mapping[str, Any]],
    max_candidates: int,
) -> dict[str, Any]:
    # Retained only for validating historical monolithic-consolidation
    # responses. The former feature-count limit is intentionally ignored.
    del max_candidates
    features = value.get("features")
    dispositions = value.get("candidate_dispositions")
    if not isinstance(features, list):
        raise ValueError("consolidation requires a features list")
    if not isinstance(dispositions, Mapping):
        raise ValueError(
            "consolidation requires candidate_dispositions for every supplied candidate"
        )
    dispositions = {str(candidate_id): row for candidate_id, row in dispositions.items()}

    def feature_name_key(name: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    def string_list(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, (str, int, float, bool)):
            return [str(raw)]
        if not isinstance(raw, Sequence):
            return []
        return list(dict.fromkeys(str(item) for item in raw if item is not None))

    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    missing_disposition_ids = sorted(candidate_ids - set(dispositions))
    if missing_disposition_ids:
        raise ValueError(
            "consolidation omitted candidate disposition(s): " f"{missing_disposition_ids[:8]}"
        )
    extra_disposition_ids = sorted(set(dispositions) - candidate_ids)
    if extra_disposition_ids:
        LOGGER.warning(
            "Stage 2 consolidation ignored %s unknown candidate disposition(s): %s",
            len(extra_disposition_ids),
            extra_disposition_ids[:8],
        )
    status_aliases = {
        "keep": "retained",
        "kept": "retained",
        "retain": "retained",
        "combine": "merged",
        "combined": "merged",
        "merge": "merged",
        "drop": "excluded",
        "dropped": "excluded",
        "exclude": "excluded",
    }
    normalized_dispositions: dict[str, dict[str, str]] = {}
    for candidate_id in sorted(candidate_ids):
        raw_disposition = dispositions[candidate_id]
        if not isinstance(raw_disposition, Mapping):
            raise ValueError(f"candidate disposition {candidate_id!r} must be an object")
        status = str(raw_disposition.get("status") or "").strip().lower()
        status = status_aliases.get(status, status)
        if status not in {"retained", "merged", "excluded"}:
            raise ValueError(
                f"candidate disposition {candidate_id!r} has unsupported status {status!r}"
            )
        feature_name = str(raw_disposition.get("feature_name") or "").strip()
        if status != "excluded" and not feature_name:
            raise ValueError(
                f"candidate disposition {candidate_id!r} with status={status!r} "
                "must name a returned feature"
            )
        normalized_dispositions[candidate_id] = {
            "status": status,
            "feature_name": feature_name,
            "reason": str(raw_disposition.get("reason") or "").strip(),
        }
    allowed_packets = {
        str(packet_id)
        for candidate in candidates
        for packet_id in string_list(candidate.get("supporting_packet_ids"))
    }
    allowed_architectures = {
        str(architecture)
        for candidate in candidates
        for architecture in [
            candidate["architecture"],
            *string_list(candidate.get("supporting_architectures")),
        ]
    }
    packet_axes: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        for packet_id in string_list(candidate.get("supporting_packet_ids")):
            per_packet = candidate.get("packet_evidence_axes") or {}
            packet_axes[str(packet_id)].update(
                str(axis)
                for axis in per_packet.get(
                    str(packet_id),
                    candidate["evidence_axes"],
                )
            )
    clean_features: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features, start=1):
        if not isinstance(feature, Mapping):
            raise ValueError(f"consolidation feature at position={feature_index} must be an object")
        name = str(feature.get("name") or feature.get("feature_name") or "").strip()
        if not name:
            raise ValueError(f"consolidation feature at position={feature_index} has no name")
        name_key = feature_name_key(name)
        matched_candidate_ids = {
            candidate_id
            for candidate_id, disposition in normalized_dispositions.items()
            if disposition["status"] != "excluded"
            and feature_name_key(disposition["feature_name"]) == name_key
        }
        if not matched_candidate_ids:
            raise ValueError(
                f"returned feature {name!r} is not referenced by any retained or merged "
                "candidate disposition"
            )
        incompatible_candidate_ids = [
            candidate_id
            for candidate_id in sorted(matched_candidate_ids)
            if not _consolidation_route_is_semantically_compatible(
                candidate_by_id[candidate_id], feature
            )
        ]
        if incompatible_candidate_ids:
            incompatible_names = [
                str(candidate_by_id[candidate_id].get("name") or candidate_id)
                for candidate_id in incompatible_candidate_ids
            ]
            raise ValueError(
                f"returned feature {name!r} has semantically incompatible candidate "
                f"route(s): {incompatible_names[:8]}. Distinct measurements must not "
                "be merged merely because packet evidence overlaps"
            )

        cited_packets = string_list(
            feature.get("supporting_packet_ids") or feature.get("packet_ids")
        )
        unknown_packets = [
            packet_id for packet_id in cited_packets if packet_id not in allowed_packets
        ]
        if unknown_packets:
            raise ValueError(
                f"returned feature {name!r} cites unknown packet ID(s): " f"{unknown_packets[:8]}"
            )
        routed_packets = {
            str(packet_id)
            for candidate_id in matched_candidate_ids
            for packet_id in string_list(candidate_by_id[candidate_id].get("supporting_packet_ids"))
        }
        unrelated_packets = sorted(set(cited_packets) - routed_packets)
        if unrelated_packets:
            LOGGER.warning(
                "Stage 2 consolidation feature=%s discarded %s known packet citation(s) "
                "not carried by candidates routed to that feature: %s",
                name,
                len(unrelated_packets),
                unrelated_packets[:8],
            )
        packets = [packet_id for packet_id in cited_packets if packet_id in routed_packets]
        for candidate_id in sorted(matched_candidate_ids):
            packets.extend(string_list(candidate_by_id[candidate_id].get("supporting_packet_ids")))
        packets = list(
            dict.fromkeys(packet_id for packet_id in packets if packet_id in allowed_packets)
        )
        if not packets:
            raise ValueError(f"returned feature {name!r} has no supplied candidate evidence")

        raw_categories = feature.get("categories_or_unit")
        if isinstance(raw_categories, Mapping):
            raw_categories = (
                raw_categories.get("categories")
                or raw_categories.get("values")
                or raw_categories.get("unit")
            )
        if raw_categories is None:
            raw_categories = feature.get("categories") or feature.get("unit")
        categories = string_list(raw_categories)

        value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
        value_type = {
            "bool": "binary",
            "boolean": "binary",
            "category": "categorical",
            "numeric": "continuous",
            "number": "continuous",
            "unknown": "ambiguous",
        }.get(value_type, value_type)
        if value_type not in ALLOWED_VALUE_TYPES:
            value_type = "ambiguous"
        if value_type in {"binary", "categorical", "ordinal"}:
            # Models sometimes serialize an enumeration as one delimited string
            # or an ordinal ontology as an integer range such as ``0-4``.
            from .plain_handoff_stage2_analysis import _validated_closed_category_values

            categories = _validated_closed_category_values(
                value_type=value_type,
                values=categories,
                source=f"returned feature {name!r}",
            )

        architectures = [
            architecture
            for architecture in string_list(feature.get("supporting_architectures"))
            if architecture in allowed_architectures
        ]
        if not architectures:
            architectures = list(
                dict.fromkeys(
                    str(architecture)
                    for candidate_id, candidate in candidate_by_id.items()
                    if candidate_id in matched_candidate_ids
                    or set(string_list(candidate.get("supporting_packet_ids"))).intersection(
                        packets
                    )
                    for architecture in [
                        candidate["architecture"],
                        *string_list(candidate.get("supporting_architectures")),
                    ]
                    if str(architecture) in allowed_architectures
                )
            )

        description = str(feature.get("description") or name).strip()
        clean_features.append(
            {
                "name": name,
                "description": description,
                "value_type": value_type,
                "categories_or_unit": categories,
                "roles": [
                    role for role in string_list(feature.get("roles")) if role in ALLOWED_ROLES
                ],
                "measurement_definition": str(
                    feature.get("measurement_definition") or description
                ).strip(),
                "missing_value_rule": str(
                    feature.get("missing_value_rule")
                    or "Return null when not documented in the pretreatment record."
                ).strip(),
                "supporting_packet_ids": packets,
                "supporting_architectures": architectures,
                "stability_summary": str(feature.get("stability_summary") or ""),
                "caveats": str(feature.get("caveats") or ""),
            }
        )
    deduplicated_features: dict[str, dict[str, Any]] = {}
    for feature in clean_features:
        key = feature_name_key(feature["name"])
        existing = deduplicated_features.get(key)
        if existing is None:
            deduplicated_features[key] = feature
            continue
        raise ValueError(f"consolidation returned duplicate feature name {feature['name']!r}")
    clean_features = list(deduplicated_features.values())
    clean_dispositions: dict[str, dict[str, str]] = {}
    features_by_name = {feature["name"]: feature for feature in clean_features}
    features_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in clean_features:
        key = feature_name_key(feature["name"])
        features_by_key[key].append(feature)
    for candidate_id in sorted(candidate_ids):
        raw_disposition = normalized_dispositions[candidate_id]
        candidate_packets = {
            str(packet_id)
            for packet_id in string_list(candidate_by_id[candidate_id].get("supporting_packet_ids"))
        }
        status = raw_disposition["status"]
        feature_name = raw_disposition["feature_name"]
        reason = raw_disposition["reason"]
        if status == "excluded":
            clean_dispositions[candidate_id] = {
                "status": "excluded",
                "feature_name": "",
                "reason": reason or "Candidate was excluded by consolidation.",
            }
            continue
        feature = features_by_name.get(feature_name)
        if feature is None and feature_name:
            key = feature_name_key(feature_name)
            matches = features_by_key.get(key, [])
            if len(matches) == 1:
                feature = matches[0]
        if feature is None:
            raise ValueError(
                f"candidate disposition {candidate_id!r} references missing returned "
                f"feature {feature_name!r}"
            )
        if not _consolidation_route_is_semantically_compatible(
            candidate_by_id[candidate_id], feature
        ):
            raise ValueError(
                f"candidate {candidate_id!r} "
                f"({candidate_by_id[candidate_id].get('name')!r}) cannot be merged "
                f"into semantically incompatible feature {feature['name']!r}"
            )
        if not candidate_packets <= set(feature["supporting_packet_ids"]):
            raise ValueError(
                f"returned feature {feature['name']!r} did not preserve all packet "
                f"evidence for candidate {candidate_id!r}"
            )
        clean_dispositions[candidate_id] = {
            "status": status,
            "feature_name": str(feature["name"]),
            "reason": reason or "Candidate was reconciled to the returned grounded feature.",
        }

    routed_features: list[dict[str, Any]] = []
    for feature in clean_features:
        axes = {
            axis
            for packet_id in feature["supporting_packet_ids"]
            for axis in packet_axes.get(packet_id, set())
        }
        derived_roles: list[str] = []
        if {"treatment", "outcome"} <= axes:
            derived_roles.append("confounder")
        elif "outcome" in axes:
            derived_roles.append("prognostic")
        if axes.intersection({"residual_effect", "matched_pair"}):
            derived_roles.append("effect_modifier")
        if not derived_roles:
            raise ValueError(
                f"returned feature {feature['name']!r} has no supported Stage 2 causal role"
            )
        feature["roles"] = derived_roles
        routed_features.append(feature)

    routed_names = {feature["name"] for feature in routed_features}
    for disposition in clean_dispositions.values():
        if disposition["status"] != "excluded" and disposition["feature_name"] not in routed_names:
            raise ValueError(
                "candidate disposition references unrouted feature "
                f"{disposition['feature_name']!r}"
            )
    used_names = {
        disposition["feature_name"]
        for disposition in clean_dispositions.values()
        if disposition["status"] != "excluded"
    }
    unused_features = [
        feature["name"] for feature in routed_features if feature["name"] not in used_names
    ]
    if unused_features:
        raise ValueError(
            "returned feature(s) have no retained or merged candidate route: "
            f"{unused_features[:8]}"
        )
    return {"features": routed_features, "candidate_dispositions": clean_dispositions}


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


def _group_roles(group: Mapping[str, Any]) -> list[str]:
    configured = _configured_feature_definitions(group)
    if configured:
        if len(configured) != 1:
            raise ValueError(
                "Stage 2 consolidation attempted to merge multiple investigator-configured "
                f"features: {[feature['name'] for feature in configured]}"
            )
        return [
            role for role in _string_values(configured[0].get("roles")) if role in ALLOWED_ROLES
        ]
    return _derive_roles(group.get("evidence_axes") or [])


def _filter_candidate_groups_by_causal_role(
    groups: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Apply the auditable causal-role filter after lossless alias consolidation."""

    retained: list[dict[str, Any]] = []
    exclusions: dict[str, str] = {}
    decisions: list[dict[str, Any]] = []
    exclusion_reason = (
        "Excluded because its Stage 1 evidence does not support a Stage 2 "
        "confounder, prognostic, or effect-modifier role."
    )
    for group in groups:
        roles = _group_roles(group)
        origin_candidate_ids = _string_values(group.get("origin_candidate_ids"))
        if roles:
            retained.append(dict(group))
            decisions.append(
                {
                    "name": str(group["name"]),
                    "status": "retained",
                    "roles": roles,
                    "origin_candidate_ids": origin_candidate_ids,
                }
            )
            continue
        for origin in origin_candidate_ids:
            exclusions[origin] = exclusion_reason
        decisions.append(
            {
                "name": str(group["name"]),
                "status": "excluded",
                "roles": [],
                "origin_candidate_ids": origin_candidate_ids,
                "reason": exclusion_reason,
            }
        )
    return retained, exclusions, decisions


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


def _alphabetical_candidate_batches(
    groups: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    round_number: int,
) -> tuple[int, list[list[dict[str, Any]]]]:
    """Sort groups and shift nonoverlapping batch boundaries between rounds."""

    if batch_size < 2:
        raise ValueError("candidate consolidation batch_size must be at least 2")
    if round_number < 1:
        raise ValueError("candidate consolidation round_number must be positive")
    ordered = [dict(group) for group in sorted(groups, key=_candidate_group_sort_key)]
    if not ordered:
        return 0, []
    if len(ordered) <= batch_size:
        return 0, [ordered]

    shift_step = max(1, batch_size // 2)
    while math.gcd(shift_step, batch_size) != 1:
        shift_step += 1
    boundary_offset = ((round_number - 1) * shift_step) % batch_size
    batches: list[list[dict[str, Any]]] = []
    cursor = 0
    if boundary_offset:
        batches.append(ordered[:boundary_offset])
        cursor = boundary_offset
    while cursor < len(ordered):
        batches.append(ordered[cursor : cursor + batch_size])
        cursor += batch_size
    return boundary_offset, batches


def _seeded_shuffle_candidate_batches(
    groups: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    seed: int,
    shuffle_round: int,
) -> list[list[dict[str, Any]]]:
    """Deterministically shuffle a sorted pool before forming bounded batches."""

    if batch_size < 2:
        raise ValueError("candidate consolidation batch_size must be at least 2")
    if shuffle_round < 1:
        raise ValueError("candidate consolidation shuffle_round must be positive")
    ordered = [dict(group) for group in sorted(groups, key=_candidate_group_sort_key)]

    def shuffled_key(group: Mapping[str, Any]) -> tuple[str, tuple[str, str, str]]:
        identity = "\0".join(
            (
                str(int(seed)),
                str(int(shuffle_round)),
                str(group.get("name") or ""),
                str(group.get("candidate_id") or ""),
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest(), _candidate_group_sort_key(
            group
        )

    shuffled = sorted(ordered, key=shuffled_key)
    return [shuffled[start : start + batch_size] for start in range(0, len(shuffled), batch_size)]


def _candidate_consolidation_batches(
    groups: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    round_number: int,
    alphabetical_rounds: int,
    seed: int,
) -> tuple[str, int | None, int | None, list[list[dict[str, Any]]]]:
    """Use shifted alphabetical partitions, then seeded shuffled partitions."""

    if alphabetical_rounds < 0:
        raise ValueError("candidate consolidation alphabetical_rounds must be nonnegative")
    if round_number <= alphabetical_rounds:
        boundary_offset, batches = _alphabetical_candidate_batches(
            groups,
            batch_size=batch_size,
            round_number=round_number,
        )
        return "alphabetical_shift", boundary_offset, None, batches
    shuffle_round = round_number - alphabetical_rounds
    return (
        "seeded_shuffle",
        None,
        shuffle_round,
        _seeded_shuffle_candidate_batches(
            groups,
            batch_size=batch_size,
            seed=seed,
            shuffle_round=shuffle_round,
        ),
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


def _derive_roles(evidence_axes: Sequence[str]) -> list[str]:
    axes = set(_canonical_evidence_axes(evidence_axes))
    roles: list[str] = []
    if {"treatment", "outcome"} <= axes:
        roles.append("confounder")
    elif "outcome" in axes:
        roles.append("prognostic")
    if axes.intersection({"residual_effect", "matched_pair"}):
        roles.append("effect_modifier")
    return roles


def _global_candidate_pool_prompt(
    *,
    groups: Sequence[Mapping[str, Any]],
    configured_feature_names: Sequence[str] = (),
    batch_ordering: str = "alphabetical_shift",
) -> list[dict[str, str]]:
    """Consolidate aliases in one bounded candidate-pool batch without filtering."""

    features = [
        _candidate_pool_feature_view(group)
        for group in sorted(groups, key=_candidate_group_sort_key)
    ]

    body: dict[str, Any] = {
        "job": "consolidate_stage2_candidate_pool",
        "task": (
            "Review this "
            + (
                "alphabetically adjacent"
                if batch_ordering == "alphabetical_shift"
                else "deterministically shuffled"
            )
            + " batch of interpreted candidate features. "
            "Partition semantic aliases and equivalent representations of each underlying "
            "patient-level measurement within the supplied batch. Every supplied feature "
            "must survive this pass either unchanged or as an input to exactly one merge. "
            "Later rounds will use new deterministic partitions of the consolidated candidates."
        ),
        "features": features,
        "rules": [
            "Every name absent from merge_directives will be retained unchanged; do not restate unchanged features.",
            "This is merge-only ontology consolidation, not feature filtering or quality review. Never exclude or drop a supplied feature.",
            "Each merge directive must contain at least two exact names from features.",
            "Treat merge_directives as a disjoint partition of alias families within this batch, not as sequential rename operations: return exactly one directive for each complete supplied alias family and never chain or split one family across directives.",
            "Each directive's inputs must list every exact supplied feature name in that alias family within this batch, including the selected canonical name when output reuses a supplied feature name.",
            "An output that equals a supplied feature name is valid only when that exact name appears in the same directive's inputs; it must not be an input of another directive or an unchanged feature.",
            "Use each feature name at most once across all merge inputs.",
            "Merge spelling variants, abbreviations, synonymous clinical names, and all clearly equivalent representations of the same underlying measurement.",
            "A general measurement name, its quantitative score, a thresholded or coarsened status, a named category, and a name containing one observed value belong together when they can all be represented by one underlying patient variable.",
            "Prefer an information-preserving underlying measurement name over a threshold, category, or observed value encoded in one candidate name.",
            "When a value-encoded or awkward alias has a clear underlying measurement in this batch, merge it into that measurement; otherwise retain it unchanged.",
            "Judge alias families jointly across this entire batch; do not require a direct lexical match between every pair of members in one family.",
            "Do not merge merely related but independently varying variables, a diagnosis with a related laboratory value, a broad concept with one independently varying component, different anatomical sites, different biomarkers, or different timepoints.",
            "Merge only true semantic aliases of the same atomic clinical variable. Inputs are aliases only when they identify the same measured dimension and can share one extraction ontology without discarding an independently varying component.",
            "Every merge output must itself be atomic: one patient-level value under one coherent ontology.",
            "The canonical output name must identify the exact clinical dimension shared by every merge input. It must not broaden them into a parent domain, umbrella, inventory, profile, or composite construct.",
            "Never introduce a broader name merely to make related candidates appear mergeable. Clinical relatedness, correlation, shared anatomy, shared domain, or membership in the same assessment does not establish semantic equivalence.",
            "Do not merge constituent variables that can vary independently. If no precise atomic target is common to every input, retain the inputs unchanged.",
            "Differences that encode only values, categories, thresholds, units, spelling, abbreviations, or reporting formats may still represent aliases of one underlying variable. Preserve the measured dimension without encoding a particular observed value in the canonical name.",
            "The output must be one concise snake_case canonical name for the exact consolidated measurement. It may reuse the best input name or provide a clearer equivalent name.",
            "When semantic equivalence is uncertain, do not merge the features.",
            "Return only exact supplied feature names in merge inputs. Return no internal IDs, provenance, definitions, explanations, unchanged feature names, or exclusion list.",
        ],
        "response": {
            "merge_directives": [
                {
                    "inputs": [
                        "all exact supplied names in one alias family, including a reused output name"
                    ],
                    "output": "one snake_case canonical feature name",
                }
            ]
        },
    }
    configured_names = list(map(str, configured_feature_names))
    if configured_names:
        body["configured_feature_names"] = configured_names
        body["rules"].extend(
            [
                "Never merge two names listed in configured_feature_names; the investigator specified them as distinct features.",
                "When one merge input is listed in configured_feature_names, output that exact configured name so its investigator-supplied ontology remains authoritative.",
            ]
        )
    return [
        {
            "role": "system",
            "content": "Consolidate aliases without filtering any features. Return JSON only.",
        },
        {
            "role": "user",
            "content": json.dumps(body, sort_keys=True, ensure_ascii=False),
        },
    ]


def _validate_global_candidate_pool_directives(
    value: Mapping[str, Any],
    *,
    group_names: Sequence[str],
    configured_feature_names: Sequence[str] = (),
    group_descriptions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Validate directives using only names and descriptions supplied in the batch."""

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


def _operationalization_prompt(
    *,
    feature_name: str,
    supporting_evidence: Sequence[str],
) -> list[dict[str, str]]:
    evidence = list(
        dict.fromkeys(str(item).strip() for item in supporting_evidence if str(item).strip())
    )
    if not evidence:
        raise ValueError("operationalization requires readable supporting evidence")
    instructions = {
        "task": (
            "Define the extraction ontology for the named candidate clinical feature. "
            "Decide its value type, allowed values or unit, and measurement rule from the "
            "candidate name and readable supporting clinical evidence."
        ),
        "rules": [
            "Define exactly the named scalar pretreatment measurement; do not rename, merge, or split it in this step.",
            "Determine value_type yourself from what the named feature means and how it is represented in the supplied evidence. No value type from an earlier discovery step is being provided.",
            "The evidence may contain unrelated clues; use only text that actually bears on the named feature.",
            "Specify a reproducible extraction target from a complete patient record.",
            "Do not invent an ad hoc score, formula, or index to force multiple distinct measurements into one scalar.",
            "Prefer value_type continuous, with a clinically meaningful unit when applicable, when the named feature can realistically be extracted as a numeric measurement. Use categorical or ordinal only when continuous measurement is infeasible or would misrepresent the feature.",
            "A continuous ontology may preserve a categorical or threshold report when a patient record lacks an exact number. Describe those evidence-supported fallback representations in the measurement rule; aggregate outer-training values will later determine continuous, categorical, or hybrid modeling.",
            "For categorical or ordinal variables, enumerate the extraction ontology; for continuous variables, provide the unit when applicable.",
            "For a binary variable, categories_or_unit must contain exactly two distinct extractable scalar values as separate array items.",
            "For a categorical or ordinal variable, categories_or_unit must contain at least two distinct extractable scalar values as separate array items.",
            "List each category exactly once. Categories that differ only by capitalization, punctuation, underscores, or spacing are duplicates and must not both appear.",
            "Never use a type label such as binary or categorical, or a combined phrase such as present-or-absent, as one ontology value.",
            "Define how absent, ambiguous, and conflicting documentation is represented.",
            "Choose one conflict_resolution strategy for multiple longitudinal observations: latest, earliest, maximum, minimum, mode, any_positive, or single_or_null.",
            "Use maximum or minimum only for continuous measurements. Use any_positive only for a binary ontology and provide its exact affirmative category as positive_category.",
            "Use single_or_null when conflicting supported values cannot be scientifically resolved by a reproducible patient-level rule.",
            "Return one flat JSON object with every response field shown below; measurement_definition and missing_value_rule are required nonempty strings.",
        ],
        "response": {
            "description": "one patient-level scalar measurement",
            "value_type": "binary|categorical|continuous|ordinal|ambiguous",
            "categories_or_unit": ["categories or one unit string"],
            "measurement_definition": "what to extract from the pretreatment record",
            "missing_value_rule": "how missing or ambiguous documentation is represented",
            "conflict_resolution": {
                "strategy": "latest|earliest|maximum|minimum|mode|any_positive|single_or_null",
                "positive_category": "exact affirmative binary category, otherwise null",
            },
            "stability_summary": "scientific support summary without provenance identifiers",
            "caveats": "remaining scientific limitations",
        },
    }
    body = {
        "candidate_feature_name": str(feature_name),
        "supporting_evidence": evidence,
    }
    return [
        {
            "role": "system",
            "content": json.dumps(
                {
                    "instruction": (
                        "Define one clinical feature ontology from its name and supporting "
                        "clinical text. Return JSON only."
                    ),
                    **instructions,
                },
                sort_keys=True,
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


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


def _colbert_supporting_text(packet: Mapping[str, Any]) -> str:
    """Return the lossless router document or the packet's readable excerpts."""

    content = packet.get("content")
    if isinstance(content, Mapping):
        hierarchy_document = str(content.get("colbert_document") or "").strip()
        if hierarchy_document:
            return hierarchy_document
    return "\n\n".join(_readable_supporting_text([packet]))


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


def _normalized_embedding_matrix(
    texts: Sequence[str],
    *,
    model_name: str,
    device: str,
    embedding_function: Callable[[Sequence[str], str, str], np.ndarray] | None,
) -> np.ndarray:
    encode = embedding_function or _encode_candidate_selection_texts
    matrix = np.asarray(encode(texts, model_name, device), dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] < 1:
        raise RuntimeError(
            "candidate registry embedding function returned unexpected shape "
            f"{matrix.shape} for {len(texts)} texts"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("candidate registry embedding function returned non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise RuntimeError("candidate registry embedding function returned a zero vector")
    return matrix / norms


def _build_candidate_registry(
    *,
    candidates: Sequence[Mapping[str, Any]],
    embedding_model: str,
    embedding_device: str,
    similarity_threshold: float,
    embedding_function: Callable[[Sequence[str], str, str], np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse exact names, then conservatively merge anchored semantic aliases."""

    if (
        isinstance(similarity_threshold, bool)
        or not isinstance(similarity_threshold, (int, float))
        or not math.isfinite(float(similarity_threshold))
        or not 0.0 < float(similarity_threshold) <= 1.0
    ):
        raise ValueError("candidate registry similarity_threshold must be between 0 and 1")
    candidate_ids = [str(candidate.get("candidate_id") or "") for candidate in candidates]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("candidate registry inputs must have candidate_id values")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate registry received duplicate candidate IDs")
    exact_groups = _materialize_exact_name_groups(candidates)
    texts = [_candidate_registry_text(group) for group in exact_groups]
    anchors = [
        _concept_identity_tokens(
            group.get("name"),
            group.get("description"),
            *(member.get("description") for member in group.get("member_measurements") or []),
        )
        for group in exact_groups
    ]
    anchor_counts = Counter(anchor for values in anchors for anchor in values)
    max_anchor_frequency = max(
        8,
        min(128, int(math.ceil(math.sqrt(max(1, len(exact_groups)))))),
    )
    informative_anchors = {
        anchor
        for anchor, count in anchor_counts.items()
        if 1 < count <= max_anchor_frequency
    }
    routing_anchors = [values.intersection(informative_anchors) for values in anchors]
    comparison_indexes = {
        index
        for index, values in enumerate(routing_anchors)
        if values
    }
    embedding_invoked = len(comparison_indexes) > 1
    matrix = (
        _normalized_embedding_matrix(
            texts,
            model_name=str(embedding_model).strip(),
            device=str(embedding_device).strip(),
            embedding_function=embedding_function,
        )
        if embedding_invoked
        else None
    )

    clusters: list[dict[str, Any]] = []
    cluster_ids_by_anchor: dict[str, set[int]] = defaultdict(set)
    assignments: list[dict[str, Any]] = []
    for index, group in enumerate(exact_groups):
        eligible_cluster_ids = sorted(
            {
                cluster_id
                for anchor in routing_anchors[index]
                for cluster_id in cluster_ids_by_anchor.get(anchor, set())
            }
        )
        best_cluster_id: int | None = None
        best_similarity: float | None = None
        if matrix is not None and index in comparison_indexes:
            for cluster_id in eligible_cluster_ids:
                similarity = float(np.dot(matrix[index], clusters[cluster_id]["centroid"]))
                if similarity < float(similarity_threshold):
                    continue
                if best_similarity is None or similarity > best_similarity:
                    best_cluster_id = cluster_id
                    best_similarity = similarity
        if best_cluster_id is None:
            cluster_id = len(clusters)
            clusters.append(
                {
                    "members": [group],
                    "member_indexes": [index],
                    "anchors": set(routing_anchors[index]),
                    "centroid": matrix[index].copy() if matrix is not None else None,
                }
            )
            for anchor in routing_anchors[index]:
                cluster_ids_by_anchor[anchor].add(cluster_id)
            assignments.append(
                {
                    "exact_group_id": str(group["candidate_id"]),
                    "registry_cluster": cluster_id + 1,
                    "action": "new_cluster",
                    "cosine_similarity": None,
                }
            )
            continue
        cluster = clusters[best_cluster_id]
        cluster["members"].append(group)
        cluster["member_indexes"].append(index)
        cluster["anchors"].update(routing_anchors[index])
        centroid = matrix[cluster["member_indexes"]].mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        cluster["centroid"] = centroid / centroid_norm
        for anchor in routing_anchors[index]:
            cluster_ids_by_anchor[anchor].add(best_cluster_id)
        assignments.append(
            {
                "exact_group_id": str(group["candidate_id"]),
                "registry_cluster": best_cluster_id + 1,
                "action": "semantic_merge",
                "cosine_similarity": best_similarity,
            }
        )

    registry: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        members = list(cluster["members"])
        canonical = _canonical_cluster_member(members)
        canonical_name = _snake_case_name(
            canonical.get("name"),
            fallback=f"measurement_{cluster_index:03d}",
        )
        registry_group = _materialize_candidate_group(
            candidate_id=f"candidate_registry_{cluster_index:04d}",
            members=members,
            canonical_name=canonical_name,
            canonical_description=_short_text(
                canonical.get("description") or canonical_name,
                max_chars=2_000,
            ),
        )
        registry.append(registry_group)
        cluster_rows.append(
            {
                "candidate_id": registry_group["candidate_id"],
                "canonical_name": registry_group["name"],
                "exact_group_names": [str(member["name"]) for member in members],
                "origin_candidate_ids": list(registry_group["origin_candidate_ids"]),
                "supporting_packet_ids": list(registry_group["supporting_packet_ids"]),
            }
        )
    return registry, {
        "raw_candidates": len(candidates),
        "exact_name_groups": len(exact_groups),
        "registry_candidates": len(registry),
        "exact_name_merges": len(candidates) - len(exact_groups),
        "semantic_merges": len(exact_groups) - len(registry),
        "semantic_embedding_invoked": embedding_invoked,
        "semantic_embedding_model": str(embedding_model).strip(),
        "semantic_embedding_device": str(embedding_device).strip(),
        "semantic_similarity_threshold": float(similarity_threshold),
        "semantic_comparison_candidates": len(comparison_indexes),
        "semantic_anchor_max_frequency": max_anchor_frequency,
        "semantic_ignored_high_frequency_anchors": sorted(
            anchor
            for anchor, count in anchor_counts.items()
            if count > max_anchor_frequency
        ),
        "assignments": assignments,
        "clusters": cluster_rows,
    }


def _packet_support_architectures(packet: Mapping[str, Any]) -> list[str]:
    content = packet.get("content")
    architectures = content.get("source_architectures") if isinstance(content, Mapping) else None
    if not architectures and packet.get("architecture"):
        architectures = [packet["architecture"]]
    return list(dict.fromkeys(_string_values(architectures)))


def _packet_support_inner_folds(packet: Mapping[str, Any]) -> list[int]:
    content = packet.get("content")
    support = content.get("support") if isinstance(content, Mapping) else None
    raw_folds = support.get("inner_folds") if isinstance(support, Mapping) else None
    folds: list[int] = []
    for value in raw_folds or []:
        try:
            fold = int(value)
        except (TypeError, ValueError):
            continue
        if fold not in folds:
            folds.append(fold)
    return folds


def _dense_candidate_packet_scores(
    queries: Sequence[str],
    documents: Sequence[str],
    *,
    embedding_model: str,
    embedding_device: str,
    embedding_function: Callable[[Sequence[str], str, str], np.ndarray] | None,
) -> np.ndarray:
    texts = list(dict.fromkeys([*queries, *documents]))
    matrix = _normalized_embedding_matrix(
        texts,
        model_name=embedding_model,
        device=embedding_device,
        embedding_function=embedding_function,
    )
    index = {text: position for position, text in enumerate(texts)}
    return np.asarray(
        [
            float(np.dot(matrix[index[query]], matrix[index[document]]))
            for query, document in zip(queries, documents)
        ],
        dtype=np.float32,
    )


def _candidate_selection_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["aggregate_support_score"]),
        -float(row["top_evidence_mean_score"]),
        -float(row["best_evidence_score"]),
        -float(row["architecture_coverage"]),
        -float(row["inner_fold_coverage"]),
        -int(row["supporting_packet_count"]),
        str(row["feature_name"]).casefold(),
        str(row["candidate_id"]),
    )


def _select_candidates_from_registry(
    *,
    candidates: Sequence[Mapping[str, Any]],
    packet_by_id: Mapping[str, Mapping[str, Any]],
    hierarchy_packet_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    top_n_per_axis: int,
    max_candidates: int,
    scoring_method: str,
    late_interaction_model: str,
    late_interaction_device: str,
    dense_embedding_model: str,
    dense_embedding_device: str,
    top_evidence_packets: int,
    document_chunk_overlap_tokens: int,
    hierarchy_top_communities: int = DEFAULT_CANDIDATE_SELECTION_HIERARCHY_TOP_COMMUNITIES,
    embedding_function: Callable[[Sequence[str], str, str], np.ndarray] | None = None,
    late_interaction_scoring_function: (
        Callable[[Sequence[str], Sequence[str], str, str], np.ndarray] | None
    ) = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank canonical candidates by evidence support and enforce fold-level caps."""

    if (
        top_n_per_axis < 1
        or max_candidates < 1
        or top_evidence_packets < 1
        or hierarchy_top_communities < 1
    ):
        raise ValueError("candidate selection limits must be positive")
    if scoring_method not in {"late_interaction", "dense_cosine"}:
        raise ValueError(f"unsupported candidate selection scoring method {scoring_method!r}")
    if not candidates:
        scoring_model = (
            late_interaction_model
            if scoring_method == "late_interaction"
            else dense_embedding_model
        )
        scoring_device = (
            late_interaction_device
            if scoring_method == "late_interaction"
            else dense_embedding_device
        )
        return [], {
            "scoring_method": scoring_method,
            "scoring_model": scoring_model,
            "scoring_device": scoring_device,
            "top_n_per_evidence_axis": top_n_per_axis,
            "max_candidates_per_fold": max_candidates,
            "top_evidence_packets_per_candidate": top_evidence_packets,
            "hierarchy_top_communities_per_candidate": hierarchy_top_communities,
            "hierarchy_packets": len(hierarchy_packet_by_id or {}),
            "document_chunk_overlap_tokens": document_chunk_overlap_tokens,
            "registry_candidates": 0,
            "candidate_packet_associations_scored": 0,
            "candidate_hierarchy_associations_scored": 0,
            "shortlisted_candidates_before_fold_cap": 0,
            "hard_cap_applied": False,
            "retained_candidates": 0,
            "dropped_registry_candidate_ids": [],
            "dropped_origin_candidate_ids": [],
            "score_aggregation": {
                "top_evidence_mean_count": top_evidence_packets,
                "relevance_weight": 0.75,
                "architecture_coverage_weight": 0.125,
                "inner_fold_coverage_weight": 0.125,
                "relevance_normalization": "clip((mean_maxsim + 1) / 2, 0, 1)",
            },
            "axis_rankings": {},
            "candidate_rankings": [],
        }
    ordered_packets = {str(packet_id): packet for packet_id, packet in packet_by_id.items()}
    ordered_hierarchy_packets = {
        str(packet_id): packet
        for packet_id, packet in (hierarchy_packet_by_id or {}).items()
    }
    candidate_by_id: dict[str, Mapping[str, Any]] = {}
    support_ids_by_candidate: dict[str, list[str]] = {}
    query_by_candidate: dict[str, str] = {}
    for position, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in candidate_by_id:
            raise ValueError(
                "candidate selection requires unique candidate IDs; invalid value at "
                f"position {position}: {candidate_id!r}"
            )
        support_ids = list(dict.fromkeys(_string_values(candidate.get("supporting_packet_ids"))))
        if not support_ids:
            raise ValueError(f"candidate selection input {candidate_id!r} has no supporting packet")
        unknown = sorted(set(support_ids) - set(ordered_packets))
        if unknown:
            raise ValueError(
                f"candidate selection input {candidate_id!r} cites unknown packets: {unknown[:8]}"
            )
        query = _natural_language_feature_name(candidate.get("name"))
        if not query:
            raise ValueError(f"candidate selection input {candidate_id!r} has no readable name")
        candidate_by_id[candidate_id] = candidate
        support_ids_by_candidate[candidate_id] = support_ids
        query_by_candidate[candidate_id] = query

    scoring_model = (
        late_interaction_model
        if scoring_method == "late_interaction"
        else dense_embedding_model
    )
    scoring_device = (
        late_interaction_device
        if scoring_method == "late_interaction"
        else dense_embedding_device
    )

    def score_associations(
        association_keys: Sequence[tuple[str, str]],
        *,
        packets_for_scoring: Mapping[str, Mapping[str, Any]],
    ) -> dict[tuple[str, str], float]:
        if not association_keys:
            return {}
        text_by_packet_id: dict[str, str] = {}
        queries: list[str] = []
        documents: list[str] = []
        for candidate_id, packet_id in association_keys:
            if packet_id not in text_by_packet_id:
                text = _colbert_supporting_text(packets_for_scoring[packet_id])
                if not text:
                    raise ValueError(
                        f"candidate selection packet {packet_id!r} has no readable "
                        "evidence text"
                    )
                text_by_packet_id[packet_id] = text
            queries.append(query_by_candidate[candidate_id])
            documents.append(text_by_packet_id[packet_id])
        if scoring_method == "late_interaction":
            if late_interaction_scoring_function is None:
                from ..models.late_interaction import score_late_interaction_pairs

                raw_scores = score_late_interaction_pairs(
                    queries,
                    documents,
                    late_interaction_model,
                    late_interaction_device,
                    document_chunk_overlap_tokens=document_chunk_overlap_tokens,
                )
            else:
                raw_scores = late_interaction_scoring_function(
                    queries,
                    documents,
                    late_interaction_model,
                    late_interaction_device,
                )
        else:
            raw_scores = _dense_candidate_packet_scores(
                queries,
                documents,
                embedding_model=dense_embedding_model,
                embedding_device=dense_embedding_device,
                embedding_function=embedding_function,
            )
        scores = np.asarray(raw_scores, dtype=np.float32)
        if scores.ndim != 1 or scores.shape[0] != len(association_keys):
            raise RuntimeError(
                "candidate support scorer returned unexpected shape "
                f"{scores.shape} for {len(association_keys)} associations"
            )
        if not np.isfinite(scores).all():
            raise RuntimeError("candidate support scorer returned non-finite scores")
        return {
            key: float(score)
            for key, score in zip(association_keys, scores.tolist())
        }

    hierarchy_association_keys = [
        (candidate_id, packet_id)
        for candidate_id in candidate_by_id
        for packet_id in sorted(ordered_hierarchy_packets)
    ]
    hierarchy_score_by_association = score_associations(
        hierarchy_association_keys,
        packets_for_scoring=ordered_hierarchy_packets,
    )
    top_hierarchy_ids_by_candidate: dict[str, list[str]] = {}
    retrieval_ids_by_candidate: dict[str, list[str]] = {}
    for candidate_id in candidate_by_id:
        ranked_hierarchy_ids = sorted(
            ordered_hierarchy_packets,
            key=lambda packet_id: (
                -hierarchy_score_by_association[(candidate_id, packet_id)],
                packet_id,
            ),
        )
        top_hierarchy_ids = ranked_hierarchy_ids[
            : min(hierarchy_top_communities, len(ranked_hierarchy_ids))
        ]
        top_hierarchy_ids_by_candidate[candidate_id] = top_hierarchy_ids
        routed_ids: list[str] = list(support_ids_by_candidate[candidate_id])
        for hierarchy_id in top_hierarchy_ids:
            content = ordered_hierarchy_packets[hierarchy_id].get("content")
            source_packet_ids = (
                _string_values(content.get("source_packet_ids"))
                if isinstance(content, Mapping)
                else []
            )
            routed_ids.extend(
                packet_id
                for packet_id in source_packet_ids
                if packet_id in ordered_packets
            )
        retrieval_ids_by_candidate[candidate_id] = list(dict.fromkeys(routed_ids))

    association_keys = [
        (candidate_id, packet_id)
        for candidate_id in candidate_by_id
        for packet_id in retrieval_ids_by_candidate[candidate_id]
    ]
    score_by_association = score_associations(
        association_keys,
        packets_for_scoring=ordered_packets,
    )

    fold_architectures = {
        architecture
        for packet in ordered_packets.values()
        for architecture in _packet_support_architectures(packet)
    }
    fold_inner_folds = {
        fold
        for packet in ordered_packets.values()
        for fold in _packet_support_inner_folds(packet)
    }
    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in candidate_by_id.items():
        support_ids = support_ids_by_candidate[candidate_id]
        retrieval_ids = retrieval_ids_by_candidate[candidate_id]
        ranked_packets = sorted(
            retrieval_ids,
            key=lambda packet_id: (
                -score_by_association[(candidate_id, packet_id)],
                packet_id,
            ),
        )
        top_packet_ids = ranked_packets[: min(top_evidence_packets, len(ranked_packets))]
        top_scores = [
            score_by_association[(candidate_id, packet_id)] for packet_id in top_packet_ids
        ]
        top_mean = float(np.mean(top_scores))
        best_score = float(top_scores[0])
        candidate_architectures = {
            architecture
            for packet_id in support_ids
            for architecture in _packet_support_architectures(ordered_packets[packet_id])
        }
        candidate_inner_folds = {
            fold
            for packet_id in support_ids
            for fold in _packet_support_inner_folds(ordered_packets[packet_id])
        }
        architecture_coverage = (
            len(candidate_architectures) / len(fold_architectures)
            if fold_architectures
            else 1.0
        )
        inner_fold_coverage = (
            len(candidate_inner_folds) / len(fold_inner_folds)
            if fold_inner_folds
            else 1.0
        )
        normalized_relevance = min(1.0, max(0.0, (top_mean + 1.0) / 2.0))
        aggregate = (
            0.75 * normalized_relevance
            + 0.125 * architecture_coverage
            + 0.125 * inner_fold_coverage
        )
        axes = _canonical_evidence_axes(candidate.get("evidence_axes")) or ["unclear"]
        rows.append(
            {
                "candidate_id": candidate_id,
                "feature_name": str(candidate.get("name") or ""),
                "natural_language_query": query_by_candidate[candidate_id],
                "evidence_axes": axes,
                "supporting_packet_count": len(support_ids),
                "retrieved_packet_count": len(retrieval_ids),
                "supporting_architectures": sorted(candidate_architectures),
                "supporting_inner_folds": sorted(candidate_inner_folds),
                "architecture_coverage": float(architecture_coverage),
                "inner_fold_coverage": float(inner_fold_coverage),
                "best_evidence_score": best_score,
                "top_evidence_mean_score": top_mean,
                "aggregate_support_score": float(aggregate),
                "ranked_packet_scores": [
                    {
                        "packet_id": packet_id,
                        "score": score_by_association[(candidate_id, packet_id)],
                        "directly_cited": packet_id in support_ids,
                        "ontology_evidence": packet_id in top_packet_ids,
                    }
                    for packet_id in ranked_packets
                ],
                "ontology_packet_ids": top_packet_ids,
                "ranked_hierarchy_packet_scores": [
                    {
                        "packet_id": packet_id,
                        "score": hierarchy_score_by_association[
                            (candidate_id, packet_id)
                        ],
                        "selected_for_descent": packet_id
                        in top_hierarchy_ids_by_candidate[candidate_id],
                    }
                    for packet_id in sorted(
                        ordered_hierarchy_packets,
                        key=lambda packet_id: (
                            -hierarchy_score_by_association[
                                (candidate_id, packet_id)
                            ],
                            packet_id,
                        ),
                    )
                ],
            }
        )

    axis_order = [
        "treatment",
        "outcome",
        "residual_effect",
        "matched_pair",
        "semantic",
        "unclear",
    ]
    axis_ranked_rows: dict[str, list[dict[str, Any]]] = {}
    axis_rank_by_candidate: dict[str, dict[str, int]] = defaultdict(dict)
    for axis in axis_order:
        ranked = sorted(
            [row for row in rows if axis in row["evidence_axes"]],
            key=_candidate_selection_sort_key,
        )
        axis_ranked_rows[axis] = ranked
        for rank, row in enumerate(ranked, start=1):
            axis_rank_by_candidate[row["candidate_id"]][axis] = rank

    # Round-robin over the axis top-N lists so a restrictive overall cap does
    # not fill entirely from the first or largest stratum.
    selected_ids: list[str] = []
    selected_set: set[str] = set()
    for rank_index in range(top_n_per_axis):
        for axis in axis_order:
            ranked = axis_ranked_rows[axis]
            if rank_index >= len(ranked):
                continue
            candidate_id = str(ranked[rank_index]["candidate_id"])
            if candidate_id in selected_set:
                continue
            selected_ids.append(candidate_id)
            selected_set.add(candidate_id)
            if len(selected_ids) >= max_candidates:
                break
        if len(selected_ids) >= max_candidates:
            break

    row_by_id = {str(row["candidate_id"]): row for row in rows}
    retained: list[dict[str, Any]] = []
    for candidate_id in selected_ids:
        candidate = dict(candidate_by_id[candidate_id])
        row = row_by_id[candidate_id]
        retained.append(
            {
                **candidate,
                "ontology_packet_ids": list(row["ontology_packet_ids"]),
                "candidate_selection": {
                    "schema_version": CANDIDATE_SELECTION_SCHEMA_VERSION,
                    "scoring_method": scoring_method,
                    "scoring_model": scoring_model,
                    "natural_language_query": row["natural_language_query"],
                    "aggregate_support_score": row["aggregate_support_score"],
                    "best_evidence_score": row["best_evidence_score"],
                    "top_evidence_mean_score": row["top_evidence_mean_score"],
                    "architecture_coverage": row["architecture_coverage"],
                    "inner_fold_coverage": row["inner_fold_coverage"],
                    "axis_ranks": axis_rank_by_candidate[candidate_id],
                    "score_by_packet": {
                        item["packet_id"]: item["score"]
                        for item in row["ranked_packet_scores"]
                    },
                    "hierarchy_packet_ids": list(
                        top_hierarchy_ids_by_candidate[candidate_id]
                    ),
                    "score_by_hierarchy_packet": {
                        item["packet_id"]: item["score"]
                        for item in row["ranked_hierarchy_packet_scores"]
                    },
                },
            }
        )

    shortlisted_without_hard_cap = {
        str(row["candidate_id"])
        for axis in axis_order
        for row in axis_ranked_rows[axis][:top_n_per_axis]
    }
    for row in rows:
        candidate_id = str(row["candidate_id"])
        row["axis_ranks"] = axis_rank_by_candidate[candidate_id]
        row["selected"] = candidate_id in selected_set
        row["selection_reason"] = (
            "selected_axis_top_n"
            if candidate_id in selected_set
            else (
                "removed_by_overall_fold_cap"
                if candidate_id in shortlisted_without_hard_cap
                else "outside_axis_top_n"
            )
        )
    dropped_origin_ids = list(
        dict.fromkeys(
            origin
            for candidate_id, candidate in candidate_by_id.items()
            if candidate_id not in selected_set
            for origin in (
                _string_values(candidate.get("origin_candidate_ids")) or [candidate_id]
            )
        )
    )
    audit = {
        "scoring_method": scoring_method,
        "scoring_model": scoring_model,
        "scoring_device": scoring_device,
        "top_n_per_evidence_axis": top_n_per_axis,
        "max_candidates_per_fold": max_candidates,
        "top_evidence_packets_per_candidate": top_evidence_packets,
        "hierarchy_top_communities_per_candidate": hierarchy_top_communities,
        "hierarchy_packets": len(ordered_hierarchy_packets),
        "document_chunk_overlap_tokens": document_chunk_overlap_tokens,
        "registry_candidates": len(candidates),
        "candidate_packet_associations_scored": len(association_keys),
        "candidate_hierarchy_associations_scored": len(
            hierarchy_association_keys
        ),
        "shortlisted_candidates_before_fold_cap": len(shortlisted_without_hard_cap),
        "hard_cap_applied": len(shortlisted_without_hard_cap) > max_candidates,
        "retained_candidates": len(retained),
        "dropped_registry_candidate_ids": [
            candidate_id for candidate_id in candidate_by_id if candidate_id not in selected_set
        ],
        "dropped_origin_candidate_ids": dropped_origin_ids,
        "score_aggregation": {
            "top_evidence_mean_count": top_evidence_packets,
            "relevance_weight": 0.75,
            "architecture_coverage_weight": 0.125,
            "inner_fold_coverage_weight": 0.125,
            "relevance_normalization": "clip((mean_maxsim + 1) / 2, 0, 1)",
        },
        "axis_rankings": {
            axis: [str(row["candidate_id"]) for row in ranked]
            for axis, ranked in axis_ranked_rows.items()
        },
        "candidate_rankings": sorted(rows, key=_candidate_selection_sort_key),
    }
    return retained, audit


def _build_and_select_candidate_registry(
    *,
    candidates: Sequence[Mapping[str, Any]],
    packet_by_id: Mapping[str, Mapping[str, Any]],
    hierarchy_packet_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    top_n_per_axis: int,
    max_candidates: int,
    registry_embedding_model: str,
    registry_embedding_device: str,
    registry_similarity_threshold: float,
    scoring_method: str,
    late_interaction_model: str,
    late_interaction_device: str,
    top_evidence_packets: int,
    document_chunk_overlap_tokens: int,
    hierarchy_top_communities: int = DEFAULT_CANDIDATE_SELECTION_HIERARCHY_TOP_COMMUNITIES,
    embedding_function: Callable[[Sequence[str], str, str], np.ndarray] | None = None,
    late_interaction_scoring_function: (
        Callable[[Sequence[str], Sequence[str], str, str], np.ndarray] | None
    ) = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry, registry_audit = _build_candidate_registry(
        candidates=candidates,
        embedding_model=registry_embedding_model,
        embedding_device=registry_embedding_device,
        similarity_threshold=registry_similarity_threshold,
        embedding_function=embedding_function,
    )
    retained, selection_audit = _select_candidates_from_registry(
        candidates=registry,
        packet_by_id=packet_by_id,
        hierarchy_packet_by_id=hierarchy_packet_by_id,
        top_n_per_axis=top_n_per_axis,
        max_candidates=max_candidates,
        scoring_method=scoring_method,
        late_interaction_model=late_interaction_model,
        late_interaction_device=late_interaction_device,
        dense_embedding_model=registry_embedding_model,
        dense_embedding_device=registry_embedding_device,
        top_evidence_packets=top_evidence_packets,
        document_chunk_overlap_tokens=document_chunk_overlap_tokens,
        hierarchy_top_communities=hierarchy_top_communities,
        embedding_function=embedding_function,
        late_interaction_scoring_function=late_interaction_scoring_function,
    )
    return retained, {
        "schema_version": CANDIDATE_SELECTION_SCHEMA_VERSION,
        "registry": registry_audit,
        "selection": selection_audit,
    }


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

    # The system message is independent of the evidence values. Compute the
    # exact JSON-list contribution without repeatedly rendering a growing body.
    template = _operationalization_prompt(
        feature_name=feature_name,
        supporting_evidence=[evidence[0]],
    )
    system_chars = len(template[0]["content"])
    empty_body_chars = len(
        json.dumps(
            {
                "candidate_feature_name": str(feature_name),
                "supporting_evidence": [],
            },
            sort_keys=True,
        )
    )
    fixed_chars_without_list = system_chars + empty_body_chars - 2
    list_chars = 2
    packed: list[str] = []
    for text_value in evidence:
        separator_chars = 2 if packed else 0
        candidate_list_chars = list_chars + separator_chars + len(json.dumps(text_value))
        if fixed_chars_without_list + candidate_list_chars > initial_prompt_budget:
            continue
        packed.append(text_value)
        list_chars = candidate_list_chars

    truncated_items = 0
    if not packed:
        available_encoded_chars = initial_prompt_budget - fixed_chars_without_list - 2
        first = evidence[0]
        best = ""
        low, high = 1, len(first)
        while low <= high:
            midpoint = (low + high) // 2
            candidate = first[:midpoint].rstrip()
            if midpoint < len(first):
                candidate = candidate.rstrip(" .") + "..."
            if candidate and len(json.dumps(candidate)) <= available_encoded_chars:
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


def _load_stage2_splits(
    *,
    provenance_path: Path | None,
    dataset_rows: int,
    outer_fold_ids: Sequence[int],
    inner_folds: int,
    seed: int,
) -> dict[int, dict[str, Any]]:
    """Read the ordinary Stage 1 split file, with a deterministic fallback."""

    rows: list[dict[str, Any]] = []
    if provenance_path is not None and Path(provenance_path).is_file():
        rows = [
            json.loads(line)
            for line in Path(provenance_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not rows:
        from sklearn.model_selection import KFold

        fold_ids = sorted(set(map(int, outer_fold_ids)))
        if len(fold_ids) < 2:
            raise FileNotFoundError(
                "full Stage 2 needs components/tfidf/split_provenance.jsonl when "
                "the handoff contains fewer than two outer folds"
            )
        splitter = KFold(n_splits=len(fold_ids), shuffle=True, random_state=int(seed))
        all_rows = np.arange(dataset_rows, dtype=int)
        for outer_fold, (fit, heldout) in zip(fold_ids, splitter.split(all_rows)):
            fit_ids = all_rows[fit]
            inner_count = min(max(2, int(inner_folds)), len(fit_ids))
            inner_rows: list[dict[str, Any]] = []
            if inner_count >= 2:
                inner_splitter = KFold(
                    n_splits=inner_count,
                    shuffle=True,
                    random_state=int(seed) + 51_000 + int(outer_fold),
                )
                for inner_fold, (inner_fit, inner_heldout) in enumerate(
                    inner_splitter.split(fit_ids), start=1
                ):
                    inner_rows.append(
                        {
                            "inner_fold": inner_fold,
                            "fit_row_ids": fit_ids[inner_fit].tolist(),
                            "heldout_row_ids": fit_ids[inner_heldout].tolist(),
                        }
                    )
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "fit_row_ids": fit_ids.tolist(),
                    "heldout_row_ids": all_rows[heldout].tolist(),
                    "inner_splits": inner_rows,
                }
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
        raise ValueError(f"split provenance is missing Stage 2 outer folds: {missing}")
    return by_fold


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
        )
        config.validate()
        self.config = config
        self.clinical_question = str(clinical_question)
        routed_completion = completion or _RoundRobinOpenAICompletion(endpoints)
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
                else extraction.runtime_endpoints or (extraction.endpoint,)
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
                routed_extraction = _RoundRobinOpenAICompletion(extraction_endpoints)
            self.extraction_completion = _ConcurrencyLimitedCompletion(
                routed_extraction,
                max_concurrency=extraction.workers,
            )
            if extraction_tokenizer is not None:
                self.extraction_tokenizer = extraction_tokenizer
            elif uses_default_extraction_transport:
                extraction_vllm = extraction.vllm
                self.extraction_tokenizer = _LazyStage2ExtractionTokenizer(
                    model=request_model,
                    cache_dir=(
                        str(extraction.vllm.download_dir)
                        if extraction.vllm is not None
                        else ""
                    ),
                    chat_template_kwargs=(
                        extraction.vllm.default_chat_template_kwargs
                        if extraction.vllm is not None
                        else None
                    ),
                )
        else:
            extraction_identity = None
            extraction_runtime_identity = None
        self.model_identity = {
            "schema_version": MODEL_IDENTITY_SCHEMA_VERSION,
            "primary": primary_identity,
            "extraction": extraction_identity,
            "extraction_runtime_continuation": extraction_runtime_identity,
            "selection_consolidation_runtime": consolidation_identity,
            "endpoint_urls_are_transport_only": True,
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
        else:
            # Adopt pre-manifest Stage 2 output only when its persisted resolved
            # model IDs agree. Endpoint URLs intentionally do not participate.
            config_path = output_dir / "config.json"
            if config_path.is_file():
                previous_config = json.loads(config_path.read_text(encoding="utf-8"))
                primary_model = str(previous_config.get("model") or "").strip()
                raw_extraction = previous_config.get("extraction_llm")
                extraction_model = (
                    str(raw_extraction.get("model") or "").strip()
                    if isinstance(raw_extraction, Mapping)
                    else ""
                )
                previous_scientific = {
                    "primary": (
                        {
                            "selected_model": primary_model,
                            "model_family": _stage2_model_family(primary_model),
                        }
                        if primary_model
                        else None
                    ),
                    "extraction": (
                        {
                            "selected_model": extraction_model,
                            "model_family": _stage2_model_family(extraction_model),
                        }
                        if extraction_model
                        else None
                    ),
                }
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
            "compiler": self.config.evidence_compiler,
            "compiler_version": EVIDENCE_COMPILER_VERSION,
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
            self.config.evidence_compiler,
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
        input_value = {
            "interpretation_schema": INTERPRETATION_SCHEMA_VERSION,
            "llm_identity": {
                "model": self.config.model,
            },
            "architecture": architecture,
            "packets": list(packets),
        }
        input_fingerprint = _value_fingerprint(input_value)
        packet_ids = [str(packet["packet_id"]) for packet in packets]
        packet_id_set = set(packet_ids)
        if len(packet_ids) != len(packet_id_set):
            raise ValueError("Stage 2 interpretation batch contains duplicate packet IDs")
        complete_path = output_dir / "complete.json"
        result_path = output_dir / "result.json"
        input_path = output_dir / "input.json"
        if complete_path.is_file() and result_path.is_file() and input_path.is_file():
            previous = json.loads(input_path.read_text(encoding="utf-8"))
            completion_state = json.loads(complete_path.read_text(encoding="utf-8"))
            previous_fingerprint = previous.get("input_fingerprint")
            if previous_fingerprint is None:
                previous_fingerprint = _value_fingerprint(
                    {
                        "interpretation_schema": previous.get("interpretation_schema"),
                        "architecture": previous.get("architecture"),
                        "packets": previous.get("packets") or [],
                    }
                )
            if (
                previous_fingerprint == input_fingerprint
                and completion_state.get("input_fingerprint") == input_fingerprint
            ):
                try:
                    cached_result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached_result = None
                if _cached_interpretation_matches_packets(
                    cached_result,
                    packet_ids=packet_id_set,
                ) and (
                    cached_result.get("rejected_packet_audit", {}).get("schema_version")
                    == INTERPRETATION_AUDIT_SCHEMA_VERSION
                ):
                    LOGGER.info("skip completed Stage 2 interpretation: %s", output_dir)
                    return cached_result
            LOGGER.info("rerun stale or inconsistent Stage 2 interpretation: %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(input_path, {**input_value, "input_fingerprint": input_fingerprint})
        packet_evidence_axes: dict[str, list[str]] = {}
        for packet in packets:
            packet_id = str(packet["packet_id"])
            packet_evidence_axes[packet_id] = _packet_observable_axes(packet)
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
            validate=lambda value: _validate_interpretation(
                value,
                packet_ids=packet_ids,
                packet_evidence_axes=packet_evidence_axes,
            ),
        )

        packet_by_id = {str(packet["packet_id"]): packet for packet in packets}
        rejected_packets = [
            packet_by_id[packet_id]
            for packet_id, disposition in initial["packet_dispositions"].items()
            if disposition.get("status") == "reviewed_no_specific_concept"
        ]
        audit_results: list[Mapping[str, Any]] = []
        if rejected_packets:
            audit_batches = _partition_rejected_packet_audit(
                rejected_packets,
                architecture=architecture,
                max_prompt_chars=self.config.max_prompt_chars,
            )
            LOGGER.info(
                "Stage 2 rejected-packet audit architecture=%s rejected_packets=%s " "batches=%s",
                architecture,
                len(rejected_packets),
                len(audit_batches),
            )
            for batch_index, batch in enumerate(audit_batches, start=1):
                batch_ids = [str(packet["packet_id"]) for packet in batch]
                audit_results.append(
                    _checkpointed_request_json(
                        output_dir=(
                            output_dir / "rejected_packet_audit" / f"batch_{batch_index:03d}"
                        ),
                        input_value={
                            "phase": "rejected_packet_audit",
                            "audit_schema": INTERPRETATION_AUDIT_SCHEMA_VERSION,
                            "interpretation_schema": INTERPRETATION_SCHEMA_VERSION,
                            "architecture": architecture,
                            "packets": list(batch),
                        },
                        messages=_rejected_packet_audit_prompt(
                            architecture=architecture,
                            packets=batch,
                        ),
                        config=self.config,
                        completion=self.completion,
                        validate=lambda value, batch_ids=batch_ids: _validate_interpretation(
                            value,
                            packet_ids=batch_ids,
                            packet_evidence_axes=packet_evidence_axes,
                        ),
                    )
                )

        result = _merge_interpretation_audit(
            packet_ids=packet_id_set,
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
                "initially_rejected_packets": len(rejected_packets),
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
        """Losslessly merge aliases across shifted and seeded-shuffled batches."""

        current = [dict(group) for group in sorted(groups, key=_candidate_group_sort_key)]
        if not current or all(_configured_feature_definitions(group) for group in current):
            return current
        prompt_limit = int(self.config.consolidation_max_prompt_chars)
        request_config = replace(self.config, max_prompt_chars=prompt_limit)
        no_change_partitions: set[tuple[tuple[str, ...], ...]] = set()
        round_summaries: list[dict[str, Any]] = []
        stopped_reason = "maximum_rounds_reached"
        process_input = {
            "phase": "iterative_candidate_pool_merge_only_consolidation",
            "consolidation_schema": CONSOLIDATION_SCHEMA_VERSION,
            "global_candidate_pool_schema": GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION,
            "outer_fold": int(outer_fold),
            "consolidation_batch_size": int(self.config.consolidation_batch_size),
            "consolidation_alphabetical_rounds": int(self.config.consolidation_alphabetical_rounds),
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
            if not current:
                stopped_reason = "candidate_pool_empty"
                break
            if all(_configured_feature_definitions(group) for group in current):
                stopped_reason = "only_explicit_features_remain"
                break
            ordering, boundary_offset, shuffle_round, batches = _candidate_consolidation_batches(
                current,
                batch_size=int(self.config.consolidation_batch_size),
                round_number=round_number,
                alphabetical_rounds=int(self.config.consolidation_alphabetical_rounds),
                seed=int(seed) + 1_000_003 * int(outer_fold),
            )
            partition_signature = tuple(
                tuple(str(group["name"]) for group in batch) for batch in batches
            )
            if partition_signature in no_change_partitions:
                stopped_reason = "repeated_no_change_partition"
                LOGGER.info(
                    "Stage 2 iterative consolidation converged before round=%s; "
                    "the unchanged candidate pool already used this partition",
                    round_number,
                )
                break

            round_dir = output_dir / f"round_{round_number:03d}" if output_dir is not None else None
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
                    batch_ordering=ordering,
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
                            "global_candidate_pool_schema": (GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION),
                            "outer_fold": int(outer_fold),
                            "round": round_number,
                            "batch": batch_number,
                            "ordering": ordering,
                            "boundary_offset": boundary_offset,
                            "shuffle_round": shuffle_round,
                            "consolidation_batch_size": int(self.config.consolidation_batch_size),
                            "consolidation_alphabetical_rounds": int(
                                self.config.consolidation_alphabetical_rounds
                            ),
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
                "ordering": ordering,
                "boundary_offset": boundary_offset,
                "shuffle_round": shuffle_round,
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
            }
            round_summaries.append(round_summary)
            if round_dir is not None:
                _write_json(
                    round_dir / "complete.json",
                    {
                        "status": "complete",
                        "completed_at": _now(),
                        "global_candidate_pool_schema": (GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION),
                        **round_summary,
                    },
                )
            LOGGER.info(
                "Stage 2 iterative consolidation round=%s offset=%s input_groups=%s "
                "ordering=%s batches=%s merge_directives=%s "
                "cross_batch_exact_merges=%s output_groups=%s changed=%s",
                round_number,
                boundary_offset,
                len(current),
                ordering,
                len(batches),
                directive_count,
                cross_batch_exact_merges,
                len(next_groups),
                changed,
            )
            current = next_groups
            if not current:
                stopped_reason = "candidate_pool_empty"
                break
            if all(_configured_feature_definitions(group) for group in current):
                stopped_reason = "only_explicit_features_remain"
                break
            if changed:
                no_change_partitions.clear()
            elif not validation_fallbacks:
                no_change_partitions.add(partition_signature)
                if len(batches) == 1:
                    stopped_reason = "single_batch_no_change"
                    break
            elif len(batches) == 1:
                stopped_reason = "single_batch_validation_fallback"
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
                    "global_candidate_pool_schema": GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION,
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
                output_dir / "candidate_pool_consolidation" if output_dir is not None else None
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
        infrastructure_audits = (
            infrastructure_failure_audit_paths(output_dir)
            if complete_path.is_file()
            else ()
        )
        if infrastructure_audits and complete_path.is_file():
            LOGGER.warning(
                "rerun completed Stage 2 outer fold=%s because %s legacy "
                "infrastructure failure(s) were checkpointed as scientific missingness",
                outer_fold,
                len(infrastructure_audits),
            )
            os.replace(
                complete_path,
                complete_path.with_name("superseded_infrastructure_complete.json"),
            )
        elif complete_path.is_file() and dataset is not None:
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
        candidates: list[dict[str, Any]] | None = None
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
        elif candidates is None:
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
            candidates = []
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

        if candidates is not None:
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
                    "global_candidate_pool_schema": GLOBAL_CANDIDATE_POOL_SCHEMA_VERSION,
                    "candidate_selection": "none_exhaustive_discovery",
                    "consolidation_batch_size": int(self.config.consolidation_batch_size),
                    "consolidation_alphabetical_rounds": int(
                        self.config.consolidation_alphabetical_rounds
                    ),
                    "consolidation_max_rounds": int(self.config.consolidation_max_rounds),
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
                "value extraction is isolated from the primary review model"
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
                inner_folds=inner_folds,
                seed=seed,
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
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=fold_workers,
                thread_name_prefix="stage2-outer-fold",
            ) as outer_executor:
                futures = {
                    outer_executor.submit(
                        self._run_outer_fold,
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
            "evidence_compiler": self.config.evidence_compiler,
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
        return PlainHandoffStage2(
            config=runtime_config,
            clinical_question=clinical_question,
            completion=runtime_primary_completion,
            extraction_completion=runtime_extraction_completion,
            extraction_tokenizer=extraction_tokenizer,
        ).run(
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
    "Stage2ExtractionLLMConfig",
    "Stage2ExplicitFeature",
    "packetize_handoff",
    "plain_stage2_config_from_mapping",
    "run_plain_handoff_stage2",
]
