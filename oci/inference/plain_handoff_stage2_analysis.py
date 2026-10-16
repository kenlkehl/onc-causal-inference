"""Patient-level extraction, empirical review, and causal estimation for Stage 2.

The implementation is deliberately file-oriented.  Every expensive extraction
batch and every review round writes an ordinary result followed by
``complete.json``.  A repeated invocation reads those files and continues at
the first unfinished operation.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from joblib.externals.loky import ProcessPoolExecutor as LokyProcessPoolExecutor

from .nuisance_diagnostics import (
    calibration_diagnostics,
    overlap_diagnostics,
    propensity_eligibility,
    validate_propensity_bounds,
)
from . import stage2_request_audit as request_audit
from . import stage2_clinical_prompts as clinical_prompts
from . import stage2_modeling_pool as modeling

from ..models.causal_forest_head import CausalForestHead
from ..models.super_learner_nuisance import SuperLearnerClassifier, SuperLearnerRegressor
from ..models.elastic_net_nuisance import (
    ElasticNetLogisticClassifier,
    ElasticNetRegressor,
)
from .stage2_elastic_net_selection import TEMPORAL_SCOPE
from . import stage2_extraction_failures as extraction_failures
from .stage2_extraction_failures import extraction_failure_column
from . import stage2_documented_unknown as documented_unknown
from .stage2_documented_unknown import (
    DOCUMENTED_UNKNOWN,
    documented_unknown_mask,
    is_documented_unknown,
    is_unknown_like,
)
from .stage2_multi_model_selection import select_stage2_features_multi_model
from .stage2_sequential_consolidation import (
    SELECTION_SCHEMA_VERSION,
    consolidate_stage2_candidates,
    latent_states_for_selected,
    materialize_selected_latents,
    measurement_definitions_for_selected,
)

LOGGER = logging.getLogger(__name__)

# Large selectors spend most of their time in Python-level proximal-gradient
# loops. Outer folds are otherwise orchestrated by threads, which makes those
# loops contend on the GIL. Isolating one large selector per outer fold keeps
# the scientific computation unchanged while allowing the independent folds
# to use separate cores. Small selectors stay in-process to avoid spawn cost.
STATISTICAL_SELECTION_PROCESS_ISOLATION_MIN_CANDIDATES = 64

EXTRACTION_CHECKPOINT_SCHEMA_VERSION = 'stage2_single_patient_extraction_v7_documented_unknown_20261016'
EXTRACTION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION = 'stage2_single_patient_feature_batch_extraction_v6_documented_unknown_20261016'
PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION = 'stage2_mode_observations_v7_documented_unknown'
PAGE_OBSERVATION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION = 'stage2_mode_observation_feature_batch_v3_documented_unknown'
PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION = 'stage2_mode_reconciliation_v8_documented_unknown'
EXTRACTION_ROUTING_SCHEMA_VERSION = 'stage2_extraction_mode_split_v1'
REVIEW_CHECKPOINT_SCHEMA_VERSION = 'stage2_aggregate_ontology_supervisor_v3_documented_unknown_20261016'
REVIEW_CONVERGENCE_SCHEMA_VERSION = "stage2_ontology_supervisor_convergence_v1"
ESTIMATION_CHECKPOINT_SCHEMA_VERSION = "stage2_outer_estimation_v12_own_fold_heldout_calibration"
STAGE2_ROLE_SELECTION_SCHEMA_VERSION = SELECTION_SCHEMA_VERSION
PRESELECTION_SNAPSHOT_SCHEMA_VERSION = "stage2_frozen_preselection_snapshot_v1"
HELDOUT_MEASUREMENT_CACHE_SCHEMA_VERSION = (
    "stage2_frozen_heldout_measurement_cache_v2_documented_unknown"
)
HELDOUT_MEASUREMENT_REUSE_SCHEMA_VERSION = (
    "stage2_heldout_measurement_reuse_v1"
)
STAGE2_RESELECTION_MIGRATION_SCHEMA_VERSION = "stage2_reselection_migration_v1"
EXTRACTION_ISSUE_SCHEMA_VERSION = "stage2_extraction_issues_v1"
PENDING_CATEGORY_ONTOLOGY_SCHEMA_VERSION = 'stage2_pending_category_ontology_v2_documented_unknown_20261016'
ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION = 'stage2_training_failure_ontology_refinement_v4_documented_unknown_20261016'
INCREMENTAL_REFINEMENT_EXTRACTION_SCHEMA_VERSION = (
    "stage2_incremental_refinement_extraction_v2_documented_unknown"
)
HARMONIZATION_CHECKPOINT_SCHEMA_VERSION = 'stage2_mixed_value_harmonization_v2_documented_unknown_20261016'
HARMONIZATION_FALLBACK_SCHEMA_VERSION = "stage2_mixed_value_harmonization_fallback_v1"
DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE = 10
DEFAULT_EXTRACTION_CHUNK_SIZE_TOKENS = 50_000
DEFAULT_EXTRACTION_CONTEXT_WINDOW_TOKENS = 131_072
DEFAULT_EXTRACTION_MAX_TOKENS = 75_000
DEFAULT_EXTRACTION_CONTEXT_MARGIN_TOKENS = 1_024
SERIAL_EXTRACTION_CHUNK_CHECKPOINT_SCHEMA_VERSION = 'stage2_serial_patient_feature_chunk_v2_documented_unknown_20261016'
SERIAL_EXTRACTION_MANIFEST_SCHEMA_VERSION = 'stage2_serial_patient_feature_extraction_v2_documented_unknown_20261016'
MAX_SERIAL_FEATURE_STATE_CHARS = 2_048


class RequestJSON(Protocol):
    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        validate: Callable[[Mapping[str, Any]], dict[str, Any]],
        *,
        request_kind: str = "interpretation",
        initial_reasoning_effort: str | None = None,
    ) -> dict[str, Any]: ...


class Stage2InfrastructureError(RuntimeError):
    """A Stage 2 request failed without producing a scientific response."""


class Stage2RequestExhaustedError(Stage2InfrastructureError):
    """A request exhausted its bounded transport or deadline budget."""


class Stage2ResponseValidationError(ValueError):
    """A completed model response remained semantically invalid after repairs."""


EXTRACTION_FIELD_REPAIR_LIMIT = 3


class _ExtractionFieldError(ValueError):
    """A response error attributable to specific supplied measurement fields."""

    def __init__(self, message: str, *, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(sorted(set(feature_names)))
        super().__init__(message)


class _ExtractionCancelledError(RuntimeError):
    """Cooperatively stop sibling extraction tasks after one task fails."""


CONFLICT_RESOLUTION_STRATEGIES = frozenset(
    {
        "latest",
        "earliest",
        "maximum",
        "minimum",
        "mode",
        "any_positive",
        "single_or_null",
    }
)

CONTINUOUS_MODELING_STRATEGIES = frozenset(
    {
        "continuous",
        "categorical",
        "continuous_with_categorical_fallback",
    }
)


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


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _configured_serial_extraction(config: Any) -> dict[str, Any]:
    """Read the token-aware full-record extraction settings from a Stage 2 config."""

    return {
        "chunk_size_tokens": int(config.extraction_chunk_size_tokens),
        "context_window_tokens": int(config.extraction_context_window_tokens),
        # Reserve room for a reasoning-enabled repair of the same lossless chunk.
        "max_output_tokens": max(
            int(config.extraction_max_tokens),
            int(config.extraction_reasoning_max_tokens or 0),
        ),
        "context_margin_tokens": int(config.extraction_context_margin_tokens),
        "deferred_retry_passes": int(config.extraction_deferred_retry_passes),
    }


def _value_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint ordered modeling values for checkpoint compatibility."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": list(map(str, frame.columns)),
                "dtypes": [str(dtype) for dtype in frame.dtypes],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def frozen_preselection_review_policy(config: Any) -> dict[str, Any]:
    """Return the ontology-review policy that produced a reusable fit matrix."""

    return {
        "review_schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
        "review_convergence_schema_version": REVIEW_CONVERGENCE_SCHEMA_VERSION,
        "ontology_refinement_schema_version": (
            ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION
        ),
        "harmonization_schema_version": HARMONIZATION_CHECKPOINT_SCHEMA_VERSION,
        "max_review_rounds": int(config.max_review_rounds),
        "ontology_refinement_min_failure_patients": int(
            config.ontology_refinement_min_failure_patients
        ),
        "max_ontology_refinement_rounds": int(config.max_ontology_refinement_rounds),
    }


def _load_frozen_preselection_snapshot(
    *,
    output_dir: Path,
    dataset: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    fit_ids: Sequence[int],
    heldout_ids: Sequence[int],
    inner_splits: Sequence[Mapping[str, Any]],
    unit_id_column: str,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    stage1_packets: Sequence[Mapping[str, Any]],
    config: Any,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any] | None,
] | None:
    """Load a migration-validated post-ontology matrix without re-extraction."""

    snapshot_dir = Path(output_dir) / "preselection"
    input_path = snapshot_dir / "input.json"
    complete_path = snapshot_dir / "complete.json"
    if not input_path.is_file() and not complete_path.is_file():
        return None
    if not input_path.is_file() or not complete_path.is_file():
        raise RuntimeError(
            f"incomplete frozen preselection snapshot under {snapshot_dir}; restore "
            "the archived run or rerun --stage2-reselect"
        )
    try:
        snapshot = json.loads(input_path.read_text(encoding="utf-8"))
        completion = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid frozen preselection snapshot under {snapshot_dir}"
        ) from exc
    if not isinstance(snapshot, Mapping) or not isinstance(completion, Mapping):
        raise RuntimeError(f"invalid frozen preselection snapshot under {snapshot_dir}")
    if (
        snapshot.get("schema_version") != PRESELECTION_SNAPSHOT_SCHEMA_VERSION
        or completion.get("schema_version") != PRESELECTION_SNAPSHOT_SCHEMA_VERSION
        or completion.get("status") != "complete"
    ):
        raise RuntimeError(
            f"incompatible frozen preselection snapshot under {snapshot_dir}"
        )
    snapshot_value = {
        str(key): copy.deepcopy(value)
        for key, value in snapshot.items()
        if key != "input_fingerprint"
    }
    input_fingerprint = _value_fingerprint(snapshot_value)
    if (
        snapshot.get("input_fingerprint") != input_fingerprint
        or completion.get("input_fingerprint") != input_fingerprint
    ):
        raise RuntimeError(
            f"frozen preselection snapshot fingerprint mismatch under {snapshot_dir}"
        )

    expected_checks = {
        "source_feature_definitions_fingerprint": _value_fingerprint(
            [dict(feature) for feature in definitions]
        ),
        "fit_row_ids_fingerprint": _value_fingerprint([int(value) for value in fit_ids]),
        "inner_splits_fingerprint": _value_fingerprint(list(inner_splits)),
        "treatment_outcome_fingerprint": _frame_fingerprint(
            dataset.iloc[list(fit_ids)][[treatment_column, outcome_column]].reset_index(
                drop=True
            )
        ),
        "fit_source_text_fingerprint": _frame_fingerprint(
            dataset.iloc[list(fit_ids)][[unit_id_column, text_column]].reset_index(
                drop=True
            )
        ),
        "stage1_packets_fingerprint": _value_fingerprint(list(stage1_packets)),
        "review_policy_fingerprint": _value_fingerprint(
            frozen_preselection_review_policy(config)
        ),
    }
    mismatches = sorted(
        key for key, expected in expected_checks.items() if snapshot.get(key) != expected
    )
    if str(snapshot.get("outcome_type") or "") != str(outcome_type):
        mismatches.append("outcome_type")
    if mismatches:
        raise RuntimeError(
            "frozen preselection snapshot does not match the current run inputs: "
            + ", ".join(mismatches)
        )

    matrix_relative = str(snapshot.get("matrix_path") or "").strip()
    if not matrix_relative:
        raise RuntimeError("frozen preselection snapshot has no matrix_path")
    matrix_path = (Path(output_dir) / matrix_relative).resolve()
    resolved_output = Path(output_dir).resolve()
    if matrix_path != resolved_output and resolved_output not in matrix_path.parents:
        raise RuntimeError("frozen preselection matrix_path escapes its outer-fold directory")
    if not matrix_path.is_file():
        raise RuntimeError(f"frozen preselection matrix is missing: {matrix_path}")
    matrix = pd.read_csv(matrix_path)
    if _frame_fingerprint(matrix) != snapshot.get("matrix_snapshot_fingerprint"):
        raise RuntimeError(f"frozen preselection matrix changed after migration: {matrix_path}")

    raw_snapshot_definitions = snapshot.get("definitions")
    if not isinstance(raw_snapshot_definitions, list) or not all(
        isinstance(feature, Mapping) for feature in raw_snapshot_definitions
    ):
        raise RuntimeError("frozen preselection snapshot definitions are invalid")
    snapshot_definitions = [dict(feature) for feature in raw_snapshot_definitions]
    names = [str(feature.get("name") or "") for feature in snapshot_definitions]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise RuntimeError("frozen preselection snapshot contains invalid feature names")
    expected_columns = ["_oci_row_id", *names]
    if list(matrix.columns) != expected_columns:
        raise RuntimeError(
            "frozen preselection matrix columns do not match its definitions"
        )
    numeric_ids = pd.to_numeric(matrix["_oci_row_id"], errors="coerce")
    if numeric_ids.isna().any() or not np.allclose(
        numeric_ids.to_numpy(dtype=float),
        np.rint(numeric_ids.to_numpy(dtype=float)),
    ):
        raise RuntimeError("frozen preselection matrix contains invalid row identifiers")
    matrix_ids = numeric_ids.astype(int).tolist()
    if matrix_ids != [int(value) for value in fit_ids] or len(set(matrix_ids)) != len(
        matrix_ids
    ):
        raise RuntimeError(
            "frozen preselection matrix rows do not match the current outer-training rows"
        )
    matrix["_oci_row_id"] = numeric_ids.astype(int)
    metadata = snapshot.get("review_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("frozen preselection snapshot review_metadata is invalid")
    raw_heldout_cache = snapshot.get("heldout_measurement_cache")
    heldout_cache: dict[str, Any] | None = None
    if raw_heldout_cache is not None:
        if not isinstance(raw_heldout_cache, Mapping):
            raise RuntimeError(
                "frozen preselection held-out measurement cache is invalid"
            )
        heldout_cache = copy.deepcopy(dict(raw_heldout_cache))
        if (
            heldout_cache.get("schema_version")
            != HELDOUT_MEASUREMENT_CACHE_SCHEMA_VERSION
        ):
            raise RuntimeError(
                "frozen preselection held-out measurement cache has an "
                "incompatible schema"
            )
        raw_cache_row_ids = heldout_cache.get("heldout_row_ids")
        if not isinstance(raw_cache_row_ids, list):
            raise RuntimeError(
                "frozen preselection held-out measurement cache has no row IDs"
            )
        try:
            cache_row_ids = [int(value) for value in raw_cache_row_ids]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "frozen preselection held-out measurement cache row IDs are invalid"
            ) from exc
        expected_heldout_ids = [int(value) for value in heldout_ids]
        if (
            cache_row_ids != expected_heldout_ids
            or len(cache_row_ids) != len(set(cache_row_ids))
            or heldout_cache.get("heldout_row_ids_fingerprint")
            != _value_fingerprint(cache_row_ids)
        ):
            raise RuntimeError(
                "frozen preselection held-out measurement cache rows do not match "
                "the current outer-heldout rows"
            )
        current_heldout_source_fingerprint = _frame_fingerprint(
            dataset.iloc[cache_row_ids][[unit_id_column, text_column]].reset_index(
                drop=True
            )
        )
        if (
            heldout_cache.get("heldout_source_text_fingerprint")
            != current_heldout_source_fingerprint
        ):
            raise RuntimeError(
                "frozen preselection held-out source text changed after migration"
            )
        extraction_llm = getattr(config, "extraction_llm", None)
        current_extraction_model = str(getattr(extraction_llm, "model", "") or "")
        if str(heldout_cache.get("extraction_model") or "") != current_extraction_model:
            raise RuntimeError(
                "frozen preselection held-out cache extraction model does not match "
                "the current extraction model"
            )
        cached_definitions = heldout_cache.get("measurement_definitions")
        if not isinstance(cached_definitions, list) or not all(
            isinstance(value, Mapping) for value in cached_definitions
        ):
            raise RuntimeError(
                "frozen preselection held-out cache definitions are invalid"
            )
        cache_names = [str(value.get("name") or "") for value in cached_definitions]
        cache_ids = [
            str(value.get("feature_id") or "") for value in cached_definitions
        ]
        if (
            any(not value for value in cache_names)
            or any(not value for value in cache_ids)
            or len(cache_names) != len(set(cache_names))
            or len(cache_ids) != len(set(cache_ids))
        ):
            raise RuntimeError(
                "frozen preselection held-out cache feature identities are invalid"
            )
    return matrix, snapshot_definitions, dict(metadata), heldout_cache


def _clean_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    raise ValueError("extracted feature values must be scalar JSON values or null")


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    key = re.sub(r"[^a-z0-9]+", "", value.lower())
    return key in {"", "unknown", "notdocumented", "missing", "nan", "na", "null", "none"}


def _normalized_category_values(*, value_type: Any, values: Sequence[Any]) -> list[str]:
    normalized_type = str(value_type or "ambiguous").strip().lower()
    normalized = [str(item).strip() for item in values if str(item).strip()]
    if normalized_type in {"binary", "categorical", "ordinal"} and len(normalized) == 1:
        separated = [
            part.strip() for part in re.split(r"\s*[,;|]\s*", normalized[0]) if part.strip()
        ]
        if len(separated) > 1:
            normalized = separated
    if normalized_type == "binary" and len(normalized) == 1:
        slash_separated = [part.strip() for part in normalized[0].split("/") if part.strip()]
        if len(slash_separated) == 2:
            normalized = slash_separated

    expand_integer_ranges = normalized_type in {"binary", "ordinal"} or (
        normalized_type == "categorical" and len(normalized) == 1
    )
    if expand_integer_ranges:
        expanded: list[str] = []
        for value in normalized:
            match = re.fullmatch(
                r"([+-]?\d+)\s*(?:-|–|—|to)\s*([+-]?\d+)",
                value,
                flags=re.IGNORECASE,
            )
            if match is None:
                expanded.append(value)
                continue
            start, stop = (int(token) for token in match.groups())
            if stop < start or stop - start > 100:
                expanded.append(value)
                continue
            expanded.extend(str(category) for category in range(start, stop + 1))
        normalized = expanded
    return list(dict.fromkeys(normalized))


_CATEGORY_SCHEMA_LABEL = re.compile(
    r"^(?:binary|boolean|categorical|category|categories|ordinal|class|classes|"
    r"level|levels|allowed\s+(?:category|categories|value|values))$",
    flags=re.IGNORECASE,
)


def _validated_closed_category_values(
    *,
    value_type: Any,
    values: Sequence[Any],
    source: str,
    keep_unknown_like: bool = False,
) -> list[str]:
    """Normalize and validate one extraction-ready closed ontology.

    This is deliberately domain-agnostic.  It validates the shape of the
    ontology, not which clinical labels should appear in it.  With
    ``keep_unknown_like`` (investigator-configured features, whose definitions
    are locked) unknown-like categories are kept as declared; the reserved
    documented_unknown value itself is never a valid category.
    """

    normalized_type = str(value_type or "ambiguous").strip().lower()
    if normalized_type not in {"binary", "categorical", "ordinal"}:
        raise ValueError(f"{source} is not a closed-ontology value type")
    categories = _normalized_category_values(
        value_type=normalized_type,
        values=values,
    )
    if keep_unknown_like:
        reserved = [category for category in categories if is_documented_unknown(category)]
        if reserved:
            raise ValueError(
                f"{source} categories_or_unit cannot declare the reserved value "
                f"{DOCUMENTED_UNKNOWN}; it is available to every variable without "
                "being listed"
            )
        unknown_like: list[str] = []
    else:
        # "Unknown"-like labels are the reserved documented_unknown state, not
        # categories, unless removing them would leave an invalid ontology.
        categories, unknown_like = documented_unknown.split_unknown_like_categories(
            normalized_type, categories
        )
    if unknown_like:
        LOGGER.info(
            "%s: unknown-like categories %s are the reserved %s value",
            source, unknown_like, DOCUMENTED_UNKNOWN,
        )
    identity_keys = [
        re.sub(r"[\W_]+", " ", category, flags=re.UNICODE).strip().casefold()
        for category in categories
    ]
    if len(identity_keys) != len(set(identity_keys)):
        duplicate_keys = sorted(
            {
                identity_key
                for identity_key in identity_keys
                if identity_keys.count(identity_key) > 1
            }
        )
        duplicate_values = [
            category
            for category, identity_key in zip(categories, identity_keys)
            if identity_key in duplicate_keys
        ]
        raise ValueError(
            f"{source} categories_or_unit must contain categories that are distinct "
            "after case and spacing normalization; return each category once and remove "
            f"these normalization-equivalent values: {duplicate_values[:12]!r}"
        )
    if normalized_type == "binary" and len(categories) != 2:
        raise ValueError(
            f"{source} binary categories_or_unit must contain exactly two distinct "
            "scalar categories as separate array items; "
            f"received {len(categories)}: {categories!r}"
        )
    if normalized_type in {"categorical", "ordinal"} and len(categories) < 2:
        raise ValueError(
            f"{source} {normalized_type} categories_or_unit must contain at least two "
            "distinct scalar categories as separate array items; "
            f"received {len(categories)}: {categories!r}"
        )
    placeholders = [
        category for category in categories if _CATEGORY_SCHEMA_LABEL.fullmatch(category.strip())
    ]
    if placeholders:
        raise ValueError(
            f"{source} categories_or_unit contains schema label(s) rather than "
            f"extractable values: {placeholders!r}"
        )
    return categories


def _declared_categories(definition: Mapping[str, Any]) -> list[str]:
    """Declared categories; the reserved documented_unknown value is never one."""
    return [
        category
        for category in _normalized_category_values(
            value_type=definition.get("value_type"),
            values=definition.get("categories_or_unit") or [],
        )
        if not is_documented_unknown(category)
    ]


def _documented_unknown_answer(value: Any, declared: Sequence[str]) -> bool:
    """The reserved value, or a whole unknown-like phrase that is not a declared category.

    A bare "unknown" stays a missing-value placeholder (null): extraction models
    commonly write it for a variable the record does not mention, and they are
    instructed to answer documented_unknown for an explicit statement.
    """
    if is_documented_unknown(value):
        return True
    return (
        isinstance(value, str)
        and is_unknown_like(value)
        and not _is_missing_scalar(value)
        and _canonical_category(value, declared) is None
    )


def _prompt_feature_definitions(
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project frozen definitions to fields needed for patient measurement.

    Normal single-patient extraction returns one scalar directly, so it must
    receive the same explicit longitudinal conflict policy that the richer
    page-observation path later applies deterministically.
    """

    output: list[dict[str, Any]] = []
    for definition in definitions:
        row = {
            key: definition.get(key)
            for key in (
                "name",
                "description",
                "value_type",
                "categories_or_unit",
                "measurement_definition",
                "missing_value_rule",
            )
        }
        if str(row.get("value_type")) in {"binary", "categorical", "ordinal"}:
            row["categories_or_unit"] = _declared_categories(row)
        if str(row.get("value_type")) == "continuous":
            row["accepted_representations"] = (
                "one JSON number, or one documented categorical/threshold string when "
                "the numeric measurement is unavailable"
            )
        row["conflict_resolution"] = _resolved_conflict_resolution(definition)
        if definition.get("clinical_label"):
            row["clinical_label"] = definition["clinical_label"]
        output.append(row)
    return output


def frozen_measurement_definition_identity(
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    """Identify exactly the feature measurement prompt represented by a cache column."""

    prompt_definition = _prompt_feature_definitions([definition])[0]
    return {
        "feature_id": str(definition.get("feature_id") or ""),
        "name": str(definition.get("name") or ""),
        "prompt_definition": prompt_definition,
        "prompt_definition_fingerprint": _value_fingerprint(prompt_definition),
    }


def _likely_positive_category(categories: Sequence[str]) -> str | None:
    """Return the affirmative member of a binary ontology when identifiable."""

    if len(categories) != 2:
        return None
    negative = re.compile(
        r"^(?:0|false|no|none|never|absent|negative|not\s+(?:present|documented|detected)|"
        r"undocumented|unknown)$",
        flags=re.IGNORECASE,
    )
    negative_matches = [category for category in categories if negative.fullmatch(category.strip())]
    if len(negative_matches) != 1:
        return None
    return next(category for category in categories if category != negative_matches[0])


def _resolved_conflict_resolution(definition: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated, explicit rule for reducing longitudinal observations.

    New ontologies persist ``conflict_resolution``.  Historical definitions are
    interpreted conservatively from their measurement text so completed
    interpretation checkpoints remain reusable.
    """

    raw = definition.get("conflict_resolution")
    if isinstance(raw, str):
        raw = {"strategy": raw}
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("conflict_resolution must be an object or strategy string")

    value_type = str(definition.get("value_type") or "ambiguous").strip().lower()
    categories = _declared_categories(definition)
    measurement_text = " ".join(
        str(definition.get(key) or "")
        for key in ("measurement_definition", "description", "missing_value_rule")
    ).lower()

    strategy_source = "explicit_ontology"
    strategy = str((raw or {}).get("strategy") or "").strip().lower().replace("-", "_")
    strategy_aliases = {
        "most_recent": "latest",
        "last": "latest",
        "first": "earliest",
        "max": "maximum",
        "min": "minimum",
        "majority": "mode",
        "present_if_ever": "any_positive",
        "null_on_conflict": "single_or_null",
    }
    strategy = strategy_aliases.get(strategy, strategy)
    if not strategy:
        strategy_source = "inferred_measurement_definition"
        if re.search(r"\b(?:maximum|maximal|highest|peak)\b", measurement_text):
            strategy = "maximum"
        elif re.search(r"\b(?:minimum|minimal|lowest)\b", measurement_text):
            strategy = "minimum"
        elif re.search(r"\b(?:earliest|first)\b", measurement_text):
            strategy = "earliest"
        elif re.search(r"\b(?:mode|most frequent|majority)\b", measurement_text):
            strategy = "mode"
        elif (
            value_type == "binary"
            and re.search(r"\b(?:ever|history of|any occurrence|presence or absence)\b", measurement_text)
            and _likely_positive_category(categories) is not None
        ):
            strategy = "any_positive"
        elif re.search(r"\b(?:single unambiguous|conflict[^.]{0,40}(?:null|missing))\b", measurement_text):
            strategy = "single_or_null"
        else:
            # The historical behavior used document order.  Make that fallback
            # explicit and auditable while preferring verified dates when present.
            strategy = "latest"
            strategy_source = "compatibility_default_latest"

    if strategy not in CONFLICT_RESOLUTION_STRATEGIES:
        raise ValueError(
            "conflict_resolution.strategy must be one of "
            f"{sorted(CONFLICT_RESOLUTION_STRATEGIES)}; received {strategy!r}"
        )
    if strategy in {"maximum", "minimum"} and value_type != "continuous":
        if raw is not None:
            raise ValueError(
                f"conflict_resolution strategy {strategy!r} requires a continuous feature"
            )
        strategy = "latest"
        strategy_source = "compatibility_default_latest_incompatible_inference"

    positive_category: str | None = None
    if strategy == "any_positive":
        raw_positive = str((raw or {}).get("positive_category") or "").strip()
        if raw_positive:
            positive_category = _canonical_category(raw_positive, categories)
        else:
            positive_category = _likely_positive_category(categories)
        if positive_category is None:
            raise ValueError(
                "any_positive conflict resolution requires one exact positive_category "
                "from a binary ontology"
            )

    return {
        "strategy": strategy,
        "positive_category": positive_category,
        "strategy_source": strategy_source,
        "dated_observations_precede_undated": strategy in {"latest", "earliest"},
        "source_order_tie_breaker": "last" if strategy != "earliest" else "first",
    }


def _page_prompt_feature_definitions(
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project definitions to the richer oversized-note observation contract."""

    return _prompt_feature_definitions(definitions)


def _refresh_conflict_resolution(definition: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive and persist a policy after a measurement ontology is revised."""

    refreshed = dict(definition)
    refreshed.pop("conflict_resolution", None)
    policy = _resolved_conflict_resolution(refreshed)
    refreshed["conflict_resolution"] = {
        "strategy": policy["strategy"],
        "positive_category": policy["positive_category"],
    }
    return refreshed


def _stale_category_ontology_audit(path: Path) -> dict[str, Any] | None:
    """Return an audit whose old closed ontology now expands more precisely."""

    if not path.is_file():
        return None
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(audit, Mapping):
        return None
    for item in audit.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        previous = [str(value) for value in item.get("allowed_categories") or []]
        current = _normalized_category_values(
            value_type=item.get("value_type"),
            values=previous,
        )
        if current != previous:
            return dict(audit)
    return None


def _supersede_stale_category_ontology_audit(
    path: Path,
    *,
    previous: Mapping[str, Any] | None,
) -> None:
    if previous is None or _stale_category_ontology_audit(path) is None:
        return
    _write_json(
        path,
        {
            "schema_version": "stage2_category_ontology_repair_v1",
            "resolution": "superseded_by_expanded_category_ontology",
            "superseded_at": _now(),
            "previous_audit": dict(previous),
        },
    )


def _feature_name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _aligned_extraction_values(
    values: Mapping[Any, Any],
    *,
    feature_names: Sequence[str],
    row_id: int,
) -> dict[str, Any]:
    """Recover harmless key drift and conservatively fill omitted features."""

    raw = {str(key): value for key, value in values.items()}
    expected = set(feature_names)
    aligned = {name: raw[name] for name in feature_names if name in raw}
    used = set(aligned)
    aliases: dict[str, str] = {}
    missing: list[str] = []
    for name in feature_names:
        if name in aligned:
            continue
        candidates = [
            candidate
            for candidate in raw
            if candidate not in used and _feature_name_key(candidate) == _feature_name_key(name)
        ]
        if len(candidates) == 1:
            candidate = candidates[0]
            aligned[name] = raw[candidate]
            used.add(candidate)
            aliases[candidate] = name
        else:
            aligned[name] = None
            missing.append(name)
    extra = sorted(set(raw) - used)
    if aliases or missing or extra:
        LOGGER.warning(
            "Stage 2 extraction normalized feature keys row_id=%s aliases=%s "
            "missing_as_null=%s extras_dropped=%s",
            row_id,
            aliases,
            missing,
            extra,
        )
    if set(aligned) != expected:  # pragma: no cover - defensive invariant
        raise RuntimeError("Stage 2 feature-key normalization changed the expected schema")
    return aligned


def _canonical_category(value: Any, declared: Sequence[str]) -> str | None:
    text = str(value).strip()
    if text in declared:
        return text
    key = re.sub(r"[^a-z0-9]+", "", text.lower())
    matches = [
        category for category in declared if re.sub(r"[^a-z0-9]+", "", category.lower()) == key
    ]
    if len(matches) == 1:
        return matches[0]
    numeric_tokens = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", text)
    if len(numeric_tokens) == 1:
        numeric_matches = [
            category
            for category in declared
            if re.findall(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", category) == numeric_tokens
        ]
        if len(numeric_matches) == 1:
            return numeric_matches[0]
    return None


class _ExtractionCategoryError(ValueError):
    """Closed-vocabulary extraction failures with enough state for safe recovery."""

    def __init__(
        self,
        *,
        issues: Sequence[Mapping[str, Any]],
        response: Mapping[str, Any],
    ) -> None:
        self.issues = tuple(dict(issue) for issue in issues)
        self.response = copy.deepcopy(dict(response))
        first = self.issues[0]
        allowed = json.dumps(
            first["allowed_categories"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        suffix = (
            f"; {len(self.issues) - 1} additional invalid categorical value(s)"
            if len(self.issues) > 1
            else ""
        )
        super().__init__(
            f"feature {first['feature_name']!r} value "
            f"{first['prior_extracted_value']!r} is invalid; allowed values are "
            f"{allowed}, {DOCUMENTED_UNKNOWN}, or null{suffix}"
        )


class _ExtractionValueError(ValueError):
    """Scalar/type extraction failures that can be repaired feature by feature."""

    def __init__(
        self,
        *,
        issues: Sequence[Mapping[str, Any]],
        response: Mapping[str, Any],
    ) -> None:
        self.issues = tuple(dict(issue) for issue in issues)
        self.response = copy.deepcopy(dict(response))
        first = self.issues[0]
        suffix = (
            f"; {len(self.issues) - 1} additional invalid value(s)" if len(self.issues) > 1 else ""
        )
        super().__init__(f"feature {first['feature_name']!r} {first['reason']}" f"{suffix}")


def _extraction_error_from_exception(
    exc: BaseException,
    error_type: type[_ExtractionCategoryError] | type[_ExtractionValueError] | type[_ExtractionFieldError],
) -> _ExtractionCategoryError | _ExtractionValueError | _ExtractionFieldError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, error_type):
            return current
        seen.add(id(current))
        # Reduced requests run while handling an earlier field failure. That
        # incidental context must not attribute a new malformed response to an
        # already excluded field; the request layer explicitly chains causes.
        current = current.__cause__ or (
            current.__context__ if error_type is not _ExtractionFieldError else None
        )
    return None


def _category_error_from_exception(exc: BaseException) -> _ExtractionCategoryError | None:
    result = _extraction_error_from_exception(exc, _ExtractionCategoryError)
    return result if isinstance(result, _ExtractionCategoryError) else None


def _value_error_from_exception(exc: BaseException) -> _ExtractionValueError | None:
    result = _extraction_error_from_exception(exc, _ExtractionValueError)
    return result if isinstance(result, _ExtractionValueError) else None


def _null_invalid_extraction_values(
    error: _ExtractionValueError,
) -> dict[str, Any]:
    patched = copy.deepcopy(error.response)
    rows_by_id = {int(row["row_id"]): row for row in patched["rows"]}
    for issue in error.issues:
        rows_by_id[int(issue["row_id"])]["values"][str(issue["feature_name"])] = None
    return patched


def _category_ontology_plan(
    error: _ExtractionCategoryError,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Deduplicate equal invalid values while preserving their response targets."""

    item_by_key: dict[str, dict[str, Any]] = {}
    targets_by_key: dict[str, list[dict[str, Any]]] = {}
    for issue in error.issues:
        key = json.dumps(
            {
                "feature_name": issue["feature_name"],
                "prior_extracted_value": issue["prior_extracted_value"],
                "allowed_categories": issue["allowed_categories"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if key not in item_by_key:
            definition = issue["definition"]
            item_by_key[key] = {
                "mapping_id": "",
                "feature_name": str(issue["feature_name"]),
                "value_type": str(definition.get("value_type") or "categorical"),
                "description": str(definition.get("description") or ""),
                "measurement_definition": str(definition.get("measurement_definition") or ""),
                "missing_value_rule": str(definition.get("missing_value_rule") or ""),
                "allowed_categories": list(issue["allowed_categories"]),
                "prior_extracted_value": issue["prior_extracted_value"],
                "occurrence_count": 0,
            }
            targets_by_key[key] = []
        item_by_key[key]["occurrence_count"] += 1
        targets_by_key[key].append(
            {
                "row_id": int(issue["row_id"]),
                "feature_name": str(issue["feature_name"]),
            }
        )

    items: list[dict[str, Any]] = []
    targets: dict[str, list[dict[str, Any]]] = {}
    for index, key in enumerate(sorted(item_by_key), start=1):
        mapping_id = f"category_mapping_{index:04d}"
        item = dict(item_by_key[key])
        item["mapping_id"] = mapping_id
        items.append(item)
        targets[mapping_id] = list(targets_by_key[key])
    return items, targets


def _category_ontology_prompt(
    items: Sequence[Mapping[str, Any]], *, validation_error: str | None = None,
) -> list[dict[str, str]]:
    if len(items) != 1:
        raise ValueError("category interpretation requires one phrase per call")
    item = items[0]
    feature = {**item, "name": item["feature_name"], "categories_or_unit": item["allowed_categories"]}
    return clinical_prompts.messages("08_map_categories", clinical_prompts.feature_text(feature)
        + "\n\nPhrase to interpret\n" + clinical_prompts.scalar(item["prior_extracted_value"])
        + ("\n\nValidation error\n" + validation_error if validation_error else ""))


def _validate_category_ontology(
    value: Mapping[str, Any],
    *,
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if set(value) == {"value"} and len(items) == 1:
        value = {"corrections": [{"mapping_id": items[0]["mapping_id"], "value": value["value"]}]}

    raw_corrections = value.get("corrections")
    if not isinstance(raw_corrections, list):
        raise ValueError("category ontology response requires a corrections array")
    item_by_id = {str(item["mapping_id"]): item for item in items}
    corrections: dict[str, Any] = {}
    for raw in raw_corrections:
        if not isinstance(raw, Mapping):
            raise ValueError("each category ontology correction must be an object")
        mapping_id = str(raw.get("mapping_id") or "")
        if mapping_id not in item_by_id or mapping_id in corrections:
            raise ValueError("category ontology returned an unknown or duplicate mapping_id")
        if "value" not in raw:
            raise ValueError(f"category mapping {mapping_id!r} omitted its value")
        extracted = _clean_scalar(raw.get("value"))
        declared = list(item_by_id[mapping_id]["allowed_categories"])
        if extracted is None:
            corrections[mapping_id] = None
            continue
        if _documented_unknown_answer(extracted, declared):
            corrections[mapping_id] = DOCUMENTED_UNKNOWN
            continue
        if _is_missing_scalar(extracted) and _canonical_category(extracted, declared) is None:
            corrections[mapping_id] = None
            continue
        canonical = _canonical_category(extracted, declared)
        if canonical is None:
            allowed = json.dumps(declared, ensure_ascii=False, separators=(",", ":"))
            raise ValueError(
                f"category mapping {mapping_id!r} returned {extracted!r}; "
                f"allowed values are {allowed}, {DOCUMENTED_UNKNOWN}, or null"
            )
        corrections[mapping_id] = canonical
    if set(corrections) != set(item_by_id):
        raise ValueError("category ontology response omitted one or more mapping IDs")
    return {
        "corrections": [
            {"mapping_id": mapping_id, "value": corrections[mapping_id]}
            for mapping_id in item_by_id
        ]
    }


def _apply_category_corrections(
    response: Mapping[str, Any],
    *,
    corrections: Mapping[str, Any],
    targets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    patched = copy.deepcopy(dict(response))
    rows_by_id = {int(row["row_id"]): row for row in patched["rows"]}
    values_by_mapping = {
        str(row["mapping_id"]): row.get("value") for row in corrections["corrections"]
    }
    for mapping_id, mapping_targets in targets.items():
        for target in mapping_targets:
            rows_by_id[int(target["row_id"])]["values"][str(target["feature_name"])] = (
                values_by_mapping[mapping_id]
            )
    return patched


def _request_validated_extraction(
    *,
    messages: Sequence[Mapping[str, str]],
    row_ids: Sequence[int],
    definitions: Sequence[Mapping[str, Any]],
    request_json: RequestJSON,
    ontology_audit_path: Path,
    messages_for_definitions: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, str]]],
    validate_response: Callable[..., dict[str, Any]] | None = None,
    prior_response: Mapping[str, Any] | None = None,
    response_schema: Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]] | None = None,
    response_instructions: Callable[[Sequence[Mapping[str, Any]]], str] | None = None,
) -> dict[str, Any]:
    """Recover field failures with smaller prompts and category failures by mapping."""

    if validate_response is None and len(row_ids) == 1:
        response_schema = response_schema or _extraction_response_schema
        response_instructions = response_instructions or _extraction_output_instructions

    issue_audit_path = ontology_audit_path.with_name("extraction_issues.json")
    pending_path = ontology_audit_path.with_name("pending_category_ontology.json")
    pending_input_fingerprint = _value_fingerprint(
        {
            "row_ids": [int(row_id) for row_id in row_ids],
            "definitions": _prompt_feature_definitions(definitions),
        }
    )
    issue_events: list[dict[str, Any]] = []
    pending_value_repair_audit: dict[str, Any] | None = None

    def validate_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
        if validate_response is not None:
            return validate_response(value, definitions=definitions)
        return _validate_extraction(
            value,
            row_ids=row_ids,
            definitions=definitions,
        )

    def request_or_resume_pending_category() -> dict[str, Any]:
        nonlocal pending_value_repair_audit
        if pending_path.is_file():
            try:
                pending = json.loads(pending_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(pending, Mapping)
                    or pending.get("schema_version")
                    != PENDING_CATEGORY_ONTOLOGY_SCHEMA_VERSION
                    or pending.get("input_fingerprint") != pending_input_fingerprint
                    or not isinstance(pending.get("issues"), list)
                    or not isinstance(pending.get("response"), Mapping)
                ):
                    raise ValueError("incompatible pending category ontology checkpoint")
                prior_events = pending.get("prior_issue_events") or []
                if not isinstance(prior_events, list):
                    raise ValueError("invalid pending category issue events")
                issue_events.extend(
                    dict(event) for event in prior_events if isinstance(event, Mapping)
                )
                raw_value_repair = pending.get("value_repair_audit")
                pending_value_repair_audit = (
                    dict(raw_value_repair)
                    if isinstance(raw_value_repair, Mapping)
                    else None
                )
                LOGGER.info(
                    "resume pending Stage 2 category ontology mapping without "
                    "repeating patient-note extraction: %s",
                    pending_path,
                )
                pending_issues = [
                    dict(issue)
                    for issue in pending["issues"]
                    if isinstance(issue, Mapping)
                ]
                if not pending_issues:
                    raise ValueError("pending category ontology checkpoint has no issues")
                raise _ExtractionCategoryError(
                    issues=pending_issues,
                    response=pending["response"],
                )
            except _ExtractionCategoryError:
                raise
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pending_path.unlink(missing_ok=True)
        with request_audit.context(
            _audit_path=str(ontology_audit_path.with_name("request_events.jsonl")),
            checkpoint_dir=str(ontology_audit_path.parent),
            patient_row_ids=list(map(int, row_ids)),
            feature_ids=[str(feature.get("feature_id") or feature["name"])
                         for feature in definitions],
        ):
            validator = validate_candidate
            if response_schema is not None:
                validator = _StructuredExtractionValidator(
                    validate_candidate,
                    {"type": "json_schema", "json_schema": {
                        "name": "clinical_measurements", "strict": True,
                        "schema": response_schema(definitions),
                    }},
                    response_instructions(definitions) if response_instructions else "",
                )
            validated_response = request_json(
                messages,
                validator,
                request_kind="extraction",
            )
        pending_path.unlink(missing_ok=True)
        return validated_response

    try:
        validated = request_or_resume_pending_category()
        _write_json(
            issue_audit_path,
            {
                "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                "completed_at": _now(),
                "events": [],
            },
        )
        return validated
    except (
        Stage2ResponseValidationError,
        _ExtractionFieldError,
        _ExtractionCategoryError,
        _ExtractionValueError,
    ) as exc:
        field_error = _extraction_error_from_exception(exc, _ExtractionFieldError)
        if field_error is not None:
            excluded = set(field_error.feature_names)
            feature_names = [str(definition["name"]) for definition in definitions]
            if not excluded or not excluded <= set(feature_names):
                raise
            remaining = [d for d in definitions if str(d["name"]) not in excluded]
            audit = {
                "schema_version": "stage2_extraction_field_recovery_v1",
                "status": "retrying_remaining_fields",
                "started_at": _now(),
                "excluded_features": sorted(excluded),
                "remaining_features": [str(d["name"]) for d in remaining],
                "validation_error": str(exc),
                "repair_limit": EXTRACTION_FIELD_REPAIR_LIMIT,
            }
            recovery_path = ontology_audit_path.with_name("field_recovery.json")
            _write_json(recovery_path, audit)
            LOGGER.warning(
                "Stage 2 extraction excludes unresolved fields %s and retries %s remaining fields",
                sorted(excluded), len(remaining),
            )
            if remaining:
                recovery_dir = ontology_audit_path.parent / "field_recovery" / _value_fingerprint(sorted(excluded))[:16]
                recovered = _request_validated_extraction(
                    messages=messages_for_definitions(remaining), row_ids=row_ids,
                    definitions=remaining, request_json=request_json,
                    ontology_audit_path=recovery_dir / "category_ontology_repair.json",
                    messages_for_definitions=messages_for_definitions,
                    validate_response=validate_response, prior_response=prior_response,
                    response_schema=response_schema, response_instructions=response_instructions,
                )
            else:
                recovered = {"rows": [{"row_id": int(row_id), "values": {}} for row_id in row_ids]}
            prior_rows = {int(row["row_id"]): row for row in (prior_response or {}).get("rows", [])}
            for row in recovered["rows"]:
                prior = prior_rows.get(int(row["row_id"]), {})
                for name in excluded:
                    row["values"][name] = prior.get("values", {}).get(name)
                if validate_response is not None:
                    state = row.setdefault("carry_forward_state", {})
                    for name in excluded:
                        state[name] = prior.get("carry_forward_state", {}).get(name)
            validated = validate_candidate(recovered)
            issue_events.extend({
                "failure_kind": "unresolved_field_after_repairs",
                "row_id": int(row_id), "feature_name": name,
                "reason": str(field_error),
            } for row_id in row_ids for name in sorted(excluded))
            _write_json(issue_audit_path, {
                "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                "completed_at": _now(), "events": issue_events,
            })
            _write_json(recovery_path, {**audit, "status": "complete", "completed_at": _now(),
                "resolution": "reextract_remaining_fields_preserve_prior_or_null_excluded"})
            return validated
        value_error = _value_error_from_exception(exc)
        value_repair_audit: dict[str, Any] | None = pending_value_repair_audit
        category_error = _category_error_from_exception(exc)
        if value_error is not None:
            issue_events.extend(
                {
                    "failure_kind": "invalid_scalar_or_value_type",
                    "row_id": int(issue["row_id"]),
                    "feature_name": str(issue["feature_name"]),
                    "reason": str(issue["reason"]),
                    "prior_extracted_value": copy.deepcopy(issue.get("prior_extracted_value")),
                }
                for issue in value_error.issues
            )
            patched = _null_invalid_extraction_values(value_error)
            value_repair_audit = {
                "schema_version": "stage2_invalid_feature_value_repair_v1",
                "resolution": "conservative_invalid_features_null",
                "original_validation_error": str(exc),
                "issues": [dict(issue) for issue in value_error.issues],
            }
            try:
                validated = validate_candidate(patched)
            except _ExtractionCategoryError as patched_category_error:
                category_error = patched_category_error
            else:
                _write_json(
                    ontology_audit_path.with_name("invalid_feature_value_repair.json"),
                    value_repair_audit,
                )
                LOGGER.warning(
                    "Stage 2 extraction retained the valid fields and replaced %s "
                    "invalid feature value(s) with null",
                    len(value_error.issues),
                )
                _write_json(
                    issue_audit_path,
                    {
                        "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                        "completed_at": _now(),
                        "events": issue_events,
                    },
                )
                return validated
        if category_error is None:
            feature_names = [str(definition["name"]) for definition in definitions]
            conservative = {
                "rows": [
                    {
                        "row_id": int(row_id),
                        "values": {name: None for name in feature_names},
                    }
                    for row_id in row_ids
                ]
            }
            if validate_response is not None and prior_response is not None:
                prior_rows = {int(row["row_id"]): row for row in prior_response.get("rows", [])}
                for row in conservative["rows"]:
                    prior = prior_rows.get(row["row_id"], {})
                    row["values"] = {name: prior.get("values", {}).get(name) for name in feature_names}
                    row["carry_forward_state"] = {
                        name: prior.get("carry_forward_state", {}).get(name) for name in feature_names
                    }
            validated = _validate_extraction(
                conservative,
                row_ids=row_ids,
                definitions=definitions,
            )
            failure_path = ontology_audit_path.with_name("extraction_failure.json")
            _write_json(
                failure_path,
                {
                    "schema_version": "stage2_extraction_failure_v1",
                    "resolution": "conservative_all_null",
                    "failed_at": _now(),
                    "row_ids": [int(row_id) for row_id in row_ids],
                    "feature_names": feature_names,
                    "validation_error": f"{type(exc).__name__}: {exc}",
                },
            )
            issue_events.extend(
                {
                    "failure_kind": "structural_response_failure",
                    "row_id": int(row_id),
                    "feature_name": None,
                    # The features of this request; the failure mask needs them.
                    "feature_names": list(feature_names),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                for row_id in row_ids
            )
            _write_json(
                issue_audit_path,
                {
                    "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                    "completed_at": _now(),
                    "events": issue_events,
                },
            )
            LOGGER.warning(
                "Stage 2 extraction remained structurally invalid after repairs; "
                "replacing %s extracted value(s) for %s patient(s) with null (%s: %s)",
                len(feature_names) * len(row_ids),
                len(row_ids),
                type(exc).__name__,
                exc,
            )
            return validated

    issue_events.extend(
        {
            "failure_kind": "out_of_ontology_category",
            "row_id": int(issue["row_id"]),
            "feature_name": str(issue["feature_name"]),
            "reason": "extracted value is outside the declared closed ontology",
            "prior_extracted_value": copy.deepcopy(issue.get("prior_extracted_value")),
            "allowed_categories": list(issue.get("allowed_categories") or []),
        }
        for issue in category_error.issues
    )
    items, targets = _category_ontology_plan(category_error)
    LOGGER.warning(
        "Stage 2 extraction exhausted ordinary repairs for %s invalid categorical "
        "value(s); requesting note-free ontology mapping",
        len(category_error.issues),
    )
    ontology_error: str | None = None
    _write_json(
        pending_path,
        {
            "schema_version": PENDING_CATEGORY_ONTOLOGY_SCHEMA_VERSION,
            "status": "awaiting_interpretation",
            "input_fingerprint": pending_input_fingerprint,
            "recorded_at": _now(),
            "issues": [dict(issue) for issue in category_error.issues],
            "response": copy.deepcopy(category_error.response),
            "prior_issue_events": [
                copy.deepcopy(event)
                for event in issue_events
                if event.get("failure_kind") != "out_of_ontology_category"
            ],
            "value_repair_audit": copy.deepcopy(value_repair_audit),
        },
    )
    try:
        corrections = {"corrections": []}
        for item in items:
            mapped = request_json(
                _category_ontology_prompt(
                    [item], validation_error=(
                        f"_ExtractionCategoryError: feature {item['feature_name']!r} "
                        f"value {item['prior_extracted_value']!r} is invalid; allowed values "
                        f"are {item['allowed_categories']!r} or null"
                    ),
                ),
                lambda value, item=item: _validate_category_ontology(value, items=[item]),
                request_kind="interpretation",
                initial_reasoning_effort="none",
            )
            corrections["corrections"].extend(mapped["corrections"])
        resolution = "llm_category_ontology"
    except Stage2ResponseValidationError as exc:
        ontology_error = f"{type(exc).__name__}: {exc}"
        resolution = "conservative_null"
        corrections = {
            "corrections": [
                {"mapping_id": str(item["mapping_id"]), "value": None} for item in items
            ]
        }
        LOGGER.warning(
            "Stage 2 category ontology mapping remained invalid; replacing %s "
            "unmappable categorical value(s) with null (%s)",
            len(category_error.issues),
            ontology_error,
        )

    patched = _apply_category_corrections(
        category_error.response,
        corrections=corrections,
        targets=targets,
    )
    validated = validate_candidate(patched)
    if value_repair_audit is not None:
        _write_json(
            ontology_audit_path.with_name("invalid_feature_value_repair.json"),
            value_repair_audit,
        )
    _write_json(
        ontology_audit_path,
        {
            "schema_version": "stage2_category_ontology_repair_v1",
            "resolution": resolution,
            "original_validation_error": str(category_error),
            "ontology_validation_error": ontology_error,
            "items": items,
            "targets": targets,
            "corrections": corrections["corrections"],
        },
    )
    _write_json(
        issue_audit_path,
        {
            "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
            "completed_at": _now(),
            "events": issue_events,
        },
    )
    pending_path.unlink(missing_ok=True)
    return validated


def _extraction_prompt(*, definitions: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if len(rows) != 1:
        raise ValueError("Stage 2 extraction prompts must contain exactly one patient's record")
    return clinical_prompts.messages("05_extract_patient",
        clinical_prompts.definitions_text(_prompt_feature_definitions(definitions))
        + "\n\nPatient record\n" + str(rows[0].get("text") or "")
        + "\n\nEnd of patient record.\n" + _extraction_output_instructions(definitions))


def _extraction_response_schema(definitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clinical_prompts.label_map(definitions)
    properties = {}
    for feature in definitions:
        value_type = feature.get("value_type", "ambiguous")
        # Numeric category labels (e.g. ECOG 3) may arrive as JSON numbers.
        # The semantic validator still maps them to declared categories and
        # rejects out-of-ontology values. A string-only grammar can otherwise
        # trap numeric-score generation in an endless whitespace sequence.
        types = (["number", "string", "null"]
                 if value_type in {"continuous", "binary", "categorical", "ordinal"}
                 else ["number", "string", "boolean", "null"])
        properties[clinical_prompts.label(feature)] = {"type": types}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _extraction_output_instructions(definitions: Sequence[Mapping[str, Any]]) -> str:
    example = {clinical_prompts.label(d): None for d in definitions}
    return (
        "Return exactly one JSON object with the complete variable names shown below as keys. "
        "Copy each key exactly, including all text on both sides of any colon. "
        "Related components are separate variables; do not replace them with a parent name "
        "or nest them inside another object. Each value is a supported scalar, "
        "documented_unknown when the record states that the value is unknown, or null. "
        "The nulls in this example are format placeholders, not suggested answers:\n"
        + json.dumps(example, ensure_ascii=False, separators=(",", ":"))
    )


@dataclass(frozen=True)
class _StructuredExtractionValidator:
    """Carry a task's wire contract alongside its semantic validator."""

    validate: Callable[[Mapping[str, Any]], dict[str, Any]]
    response_format: Mapping[str, Any]
    repair_instructions: str

    def __call__(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return self.validate(value)


def _serial_response_schema(definitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clinical_prompts.label_map(definitions)
    labels = [clinical_prompts.label(d) for d in definitions]
    values = _extraction_response_schema(definitions)["properties"]
    notes = {name: {"type": ["string", "null"], "maxLength": MAX_SERIAL_FEATURE_STATE_CHARS}
             for name in labels}
    return {"type": "object", "properties": {
        name: {"type": "object", "properties": properties, "required": labels,
               "additionalProperties": False}
        for name, properties in (("values", values), ("decision_notes", notes))
    }, "required": ["values", "decision_notes"], "additionalProperties": False}


def _serial_output_instructions(definitions: Sequence[Mapping[str, Any]]) -> str:
    example = {key: {clinical_prompts.label(d): None for d in definitions}
               for key in ("values", "decision_notes")}
    return (
        "Update only the listed clinical variables using the record section and prior values. "
        "Return exactly one JSON object with values and decision_notes, each containing every "
        "requested variable. Copy each complete key exactly, including all text on both sides "
        "of any colon; keep related components separate. Values are supported scalars, "
        "documented_unknown when the record states that the value is unknown, or null. "
        "Decision notes retain the "
        "date or conflict information needed for the next section, as a short string or null. "
        "The nulls in this example are format placeholders, not suggested answers:\n"
        + json.dumps(example, ensure_ascii=False, separators=(",", ":"))
    )


def _serial_extraction_prompt(*, definitions: Sequence[Mapping[str, Any]], row_id: int,
    chunk_text: str, prior_values: Mapping[str, Any], prior_feature_state: Mapping[str, Any],
    chunk_index: int, char_start: int, char_end: int, document_chars: int) -> list[dict[str, str]]:
    prior = "\n".join(f"- {clinical_prompts.label(d)}: {clinical_prompts.scalar(prior_values.get(d['name']))}; "
        f"decision note: {clinical_prompts.scalar(prior_feature_state.get(d['name']))}" for d in definitions)
    return clinical_prompts.messages("06_serial_chunk",
        clinical_prompts.definitions_text(_prompt_feature_definitions(definitions))
        + "\n\nValues found so far\n" + prior + "\n\nNext section of the record\n" + chunk_text
        + "\n\nEnd of record section.\n" + _serial_output_instructions(definitions))


def _page_extraction_prompt_template(*, definitions: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    messages = clinical_prompts.messages("07_page_observations", "")
    # Preserve the separator here; messages() normally strips trailing whitespace
    # after the patient text has been appended.
    messages[-1]["content"] = (
        clinical_prompts.definitions_text(_page_prompt_feature_definitions(definitions), include_conflict=False)
        + "\n\nRecord section\n")
    return messages


def _page_output_instructions() -> str:
    return (
        "Return exactly one JSON object with an observations array. Each observation has "
        "exactly feature, value, quote, and governing_date_quote. Copy the complete clinical "
        "variable name into feature. Use a supported scalar value (documented_unknown when the "
        "text states that the value is unknown), a short supporting quote, "
        "and the linked date quote or null. Include only occurrences that support the requested "
        "attribute under its measurement definition; do not infer it from other measurements. "
        "For example, severity alone does not establish symptom frequency. "
        "Omit missing-value placeholders. Preserve real repeated occurrences, reporting each "
        "once in source order, and stop after the last supported occurrence. "
        'Return {"observations": []} if there are none.'
    )


def _page_response_schema(definitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    properties = _extraction_response_schema(definitions)["properties"]
    # Leave category normalization to the semantic validator, including numeric
    # category labels. Null placeholders remain valid JSON and are omitted locally.
    value_types = sorted({t for schema in properties.values() for t in schema["type"]})
    observation = {"type": "object", "properties": {
        "feature": {"type": "string", "enum": list(properties)},
        "value": {"type": value_types},
        "quote": {"type": ["string", "null"]},
        "governing_date_quote": {"type": ["string", "null"]},
    }, "required": ["feature", "value", "quote", "governing_date_quote"],
        "additionalProperties": False}
    return {"type": "object", "properties": {
        "observations": {"type": "array", "items": observation},
    }, "required": ["observations"], "additionalProperties": False}


def _render_page_extraction_prompt(template: Sequence[Mapping[str, str]], text: str) -> list[dict[str, str]]:
    messages = [dict(message) for message in template]
    messages[-1]["content"] = (messages[-1]["content"] + text
        + "\n\nEnd of record section.\n" + _page_output_instructions()).strip()
    return messages


def _page_extraction_prompt(*, definitions: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> list[dict[str, str]]:
    return _render_page_extraction_prompt(
        _page_extraction_prompt_template(definitions=definitions), str(row.get("text") or ""))


def _prompt_chars(messages: Sequence[Mapping[str, str]]) -> int:
    """Return the exact rendered content characters sent to the endpoint."""

    return sum(len(str(message.get("content") or "")) for message in messages)


def _token_id_count(encoded: Any) -> int:
    """Count one unbatched token-id sequence from common tokenizer outputs."""

    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    shape = getattr(encoded, "shape", None)
    if shape is not None and len(shape):
        return int(shape[-1])
    if hasattr(encoded, "tolist") and not isinstance(encoded, list):
        encoded = encoded.tolist()
    if isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes, bytearray)):
        if encoded and isinstance(encoded[0], Sequence):
            return len(encoded[0])
        return len(encoded)
    raise TypeError("tokenizer did not return a countable input_ids sequence")


def prompt_token_count(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
) -> int:
    """Count the endpoint-ready chat prompt, including generation framing."""

    if tokenizer is None:
        raise ValueError("a tokenizer is required for token-bounded Stage 2 extraction")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError(
            "the extraction tokenizer must implement apply_chat_template so Stage 2 "
            "can enforce the model context window exactly"
        )
    encoded = apply_chat_template(
        [dict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=True,
    )
    return _token_id_count(encoded)


def _text_token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    return _token_id_count(encoded)


def _partition_feature_definitions(
    definitions: Sequence[Mapping[str, Any]],
    *,
    feature_batch_size: int,
) -> list[list[Mapping[str, Any]]]:
    """Return stable consecutive feature slices for extraction prompts."""

    if (
        isinstance(feature_batch_size, bool)
        or not isinstance(feature_batch_size, int)
        or feature_batch_size < 1
    ):
        raise ValueError("feature_batch_size must be a positive integer")
    return [
        list(definitions[start : start + feature_batch_size])
        for start in range(0, len(definitions), feature_batch_size)
    ]


def _normalize_single_patient_wrapper(value, *, row_ids, feature_names):
    """Repair only exact, complete single-patient wrappers; never invent values."""
    if len(row_ids) != 1 or not isinstance(value, Mapping):
        return value
    if isinstance(value.get("rows"), list):
        return value
    expected = set(feature_names)
    normalized = None
    shape = None
    if set(value) == expected:
        normalized = {"row_id": int(row_ids[0]), "values": dict(value)}
        shape = "flat_exact_feature_map"
    elif set(value) in ({"values"}, {"row_id", "values"}):
        values = value.get("values")
        if isinstance(values, Mapping) and set(values) == expected:
            normalized = {"row_id": value.get("row_id", int(row_ids[0])), "values": dict(values)}
            shape = "single_row_object"
    elif set(value) == {"rows"} and isinstance(value["rows"], Mapping):
        row = value["rows"]
        if (set(row) == {"row_id", "values"} and isinstance(row["values"], Mapping)
                and set(row["values"]) == expected):
            normalized = dict(row)
            shape = "rows_object_instead_of_array"
    if normalized is None:
        return value
    request_audit.event("response_shape_normalized", original_shape=shape,
                        features=len(expected), inferred_measurements=0)
    return {"rows": [normalized]}


def _named_extraction_values(values, definitions):
    """Attribute name failures by exact alignment; never guess a renamed value."""
    try:
        return clinical_prompts.named_values(values, definitions)
    except ValueError as exc:
        if not isinstance(values, Mapping):
            raise
        names = clinical_prompts.label_map(definitions)
        matched: set[str] = set()
        duplicates: set[str] = set()
        for key in values:
            try:
                name = clinical_prompts.resolve_label(key, names)
            except ValueError:
                continue
            if name in matched:
                duplicates.add(name)
            matched.add(name)
        affected = (set(names.values()) - matched) | duplicates
        if affected:
            raise _ExtractionFieldError(str(exc), feature_names=sorted(affected)) from exc
        raise


def _validate_extraction(
    value: Mapping[str, Any],
    *,
    row_ids: Sequence[int],
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(row_ids) == 1 and "rows" not in value and "values" not in value:
        request_audit.event("response_shape_normalized", original_shape="clinical_variable_map",
                            features=len(definitions), inferred_measurements=0)
        value = {"rows": [{"row_id": int(row_ids[0]),
                  "values": _named_extraction_values(value, definitions)}]}

    value = _normalize_single_patient_wrapper(
        value, row_ids=row_ids, feature_names=[str(feature["name"]) for feature in definitions],
    )
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("extraction response requires a rows array")
    expected_rows = {int(row_id) for row_id in row_ids}
    feature_names = [str(feature["name"]) for feature in definitions]
    by_row: dict[int, dict[str, Any]] = {}
    definitions_by_name = {str(feature["name"]): feature for feature in definitions}
    category_issues: list[dict[str, Any]] = []
    value_issues: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("each extraction row must be an object")
        row_id = int(raw.get("row_id"))
        if row_id not in expected_rows or row_id in by_row:
            raise ValueError("extraction returned an unknown or duplicate row_id")
        values = raw.get("values")
        if not isinstance(values, Mapping):
            raise ValueError("each extraction row requires a values object")
        aligned_values = _aligned_extraction_values(
            values,
            feature_names=feature_names,
            row_id=row_id,
        )
        clean_values: dict[str, Any] = {}
        for name in feature_names:
            definition = definitions_by_name[name]
            value_type = str(definition.get("value_type") or "ambiguous")
            declared = _declared_categories(definition)
            try:
                extracted = _clean_scalar(aligned_values[name])
            except ValueError:
                extracted = aligned_values[name]
                value_issues.append(
                    {
                        "row_id": row_id,
                        "feature_name": name,
                        "value_type": value_type,
                        "reason": (
                            "requires one scalar JSON value or null, but the model "
                            f"returned {type(extracted).__name__}"
                        ),
                        "prior_extracted_value": extracted,
                    }
                )
                clean_values[name] = extracted
                continue
            if _documented_unknown_answer(extracted, declared):
                # The record states that the value is unknown (every value type).
                clean_values[name] = DOCUMENTED_UNKNOWN
                continue
            if documented_unknown.is_malformed_documented_unknown(extracted):
                value_issues.append(
                    {
                        "row_id": row_id,
                        "feature_name": name,
                        "value_type": value_type,
                        "reason": (
                            f"must write the reserved value exactly as {DOCUMENTED_UNKNOWN}, "
                            "with no other text"
                        ),
                        "prior_extracted_value": extracted,
                    }
                )
                clean_values[name] = extracted
                continue
            if _is_missing_scalar(extracted) and _canonical_category(extracted, declared) is None:
                extracted = None
            if extracted is not None and value_type == "continuous":
                if isinstance(extracted, bool) or not isinstance(extracted, (int, float, str)):
                    value_issues.append(
                        {
                            "row_id": row_id,
                            "feature_name": name,
                            "value_type": value_type,
                            "reason": (
                                "requires one JSON number, one documented categorical or "
                                "threshold string, or null"
                            ),
                            "prior_extracted_value": extracted,
                        }
                    )
                    clean_values[name] = extracted
                    continue
                if isinstance(extracted, (int, float)):
                    extracted = float(extracted)
                else:
                    extracted = extracted.strip()
            elif extracted is not None and value_type in {"binary", "categorical", "ordinal"}:
                canonical = _canonical_category(extracted, declared) if declared else str(extracted)
                if canonical is None:
                    category_issues.append(
                        {
                            "row_id": row_id,
                            "feature_name": name,
                            "prior_extracted_value": extracted,
                            "allowed_categories": list(declared),
                            "definition": dict(definition),
                        }
                    )
                else:
                    extracted = canonical
            clean_values[name] = extracted
        by_row[row_id] = {"row_id": row_id, "values": clean_values}
        if "carry_forward_state" in raw:
            by_row[row_id]["carry_forward_state"] = copy.deepcopy(
                raw.get("carry_forward_state")
            )
    if set(by_row) != expected_rows:
        raise ValueError("extraction response omitted one or more supplied rows")
    normalized_response = {"rows": [by_row[int(row_id)] for row_id in row_ids]}
    if value_issues:
        raise _ExtractionValueError(
            issues=value_issues,
            response=normalized_response,
        )
    if category_issues:
        raise _ExtractionCategoryError(issues=category_issues, response=normalized_response)
    return normalized_response


def _validate_serial_extraction(
    value: Mapping[str, Any],
    *,
    row_id: int,
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate cumulative values plus bounded policy metadata for the next chunk."""
    if set(value) == {"values", "decision_notes"}:
        for field in ("values", "decision_notes"):
            if not isinstance(value[field], Mapping):
                raise ValueError(f"{field} must be a JSON object keyed by every requested clinical variable")
        value = {"rows": [{"row_id": int(row_id),
            "values": _named_extraction_values(value["values"], definitions),
            "carry_forward_state": _named_extraction_values(value["decision_notes"], definitions)}]}


    feature_names = [str(definition["name"]) for definition in definitions]
    value = _normalize_single_patient_wrapper(
        value, row_ids=[row_id], feature_names=feature_names,
    )
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError(
            "Return one JSON object with exactly values and decision_notes; "
            "both must be objects keyed by every requested clinical variable"
        )
    row = rows[0]
    raw_state = row.get("carry_forward_state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("Include decision_notes as an object keyed by every requested clinical variable")
    if set(map(str, raw_state)) != set(feature_names):
        raise ValueError(
            "decision_notes must contain every requested clinical variable exactly once"
        )
    state: dict[str, str | None] = {}
    for name in feature_names:
        raw = raw_state.get(name)
        if raw is None:
            state[name] = None
            continue
        if isinstance(raw, bool):
            raw = "true" if raw else "false"
        elif isinstance(raw, int):
            raw = str(raw)
        elif isinstance(raw, float) and math.isfinite(raw):
            raw = json.dumps(raw, ensure_ascii=False, allow_nan=False)
        if not isinstance(raw, str):
            # Carry-forward state is bounded prompt metadata, not an extracted
            # scientific value. Preserve scalar evidence losslessly as text and
            # conservatively drop malformed containers/non-finite values rather
            # than crashing after an otherwise valid ontology correction.
            state[name] = None
            continue
        rendered = raw.strip()
        if len(rendered) > MAX_SERIAL_FEATURE_STATE_CHARS:
            raise _ExtractionFieldError(
                f"serial carry_forward_state for {name!r} exceeds "
                f"{MAX_SERIAL_FEATURE_STATE_CHARS} characters", feature_names=[name],
            )
        state[name] = rendered or None
    # Value/category errors retain a partial response for downstream recovery.
    # Check metadata first so every such response is safe to carry to the next
    # chunk, including after nulling or mapping an invalid scientific value.
    return _validate_extraction(
        {**value, "rows": [{**row, "carry_forward_state": state}]},
        row_ids=[row_id],
        definitions=definitions,
    )


class _PageObservationValidationError(ValueError):
    """Invalid provenance rows while retaining every independently valid observation."""

    def __init__(
        self,
        *,
        issues: Sequence[Mapping[str, Any]],
        response: Mapping[str, Any],
    ) -> None:
        self.issues = tuple(dict(issue) for issue in issues)
        self.response = copy.deepcopy(dict(response))
        first = self.issues[0]
        suffix = f"; {len(self.issues) - 1} additional issue(s)" if len(self.issues) > 1 else ""
        super().__init__(
            f"page observation {first.get('observation_index')} for feature "
            f"{first.get('feature_name')!r} is invalid: {first.get('reason')}{suffix}"
        )


def _page_observation_error_from_exception(
    exc: BaseException,
) -> _PageObservationValidationError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, _PageObservationValidationError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _quote_span(
    *,
    text: str,
    quote: Any,
    reported_start: Any,
    reported_end: Any,
    label: str,
) -> tuple[str, int | None, int | None, str]:
    """Keep supplied evidence; locate it opportunistically for the audit only."""

    if not isinstance(quote, str) or not quote.strip():
        raise ValueError(f"{label} must be nonempty text")
    start: int | None = None
    end: int | None = None
    if not isinstance(reported_start, bool) and isinstance(reported_start, int):
        start = int(reported_start)
    if not isinstance(reported_end, bool) and isinstance(reported_end, int):
        end = int(reported_end)
    if (
        start is not None
        and end is not None
        and 0 <= start < end <= len(text)
        and text[start:end] == quote
    ):
        return quote, start, end, "reported_exact"

    matches: list[int] = []
    cursor = 0
    while True:
        match = text.find(quote, cursor)
        if match < 0:
            break
        matches.append(match)
        cursor = match + 1
    if not matches:
        return quote, None, None, "not_located"
    if start is None:
        if len(matches) != 1:
            return quote, None, None, "ambiguous_location"
        selected = matches[0]
        method = "unique_exact_match"
    else:
        ranked = sorted(matches, key=lambda candidate: (abs(candidate - start), candidate))
        if len(ranked) > 1 and abs(ranked[0] - start) == abs(ranked[1] - start):
            return quote, None, None, "ambiguous_location"
        selected = ranked[0]
        method = "nearest_exact_match"
    return quote, selected, selected + len(quote), method


def _canonical_observation_time(value: Any) -> str | None:
    if value is None or _is_missing_scalar(value):
        return None
    if not isinstance(value, str):
        raise ValueError("recorded_at must be an ISO-8601 string or null")
    text = value.strip()
    if re.fullmatch(r"\d{4}", text):
        return text
    if re.fullmatch(r"\d{4}-\d{2}", text):
        try:
            datetime.strptime(text, "%Y-%m")
        except ValueError as exc:
            raise ValueError("recorded_at must be a valid ISO-8601 month") from exc
        return text
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            datetime.fromisoformat(text)
        except ValueError as exc:  # pragma: no cover - guarded by datetime
            raise ValueError("recorded_at must be a valid ISO-8601 date") from exc
        return text
    rendered = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(rendered)
    except ValueError as exc:
        raise ValueError("recorded_at must be a valid ISO-8601 date or datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def _canonical_time_evidence(value: Any) -> str:
    """Normalize the model-supplied date text locally."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("recorded_at_evidence must be a nonempty date or datetime quote string")
    text = value.strip()
    if re.fullmatch(r"\d{4}", text):
        return text
    if re.fullmatch(r"\d{4}[-/]\d{1,2}", text):
        year, month = (int(part) for part in re.split(r"[-/]", text))
        if not 1 <= month <= 12:
            raise ValueError("recorded_at_evidence contains an invalid calendar month")
        return f"{year:04d}-{month:02d}"
    numeric_month_year = re.fullmatch(r"(\d{1,2})[-/](\d{4})", text)
    if numeric_month_year is not None:
        month, year = (int(part) for part in numeric_month_year.groups())
        if not 1 <= month <= 12:
            raise ValueError("recorded_at_evidence contains an invalid calendar month")
        return f"{year:04d}-{month:02d}"
    month_year = re.fullmatch(
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    if month_year is not None:
        parsed_month = datetime.strptime(month_year.group(1)[:3].title(), "%b").month
        return f"{int(month_year.group(2)):04d}-{parsed_month:02d}"
    try:
        parsed = pd.to_datetime(text, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "recorded_at_evidence must itself be a parseable date or datetime quote"
        ) from exc
    if isinstance(parsed, pd.DatetimeIndex):
        if len(parsed) != 1:  # pragma: no cover - scalar input invariant
            raise ValueError("recorded_at_evidence must contain one date or datetime")
        parsed = parsed[0]
    timestamp = pd.Timestamp(parsed)
    if not re.search(r"\d{1,2}:\d{2}", text):
        return timestamp.date().isoformat()
    return timestamp.isoformat().replace("+00:00", "Z")


def _validate_page_observations(
    value: Mapping[str, Any],
    *,
    page: Mapping[str, Any],
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate reported observations, retaining quotations without a match gate."""

    if set(value) == {"observations"}:
        value = {"rows": [{"row_id": int(page["row_id"]), "observations": value["observations"]}]}
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError("page extraction response requires exactly one row object")
    expected_row_id = int(page["row_id"])
    try:
        row_id = int(rows[0].get("row_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("page extraction row_id must be the supplied integer") from exc
    if row_id != expected_row_id:
        raise ValueError("page extraction returned an unknown row_id")
    raw_observations = rows[0].get("observations")
    if not isinstance(raw_observations, list):
        raise ValueError("page extraction row requires an observations array")

    text = str(page.get("text") or "")
    page_meta = dict(page.get("page") or {})
    page_index = int(page_meta.get("page_index"))
    page_char_start = int(page_meta.get("char_start"))
    definition_by_name = {str(definition["name"]): definition for definition in definitions}
    labels = clinical_prompts.label_map(definitions)
    normalized: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    omitted_missing = 0

    for observation_index, raw in enumerate(raw_observations, start=1):
        feature_name = ""
        try:
            if not isinstance(raw, Mapping):
                raise ValueError("observation must be an object")
            if "feature" in raw:
                if set(raw) != {"feature", "value", "quote", "governing_date_quote"}:
                    raise ValueError("each observation requires feature, value, quote, and governing_date_quote")
                feature_name = clinical_prompts.resolve_label(raw["feature"], labels)
            else:
                feature_name = str(raw.get("feature_name") or "")
            if feature_name not in definition_by_name:
                raise ValueError("feature_name is not one of the supplied features")
            if "value" not in raw:
                raise ValueError("observation must include value")
            scalar_response = _validate_extraction(
                {"rows": [{"row_id": row_id, "values": {feature_name: raw["value"]}}]},
                row_ids=[row_id], definitions=[definition_by_name[feature_name]],
            )
            scalar = scalar_response["rows"][0]["values"][feature_name]
            if scalar is None:
                # A missing-value placeholder is not a value-bearing occurrence
                # and cannot vote in the mode. Omit it before date/quote parsing;
                # declared negative categories remain nonmissing after validation.
                omitted_missing += 1
                continue
            if "feature" in raw:
                date_quote = raw["governing_date_quote"]
                quote, quote_start, quote_end, _ = _quote_span(text=text, quote=raw["quote"],
                    reported_start=None, reported_end=None, label="observation quote")
                date_start = date_end = None
                if (quote_start is not None and isinstance(date_quote, str)
                        and text.count(date_quote) > 1 and quote.count(date_quote) == 1):
                    date_start = quote_start + quote.index(date_quote)
                    date_end = date_start + len(date_quote)
                raw = {"feature_name": feature_name, "value": raw["value"], "evidence": quote,
                       "evidence_start": quote_start, "evidence_end": quote_end,
                       "recorded_at": _canonical_time_evidence(date_quote) if date_quote is not None else None,
                       "recorded_at_evidence": date_quote, "recorded_at_start": date_start, "recorded_at_end": date_end}
            evidence, evidence_start, evidence_end, evidence_offset_resolution = (
                _quote_span(
                    text=text,
                    quote=raw.get("evidence"),
                    reported_start=raw.get("evidence_start"),
                    reported_end=raw.get("evidence_end"),
                    label="evidence",
                )
            )
            recorded_at = _canonical_observation_time(raw.get("recorded_at"))
            recorded_at_evidence: str | None = None
            recorded_at_start: int | None = None
            recorded_at_end: int | None = None
            recorded_at_offset_resolution: str | None = None
            if recorded_at is not None:
                (
                    recorded_at_evidence,
                    recorded_at_start,
                    recorded_at_end,
                    recorded_at_offset_resolution,
                ) = _quote_span(
                    text=text,
                    quote=raw.get("recorded_at_evidence"),
                    reported_start=raw.get("recorded_at_start"),
                    reported_end=raw.get("recorded_at_end"),
                    label="recorded_at_evidence",
                )
                source_recorded_at = _canonical_time_evidence(recorded_at_evidence)
                if recorded_at != source_recorded_at:
                    raise ValueError(
                        "recorded_at is inconsistent with the supplied date text"
                    )
                recorded_at = source_recorded_at
            elif any(
                raw.get(key) is not None
                for key in (
                    "recorded_at_evidence",
                    "recorded_at_start",
                    "recorded_at_end",
                )
            ):
                raise ValueError(
                    "recorded_at evidence and offsets must be null when recorded_at is null"
                )

            retained_index = len(normalized) + 1
            identity_value = {
                "row_id": row_id,
                "feature_name": feature_name,
                "value": scalar,
                "page_index": page_index,
                "observation_index": retained_index,
                "recorded_at": recorded_at,
            }
            normalized.append(
                {
                    "observation_id": f"observation_{_value_fingerprint(identity_value)[:20]}",
                    "feature_name": feature_name,
                    "value": scalar,
                    "evidence": evidence,
                    "evidence_start": evidence_start,
                    "evidence_end": evidence_end,
                    "source_start": page_char_start + evidence_start if evidence_start is not None else None,
                    "source_end": page_char_start + evidence_end if evidence_end is not None else None,
                    "page_index": page_index,
                    "observation_index": retained_index,
                    "evidence_status": "model_reported",
                    "recorded_at": recorded_at,
                    "recorded_at_evidence": recorded_at_evidence,
                    "recorded_at_start": recorded_at_start,
                    "recorded_at_end": recorded_at_end,
                    "recorded_at_source_start": (
                        page_char_start + recorded_at_start
                        if recorded_at_start is not None
                        else None
                    ),
                    "recorded_at_source_end": (
                        page_char_start + recorded_at_end
                        if recorded_at_end is not None
                        else None
                    ),
                    "offset_resolution": evidence_offset_resolution,
                    "recorded_at_offset_resolution": recorded_at_offset_resolution,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                {
                    "observation_index": observation_index,
                    "feature_name": feature_name or None,
                    "reason": str(exc),
                    "raw_observation": copy.deepcopy(raw),
                }
            )

    by_id = {str(observation["observation_id"]): observation for observation in normalized}
    normalized_response = {
        "rows": [
            {
                "row_id": row_id,
                "observations": sorted(
                    by_id.values(),
                    key=lambda observation: int(observation["observation_index"]),
                ),
            }
        ]
    }
    if omitted_missing:
        request_audit.event("page_missing_observations_omitted", count=omitted_missing)
    if issues:
        raise _PageObservationValidationError(issues=issues, response=normalized_response)
    return normalized_response


def _request_validated_page_observations(
    *,
    messages: Sequence[Mapping[str, str]],
    page: Mapping[str, Any],
    definitions: Sequence[Mapping[str, Any]],
    request_json: RequestJSON,
    audit_dir: Path,
) -> dict[str, Any]:
    """Request mode observations, retaining valid values after exhausted repairs."""

    issue_path = audit_dir / "extraction_issues.json"
    try:
        with request_audit.context(
            _audit_path=str(audit_dir / "request_events.jsonl"),
            checkpoint_dir=str(audit_dir),
            patient_row_ids=[int(page["row_id"])],
            page_index=page.get("page_index"),
            feature_ids=[str(feature.get("feature_id") or feature["name"])
                         for feature in definitions],
        ):
            validated = request_json(
                messages,
                _StructuredExtractionValidator(
                    validate=lambda candidate: _validate_page_observations(
                        candidate, page=page, definitions=definitions),
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "stage2_page_observations", "strict": True,
                        "schema": _page_response_schema(definitions),
                    }},
                    repair_instructions=_page_output_instructions(),
                ),
                request_kind="extraction",
            )
        _write_json(
            issue_path,
            {
                "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                "completed_at": _now(),
                "events": [],
            },
        )
        return validated
    except (Stage2ResponseValidationError, _PageObservationValidationError) as exc:
        observation_error = _page_observation_error_from_exception(exc)
        row_id = int(page["row_id"])
        if observation_error is not None:
            events = [
                {
                    "failure_kind": "invalid_page_observation",
                    "row_id": row_id,
                    "feature_name": issue.get("feature_name"),
                    "reason": str(issue.get("reason") or ""),
                    "observation_index": int(issue.get("observation_index") or 0),
                }
                for issue in observation_error.issues
            ]
            _write_json(
                audit_dir / "invalid_page_observation_repair.json",
                {
                    "schema_version": "stage2_invalid_page_observation_repair_v1",
                    "resolution": "retain_valid_drop_invalid",
                    "original_validation_error": str(exc),
                    "issues": [dict(issue) for issue in observation_error.issues],
                },
            )
            _write_json(
                issue_path,
                {
                    "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                    "completed_at": _now(),
                    "events": events,
                },
            )
            LOGGER.warning(
                "Stage 2 page extraction retained valid observations and dropped %s "
                "invalid observation(s) for row %s",
                len(observation_error.issues),
                row_id,
            )
            return copy.deepcopy(observation_error.response)

        conservative = {"rows": [{"row_id": row_id, "observations": []}]}
        _write_json(
            audit_dir / "extraction_failure.json",
            {
                "schema_version": "stage2_page_observation_failure_v1",
                "resolution": "conservative_no_observations",
                "failed_at": _now(),
                "row_id": row_id,
                "feature_names": [str(definition["name"]) for definition in definitions],
                "validation_error": f"{type(exc).__name__}: {exc}",
            },
        )
        _write_json(
            issue_path,
            {
                "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                "completed_at": _now(),
                "events": [
                    {
                        "failure_kind": "structural_page_observation_failure",
                        "row_id": row_id,
                        "feature_name": None,
                        "feature_names": [str(definition["name"]) for definition in definitions],
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                ],
            },
        )
        LOGGER.warning(
            "Stage 2 page observation response remained structurally invalid; "
            "using no observations for row %s (%s: %s)",
            row_id,
            type(exc).__name__,
            exc,
        )
        return conservative


def _observation_value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _observation_time_sort_value(observation: Mapping[str, Any]) -> int | None:
    recorded_at = observation.get("recorded_at")
    if not recorded_at:
        return None
    timestamp = pd.Timestamp(str(recorded_at))
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.value)


def _observation_order(observation: Mapping[str, Any], *, latest: bool = False) -> tuple[int, int]:
    # Current observations use the requested source-order response sequence.
    # Historical observations may only carry absolute source offsets.
    if "observation_index" in observation:
        return int(observation["page_index"]), int(observation["observation_index"])
    return 0, int(observation["source_end" if latest else "source_start"])


def _select_temporal_observation(
    observations: Sequence[Mapping[str, Any]],
    *,
    latest: bool,
) -> tuple[dict[str, Any], str]:
    dated = [
        (observation, _observation_time_sort_value(observation))
        for observation in observations
        if observation.get("recorded_at")
    ]
    if dated:
        if latest:
            selected, _timestamp = max(
                dated,
                key=lambda item: (
                    int(item[1]),
                    _observation_order(item[0], latest=True),
                    str(item[0]["observation_id"]),
                ),
            )
        else:
            selected, _timestamp = min(
                dated,
                key=lambda item: (
                    int(item[1]),
                    _observation_order(item[0]),
                    str(item[0]["observation_id"]),
                ),
            )
        return dict(selected), "reported_recorded_at"
    selected = (max if latest else min)(
        observations,
        key=lambda observation: (
            _observation_order(observation, latest=latest),
            str(observation["observation_id"]),
        ),
    )
    return dict(selected), "reported_source_order"


def _resolve_feature_observations(
    *,
    definition: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    policy = _resolved_conflict_resolution(definition)
    unique = {
        str(observation["observation_id"]): dict(observation) for observation in observations
    }
    ordered = sorted(
        unique.values(),
        key=lambda observation: (
            _observation_order(observation),
            str(observation["observation_id"]),
        ),
    )
    decision: dict[str, Any] = {
        "feature_name": str(definition["name"]),
        "policy": policy,
        "observation_count": len(ordered),
        "distinct_value_count": len(
            {_observation_value_key(observation.get("value")) for observation in ordered}
        ),
        "observations": ordered,
        "selected_observation_id": None,
        "resolution": "no_observations",
        "value": None,
    }
    if not ordered:
        return None, decision
    # A documented value takes precedence over statements that the value is
    # unknown; documented_unknown is selected only when nothing else is.
    unknown = [o for o in ordered if is_documented_unknown(o.get("value"))]
    if unknown:
        decision["documented_unknown_observations"] = len(unknown)
        definite = [o for o in ordered if not is_documented_unknown(o.get("value"))]
        if not definite:
            selected, basis = _select_temporal_observation(unknown, latest=True)
            decision.update(
                {
                    "selected_observation_id": selected["observation_id"],
                    "resolution": "documented_unknown_only",
                    "selection_basis": basis,
                    "value": DOCUMENTED_UNKNOWN,
                }
            )
            return DOCUMENTED_UNKNOWN, decision
        ordered = definite

    values_by_key: dict[str, list[dict[str, Any]]] = {}
    for observation in ordered:
        values_by_key.setdefault(_observation_value_key(observation["value"]), []).append(
            observation
        )
    if len(values_by_key) == 1:
        selected, basis = _select_temporal_observation(ordered, latest=True)
        decision.update(
            {
                "selected_observation_id": selected["observation_id"],
                "resolution": "unanimous_value",
                "selection_basis": basis,
                "value": selected["value"],
            }
        )
        return selected["value"], decision

    strategy = str(policy["strategy"])
    selected: dict[str, Any] | None = None
    basis = ""
    if strategy in {"latest", "earliest"}:
        selected, basis = _select_temporal_observation(
            ordered,
            latest=strategy == "latest",
        )
    elif strategy in {"maximum", "minimum"}:
        numeric = [
            observation
            for observation in ordered
            if isinstance(observation.get("value"), (int, float))
            and not isinstance(observation.get("value"), bool)
        ]
        if numeric:
            target_value = (max if strategy == "maximum" else min)(
                float(observation["value"]) for observation in numeric
            )
            extrema = [
                observation
                for observation in numeric
                if float(observation["value"]) == target_value
            ]
            selected, temporal_basis = _select_temporal_observation(extrema, latest=True)
            basis = f"{strategy}_numeric_then_{temporal_basis}"
    elif strategy == "mode":
        largest_count = max(len(group) for group in values_by_key.values())
        modal = [
            observation
            for group in values_by_key.values()
            if len(group) == largest_count
            for observation in group
        ]
        selected, temporal_basis = _select_temporal_observation(modal, latest=True)
        basis = f"mode_then_{temporal_basis}"
    elif strategy == "any_positive":
        positive = str(policy["positive_category"])
        matching = [observation for observation in ordered if observation["value"] == positive]
        selected, temporal_basis = _select_temporal_observation(
            matching or ordered,
            latest=True,
        )
        basis = (
            f"any_positive_then_{temporal_basis}"
            if matching
            else f"no_positive_then_{temporal_basis}"
        )
    elif strategy == "single_or_null":
        basis = "conflicting_values_are_null"

    if selected is None:
        decision.update(
            {
                "resolution": "conflict_null",
                "selection_basis": basis or "no_valid_deterministic_selection",
            }
        )
        return None, decision
    decision.update(
        {
            "selected_observation_id": selected["observation_id"],
            "resolution": "selected_by_policy",
            "selection_basis": basis,
            "value": selected["value"],
        }
    )
    return selected["value"], decision


def _resolve_page_observations(
    *,
    definitions: Sequence[Mapping[str, Any]],
    page_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    observations_by_feature: dict[str, list[dict[str, Any]]] = {
        str(definition["name"]): [] for definition in definitions
    }
    for page_result in page_results:
        for observation in page_result.get("observations") or []:
            feature_name = str(observation.get("feature_name") or "")
            if feature_name not in observations_by_feature:
                raise ValueError("page result contains an observation for an unknown feature")
            observations_by_feature[feature_name].append(dict(observation))

    values: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for definition in definitions:
        feature_name = str(definition["name"])
        value, decision = _resolve_feature_observations(
            definition=definition,
            observations=observations_by_feature[feature_name],
        )
        values[feature_name] = value
        decisions[feature_name] = decision
    return values, decisions


def _preferred_lossless_page_end(source: str, *, start: int, hard_end: int) -> tuple[int, str]:
    """Prefer a nearby clinical-note or text boundary without dropping characters."""

    if hard_end >= len(source):
        return len(source), "document_end"
    width = hard_end - start
    if width <= 1:
        return hard_end, "hard_character_limit"
    minimum = start + max(1, int(width * 0.8))
    for separator, label in (
        ("\n\n<new_note>\n\n", "new_note_separator"),
        ("\n\n", "paragraph_boundary"),
        ("\n", "line_boundary"),
        (". ", "sentence_boundary"),
        (" ", "word_boundary"),
    ):
        position = source.rfind(separator, minimum, hard_end)
        if position >= minimum:
            return position + len(separator), label
    return hard_end, "hard_character_limit"


def _lossless_extraction_pages(
    row: Mapping[str, Any],
    *,
    definition_batches: Sequence[Sequence[Mapping[str, Any]]],
    max_prompt_chars: int,
    tokenizer: Any = None,
    input_token_budget: int | None = None,
    prompt_templates: Sequence[Sequence[Mapping[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    """Split one note into exact contiguous pages within character and token caps."""

    source = str(row.get("text") or "")
    row_id = int(row["row_id"])
    if not source:
        raise ValueError(
            "an empty Stage 2 row exceeded the prompt budget before note text was added; "
            "increase stage2.extraction_max_prompt_chars or shorten the feature definitions"
        )
    # The page prompt contains a fixed instruction/definition prefix, source
    # text, and output reminder. Prepare each prefix once per feature batch, rather
    # than rebuilding all clinical definitions at every binary-search step.
    if prompt_templates is None:
        prompt_templates = [
            _page_extraction_prompt_template(definitions=batch)
            for batch in definition_batches
        ]

    def fits(text: str) -> bool:
        for template in prompt_templates:
            messages = _render_page_extraction_prompt(template, text)
            if _prompt_chars(messages) > int(max_prompt_chars):
                return False
            if tokenizer is not None:
                if prompt_token_count(tokenizer, messages) > int(input_token_budget):
                    return False
        return True

    pages: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(source):
        # Most records fit in one page. Checking that first avoids repeatedly
        # tokenizing near-identical prefixes just to rediscover the document end.
        if fits(source[cursor:]):
            pages.append({
                "row_id": row_id,
                "text": source[cursor:],
                "page": {"page_index": len(pages) + 1, "char_start": cursor,
                         "char_end": len(source), "document_chars": len(source)},
            })
            cursor = len(source)
            break
        low = cursor + 1
        high = len(source) - 1
        best: dict[str, Any] | None = None
        while low <= high:
            end = (low + high) // 2
            candidate = {
                "row_id": row_id,
                "text": source[cursor:end],
                "page": {
                    "page_index": len(pages) + 1,
                    "char_start": cursor,
                    "char_end": end,
                    "document_chars": len(source),
                },
            }
            if fits(candidate["text"]):
                best = candidate
                low = end + 1
            else:
                high = end - 1
        if best is None:
            raise ValueError(
                "Stage 2 feature definitions and prompt envelope leave no room for even "
                "one source character; increase stage2.extraction_max_prompt_chars or "
                "shorten the feature definitions"
            )
        preferred_end, _boundary = _preferred_lossless_page_end(
            source,
            start=cursor,
            hard_end=int(best["page"]["char_end"]),
        )
        if preferred_end != int(best["page"]["char_end"]):
            best = {
                "row_id": row_id,
                "text": source[cursor:preferred_end],
                "page": {
                    "page_index": len(pages) + 1,
                    "char_start": cursor,
                    "char_end": preferred_end,
                    "document_chars": len(source),
                },
            }
        pages.append(best)
        cursor = int(best["page"]["char_end"])
    if "".join(str(page["text"]) for page in pages) != source:
        raise RuntimeError("Stage 2 lossless page planner changed patient text")
    return pages


def _serial_extraction_required(
    *,
    row: Mapping[str, Any],
    definitions: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    chunk_size_tokens: int,
    input_token_budget: int,
    max_prompt_chars: int,
) -> bool:
    """Return whether one patient/feature slice needs an ordered serial pass."""

    text = str(row.get("text") or "")
    messages = _extraction_prompt(definitions=definitions, rows=[row])
    if tokenizer is None:
        return _prompt_chars(messages) > int(max_prompt_chars)
    return (
        _text_token_count(tokenizer, text) > int(chunk_size_tokens)
        or prompt_token_count(tokenizer, messages) > int(input_token_budget)
        or _prompt_chars(messages) > int(max_prompt_chars)
    )


def _next_serial_extraction_chunk(
    *,
    source: str,
    cursor: int,
    row_id: int,
    definitions: Sequence[Mapping[str, Any]],
    prior_values: Mapping[str, Any],
    prior_feature_state: Mapping[str, Any],
    chunk_index: int,
    tokenizer: Any,
    chunk_size_tokens: int,
    input_token_budget: int,
    max_prompt_chars: int,
) -> dict[str, Any]:
    """Find the largest exact contiguous source prefix inside every prompt cap."""

    def candidate(end: int) -> dict[str, Any]:
        chunk_text = source[cursor:end]
        messages = _serial_extraction_prompt(
            definitions=definitions,
            row_id=row_id,
            chunk_text=chunk_text,
            prior_values=prior_values,
            prior_feature_state=prior_feature_state,
            chunk_index=chunk_index,
            char_start=cursor,
            char_end=end,
            document_chars=len(source),
        )
        return {
            "text": chunk_text,
            "char_start": int(cursor),
            "char_end": int(end),
            "source_tokens": _text_token_count(tokenizer, chunk_text) if tokenizer is not None else None,
            "prompt_tokens": prompt_token_count(tokenizer, messages) if tokenizer is not None else None,
            "prompt_chars": _prompt_chars(messages),
            "messages": messages,
        }

    def fits(value: Mapping[str, Any]) -> bool:
        return (
            (tokenizer is None or int(value["source_tokens"]) <= int(chunk_size_tokens))
            and (tokenizer is None or int(value["prompt_tokens"]) <= int(input_token_budget))
            and int(value["prompt_chars"]) <= int(max_prompt_chars)
        )

    low = int(cursor) + 1
    high = len(source)
    best: dict[str, Any] | None = None
    while low <= high:
        end = (low + high) // 2
        value = candidate(end)
        if fits(value):
            best = value
            low = end + 1
        else:
            high = end - 1
    if best is None:
        empty = candidate(cursor)
        raise ValueError(
            "Stage 2 serial extraction prompt leaves no room for one source character: "
            f"empty_prompt_tokens={empty['prompt_tokens']}, "
            f"input_token_budget={input_token_budget}, "
            f"empty_prompt_chars={empty['prompt_chars']}, "
            f"max_prompt_chars={max_prompt_chars}. Reduce the feature batch size or "
            "extraction_max_tokens, or increase the configured context window."
        )

    preferred_end, boundary = _preferred_lossless_page_end(
        source,
        start=cursor,
        hard_end=int(best["char_end"]),
    )
    if preferred_end != int(best["char_end"]):
        preferred = candidate(preferred_end)
        if fits(preferred):
            best = preferred
        else:  # Token counts can very rarely be non-monotonic at a BPE boundary.
            boundary = "hard_token_limit"
    else:
        boundary = "document_end" if preferred_end == len(source) else "hard_token_limit"
    best["boundary"] = boundary
    return best


def _serial_extract_feature_batch(
    *,
    parent_dir: Path,
    row: Mapping[str, Any],
    definitions: Sequence[Mapping[str, Any]],
    request_json: RequestJSON,
    request_identity: Mapping[str, Any],
    tokenizer: Any,
    chunk_size_tokens: int,
    context_window_tokens: int,
    max_output_tokens: int,
    context_margin_tokens: int,
    max_prompt_chars: int,
) -> dict[str, Any]:
    """Process one patient's feature slice serially with resumable carried state."""

    input_token_budget = (
        int(context_window_tokens)
        - int(max_output_tokens)
        - int(context_margin_tokens)
    )
    if input_token_budget < 1:
        raise ValueError(
            "Stage 2 extraction context window leaves no input-token budget after "
            "reserving the configured output ceiling and safety margin"
        )
    row_id = int(row["row_id"])
    source = str(row.get("text") or "")
    feature_names = [str(definition["name"]) for definition in definitions]
    prior_values: dict[str, Any] = {name: None for name in feature_names}
    prior_feature_state: dict[str, str | None] = {
        name: None for name in feature_names
    }
    serial_input = {
        "schema_version": SERIAL_EXTRACTION_MANIFEST_SCHEMA_VERSION,
        "request_identity": dict(request_identity),
        "definitions": _prompt_feature_definitions(definitions),
        "row_id": row_id,
        "document_chars": len(source),
        "document_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "chunk_size_tokens": int(chunk_size_tokens),
        "context_window_tokens": int(context_window_tokens),
        "max_output_tokens": int(max_output_tokens),
        "context_margin_tokens": int(context_margin_tokens),
        "input_token_budget": int(input_token_budget),
        "max_prompt_chars": int(max_prompt_chars),
        "token_budget_enforced": tokenizer is not None,
    }
    serial_fingerprint = _value_fingerprint(serial_input)
    if not source:
        raise ValueError(
            "an empty Stage 2 record exceeded the one-shot token envelope before "
            "clinical text was added; reduce the feature batch size or extraction output cap"
        )

    cursor = 0
    chunk_index = 1
    chunk_manifest: list[dict[str, Any]] = []
    while cursor < len(source):
        planned = _next_serial_extraction_chunk(
            source=source,
            cursor=cursor,
            row_id=row_id,
            definitions=definitions,
            prior_values=prior_values,
            prior_feature_state=prior_feature_state,
            chunk_index=chunk_index,
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            input_token_budget=input_token_budget,
            max_prompt_chars=max_prompt_chars,
        )
        chunk_dir = parent_dir / "serial_chunks" / f"chunk_{chunk_index:05d}"
        result_path = chunk_dir / "result.json"
        complete_path = chunk_dir / "complete.json"
        input_path = chunk_dir / "input.json"
        ontology_audit_path = chunk_dir / "category_ontology_repair.json"
        failure_path = chunk_dir / "extraction_failure.json"
        chunk_input = {
            "schema_version": SERIAL_EXTRACTION_CHUNK_CHECKPOINT_SCHEMA_VERSION,
            "serial_input_fingerprint": serial_fingerprint,
            "request_identity": dict(request_identity),
            "definitions": _prompt_feature_definitions(definitions),
            "row_id": row_id,
            "prior_values": copy.deepcopy(prior_values),
            "prior_feature_state": copy.deepcopy(prior_feature_state),
            "chunk": {
                "chunk_index": chunk_index,
                "char_start": int(planned["char_start"]),
                "char_end": int(planned["char_end"]),
                "document_chars": len(source),
                "source_tokens": planned["source_tokens"],
                "prompt_tokens": planned["prompt_tokens"],
                "prompt_chars": int(planned["prompt_chars"]),
                "boundary": str(planned["boundary"]),
                "text": str(planned["text"]),
            },
        }
        input_fingerprint = _value_fingerprint(chunk_input)
        stale_audit = _stale_category_ontology_audit(ontology_audit_path)
        result: dict[str, Any] | None = None
        if complete_path.is_file() and result_path.is_file() and stale_audit is None:
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    isinstance(completion, Mapping)
                    and completion.get("schema_version")
                    == SERIAL_EXTRACTION_CHUNK_CHECKPOINT_SCHEMA_VERSION
                    and completion.get("input_fingerprint") == input_fingerprint
                    and isinstance(cached, Mapping)
                ):
                    result = _validate_extraction(
                        cached,
                        row_ids=[row_id],
                        definitions=definitions,
                    )
                    result = _validate_serial_extraction(
                        result,
                        row_id=row_id,
                        definitions=definitions,
                    )
            except (
                KeyError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                result = None
            if result is None:
                LOGGER.info("rerun incompatible Stage 2 serial chunk: %s", chunk_dir)
        if result is None:
            same_incomplete_input = False
            if input_path.is_file():
                try:
                    prior_input = json.loads(input_path.read_text(encoding="utf-8"))
                    same_incomplete_input = (
                        isinstance(prior_input, Mapping)
                        and prior_input.get("input_fingerprint") == input_fingerprint
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            if not same_incomplete_input:
                for stale_name in (
                    "category_ontology_repair.json",
                    "extraction_failure.json",
                    "extraction_issues.json",
                    "invalid_feature_value_repair.json",
                    "pending_category_ontology.json",
                ):
                    (chunk_dir / stale_name).unlink(missing_ok=True)
            chunk_dir.mkdir(parents=True, exist_ok=True)
            _write_json(input_path, {**chunk_input, "input_fingerprint": input_fingerprint})
            result = _request_validated_extraction(
                messages=planned["messages"],
                row_ids=[row_id],
                definitions=definitions,
                request_json=request_json,
                ontology_audit_path=ontology_audit_path,
                messages_for_definitions=lambda subset: _serial_extraction_prompt(
                    definitions=subset, row_id=row_id, chunk_text=str(planned["text"]),
                    prior_values=prior_values, prior_feature_state=prior_feature_state,
                    chunk_index=chunk_index, char_start=int(planned["char_start"]),
                    char_end=int(planned["char_end"]), document_chars=len(source),
                ),
                validate_response=lambda value, *, definitions: _validate_serial_extraction(
                    value,
                    row_id=row_id,
                    definitions=definitions,
                ),
                prior_response={"rows": [{"row_id": row_id, "values": prior_values,
                    "carry_forward_state": prior_feature_state}]},
                response_schema=_serial_response_schema,
                response_instructions=_serial_output_instructions,
            )
            if failure_path.is_file():
                # A malformed later response must not erase validated state from
                # earlier chunks. The failure remains in its audit ledger.
                result = _validate_serial_extraction(
                    {
                        "rows": [
                            {
                                "row_id": row_id,
                                "values": prior_values,
                                "carry_forward_state": prior_feature_state,
                            }
                        ]
                    },
                    row_id=row_id,
                    definitions=definitions,
                )
            _write_json(result_path, result)
            _supersede_stale_category_ontology_audit(
                ontology_audit_path,
                previous=stale_audit,
            )
            _write_json(
                complete_path,
                {
                    "status": "complete",
                    "schema_version": SERIAL_EXTRACTION_CHUNK_CHECKPOINT_SCHEMA_VERSION,
                    "input_fingerprint": input_fingerprint,
                    "completed_at": _now(),
                    "row_id": row_id,
                    "chunk_index": chunk_index,
                    "char_start": int(planned["char_start"]),
                    "char_end": int(planned["char_end"]),
                    "source_tokens": planned["source_tokens"],
                    "prompt_tokens": planned["prompt_tokens"],
                    "structural_failure_carried_prior_state": failure_path.is_file(),
                },
            )
        prior_values = dict(result["rows"][0]["values"])
        prior_feature_state = dict(result["rows"][0]["carry_forward_state"])
        chunk_manifest.append(
            {
                "chunk_index": chunk_index,
                "char_start": int(planned["char_start"]),
                "char_end": int(planned["char_end"]),
                "source_tokens": planned["source_tokens"],
                "prompt_tokens": planned["prompt_tokens"],
                "boundary": str(planned["boundary"]),
                "input_fingerprint": input_fingerprint,
            }
        )
        cursor = int(planned["char_end"])
        chunk_index += 1

    result = _validate_extraction(
        {"rows": [{"row_id": row_id, "values": prior_values}]},
        row_ids=[row_id],
        definitions=definitions,
    )
    lossless_source_coverage = (
        bool(chunk_manifest)
        and int(chunk_manifest[0]["char_start"]) == 0
        and int(chunk_manifest[-1]["char_end"]) == len(source)
        and all(
            int(left["char_end"]) == int(right["char_start"])
            for left, right in zip(chunk_manifest, chunk_manifest[1:])
        )
    )
    if not lossless_source_coverage:  # pragma: no cover - planner invariant
        raise RuntimeError("Stage 2 serial extraction did not cover the source contiguously")
    _write_json(
        parent_dir / "serial_extraction.json",
        {
            **serial_input,
            "input_fingerprint": serial_fingerprint,
            "chunks": chunk_manifest,
            "lossless_source_coverage": lossless_source_coverage,
        },
    )
    _write_json(
        parent_dir / "serial_complete.json",
        {
            "status": "complete",
            "schema_version": SERIAL_EXTRACTION_MANIFEST_SCHEMA_VERSION,
            "input_fingerprint": serial_fingerprint,
            "completed_at": _now(),
            "row_id": row_id,
            "chunks": len(chunk_manifest),
            "document_chars": len(source),
        },
    )
    _write_json(
        parent_dir / "extraction_issues.json",
        {
            "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
            "completed_at": _now(),
            "events": [],
            "delegated_to_serial_chunks": True,
        },
    )
    return result


def _summarize_extraction_failures(
    *,
    output_dir: Path,
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate feature-attributable failures across distinct patients."""

    definition_names = {str(definition["name"]) for definition in definitions}
    patterns: dict[tuple[str, str, str], dict[str, Any]] = {}
    structural_rows: set[int] = set()
    # Every issue event is written only after the request's repairs are
    # exhausted. Cells named by an event are candidate extraction failures;
    # the mask keeps those whose final value is null.
    terminal_cells: dict[str, set[int]] = {}
    # A split run supersedes the old all-variable page path. Historical failed
    # quote checks must not enter the new measurement-quality review.
    issue_root = output_dir / "by_strategy" if (output_dir / "extraction_routing.json").is_file() else output_dir
    issue_files = sorted(issue_root.rglob("extraction_issues.json"))
    for path in issue_files:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            audit = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(audit, Mapping):
            continue
        for raw_event in audit.get("events") or []:
            if not isinstance(raw_event, Mapping):
                continue
            try:
                row_id = int(raw_event["row_id"])
            except (KeyError, TypeError, ValueError):
                continue
            feature_name = str(raw_event.get("feature_name") or "")
            failure_kind = str(raw_event.get("failure_kind") or "unknown")
            reason = str(raw_event.get("reason") or "")
            if not feature_name:
                structural_rows.add(row_id)
                covered = raw_event.get("feature_names")
                covered_names = (
                    {str(name) for name in covered}
                    if isinstance(covered, list)
                    else set(definition_names)
                )
                for name in covered_names & definition_names:
                    terminal_cells.setdefault(name, set()).add(row_id)
                continue
            if feature_name not in definition_names:
                continue
            terminal_cells.setdefault(feature_name, set()).add(row_id)
            signature_reason = reason if failure_kind == "invalid_scalar_or_value_type" else ""
            key = (feature_name, failure_kind, signature_reason)
            pattern = patterns.setdefault(
                key,
                {
                    "feature_name": feature_name,
                    "failure_kind": failure_kind,
                    "reason": reason,
                    "patient_row_ids": set(),
                    "example_values": [],
                    "allowed_categories": list(raw_event.get("allowed_categories") or []),
                },
            )
            pattern["patient_row_ids"].add(row_id)
            if "prior_extracted_value" in raw_event:
                example = copy.deepcopy(raw_event.get("prior_extracted_value"))
                example_key = json.dumps(
                    example,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                existing_keys = {
                    json.dumps(
                        value,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    for value in pattern["example_values"]
                }
                if example_key not in existing_keys and len(pattern["example_values"]) < 12:
                    pattern["example_values"].append(example)

    rendered_patterns = []
    for pattern in patterns.values():
        row_ids = sorted(pattern.pop("patient_row_ids"))
        rendered_patterns.append(
            {
                **pattern,
                "patient_count": len(row_ids),
                "patient_row_ids": row_ids,
            }
        )
    rendered_patterns.sort(
        key=lambda pattern: (
            -int(pattern["patient_count"]),
            str(pattern["feature_name"]),
            str(pattern["failure_kind"]),
            str(pattern["reason"]),
        )
    )
    summary = {
        "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
        "completed_at": _now(),
        "issue_files": len(issue_files),
        "feature_failure_patterns": rendered_patterns,
        "structural_failure_patient_count": len(structural_rows),
        "structural_failure_patient_row_ids": sorted(structural_rows),
        "terminal_failure_cells": {
            name: sorted(rows) for name, rows in sorted(terminal_cells.items())
        },
        "terminal_failure_policy": extraction_failures.TERMINAL_FAILURE_POLICY,
    }
    _write_json(output_dir / "failure_summary.json", summary)
    return summary


def extract_rows(
    *,
    dataset: pd.DataFrame,
    row_ids: Sequence[int],
    text_column: str,
    definitions: Sequence[Mapping[str, Any]],
    output_dir: Path,
    request_json: RequestJSON,
    workers: int,
    max_prompt_chars: int,
    feature_batch_size: int = DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE,
    request_identity: Mapping[str, Any] | None = None,
    tokenizer: Any | None = None,
    chunk_size_tokens: int = DEFAULT_EXTRACTION_CHUNK_SIZE_TOKENS,
    context_window_tokens: int = DEFAULT_EXTRACTION_CONTEXT_WINDOW_TOKENS,
    max_output_tokens: int = DEFAULT_EXTRACTION_MAX_TOKENS,
    context_margin_tokens: int = DEFAULT_EXTRACTION_CONTEXT_MARGIN_TOKENS,
    deferred_retry_passes: int = 1,
    _source_rows: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Extract patient/feature batches by full-record LLM reading."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if (isinstance(deferred_retry_passes, bool) or not isinstance(deferred_retry_passes, int)
            or deferred_retry_passes < 0):
        raise ValueError("deferred_retry_passes must be a nonnegative integer")
    mode_definitions = [d for d in definitions if _resolved_conflict_resolution(d)["strategy"] == "mode"]
    value_definitions = [d for d in definitions if _resolved_conflict_resolution(d)["strategy"] != "mode"]
    if mode_definitions and value_definitions:
        # Use disjoint feature sets and checkpoint trees. Each child retains
        # patient/chunk recovery; both share the caller's request admission cap.
        _write_json(output_dir / "extraction_routing.json", {
            "schema_version": EXTRACTION_ROUTING_SCHEMA_VERSION,
            "mode_features": [d["name"] for d in mode_definitions],
            "value_features": [d["name"] for d in value_definitions],
            "quotation_policy": "mode_only_advisory",
        })
        LOGGER.info("Stage 2 extraction routing root=%s scalar_features=%s mode_features=%s quotation_policy=mode_only_advisory",
                    output_dir, len(value_definitions), len(mode_definitions))
        cancelled = threading.Event()
        request_slots = threading.BoundedSemaphore(max(1, int(workers)))

        def split_request(*args, **kwargs):
            # Both strategies can use the complete pool as work becomes ready.
            # Preserve the caller's limit even without a global LLM admission
            # wrapper (e.g. standalone extraction and custom request clients).
            with request_slots:
                if cancelled.is_set():
                    raise _ExtractionCancelledError("Sibling extraction strategy failed")
                return request_json(*args, **kwargs)

        def extract_group(name, group, group_workers):
            return extract_rows(
                dataset=dataset, row_ids=row_ids, text_column=text_column,
                definitions=group, output_dir=output_dir / "by_strategy" / name,
                _source_rows=_source_rows,
                request_json=split_request, workers=group_workers,
                max_prompt_chars=max_prompt_chars, feature_batch_size=feature_batch_size,
                request_identity=request_identity, tokenizer=tokenizer,
                chunk_size_tokens=chunk_size_tokens, context_window_tokens=context_window_tokens,
                max_output_tokens=max_output_tokens, context_margin_tokens=context_margin_tokens,
                deferred_retry_passes=deferred_retry_passes,
            )

        frames = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, max(1, int(workers)))) as pool:
            futures = [pool.submit(extract_group, name, group, max(1, int(workers)))
                       for name, group in (("values", value_definitions), ("mode", mode_definitions))]
            try:
                for future in concurrent.futures.as_completed(futures):
                    frames.append(future.result())
            except BaseException:
                cancelled.set()
                for future in futures:
                    future.cancel()
                raise
        frame = frames[0].merge(frames[1], on="_oci_row_id", validate="one_to_one")
        frame = frame.set_index("_oci_row_id").loc[list(row_ids)].reset_index()
        frame = frame[["_oci_row_id", *[str(d["name"]) for d in definitions]]]
        _write_frame(output_dir / "extracted.csv", frame)
        summary = _summarize_extraction_failures(output_dir=output_dir, definitions=definitions)
        _write_json(output_dir / "complete.json", {
            "status": "complete", "schema_version": EXTRACTION_ROUTING_SCHEMA_VERSION,
            "completed_at": _now(), "rows": len(frame), "features": len(definitions),
            "mode_features": len(mode_definitions), "value_features": len(value_definitions),
            "feature_failure_patterns": len(summary["feature_failure_patterns"]),
            "structural_failure_patients": summary["structural_failure_patient_count"],
        })
        return frame
    cancellation = threading.Event()

    def guarded_request_json(
        messages: Sequence[Mapping[str, str]],
        validate: Callable[[Mapping[str, Any]], dict[str, Any]],
        *,
        request_kind: str = "interpretation",
        initial_reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        if cancellation.is_set():
            raise _ExtractionCancelledError(
                "Stage 2 extraction cancelled after a sibling task failed"
            )
        return request_json(
            messages,
            validate,
            request_kind=request_kind,
            **({"initial_reasoning_effort": initial_reasoning_effort}
               if initial_reasoning_effort is not None else {}),
        )
    extraction_request_identity = dict(request_identity or {})
    feature_names = [str(feature["name"]) for feature in definitions]
    definition_batches = _partition_feature_definitions(
        definitions,
        feature_batch_size=feature_batch_size,
    )
    if not definitions:
        frame = pd.DataFrame({"_oci_row_id": [int(value) for value in row_ids]})
        _write_frame(output_dir / "extracted.csv", frame)
        failure_summary = _summarize_extraction_failures(
            output_dir=output_dir,
            definitions=definitions,
        )
        _write_json(
            output_dir / "complete.json",
            {
                "status": "complete",
                "rows": len(frame),
                "feature_failure_patterns": len(failure_summary["feature_failure_patterns"]),
                "structural_failure_patients": failure_summary["structural_failure_patient_count"],
            },
        )
        return frame

    request_rows = list(_source_rows) if _source_rows is not None else [
        {
            "row_id": int(row_id),
            "text": (
                ""
                if pd.isna(dataset.iloc[int(row_id)][text_column])
                else str(dataset.iloc[int(row_id)][text_column])
            ),
        }
        for row_id in row_ids
    ]
    extraction_definitions = _prompt_feature_definitions(definitions)
    page_extraction_definitions = _page_prompt_feature_definitions(definitions)
    requires_page_observations = any(
        _resolved_conflict_resolution(d)["strategy"] == "mode" for d in definitions
    )
    if tokenizer is not None:
        if int(chunk_size_tokens) < 1:
            raise ValueError("chunk_size_tokens must be positive")
        if int(context_margin_tokens) < 0:
            raise ValueError("context_margin_tokens must be nonnegative")
        if int(context_window_tokens) - int(max_output_tokens) - int(
            context_margin_tokens
        ) < 1:
            raise ValueError(
                "serial extraction context window leaves no input-token budget"
            )
        # Token-aware serial extraction supersedes the older character-page
        # fallback. Every patient remains one independent outer task; chunks
        # within that task execute strictly in source order.
        batches = [[row] for row in request_rows]
        oversized_rows: list[Mapping[str, Any]] = []
    else:
        # Character-only callers also use scalar carried state for long records.
        # Quotations are reserved for mode variables, regardless of note length.
        batches = [[row] for row in request_rows]
        oversized_rows = []

    if requires_page_observations:
        batches = [[row] for row in request_rows if not row["text"]]
        oversized_rows = [row for row in request_rows if row["text"]]

    # Plan within each patient task so extraction can begin before the entire
    # cohort has been paginated. Only immutable prompt prefixes are shared.
    page_prompt_templates = [
        _page_extraction_prompt_template(definitions=batch)
        for batch in definition_batches
    ] if oversized_rows else []

    if len(definition_batches) > 1:
        LOGGER.info(
            "Stage 2 extraction features=%s feature_batch_size=%s "
            "feature_batches_per_patient=%s patients=%s paged_patients=%s; "
            "page planning runs incrementally within patient workers",
            len(definitions),
            feature_batch_size,
            len(definition_batches),
            len(request_rows),
            len(oversized_rows),
        )

    def request_feature_batches(
        *,
        parent_dir: Path,
        row: Mapping[str, Any],
        parent_schema_version: str,
    ) -> dict[str, Any]:
        """Extract and checkpoint each feature slice for one patient or page."""

        if len(definition_batches) < 2:  # pragma: no cover - caller invariant
            raise RuntimeError("feature-batched extraction requires multiple feature batches")
        row_id = int(row["row_id"])
        merged_values: dict[str, Any] = {}
        for feature_batch_index, batch_definitions in enumerate(
            definition_batches,
            start=1,
        ):
            feature_dir = parent_dir / "feature_batches" / f"batch_{feature_batch_index:05d}"
            result_path = feature_dir / "result.json"
            complete_path = feature_dir / "complete.json"
            ontology_audit_path = feature_dir / "category_ontology_repair.json"
            batch_input = {
                "schema_version": EXTRACTION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION,
                "parent_schema_version": parent_schema_version,
                "request_identity": extraction_request_identity,
                "definitions": _prompt_feature_definitions(batch_definitions),
                "row": dict(row),
            }
            input_fingerprint = _value_fingerprint(batch_input)
            stale_audit = _stale_category_ontology_audit(ontology_audit_path)
            result: dict[str, Any] | None = None
            if complete_path.is_file() and result_path.is_file() and stale_audit is None:
                try:
                    completion = json.loads(complete_path.read_text(encoding="utf-8"))
                    cached = json.loads(result_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(completion, Mapping)
                        and completion.get("schema_version")
                        == EXTRACTION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION
                        and completion.get("input_fingerprint") == input_fingerprint
                        and isinstance(cached, Mapping)
                    ):
                        result = _validate_extraction(
                            cached,
                            row_ids=[row_id],
                            definitions=batch_definitions,
                        )
                except (
                    KeyError,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    result = None
                if result is None:
                    LOGGER.info(
                        "rerun incompatible Stage 2 extraction feature batch: %s",
                        feature_dir,
                    )
            if result is None:
                if stale_audit is not None:
                    LOGGER.info(
                        "retry stale Stage 2 category ontology feature batch: %s",
                        feature_dir,
                    )
                feature_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    feature_dir / "input.json",
                    {**batch_input, "input_fingerprint": input_fingerprint},
                )
                use_serial = _serial_extraction_required(
                    row=row,
                    definitions=batch_definitions,
                    tokenizer=tokenizer,
                    chunk_size_tokens=int(chunk_size_tokens),
                    input_token_budget=(
                        int(context_window_tokens)
                        - int(max_output_tokens)
                        - int(context_margin_tokens)
                    ),
                    max_prompt_chars=int(max_prompt_chars),
                )
                if use_serial:
                    result = _serial_extract_feature_batch(
                        parent_dir=feature_dir,
                        row=row,
                        definitions=batch_definitions,
                        request_json=guarded_request_json,
                        request_identity=extraction_request_identity,
                        tokenizer=tokenizer,
                        chunk_size_tokens=int(chunk_size_tokens),
                        context_window_tokens=int(context_window_tokens),
                        max_output_tokens=int(max_output_tokens),
                        context_margin_tokens=int(context_margin_tokens),
                        max_prompt_chars=int(max_prompt_chars),
                    )
                else:
                    messages = _extraction_prompt(
                        definitions=batch_definitions,
                        rows=[row],
                    )
                    if _prompt_chars(messages) > int(max_prompt_chars):  # pragma: no cover
                        raise RuntimeError(
                            "Stage 2 extraction planner emitted an oversized feature batch"
                        )
                    result = _request_validated_extraction(
                        messages=messages,
                        row_ids=[row_id],
                        definitions=batch_definitions,
                        request_json=guarded_request_json,
                        ontology_audit_path=ontology_audit_path,
                        messages_for_definitions=lambda subset: _extraction_prompt(
                            definitions=subset, rows=[row],
                        ),
                    )
                _write_json(result_path, result)
                _supersede_stale_category_ontology_audit(
                    ontology_audit_path,
                    previous=stale_audit,
                )
                _write_json(
                    complete_path,
                    {
                        "status": "complete",
                        "schema_version": (EXTRACTION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION),
                        "input_fingerprint": input_fingerprint,
                        "completed_at": _now(),
                        "row_id": row_id,
                        "features": len(batch_definitions),
                        "feature_batch": feature_batch_index,
                    },
                )
            merged_values.update(dict(result["rows"][0]["values"]))

        merged = _validate_extraction(
            {"rows": [{"row_id": row_id, "values": merged_values}]},
            row_ids=[row_id],
            definitions=definitions,
        )
        # Any parent-level issue ledger belongs to an older all-feature request.
        # Active issue ledgers now live beside the feature-slice checkpoints.
        _write_json(
            parent_dir / "extraction_issues.json",
            {
                "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                "completed_at": _now(),
                "events": [],
                "delegated_to_feature_batches": True,
            },
        )
        return merged

    def request_page_feature_batches(
        *,
        parent_dir: Path,
        page: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Extract provenance observations for each feature slice of one page."""

        if len(definition_batches) < 2:  # pragma: no cover - caller invariant
            raise RuntimeError("page feature batching requires multiple feature batches")
        row_id = int(page["row_id"])
        merged_observations: list[dict[str, Any]] = []
        for feature_batch_index, batch_definitions in enumerate(
            definition_batches,
            start=1,
        ):
            feature_dir = parent_dir / "feature_batches" / f"batch_{feature_batch_index:05d}"
            result_path = feature_dir / "result.json"
            complete_path = feature_dir / "complete.json"
            batch_input = {
                "schema_version": PAGE_OBSERVATION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION,
                "parent_schema_version": PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                "request_identity": extraction_request_identity,
                "definitions": _page_prompt_feature_definitions(batch_definitions),
                "page": dict(page),
            }
            input_fingerprint = _value_fingerprint(batch_input)
            result: dict[str, Any] | None = None
            if complete_path.is_file() and result_path.is_file():
                try:
                    completion = json.loads(complete_path.read_text(encoding="utf-8"))
                    cached = json.loads(result_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(completion, Mapping)
                        and completion.get("schema_version")
                        == PAGE_OBSERVATION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION
                        and completion.get("input_fingerprint") == input_fingerprint
                        and isinstance(cached, Mapping)
                    ):
                        result = _validate_page_observations(
                            cached,
                            page=page,
                            definitions=batch_definitions,
                        )
                except (
                    KeyError,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    result = None
                if result is None:
                    LOGGER.info(
                        "rerun incompatible Stage 2 page observation feature batch: %s",
                        feature_dir,
                    )
            if result is None:
                feature_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    feature_dir / "input.json",
                    {**batch_input, "input_fingerprint": input_fingerprint},
                )
                messages = _page_extraction_prompt(
                    definitions=batch_definitions,
                    row=page,
                )
                if _prompt_chars(messages) > int(max_prompt_chars):  # pragma: no cover
                    raise RuntimeError(
                        "Stage 2 page observation planner emitted an oversized feature batch"
                    )
                result = _request_validated_page_observations(
                    messages=messages,
                    page=page,
                    definitions=batch_definitions,
                    request_json=guarded_request_json,
                    audit_dir=feature_dir,
                )
                _write_json(result_path, result)
                _write_json(
                    complete_path,
                    {
                        "status": "complete",
                        "schema_version": (
                            PAGE_OBSERVATION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION
                        ),
                        "input_fingerprint": input_fingerprint,
                        "completed_at": _now(),
                        "row_id": row_id,
                        "features": len(batch_definitions),
                        "feature_batch": feature_batch_index,
                        "observations": len(result["rows"][0]["observations"]),
                    },
                )
            merged_observations.extend(result["rows"][0]["observations"])

        merged = _validate_page_observations(
            {"rows": [{"row_id": row_id, "observations": merged_observations}]},
            page=page,
            definitions=definitions,
        )
        _write_json(
            parent_dir / "extraction_issues.json",
            {
                "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                "completed_at": _now(),
                "events": [],
                "delegated_to_page_feature_batches": True,
            },
        )
        return merged

    def run_batch(index: int, batch: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if len(batch) != 1:  # pragma: no cover - enforced by the planner
            raise RuntimeError("Stage 2 extraction planner created a multi-patient batch")
        batch_dir = output_dir / "batches" / f"batch_{index:05d}"
        result_path = batch_dir / "result.json"
        complete_path = batch_dir / "complete.json"
        manifest_path = batch_dir / "row_ids.json"
        ontology_audit_path = batch_dir / "category_ontology_repair.json"
        row_ids = [int(row["row_id"]) for row in batch]
        input_fingerprint = _value_fingerprint(
            {
                "schema_version": EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                "request_identity": extraction_request_identity,
                "definitions": extraction_definitions,
                "rows": list(batch),
            }
        )
        stale_audit = _stale_category_ontology_audit(ontology_audit_path)
        if (
            stale_audit is None
            and complete_path.is_file()
            and result_path.is_file()
            and manifest_path.is_file()
        ):
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                saved_row_ids = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    isinstance(completion, Mapping)
                    and completion.get("status") == "complete"
                    and completion.get("schema_version") == EXTRACTION_CHECKPOINT_SCHEMA_VERSION
                    and completion.get("input_fingerprint") == input_fingerprint
                    and isinstance(cached, Mapping)
                    and isinstance(saved_row_ids, list)
                    and [int(value) for value in saved_row_ids] == row_ids
                ):
                    return list(
                        _validate_extraction(cached, row_ids=row_ids, definitions=definitions)["rows"]
                    )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass

        if complete_path.is_file() or result_path.is_file():
            LOGGER.info("rerun incompatible Stage 2 extraction checkpoint: %s", batch_dir)
        if stale_audit is not None:
            LOGGER.info("retry stale Stage 2 category ontology batch: %s", batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        _write_json(batch_dir / "row_ids.json", row_ids)
        if len(definition_batches) == 1:
            use_serial = _serial_extraction_required(
                row=batch[0],
                definitions=definitions,
                tokenizer=tokenizer,
                chunk_size_tokens=int(chunk_size_tokens),
                input_token_budget=(
                    int(context_window_tokens)
                    - int(max_output_tokens)
                    - int(context_margin_tokens)
                ),
                max_prompt_chars=int(max_prompt_chars),
            )
            if use_serial:
                result = _serial_extract_feature_batch(
                    parent_dir=batch_dir,
                    row=batch[0],
                    definitions=definitions,
                    request_json=guarded_request_json,
                    request_identity=extraction_request_identity,
                    tokenizer=tokenizer,
                    chunk_size_tokens=int(chunk_size_tokens),
                    context_window_tokens=int(context_window_tokens),
                    max_output_tokens=int(max_output_tokens),
                    context_margin_tokens=int(context_margin_tokens),
                    max_prompt_chars=int(max_prompt_chars),
                )
            else:
                messages = _extraction_prompt(
                    definitions=definitions,
                    rows=batch,
                )
                result = _request_validated_extraction(
                    messages=messages,
                    row_ids=row_ids,
                    definitions=definitions,
                    request_json=guarded_request_json,
                    ontology_audit_path=ontology_audit_path,
                    messages_for_definitions=lambda subset: _extraction_prompt(
                        definitions=subset, rows=batch,
                    ),
                )
                if _prompt_chars(messages) > int(max_prompt_chars):  # pragma: no cover
                    raise RuntimeError("Stage 2 extraction planner emitted an oversized batch")
        else:
            result = request_feature_batches(
                parent_dir=batch_dir,
                row=batch[0],
                parent_schema_version=EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
            )
        _write_json(result_path, result)
        _supersede_stale_category_ontology_audit(
            ontology_audit_path,
            previous=stale_audit,
        )
        _write_json(
            complete_path,
            {
                "status": "complete",
                "schema_version": EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "completed_at": _now(),
                "rows": 1,
                "features": len(definitions),
                "feature_batches": len(definition_batches),
                "feature_batch_size": feature_batch_size,
            },
        )
        return list(result["rows"])

    def run_page(page: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        page_meta = dict(page["page"])
        row_id = int(page["row_id"])
        page_index = int(page_meta["page_index"])
        page_dir = output_dir / "pages" / f"row_{row_id:08d}" / f"page_{page_index:05d}"
        result_path = page_dir / "result.json"
        complete_path = page_dir / "complete.json"
        ontology_audit_path = page_dir / "category_ontology_repair.json"
        input_fingerprint = _value_fingerprint(
            {
                "schema_version": PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                "request_identity": extraction_request_identity,
                "definitions": page_extraction_definitions,
                "page": dict(page),
            }
        )
        stale_audit = _stale_category_ontology_audit(ontology_audit_path)
        if complete_path.is_file() and result_path.is_file() and stale_audit is None:
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
                stored = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    completion.get("schema_version") == PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION
                    and completion.get("input_fingerprint") == input_fingerprint
                ):
                    validated = _validate_page_observations(
                        stored,
                        page=page,
                        definitions=definitions,
                    )
                    return page_meta, dict(validated["rows"][0])
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            LOGGER.info("rerun incompatible Stage 2 extraction page: %s", page_dir)
        if stale_audit is not None:
            LOGGER.info("retry stale Stage 2 category ontology page: %s", page_dir)
        page_dir.mkdir(parents=True, exist_ok=True)
        _write_json(page_dir / "page.json", page_meta)
        if len(definition_batches) == 1:
            messages = _page_extraction_prompt(
                definitions=definitions,
                row=page,
            )
            if _prompt_chars(messages) > int(max_prompt_chars):  # pragma: no cover
                raise RuntimeError(
                    "Stage 2 page observation planner emitted an oversized page"
                )
            result = _request_validated_page_observations(
                messages=messages,
                page=page,
                definitions=definitions,
                request_json=guarded_request_json,
                audit_dir=page_dir,
            )
        else:
            result = request_page_feature_batches(
                parent_dir=page_dir,
                page=page,
            )
        _write_json(result_path, result)
        _supersede_stale_category_ontology_audit(
            ontology_audit_path,
            previous=stale_audit,
        )
        _write_json(
            complete_path,
            {
                "status": "complete",
                "schema_version": PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "completed_at": _now(),
                "features": len(definitions),
                "feature_batches": len(definition_batches),
                "feature_batch_size": feature_batch_size,
                "observations": len(result["rows"][0]["observations"]),
                **page_meta,
            },
        )
        return page_meta, dict(result["rows"][0])

    def run_paged_row(row: Mapping[str, Any]):
        if cancellation.is_set():
            raise _ExtractionCancelledError("Stage 2 extraction cancelled before page planning")
        row_id = int(row["row_id"])
        planning_path = output_dir / "pages" / f"row_{row_id:08d}" / "planning_status.json"
        started = time.monotonic()
        _write_json(planning_path, {"status": "planning", "row_id": row_id,
                                   "started_at": _now(), "document_chars": len(row["text"])})
        pages = _lossless_extraction_pages(
            row, definition_batches=definition_batches,
            max_prompt_chars=int(max_prompt_chars), tokenizer=tokenizer,
            input_token_budget=int(context_window_tokens) - int(max_output_tokens) - int(context_margin_tokens),
            prompt_templates=page_prompt_templates,
        )
        duration = round(time.monotonic() - started, 3)
        _write_json(planning_path, {"status": "complete", "row_id": row_id,
                                   "completed_at": _now(), "pages": len(pages),
                                   "duration_seconds": duration})
        LOGGER.info("Stage 2 extraction pages planned root=%s row_id=%s pages=%s seconds=%.3f",
                    output_dir, row_id, len(pages), duration)
        # Page/feature checkpoints keep partial patients reusable on a retry.
        return [run_page(page) for page in pages]

    completed: list[tuple[int, list[dict[str, Any]]]] = []
    completed_pages: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    tasks = [("batch", index, batch) for index, batch in enumerate(batches, start=1)]
    tasks += [("paged_row", int(row["row_id"]), row) for row in oversized_rows]
    task_count = len(tasks)
    failure_path = output_dir / "deferred_extraction.json"
    failures = (list(json.loads(failure_path.read_text(encoding="utf-8"))["failures"])
                if failure_path.exists() else [])
    active_task = None

    def task_label(task):
        kind, index, payload = task
        return {"kind": kind, "batch": index if kind == "batch" else None,
                "row_ids": ([int(row["row_id"]) for row in payload] if kind == "batch"
                            else [int(payload["row_id"])]),
                "page_index": payload.get("page_index") if kind == "page" else None}

    def save_deferred(status, retry_pass, unresolved):
        _write_json(failure_path, {
            "schema_version": "stage2_deferred_extraction_v1", "updated_at": _now(),
            "status": status, "retry_pass": retry_pass,
            "maximum_retry_passes": deferred_retry_passes,
            "failures": failures, "unresolved": [task_label(task) for task in unresolved],
        })

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(int(workers), max(1, task_count)))
    ) as executor:
        task_futures = {}
        try:
            for retry_pass in range(deferred_retry_passes + 1):
                task_futures = {
                    (executor.submit(run_batch, index, payload) if kind == "batch"
                     else executor.submit(run_paged_row, payload)): (kind, index, payload)
                    for kind, index, payload in tasks
                }
                deferred = []
                consecutive_failures = 0
                for future in concurrent.futures.as_completed(task_futures):
                    active_task = task_futures[future]
                    kind, index, _payload = active_task
                    try:
                        result = future.result()
                    except Stage2RequestExhaustedError as error:
                        if deferred_retry_passes == 0:
                            raise
                        deferred.append((active_task, error))
                        consecutive_failures += 1
                        failures.append({**task_label(active_task), "retry_pass": retry_pass,
                                         "failed_at": _now(), "error": str(error)[:2000]})
                        save_deferred("deferred", retry_pass, [task for task, _ in deferred])
                        LOGGER.warning(
                            "Stage 2 extraction deferred root=%s task=%s retry_pass=%s: %s",
                            output_dir, task_label(active_task), retry_pass, error,
                        )
                        # Avoid marching through the entire cohort during a
                        # persistent outage. Isolated stragglers can be deferred.
                        if consecutive_failures >= max(3, int(workers)):
                            unresolved = [task for task, _ in deferred]
                            unresolved += [task for f, task in task_futures.items() if not f.done()]
                            save_deferred("aborted_consecutive_failures", retry_pass, unresolved)
                            raise
                        continue
                    consecutive_failures = 0
                    if kind == "batch":
                        completed.append((index, result))
                    else:
                        completed_pages[index] = result
                if not deferred:
                    if failures or failure_path.exists():
                        save_deferred("resolved", retry_pass, [])
                    break
                tasks = [task for task, _error in deferred]
                if retry_pass == deferred_retry_passes:
                    save_deferred("unresolved", retry_pass, tasks)
                    active_task, error = deferred[0]
                    raise error
                save_deferred("retrying_after_other_tasks", retry_pass + 1, tasks)
        except BaseException:
            # Log before executor shutdown waits for in-flight requests. Include
            # the checkpoint location so interleaved folds remain distinguishable.
            LOGGER.exception(
                "Stage 2 extraction failed root=%s batch=%s row_id=%s; "
                "cancelling queued work and waiting for in-flight requests",
                output_dir,
                active_task[1] if active_task and active_task[0] == "batch" else None,
                active_task[1] if active_task and active_task[0] == "paged_row" else None,
            )
            cancellation.set()
            for pending in task_futures:
                pending.cancel()
            raise
    values_by_row = {
        int(row["row_id"]): dict(row["values"])
        for _index, rows in sorted(completed)
        for row in rows
    }

    def reconcile_row(
        row_id: int,
        page_values: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> dict[str, Any]:
        reconciliation_dir = output_dir / "pages" / f"row_{row_id:08d}" / "reconciliation"
        result_path = reconciliation_dir / "result.json"
        decisions_path = reconciliation_dir / "decisions.json"
        complete_path = reconciliation_dir / "complete.json"
        ordered = sorted(page_values, key=lambda item: int(item[0]["page_index"]))
        page_results = [
            {
                **dict(meta),
                "observations": [dict(value) for value in result["observations"]],
            }
            for meta, result in ordered
        ]
        reconciliation_fingerprint = _value_fingerprint(
            {
                "schema_version": PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION,
                "request_identity": extraction_request_identity,
                "row_id": int(row_id),
                "definitions": page_extraction_definitions,
                "page_results": page_results,
            }
        )
        if complete_path.is_file() and result_path.is_file() and decisions_path.is_file():
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
                stored = json.loads(result_path.read_text(encoding="utf-8"))
                decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
                if (
                    completion.get("schema_version")
                    == PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION
                    and completion.get("input_fingerprint") == reconciliation_fingerprint
                    and isinstance(decisions, Mapping)
                ):
                    validated = _validate_extraction(
                        stored,
                        row_ids=[row_id],
                        definitions=definitions,
                    )
                    return dict(validated["rows"][0]["values"])
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            LOGGER.info(
                "rerun incompatible Stage 2 page reconciliation: %s",
                reconciliation_dir,
            )
        merged_values, decisions = _resolve_page_observations(
            definitions=definitions,
            page_results=page_results,
        )
        result = _validate_extraction(
            {"rows": [{"row_id": int(row_id), "values": merged_values}]},
            row_ids=[row_id],
            definitions=definitions,
        )
        reconciliation_dir.mkdir(parents=True, exist_ok=True)
        _write_json(reconciliation_dir / "page_manifest.json", page_results)
        _write_json(
            decisions_path,
            {
                "schema_version": PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION,
                "row_id": int(row_id),
                "decisions": decisions,
            },
        )
        _write_json(result_path, result)
        _write_json(
            reconciliation_dir / "extraction_issues.json",
            {
                "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
                "completed_at": _now(),
                "events": [],
                "deterministic_reconciliation": True,
            },
        )
        _write_json(
            complete_path,
            {
                "status": "complete",
                "schema_version": PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION,
                "input_fingerprint": reconciliation_fingerprint,
                "completed_at": _now(),
                "pages": len(page_results),
                "observations": sum(
                    len(page_result["observations"]) for page_result in page_results
                ),
                "features": len(definitions),
                "conflicts": sum(
                    int(decision["distinct_value_count"] > 1)
                    for decision in decisions.values()
                ),
                "null_conflicts": sum(
                    int(decision["resolution"] == "conflict_null")
                    for decision in decisions.values()
                ),
                "reconciliation_method": "deterministic_reported_occurrences",
            },
        )
        return dict(result["rows"][0]["values"])

    if completed_pages:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(int(workers), len(completed_pages)))
        ) as executor:
            reconciliation_futures = {
                executor.submit(reconcile_row, row_id, page_values): row_id
                for row_id, page_values in completed_pages.items()
            }
            for future in concurrent.futures.as_completed(reconciliation_futures):
                row_id = reconciliation_futures[future]
                values_by_row[row_id] = future.result()
    records = []
    for row_id in row_ids:
        record: dict[str, Any] = {"_oci_row_id": int(row_id)}
        record.update({name: values_by_row[int(row_id)].get(name) for name in feature_names})
        records.append(record)
    frame = pd.DataFrame(records, columns=["_oci_row_id", *feature_names])
    _write_frame(output_dir / "extracted.csv", frame)
    failure_summary = _summarize_extraction_failures(
        output_dir=output_dir,
        definitions=definitions,
    )
    _write_json(
        output_dir / "complete.json",
        {
            "status": "complete",
            "completed_at": _now(),
            "rows": len(frame),
            "features": len(feature_names),
            "feature_batch_size": feature_batch_size,
            "feature_batches_per_patient": len(definition_batches),
            "batches": len(batches),
            "paged_rows": len(oversized_rows),
            "pages": sum(len(pages) for pages in completed_pages.values()),
            "serial_patient_feature_passes": len(
                list((output_dir / "batches").glob("**/serial_complete.json"))
            ),
            "feature_failure_patterns": len(failure_summary["feature_failure_patterns"]),
            "structural_failure_patients": failure_summary["structural_failure_patient_count"],
        },
    )
    return frame


def _feature_modeling_strategy(feature: Mapping[str, Any]) -> str:
    value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
    if value_type != "continuous":
        return "categorical"
    harmonization = feature.get("harmonization_plan")
    if isinstance(harmonization, Mapping):
        target = str(harmonization.get("target_representation") or "").strip().lower()
        if target not in {"continuous", "categorical"}:
            raise ValueError(
                f"continuous feature {feature.get('name')!r} has an invalid " "harmonization target"
            )
        return target
    strategy = str(feature.get("modeling_strategy") or "continuous").strip().lower()
    if strategy not in CONTINUOUS_MODELING_STRATEGIES:
        raise ValueError(
            f"continuous feature {feature.get('name')!r} has unsupported "
            f"modeling_strategy {strategy!r}"
        )
    return strategy


def _normalized_feature_modeling_definition(
    feature: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(feature)
    if str(normalized.get("value_type") or "ambiguous").strip().lower() == "continuous":
        normalized["modeling_strategy"] = _feature_modeling_strategy(normalized)
    else:
        normalized.pop("modeling_strategy", None)
    return normalized


def feature_summaries(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    row_count = max(1, len(frame))
    for feature in definitions:
        name = str(feature["name"])
        series = frame[name] if name in frame else pd.Series([None] * len(frame))
        nonmissing = series.dropna()
        counts = nonmissing.astype(str).value_counts()
        dominant = float(counts.iloc[0] / len(nonmissing)) if len(nonmissing) else 1.0
        unknown_rows = int(documented_unknown_mask(nonmissing).sum())
        summary: dict[str, Any] = {
            "feature_id": str(feature["feature_id"]),
            "name": name,
            "rows": len(frame),
            "nonmissing": int(len(nonmissing)),
            "nonmissing_fraction": float(len(nonmissing) / row_count),
            # Counted in nonmissing: the record states that the value is unknown.
            "documented_unknown": unknown_rows,
            "documented_unknown_fraction": float(unknown_rows / row_count),
            "unique_nonmissing": int(nonmissing.nunique()),
            "dominant_value_fraction": dominant,
            "most_common_values": {str(key): int(count) for key, count in counts.head(8).items()},
        }
        if str(feature.get("value_type")) == "continuous":
            nonmissing = documented_unknown.without_documented_unknown(nonmissing).dropna()
            numeric_values = pd.to_numeric(nonmissing, errors="coerce")
            numeric = numeric_values.dropna()
            categorical = nonmissing.loc[numeric_values.isna()].astype(str)
            categorical_counts = categorical.value_counts()
            summary.update(
                {
                    "numeric_nonmissing": int(len(numeric)),
                    "numeric_nonmissing_fraction": float(len(numeric) / row_count),
                    "categorical_fallback_nonmissing": int(len(categorical)),
                    "categorical_fallback_nonmissing_fraction": float(len(categorical) / row_count),
                    "categorical_fallback_values": {
                        str(key): int(count) for key, count in categorical_counts.head(12).items()
                    },
                    "numeric_mean": float(numeric.mean()) if len(numeric) else None,
                    "numeric_sd": float(numeric.std(ddof=0)) if len(numeric) else None,
                    "recommended_modeling_strategy": (
                        "continuous_with_categorical_fallback"
                        if len(numeric) and len(categorical)
                        else "categorical" if len(categorical) else "continuous"
                    ),
                }
            )
        summaries.append(summary)
    return summaries


def _mixed_value_observations(
    frame: pd.DataFrame,
    feature: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Describe a continuous extraction containing numeric and text values."""

    name = str(feature["name"])
    series = frame[name] if name in frame else pd.Series([None] * len(frame))
    # documented_unknown is a reserved state, not a phrase to harmonize.
    nonmissing = documented_unknown.without_documented_unknown(series).dropna()
    numeric_values = pd.to_numeric(nonmissing, errors="coerce")
    numeric = numeric_values.dropna().astype(float)
    categorical = nonmissing.loc[numeric_values.isna()].astype(str)
    if not len(numeric) or not len(categorical):
        return None
    categorical_counts = categorical.value_counts(dropna=False)
    quantile_probabilities = np.linspace(0.0, 1.0, min(21, len(numeric)))
    quantiles = [
        {
            "probability": float(probability),
            "value": float(numeric.quantile(float(probability))),
        }
        for probability in quantile_probabilities
    ]
    return {
        "numeric_count": int(len(numeric)),
        "numeric_min": float(numeric.min()),
        "numeric_max": float(numeric.max()),
        "numeric_quantiles": quantiles,
        "categorical_count": int(len(categorical)),
        "categorical_values": [
            {"raw_value": str(raw_value), "count": int(count)}
            for raw_value, count in categorical_counts.items()
        ],
    }


def _harmonization_prompt(*, feature: Mapping[str, Any], observations: Mapping[str, Any],
    prior_plan: Mapping[str, Any] | None) -> list[dict[str, str]]:
    return clinical_prompts.messages("10_harmonize_values", clinical_prompts.feature_text(feature)
        + "\n\nObserved values\n" + clinical_prompts.readable(observations))


def _harmonization_delta_prompt(*, feature: Mapping[str, Any], prior_plan: Mapping[str, Any],
    new_categorical_values: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if len(new_categorical_values) != 1:
        raise ValueError("value-map extension requires one phrase per call")
    representation = {key: prior_plan[key] for key in ("target_representation", "canonical_categories", "numeric_bin_rules")}
    return clinical_prompts.messages("11_extend_value_map", clinical_prompts.feature_text(feature)
        + "\n\nRepresentation and category definitions\n" + clinical_prompts.readable(representation)
        + "\n\nPhrase to interpret\n" + str(new_categorical_values[0]["raw_value"]))


def _bin_contains(value: float, rule: Mapping[str, Any]) -> bool:
    lower = rule.get("lower_bound")
    upper = rule.get("upper_bound")
    lower_ok = (
        lower is None
        or value > float(lower)
        or (bool(rule.get("lower_inclusive")) and value == float(lower))
    )
    upper_ok = (
        upper is None
        or value < float(upper)
        or (bool(rule.get("upper_inclusive")) and value == float(upper))
    )
    return bool(lower_ok and upper_ok)


_HARMONIZATION_MAPPING_NORMALIZATION_COUNT_FIELDS = (
    "non_object_entries_dropped",
    "empty_raw_value_entries_dropped",
)
_HARMONIZATION_MAPPING_NORMALIZATION_VALUE_FIELDS = (
    "extra_raw_values_dropped",
    "identical_duplicate_raw_values_deduplicated",
    "conflicting_duplicate_raw_values_mapped_to_null",
    "missing_raw_values_mapped_to_null",
    "invalid_canonical_raw_values_mapped_to_null",
)


def _finalize_harmonization_mapping_normalization(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        field: int(audit.get(field) or 0)
        for field in _HARMONIZATION_MAPPING_NORMALIZATION_COUNT_FIELDS
    }
    payload.update(
        {
            field: sorted(
                {
                    str(item)
                    for item in audit.get(field) or []
                    if str(item)
                }
            )
            for field in _HARMONIZATION_MAPPING_NORMALIZATION_VALUE_FIELDS
        }
    )
    payload["status"] = (
        "normalized"
        if any(payload[field] for field in _HARMONIZATION_MAPPING_NORMALIZATION_COUNT_FIELDS)
        or any(payload[field] for field in _HARMONIZATION_MAPPING_NORMALIZATION_VALUE_FIELDS)
        else "unchanged"
    )
    return payload


def _merge_harmonization_mapping_normalizations(
    *audits: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    present = [audit for audit in audits if isinstance(audit, Mapping)]
    if not present:
        return None
    combined: dict[str, Any] = {
        field: sum(int(audit.get(field) or 0) for audit in present)
        for field in _HARMONIZATION_MAPPING_NORMALIZATION_COUNT_FIELDS
    }
    combined.update(
        {
            field: [
                item
                for audit in present
                for item in audit.get(field) or []
            ]
            for field in _HARMONIZATION_MAPPING_NORMALIZATION_VALUE_FIELDS
        }
    )
    finalized = _finalize_harmonization_mapping_normalization(combined)
    return finalized if finalized["status"] == "normalized" else None


def _normalize_harmonization_value_map(
    raw_mapping: Any,
    *,
    expected_raw_values: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservatively repair bookkeeping defects in an otherwise usable value map."""

    if not isinstance(raw_mapping, list):
        raise ValueError("harmonization requires categorical_value_map")
    expected = list(dict.fromkeys(str(raw_value) for raw_value in expected_raw_values))
    expected_set = set(expected)
    mapping_by_raw: dict[str, Any] = {}
    conflicts: set[str] = set()
    audit: dict[str, Any] = {
        "non_object_entries_dropped": 0,
        "empty_raw_value_entries_dropped": 0,
        "extra_raw_values_dropped": [],
        "identical_duplicate_raw_values_deduplicated": [],
        "conflicting_duplicate_raw_values_mapped_to_null": [],
        "missing_raw_values_mapped_to_null": [],
        "invalid_canonical_raw_values_mapped_to_null": [],
    }
    for row in raw_mapping:
        if not isinstance(row, Mapping):
            audit["non_object_entries_dropped"] += 1
            continue
        raw_value = str(row.get("raw_value") or "")
        if not raw_value:
            audit["empty_raw_value_entries_dropped"] += 1
            continue
        if raw_value not in expected_set:
            audit["extra_raw_values_dropped"].append(raw_value)
            continue
        canonical = row.get("canonical_value")
        if raw_value in mapping_by_raw:
            if raw_value in conflicts or mapping_by_raw[raw_value] != canonical:
                mapping_by_raw[raw_value] = None
                conflicts.add(raw_value)
                audit["conflicting_duplicate_raw_values_mapped_to_null"].append(raw_value)
            else:
                audit["identical_duplicate_raw_values_deduplicated"].append(raw_value)
            continue
        mapping_by_raw[raw_value] = canonical
    for raw_value in expected:
        if raw_value not in mapping_by_raw:
            mapping_by_raw[raw_value] = None
            audit["missing_raw_values_mapped_to_null"].append(raw_value)
    return mapping_by_raw, _finalize_harmonization_mapping_normalization(audit)


def _clean_harmonization_mapping_values(
    mapping_by_raw: Mapping[str, Any],
    *,
    expected_raw_values: Sequence[str],
    target: str,
    categories: Sequence[str],
    normalization: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = dict(normalization)
    invalid = list(audit.get("invalid_canonical_raw_values_mapped_to_null") or [])
    clean_mapping: list[dict[str, Any]] = []
    for raw_value in expected_raw_values:
        canonical = mapping_by_raw[str(raw_value)]
        if canonical is not None and target == "continuous":
            if (
                isinstance(canonical, bool)
                or not isinstance(canonical, (int, float))
                or not math.isfinite(float(canonical))
            ):
                canonical = None
                invalid.append(str(raw_value))
            else:
                canonical = float(canonical)
        elif canonical is not None:
            canonical = str(canonical)
            if canonical not in categories:
                canonical = None
                invalid.append(str(raw_value))
        clean_mapping.append(
            {"raw_value": str(raw_value), "canonical_value": canonical}
        )
    audit["invalid_canonical_raw_values_mapped_to_null"] = invalid
    clean_mapping.sort(key=lambda row: str(row["raw_value"]))
    return clean_mapping, _finalize_harmonization_mapping_normalization(audit)


def _validate_harmonization_plan(
    value: Mapping[str, Any],
    *,
    feature: Mapping[str, Any],
    observations: Mapping[str, Any],
) -> dict[str, Any]:
    if "status" in value:
        from .stage2_value_interpretation import compile_value_interpretations
        value = compile_value_interpretations(value, observations=observations)

    target = str(value.get("target_representation") or "").strip().lower()
    if target not in {"continuous", "categorical"}:
        raise ValueError("harmonization target_representation must be continuous or categorical")
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("harmonization requires a reason")
    expected_raw_values = [
        str(row["raw_value"]) for row in observations.get("categorical_values") or []
    ]
    mapping_by_raw, mapping_normalization = _normalize_harmonization_value_map(
        value.get("categorical_value_map"),
        expected_raw_values=expected_raw_values,
    )

    raw_categories = value.get("canonical_categories")
    raw_bins = value.get("numeric_bin_rules")
    if not isinstance(raw_categories, list) or not isinstance(raw_bins, list):
        raise ValueError("harmonization requires canonical_categories and numeric_bin_rules arrays")
    categories: list[str] = []
    bins: list[dict[str, Any]] = []
    if target == "continuous":
        if raw_categories or raw_bins:
            raise ValueError("continuous harmonization requires empty categories and numeric bins")
    else:
        categories = [str(item).strip() for item in raw_categories]
        if (
            len(categories) < 2
            or any(not item for item in categories)
            or len(set(categories)) != len(categories)
        ):
            raise ValueError("categorical harmonization requires at least two distinct categories")
        if not raw_bins:
            raise ValueError("categorical harmonization requires numeric_bin_rules")
        for index, raw_rule in enumerate(raw_bins):
            if not isinstance(raw_rule, Mapping):
                raise ValueError("each numeric bin rule must be an object")
            for label in ("lower_inclusive", "upper_inclusive"):
                if not isinstance(raw_rule.get(label), bool):
                    raise ValueError(f"numeric bin {label} must be a boolean")
            lower = raw_rule.get("lower_bound")
            upper = raw_rule.get("upper_bound")
            for label, bound in (("lower_bound", lower), ("upper_bound", upper)):
                if bound is not None:
                    if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                        raise ValueError(f"numeric bin {label} must be a number or null")
                    if not math.isfinite(float(bound)):
                        raise ValueError(f"numeric bin {label} must be finite or null")
            lower = float(lower) if lower is not None else None
            upper = float(upper) if upper is not None else None
            if lower is not None and upper is not None and (
                lower > upper or (lower == upper and not (raw_rule["lower_inclusive"] and raw_rule["upper_inclusive"]))
            ):
                raise ValueError("numeric bins require ordered bounds; a point interval must include its boundary")
            canonical = str(raw_rule.get("canonical_value") or "")
            if canonical not in categories:
                raise ValueError("numeric bin values must be canonical categories")
            rule = {
                "lower_bound": lower,
                "lower_inclusive": bool(raw_rule.get("lower_inclusive")),
                "upper_bound": upper,
                "upper_inclusive": bool(raw_rule.get("upper_inclusive")),
                "canonical_value": canonical,
            }
            if index == 0 and lower is not None:
                raise ValueError("the first numeric bin lower_bound must be null")
            if index > 0:
                prior = bins[-1]
                prior_upper = prior["upper_bound"]
                if (
                    prior_upper is None
                    or lower is None
                    or not math.isclose(
                        float(prior_upper), float(lower), rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    raise ValueError("numeric bins must be ordered and contiguous")
                if bool(prior["upper_inclusive"]) == bool(rule["lower_inclusive"]):
                    raise ValueError("adjacent numeric bins require exactly one inclusive boundary")
            bins.append(rule)
        if bins[-1]["upper_bound"] is not None:
            raise ValueError("the final numeric bin upper_bound must be null")
        observed_numeric = [
            float(row["value"]) for row in observations.get("numeric_quantiles") or []
        ]
        if any(sum(_bin_contains(item, rule) for rule in bins) != 1 for item in observed_numeric):
            raise ValueError("numeric bins must assign every observed numeric summary value once")

    clean_mapping, mapping_normalization = _clean_harmonization_mapping_values(
        mapping_by_raw,
        expected_raw_values=expected_raw_values,
        target=target,
        categories=categories,
        normalization=mapping_normalization,
    )
    combined_normalization = _merge_harmonization_mapping_normalizations(
        (
            value.get("categorical_value_map_normalization")
            if isinstance(value.get("categorical_value_map_normalization"), Mapping)
            else None
        ),
        mapping_normalization,
    )
    result = {
        "schema_version": HARMONIZATION_CHECKPOINT_SCHEMA_VERSION,
        "feature_id": str(feature["feature_id"]),
        "target_representation": target,
        "reason": reason,
        "canonical_categories": categories,
        "categorical_value_map": clean_mapping,
        "numeric_bin_rules": bins,
        "unmapped_value_rule": "null",
        "training_observations_fingerprint": _value_fingerprint(observations),
    }
    if combined_normalization is not None:
        result["categorical_value_map_normalization"] = combined_normalization
    return result


def _validate_harmonization_delta(
    value: Mapping[str, Any],
    *,
    prior_plan: Mapping[str, Any],
    new_raw_values: Sequence[str],
) -> dict[str, Any]:
    if set(value) == {"value"} and len(new_raw_values) == 1:
        value = {"categorical_value_map": [{"raw_value": str(new_raw_values[0]), "canonical_value": value["value"]}]}

    mapping_by_raw, normalization = _normalize_harmonization_value_map(
        value.get("categorical_value_map"),
        expected_raw_values=new_raw_values,
    )
    clean_mapping, normalization = _clean_harmonization_mapping_values(
        mapping_by_raw,
        expected_raw_values=new_raw_values,
        target=str(prior_plan["target_representation"]),
        categories=[str(item) for item in prior_plan.get("canonical_categories") or []],
        normalization=normalization,
    )
    result: dict[str, Any] = {"categorical_value_map": clean_mapping}
    if normalization["status"] == "normalized":
        result["categorical_value_map_normalization"] = normalization
    return result


def _validate_prior_harmonization_plan_for_extension(
    prior_plan: Mapping[str, Any],
    *,
    feature: Mapping[str, Any],
    observations: Mapping[str, Any],
) -> dict[str, Any]:
    current_counts = {
        str(row["raw_value"]): int(row.get("count") or 0)
        for row in observations.get("categorical_values") or []
    }
    prior_raw_values = list(
        dict.fromkeys(
            str(row.get("raw_value") or "")
            for row in prior_plan.get("categorical_value_map") or []
            if isinstance(row, Mapping) and str(row.get("raw_value") or "")
        )
    )
    prior_observations = copy.deepcopy(dict(observations))
    prior_observations["categorical_values"] = [
        {"raw_value": raw_value, "count": current_counts.get(raw_value, 0)}
        for raw_value in prior_raw_values
    ]
    return _validate_harmonization_plan(
        prior_plan,
        feature=feature,
        observations=prior_observations,
    )


def _harmonization_validation_fallback(
    *,
    feature: Mapping[str, Any],
    observations: Mapping[str, Any],
    validation_error: ValueError,
    status: str,
    unresolved_raw_values: Sequence[str],
    retained_prior_plan: bool,
) -> dict[str, Any]:
    return {
        "schema_version": HARMONIZATION_FALLBACK_SCHEMA_VERSION,
        "status": status,
        "feature_id": str(feature["feature_id"]),
        "recorded_at": _now(),
        "validation_error": f"{type(validation_error).__name__}: {validation_error}",
        "retained_prior_plan": retained_prior_plan,
        "unresolved_raw_values": sorted(set(map(str, unresolved_raw_values))),
        "unresolved_value_rule": "null" if retained_prior_plan else "retain_raw_hybrid_value",
        "modeling_strategy": (
            None if retained_prior_plan else "continuous_with_categorical_fallback"
        ),
        "numeric_training_values": int(observations.get("numeric_count") or 0),
        "categorical_training_values": int(observations.get("categorical_count") or 0),
        "training_observations_fingerprint": _value_fingerprint(observations),
    }


def _extend_prior_plan_with_null_mappings(
    prior_plan: Mapping[str, Any],
    *,
    feature: Mapping[str, Any],
    observations: Mapping[str, Any],
    new_raw_values: Sequence[str],
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(prior_plan))
    candidate["categorical_value_map"] = [
        *[dict(row) for row in prior_plan.get("categorical_value_map") or []],
        *[
            {"raw_value": str(raw_value), "canonical_value": None}
            for raw_value in new_raw_values
        ],
    ]
    null_extension_normalization = _finalize_harmonization_mapping_normalization(
        {
            "missing_raw_values_mapped_to_null": list(new_raw_values),
        }
    )
    combined_normalization = _merge_harmonization_mapping_normalizations(
        (
            prior_plan.get("categorical_value_map_normalization")
            if isinstance(prior_plan.get("categorical_value_map_normalization"), Mapping)
            else None
        ),
        null_extension_normalization,
    )
    if combined_normalization is not None:
        candidate["categorical_value_map_normalization"] = combined_normalization
    return _validate_harmonization_plan(
        candidate,
        feature=feature,
        observations=observations,
    )


def _request_harmonization_plan(
    *,
    feature: Mapping[str, Any],
    observations: Mapping[str, Any],
    prior_plan: Mapping[str, Any] | None,
    output_dir: Path,
    request_json: RequestJSON,
    max_prompt_chars: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    input_value = {
        "schema_version": HARMONIZATION_CHECKPOINT_SCHEMA_VERSION,
        "feature": {
            key: copy.deepcopy(feature.get(key))
            for key in (
                "feature_id",
                "name",
                "description",
                "categories_or_unit",
                "measurement_definition",
                "missing_value_rule",
            )
        },
        "observations": copy.deepcopy(observations),
        "prior_plan": copy.deepcopy(prior_plan),
    }
    input_fingerprint = _value_fingerprint(input_value)
    result_path = output_dir / "result.json"
    fallback_path = output_dir / "fallback.json"
    complete_path = output_dir / "complete.json"
    if complete_path.is_file():
        try:
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                completion.get("schema_version") == HARMONIZATION_CHECKPOINT_SCHEMA_VERSION
                and completion.get("input_fingerprint") == input_fingerprint
            ):
                fallback = (
                    json.loads(fallback_path.read_text(encoding="utf-8"))
                    if completion.get("validation_fallback") is True
                    and fallback_path.is_file()
                    else None
                )
                if result_path.is_file():
                    plan = _validate_harmonization_plan(
                        json.loads(result_path.read_text(encoding="utf-8")),
                        feature=feature,
                        observations=observations,
                    )
                    return plan, fallback, False
                if (
                    isinstance(fallback, Mapping)
                    and fallback.get("schema_version")
                    == HARMONIZATION_FALLBACK_SCHEMA_VERSION
                ):
                    return None, dict(fallback), False
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    validated_prior: dict[str, Any] | None = None
    if isinstance(prior_plan, Mapping):
        try:
            validated_prior = _validate_prior_harmonization_plan_for_extension(
                prior_plan,
                feature=feature,
                observations=observations,
            )
        except ValueError:
            validated_prior = None
    expected_rows = list(observations.get("categorical_values") or [])
    expected_raw_values = [str(row["raw_value"]) for row in expected_rows]
    new_rows: list[Mapping[str, Any]] = []
    if validated_prior is not None:
        prior_raw_values = {
            str(row["raw_value"])
            for row in validated_prior.get("categorical_value_map") or []
        }
        new_rows = [
            row for row in expected_rows if str(row["raw_value"]) not in prior_raw_values
        ]
        if not new_rows:
            return validated_prior, None, False
        messages = _harmonization_delta_prompt(
            feature=feature,
            prior_plan=validated_prior,
            new_categorical_values=new_rows[:1],
        )
        request_mode = "prior_plan_delta"
    else:
        messages = _harmonization_prompt(
            feature=feature,
            observations=observations,
            prior_plan=prior_plan,
        )
        request_mode = "full_plan"
    prompt_chars = _prompt_chars(messages)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "input.json",
        {
            **input_value,
            "input_fingerprint": input_fingerprint,
            "request_mode": request_mode,
            "new_categorical_values": copy.deepcopy(new_rows),
        },
    )
    request_performed = False
    validation_error: ValueError | None = None
    plan: dict[str, Any] | None = None
    try:
        if prompt_chars > int(max_prompt_chars):
            raise ValueError(
                "Stage 2 mixed-value harmonization prompt exceeds max_prompt_chars "
                f"({prompt_chars} > {max_prompt_chars}) for {feature.get('name')!r}"
            )
        request_performed = True
        if validated_prior is not None:
            new_raw_values = [str(row["raw_value"]) for row in new_rows]
            delta = {"categorical_value_map": []}
            for new_row in new_rows:
                token_delta = request_json(
                    _harmonization_delta_prompt(feature=feature, prior_plan=validated_prior,
                                               new_categorical_values=[new_row]),
                    lambda response, raw=str(new_row["raw_value"]): _validate_harmonization_delta(
                        response, prior_plan=validated_prior, new_raw_values=[raw]),
                    request_kind="interpretation",
                )
                delta["categorical_value_map"].extend(token_delta["categorical_value_map"])
            candidate = copy.deepcopy(validated_prior)
            candidate["categorical_value_map"] = [
                *[dict(row) for row in validated_prior["categorical_value_map"]],
                *[dict(row) for row in delta["categorical_value_map"]],
            ]
            delta_normalization = delta.get("categorical_value_map_normalization")
            combined_normalization = _merge_harmonization_mapping_normalizations(
                (
                    validated_prior.get("categorical_value_map_normalization")
                    if isinstance(
                        validated_prior.get("categorical_value_map_normalization"), Mapping
                    )
                    else None
                ),
                delta_normalization if isinstance(delta_normalization, Mapping) else None,
            )
            if combined_normalization is not None:
                candidate["categorical_value_map_normalization"] = combined_normalization
            plan = _validate_harmonization_plan(
                candidate,
                feature=feature,
                observations=observations,
            )
        else:
            plan = request_json(
                messages,
                lambda response: _validate_harmonization_plan(
                    response,
                    feature=feature,
                    observations=observations,
                ),
                request_kind="interpretation",
            )
    except Stage2ResponseValidationError as exc:
        validation_error = exc

    fallback: dict[str, Any] | None = None
    if validation_error is not None:
        if validated_prior is not None:
            unresolved = [str(row["raw_value"]) for row in new_rows]
            plan = _extend_prior_plan_with_null_mappings(
                validated_prior,
                feature=feature,
                observations=observations,
                new_raw_values=unresolved,
            )
            fallback = _harmonization_validation_fallback(
                feature=feature,
                observations=observations,
                validation_error=validation_error,
                status="prior_plan_extended_with_null_mappings",
                unresolved_raw_values=unresolved,
                retained_prior_plan=True,
            )
        else:
            plan = None
            fallback = _harmonization_validation_fallback(
                feature=feature,
                observations=observations,
                validation_error=validation_error,
                status="hybrid_modeling_without_harmonization_plan",
                unresolved_raw_values=expected_raw_values,
                retained_prior_plan=False,
            )
        _write_json(fallback_path, fallback)
        LOGGER.warning(
            "Stage 2 harmonization remained invalid for feature=%s; using audited "
            "fallback status=%s (%s)",
            feature.get("feature_id"),
            fallback["status"],
            validation_error,
        )
    if plan is not None:
        _write_json(result_path, plan)
    _write_json(
        complete_path,
        {
            "status": (
                "complete_with_validation_fallback" if fallback is not None else "complete"
            ),
            "schema_version": HARMONIZATION_CHECKPOINT_SCHEMA_VERSION,
            "fallback_schema_version": (
                HARMONIZATION_FALLBACK_SCHEMA_VERSION if fallback is not None else None
            ),
            "input_fingerprint": input_fingerprint,
            "completed_at": _now(),
            "prompt_chars": prompt_chars,
            "request_mode": request_mode,
            "request_performed": request_performed,
            "validation_fallback": fallback is not None,
        },
    )
    return plan, fallback, request_performed


def _apply_one_harmonization_plan(
    series: pd.Series,
    plan: Mapping[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    exact_mapping = {
        str(row["raw_value"]): row.get("canonical_value")
        for row in plan.get("categorical_value_map") or []
    }
    normalized_targets: dict[str, set[Any]] = {}
    for raw_value, canonical in exact_mapping.items():
        normalized_targets.setdefault(raw_value.strip().casefold(), set()).add(canonical)
    normalized_mapping = {
        key: next(iter(values)) for key, values in normalized_targets.items() if len(values) == 1
    }
    target = str(plan["target_representation"])
    bins = list(plan.get("numeric_bin_rules") or [])
    output: list[Any] = []
    unmapped: list[str] = []
    mapped_numeric = 0
    mapped_categorical = 0
    for raw in series.tolist():
        if raw is None or bool(pd.isna(raw)):
            output.append(None)
            continue
        if is_documented_unknown(raw):
            output.append(DOCUMENTED_UNKNOWN)
            continue
        numeric = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            numeric_value = float(numeric)
            if target == "continuous":
                output.append(numeric_value)
            else:
                matches = [rule for rule in bins if _bin_contains(numeric_value, rule)]
                if len(matches) == 1:
                    output.append(str(matches[0]["canonical_value"]))
                    mapped_numeric += 1
                else:  # pragma: no cover - validated plans are exhaustive
                    output.append(None)
                    unmapped.append(str(raw))
            continue
        raw_value = str(raw)
        if raw_value in exact_mapping:
            output.append(exact_mapping[raw_value])
            mapped_categorical += 1
        elif raw_value.strip().casefold() in normalized_mapping:
            output.append(normalized_mapping[raw_value.strip().casefold()])
            mapped_categorical += 1
        else:
            output.append(None)
            unmapped.append(raw_value)
    return pd.Series(output, index=series.index, dtype=object), {
        "target_representation": target,
        "rows": int(len(series)),
        "mapped_numeric_rows": int(mapped_numeric),
        "mapped_categorical_rows": int(mapped_categorical),
        "unmapped_nonmissing_rows": int(len(unmapped)),
        "unmapped_value_examples": list(dict.fromkeys(unmapped))[:12],
    }


def _apply_harmonization_plans(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    harmonized = frame.copy()
    features: list[dict[str, Any]] = []
    for feature in definitions:
        plan = feature.get("harmonization_plan")
        if not isinstance(plan, Mapping):
            continue
        name = str(feature["name"])
        series = (
            harmonized[name]
            if name in harmonized
            else pd.Series([None] * len(harmonized), index=harmonized.index)
        )
        harmonized[name], audit = _apply_one_harmonization_plan(series, plan)
        features.append(
            {
                "feature_id": str(feature["feature_id"]),
                "name": name,
                **audit,
            }
        )
    return harmonized, {
        "schema_version": "stage2_applied_value_harmonization_v1",
        "scope": scope,
        "rows": int(len(frame)),
        "features_harmonized": len(features),
        "features": features,
    }


def _harmonize_training_extraction(
    *,
    extracted: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    output_dir: Path,
    request_json: RequestJSON,
    max_prompt_chars: int,
    workers: int = 1,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from .stage2_parallel import ordered_map

    def harmonize_feature(raw_feature):
        newly_requested, fallbacks, mapping_normalizations = [], [], []
        feature = dict(raw_feature)
        if str(feature.get("value_type") or "").strip().lower() == "continuous":
            observations = _mixed_value_observations(extracted, feature)
            if observations is not None:
                prior_plan = feature.get("harmonization_plan")
                prior_fallback = feature.get("harmonization_fallback")
                if isinstance(prior_plan, Mapping):
                    observed_raw = {
                        str(row["raw_value"]) for row in observations["categorical_values"]
                    }
                    for row in prior_plan.get("categorical_value_map") or []:
                        raw_value = str(row.get("raw_value") or "")
                        if raw_value and raw_value not in observed_raw:
                            observations["categorical_values"].append(
                                {"raw_value": raw_value, "count": 0}
                            )
                fallback: dict[str, Any] | None = None
                request_performed = False
                if (
                    not isinstance(prior_plan, Mapping)
                    and isinstance(prior_fallback, Mapping)
                    and prior_fallback.get("status")
                    == "hybrid_modeling_without_harmonization_plan"
                ):
                    plan = None
                    fallback = dict(prior_fallback)
                else:
                    feature_dir = output_dir / str(feature["feature_id"])
                    plan, fallback, request_performed = _request_harmonization_plan(
                        feature=feature,
                        observations=observations,
                        prior_plan=(prior_plan if isinstance(prior_plan, Mapping) else None),
                        output_dir=feature_dir,
                        request_json=request_json,
                        max_prompt_chars=max_prompt_chars,
                    )
                if request_performed:
                    newly_requested.append(str(feature["feature_id"]))
                if fallback is not None:
                    feature["harmonization_fallback"] = copy.deepcopy(fallback)
                    fallbacks.append(
                        {
                            **copy.deepcopy(fallback),
                            "reused_from_prior_round": not request_performed,
                        }
                    )
                elif request_performed:
                    feature.pop("harmonization_fallback", None)
                if plan is not None:
                    feature["harmonization_plan"] = plan
                    feature["modeling_strategy"] = str(plan["target_representation"])
                    normalization = plan.get("categorical_value_map_normalization")
                    if isinstance(normalization, Mapping):
                        mapping_normalizations.append(
                            {
                                "feature_id": str(feature["feature_id"]),
                                "name": str(feature["name"]),
                                **copy.deepcopy(normalization),
                            }
                        )
                else:
                    feature.pop("harmonization_plan", None)
                    if (
                        fallback is not None
                        and fallback.get("status")
                        == "hybrid_modeling_without_harmonization_plan"
                        and (
                            request_performed
                            or not isinstance(prior_fallback, Mapping)
                        )
                    ):
                        feature["modeling_strategy"] = (
                            "continuous_with_categorical_fallback"
                        )
                    elif str(feature.get("modeling_strategy") or "").strip().lower() not in (
                        CONTINUOUS_MODELING_STRATEGIES
                    ):
                        feature["modeling_strategy"] = (
                            "continuous_with_categorical_fallback"
                        )
        return (_normalized_feature_modeling_definition(feature),
                newly_requested, fallbacks, mapping_normalizations)

    rows = ordered_map(harmonize_feature, definitions, workers=workers,
                       thread_name="stage2-harmonization")
    updated = [row[0] for row in rows]
    newly_requested = [value for row in rows for value in row[1]]
    fallbacks = [value for row in rows for value in row[2]]
    mapping_normalizations = [value for row in rows for value in row[3]]
    harmonized, application = _apply_harmonization_plans(
        extracted,
        updated,
        scope="outer_training",
    )
    report = {
        **application,
        "plans_requested_from_llm": len(newly_requested),
        "features_requested_from_llm": newly_requested,
        "harmonization_validation_fallbacks": len(fallbacks),
        "features_with_harmonization_validation_fallback": [
            str(fallback["feature_id"]) for fallback in fallbacks
        ],
        "fallbacks": fallbacks,
        "normalized_categorical_value_maps": len(mapping_normalizations),
        "categorical_value_map_normalizations": mapping_normalizations,
    }
    _write_frame(output_dir / "extracted_harmonized.csv", harmonized)
    _write_json(output_dir / "harmonization.json", report)
    return harmonized, updated, report


def _assert_extraction_health(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    minimum_row_nonmissing_fraction: float,
    audit_path: Path,
) -> dict[str, Any]:
    """Reject final extraction matrices that are effectively all missing."""

    feature_names = [str(feature["name"]) for feature in definitions]
    if not feature_names:
        audit = {
            "schema_version": "stage2_final_extraction_health_v1",
            "status": "not_applicable",
            "scope": scope,
            "rows": int(len(frame)),
            "features": 0,
        }
        _write_json(audit_path, audit)
        return audit
    missing_columns = sorted(set(feature_names) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Stage 2 final {scope} extraction is missing feature columns: " f"{missing_columns}"
        )
    values = frame[feature_names]
    rows_with_any = values.notna().any(axis=1)
    row_fraction = float(rows_with_any.mean()) if len(values) else 0.0
    nonmissing_cells = int(values.notna().to_numpy(dtype=bool).sum())
    total_cells = int(values.shape[0] * values.shape[1])
    audit = {
        "schema_version": "stage2_final_extraction_health_v1",
        "status": ("ok" if row_fraction >= float(minimum_row_nonmissing_fraction) else "failed"),
        "scope": scope,
        "rows": int(len(values)),
        "features": int(len(feature_names)),
        "rows_with_any_nonmissing": int(rows_with_any.sum()),
        "all_null_rows": int((~rows_with_any).sum()),
        "row_nonmissing_fraction": row_fraction,
        "minimum_row_nonmissing_fraction": float(minimum_row_nonmissing_fraction),
        "nonmissing_cells": nonmissing_cells,
        "nonmissing_cell_fraction": (float(nonmissing_cells / total_cells) if total_cells else 0.0),
        "definitions_fingerprint": _value_fingerprint(list(definitions)),
    }
    _write_json(audit_path, audit)
    if audit["status"] != "ok":
        raise ValueError(
            f"Stage 2 final {scope} extraction is catastrophically sparse: "
            f"only {audit['rows_with_any_nonmissing']}/{audit['rows']} rows contain "
            "any retained feature value "
            f"({row_fraction:.3f} < {minimum_row_nonmissing_fraction:.3f})"
        )
    return audit


DEFAULT_MIN_INFORMATIVE_PATIENTS = 20
DEFAULT_MAX_EXTRACTION_FAILURE_FRACTION = 0.10
# Fail-closed sanity check, not a modeling rule: final matrices in which fewer
# than 5% of patients have any retained value indicate a broken extraction.
CATASTROPHIC_ROW_NONMISSING_FRACTION = 0.05
CANDIDATE_FILTER_DESCRIPTION = (
    "Outer-training candidates are kept only when (1) the fraction of training "
    "patients whose extraction failed after all repairs is at most "
    "stage2.max_extraction_failure_fraction; (2) at least "
    "stage2.min_informative_patients training patients have an observed value; and "
    "(3) the feature is not nearly constant: categorical/binary/ordinal features "
    "need at least two levels (counting 'not documented' as a level, never counting "
    "failed extractions) each with at least min_informative_patients patients, and "
    "continuous features need non-identical observed values. A documented_unknown "
    "value (the record states that the value is unknown, indeterminate, pending, not "
    "assessed, or equivocal) counts as an observed state and, for categorical features, "
    "as its own level: it is recorded patient information, encoded as its own "
    "indicator. Retained features with training extraction failures get a grouped "
    "failure indicator column."
)


def _observed_distinct_values(series: pd.Series, strategy: str) -> int:
    observed = series.dropna()
    if strategy in {"continuous", "continuous_with_categorical_fallback"}:
        numeric = pd.to_numeric(observed, errors="coerce")
        keys = [
            ("number", float(value)) if pd.notna(value) else ("text", str(raw))
            for raw, value in zip(observed.tolist(), numeric.tolist())
        ]
        return len(set(keys))
    return int(observed.astype(str).nunique())


def _filter_sparse_training_candidates(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    *,
    audit_path: Path,
    health_audit_path: Path,
    min_informative_patients: int = DEFAULT_MIN_INFORMATIVE_PATIENTS,
    max_extraction_failure_fraction: float = DEFAULT_MAX_EXTRACTION_FAILURE_FRACTION,
    failure_mask: pd.DataFrame | None = None,
    treatment: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Apply the count-based sparsity rule and the extraction-failure rule.

    Uses outer-training rows only. ``failure_mask`` (``_oci_row_id`` plus one 0/1
    column per measured feature) marks cells whose extraction failed after all
    repairs; those cells are neither "observed" nor "not documented".
    """

    if (
        isinstance(min_informative_patients, bool)
        or not isinstance(min_informative_patients, (int, np.integer))
        or int(min_informative_patients) < 1
    ):
        raise ValueError("stage2.min_informative_patients must be a positive integer")
    if (
        isinstance(max_extraction_failure_fraction, bool)
        or not isinstance(max_extraction_failure_fraction, (int, float))
        or not math.isfinite(float(max_extraction_failure_fraction))
        or not 0.0 <= float(max_extraction_failure_fraction) <= 1.0
    ):
        raise ValueError("stage2.max_extraction_failure_fraction must be in [0, 1]")
    k = int(min_informative_patients)
    max_failure = float(max_extraction_failure_fraction)
    names = [str(feature["name"]) for feature in definitions]
    # A missing column is a malformed extraction, not an all-null measurement.
    values = frame[names]
    rows = len(frame)
    if not rows:
        raise ValueError("Stage 2 candidate filtering requires training rows")
    row_ids = [int(value) for value in frame["_oci_row_id"]]
    mask_by_id = failure_mask.set_index("_oci_row_id") if failure_mask is not None else None
    if mask_by_id is not None and not set(row_ids).issubset(set(map(int, mask_by_id.index))):
        raise ValueError("extraction-failure mask does not cover the training rows")
    retained: list[dict[str, Any]] = []
    decisions = []
    failure_columns: dict[str, np.ndarray] = {}
    all_failed: dict[str, np.ndarray] = {}
    for feature in definitions:
        name = str(feature["name"])
        series = values[name]
        if mask_by_id is not None and name in mask_by_id.columns:
            failed = mask_by_id.loc[row_ids, name].fillna(0).to_numpy(dtype=float) > 0.5
        elif extraction_failure_column(name) in frame.columns:
            failed = extraction_failures.failure_flags(
                frame, {**feature, "extraction_failure_indicator": True}
            ).to_numpy(dtype=bool)
        else:
            failed = np.zeros(rows, dtype=bool)
        failed &= series.isna().to_numpy(dtype=bool)
        all_failed[name] = failed.astype(int)
        # documented_unknown cells are observed (non-null) states.
        observed_mask = series.notna().to_numpy(dtype=bool)
        observed = int(observed_mask.sum())
        unknown_rows = int(documented_unknown_mask(series).sum())
        failed_rows = int(failed.sum())
        not_documented = rows - observed - failed_rows
        failure_fraction = failed_rows / rows
        strategy = _feature_modeling_strategy(feature)
        level_counts: dict[str, int] = {}
        if strategy in {"continuous", "continuous_with_categorical_fallback"}:
            distinct = _observed_distinct_values(series, strategy)
            informative_levels = None
        else:
            level_counts = {
                str(level): int(count)
                for level, count in series.dropna().astype(str).value_counts().items()
            }
            if not_documented:
                level_counts["__not_documented__"] = int(not_documented)
            distinct = len([level for level in level_counts if level != "__not_documented__"])
            informative_levels = sum(count >= k for count in level_counts.values())
        if failure_fraction > max_failure:
            action, reason = "drop", "extraction_failure_fraction_above_maximum"
        elif observed < k:
            action, reason = "drop", "fewer_than_min_informative_patients_observed"
        elif informative_levels is None and distinct < 2:
            action, reason = "drop", "constant_observed_values"
        elif informative_levels is not None and informative_levels < 2:
            action, reason = "drop", "fewer_than_two_levels_with_min_informative_patients"
        else:
            action, reason = "keep", "informative"
        decision = {
            "feature_id": str(feature["feature_id"]),
            "name": name,
            "modeling_strategy": strategy,
            "observed_rows": observed,
            "documented_unknown_rows": unknown_rows,
            "not_documented_rows": not_documented,
            "extraction_failed_rows": failed_rows,
            "extraction_failure_fraction": failure_fraction,
            "missing_rows": rows - observed,
            "missing_fraction": (rows - observed) / rows,
            "distinct_observed_values": distinct,
            **(
                {
                    "level_counts": dict(sorted(level_counts.items())[:50]),
                    "levels_with_min_informative_patients": informative_levels,
                }
                if informative_levels is not None
                else {}
            ),
            "configured_explicit_feature": bool(feature.get("configured_explicit_feature")),
            "action": action,
            "reason": reason,
        }
        if action == "keep":
            kept = dict(feature)
            if failed_rows:
                kept["extraction_failure_indicator"] = True
                failure_columns[name] = failed.astype(int)
                decision["extraction_failure_indicator"] = True
            else:
                kept.pop("extraction_failure_indicator", None)
            retained.append(kept)
        decisions.append(decision)
    failure_report = extraction_failures.failure_rate_report(
        pd.DataFrame({"_oci_row_id": row_ids, **all_failed}),
        definitions,
        scope="outer_training",
        treatment=treatment,
    )
    unknown_report = documented_unknown.documented_unknown_report(
        frame, definitions, scope="outer_training", treatment=treatment,
    )
    summary = {
        "schema_version": "stage2_candidate_informativeness_filter_v3_documented_unknown",
        "status": "failed" if definitions and not retained else "complete",
        "scope": "outer_training",
        "rule": "count_based_informativeness_and_extraction_failure",
        "description": CANDIDATE_FILTER_DESCRIPTION,
        "min_informative_patients": k,
        "max_extraction_failure_fraction": max_failure,
        "rows": rows,
        "candidates_before": len(definitions),
        "candidates_retained": len(retained),
        "candidates_dropped": len(definitions) - len(retained),
        "dropped_feature_ids": [d["feature_id"] for d in decisions if d["action"] == "drop"],
        "dropped_by_reason": dict(
            Counter(d["reason"] for d in decisions if d["action"] == "drop")
        ),
        "extraction_failure_mask_available": failure_mask is not None,
        "retained_with_failure_indicator": sorted(failure_columns),
        "training_extraction_failure_report": failure_report,
        "training_documented_unknown_report": unknown_report,
        "input_matrix_fingerprint": _frame_fingerprint(frame),
        "audit_path": str(audit_path),
    }
    _write_json(audit_path, {**summary, "features": decisions})
    LOGGER.info(
        "Stage 2 candidate informativeness filter retained %d/%d features on %d training rows",
        len(retained), len(definitions), rows,
    )
    if definitions and not retained:
        # Preserve the existing catastrophic-extraction diagnostic. Even when
        # the union has adequate coverage, losing every candidate must not
        # silently turn an extraction failure into an unadjusted analysis.
        health = _assert_extraction_health(
            frame, definitions, scope="training",
            minimum_row_nonmissing_fraction=CATASTROPHIC_ROW_NONMISSING_FRACTION,
            audit_path=health_audit_path,
        )
        _write_json(health_audit_path, {
            **health, "status": "failed", "reason": "all_candidates_uninformative_or_unreliable",
        })
        raise ValueError(
            "Stage 2 training extraction is catastrophically sparse: no candidate "
            "passes the informativeness and extraction-failure rules"
        )
    retained_names = [str(f["name"]) for f in retained]
    result = frame[["_oci_row_id", *retained_names]].copy()
    for name in retained_names:
        if name in failure_columns:
            result[extraction_failure_column(name)] = failure_columns[name]
    return result, retained, summary


class _FeatureEncoder:
    """Fit-on-training feature encoder for nuisance models and forests.

    A feature flagged ``extraction_failure_indicator`` gets one extra 0/1
    column from its companion ``<name>__oci_extraction_failed`` column;
    failed cells are excluded from the "missing"/"not documented" encoding
    (continuous values are median-imputed, category indicators stay 0).

    A feature whose training rows contain the reserved ``documented_unknown``
    value gets one more 0/1 column, last in its group; those cells are
    excluded from the missing indicator, the fallback categories and every
    category level (continuous values are median-imputed). A held-out
    ``documented_unknown`` for a feature without training occurrences is
    encoded as missing.
    """

    def __init__(self, definitions: Sequence[Mapping[str, Any]]) -> None:
        self.definitions = list(definitions)
        self.encodings: list[tuple[str, str, Any]] = []
        self.failure_indicators: dict[str, Mapping[str, Any]] = {}
        self.unknown_indicators: set[str] = set()

    def fit(self, frame: pd.DataFrame) -> "_FeatureEncoder":
        self.encodings = []
        self.failure_indicators = {
            str(feature["name"]): feature
            for feature in self.definitions
            if extraction_failures.has_failure_indicator(feature)
        }
        self.unknown_indicators = set()
        for feature in self.definitions:
            name = str(feature["name"])
            modeling_strategy = _feature_modeling_strategy(feature)
            raw_series = frame[name] if name in frame else pd.Series([None] * len(frame))
            if bool(documented_unknown_mask(raw_series).any()):
                self.unknown_indicators.add(name)
            series = documented_unknown.without_documented_unknown(raw_series)
            if modeling_strategy in {
                "continuous",
                "continuous_with_categorical_fallback",
            }:
                numeric = pd.to_numeric(series, errors="coerce")
                median = float(numeric.median()) if numeric.notna().any() else 0.0
                scale = float(numeric.fillna(median).std(ddof=0))
                if modeling_strategy == "continuous":
                    parameters: Any = (median, scale or 1.0)
                else:
                    fallback_mask = series.notna() & numeric.isna()
                    fallback_categories = sorted(
                        str(item) for item in series.loc[fallback_mask].astype(str).unique()
                    )
                    parameters = (median, scale or 1.0, fallback_categories)
                self.encodings.append((name, modeling_strategy, parameters))
            else:
                harmonization = feature.get("harmonization_plan")
                harmonized_categories = (
                    [str(item) for item in harmonization.get("canonical_categories") or []]
                    if isinstance(harmonization, Mapping)
                    and str(harmonization.get("target_representation") or "") == "categorical"
                    else []
                )
                closed_ontology = bool(harmonized_categories) or str(
                    feature.get("value_type") or "ambiguous"
                ) in {
                    "binary",
                    "categorical",
                    "ordinal",
                }
                declared = (
                    harmonized_categories
                    if harmonized_categories
                    else _declared_categories(feature) if closed_ontology else []
                )
                observed = [str(item) for item in series.dropna().astype(str).unique()]
                categories = list(dict.fromkeys([*declared, *sorted(observed), "__missing__"]))
                encoding = "categorical" if closed_ontology else "categorical_with_other"
                self.encodings.append((name, encoding, categories))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for name, value_type, parameters in self.encodings:
            raw_series = frame[name] if name in frame else pd.Series([None] * len(frame))
            tracked = name in self.failure_indicators
            failed = (
                extraction_failures.failure_flags(frame, self.failure_indicators[name])
                .to_numpy(dtype=bool)
                if tracked and name in frame
                else np.zeros(len(frame), dtype=bool)
            )
            unknown_tracked = name in self.unknown_indicators
            unknown = documented_unknown_mask(raw_series).to_numpy(dtype=bool)
            # Without training occurrences a documented unknown is encoded as missing.
            series = documented_unknown.without_documented_unknown(raw_series)
            # Cells excluded from the missing indicator: failures and tracked unknowns.
            not_missing = failed | (unknown & unknown_tracked)
            if value_type == "continuous":
                median, scale = parameters
                numeric = pd.to_numeric(series, errors="coerce")
                missing = (numeric.isna().to_numpy(dtype=bool) & ~not_missing).astype(float)
                values = (numeric.fillna(median).to_numpy(dtype=float) - median) / scale
                columns.extend([values, missing])
            elif value_type == "continuous_with_categorical_fallback":
                median, scale, categories = parameters
                numeric = pd.to_numeric(series, errors="coerce")
                raw_missing = series.isna().to_numpy(dtype=bool) & ~not_missing
                categorical_mask = series.notna() & numeric.isna()
                normalized = series.where(categorical_mask, "").astype(str)
                values = (numeric.fillna(median).to_numpy(dtype=float) - median) / scale
                columns.extend(
                    [
                        values,
                        numeric.notna().to_numpy(dtype=float),
                        raw_missing.astype(float),
                    ]
                )
                for category in categories:
                    columns.append(
                        (categorical_mask & (normalized == category)).to_numpy(dtype=float)
                    )
                columns.append(
                    (categorical_mask & ~normalized.isin(categories)).to_numpy(dtype=float)
                )
            else:
                normalized = series.where(series.notna(), "__missing__").astype(str)
                for category in parameters:
                    columns.append(
                        ((normalized == category).to_numpy(dtype=bool) & ~not_missing).astype(
                            float
                        )
                    )
                if value_type == "categorical_with_other":
                    columns.append(
                        ((~normalized.isin(parameters)).to_numpy(dtype=bool) & ~not_missing)
                        .astype(float)
                    )
            if tracked:
                columns.append(failed.astype(float))
            if unknown_tracked:
                columns.append(unknown.astype(float))
        if not columns:
            return np.empty((len(frame), 0), dtype=float)
        return np.column_stack(columns).astype(float, copy=False)


def _definitions_for_roles(
    definitions: Sequence[Mapping[str, Any]], roles: set[str]
) -> list[Mapping[str, Any]]:
    return [
        feature
        for feature in definitions
        if set(map(str, feature.get("roles") or [])).intersection(roles)
    ]


def _definitions_for_nuisance_role(
    definitions: Sequence[Mapping[str, Any]],
    role: str,
) -> list[Mapping[str, Any]]:
    """Use the persisted treatment/outcome supports of the retained features."""

    if role not in {"treatment", "outcome"}:
        raise ValueError("nuisance role must be treatment or outcome")
    definitions = list(definitions)
    selected = [
        feature
        for feature in definitions
        if role in set(map(str, feature.get("nuisance_model_roles") or []))
    ]
    if role == "outcome":
        selected_ids = {str(feature.get("feature_id") or feature["name"]) for feature in selected}
        selected.extend(
            feature
            for feature in definitions
            if "effect_modifier" in set(map(str, feature.get("roles") or []))
            and str(feature.get("feature_id") or feature["name"]) not in selected_ids
        )
    return selected


class _ConstantClassifier:
    classes_ = np.asarray([0, 1], dtype=int)

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probability = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - probability, probability])


class _ConstantRegressor:
    def __init__(self, mean: float) -> None:
        self.mean = float(mean)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.mean, dtype=float)


def _is_super_learner(learner: Mapping[str, Any] | None) -> bool:
    if learner is None or learner.get("name") == "elastic_net":
        return False
    if learner.get("name") != "super_learner":
        raise ValueError(f"unsupported nuisance learner: {learner.get('name')!r}")
    return True


def _fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    learner: Mapping[str, Any] | None = None,
) -> Any:
    """Final nuisance classifier: elastic net (``learner`` None) or the configured stack.

    Every model is fitted inside a modeling-pool job limited to one thread.
    """
    if len(np.unique(y)) < 2 or x.shape[1] == 0:
        return _ConstantClassifier(float(np.mean(y)))
    if _is_super_learner(learner):
        model: Any = SuperLearnerClassifier(
            learners=tuple(learner["learners"]), cv_folds=int(learner["cv_folds"]),
            random_state=seed, n_jobs=1,
        )
    else:
        model = ElasticNetLogisticClassifier(random_state=seed, n_jobs=1)
    model.fit(x, y.astype(int))
    return model


def _predict_probability(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return probabilities[:, classes.index(1)].astype(float)


def _fit_regressor(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    learner: Mapping[str, Any] | None = None,
) -> Any:
    if x.shape[1] == 0:
        return _ConstantRegressor(float(np.mean(y)))
    if _is_super_learner(learner):
        model: Any = SuperLearnerRegressor(
            learners=tuple(learner["learners"]), cv_folds=int(learner["cv_folds"]),
            random_state=seed, n_jobs=1,
        )
    else:
        model = ElasticNetRegressor(random_state=seed, n_jobs=1)
    model.fit(x, y)
    return model


def _dr_score(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: np.ndarray,
    *,
    clip: float,
) -> np.ndarray:
    e = np.clip(propensity, clip, 1.0 - clip)
    score = (
        mu1
        - mu0
        + treatment * (outcome - mu1) / e
        - (1.0 - treatment) * (outcome - mu0) / (1.0 - e)
    )
    return np.where(np.isfinite(score), score, np.nan)


def _fallback_inner_splits(
    row_ids: Sequence[int], *, folds: int, seed: int
) -> list[dict[str, Any]]:
    from sklearn.model_selection import KFold

    row_ids = np.asarray(row_ids, dtype=int)
    count = min(max(2, int(folds)), len(row_ids))
    if count < 2:
        return []
    splitter = KFold(n_splits=count, shuffle=True, random_state=seed)
    return [
        {
            "inner_fold": index,
            "fit_row_ids": row_ids[fit].tolist(),
            "heldout_row_ids": row_ids[heldout].tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(row_ids), start=1)
    ]


def _ontology_refinement_prompt(*, feature: Mapping[str, Any], failure_patterns: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    patterns = [{k: copy.deepcopy(p.get(k)) for k in ("failure_kind", "reason", "patient_count", "example_values", "allowed_categories")} for p in failure_patterns]
    return clinical_prompts.messages("09_refine_ontology", clinical_prompts.feature_text(feature)
        + "\n\nFailed answers\n" + clinical_prompts.readable(patterns))


def _validate_ontology_refinement(
    value: Mapping[str, Any],
    *,
    feature: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a clarification or an explicit decomposition into scalar features."""
    if value.get("action") == "revise" and "definition" in value:
        fields = {"description", "value_type", "categories_or_unit", "measurement_definition", "missing_value_rule"}
        if set(value) != {"action", "reason", "definition"} or not isinstance(value["definition"], Mapping) or set(value["definition"]) != fields:
            raise ValueError("a revision requires reason and a complete clinical definition")
        value = {"action": "revise", "reason": value["reason"], **value["definition"]}


    if feature.get("configured_explicit_feature") is True:
        raise ValueError("investigator-configured feature ontology cannot be revised")
    action = str(value.get("action") or "").strip().lower()
    if action not in {"keep", "revise", "split"}:
        raise ValueError("ontology refinement action must be keep, revise, or split")
    decision: dict[str, Any] = {
        "feature_id": str(feature["feature_id"]),
        "feature_name": str(feature["name"]),
        "action": action,
        "reason": str(value.get("reason") or "").strip(),
    }
    if action == "keep":
        return decision

    if action == "split":
        components = value.get("components")
        if not isinstance(components, list) or not 2 <= len(components) <= 20:
            raise ValueError("split requires 2 to 20 independently measurable components")
        normalized_components = []
        seen = set()
        for component in components:
            if not isinstance(component, Mapping):
                raise ValueError("each split component must be a clinical definition")
            name = str(component.get("name") or "").strip()
            key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            if not key or key in seen:
                raise ValueError("split component names must be distinct clinical names")
            seen.add(key)
            normalized = _validate_ontology_refinement(
                {**component, "action": "revise", "reason": decision["reason"]},
                feature=feature,
            )
            if normalized["value_type"] == "binary":
                missingness_contract = (
                    "Return null when this component is not mentioned, and "
                    "documented_unknown when the record states that it was not assessed "
                    "or is unknown. A negative answer requires an explicit negative "
                    "finding for this component; never infer it from other components "
                    "being present. "
                )
                if not normalized["missing_value_rule"].startswith(missingness_contract):
                    normalized["missing_value_rule"] = missingness_contract + normalized["missing_value_rule"]
            normalized_components.append({
                "name": name,
                **{k: normalized[k] for k in ONTOLOGY_DEFINITION_FIELDS},
            })
        decision["components"] = normalized_components
        return decision

    description = str(value.get("description") or "").strip()
    value_type = str(value.get("value_type") or "").strip().lower()
    categories = value.get("categories_or_unit")
    measurement_definition = str(value.get("measurement_definition") or "").strip()
    missing_value_rule = str(value.get("missing_value_rule") or "").strip()
    if not description:
        raise ValueError("a revised ontology requires description")
    if value_type not in {"binary", "categorical", "continuous", "ordinal", "ambiguous"}:
        raise ValueError("a revised ontology requires an operational value_type")
    if not isinstance(categories, list):
        raise ValueError("a revised ontology requires a categories_or_unit array")
    clean_categories = [str(item).strip() for item in categories if str(item).strip()]
    if value_type in {"binary", "categorical", "ordinal"}:
        clean_categories = _validated_closed_category_values(
            value_type=value_type,
            values=clean_categories,
            source=f"refined feature {feature['feature_id']!r}",
        )
    elif value_type == "ambiguous" and clean_categories:
        raise ValueError("an ambiguous scalar has no declared categories or unit")
    elif value_type == "continuous" and len(clean_categories) > 1:
        raise ValueError("a revised continuous ontology may name at most one unit")
    if not measurement_definition:
        raise ValueError("a revised ontology requires measurement_definition")
    if not missing_value_rule:
        raise ValueError("a revised ontology requires missing_value_rule")
    decision.update(
        {
            "description": description,
            "value_type": value_type,
            "categories_or_unit": clean_categories,
            "measurement_definition": measurement_definition,
            "missing_value_rule": missing_value_rule,
        }
    )
    return decision


ONTOLOGY_DEFINITION_FIELDS = (
    "description", "value_type", "categories_or_unit", "measurement_definition",
    "missing_value_rule",
)


def _apply_ontology_decision(feature, decision):
    """Keep provenance and stable identities when replacing a composite variable."""
    if decision is None or decision["action"] == "keep":
        return [dict(feature)]
    if decision["action"] == "split":
        result = []
        for component in decision["components"]:
            # Parent identity namespaces components, so independent reviews cannot
            # collide with each other or overwrite existing investigator features.
            key = re.sub(r"[^a-z0-9]+", "_", component["name"].lower()).strip("_")
            child = {**feature, **{k: component[k] for k in ONTOLOGY_DEFINITION_FIELDS},
                     "name": f"{feature['name']}__{key}",
                     "feature_id": f"{feature['feature_id']}__{key}",
                     "clinical_label": f"{clinical_prompts.label(feature)}: {component['name']}",
                     "ontology_parent_feature_id": feature["feature_id"],
                     "ontology_split_reason": decision["reason"]}
            for field in ("harmonization_plan", "harmonization_fallback"):
                child.pop(field, None)
            result.append(_refresh_conflict_resolution(child))
        return result
    revised = {**feature, **{k: decision[k] for k in ONTOLOGY_DEFINITION_FIELDS}}
    if all(revised.get(k) == feature.get(k) for k in ONTOLOGY_DEFINITION_FIELDS):
        return [dict(feature)]
    for field in ("harmonization_plan", "harmonization_fallback"):
        revised.pop(field, None)
    return [_refresh_conflict_resolution(revised)]


def _request_scalar_ontology_decision(*, messages, feature, request_json, output_dir):
    """Independently check semantic type changes before cohort-wide extraction."""
    original_messages = list(messages)
    checks = []
    for attempt in range(3):
        decision = request_json(messages, lambda v: _validate_ontology_refinement(v, feature=feature),
                                request_kind="interpretation")
        risky_conversion = (decision["action"] == "revise"
            and feature.get("value_type") == "ambiguous"
            and decision["value_type"] in {"binary", "categorical", "ordinal"})
        if decision["action"] != "split" and not risky_conversion:
            return decision
        proposed = _apply_ontology_decision(feature, decision)

        def validate_check(value):
            if set(value) != {"valid", "reason"} or not isinstance(value["valid"], bool):
                raise ValueError("return valid (boolean) and reason (text)")
            if not isinstance(value["reason"], str) or not value["reason"].strip():
                raise ValueError("explain whether the proposed scalar contract preserves the measurement")
            return dict(value)

        check = request_json(clinical_prompts.messages("ontology_scalar_contract",
            "Original variable\n" + clinical_prompts.feature_text(feature)
            + "\n\nProposed definitions\n" + clinical_prompts.definitions_text(proposed)),
            validate_check, request_kind="interpretation")
        checks.append({"attempt": attempt + 1, "decision": decision, "check": check})
        _write_json(Path(output_dir) / "scalar_contract_checks.json", {"checks": checks})
        if check["valid"]:
            return decision
        messages = [*original_messages, {"role": "assistant", "content": json.dumps(decision)},
                    {"role": "user", "content": "The proposed definition failed its scalar contract check: "
                     + check["reason"] + " Correct the definition or split it into independent scalar variables. "
                     "Ambiguous is an allowed type; never invent category labels to satisfy a schema."}]
    # Keeping the original measurements is safer than replacing them with a
    # definition known to be incompatible. This is explicit in the audit.
    return {"feature_id": feature["feature_id"], "feature_name": feature["name"],
            "action": "keep", "reason": "Scalar contract check rejected proposed changes: " + checks[-1]["check"]["reason"],
            "validation_fallback": True}


def _aggregate_ontology_supervisor_prompt(*, feature: Mapping[str, Any], summary: Mapping[str, Any],
    failure_patterns: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    patterns = [{k: copy.deepcopy(p.get(k)) for k in ("failure_kind", "reason", "patient_count", "example_values", "allowed_categories")} for p in failure_patterns]
    return clinical_prompts.messages("12_supervise_ontology", clinical_prompts.feature_text(feature)
        + "\n\nCurrent extracted values\n" + clinical_prompts.readable(summary)
        + "\n\nEarlier failed answers (patients can also appear in the current summary)\n"
        + clinical_prompts.readable(patterns))


def _request_aggregate_ontology_supervisor(
    *,
    definitions: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    failure_summary: Mapping[str, Any],
    output_dir: Path,
    request_json: RequestJSON,
    workers: int,
    request_identity: Mapping[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    """Review changed schemas and reuse identical feature reviews across rounds."""

    summaries_by_id = {str(row["feature_id"]): dict(row) for row in summaries}
    failures_by_name: dict[str, list[dict[str, Any]]] = {}
    for pattern in failure_summary.get("feature_failure_patterns") or []:
        if isinstance(pattern, Mapping) and str(pattern.get("feature_name") or ""):
            failures_by_name.setdefault(str(pattern["feature_name"]), []).append(dict(pattern))
    identity = dict(request_identity or {})
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_cache_dir = (
        Path(cache_dir)
        if cache_dir is not None
        else output_dir.parent.parent / "supervisor_cache"
    )

    # Adopt valid leaf checkpoints from earlier rounds as well as the new
    # content-addressed cache.  This makes the optimization effective for a
    # run that was started before the cache directory existed.
    cached_dirs_by_fingerprint: dict[str, Path] = {}
    supervision_root = output_dir.parent.parent
    checkpoint_paths = [
        *sorted(supervision_root.glob("round_*/supervisor/feature_*/complete.json")),
        *sorted(shared_cache_dir.glob("*/*/complete.json")),
    ]
    for cached_complete_path in checkpoint_paths:
        cached_feature_dir = cached_complete_path.parent
        if cached_feature_dir == output_dir or output_dir in cached_feature_dir.parents:
            continue
        try:
            cached_completion = json.loads(
                cached_complete_path.read_text(encoding="utf-8")
            )
            cached_fingerprint = str(
                cached_completion.get("input_fingerprint") or ""
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if cached_fingerprint:
            cached_dirs_by_fingerprint.setdefault(
                cached_fingerprint,
                cached_feature_dir,
            )

    jobs: list[tuple[int, dict[str, Any]]] = []
    decisions: dict[str, dict[str, Any]] = {}
    for index, raw_feature in enumerate(definitions, start=1):
        feature = dict(raw_feature)
        feature_id = str(feature["feature_id"])
        if feature.get("configured_explicit_feature") is True:
            decisions[feature_id] = {
                "feature_id": feature_id,
                "feature_name": str(feature["name"]),
                "action": "keep",
                "reason": "Investigator-specified ontology is locked.",
                "configured_explicit_feature": True,
            }
        else:
            jobs.append((index, feature))

    def cached_decision(
        directory: Path,
        *,
        fingerprint: str,
        feature: Mapping[str, Any],
        permit_validation_fallback: bool,
    ) -> dict[str, Any] | None:
        result_path = directory / "result.json"
        complete_path = directory / "complete.json"
        if not result_path.is_file() or not complete_path.is_file():
            return None
        try:
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            if completion.get("input_fingerprint") != fingerprint:
                return None
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            validated = _validate_ontology_refinement(cached, feature=feature)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not permit_validation_fallback and validated.get("validation_fallback") is True:
            return None
        return validated

    def write_checkpoint(
        directory: Path,
        *,
        input_value: Mapping[str, Any],
        fingerprint: str,
        decision: Mapping[str, Any],
        reuse_source: str | None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            directory / "input.json",
            {**dict(input_value), "input_fingerprint": fingerprint},
        )
        _write_json(directory / "result.json", decision)
        _write_json(
            directory / "complete.json",
            {
                "status": "complete",
                "schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
                "input_fingerprint": fingerprint,
                "completed_at": _now(),
                "action": decision["action"],
                "reused": reuse_source is not None,
                "reuse_source": reuse_source,
            },
        )

    def request_one(
        job: tuple[int, dict[str, Any]],
    ) -> tuple[str, dict[str, Any], str]:
        index, feature = job
        feature_id = str(feature["feature_id"])
        summary = summaries_by_id.get(feature_id, {})
        failures = failures_by_name.get(str(feature["name"]), [])
        input_value = {
            "schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
            "primary_request_identity": identity,
            "feature": {
                key: copy.deepcopy(feature.get(key))
                for key in (
                    "feature_id",
                    "name",
                    "description",
                    "value_type",
                    "categories_or_unit",
                    "measurement_definition",
                    "missing_value_rule",
                )
            },
            "aggregate_extraction_summary": summary,
            "aggregate_validation_failures": failures,
        }
        fingerprint = _value_fingerprint(input_value)
        feature_dir = output_dir / f"feature_{index:04d}"
        local = cached_decision(
            feature_dir,
            fingerprint=fingerprint,
            feature=feature,
            permit_validation_fallback=True,
        )
        if local is not None:
            return feature_id, local, "local_checkpoint"

        cache_feature_dir = (
            shared_cache_dir / fingerprint[:2] / fingerprint
        )
        reusable = cached_decision(
            cache_feature_dir,
            fingerprint=fingerprint,
            feature=feature,
            permit_validation_fallback=False,
        )
        reuse_source = "content_addressed_cache"
        if reusable is None:
            adopted_dir = cached_dirs_by_fingerprint.get(fingerprint)
            if adopted_dir is not None:
                reusable = cached_decision(
                    adopted_dir,
                    fingerprint=fingerprint,
                    feature=feature,
                    permit_validation_fallback=False,
                )
                reuse_source = "prior_round_checkpoint"
        if reusable is not None:
            write_checkpoint(
                feature_dir,
                input_value=input_value,
                fingerprint=fingerprint,
                decision=reusable,
                reuse_source=reuse_source,
            )
            if not (cache_feature_dir / "complete.json").is_file():
                write_checkpoint(
                    cache_feature_dir,
                    input_value=input_value,
                    fingerprint=fingerprint,
                    decision=reusable,
                    reuse_source=reuse_source,
                )
            return feature_id, reusable, reuse_source

        feature_dir.mkdir(parents=True, exist_ok=True)
        _write_json(feature_dir / "input.json", {**input_value, "input_fingerprint": fingerprint})
        try:
            decision = _request_scalar_ontology_decision(
                messages=_aggregate_ontology_supervisor_prompt(
                    feature=feature,
                    summary=summary,
                    failure_patterns=failures,
                ),
                feature=feature, request_json=request_json, output_dir=feature_dir,
            )
        except Stage2ResponseValidationError as exc:
            decision = {
                "feature_id": feature_id,
                "feature_name": str(feature["name"]),
                "action": "keep",
                "reason": f"Invalid supervisor response; conservative keep: {exc}",
                "validation_fallback": True,
            }
        write_checkpoint(
            feature_dir,
            input_value=input_value,
            fingerprint=fingerprint,
            decision=decision,
            reuse_source=None,
        )
        if decision.get("validation_fallback") is not True:
            write_checkpoint(
                cache_feature_dir,
                input_value=input_value,
                fingerprint=fingerprint,
                decision=decision,
                reuse_source=None,
            )
        return feature_id, decision, "model"

    request_sources: Counter[str] = Counter()
    if jobs:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(int(workers), len(jobs)))
        ) as executor:
            futures = [executor.submit(request_one, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                feature_id, decision, source = future.result()
                decisions[feature_id] = decision
                request_sources[source] += 1

    updated: list[dict[str, Any]] = []
    changed_ids: list[str] = []
    for raw_feature in definitions:
        feature = dict(raw_feature)
        decision = decisions[str(feature["feature_id"])]
        replacements = _apply_ontology_decision(feature, decision)
        if replacements != [feature]:
            changed_ids.extend(str(f["feature_id"]) for f in replacements)
        updated.extend(replacements)

    report = {
        "schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
        "completed_at": _now(),
        "features_reviewed": len(definitions),
        "review_candidate_features": len(jobs),
        "model_requested_features": int(request_sources["model"]),
        "cache_reused_features": int(
            request_sources["content_addressed_cache"]
            + request_sources["prior_round_checkpoint"]
        ),
        "local_checkpoint_reused_features": int(
            request_sources["local_checkpoint"]
        ),
        "review_request_sources": dict(sorted(request_sources.items())),
        "changed_feature_ids": changed_ids,
        "decisions": [decisions[str(feature["feature_id"])] for feature in definitions],
        "prohibited_information_supplied": False,
    }
    _write_json(output_dir / "result.json", {"definitions": updated, **report})
    _write_json(output_dir / "complete.json", {"status": "complete", **report})
    return updated, bool(changed_ids), report


def _repeated_ontology_failure_patterns(
    summary: Mapping[str, Any],
    *,
    minimum_patients: int,
) -> dict[str, list[dict[str, Any]]]:
    """Select feature-specific patterns repeated across enough distinct patients."""

    selected: dict[str, list[dict[str, Any]]] = {}
    for raw_pattern in summary.get("feature_failure_patterns") or []:
        if not isinstance(raw_pattern, Mapping):
            continue
        if int(raw_pattern.get("patient_count") or 0) < int(minimum_patients):
            continue
        feature_name = str(raw_pattern.get("feature_name") or "")
        if feature_name:
            selected.setdefault(feature_name, []).append(dict(raw_pattern))
    return selected


def _request_ontology_refinements(
    *,
    definitions: Sequence[Mapping[str, Any]],
    repeated_patterns: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    request_json: RequestJSON,
    workers: int,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    """Checkpoint independent ontology decisions and preserve explicit definitions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    features_by_name = {str(feature["name"]): dict(feature) for feature in definitions}
    unknown_names = sorted(set(repeated_patterns) - set(features_by_name))
    if unknown_names:
        raise ValueError(
            "ontology refinement received failure patterns for unknown feature names: "
            f"{unknown_names}"
        )
    input_value = {
        "schema_version": ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION,
        "definitions": list(definitions),
        "repeated_failure_patterns": {
            name: list(patterns) for name, patterns in repeated_patterns.items()
        },
    }
    input_fingerprint = _value_fingerprint(input_value)
    _write_json(
        output_dir / "input.json",
        {**input_value, "input_fingerprint": input_fingerprint},
    )

    decisions_by_name: dict[str, dict[str, Any]] = {}
    jobs: list[tuple[int, str, dict[str, Any], list[dict[str, str]]]] = []
    for index, feature in enumerate(definitions, start=1):
        name = str(feature["name"])
        if name not in repeated_patterns:
            continue
        if feature.get("configured_explicit_feature") is True:
            decisions_by_name[name] = {
                "feature_id": str(feature["feature_id"]),
                "feature_name": name,
                "action": "keep",
                "reason": (
                    "Repeated failures were recorded, but the investigator-configured "
                    "ontology is immutable."
                ),
                "configured_explicit_feature": True,
            }
            continue
        jobs.append(
            (
                index,
                name,
                dict(feature),
                _ontology_refinement_prompt(
                    feature=feature,
                    failure_patterns=repeated_patterns[name],
                ),
            )
        )

    def request_one(
        job: tuple[int, str, dict[str, Any], list[dict[str, str]]],
    ) -> tuple[str, dict[str, Any]]:
        index, name, feature, messages = job
        feature_dir = output_dir / f"feature_{index:03d}"
        feature_input = {
            "schema_version": ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION,
            "feature": feature,
            "failure_patterns": list(repeated_patterns[name]),
        }
        feature_fingerprint = _value_fingerprint(feature_input)
        result_path = feature_dir / "result.json"
        complete_path = feature_dir / "complete.json"
        if result_path.is_file() and complete_path.is_file():
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                if completion.get("input_fingerprint") == feature_fingerprint:
                    return name, _validate_ontology_refinement(cached, feature=feature)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        feature_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            feature_dir / "input.json",
            {**feature_input, "input_fingerprint": feature_fingerprint},
        )
        try:
            decision = _request_scalar_ontology_decision(
                messages=messages, feature=feature, request_json=request_json,
                output_dir=feature_dir,
            )
        except Stage2ResponseValidationError as exc:
            decision = {
                "feature_id": str(feature["feature_id"]),
                "feature_name": name,
                "action": "keep",
                "reason": (
                    "Ontology refinement response remained invalid; retained the prior "
                    f"ontology: {type(exc).__name__}: {exc}"
                ),
                "validation_fallback": True,
            }
            _write_json(
                feature_dir / "fallback.json",
                {
                    "status": "conservative_keep",
                    "completed_at": _now(),
                    "validation_error": f"{type(exc).__name__}: {exc}",
                },
            )
        _write_json(result_path, decision)
        _write_json(
            complete_path,
            {
                "status": "complete",
                "schema_version": ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION,
                "input_fingerprint": feature_fingerprint,
                "completed_at": _now(),
                "action": decision["action"],
            },
        )
        return name, decision

    if jobs:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(int(workers), len(jobs)))
        ) as executor:
            futures = {executor.submit(request_one, job): job[1] for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                name, decision = future.result()
                decisions_by_name[name] = decision

    updated: list[dict[str, Any]] = []
    changed_names: list[str] = []
    for raw_feature in definitions:
        feature = dict(raw_feature)
        name = str(feature["name"])
        decision = decisions_by_name.get(name)
        replacements = _apply_ontology_decision(feature, decision)
        if replacements != [feature]:
            changed_names.extend(f["name"] for f in replacements)
        updated.extend(replacements)

    ordered_decisions = [
        decisions_by_name[str(feature["name"])]
        for feature in definitions
        if str(feature["name"]) in decisions_by_name
    ]
    report = {
        "schema_version": ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION,
        "input_fingerprint": input_fingerprint,
        "completed_at": _now(),
        "triggered_features": len(repeated_patterns),
        "model_requested_features": len(jobs),
        "immutable_explicit_features": sum(
            decision.get("configured_explicit_feature") is True for decision in ordered_decisions
        ),
        "changed_feature_names": changed_names,
        "decisions": ordered_decisions,
    }
    _write_json(output_dir / "result.json", {"definitions": updated, **report})
    _write_json(output_dir / "complete.json", {"status": "complete", **report})
    return updated, bool(changed_names), report


def _feature_extraction_fingerprint(feature: Mapping[str, Any]) -> str:
    """Fingerprint only fields that can change a patient extraction prompt."""

    return _value_fingerprint(_prompt_feature_definitions([feature])[0])


def _features_requiring_reextraction(
    *,
    prior_definitions: Sequence[Mapping[str, Any]],
    definitions: Sequence[Mapping[str, Any]],
    prior_extracted: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return current definitions whose patient-measurement prompt changed."""

    prior_by_id: dict[str, Mapping[str, Any]] = {}
    for feature in prior_definitions:
        feature_id = str(feature["feature_id"])
        if feature_id in prior_by_id:
            raise ValueError(f"duplicate prior feature_id {feature_id!r}")
        prior_by_id[feature_id] = feature

    changed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for raw_feature in definitions:
        feature = dict(raw_feature)
        feature_id = str(feature["feature_id"])
        name = str(feature["name"])
        if feature_id in seen_ids:
            raise ValueError(f"duplicate current feature_id {feature_id!r}")
        if name in seen_names:
            raise ValueError(f"duplicate current feature name {name!r}")
        seen_ids.add(feature_id)
        seen_names.add(name)
        prior = prior_by_id.get(feature_id)
        if (
            prior is None
            or str(prior.get("name") or "") != name
            or name not in prior_extracted.columns
            or _feature_extraction_fingerprint(prior)
            != _feature_extraction_fingerprint(feature)
        ):
            changed.append(feature)
    return changed


def _validated_extraction_index(
    frame: pd.DataFrame,
    *,
    row_ids: Sequence[int],
    required_feature_names: Sequence[str],
    source: str,
) -> pd.DataFrame:
    """Validate and index one extraction matrix without changing row identity."""

    if "_oci_row_id" not in frame.columns:
        raise ValueError(f"{source} extraction is missing _oci_row_id")
    missing = [name for name in required_feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"{source} extraction is missing feature columns: {missing}")
    indexed = frame.copy()
    try:
        indexed["_oci_row_id"] = indexed["_oci_row_id"].map(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} extraction has invalid row identifiers") from exc
    if indexed["_oci_row_id"].duplicated().any():
        raise ValueError(f"{source} extraction has duplicate row identifiers")
    expected = [int(row_id) for row_id in row_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("refinement extraction received duplicate requested row identifiers")
    observed = set(indexed["_oci_row_id"].tolist())
    if observed != set(expected):
        raise ValueError(
            f"{source} extraction row identifiers do not match the requested patients"
        )
    return indexed.set_index("_oci_row_id", drop=False).loc[expected]


def _merge_incremental_failure_summaries(
    *,
    prior_summary: Mapping[str, Any],
    delta_summary: Mapping[str, Any],
    current_feature_names: Sequence[str],
    changed_feature_names: Sequence[str],
) -> dict[str, Any]:
    """Replace changed-feature failures while retaining reused-feature provenance."""

    current = set(map(str, current_feature_names))
    changed = set(map(str, changed_feature_names))
    if not changed.issubset(current):
        raise ValueError("changed refinement features are not present in current definitions")

    patterns: list[dict[str, Any]] = []
    for raw_pattern in prior_summary.get("feature_failure_patterns") or []:
        if not isinstance(raw_pattern, Mapping):
            continue
        name = str(raw_pattern.get("feature_name") or "")
        if name in current and name not in changed:
            patterns.append(copy.deepcopy(dict(raw_pattern)))
    for raw_pattern in delta_summary.get("feature_failure_patterns") or []:
        if not isinstance(raw_pattern, Mapping):
            continue
        name = str(raw_pattern.get("feature_name") or "")
        if name not in changed:
            raise ValueError(
                "delta extraction reported a failure for a feature that was not "
                f"re-extracted: {name!r}"
            )
        patterns.append(copy.deepcopy(dict(raw_pattern)))
    patterns.sort(
        key=lambda pattern: (
            -int(pattern.get("patient_count") or 0),
            str(pattern.get("feature_name") or ""),
            str(pattern.get("failure_kind") or ""),
            str(pattern.get("reason") or ""),
        )
    )

    structural_rows = {
        int(row_id)
        for summary in (prior_summary, delta_summary)
        for row_id in summary.get("structural_failure_patient_row_ids") or []
    }
    prior_cells = extraction_failures.terminal_failure_cells(prior_summary)
    delta_cells = extraction_failures.terminal_failure_cells(delta_summary)
    unexpected = sorted(set(delta_cells) - changed)
    if unexpected:
        raise ValueError(
            "delta extraction reported failed cells for features that were not "
            f"re-extracted: {unexpected}"
        )
    merged_cells = {
        name: sorted(delta_cells.get(name, set()) if name in changed else prior_cells.get(name, set()))
        for name in sorted(current)
        if (delta_cells.get(name) if name in changed else prior_cells.get(name))
    }
    return {
        "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
        "completed_at": _now(),
        "issue_files": int(prior_summary.get("issue_files") or 0)
        + int(delta_summary.get("issue_files") or 0),
        "feature_failure_patterns": patterns,
        **({"decision_feature_summary": {
            name: copy.deepcopy(
                (delta_summary if name in changed else prior_summary)
                .get("decision_feature_summary", {}).get(name, {})
            ) for name in sorted(current)
        }} if "decision_feature_summary" in prior_summary or "decision_feature_summary" in delta_summary else {}),
        "structural_failure_patient_count": len(structural_rows),
        "structural_failure_patient_row_ids": sorted(structural_rows),
        "terminal_failure_cells": merged_cells,
        "terminal_failure_policy": extraction_failures.TERMINAL_FAILURE_POLICY,
        "incremental_refinement": {
            "schema_version": INCREMENTAL_REFINEMENT_EXTRACTION_SCHEMA_VERSION,
            "reextracted_feature_names": sorted(changed),
            "reused_feature_names": sorted(current - changed),
            "structural_failure_policy": "union_reused_and_delta_patient_rows",
        },
    }


def _extract_changed_features_and_merge(
    *,
    dataset: pd.DataFrame,
    row_ids: Sequence[int],
    text_column: str,
    definitions: Sequence[Mapping[str, Any]],
    prior_extracted: pd.DataFrame,
    prior_definitions: Sequence[Mapping[str, Any]],
    prior_failure_summary: Mapping[str, Any],
    output_dir: Path,
    request_json: RequestJSON,
    workers: int,
    max_prompt_chars: int,
    feature_batch_size: int,
    request_identity: Mapping[str, Any] | None,
    tokenizer: Any | None,
    chunk_size_tokens: int,
    context_window_tokens: int,
    max_output_tokens: int,
    context_margin_tokens: int,
    deferred_retry_passes: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract changed features only and materialize a complete merged matrix."""

    current = [dict(feature) for feature in definitions]
    current_names = [str(feature["name"]) for feature in current]

    changed_definitions = _features_requiring_reextraction(
        prior_definitions=prior_definitions,
        definitions=current,
        prior_extracted=prior_extracted,
    )
    changed_names = [str(feature["name"]) for feature in changed_definitions]
    delta_dir = output_dir / "changed_features"
    LOGGER.info(
        "Stage 2 incremental refinement re-extracting features=%s/%s names=%s",
        len(changed_definitions),
        len(current),
        changed_names,
    )
    delta = extract_rows(
        dataset=dataset,
        row_ids=row_ids,
        text_column=text_column,
        definitions=changed_definitions,
        output_dir=delta_dir,
        request_json=request_json,
        workers=workers,
        max_prompt_chars=max_prompt_chars,
        feature_batch_size=feature_batch_size,
        request_identity=request_identity,
        tokenizer=tokenizer,
        chunk_size_tokens=chunk_size_tokens,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        context_margin_tokens=context_margin_tokens,
        deferred_retry_passes=deferred_retry_passes,
    )
    prior_names = [str(feature["name"]) for feature in prior_definitions]
    prior_indexed = _validated_extraction_index(
        prior_extracted,
        row_ids=row_ids,
        required_feature_names=[
            name for name in current_names if name not in set(changed_names)
        ],
        source="prior",
    )
    delta_indexed = _validated_extraction_index(
        delta,
        row_ids=row_ids,
        required_feature_names=changed_names,
        source="delta",
    )
    ordered_row_ids = [int(row_id) for row_id in row_ids]
    merged = pd.DataFrame({"_oci_row_id": ordered_row_ids})
    changed = set(changed_names)
    for name in current_names:
        source = delta_indexed if name in changed else prior_indexed
        merged[name] = source.loc[ordered_row_ids, name].tolist()

    delta_summary = json.loads(
        (delta_dir / "failure_summary.json").read_text(encoding="utf-8")
    )
    merged_summary = _merge_incremental_failure_summaries(
        prior_summary=prior_failure_summary,
        delta_summary=delta_summary,
        current_feature_names=current_names,
        changed_feature_names=changed_names,
    )
    merge_input = {
        "schema_version": INCREMENTAL_REFINEMENT_EXTRACTION_SCHEMA_VERSION,
        "prior_definition_fingerprints": {
            str(feature["feature_id"]): _feature_extraction_fingerprint(feature)
            for feature in prior_definitions
        },
        "current_definition_fingerprints": {
            str(feature["feature_id"]): _feature_extraction_fingerprint(feature)
            for feature in current
        },
        "prior_feature_names": prior_names,
        "current_feature_names": current_names,
        "reextracted_feature_names": changed_names,
        "prior_frame_fingerprint": _frame_fingerprint(prior_extracted),
        "delta_frame_fingerprint": _frame_fingerprint(delta),
    }
    input_fingerprint = _value_fingerprint(merge_input)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "incremental_merge_input.json",
        {**merge_input, "input_fingerprint": input_fingerprint},
    )
    _write_frame(output_dir / "extracted.csv", merged)
    _write_json(output_dir / "failure_summary.json", merged_summary)
    delta_completion = json.loads(
        (delta_dir / "complete.json").read_text(encoding="utf-8")
    )
    _write_json(
        output_dir / "complete.json",
        {
            "status": "complete",
            "schema_version": INCREMENTAL_REFINEMENT_EXTRACTION_SCHEMA_VERSION,
            "input_fingerprint": input_fingerprint,
            "completed_at": _now(),
            "rows": len(merged),
            "features": len(current),
            "reextraction_scope": "changed_features_only",
            "reextracted_features": len(changed_names),
            "reextracted_feature_names": changed_names,
            "reused_features": len(current) - len(changed_names),
            "feature_batch_size": feature_batch_size,
            "feature_batches_per_patient": int(
                delta_completion.get("feature_batches_per_patient") or 0
            ),
            "batches": int(delta_completion.get("batches") or len(row_ids)),
            "paged_rows": int(delta_completion.get("paged_rows") or 0),
            "pages": int(delta_completion.get("pages") or 0),
            "serial_patient_feature_passes": int(
                delta_completion.get("serial_patient_feature_passes") or 0
            ),
            "feature_failure_patterns": len(
                merged_summary["feature_failure_patterns"]
            ),
            "structural_failure_patients": merged_summary[
                "structural_failure_patient_count"
            ],
            "delta_extraction_dir": str(delta_dir),
        },
    )
    return merged, merged_summary


def _extract_training_with_ontology_feedback(
    *,
    dataset: pd.DataFrame,
    row_ids: Sequence[int],
    text_column: str,
    definitions: Sequence[Mapping[str, Any]],
    output_dir: Path,
    feedback_dir: Path,
    request_json: RequestJSON,
    workers: int,
    max_prompt_chars: int,
    feature_batch_size: int,
    minimum_failure_patients: int,
    max_refinement_rounds: int,
    request_identity: Mapping[str, Any] | None = None,
    tokenizer: Any | None = None,
    chunk_size_tokens: int = DEFAULT_EXTRACTION_CHUNK_SIZE_TOKENS,
    context_window_tokens: int = DEFAULT_EXTRACTION_CONTEXT_WINDOW_TOKENS,
    max_output_tokens: int = DEFAULT_EXTRACTION_MAX_TOKENS,
    context_margin_tokens: int = DEFAULT_EXTRACTION_CONTEXT_MARGIN_TOKENS,
    deferred_retry_passes: int = 1,
    interpretation_workers: int = 1,
    prior_extracted: pd.DataFrame | None = None,
    prior_definitions: Sequence[Mapping[str, Any]] | None = None,
    prior_failure_summary: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """Extract training rows and incrementally repair changed feature ontologies."""

    supplied_prior = (
        prior_extracted is not None,
        prior_definitions is not None,
        prior_failure_summary is not None,
    )
    if any(supplied_prior) and not all(supplied_prior):
        raise ValueError(
            "incremental refinement requires prior_extracted, prior_definitions, "
            "and prior_failure_summary together"
        )

    current = [dict(feature) for feature in definitions]
    extraction_dir = output_dir
    rounds: list[dict[str, Any]] = []
    stopped_reason = "maximum_refinement_rounds_reached"
    extracted: pd.DataFrame | None = None
    summary: dict[str, Any] | None = None
    source_extracted = prior_extracted
    source_definitions = (
        [dict(feature) for feature in prior_definitions]
        if prior_definitions is not None
        else None
    )
    source_summary = (
        copy.deepcopy(dict(prior_failure_summary))
        if prior_failure_summary is not None
        else None
    )
    for pass_index in range(0, int(max_refinement_rounds) + 1):
        if source_extracted is None:
            extracted = extract_rows(
                dataset=dataset,
                row_ids=row_ids,
                text_column=text_column,
                definitions=current,
                output_dir=extraction_dir,
                request_json=request_json,
                workers=workers,
                max_prompt_chars=max_prompt_chars,
                feature_batch_size=feature_batch_size,
                request_identity=request_identity,
                tokenizer=tokenizer,
                chunk_size_tokens=chunk_size_tokens,
                context_window_tokens=context_window_tokens,
                max_output_tokens=max_output_tokens,
                context_margin_tokens=context_margin_tokens,
                deferred_retry_passes=deferred_retry_passes,
            )
            summary = json.loads(
                (extraction_dir / "failure_summary.json").read_text(encoding="utf-8")
            )
        else:
            if source_definitions is None or source_summary is None:  # pragma: no cover
                raise RuntimeError("incremental refinement source state is incomplete")
            extracted, summary = _extract_changed_features_and_merge(
                dataset=dataset,
                row_ids=row_ids,
                text_column=text_column,
                definitions=current,
                prior_extracted=source_extracted,
                prior_definitions=source_definitions,
                prior_failure_summary=source_summary,
                output_dir=extraction_dir,
                request_json=request_json,
                workers=workers,
                max_prompt_chars=max_prompt_chars,
                feature_batch_size=feature_batch_size,
                request_identity=request_identity,
                tokenizer=tokenizer,
                chunk_size_tokens=chunk_size_tokens,
                context_window_tokens=context_window_tokens,
                max_output_tokens=max_output_tokens,
                context_margin_tokens=context_margin_tokens,
                deferred_retry_passes=deferred_retry_passes,
            )
        repeated = _repeated_ontology_failure_patterns(
            summary,
            minimum_patients=minimum_failure_patients,
        )
        if not repeated:
            stopped_reason = "no_repeated_feature_failures"
            break
        if pass_index >= int(max_refinement_rounds):
            break
        round_number = pass_index + 1
        round_dir = feedback_dir / f"round_{round_number:03d}"
        updated, changed, report = _request_ontology_refinements(
            definitions=current,
            repeated_patterns=repeated,
            output_dir=round_dir,
            request_json=request_json,
            workers=interpretation_workers,
        )
        rounds.append(
            {
                "round": round_number,
                "source_extraction_dir": str(extraction_dir),
                "repeated_failure_features": sorted(repeated),
                "changed_feature_names": list(report["changed_feature_names"]),
                "immutable_explicit_features": int(report["immutable_explicit_features"]),
            }
        )
        if not changed:
            stopped_reason = "no_ontology_changes"
            break
        source_extracted = extracted
        source_definitions = current
        source_summary = summary
        current = updated
        extraction_dir = round_dir / "extraction"

    if extracted is None or summary is None:  # pragma: no cover - loop always runs once
        raise RuntimeError("ontology feedback loop did not perform extraction")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_result = {
        "schema_version": ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION,
        "completed_at": _now(),
        "minimum_failure_patients": int(minimum_failure_patients),
        "maximum_refinement_rounds": int(max_refinement_rounds),
        "rounds_executed": len(rounds),
        "stopped_reason": stopped_reason,
        "rounds": rounds,
        "definitions": current,
    }
    _write_json(feedback_dir / "final_failure_summary.json", summary)
    _write_json(feedback_dir / "result.json", feedback_result)
    _write_json(feedback_dir / "complete.json", {"status": "complete", **feedback_result})
    return extracted, current, len(rounds)


SUPER_LEARNER_DESCRIPTION = (
    "Each nuisance model (e, mu0 on comparator rows, mu1 on treated rows) is a stacked "
    "super learner: the elastic net, histogram gradient boosting and a random forest "
    "(or the configured subset) are fitted in V-fold internal cross-validation inside "
    "the model's training rows (stratified for binary targets; fewer folds when a class "
    "or arm is small); non-negative weights summing to one minimize the internal-CV log "
    "loss (binary) or squared error (continuous, normalized NNLS); learners with "
    "positive weight are refitted on all training rows. The fit audits record the "
    "weights, per-learner and stack CV losses, and each learner's own audit. The "
    "held-out own-fold Platt calibration applies to the stacked predictions as to any "
    "learner."
)
NUISANCE_MODEL_STORE_SCHEMA_VERSION = "stage2_nuisance_model_store_v1"
NUISANCE_MODEL_MANIFEST = "manifest.json"
_STORABLE_NUISANCE_MODELS = (
    ElasticNetLogisticClassifier,
    ElasticNetRegressor,
    SuperLearnerClassifier,
    SuperLearnerRegressor,
    _ConstantClassifier,
    _ConstantRegressor,
)


def _nuisance_library_versions() -> dict[str, str]:
    import platform

    import joblib
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _nuisance_model_identity(
    *,
    dataset: pd.DataFrame,
    candidate_fit: pd.DataFrame,
    candidate_definitions: Sequence[Mapping[str, Any]],
    fit_ids: Sequence[int],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    binary: bool,
    seed: int,
    learner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the inner-fold nuisance models depend on, plus code and library versions."""

    fit_ids = [int(value) for value in fit_ids]
    return {
        "nuisance_learner": dict(learner) if learner is not None else {"name": "elastic_net"},
        "schema_version": NUISANCE_MODEL_STORE_SCHEMA_VERSION,
        "definitions": list(candidate_definitions),
        "training_measurements": _frame_fingerprint(candidate_fit),
        "training_labels": _frame_fingerprint(
            dataset.iloc[fit_ids][[treatment_column, outcome_column]].reset_index(drop=True)
        ),
        "fit_row_ids": fit_ids,
        "inner_splits": [dict(split) for split in inner_splits],
        "binary": bool(binary),
        "seed": int(seed) + FINAL_NUISANCE_SEED_OFFSET,
        "sources": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "plain_handoff_stage2_analysis.py",
                "../models/elastic_net_nuisance.py",
                "../models/super_learner_nuisance.py",
            )
        },
        "libraries": _nuisance_library_versions(),
    }


def _store_nuisance_model(store: Mapping[str, Any], model: Any) -> None:
    import joblib

    path = Path(store["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(model, temporary)
    os.replace(temporary, path)


def _load_stored_nuisance_model(store: Mapping[str, Any], columns: int) -> Any | None:
    """Return a stored fitted model, or None (refit) on any mismatch or error.

    The file is unpickled only after its SHA-256 matches the store manifest,
    whose fingerprint covers the training inputs, the source code and the
    library versions.
    """

    import io

    import joblib

    try:
        manifest_path = Path(store["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != NUISANCE_MODEL_STORE_SCHEMA_VERSION
            or manifest.get("fingerprint") != store["fingerprint"]
        ):
            return None
        path = Path(store["path"])
        expected = (manifest.get("models") or {}).get(
            path.relative_to(manifest_path.parent).as_posix()
        )
        payload = path.read_bytes()
        if expected is None or hashlib.sha256(payload).hexdigest() != expected:
            return None
        model = joblib.load(io.BytesIO(payload))
        if not isinstance(model, _STORABLE_NUISANCE_MODELS):
            return None
        if int(getattr(model, "n_features_in_", columns)) != int(columns):
            return None
        return model
    except Exception:  # fail closed to refitting
        LOGGER.warning("stored nuisance model unusable; refitting: %s", store.get("path"),
                       exc_info=True)
        return None


def _write_nuisance_model_manifest(root: Path, identity: Mapping[str, Any], fingerprint: str) -> None:
    models = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.joblib"))
    }
    _write_json(root / NUISANCE_MODEL_MANIFEST, {
        "schema_version": NUISANCE_MODEL_STORE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "identity": dict(identity),
        "models": models,
        "serialization": "joblib; loaded only when the file SHA-256 matches this manifest",
    })


def _fit_nuisance_prediction(
    x, y, valid, *, binary, seed, audit_path=None, split_at=None, model_store=None,
    learner=None,
):
    """Fit one nuisance model (or load it from a model store) and predict ``valid``.

    With ``split_at``, rows ``valid[:split_at]`` and ``valid[split_at:]`` are
    predicted in separate calls, so the first block's values do not depend on
    the rows appended after it. ``model_store`` (``path``, ``manifest``,
    ``fingerprint``, ``mode``) saves the fitted model (``"save"``) or reuses a
    verified stored one (``"load"``), falling back to fitting.
    """
    model, source = None, "fitted"
    if model_store is not None and model_store.get("mode") == "load":
        model = _load_stored_nuisance_model(model_store, np.asarray(x).shape[1])
        if model is not None:
            source = "reused_stored_model"
    if model is None:
        model = (
            _fit_classifier(x, y, seed=seed, learner=learner) if binary
            else _fit_regressor(x, y, seed=seed, learner=learner)
        )
        if model_store is not None and model_store.get("mode") == "save":
            _store_nuisance_model(model_store, model)
    if audit_path is not None:
        # The elastic-net wrappers report standardization, the selected penalty
        # and CV/refit iteration counts (including whether a limit was reached).
        fit_audit = getattr(model, "fit_audit", None)
        _write_json(Path(audit_path), {
            "schema_version": "stage2_nuisance_fit_audit_v1",
            "rows": int(len(y)),
            "columns": int(np.asarray(x).shape[1]),
            "fit_audit": (
                fit_audit() if callable(fit_audit)
                else {"estimator": type(model).__name__, "fit_mode": "constant_or_external"}
            ),
            **({"model_source": source} if model_store is not None else {}),
        })

    def predict(rows):
        return _predict_probability(model, rows) if binary else np.asarray(model.predict(rows), float)

    if split_at is None or not 0 < int(split_at) < len(valid):
        return predict(valid)
    return np.concatenate((predict(valid[: int(split_at)]), predict(valid[int(split_at):])))


def _nuisance_prediction_futures(
    x_t, x_y, treatment, outcome, valid_t, valid_y, *, binary, seed,
    checkpoint_dir=None, checkpoint_fingerprint="uncached", name="full_training",
    audit_dir=None, split_at=None, model_store=None, learner=None,
):
    from functools import partial
    from .stage2_final_modeling import prediction_future

    def audit_path(target):
        return None if audit_dir is None else Path(audit_dir) / name / f"{target}.json"

    def store(target):
        if model_store is None:
            return None
        root = Path(model_store["root"])
        return {
            "path": str(root / name / f"{target}.joblib"),
            "manifest": str(root / NUISANCE_MODEL_MANIFEST),
            "fingerprint": model_store["fingerprint"],
            "mode": model_store["mode"],
        }

    pending = [prediction_future(
        checkpoint_dir, f"{name}/propensity", checkpoint_fingerprint,
        partial(_fit_nuisance_prediction, x_t, treatment, valid_t, binary=True, seed=seed,
                audit_path=audit_path("propensity"), split_at=split_at,
                model_store=store("propensity"), learner=learner),
        len(valid_t),
    )]
    for arm in (0, 1):
        mask = treatment.astype(int) == arm
        if not mask.any():
            mask = np.ones(len(treatment), dtype=bool)
        pending.append(prediction_future(
            checkpoint_dir, f"{name}/mu{arm}", checkpoint_fingerprint,
            partial(_fit_nuisance_prediction, x_y[mask], outcome[mask], valid_y,
                    binary=binary, seed=seed + arm, audit_path=audit_path(f"mu{arm}"),
                    split_at=split_at, model_store=store(f"mu{arm}"), learner=learner),
            len(valid_y),
        ))
    return pending


def _nuisance_fit_audit_summary(audit_dir: Path) -> dict[str, Any]:
    """Summarize persisted nuisance fit audits (elastic-net iteration limits, stacks).

    For a super learner the elastic-net fields come from its refitted elastic
    net, and the internal-CV elastic-net fits count toward the CV limit flag.
    """

    records = []
    for path in sorted(Path(audit_dir).rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        audit = value.get("fit_audit") if isinstance(value, Mapping) else None
        if not isinstance(audit, Mapping):
            continue
        stacked = "stack_weights" in audit
        elastic = (audit.get("base_learners") or {}).get("elastic_net") or {} if stacked else audit
        optimization = elastic.get("optimization") or {}
        standardization = elastic.get("standardization") or {}
        internal_limit = stacked and any(
            fold.get("iteration_limit_reached")
            for fold in (audit.get("internal_cv") or {}).get("elastic_net", [])
        )
        records.append({
            "fit": str(path.relative_to(audit_dir).with_suffix("")),
            "fit_mode": audit.get("fit_mode"),
            "selected_regularization": elastic.get("selected_regularization"),
            "standardized": standardization.get("applied"),
            "refit_iterations": optimization.get("refit_iterations"),
            "refit_iteration_limit_reached": bool(optimization.get("refit_iteration_limit_reached")),
            "cv_iteration_limit_reached": bool(
                optimization.get("cv_iteration_limit_reached") or internal_limit
            ),
            "configured_max_iter": optimization.get("configured_max_iter"),
            **({"model_source": value["model_source"]} if "model_source" in value else {}),
            **(
                {
                    "stack_weights": audit.get("stack_weights"),
                    "learner_cv_loss": audit.get("learner_cv_loss"),
                    "stack_cv_loss": audit.get("stack_cv_loss"),
                    "effective_cv_folds": audit.get("effective_cv_folds"),
                }
                if stacked
                else {}
            ),
        })
    sources = Counter(r["model_source"] for r in records if "model_source" in r)
    return {
        "audit_dir": str(audit_dir),
        "fits_audited": len(records),
        **({"model_sources": dict(sorted(sources.items()))} if sources else {}),
        "refit_iteration_limit_reached": [r["fit"] for r in records if r["refit_iteration_limit_reached"]],
        "cv_iteration_limit_reached": [r["fit"] for r in records if r["cv_iteration_limit_reached"]],
        "standardized_fits": sum(r["standardized"] is True for r in records),
        "fits": records,
    }


def _nuisance_learner_summary(
    learner: Mapping[str, Any] | None, audit_summary: Mapping[str, Any]
) -> dict[str, Any]:
    """The learner and, for a super learner, mean stack weights and CV losses per nuisance."""

    learner = dict(learner) if learner is not None else {"name": "elastic_net"}
    summary: dict[str, Any] = {"learner": learner["name"], "configuration": learner}
    if learner["name"] != "super_learner":
        return summary
    by_target: dict[str, list[Mapping[str, Any]]] = {}
    for record in audit_summary.get("fits") or []:
        if "stack_weights" in record:
            by_target.setdefault(str(record["fit"]).rsplit("/", 1)[-1], []).append(record)

    def mean_by_learner(records, key):
        values: dict[str, list[float]] = {}
        for record in records:
            for name, value in (record.get(key) or {}).items():
                if value is not None:
                    values.setdefault(name, []).append(float(value))
        return {name: float(np.mean(items)) for name, items in values.items()}

    summary["nuisances"] = {
        target: {
            "fits": len(records),
            "fit_modes": dict(Counter(str(r.get("fit_mode")) for r in records)),
            "mean_stack_weights": mean_by_learner(records, "stack_weights"),
            "mean_learner_cv_loss": mean_by_learner(records, "learner_cv_loss"),
            "mean_stack_cv_loss": (
                float(np.mean([r["stack_cv_loss"] for r in records
                               if r.get("stack_cv_loss") is not None]))
                if any(r.get("stack_cv_loss") is not None for r in records) else None
            ),
        }
        for target, records in sorted(by_target.items())
    }
    summary["fits_without_stack"] = int(audit_summary.get("fits_audited", 0)) - sum(
        len(records) for records in by_target.values()
    )
    return summary


def _cross_fitted_nuisance(
    *,
    dataset: pd.DataFrame,
    extracted: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    fit_ids: Sequence[int],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    binary: bool,
    seed: int,
    checkpoint_dir: Path | None = None,
    checkpoint_fingerprint: str = "uncached",
    all_candidates: bool = False,
    audit_dir: Path | None = None,
    heldout_extracted: pd.DataFrame | None = None,
    model_store: Mapping[str, Any] | None = None,
    learner: Mapping[str, Any] | None = None,
    heldout_fold_predictions: bool = False,
) -> tuple[Any, ...]:
    """Inner cross-fitted e/mu0/mu1 for the outer-training rows.

    With ``heldout_extracted`` (outer-held-out measurements, no labels), every
    inner-fold model also predicts the held-out rows in the same prediction job,
    and six arrays are returned: training e, mu0, mu1 followed by the unweighted
    average of the K inner-fold models' held-out e, mu0, mu1 (probability scale
    for binary targets, uncalibrated). ``heldout_fold_predictions`` appends a
    seventh element: one record per fold model with its inner-validation
    positions and its own held-out predictions. Each fold model predicts its
    inner-validation rows and the held-out rows in separate calls, so the
    training values are identical with and without ``heldout_extracted``.
    """
    from .stage2_final_modeling import prediction_result

    fit_ids = [int(value) for value in fit_ids]
    position = {row_id: index for index, row_id in enumerate(fit_ids)}
    e = np.full(len(fit_ids), np.nan)
    mu0 = np.full(len(fit_ids), np.nan)
    mu1 = np.full(len(fit_ids), np.nan)
    extracted_by_id = extracted.set_index("_oci_row_id", drop=False)
    propensity_defs = list(definitions) if all_candidates else _definitions_for_nuisance_role(definitions, "treatment")
    outcome_defs = list(definitions) if all_candidates else _definitions_for_nuisance_role(definitions, "outcome")
    pending = []
    for fold_index, fold in enumerate(inner_splits, start=1):
        train_ids = [int(value) for value in fold["fit_row_ids"] if int(value) in position]
        valid_ids = [int(value) for value in fold["heldout_row_ids"] if int(value) in position]
        if not train_ids or not valid_ids:
            continue
        train_features = extracted_by_id.loc[train_ids].reset_index(drop=True)
        valid_features = extracted_by_id.loc[valid_ids].reset_index(drop=True)
        train_data = dataset.iloc[train_ids]
        t_train = train_data[treatment_column].to_numpy(dtype=float)
        y_train = train_data[outcome_column].to_numpy(dtype=float)
        t_encoder = _FeatureEncoder(propensity_defs).fit(train_features)
        y_encoder = _FeatureEncoder(outcome_defs).fit(train_features)
        x_t_train = t_encoder.transform(train_features)
        x_t_valid = t_encoder.transform(valid_features)
        x_y_train = y_encoder.transform(train_features)
        x_y_valid = y_encoder.transform(valid_features)
        if heldout_extracted is not None:
            # The same fold model scores its inner-validation rows and the
            # outer-held-out rows; the latter are appended after the former.
            x_t_valid = np.vstack((x_t_valid, t_encoder.transform(heldout_extracted)))
            x_y_valid = np.vstack((x_y_valid, y_encoder.transform(heldout_extracted)))
        valid_positions = [position[row_id] for row_id in valid_ids]
        fold_number = int(fold.get("inner_fold", fold_index))
        jobs = _nuisance_prediction_futures(
            x_t_train, x_y_train, t_train, y_train, x_t_valid, x_y_valid,
            binary=binary, seed=seed + fold_index,
            checkpoint_dir=checkpoint_dir, checkpoint_fingerprint=checkpoint_fingerprint,
            name=f"cross_fit/fold_{fold_index:03d}", audit_dir=audit_dir,
            split_at=len(valid_ids) if heldout_extracted is not None else None,
            model_store=model_store,
            learner=learner,
        )
        pending.append((fold_number, valid_positions, jobs))
    heldout_rows = 0 if heldout_extracted is None else len(heldout_extracted)
    heldout_sums = [np.zeros(heldout_rows) for _ in range(3)]
    fold_models = []
    for fold_number, valid_positions, jobs in pending:
        fold_heldout = []
        for index, (target, job) in enumerate(zip((e, mu0, mu1), jobs)):
            values = prediction_result(job, len(valid_positions) + heldout_rows)
            target[valid_positions] = values[: len(valid_positions)]
            heldout_sums[index] += values[len(valid_positions):]
            fold_heldout.append(values[len(valid_positions):])
        fold_models.append({
            "inner_fold": fold_number,
            "validation_positions": list(valid_positions),
            "heldout": tuple(fold_heldout),
        })
    if np.isnan(e).any() or np.isnan(mu0).any() or np.isnan(mu1).any():
        raise ValueError(
            "inner splits do not provide one nuisance prediction for every outer-fit row"
        )
    if heldout_extracted is None:
        return e, mu0, mu1
    if not pending:
        raise ValueError("inner splits fitted no nuisance models for the held-out ensemble")
    heldout_e, heldout_mu0, heldout_mu1 = (values / len(pending) for values in heldout_sums)
    if heldout_fold_predictions:
        return e, mu0, mu1, heldout_e, heldout_mu0, heldout_mu1, fold_models
    return e, mu0, mu1, heldout_e, heldout_mu0, heldout_mu1


def _aligned_measurement_frame(
    frame: pd.DataFrame,
    row_ids: Sequence[int],
    definitions: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> pd.DataFrame:
    """Return one row per requested patient, in order, with every candidate column."""

    names = [str(feature["name"]) for feature in definitions]
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} nuisance-candidate measurements lack columns: {missing}")
    ids = [int(value) for value in frame["_oci_row_id"]]
    if len(set(ids)) != len(ids) or set(ids) != set(map(int, row_ids)):
        raise ValueError(
            f"{label} nuisance-candidate measurements do not cover exactly the "
            "outer-fold rows"
        )
    companions = [
        extraction_failure_column(name)
        for name in names
        if extraction_failure_column(name) in frame.columns
    ]
    return (
        frame.set_index("_oci_row_id", drop=False)
        .loc[[int(value) for value in row_ids], ["_oci_row_id", *names, *companions]]
        .reset_index(drop=True)
    )


def _inner_fold_assignment(
    fit_ids: Sequence[int], inner_splits: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Map each outer-training row to the inner fold that holds it out."""

    position = {int(row_id): index for index, row_id in enumerate(fit_ids)}
    assignment = np.zeros(len(fit_ids), dtype=int)
    for fold_index, fold in enumerate(inner_splits, start=1):
        number = int(fold.get("inner_fold", fold_index))
        for row_id in fold["heldout_row_ids"]:
            if int(row_id) in position:
                assignment[position[int(row_id)]] = number
    if (assignment == 0).any():
        raise ValueError("inner splits do not hold out every outer-training row once")
    return assignment


NUISANCE_RECALIBRATION_DESCRIPTION = (
    "Training out-of-fold e_hat, mu0_hat and mu1_hat are the raw predictions of the "
    "inner-fold model that did not see each row; they are not recalibrated. Held-out "
    "predictions are calibrated per inner-fold model: model k's Platt calibrator "
    "(logistic regression of the label on [1, logit(prediction)]) is fit on model k's "
    "own inner-validation rows (its out-of-fold predictions and those rows' training "
    "labels; for mu0 and mu1, the validation rows in the matching arm), applied to model "
    "k's held-out predictions, and the K calibrated predictions are averaged. A "
    "calibrator with fewer than 20 rows, a single class, constant predictions, "
    "separation or a nonpositive slope is the identity map. No held-out label is used. "
    "Continuous-outcome mu_hat is not recalibrated; its held-out value is the average of "
    "the K fold models."
)
# Version of the training/held-out nuisance post-processing; part of the
# selection identity under orthogonal_tests and of the model-store identity.
NUISANCE_CONSTRUCTION_VERSION = "raw_training_out_of_fold_own_fold_heldout_platt_v1"


def _recalibrated_nuisances(
    *,
    treatment: np.ndarray,
    outcome: np.ndarray,
    binary: bool,
    fit: tuple[np.ndarray, np.ndarray, np.ndarray],
    heldout: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    heldout_folds: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Raw training OOF nuisances and own-fold Platt-calibrated held-out nuisances.

    ``heldout_folds`` (required with ``heldout``) lists each inner-fold model's
    inner-validation positions and held-out predictions. Calibrators use only
    training rows and labels; training values never depend on ``heldout``.
    """

    from ..utils.calibration import PlattRecalibration

    raw_fit = tuple(np.asarray(v, dtype=float) for v in fit)
    treatment, outcome = np.asarray(treatment, dtype=float), np.asarray(outcome, dtype=float)
    audit: dict[str, Any] = {
        "schema_version": "stage2_nuisance_recalibration_v2_own_fold_heldout",
        "method": "raw_training_out_of_fold_own_fold_heldout_platt",
        "description": NUISANCE_RECALIBRATION_DESCRIPTION,
        "training": "raw_inner_out_of_fold_predictions_not_recalibrated",
    }
    if heldout is None:
        return {"fit": raw_fit, "heldout": None, "audit": audit}
    if not heldout_folds:
        raise ValueError("held-out calibration requires each fold model's held-out predictions")
    raw_heldout = tuple(np.asarray(v, dtype=float) for v in heldout)
    final_heldout = list(raw_heldout)
    targets = [("propensity", 0, treatment, None)]
    if binary:
        targets += [("mu0", 1, outcome, 0), ("mu1", 2, outcome, 1)]
    else:
        audit.update(
            mu0={"method": "none", "reason": "continuous outcome"},
            mu1={"method": "none", "reason": "continuous outcome"},
        )
    for name, index, label, arm in targets:
        calibrated, per_fold = [], {}
        for fold in heldout_folds:
            rows = np.asarray(fold["validation_positions"], dtype=int)
            if arm is not None:
                rows = rows[treatment[rows] == arm]
            calibrator = PlattRecalibration.fit(raw_fit[index][rows], label[rows])
            calibrated.append(calibrator.transform(np.asarray(fold["heldout"][index], float)))
            per_fold[str(fold["inner_fold"])] = {
                **calibrator.public_dict(),
                "fit_rows": int(len(rows)),
                "fit_scope": (
                    "this fold model's inner-validation rows"
                    + ("" if arm is None else f" with treatment == {arm}")
                ),
            }
        final_heldout[index] = np.mean(np.vstack(calibrated), axis=0)
        audit[name] = {
            "method": "own_fold_platt_then_average",
            "fold_models": len(heldout_folds),
            "identity_calibrators": sum(
                not str(item.get("status")).startswith("fitted") for item in per_fold.values()
            ),
            "per_fold": per_fold,
        }
    return {
        "fit": raw_fit,
        "heldout": tuple(np.asarray(v, dtype=float) for v in final_heldout),
        "audit": audit,
    }


# The final nuisance construction's seed offset; modifier testing reuses it.
FINAL_NUISANCE_SEED_OFFSET = 30_000


def _outer_training_nuisances(
    *,
    dataset: pd.DataFrame,
    candidate_fit: pd.DataFrame,
    candidate_definitions: Sequence[Mapping[str, Any]],
    fit_ids: Sequence[int],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    binary: bool,
    seed: int,
    checkpoint_dir: Path | None,
    checkpoint_fingerprint: str,
    audit_dir: Path | None,
    candidate_heldout: pd.DataFrame | None = None,
    model_store_root: Path | None = None,
    model_store_mode: str | None = None,
    learner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """All-candidate, inner-cross-fitted e/mu0/mu1 (raw training OOF; own-fold
    Platt-calibrated held-out values; see ``_recalibrated_nuisances``).

    ``learner`` is the nuisance learner (``learner_spec``); None is the elastic
    net.

    The final estimation (with ``candidate_heldout``) and the modifier tests
    (training rows only, before held-out measurements exist) share this
    construction: same candidates, folds, seeds and standardization. Training
    values are the raw out-of-fold predictions and are identical in both calls.

    ``model_store_root`` with ``model_store_mode="save"`` (modifier tests)
    persists every inner-fold model under ``<root>/<fingerprint>``; with
    ``"load"`` (final estimation) those verified models score the held-out rows
    instead of being refitted. The fingerprint covers the training inputs, the
    source code and the library versions; any mismatch or load error refits.
    The held-out calibrators are recomputed from the stored models' identical
    training out-of-fold predictions and their own folds' training labels.
    """

    fit_ids = [int(value) for value in fit_ids]
    fit_data = dataset.iloc[fit_ids]
    treatment = fit_data[treatment_column].to_numpy(dtype=float)
    outcome = fit_data[outcome_column].to_numpy(dtype=float)
    inner_fold_ids = _inner_fold_assignment(fit_ids, inner_splits)
    store, store_report = None, {"status": "not_configured"}
    if model_store_root is not None and model_store_mode is not None:
        if model_store_mode not in {"save", "load"}:
            raise ValueError("model_store_mode must be 'save' or 'load'")
        identity = _nuisance_model_identity(
            dataset=dataset,
            candidate_fit=candidate_fit,
            candidate_definitions=candidate_definitions,
            fit_ids=fit_ids,
            inner_splits=inner_splits,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            binary=binary,
            seed=seed,
            learner=learner,
        )
        store_fingerprint = _value_fingerprint(identity)
        store_root = Path(model_store_root) / store_fingerprint[:20]
        store_report = {
            "status": "saved" if model_store_mode == "save" else "absent",
            "mode": model_store_mode,
            "root": str(store_root),
            "fingerprint": store_fingerprint,
        }
        if model_store_mode == "save" or (store_root / NUISANCE_MODEL_MANIFEST).is_file():
            store = {"root": str(store_root), "mode": model_store_mode,
                     "fingerprint": store_fingerprint}
            if model_store_mode == "load":
                store_report["status"] = "available"
    raw = _cross_fitted_nuisance(
        dataset=dataset,
        extracted=candidate_fit,
        definitions=candidate_definitions,
        fit_ids=fit_ids,
        inner_splits=inner_splits,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        binary=binary,
        seed=seed + FINAL_NUISANCE_SEED_OFFSET,
        checkpoint_dir=checkpoint_dir,
        checkpoint_fingerprint=checkpoint_fingerprint,
        # The pooled support was learned across all outer-training patients.
        # It must not screen predictors for their own OOF nuisance predictions.
        all_candidates=True,
        audit_dir=audit_dir,
        # Held-out rows: the average of the same K inner-fold models (the
        # cross-fitting ensemble) that produced the training OOF predictions.
        # No separate full-data fit.
        heldout_extracted=candidate_heldout,
        model_store=store,
        learner=learner,
        heldout_fold_predictions=candidate_heldout is not None,
    )
    if store is not None and model_store_mode == "save":
        _write_nuisance_model_manifest(Path(store["root"]), identity, store["fingerprint"])
    raw_fit = tuple(raw[:3])
    raw_heldout = tuple(raw[3:6]) if candidate_heldout is not None else None
    recalibrated = _recalibrated_nuisances(
        treatment=treatment,
        outcome=outcome,
        binary=binary,
        fit=raw_fit,
        heldout=raw_heldout,
        heldout_folds=raw[6] if candidate_heldout is not None else None,
    )
    return {
        "treatment": treatment,
        "outcome": outcome,
        "inner_fold_ids": inner_fold_ids,
        "raw_fit": raw_fit,
        "raw_heldout": raw_heldout,
        "recalibrated": recalibrated,
        "model_store": store_report,
    }


def estimate_outer_fold(
    *,
    dataset: pd.DataFrame,
    extracted_fit: pd.DataFrame,
    extracted_heldout: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    unit_id_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    inner_folds: int,
    seed: int,
    propensity_clip: float,
    estimation_trees: int,
    output_dir: Path,
    min_propensity: float | None = None,
    max_propensity: float | None = None,
    estimator: str = "causal_forest",
    statistical_policy: Any = None,
    nuisance_candidate_definitions: Sequence[Mapping[str, Any]],
    nuisance_candidate_fit: pd.DataFrame,
    nuisance_candidate_heldout: pd.DataFrame,
    nuisance_model_store: Path | None = None,
    nuisance_learner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate one outer fold with one honest nuisance construction everywhere.

    ``nuisance_learner`` (``learner_spec``; None is the elastic net) fits e,
    mu0 and mu1.

    ``nuisance_model_store`` is the root where the modifier tests stored their
    inner-fold nuisance models; verified models are reused to score the
    held-out rows, otherwise they are refitted (identical values either way).

    Training rows get the raw inner-cross-fitted, all-candidate nuisances of
    the configured learner; held-out rows get the average of those same K
    inner-fold models' predictions, each Platt-calibrated on that model's own
    inner-validation rows (training labels only). These same e/mu0/mu1 feed eligibility, the
    held-out AIPW scores, the DINA offsets, the interaction model's unpenalized
    adjustment covariates, the forest's prognostic column and the diagnostics.
    Held-out treatment/outcome labels are read only after effect predictions
    are frozen.
    """
    from .stage2_effect_estimators import (
        NUISANCE_ADJUSTMENT_DESCRIPTION,
        PROGNOSTIC_FOREST_DESCRIPTION,
        SELECTED_MODIFIER_PENALTY_DESCRIPTION,
        effect_model_family,
        ensemble_estimators,
        fit_interaction_effect,
        nuisance_adjustment,
        prognostic_score,
        selected_modifier_penalty,
        selected_modifier_penalty_summary,
    )

    model_family = effect_model_family(estimator, outcome_type)
    if estimator in {"linear_interactions", "dina", "ensemble"}:
        if statistical_policy is None:
            raise ValueError(
                "linear interaction, DINA or ensemble estimation requires a statistical policy"
            )
        statistical_policy.validate()
    # Under orthogonal_tests the interaction model does not shrink the
    # test-selected modifiers again (guarded; see fit_interaction_effect).
    orthogonal_interactions = (
        estimator in {"linear_interactions", "ensemble"}
        and statistical_policy.multi_model.modifier_selection.rule == "orthogonal_tests"
    )
    modifier_penalty = (
        selected_modifier_penalty(statistical_policy, outcome_type)
        if orthogonal_interactions
        else None
    )
    validate_propensity_bounds(min_propensity, max_propensity)
    complete_path = output_dir / "complete.json"
    diagnostics_path = output_dir / "diagnostics.json"
    estimation_input_fingerprint = _value_fingerprint(
        {
            "schema_version": ESTIMATION_CHECKPOINT_SCHEMA_VERSION,
            "definitions": list(definitions),
            "split": dict(split),
            "unit_id_column": unit_id_column,
            "treatment_column": treatment_column,
            "outcome_column": outcome_column,
            "outcome_type": outcome_type,
            "inner_folds": inner_folds,
            "seed": seed,
            "min_propensity": min_propensity,
            "max_propensity": max_propensity,
            "propensity_clip": propensity_clip,
            "estimation_trees": estimation_trees,
            "estimator": estimator,
            "nuisance_learner": (
                dict(nuisance_learner) if nuisance_learner is not None else {"name": "elastic_net"}
            ),
            "final_fit_sources": {
                name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in (
                    "plain_handoff_stage2_analysis.py", "stage2_final_modeling.py",
                    "stage2_effect_estimators.py", "../utils/calibration.py",
                    "../models/elastic_net_nuisance.py", "../models/causal_forest_head.py",
                    "../models/super_learner_nuisance.py",
                )
            },
            "interaction_policy": (
                statistical_policy.public_dict()
                if estimator in {"linear_interactions", "dina", "ensemble"}
                else None
            ),
            "selected_modifier_penalty": (
                modifier_penalty.public_dict() if modifier_penalty is not None else None
            ),
            "interaction_source_fingerprint": (
                _value_fingerprint(
                    [
                        (Path(__file__).parent / name).read_text()
                        for name in (
                            "stage2_effect_estimators.py",
                            "stage2_dina.py",
                            "../models/dina.py",
                            "stage2_multi_model_selection.py",
                            "stage2_elastic_net_selection.py",
                        )
                    ]
                )
                if estimator in {"linear_interactions", "dina", "ensemble"}
                else None
            ),
            "dataset_modeling_fingerprint": _frame_fingerprint(
                dataset[[unit_id_column, treatment_column, outcome_column]]
            ),
            "extracted_fit_fingerprint": _frame_fingerprint(extracted_fit),
            "extracted_heldout_fingerprint": _frame_fingerprint(extracted_heldout),
            "nuisance_candidates": {
                "definitions": list(nuisance_candidate_definitions),
                "training_measurements": _frame_fingerprint(nuisance_candidate_fit),
                "heldout_measurements": _frame_fingerprint(nuisance_candidate_heldout),
            },
        }
    )
    if complete_path.is_file() and diagnostics_path.is_file():
        try:
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                completion.get("schema_version") == ESTIMATION_CHECKPOINT_SCHEMA_VERSION
                and completion.get("input_fingerprint") == estimation_input_fingerprint
            ):
                return json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except (
            AttributeError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            pass
        LOGGER.info("rerun incompatible Stage 2 outer-fold estimation: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from . import stage2_final_modeling as final_modeling

    # Worker counts are operational: changing them can reuse these predictions.
    # New source/input identities use separate directories, preserving prior fits.
    final_fit_fingerprint = estimation_input_fingerprint
    final_fit_dir = output_dir / "fit_checkpoints" / final_fit_fingerprint
    LOGGER.info("Stage 2 final estimation started output=%s shared_modeling_pool=%s",
                output_dir, modeling.enabled())
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    heldout_ids = [int(value) for value in split["heldout_row_ids"]]
    binary = str(outcome_type) == "binary"
    fit_data = dataset.iloc[fit_ids]
    heldout_data = dataset.iloc[heldout_ids]
    t_fit = fit_data[treatment_column].to_numpy(dtype=float)
    y_fit = fit_data[outcome_column].to_numpy(dtype=float)

    adjustment_defs = _definitions_for_roles(definitions, {"confounder"})
    # Statistically selected task supports are reported for transparency only.
    # Every nuisance model below uses the complete candidate catalog (M-02).
    propensity_defs = _definitions_for_nuisance_role(definitions, "treatment")
    outcome_defs = _definitions_for_nuisance_role(definitions, "outcome")
    effect_defs = _definitions_for_roles(definitions, {"effect_modifier"})
    effect_ids = {str(feature["feature_id"]) for feature in effect_defs}
    pure_confounder_defs = [
        feature for feature in adjustment_defs if str(feature["feature_id"]) not in effect_ids
    ]
    for frame, ids, label in (
        (extracted_fit, fit_ids, "training"),
        (extracted_heldout, heldout_ids, "held-out"),
    ):
        if [int(value) for value in frame["_oci_row_id"]] != ids:
            raise ValueError(f"{label} measurements must follow the outer-fold row order")
    candidate_fit = _aligned_measurement_frame(
        nuisance_candidate_fit, fit_ids, nuisance_candidate_definitions, label="training"
    )
    candidate_heldout = _aligned_measurement_frame(
        nuisance_candidate_heldout, heldout_ids, nuisance_candidate_definitions,
        label="held-out",
    )
    _write_frame(output_dir / "nuisance_candidate_fit.csv", candidate_fit)
    _write_frame(output_dir / "nuisance_candidate_heldout.csv", candidate_heldout)
    _write_json(
        output_dir / "nuisance_candidate_definitions.json",
        {"features": [dict(feature) for feature in nuisance_candidate_definitions]},
    )
    # Eligibility for effect fitting must not use in-sample propensity predictions.
    supplied_inner = list(split.get("inner_splits") or []) or _fallback_inner_splits(
        fit_ids, folds=inner_folds, seed=seed
    )
    # Fit audits live beside (not inside) the prediction checkpoints.
    nuisance_audit_dir = output_dir / "nuisance_fit_audits" / final_fit_fingerprint[:20]
    nuisances = _outer_training_nuisances(
        dataset=dataset,
        candidate_fit=candidate_fit,
        candidate_definitions=nuisance_candidate_definitions,
        fit_ids=fit_ids,
        inner_splits=supplied_inner,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        binary=binary,
        seed=seed,
        checkpoint_dir=final_fit_dir,
        checkpoint_fingerprint=final_fit_fingerprint,
        audit_dir=nuisance_audit_dir,
        candidate_heldout=candidate_heldout,
        model_store_root=nuisance_model_store,
        model_store_mode="load" if nuisance_model_store is not None else None,
        learner=nuisance_learner,
    )
    inner_fold_ids = nuisances["inner_fold_ids"]
    raw_fit_propensity, raw_fit_mu0, raw_fit_mu1 = nuisances["raw_fit"]
    raw_propensity, raw_mu0, raw_mu1 = nuisances["raw_heldout"]
    LOGGER.info("Stage 2 final nuisances completed output=%s", output_dir)
    recalibrated = nuisances["recalibrated"]
    fit_propensity, fit_mu0, fit_mu1 = recalibrated["fit"]
    propensity, mu0, mu1 = recalibrated["heldout"]
    fit_keep = propensity_eligibility(fit_propensity, min_propensity, max_propensity)
    _write_frame(
        output_dir / "nuisance_fit_predictions.csv",
        pd.DataFrame(
            {
                "_oci_row_id": fit_ids,
                "inner_fold": inner_fold_ids,
                "treatment": t_fit,
                "outcome": y_fit,
                "propensity": fit_propensity,
                "mu0": fit_mu0,
                "mu1": fit_mu1,
                "propensity_uncalibrated": raw_fit_propensity,
                "mu0_uncalibrated": raw_fit_mu0,
                "mu1_uncalibrated": raw_fit_mu1,
                "effect_eligible": fit_keep,
            }
        ),
    )
    fit_overlap = overlap_diagnostics(fit_propensity, min_propensity, max_propensity)
    _write_json(output_dir / "propensity_overlap.json", {"fit": fit_overlap})
    if fit_keep.sum() < 4 or min(np.sum(t_fit[fit_keep] == arm) for arm in (0, 1)) < 2:
        raise ValueError(
            "propensity bounds leave insufficient training rows or treatment arms for effect estimation"
        )
    eligible_fit = extracted_fit.iloc[np.flatnonzero(fit_keep)]
    heldout_keep = propensity_eligibility(propensity, min_propensity, max_propensity)
    overlap = {
        "fit": fit_overlap,
        "heldout": overlap_diagnostics(propensity, min_propensity, max_propensity),
        "fit_eligibility_source": "inner_out_of_fold_all_candidate_raw_propensity",
        "heldout_eligibility_source": (
            "inner_fold_ensemble_all_candidate_own_fold_calibrated_propensity"
        ),
    }
    _write_json(output_dir / "propensity_overlap.json", overlap)
    if not heldout_keep.any():
        raise ValueError("propensity bounds leave no held-out patients for effect estimation")

    def fit_estimator(name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Fit one final effect estimator on the eligible training rows.

        Returns held-out effect predictions (eligible rows), the fit audit and
        architecture diagnostics. No held-out labels enter any estimator.
        """
        if name == "causal_forest":
            effect_encoder = _FeatureEncoder(effect_defs).fit(eligible_fit)
            control_encoder = _FeatureEncoder(pure_confounder_defs).fit(eligible_fit)
            x_effect_fit = effect_encoder.transform(extracted_fit)
            x_effect_heldout = effect_encoder.transform(extracted_heldout)
            selected_modifier_columns = int(x_effect_fit.shape[1])
            w_fit = control_encoder.transform(extracted_fit)
            # M-04: on the risk-difference scale the CATE moves with baseline risk
            # even under a constant odds ratio, so X carries the honest prognostic
            # score (OOF for training rows, inner-fold ensemble for held-out rows).
            prognostic_in_x = binary
            if prognostic_in_x:
                x_effect_fit = np.column_stack(
                    (x_effect_fit, prognostic_score(fit_mu0, binary=True))
                )
                x_effect_heldout = np.column_stack(
                    (x_effect_heldout, prognostic_score(mu0, binary=True))
                )
            if x_effect_fit.shape[1] == 0:
                # EconML requires X for heterogeneous-effect prediction. A constant X
                # yields one fold-level treatment effect when no modifier survived
                # (continuous outcomes only; binary outcomes always carry the
                # prognostic score).
                x_effect_fit = np.ones((len(extracted_fit), 1), dtype=float)
                x_effect_heldout = np.ones((len(extracted_heldout), 1), dtype=float)
                constant_effect_design = True
            else:
                constant_effect_design = False
            controls_fit = w_fit[fit_keep] if w_fit.shape[1] else None
            with modeling.threaded_fit() as forest_jobs:
                causal_forest = CausalForestHead(
                    n_estimators=int(estimation_trees),
                    max_depth=None,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    honest=True,
                    inference=True,
                    random_state=seed + 20_000,
                    tune_model=False,
                    subforest_size=next(size for size in (4, 3, 2, 1) if int(estimation_trees) % size == 0),
                    n_jobs=forest_jobs,
                    outcome_type=outcome_type,
                )
                causal_forest.fit(
                    x_effect_fit[fit_keep],
                    t_fit[fit_keep],
                    y_fit[fit_keep],
                    W=controls_fit,
                )
                effect_predictions = causal_forest.predict(
                    x_effect_heldout[heldout_keep],
                    return_ci=True,
                )

                fit_audit = causal_forest.fit_audit()
            architecture_diagnostics = {
                "causal_forest_trees": int(estimation_trees),
                "causal_forest_honest": True,
                "causal_forest_inference": True,
                "causal_forest_tuned": False,
                "causal_forest_fit_audit": fit_audit,
                "pure_confounders_in_w": len(pure_confounder_defs),
                "dual_role_features_in_x_only": len(
                    [f for f in adjustment_defs if str(f["feature_id"]) in effect_ids]
                ),
                "selected_modifier_columns_in_x": selected_modifier_columns,
                "prognostic_score_in_x": prognostic_in_x,
                "prognostic_score": (
                    {
                        "column": "logit_prognostic_mu0",
                        "training_source": "inner_out_of_fold_all_candidate_raw_mu0",
                        "heldout_source": (
                            "inner_fold_ensemble_all_candidate_own_fold_calibrated_mu0"
                        ),
                        "description": PROGNOSTIC_FOREST_DESCRIPTION,
                    }
                    if prognostic_in_x
                    else {"column": None, "reason": "continuous outcome: no scale-induced heterogeneity"}
                ),
                "constant_effect_design": constant_effect_design,
            }
        elif name == "dina":
            from functools import partial
            from .stage2_dina import fit_effect as fit_dina_effect
            from .stage2_dina import unpack_nuisances

            result = modeling.call(partial(
                fit_dina_effect,
                train=eligible_fit,
                valid=extracted_heldout.iloc[np.flatnonzero(heldout_keep)],
                definitions=definitions,
                modifier_ids=effect_ids,
                treatment=t_fit[fit_keep],
                outcome=y_fit[fit_keep],
                policy=statistical_policy,
                seed=seed + 20_000,
                nuisance=unpack_nuisances({
                    "training_propensity": fit_propensity[fit_keep],
                    "training_mu0": fit_mu0[fit_keep],
                    "training_mu1": fit_mu1[fit_keep],
                    "validation_propensity": propensity[heldout_keep],
                    "validation_mu0": mu0[heldout_keep],
                    "validation_mu1": mu1[heldout_keep],
                }),
            ))
            effect_predictions = {
                "tau_pred": result["tau"],
                "log_odds_ratio": result["delta"],
                "dina_mu0": result["mu0"],
                "dina_mu1": result["mu1"],
            }
            fit_audit = result["audit"]
            architecture_diagnostics = {
                "dina_fit_audit": fit_audit,
                "dina_offsets": (
                    "training rows: raw inner out-of-fold all-candidate e/mu0/mu1; "
                    "held-out rows: inner-fold ensemble of own-fold calibrated e/mu0/mu1 "
                    "(one construction for fitting and prediction)"
                ),
                "cate_confidence_intervals": "not_computed_for_grouped_dina",
            }
        else:
            # Shared with the inner validation search; no held-out labels enter this API.
            result = fit_interaction_effect(
                train=eligible_fit,
                valid=extracted_heldout.iloc[np.flatnonzero(heldout_keep)],
                definitions=definitions,
                modifier_ids=effect_ids,
                treatment=t_fit[fit_keep],
                outcome=y_fit[fit_keep],
                binary=binary,
                policy=statistical_policy,
                seed=seed + 20_000,
                path_submit=final_modeling.penalty_path_submitter(
                    final_fit_dir, final_fit_fingerprint
                ),
                adjustment=nuisance_adjustment(
                    train_propensity=fit_propensity[fit_keep],
                    valid_propensity=propensity[heldout_keep],
                    train_mu0=fit_mu0[fit_keep],
                    valid_mu0=mu0[heldout_keep],
                    binary=binary,
                ),
                unpenalized_selected=modifier_penalty,
            )
            effect_predictions = {"tau_pred": result["tau"]}
            fit_audit = result["audit"]
            architecture_diagnostics = {
                "interaction_fit_audit": fit_audit,
                "cate_confidence_intervals": "not_computed_for_penalized_interaction_model",
                "no_selected_interactions": not effect_ids,
                **(
                    {"selected_modifier_penalty": selected_modifier_penalty_summary(fit_audit)}
                    if orthogonal_interactions
                    else {}
                ),
            }
        return effect_predictions, fit_audit, architecture_diagnostics

    ensemble = None
    if estimator == "ensemble":
        from .stage2_multi_model_selection import NotEstimable

        fitted: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
        unavailable_components: dict[str, Any] = {}
        for name in ensemble_estimators(outcome_type):
            try:
                fitted[name] = fit_estimator(name)
            except NotEstimable as exc:
                unavailable_components[name] = {
                    "status": "not_estimable",
                    "reason": str(exc),
                    **({"optimizer_diagnostics": exc.diagnostics}
                       if hasattr(exc, "diagnostics") else {}),
                }
                LOGGER.warning(
                    "Stage 2 ensemble component %s not estimable: %s", name, exc
                )
        if not fitted:
            raise ValueError("no ensemble component estimator was estimable")
        component_tau = {
            name: np.asarray(value[0]["tau_pred"], dtype=float).reshape(-1)
            for name, value in fitted.items()
        }
        effect_predictions = {
            # The unweighted mean of the fitted components' CATEs.
            "tau_pred": np.mean(np.vstack(list(component_tau.values())), axis=0),
            **{f"component_tau_{name}": values for name, values in component_tau.items()},
        }
        if "causal_forest" in fitted:
            for key in ("tau_lower", "tau_upper", "tau_std"):
                if fitted["causal_forest"][0].get(key) is not None:
                    effect_predictions[f"causal_forest_{key}"] = fitted["causal_forest"][0][key]
        if "dina" in fitted:
            for key in ("log_odds_ratio", "dina_mu0", "dina_mu1"):
                effect_predictions[key] = fitted["dina"][0][key]
        ensemble = {
            "estimator_policy": "ensemble",
            "combination": "unweighted_mean_of_fitted_component_cates",
            "component_estimators": list(ensemble_estimators(outcome_type)),
            "fitted_estimators": list(fitted),
            "unavailable_estimators": unavailable_components,
            "component_model_families": {
                name: effect_model_family(name, outcome_type) for name in fitted
            },
            "component_audits": {name: value[2] for name, value in fitted.items()},
            "interval_columns": (
                "estimated_cate_lower_95, estimated_cate_upper_95 and "
                "estimated_cate_standard_error are empty: the causal forest's standard "
                "error does not describe the ensemble average. The forest's own interval "
                "is reported in estimated_cate_causal_forest_lower_95/_upper_95/"
                "_standard_error."
            ),
        }
        # Each component's full fit audit is in ensemble["component_audits"].
        fit_audit = {
            "estimator_policy": "ensemble",
            "fitted_estimators": list(fitted),
            "outcome_model_contract": {
                "effect_scale": (
                    "probability_difference" if binary else "outcome_difference"
                ),
                "combination": "unweighted_mean_of_fitted_component_cates",
                "components": {
                    name: value[1].get("outcome_model_contract")
                    for name, value in fitted.items()
                },
            },
        }
        architecture_diagnostics = {
            "estimator_policy": "ensemble",
            "ensemble": ensemble,
            "cate_confidence_intervals": "not_computed_for_estimator_ensemble",
        }
    else:
        effect_predictions, fit_audit, architecture_diagnostics = fit_estimator(estimator)

    def scatter_prediction(key: str) -> np.ndarray:
        values = np.full(len(heldout_ids), np.nan)
        prediction = effect_predictions.get(key)
        if prediction is not None:
            values[heldout_keep] = np.asarray(prediction, dtype=float).reshape(-1)
        return values

    # Effect predictions are frozen here; held-out labels are read only below.
    cate = scatter_prediction("tau_pred")
    cate_lower_values = scatter_prediction("tau_lower")
    cate_upper_values = scatter_prediction("tau_upper")
    cate_std_values = scatter_prediction("tau_std")
    t_heldout = heldout_data[treatment_column].to_numpy(dtype=float)
    y_heldout = heldout_data[outcome_column].to_numpy(dtype=float)
    aipw = np.full(len(heldout_ids), np.nan)
    aipw[heldout_keep] = _dr_score(
        y_heldout[heldout_keep],
        t_heldout[heldout_keep],
        mu0[heldout_keep],
        mu1[heldout_keep],
        propensity[heldout_keep],
        clip=propensity_clip,
    )
    predictions = pd.DataFrame(
        {
            "_oci_row_id": heldout_ids,
            unit_id_column: heldout_data[unit_id_column].tolist(),
            "treatment": t_heldout,
            "outcome": y_heldout,
            "propensity": propensity,
            "mu0": mu0,
            "mu1": mu1,
            "effect_eligible": heldout_keep,
            "aipw_score": aipw,
            **(
                {
                    "dina_log_odds_ratio": scatter_prediction("log_odds_ratio"),
                    "dina_mu0": scatter_prediction("dina_mu0"),
                    "dina_mu1": scatter_prediction("dina_mu1"),
                }
                if estimator == "dina"
                or (ensemble is not None and "dina" in ensemble["fitted_estimators"])
                else {}
            ),
            "estimated_cate": cate,
            "estimated_cate_lower_95": cate_lower_values,
            "estimated_cate_upper_95": cate_upper_values,
            "estimated_cate_standard_error": cate_std_values,
            **(
                {
                    # Columns for causal_forest and linear_interactions are always
                    # present (empty if not estimable); DINA only when fitted.
                    **{
                        f"estimated_cate_{name}": scatter_prediction(f"component_tau_{name}")
                        for name in ensemble["component_estimators"]
                        if name != "dina" or name in ensemble["fitted_estimators"]
                    },
                    "estimated_cate_causal_forest_lower_95": scatter_prediction(
                        "causal_forest_tau_lower"
                    ),
                    "estimated_cate_causal_forest_upper_95": scatter_prediction(
                        "causal_forest_tau_upper"
                    ),
                    "estimated_cate_causal_forest_standard_error": scatter_prediction(
                        "causal_forest_tau_std"
                    ),
                }
                if ensemble is not None
                else {}
            ),
            "propensity_uncalibrated": raw_propensity,
            "mu0_uncalibrated": raw_mu0,
            "mu1_uncalibrated": raw_mu1,
        }
    )
    _write_frame(output_dir / "predictions.csv", predictions)
    finite = aipw[np.isfinite(aipw)]
    if not len(finite):
        raise ValueError("outer-fold estimation produced no finite AIPW scores")
    ate = float(np.mean(finite))
    standard_error = (
        float(np.std(finite, ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else None
    )

    def calibration_set(e_fit, m0_fit, m1_fit, e_held, m0_held, m1_held):
        return {
            "fit_out_of_fold_treatment": calibration_diagnostics(t_fit, e_fit, binary=True),
            "fit_out_of_fold_outcome_factual": calibration_diagnostics(
                y_fit, np.where(t_fit == 1, m1_fit, m0_fit), binary=binary
            ),
            "fit_out_of_fold_outcome_by_arm": {
                str(arm): calibration_diagnostics(
                    y_fit[t_fit == arm], (m1_fit if arm else m0_fit)[t_fit == arm],
                    binary=binary,
                )
                for arm in (0, 1)
            },
            "heldout_treatment": calibration_diagnostics(t_heldout, e_held, binary=True),
            "heldout_outcome_factual": calibration_diagnostics(
                y_heldout, np.where(t_heldout == 1, m1_held, m0_held), binary=binary
            ),
            "heldout_outcome_by_arm": {
                str(arm): calibration_diagnostics(
                    y_heldout[t_heldout == arm],
                    (m1_held if arm else m0_held)[t_heldout == arm],
                    binary=binary,
                )
                for arm in (0, 1)
            },
        }

    nuisance_calibration = calibration_set(
        fit_propensity, fit_mu0, fit_mu1, propensity, mu0, mu1
    )
    nuisance_calibration.update(
        {
            "eligible_heldout_treatment": calibration_diagnostics(
                t_heldout[heldout_keep], propensity[heldout_keep], binary=True
            ),
            "eligible_heldout_outcome_factual": calibration_diagnostics(
                y_heldout[heldout_keep],
                np.where(t_heldout == 1, mu1, mu0)[heldout_keep],
                binary=binary,
            ),
            "predictions_used_downstream": (
                "raw_training_out_of_fold_and_own_fold_calibrated_heldout"
            ),
            "before_recalibration": calibration_set(
                raw_fit_propensity, raw_fit_mu0, raw_fit_mu1, raw_propensity, raw_mu0, raw_mu1
            ),
            "recalibration": recalibrated["audit"],
        }
    )

    def slope_pair(name):
        before = nuisance_calibration["before_recalibration"][name]
        after = nuisance_calibration[name]
        return {
            "before": {k: before.get(k) for k in ("calibration_intercept", "calibration_slope")},
            "after": {k: after.get(k) for k in ("calibration_intercept", "calibration_slope")},
        }

    nuisance_calibration["calibration_slope_summary"] = {
        name: slope_pair(name)
        for name in (
            "fit_out_of_fold_treatment",
            "fit_out_of_fold_outcome_factual",
            "heldout_treatment",
            "heldout_outcome_factual",
        )
    }
    super_learner_nuisances = _is_super_learner(nuisance_learner)
    nuisance_audit_summary = _nuisance_fit_audit_summary(nuisance_audit_dir)
    methodology_notes = {
        "unified_nuisances": (
            "Training out-of-fold and outer-held-out nuisances use one construction: "
            + (
                "super-learner stacks (" + ", ".join(nuisance_learner["learners"])
                + "; non-negative weights summing to one, learned by internal "
                "cross-validation inside each fit's training rows) "
                if super_learner_nuisances
                else "elastic nets "
            )
            + "on the complete nuisance-candidate catalog with fold-local "
            "regularization, fitted in the K inner cross-fitting folds. Training rows get "
            "the raw prediction of the fold model that did not see them; held-out rows "
            "(whose patients were measured on the full catalog) get the average of the "
            "same K fold models' predictions, each first Platt-calibrated on that model's "
            "own inner-validation rows (probability scale for binary targets). The same "
            "e/mu0/mu1 feed eligibility, AIPW, DINA offsets, interaction-model adjustment "
            "covariates, the forest prognostic score and these diagnostics."
        ),
        "nuisance_recalibration": recalibrated["audit"]["description"],
        "elastic_net_standardization": (
            "Elastic-net nuisance wrappers standardize their design with a StandardScaler "
            "fit on each fit's own rows; refit iteration counts are in the fit audits."
        ),
        **(
            {"super_learner_nuisances": SUPER_LEARNER_DESCRIPTION}
            if super_learner_nuisances
            else {}
        ),
        **(
            {"interaction_model_nuisance_adjustment": NUISANCE_ADJUSTMENT_DESCRIPTION}
            if estimator == "linear_interactions"
            or (ensemble is not None and "linear_interactions" in ensemble["fitted_estimators"])
            else {}
        ),
        **(
            {"interaction_model_selected_modifier_penalty": SELECTED_MODIFIER_PENALTY_DESCRIPTION}
            if modifier_penalty is not None and (
                estimator == "linear_interactions"
                or (
                    ensemble is not None
                    and "linear_interactions" in ensemble["fitted_estimators"]
                )
            )
            else {}
        ),
        **(
            {"forest_prognostic_score": PROGNOSTIC_FOREST_DESCRIPTION}
            if binary and (
                estimator == "causal_forest"
                or (ensemble is not None and "causal_forest" in ensemble["fitted_estimators"])
            )
            else {}
        ),
        **(
            {"estimator_ensemble": (
                "estimated_cate is the unweighted mean of the CATEs of every final "
                "estimator fitted for this outcome type (causal forest, linear "
                "interactions and, for binary outcomes, DINA), each fitted exactly as when "
                "chosen alone. Per-estimator CATEs are in estimated_cate_<estimator>."
            )}
            if ensemble is not None
            else {}
        ),
    }
    diagnostics = {
        "model_family": model_family,
        "estimator": estimator,
        "effect_model_fit_audit": fit_audit,
        **architecture_diagnostics,
        "methodology_notes": methodology_notes,
        "primary_ate_estimator": "outer_cross_fitted_aipw",
        **(
            {
                "nuisance_model_family": "super_learner",
                "binary_nuisance_model": (
                    "oci.models.super_learner_nuisance.SuperLearnerClassifier"
                ),
                "continuous_nuisance_model": (
                    "oci.models.super_learner_nuisance.SuperLearnerRegressor"
                ),
            }
            if super_learner_nuisances
            else {
                "nuisance_model_family": "elastic_net",
                "binary_nuisance_model": (
                    "oci.models.elastic_net_nuisance.ElasticNetLogisticClassifier"
                ),
                "continuous_nuisance_model": ("oci.models.elastic_net_nuisance.ElasticNetRegressor"),
            }
        ),
        "rows": len(heldout_ids),
        "fit_rows": len(fit_ids),
        "effect_fit_rows": int(fit_keep.sum()),
        "effect_estimation_rows": int(heldout_keep.sum()),
        "estimand": (
            "average_treatment_effect_in_propensity_eligible_population"
            if min_propensity is not None or max_propensity is not None
            else "average_treatment_effect"
        ),
        "propensity_overlap": overlap,
        "nuisance_calibration": nuisance_calibration,
        "features": len(definitions),
        "confounders": len(adjustment_defs),
        "treatment_nuisance_features": len(nuisance_candidate_definitions),
        "outcome_nuisance_features": len(nuisance_candidate_definitions),
        "nuisance_routing": {
            "treatment_feature_ids": [str(f["feature_id"]) for f in propensity_defs],
            "outcome_feature_ids": [str(f["feature_id"]) for f in outcome_defs],
            "selected_task_supports_used_for_nuisances": False,
            "training_crossfit_design": "all_candidates_with_fold_local_regularization",
            "heldout_design": "average_of_the_inner_fold_all_candidate_models",
            "heldout_ensemble_models": int(len(set(inner_fold_ids.tolist()))),
            "training_crossfit_candidates": len(nuisance_candidate_definitions),
            "heldout_candidates": len(nuisance_candidate_definitions),
            "recalibration": "own_fold_platt_heldout_only",
            "dina_reuses_external_nuisances": estimator == "dina"
            or (ensemble is not None and "dina" in ensemble["fitted_estimators"]),
        },
        "nuisance_fit_audit_summary": nuisance_audit_summary,
        "nuisance_learner": _nuisance_learner_summary(nuisance_learner, nuisance_audit_summary),
        **(
            {"nuisance_model_store": nuisances["model_store"]}
            if nuisance_model_store is not None
            else {}
        ),
        "effect_modifiers": len(effect_defs),
        "ate_aipw": ate,
        "standard_error": standard_error,
        "confidence_interval_95": (
            [ate - 1.96 * standard_error, ate + 1.96 * standard_error]
            if standard_error is not None
            else None
        ),
        "mean_estimated_cate": float(np.nanmean(cate)) if np.isfinite(cate).any() else None,
        **(
            {
                "mean_estimated_cate_by_estimator": {
                    key.removeprefix("component_tau_"): float(np.mean(values))
                    for key, values in effect_predictions.items()
                    if key.startswith("component_tau_")
                }
            }
            if ensemble is not None
            else {}
        ),
        **(
            {
                "mean_causal_forest_lower_95": (
                    float(np.nanmean(cate_lower_values))
                    if np.isfinite(cate_lower_values).any()
                    else None
                ),
                "mean_causal_forest_upper_95": (
                    float(np.nanmean(cate_upper_values))
                    if np.isfinite(cate_upper_values).any()
                    else None
                ),
            }
            if estimator == "causal_forest"
            else {}
        ),
        "propensity_min": float(np.min(propensity)) if len(propensity) else None,
        "propensity_max": float(np.max(propensity)) if len(propensity) else None,
        "propensity_clip": propensity_clip,
        "clipped_low_rows": int(np.sum(propensity < propensity_clip)),
        "clipped_high_rows": int(np.sum(propensity > 1.0 - propensity_clip)),
        "predictions_path": str(output_dir / "predictions.csv"),
    }
    LOGGER.info(
        "Stage 2 final nuisance calibration=%s overlap=%s",
        diagnostics["nuisance_calibration"],
        overlap,
    )
    _write_json(diagnostics_path, diagnostics)
    _write_json(
        complete_path,
        {
            "status": "complete",
            "schema_version": ESTIMATION_CHECKPOINT_SCHEMA_VERSION,
            "input_fingerprint": estimation_input_fingerprint,
            "outcome_type": outcome_type,
            "model_family": model_family,
            "outcome_model_contract": fit_audit["outcome_model_contract"],
            "completed_at": _now(),
            "rows": len(heldout_ids),
        },
    )
    return diagnostics


def _validated_heldout_measurement_frame(
    frame: pd.DataFrame,
    *,
    heldout_ids: Sequence[int],
    feature_names: Sequence[str],
    source: str,
) -> pd.DataFrame:
    expected_columns = ["_oci_row_id", *map(str, feature_names)]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"{source} columns do not match its measurement definitions: "
            f"expected={expected_columns!r}, actual={list(frame.columns)!r}"
        )
    numeric_ids = pd.to_numeric(frame["_oci_row_id"], errors="coerce")
    if numeric_ids.isna().any() or not np.allclose(
        numeric_ids.to_numpy(dtype=float),
        np.rint(numeric_ids.to_numpy(dtype=float)),
    ):
        raise ValueError(f"{source} contains invalid row identifiers")
    actual_ids = numeric_ids.astype(int).tolist()
    expected_ids = [int(value) for value in heldout_ids]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise ValueError(f"{source} rows do not match the outer-heldout partition")
    validated = frame.copy()
    validated["_oci_row_id"] = numeric_ids.astype(int)
    return validated


def _load_reusable_archived_heldout_measurements(
    *,
    output_dir: Path,
    heldout_ids: Sequence[int],
    measurement_definitions: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
) -> tuple[pd.DataFrame, set[str], dict[str, Any]]:
    """Load only definition-identical raw measurements from the reselection archive."""

    empty = pd.DataFrame({"_oci_row_id": [int(value) for value in heldout_ids]})
    audit: dict[str, Any] = {
        "schema_version": HELDOUT_MEASUREMENT_REUSE_SCHEMA_VERSION,
        "status": "cache_rejected",
        "cache_schema_version": cache.get("schema_version"),
        "required_features": len(measurement_definitions),
        "cache_available_features": 0,
        "reused_features": [],
        "definition_incompatible_features": [],
    }
    try:
        stage2_dir = Path(output_dir).parent.resolve()
        state_path = stage2_dir / "reselection_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise ValueError("reselection state is not an object")
        if (
            state.get("schema_version")
            != STAGE2_RESELECTION_MIGRATION_SCHEMA_VERSION
            or state.get("status") not in {"prepared", "complete"}
        ):
            raise ValueError("reselection state is not a prepared compatible migration")
        archive_relative = Path(str(state.get("archive_path") or ""))
        if (
            not archive_relative.parts
            or archive_relative.is_absolute()
            or ".." in archive_relative.parts
        ):
            raise ValueError("reselection archive path is invalid")
        archive_dir = (stage2_dir / archive_relative).resolve()
        if stage2_dir not in archive_dir.parents:
            raise ValueError("reselection archive escapes the Stage 2 directory")
        manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("reselection_id") != state.get("reselection_id")
            or manifest.get("policy_fingerprint") != state.get("policy_fingerprint")
        ):
            raise ValueError("reselection archive manifest does not match its state")
        source_relative = Path(str(cache.get("source_artifact_path") or ""))
        if (
            not source_relative.parts
            or source_relative.is_absolute()
            or ".." in source_relative.parts
        ):
            raise ValueError("cached held-out artifact path is invalid")
        archive_artifacts = (archive_dir / "artifacts").resolve()
        source_path = (archive_artifacts / source_relative).resolve()
        if archive_artifacts not in source_path.parents:
            raise ValueError("cached held-out artifact escapes its archive")
        cached_frame = pd.read_csv(source_path)
        if _frame_fingerprint(cached_frame) != cache.get("raw_frame_fingerprint"):
            raise ValueError("cached held-out artifact fingerprint changed")
        raw_cached_definitions = cache.get("measurement_definitions")
        if not isinstance(raw_cached_definitions, list) or not all(
            isinstance(value, Mapping) for value in raw_cached_definitions
        ):
            raise ValueError("cached held-out measurement definitions are invalid")
        cached_definitions = [dict(value) for value in raw_cached_definitions]
        cached_names = [str(value.get("name") or "") for value in cached_definitions]
        cached_frame = _validated_heldout_measurement_frame(
            cached_frame,
            heldout_ids=heldout_ids,
            feature_names=cached_names,
            source="archived held-out measurement cache",
        )
        audit["cache_available_features"] = len(cached_definitions)
        audit["source_artifact_path"] = str(source_relative)
        audit["source_frame_fingerprint"] = cache.get("raw_frame_fingerprint")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        audit["rejection_reason"] = f"{type(exc).__name__}: {exc}"
        LOGGER.warning(
            "Stage 2 held-out measurement cache rejected; extracting all required "
            "components instead: %s",
            audit["rejection_reason"],
        )
        return empty, set(), audit

    cached_by_id = {
        str(value.get("feature_id") or ""): value for value in cached_definitions
    }
    reusable_names: list[str] = []
    incompatible_names: list[str] = []
    for definition in measurement_definitions:
        current_identity = frozen_measurement_definition_identity(definition)
        cached_identity = cached_by_id.get(current_identity["feature_id"])
        name = str(definition["name"])
        if cached_identity == current_identity and name in cached_frame.columns:
            reusable_names.append(name)
        elif cached_identity is not None:
            incompatible_names.append(name)
    audit["status"] = "cache_accepted"
    audit["reused_features"] = reusable_names
    audit["definition_incompatible_features"] = incompatible_names
    return (
        cached_frame[["_oci_row_id", *reusable_names]].copy(),
        set(reusable_names),
        audit,
    )


def _extract_outer_heldout_measurements(
    *,
    dataset: pd.DataFrame,
    heldout_ids: Sequence[int],
    text_column: str,
    measurement_definitions: Sequence[Mapping[str, Any]],
    output_dir: Path,
    request_json: RequestJSON,
    workers: int,
    max_prompt_chars: int,
    feature_batch_size: int,
    request_identity: Mapping[str, Any],
    tokenizer: Any | None,
    frozen_cache: Mapping[str, Any] | None,
    serial_extraction: Mapping[str, Any],
) -> pd.DataFrame:
    """Reuse archived raw components and extract only cache misses."""

    heldout_dir = Path(output_dir) / "extraction" / "heldout"
    if frozen_cache is None:
        return extract_rows(
            dataset=dataset,
            row_ids=heldout_ids,
            text_column=text_column,
            definitions=measurement_definitions,
            output_dir=heldout_dir,
            request_json=request_json,
            workers=workers,
            max_prompt_chars=max_prompt_chars,
            feature_batch_size=feature_batch_size,
            request_identity=request_identity,
            tokenizer=tokenizer,
            **dict(serial_extraction),
        )

    cached_frame, reused_names, audit = (
        _load_reusable_archived_heldout_measurements(
            output_dir=output_dir,
            heldout_ids=heldout_ids,
            measurement_definitions=measurement_definitions,
            cache=frozen_cache,
        )
    )
    missing_definitions = [
        definition
        for definition in measurement_definitions
        if str(definition["name"]) not in reused_names
    ]
    missing_names = [str(definition["name"]) for definition in missing_definitions]
    if missing_definitions:
        newly_extracted = extract_rows(
            dataset=dataset,
            row_ids=heldout_ids,
            text_column=text_column,
            definitions=missing_definitions,
            output_dir=heldout_dir / "new_measurements",
            request_json=request_json,
            workers=workers,
            max_prompt_chars=max_prompt_chars,
            feature_batch_size=feature_batch_size,
            request_identity=request_identity,
            tokenizer=tokenizer,
            **dict(serial_extraction),
        )
        newly_extracted = _validated_heldout_measurement_frame(
            newly_extracted,
            heldout_ids=heldout_ids,
            feature_names=missing_names,
            source="new held-out measurements",
        )
    else:
        newly_extracted = pd.DataFrame(
            {"_oci_row_id": [int(value) for value in heldout_ids]}
        )

    combined = cached_frame.merge(
        newly_extracted,
        on="_oci_row_id",
        how="inner",
        sort=False,
        validate="one_to_one",
    )
    required_names = [str(definition["name"]) for definition in measurement_definitions]
    combined = _validated_heldout_measurement_frame(
        combined[["_oci_row_id", *required_names]],
        heldout_ids=heldout_ids,
        feature_names=required_names,
        source="combined held-out measurements",
    )
    _write_frame(heldout_dir / "extracted.csv", combined)
    audit.update(
        {
            "completed_at": _now(),
            "rows": len(combined),
            "required_features": len(required_names),
            "reused_feature_count": len(reused_names),
            "newly_extracted_feature_count": len(missing_names),
            "newly_extracted_features": missing_names,
            "reused_measurement_model": str(
                frozen_cache.get("extraction_model") or ""
            ),
            "new_measurement_request_identity": dict(request_identity),
            "mixed_extractor_models": bool(
                missing_names
                and reused_names
                and str(frozen_cache.get("extraction_model") or "")
                != str(request_identity.get("model") or "")
            ),
            "combined_frame_fingerprint": _frame_fingerprint(combined),
        }
    )
    _write_json(heldout_dir / "measurement_reuse.json", audit)
    _write_json(
        heldout_dir / "complete.json",
        {
            "status": "complete",
            "schema_version": HELDOUT_MEASUREMENT_REUSE_SCHEMA_VERSION,
            "completed_at": _now(),
            "rows": len(combined),
            "features": len(required_names),
            "reused_features": len(reused_names),
            "newly_extracted_features": len(missing_names),
            "combined_frame_fingerprint": _frame_fingerprint(combined),
        },
    )
    return combined


def _heldout_failure_summary(
    output_dir: Path, *, frozen_cache_used: bool
) -> tuple[dict[str, Any], str]:
    """Load the failure summary of the held-out extraction that produced the values."""

    heldout_dir = Path(output_dir) / "extraction" / "heldout"
    path = (
        heldout_dir / "new_measurements" / "failure_summary.json"
        if frozen_cache_used
        else heldout_dir / "failure_summary.json"
    )
    if not path.is_file():
        return {}, (
            "unavailable_reused_archived_measurements"
            if frozen_cache_used
            else "unavailable_no_failure_summary"
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    return dict(summary), (
        "newly_extracted_heldout_issue_events_archived_measurements_unknown"
        if frozen_cache_used
        else "heldout_extraction_issue_events"
    )


def _run_stage2_statistical_selection(
    selector_arguments: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[Any]]:
    """Run a large fold selector outside the thread-orchestration GIL."""

    arguments = dict(selector_arguments)
    from . import stage2_modeling_pool as modeling

    # The multi-model coordinator submits leaf jobs to the shared run pool.
    # Putting the coordinator in another process would create nested pools.
    candidate_count = len(arguments.get("definitions") or [])
    if (
        modeling.enabled()
        or candidate_count < STATISTICAL_SELECTION_PROCESS_ISOLATION_MIN_CANDIDATES
    ):
        return select_stage2_features_multi_model(**arguments)
    LOGGER.info(
        "Stage 2 statistical selection process isolation candidates=%s threshold=%s",
        candidate_count,
        STATISTICAL_SELECTION_PROCESS_ISOLATION_MIN_CANDIDATES,
    )
    # Loky serializes the submitted callable and does not re-import an
    # interactive ``__main__`` module.  The large-candidate path therefore
    # remains process isolated when called from notebooks, ``python -c``, or
    # stdin, where the stdlib spawn executor cannot safely bootstrap.
    with LokyProcessPoolExecutor(max_workers=1) as executor:
        return executor.submit(
            select_stage2_features_multi_model,
            **arguments,
        ).result()


def _config_nuisance_learner(config: Any) -> dict[str, Any]:
    """The configured final nuisance learner (elastic net for configs without the key)."""
    spec = getattr(config, "nuisance_learner_spec", None)
    return spec() if callable(spec) else {"name": "elastic_net"}


def initial_feature_modeling_definitions(definitions):
    """Normalize the contracts used by both preparation and initial extraction."""
    current = []
    for raw_feature in definitions:
        feature = dict(raw_feature)
        value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
        if value_type in {"binary", "categorical", "ordinal"}:
            feature["categories_or_unit"] = _validated_closed_category_values(
                value_type=value_type,
                values=feature.get("categories_or_unit") or [],
                source=f"feature {feature.get('name')!r}",
                keep_unknown_like=feature.get("configured_explicit_feature") is True,
            )
        current.append(_normalized_feature_modeling_definition(feature))
    return current


def run_fold_analysis(
    *,
    dataset: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    clinical_question: str,
    unit_id_column: str,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    inner_folds: int,
    seed: int,
    output_dir: Path,
    request_json: RequestJSON,
    selection_consolidation_request_json: RequestJSON | None = None,
    config: Any,
    extraction_tokenizer: Any | None = None,
    stage1_packets: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run extraction supervision, fold-local selection, and causal-forest estimation."""

    # Study context is supplied only to modeling reviews, never extraction supervision.
    ontology_refinement_min_failure_patients = int(config.ontology_refinement_min_failure_patients)
    max_ontology_refinement_rounds = int(config.max_ontology_refinement_rounds)
    extraction_feature_batch_size = int(config.extraction_feature_batch_size)
    serial_extraction = _configured_serial_extraction(config)
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    heldout_ids = [int(value) for value in split["heldout_row_ids"]]
    inner_splits = list(split.get("inner_splits") or []) or _fallback_inner_splits(
        fit_ids,
        folds=inner_folds,
        seed=seed,
    )
    extraction_llm = getattr(config, "extraction_llm", None)
    configured_extraction_model = str(getattr(extraction_llm, "model", ""))
    runtime_extraction_model = str(
        getattr(extraction_llm, "runtime_model", "") or ""
    ).strip()
    extraction_identity = {
        "model": runtime_extraction_model or configured_extraction_model,
        "configured_checkpoint_model": configured_extraction_model,
        "runtime_continuation_model": runtime_extraction_model or None,
    }
    primary_identity = {
        "model": str(getattr(config, "model", "")),
    }
    extraction_workers = int(
        getattr(extraction_llm, "workers", getattr(config, "workers", 1))
    )

    maximum_review_rounds = int(getattr(config, "max_review_rounds", 1))
    frozen_snapshot = _load_frozen_preselection_snapshot(
        output_dir=output_dir,
        dataset=dataset,
        definitions=definitions,
        fit_ids=fit_ids,
        heldout_ids=heldout_ids,
        inner_splits=inner_splits,
        unit_id_column=unit_id_column,
        text_column=text_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        stage1_packets=stage1_packets,
        config=config,
    )
    frozen_review_convergence: dict[str, Any] | None = None
    frozen_heldout_measurement_cache: dict[str, Any] | None = None
    if frozen_snapshot is None:
        current = initial_feature_modeling_definitions(definitions)
        review_rounds = 0
        ontology_refinement_rounds = 0
        review_converged = False
        review_history: list[dict[str, Any]] = []
        final_fit_all: pd.DataFrame | None = None
        final_fit_raw: pd.DataFrame | None = None
        final_fit_definitions: list[dict[str, Any]] | None = None
        final_fit_failure_summary: dict[str, Any] | None = None
    else:
        (
            frozen_matrix,
            frozen_definitions,
            frozen_metadata,
            frozen_heldout_measurement_cache,
        ) = frozen_snapshot
        current = [
            _normalized_feature_modeling_definition(feature)
            for feature in frozen_definitions
        ]
        review_rounds = int(frozen_metadata.get("review_rounds") or 0)
        ontology_refinement_rounds = int(
            frozen_metadata.get("ontology_refinement_rounds") or 0
        )
        review_converged = bool(frozen_metadata.get("review_converged"))
        stored_convergence = frozen_metadata.get("review_convergence")
        frozen_review_convergence = (
            copy.deepcopy(dict(stored_convergence))
            if isinstance(stored_convergence, Mapping)
            else None
        )
        review_history = list(
            (frozen_review_convergence or {}).get("history") or []
        )
        final_fit_all = frozen_matrix
        final_fit_raw = frozen_matrix.copy()
        final_fit_definitions = copy.deepcopy(current)
        final_fit_failure_summary = {}
        LOGGER.info(
            "reuse frozen Stage 2 preselection snapshot: %s",
            Path(output_dir) / "preselection",
        )

    review_round_indices = (
        range(1, maximum_review_rounds + 1) if frozen_snapshot is None else ()
    )
    for round_index in review_round_indices:
        review_rounds = round_index
        round_dir = output_dir / "ontology_supervision" / f"round_{round_index:03d}"
        _write_json(round_dir / "definitions_before_extraction.json", {"features": current})
        raw_extracted, extracted_definitions, feedback_rounds = (
            _extract_training_with_ontology_feedback(
                dataset=dataset,
                row_ids=fit_ids,
                text_column=text_column,
                definitions=current,
                output_dir=round_dir / "extraction",
                feedback_dir=round_dir / "failure_ontology_refinement",
                request_json=request_json,
                interpretation_workers=int(getattr(config, "workers", 1)),
                workers=extraction_workers,
                max_prompt_chars=int(config.extraction_max_prompt_chars),
                feature_batch_size=extraction_feature_batch_size,
                minimum_failure_patients=ontology_refinement_min_failure_patients,
                max_refinement_rounds=max_ontology_refinement_rounds,
                request_identity=extraction_identity,
                tokenizer=extraction_tokenizer,
                prior_extracted=final_fit_raw,
                prior_definitions=final_fit_definitions,
                prior_failure_summary=final_fit_failure_summary,
                **serial_extraction,
            )
        )
        ontology_refinement_rounds += feedback_rounds
        extracted_definitions = [
            _normalized_feature_modeling_definition(feature)
            for feature in extracted_definitions
        ]
        failure_summary = json.loads(
            (
                round_dir
                / "failure_ontology_refinement"
                / "final_failure_summary.json"
            ).read_text(encoding="utf-8")
        )
        extracted, extracted_definitions, harmonization = _harmonize_training_extraction(
            extracted=raw_extracted,
            definitions=extracted_definitions,
            output_dir=round_dir / "harmonization",
            request_json=request_json,
            workers=int(getattr(config, "workers", 1)),
            max_prompt_chars=int(config.max_prompt_chars),
        )
        summaries = feature_summaries(extracted, extracted_definitions)
        _write_json(round_dir / "aggregate_extraction_summary.json", summaries)
        reviewed, changed, review_report = _request_aggregate_ontology_supervisor(
            definitions=extracted_definitions,
            summaries=summaries,
            failure_summary=failure_summary,
            output_dir=round_dir / "supervisor",
            request_json=request_json,
            workers=int(getattr(config, "workers", 1)),
            request_identity=primary_identity,
        )
        final_fit_all = extracted
        final_fit_raw = raw_extracted
        final_fit_definitions = extracted_definitions
        final_fit_failure_summary = failure_summary
        current = [
            _normalized_feature_modeling_definition(feature) for feature in reviewed
        ]
        review_history.append(
            {
                "round": round_index,
                "failure_ontology_refinement_rounds": feedback_rounds,
                "harmonization": harmonization,
                "supervisor_changed_feature_ids": list(
                    review_report["changed_feature_ids"]
                ),
            }
        )
        _write_json(
            round_dir / "complete.json",
            {
                "status": "complete",
                "schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
                "completed_at": _now(),
                "ontology_changed": changed,
                "features": len(current),
            },
        )
        if not changed:
            review_converged = True
            break

    if (
        final_fit_all is None
        or final_fit_raw is None
        or final_fit_definitions is None
        or final_fit_failure_summary is None
    ):
        raise RuntimeError("Stage 2 ontology supervision did not perform extraction")

    # A revision in the last allowed supervisor round has not yet been applied
    # to patient text. Re-extract only changed candidates before statistical tests.
    if frozen_snapshot is not None:
        pass
    elif _value_fingerprint(final_fit_definitions) != _value_fingerprint(current):
        final_fit_raw, current, feedback_rounds = _extract_training_with_ontology_feedback(
            dataset=dataset,
            row_ids=fit_ids,
            text_column=text_column,
            definitions=current,
            output_dir=output_dir / "extraction" / "all_candidates_fit",
            feedback_dir=output_dir / "extraction" / "all_candidates_fit_refinement",
            request_json=request_json,
            interpretation_workers=int(getattr(config, "workers", 1)),
            workers=extraction_workers,
            max_prompt_chars=int(config.extraction_max_prompt_chars),
            feature_batch_size=extraction_feature_batch_size,
            minimum_failure_patients=ontology_refinement_min_failure_patients,
            max_refinement_rounds=max_ontology_refinement_rounds,
            request_identity=extraction_identity,
            tokenizer=extraction_tokenizer,
            prior_extracted=final_fit_raw,
            prior_definitions=final_fit_definitions,
            prior_failure_summary=final_fit_failure_summary,
            **serial_extraction,
        )
        ontology_refinement_rounds += feedback_rounds
        final_fit_failure_summary = json.loads(
            (
                output_dir / "extraction" / "all_candidates_fit_refinement"
                / "final_failure_summary.json"
            ).read_text(encoding="utf-8")
        )
        final_fit_all, current, _ = _harmonize_training_extraction(
            extracted=final_fit_raw,
            definitions=current,
            output_dir=output_dir / "extraction" / "all_candidates_fit_harmonization",
            request_json=request_json,
            workers=int(getattr(config, "workers", 1)),
            max_prompt_chars=int(config.max_prompt_chars),
        )
    else:
        _write_frame(
            output_dir / "extraction" / "all_candidates_fit" / "extracted.csv",
            final_fit_all,
        )

    # Cells whose extraction failed after every repair (S2A-004). A frozen
    # preselection snapshot carries no failure provenance; that is recorded.
    min_informative_patients = int(
        getattr(config, "min_informative_patients", DEFAULT_MIN_INFORMATIVE_PATIENTS)
    )
    max_extraction_failure_fraction = float(
        getattr(
            config, "max_extraction_failure_fraction", DEFAULT_MAX_EXTRACTION_FAILURE_FRACTION
        )
    )
    fit_failure_mask_available = frozen_snapshot is None
    fit_failure_mask = extraction_failures.failure_mask(
        final_fit_raw.set_index("_oci_row_id", drop=False).loc[fit_ids].reset_index(drop=True)
        if "_oci_row_id" in final_fit_raw
        else final_fit_raw,
        final_fit_failure_summary,
        current,
    )
    _write_frame(output_dir / "extraction" / "failure_mask_fit.csv", fit_failure_mask)
    fit_treatment = dataset.iloc[fit_ids][treatment_column].to_numpy(dtype=float)
    # Measure coverage after repair/review, before any supervised modeling.
    # Keep the complete extraction checkpoint available for audit and reuse.
    final_fit_all, current, missingness_report = _filter_sparse_training_candidates(
        final_fit_all, current,
        audit_path=output_dir / "extraction" / "candidate_missingness_filter.json",
        health_audit_path=output_dir / "extraction" / "fit_health.json",
        min_informative_patients=min_informative_patients,
        max_extraction_failure_fraction=max_extraction_failure_fraction,
        failure_mask=fit_failure_mask,
        treatment=fit_treatment,
    )
    missingness_report["extraction_failure_mask_source"] = (
        "training_extraction_issue_events"
        if fit_failure_mask_available
        else "unavailable_for_frozen_preselection_snapshot"
    )
    _write_json(
        output_dir / "extraction" / "failure_report_fit.json",
        missingness_report["training_extraction_failure_report"],
    )
    if missingness_report.get("training_documented_unknown_report") is not None:
        _write_json(
            output_dir / "extraction" / "documented_unknown_report_fit.json",
            missingness_report["training_documented_unknown_report"],
        )
    preselection_matrix_relative = (
        Path("extraction") / "coverage_filtered_fit"
        / _frame_fingerprint(final_fit_all) / "extracted.csv"
    )
    _write_frame(output_dir / preselection_matrix_relative, final_fit_all)

    # Supervised definition search is separate from the outcome-blind ontology
    # supervisor. Keep the latter's cache reusable when this search is enabled.
    from .stage2_estimand_ontology_config import EstimandOntologyConfig

    estimand_policy = getattr(config, "estimand_ontology", EstimandOntologyConfig())
    estimand_report = {"status": "disabled"}
    if estimand_policy.enabled or any(f.get("estimand_ontology") for f in current):
        from .stage2_estimand_ontology import refine_estimand_ontologies

        def extract_ontology_alternatives(alternatives, directory):
            # This callback exposes only training text to extraction. Its
            # definition stays frozen during comparison; harmonization observes
            # values without access to treatment or outcome.
            measured = extract_rows(
                dataset=dataset[[text_column]], row_ids=fit_ids, text_column=text_column,
                definitions=alternatives, output_dir=directory / "extraction",
                request_json=request_json, workers=extraction_workers,
                max_prompt_chars=int(config.extraction_max_prompt_chars),
                feature_batch_size=extraction_feature_batch_size,
                request_identity=extraction_identity, tokenizer=extraction_tokenizer,
                **serial_extraction,
            )
            measured, harmonized_definitions, _ = _harmonize_training_extraction(
                extracted=measured, definitions=alternatives,
                output_dir=directory / "harmonization", request_json=request_json,
                workers=int(getattr(config, "workers", 1)),
                max_prompt_chars=int(config.max_prompt_chars),
            )
            return measured, harmonized_definitions

        search_labels = dataset.iloc[fit_ids][[treatment_column, outcome_column]].copy()
        search_labels.columns = ["treatment", "outcome"]
        search_labels.insert(0, "_oci_row_id", fit_ids)
        final_fit_all, current, estimand_report = refine_estimand_ontologies(
            extracted_fit=final_fit_all, labels=search_labels.reset_index(drop=True),
            definitions=current, inner_splits=inner_splits, outcome_type=outcome_type,
            clinical_question=clinical_question, request_json=request_json,
            extract_alternatives=extract_ontology_alternatives,
            output_dir=output_dir / "estimand_ontology", policy=estimand_policy,
            source_text_fingerprint=_frame_fingerprint(dataset.iloc[fit_ids][[text_column]]),
            request_identity={"primary": primary_identity, "extraction": extraction_identity},
            seed=seed,
            proposal_workers=int(getattr(config, "workers", 1)),
            propensity_bounds=(
                getattr(config, "min_propensity", None) if getattr(config, "min_propensity", None) is not None else 0.1,
                getattr(config, "max_propensity", None) if getattr(config, "max_propensity", None) is not None else 0.9,
            ),
        )
        # Alternative definitions create new measurements; enforce the same
        # coverage requirement before they reach downstream selection.
        final_fit_all, current, alternative_missingness_report = _filter_sparse_training_candidates(
            final_fit_all, current,
            audit_path=output_dir / "extraction" / "estimand_candidate_missingness_filter.json",
            health_audit_path=output_dir / "extraction" / "fit_health.json",
            min_informative_patients=min_informative_patients,
            max_extraction_failure_fraction=max_extraction_failure_fraction,
            failure_mask=fit_failure_mask,
            treatment=fit_treatment,
        )
        missingness_report["after_estimand_refinement"] = alternative_missingness_report
        # A frozen preselection snapshot may point at the original matrix.
        # Never overwrite it when adding/removing alternative measurements.
        preselection_matrix_relative = (Path("extraction") / "estimand_candidates_fit"
                                    / _frame_fingerprint(final_fit_all) / "extracted.csv")
        _write_frame(output_dir / preselection_matrix_relative, final_fit_all)

    selection_dir = output_dir / "selection"
    statistical_policy = getattr(config, "statistical_selection", None)
    if statistical_policy is None:
        raise ValueError("Stage 2 config is missing statistical_selection policy")
    statistical_policy = replace(
        statistical_policy,
        min_propensity=getattr(config, "min_propensity", None),
        max_propensity=getattr(config, "max_propensity", None),
    )
    consolidation_policy = getattr(config, "selection_consolidation", None)
    if consolidation_policy is None:
        raise ValueError("Stage 2 config is missing selection_consolidation policy")
    selection_input = {
        "schema_version": STAGE2_ROLE_SELECTION_SCHEMA_VERSION,
        "prompt_version": clinical_prompts.PROMPT_VERSION,
        "study_context": {"clinical_question": clinical_question, "outcome_type": outcome_type},
        "temporal_scope": TEMPORAL_SCOPE,
        "extracted_fit_fingerprint": _frame_fingerprint(final_fit_all),
        "treatment_outcome_fingerprint": _frame_fingerprint(
            dataset.iloc[fit_ids][[treatment_column, outcome_column]].reset_index(drop=True)
        ),
        "stage1_packets_fingerprint": _value_fingerprint(list(stage1_packets)),
        "definitions": current,
        "inner_splits": inner_splits,
        "outcome_type": outcome_type,
        "selection_consolidation_policy": consolidation_policy.scientific_dict(),
        "preselection_matrix_path": str(preselection_matrix_relative),
        "candidate_missingness_filter": {
            "schema_version": missingness_report["schema_version"],
            "min_informative_patients": min_informative_patients,
            "max_extraction_failure_fraction": max_extraction_failure_fraction,
            "scope": "outer_training",
            "audit_fingerprint": _value_fingerprint(missingness_report),
        },
        **({"estimand_ontology": {"policy": estimand_policy.public_dict(),
                                   "search_fingerprint": estimand_report.get("input_fingerprint")}}
           if estimand_policy.enabled else {}),
        "selection_consolidation_llm_model": str(getattr(config, "model", "")),
        "statistical_component_schema_version": (
            statistical_policy.multi_model.public_dict()["schema_version"]
        ),
        # Final-estimation-only keys (the selected-modifier penalty) are omitted:
        # changing them refits the estimation and reuses this selection.
        "statistical_selection_policy": statistical_policy.selection_dict(),
        "role_adjudication_policy": config.role_adjudication.public_dict(),
        "role_adjudication_llm_model": str(getattr(config, "model", "")),
        "predictive_nuisance_sources": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("stage2_nuisance_selection.py", "stage2_multi_model_selection.py")
        },
        # A completed selection must not bypass a changed ranking criterion.
        # Numerical evidence has its own independent identity and remains reusable.
        "modifier_ranking_source_sha256": hashlib.sha256(
            Path(__file__).with_name("stage2_modifier_ranking.py").read_bytes()
        ).hexdigest(),
    }
    if statistical_policy.multi_model.modifier_selection.rule == "orthogonal_tests":
        # The AIPW projection test clips propensities exactly as the AIPW ATE does.
        # The trees matter only to final estimation, which has its own identity.
        selection_input["modifier_test_propensity_clip"] = float(config.propensity_clip)
        # The testing nuisances use the final-estimation nuisance learner and
        # its training construction (raw out-of-fold predictions).
        selection_input["modifier_test_nuisance_learner"] = _config_nuisance_learner(config)
        selection_input["modifier_test_nuisance_construction"] = NUISANCE_CONSTRUCTION_VERSION
        selection_input["modifier_tests_source_sha256"] = hashlib.sha256(
            Path(__file__).with_name("stage2_modifier_tests.py").read_bytes()
        ).hexdigest()
    else:
        # The balanced_evidence count search scores causal forests of this size.
        selection_input["modifier_count_estimation_trees"] = int(config.estimation_trees)
    if getattr(
        getattr(statistical_policy.multi_model, "matched_batch", None), "enabled", False
    ):
        selection_input["matched_batch_source_sha256"] = hashlib.sha256(
            Path(__file__).with_name("stage2_matched_batch.py").read_bytes()
        ).hexdigest()
    selection_fingerprint = _value_fingerprint(selection_input)
    selection_report_path = selection_dir / "elastic_net_selection.json"
    selected_path = selection_dir / "selected_definitions.json"
    dependencies_path = selection_dir / "measurement_definitions.json"
    latent_states_path = selection_dir / "selected_latent_states.json"
    nuisance_candidates_path = selection_dir / "nuisance_candidates.json"
    selection_complete_path = selection_dir / "complete.json"
    selection_input_path = selection_dir / "input.json"
    selection_report: dict[str, Any] | None = None
    selected: list[dict[str, Any]] | None = None
    measurement_definitions: list[dict[str, Any]] | None = None
    latent_states: list[dict[str, Any]] | None = None
    modeling_fit_frame: pd.DataFrame | None = None
    nuisance_candidates = None
    if (
        selection_report_path.is_file()
        and selected_path.is_file()
        and dependencies_path.is_file()
        and latent_states_path.is_file()
        and selection_complete_path.is_file()
        and selection_input_path.is_file()
        and nuisance_candidates_path.is_file()
    ):
        try:
            completion = json.loads(selection_complete_path.read_text(encoding="utf-8"))
            prior_input = json.loads(selection_input_path.read_text(encoding="utf-8"))
            cached_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
            cached_selected = json.loads(selected_path.read_text(encoding="utf-8"))
            cached_dependencies = json.loads(dependencies_path.read_text(encoding="utf-8"))
            cached_latent_states = json.loads(latent_states_path.read_text(encoding="utf-8"))
            if (
                completion.get("input_fingerprint") == selection_fingerprint
                and prior_input.get("input_fingerprint") == selection_fingerprint
                and cached_report.get("schema_version")
                == STAGE2_ROLE_SELECTION_SCHEMA_VERSION
                and isinstance(cached_selected.get("features"), list)
                and isinstance(cached_dependencies.get("features"), list)
                and isinstance(cached_latent_states.get("latents"), list)
            ):
                selection_report = dict(cached_report)
                selected = [dict(feature) for feature in cached_selected["features"]]
                measurement_definitions = [
                    dict(feature) for feature in cached_dependencies["features"]
                ]
                latent_states = [dict(item) for item in cached_latent_states["latents"]]
                nuisance_candidates = json.loads(nuisance_candidates_path.read_text(encoding="utf-8"))
                if (nuisance_candidates.get("input_fingerprint") != selection_fingerprint
                        or not isinstance(nuisance_candidates.get("features"), list)
                        or not isinstance(nuisance_candidates.get("latents"), list)):
                    raise ValueError("incompatible nuisance candidate checkpoint")
                modeling_fit_frame = materialize_selected_latents(
                    frame=final_fit_all,
                    latent_states=nuisance_candidates["latents"],
                    measurement_definitions=current,
                )
                LOGGER.info(
                    "skip completed Stage 2 all-evidence role selection: %s",
                    selection_dir,
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            selection_report = None
            selected = None
            measurement_definitions = None
            latent_states = None
    elif selection_complete_path.is_file():
        try:
            incompatible = json.loads(
                selection_complete_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            incompatible = {}
        if incompatible.get("schema_version") not in {
            None,
            STAGE2_ROLE_SELECTION_SCHEMA_VERSION,
        }:
            raise RuntimeError(
                "this outer fold contains an incompatible Stage 2 selection schema. "
                "Preserve it for audit and use a fresh Stage 2 output directory."
            )
    if (
        selection_report is None
        or selected is None
        or measurement_definitions is None
        or latent_states is None
    ):
        _write_json(
            selection_input_path,
            {**selection_input, "input_fingerprint": selection_fingerprint},
        )
        (
            consolidated_fit,
            consolidated_definitions,
            consolidation_report,
            all_latent_entries,
        ) = consolidate_stage2_candidates(
            extracted_fit=final_fit_all,
            definitions=current,
            request_json=selection_consolidation_request_json or request_json,
            policy=consolidation_policy,
            output_dir=selection_dir / "candidate_consolidation",
            request_model=str(getattr(config, "model", "")),
            request_runtime_identity=(
                {
                    "endpoint": str(consolidation_policy.runtime_llm_endpoint),
                    "model": str(consolidation_policy.runtime_llm_model),
                }
                if str(consolidation_policy.runtime_llm_endpoint).strip()
                else None
            ),
        )
        nuisance_candidates = {
            "input_fingerprint": selection_fingerprint,
            "features": consolidated_definitions,
            "latents": all_latent_entries,
        }
        _write_json(nuisance_candidates_path, nuisance_candidates)
        (
            statistically_selected,
            elastic_net_report,
            _elastic_net_dependencies,
            _elastic_net_latent_states,
        ) = _run_stage2_statistical_selection(
            {
                "dataset": dataset,
                "extracted_fit": consolidated_fit,
                "definitions": consolidated_definitions,
                "inner_splits": inner_splits,
                "treatment_column": treatment_column,
                "outcome_column": outcome_column,
                "outcome_type": outcome_type,
                "seed": seed,
                "policy": statistical_policy,
                "checkpoint_dir": selection_dir / "multi_model",
            }
        )
        elastic_net_report["study_context"] = {"clinical_question": str(clinical_question), "outcome_type": str(outcome_type)}
        _write_json(
            selection_dir / "statistical_evidence.json",
            elastic_net_report,
        )
        nuisance_rows = elastic_net_report.get("cross_fitted_nuisance_models", {}).get(
            "predictions", []
        )
        if nuisance_rows:
            _write_frame(selection_dir / "nuisance_predictions.csv", pd.DataFrame(nuisance_rows))
        # Numerical nuisance retention is final; no LLM decides causal roles.
        selected = statistically_selected
        role_adjudication_report = {
            "schema_version": "stage2_predictive_nuisance_selection_v1",
            "status": "replaced_by_statistical_nuisance_selection",
            "selection_authority": "multi_model",
            "llm_roles_are_binding": False,
            "llm_requests": 0,
            "decisions": elastic_net_report.get("decisions", []),
            "retained_feature_ids": [str(f["feature_id"]) for f in selected],
        }
        modifier_count_report = None
        if consolidated_definitions:
            from .stage2_modifier_count import select_modifier_count

            (
                selected,
                role_adjudication_report,
                modifier_count_report,
            ) = select_modifier_count(
                dataset=dataset,
                extracted_fit=consolidated_fit,
                definitions=consolidated_definitions,
                inner_splits=inner_splits,
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                outcome_type=outcome_type,
                seed=seed,
                policy=statistical_policy,
                selected=selected,
                role_report=role_adjudication_report,
                statistical_report=elastic_net_report,
                request_json=request_json,
                role_policy=config.role_adjudication,
                output_dir=selection_dir / "modifier_count",
                numerical_checkpoint_dir=selection_dir / "multi_model",
                model_identity=str(getattr(config, "model", "")),
                estimation_trees=int(config.estimation_trees),
                run_numerical=_run_stage2_statistical_selection,
                fit_row_ids=fit_ids,
                propensity_clip=float(config.propensity_clip),
                tests_path=selection_dir / "modifier_tests.json",
                nuisance_model_store=selection_dir / "nuisance_models",
                nuisance_learner=_config_nuisance_learner(config),
            )
        measurement_definitions = measurement_definitions_for_selected(
            selected,
            current,
        )
        latent_states = latent_states_for_selected(selected, all_latent_entries)
        selection_report = {
            **elastic_net_report,
            "schema_version": STAGE2_ROLE_SELECTION_SCHEMA_VERSION,
            "statistical_component_schema_version": elastic_net_report["schema_version"],
            "statistical_provisional_decisions": elastic_net_report.get(
                "decisions", []
            ),
            "role_adjudication": role_adjudication_report,
            "final_role_assignment": (
                (
                    "predictive_nuisances_and_orthogonal_test_modifiers"
                    if modifier_count_report.get("rule") == "orthogonal_tests"
                    else "predictive_nuisances_and_nested_r_loss_modifiers"
                )
                if modifier_count_report is not None
                else "predictive_nuisances_and_joint_orthogonal_modifiers"
            ),
            "decisions": role_adjudication_report.get(
                "decisions", elastic_net_report.get("decisions", [])
            ),
            "retained_feature_ids": [
                str(feature["feature_id"]) for feature in selected
            ],
            "latent_construction": (
                "sequential_semantic_association_consolidation"
                if consolidation_policy.enabled
                else "disabled"
            ),
            "pairwise_association_screen": (
                "unsupervised_preselection_consolidation_only"
                if consolidation_policy.enabled
                else "disabled"
            ),
            "preselection_consolidation": consolidation_report,
            "candidate_missingness_filter": missingness_report,
            **({"estimand_ontology": estimand_report} if estimand_policy.enabled else {}),
            "candidate_counts": {
                "before_consolidation": len(current),
                "after_consolidation": len(consolidated_definitions),
                "selected": len(selected),
            },
            "measurement_dependency_feature_ids": [
                str(feature["feature_id"]) for feature in measurement_definitions
            ],
            "selected_latent_ids": [
                str(item["latent_id"]) for item in latent_states
            ],
            **(
                {"modifier_count_selection": modifier_count_report}
                if modifier_count_report is not None
                else {}
            ),
        }
        modeling_fit_frame = consolidated_fit
        _write_json(selection_report_path, selection_report)
        _write_json(selected_path, {"features": selected})
        _write_json(dependencies_path, {"features": measurement_definitions})
        _write_json(latent_states_path, {"latents": latent_states})
        _write_json(
            selection_complete_path,
            {
                "status": "complete",
                "schema_version": STAGE2_ROLE_SELECTION_SCHEMA_VERSION,
                "completed_at": _now(),
                "input_fingerprint": selection_fingerprint,
                "retained_features": len(selected),
                "measurement_dependencies": len(measurement_definitions),
                "selected_latents": len(latent_states),
            },
        )

    if modeling_fit_frame is None:
        raise RuntimeError("Stage 2 selection did not produce a modeling fit frame")
    selected_names = [str(feature["name"]) for feature in selected]
    fit_selected = modeling_fit_frame[
        [
            "_oci_row_id",
            *selected_names,
            *extraction_failures.companion_columns(modeling_fit_frame, selected),
        ]
    ].copy()
    _write_frame(output_dir / "extraction" / "fit" / "extracted.csv", fit_selected)
    _write_frame(output_dir / "extraction" / "fit" / "harmonized.csv", fit_selected)
    if selected:
        _assert_extraction_health(
            fit_selected,
            selected,
            scope="training",
            minimum_row_nonmissing_fraction=CATASTROPHIC_ROW_NONMISSING_FRACTION,
            audit_path=output_dir / "extraction" / "fit_health.json",
        )
    else:
        _write_json(
            output_dir / "extraction" / "fit_health.json",
            {
                "schema_version": "stage2_final_extraction_health_v1",
                "status": "not_applicable_no_selected_features",
                "scope": "training",
                "rows": len(fit_selected),
                "features": 0,
            },
        )

    # Held-out patients are measured on the COMPLETE nuisance-candidate catalog
    # (not only the selected features' dependencies), so held-out nuisances can
    # use the same all-candidate construction as the training OOF nuisances.
    # Definitions and harmonization plans are the frozen training ones.
    candidate_definitions = list(nuisance_candidates["features"])
    candidate_names = [str(feature["name"]) for feature in candidate_definitions]
    heldout_measurement_definitions = measurement_definitions_for_selected(
        candidate_definitions, current
    )
    heldout_latent_states = latent_states_for_selected(
        candidate_definitions, nuisance_candidates["latents"]
    )
    missing_selected = sorted(
        str(feature["feature_id"])
        for feature in measurement_definitions
        if str(feature["feature_id"])
        not in {str(item["feature_id"]) for item in heldout_measurement_definitions}
    )
    if missing_selected:
        raise RuntimeError(
            "selected measurement dependencies are missing from the nuisance-candidate "
            f"catalog: {missing_selected}"
        )
    heldout_raw = _extract_outer_heldout_measurements(
        dataset=dataset,
        heldout_ids=heldout_ids,
        text_column=text_column,
        measurement_definitions=heldout_measurement_definitions,
        output_dir=output_dir,
        request_json=request_json,
        workers=extraction_workers,
        max_prompt_chars=int(config.extraction_max_prompt_chars),
        feature_batch_size=extraction_feature_batch_size,
        request_identity=extraction_identity,
        tokenizer=extraction_tokenizer,
        frozen_cache=frozen_heldout_measurement_cache,
        serial_extraction=serial_extraction,
    )
    heldout_measurements, heldout_harmonization = _apply_harmonization_plans(
        heldout_raw,
        heldout_measurement_definitions,
        scope="outer_heldout",
    )
    _write_frame(
        output_dir / "extraction" / "heldout" / "measurement_harmonized.csv",
        heldout_measurements,
    )
    _write_json(
        output_dir / "extraction" / "heldout" / "harmonization.json",
        heldout_harmonization,
    )
    heldout_with_latents = materialize_selected_latents(
        frame=heldout_measurements,
        latent_states=heldout_latent_states,
        measurement_definitions=heldout_measurement_definitions,
    )
    # Held-out extraction failures use the same rule as training. Companion
    # indicator columns exist only for features with training failures; a
    # held-out failure in any other feature is encoded as missing and reported.
    heldout_failure_summary, heldout_failure_source = _heldout_failure_summary(
        output_dir, frozen_cache_used=frozen_heldout_measurement_cache is not None
    )
    heldout_failure_mask = extraction_failures.failure_mask(
        heldout_raw, heldout_failure_summary, heldout_measurement_definitions
    )
    _write_frame(output_dir / "extraction" / "failure_mask_heldout.csv", heldout_failure_mask)
    heldout_with_latents = extraction_failures.attach_failure_columns(
        heldout_with_latents, heldout_failure_mask, candidate_definitions
    )
    heldout_candidates = heldout_with_latents[
        [
            "_oci_row_id",
            *candidate_names,
            *extraction_failures.companion_columns(heldout_with_latents, candidate_definitions),
        ]
    ].copy()
    heldout_extraction = heldout_with_latents[
        [
            "_oci_row_id",
            *selected_names,
            *extraction_failures.companion_columns(heldout_with_latents, selected),
        ]
    ].copy()
    _write_frame(
        output_dir / "extraction" / "heldout" / "harmonized.csv",
        heldout_extraction,
    )
    if selected:
        _assert_extraction_health(
            heldout_extraction,
            selected,
            scope="heldout",
            minimum_row_nonmissing_fraction=CATASTROPHIC_ROW_NONMISSING_FRACTION,
            audit_path=output_dir / "extraction" / "heldout_health.json",
        )
    else:
        _write_json(
            output_dir / "extraction" / "heldout_health.json",
            {
                "schema_version": "stage2_final_extraction_health_v1",
                "status": "not_applicable_no_selected_features",
                "scope": "heldout",
                "rows": len(heldout_extraction),
                "features": 0,
            },
        )

    harmonization_validation_fallbacks = [
        {
            "feature_id": str(feature["feature_id"]),
            "name": str(feature["name"]),
            **copy.deepcopy(feature["harmonization_fallback"]),
        }
        for feature in measurement_definitions
        if isinstance(feature.get("harmonization_fallback"), Mapping)
    ]
    review_convergence = frozen_review_convergence or {
        "schema_version": REVIEW_CONVERGENCE_SCHEMA_VERSION,
        "status": "converged" if review_converged else "maximum_rounds_reached",
        "converged": review_converged,
        "review_rounds": review_rounds,
        "maximum_review_rounds": maximum_review_rounds,
        "continued_with_latest_ontology": not review_converged,
        "history": review_history,
    }
    if frozen_review_convergence is not None:
        review_convergence = {
            **review_convergence,
            "reused_frozen_preselection_snapshot": True,
        }
    _write_json(output_dir / "ontology_supervision" / "convergence.json", review_convergence)
    from .stage2_effect_estimators import effect_model_family

    modifier_selection = statistical_policy.multi_model.modifier_selection
    count_report = selection_report.get("modifier_count_selection")
    if count_report is None and modifier_selection.rule == "orthogonal_tests" and (
        modifier_selection.resolved_estimator_policy != "dina" or outcome_type == "binary"
    ):
        # Without candidates no selection runs; the configured policy still applies.
        final_estimator = modifier_selection.resolved_estimator_policy
    else:
        final_estimator = (count_report or {}).get("chosen_estimator", "causal_forest")
    _write_json(
        output_dir / "final_definitions.json",
        {
            "schema_version": STAGE2_ROLE_SELECTION_SCHEMA_VERSION,
            "features": selected,
            "measurement_dependencies": measurement_definitions,
            "heldout_measurement_scope": "complete_nuisance_candidate_catalog",
            "heldout_measurement_definitions": [
                str(feature["feature_id"]) for feature in heldout_measurement_definitions
            ],
            "selected_latent_states_artifact": str(latent_states_path),
            "all_candidate_features": len(current),
            "active_candidates_after_consolidation": int(
                selection_report.get("candidate_counts", {}).get(
                    "after_consolidation", len(current)
                )
            ),
            "selection_consolidation_artifact": str(
                selection_dir / "candidate_consolidation" / "report.json"
            ),
            "review_rounds": review_rounds,
            "evaluation_rounds": review_rounds,
            "review_converged": review_converged,
            "review_convergence": review_convergence,
            "ontology_refinement_rounds": ontology_refinement_rounds,
            "selection_artifact": str(selection_dir / "elastic_net_selection.json"),
            "candidate_missingness_filter": missingness_report,
            **({"estimand_ontology": estimand_report} if estimand_policy.enabled else {}),
            "screening_model_family": "multi_model_evidence_with_predictive_nuisance_selection",
            "final_model_family": effect_model_family(final_estimator, outcome_type),
            "final_estimator": final_estimator,
            "harmonization_validation_fallbacks": harmonization_validation_fallbacks,
        },
    )
    combined = pd.concat([fit_selected, heldout_extraction], ignore_index=True).sort_values(
        "_oci_row_id"
    )
    _write_frame(output_dir / "extraction" / "extracted_features.csv", combined)
    diagnostics = estimate_outer_fold(
        dataset=dataset,
        extracted_fit=fit_selected,
        extracted_heldout=heldout_extraction,
        definitions=selected,
        split=split,
        unit_id_column=unit_id_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        inner_folds=inner_folds,
        seed=seed,
        propensity_clip=float(config.propensity_clip),
        min_propensity=getattr(config, "min_propensity", None),
        max_propensity=getattr(config, "max_propensity", None),
        estimation_trees=int(config.estimation_trees),
        estimator=final_estimator,
        statistical_policy=statistical_policy,
        output_dir=output_dir / "estimation",
        nuisance_candidate_definitions=candidate_definitions,
        nuisance_candidate_fit=modeling_fit_frame,
        nuisance_candidate_heldout=heldout_candidates,
        # Inner-fold nuisance models stored by the modifier tests, if any.
        nuisance_model_store=selection_dir / "nuisance_models",
        nuisance_learner=_config_nuisance_learner(config),
    )
    # Reporting only, after this fold's predictions are frozen: benchmark refits
    # for the unmeasured-confounding sensitivity analysis (training-row fits,
    # scored on this fold's held-out rows). No selection or estimation reads
    # them, and a failure is recorded in the benchmark file, not raised.
    from .stage2_sensitivity import benchmark_fold

    benchmark_fold(
        config=config,
        estimation_dir=output_dir / "estimation",
        dataset=dataset,
        candidate_definitions=candidate_definitions,
        candidate_fit=modeling_fit_frame,
        candidate_heldout=heldout_candidates,
        selected_definitions=selected,
        split=split,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        seed=seed,
        propensity_clip=float(config.propensity_clip),
    )
    # Held-out treatment labels are used here only for a descriptive
    # differential-measurement report, after effect predictions are frozen.
    heldout_failure_report = extraction_failures.failure_rate_report(
        heldout_failure_mask,
        heldout_measurement_definitions,
        scope="outer_heldout",
        treatment=dataset.iloc[heldout_ids][treatment_column].to_numpy(dtype=float),
    )
    flagged_names = {
        str(feature["name"])
        for feature in candidate_definitions
        if extraction_failures.has_failure_indicator(feature)
    }
    heldout_failure_report.update(
        failure_source=heldout_failure_source,
        encoded_as_missing_without_training_indicator=sorted(
            record["name"]
            for record in heldout_failure_report["per_feature"]
            if record["failed_rows"] and record["name"] not in flagged_names
        ),
        reported_after_effect_predictions_were_frozen=True,
    )
    _write_json(output_dir / "extraction" / "failure_report_heldout.json", heldout_failure_report)
    # Same reporting rule as the failure report: held-out treatment labels are
    # read only after the effect predictions are frozen.
    heldout_unknown_report = documented_unknown.documented_unknown_report(
        heldout_candidates, candidate_definitions, scope="outer_heldout",
        treatment=dataset.iloc[heldout_ids][treatment_column].to_numpy(dtype=float),
    )
    unknown_tracked = {
        str(feature["name"])
        for feature in candidate_definitions
        if bool(documented_unknown_mask(
            modeling_fit_frame[str(feature["name"])]
            if str(feature["name"]) in modeling_fit_frame
            else pd.Series(dtype=object)
        ).any())
    }
    heldout_unknown_report.update(
        encoded_as_missing_without_training_occurrences=sorted(
            record["name"]
            for record in heldout_unknown_report["per_feature"]
            if record["documented_unknown_rows"] and record["name"] not in unknown_tracked
        ),
        reported_after_effect_predictions_were_frozen=True,
    )
    _write_json(
        output_dir / "extraction" / "documented_unknown_report_heldout.json",
        heldout_unknown_report,
    )
    extraction_failure_summary = {
        "description": extraction_failures.TERMINAL_FAILURE_POLICY,
        "training": missingness_report["training_extraction_failure_report"],
        "training_mask_source": missingness_report.get("extraction_failure_mask_source"),
        "heldout": heldout_failure_report,
        "training_mask_path": str(output_dir / "extraction" / "failure_mask_fit.csv"),
        "heldout_mask_path": str(output_dir / "extraction" / "failure_mask_heldout.csv"),
        "features_with_grouped_failure_indicator": sorted(flagged_names),
        "documented_unknown": {
            "description": documented_unknown.DOCUMENTED_UNKNOWN_POLICY,
            "training": missingness_report.get("training_documented_unknown_report"),
            "heldout": heldout_unknown_report,
            "features_with_grouped_documented_unknown_indicator": sorted(unknown_tracked),
        },
    }
    diagnostics["extraction_failures"] = extraction_failure_summary
    _write_json(output_dir / "estimation" / "extraction_failure_diagnostics.json",
                extraction_failure_summary)
    return {
        "features": selected,
        "review_rounds": review_rounds,
        "evaluation_rounds": review_rounds,
        "review_converged": review_converged,
        "review_convergence": review_convergence,
        "ontology_refinement_rounds": ontology_refinement_rounds,
        "screening_model_family": "multi_model_evidence_with_predictive_nuisance_selection",
        "selection": selection_report,
        "candidate_missingness_filter": missingness_report,
        "extraction_failures": extraction_failure_summary,
        "measurement_dependencies": measurement_definitions,
        "harmonization_validation_fallbacks": harmonization_validation_fallbacks,
        **({"estimand_ontology": estimand_report} if estimand_policy.enabled else {}),
        "estimation": diagnostics,
    }


__all__ = [
    "HELDOUT_MEASUREMENT_CACHE_SCHEMA_VERSION",
    "HELDOUT_MEASUREMENT_REUSE_SCHEMA_VERSION",
    "PRESELECTION_SNAPSHOT_SCHEMA_VERSION",
    "Stage2InfrastructureError",
    "Stage2RequestExhaustedError",
    "Stage2ResponseValidationError",
    "frozen_measurement_definition_identity",
    "frozen_preselection_review_policy",
    "run_fold_analysis",
]
