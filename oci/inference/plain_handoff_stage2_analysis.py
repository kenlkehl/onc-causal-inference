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
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence

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

from ..models.causal_forest_head import CausalForestHead
from ..models.elastic_net_nuisance import (
    ElasticNetLogisticClassifier,
    ElasticNetRegressor,
)
from .stage2_elastic_net_selection import (
    SCHEMA_VERSION as ELASTIC_NET_COMPONENT_SCHEMA_VERSION,
    TEMPORAL_SCOPE,
    select_stage2_features_elastic_net,
)
from .stage2_role_adjudication import (
    SCHEMA_VERSION as ROLE_ADJUDICATION_COMPONENT_SCHEMA_VERSION,
    adjudicate_stage2_roles,
)
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

EXTRACTION_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_single_patient_extraction_v6_conflict_resolution_independent_small_model"
)
EXTRACTION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_single_patient_feature_batch_extraction_v5_conflict_resolution_independent_small_model"
)
PAGE_EXTRACTION_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_single_patient_page_observations_v5_provenance"
)
PAGE_OBSERVATION_FEATURE_BATCH_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_single_patient_page_observation_feature_batch_v1_provenance"
)
PAGE_RECONCILIATION_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_deterministic_page_reconciliation_v6_provenance"
)
REVIEW_CHECKPOINT_SCHEMA_VERSION = "stage2_aggregate_ontology_supervisor_v1"
REVIEW_CONVERGENCE_SCHEMA_VERSION = "stage2_ontology_supervisor_convergence_v1"
ESTIMATION_CHECKPOINT_SCHEMA_VERSION = "stage2_outer_estimation_v8_calibration_overlap"
STAGE2_ROLE_SELECTION_SCHEMA_VERSION = SELECTION_SCHEMA_VERSION
PRESELECTION_SNAPSHOT_SCHEMA_VERSION = "stage2_frozen_preselection_snapshot_v1"
HELDOUT_MEASUREMENT_CACHE_SCHEMA_VERSION = (
    "stage2_frozen_heldout_measurement_cache_v1"
)
HELDOUT_MEASUREMENT_REUSE_SCHEMA_VERSION = (
    "stage2_heldout_measurement_reuse_v1"
)
STAGE2_RESELECTION_MIGRATION_SCHEMA_VERSION = "stage2_reselection_migration_v1"
EXTRACTION_ISSUE_SCHEMA_VERSION = "stage2_extraction_issues_v1"
PENDING_CATEGORY_ONTOLOGY_SCHEMA_VERSION = "stage2_pending_category_ontology_v1"
ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_training_failure_ontology_refinement_v2_request_policy"
)
INCREMENTAL_REFINEMENT_EXTRACTION_SCHEMA_VERSION = (
    "stage2_incremental_refinement_extraction_v1_feature_delta"
)
HARMONIZATION_CHECKPOINT_SCHEMA_VERSION = "stage2_mixed_value_harmonization_v1_llm_training_only"
HARMONIZATION_FALLBACK_SCHEMA_VERSION = "stage2_mixed_value_harmonization_fallback_v1"
# Compatibility defaults for Stage 2 config objects created before ontology
# refinement was added.  Keeping this boundary tolerant also protects a
# long-running workflow if an older caller passes a config object directly.
DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS = 3
DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS = 2
DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE = 10
DEFAULT_EXTRACTION_CHUNK_SIZE_TOKENS = 50_000
DEFAULT_EXTRACTION_CONTEXT_WINDOW_TOKENS = 131_072
DEFAULT_EXTRACTION_MAX_TOKENS = 75_000
DEFAULT_EXTRACTION_CONTEXT_MARGIN_TOKENS = 1_024
SERIAL_EXTRACTION_CHUNK_CHECKPOINT_SCHEMA_VERSION = (
    "stage2_serial_patient_feature_chunk_v1_carried_validated_state"
)
SERIAL_EXTRACTION_MANIFEST_SCHEMA_VERSION = (
    "stage2_serial_patient_feature_extraction_v1_lossless_ordered_chunks"
)
MAX_SERIAL_FEATURE_STATE_CHARS = 2_048
DEFAULT_SCREENING_TREES = 200
DEFAULT_MAX_EVALUATION_ROUNDS = 10
DEFAULT_STABILITY_SELECTION_ROUNDS = 3
DEFAULT_STABILITY_SELECTION_FREQUENCY = 2.0 / 3.0
DEFAULT_EFFECT_MODIFIER_NEGATIVE_MARGIN_FRACTION = 0.01
DEFAULT_EFFECT_MODIFIER_NEGATIVE_FOLD_FRACTION = 0.6


class RequestJSON(Protocol):
    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        validate: Callable[[Mapping[str, Any]], dict[str, Any]],
        *,
        request_kind: str = "interpretation",
    ) -> dict[str, Any]: ...


class Stage2InfrastructureError(RuntimeError):
    """A Stage 2 request failed without producing a scientific response."""


class Stage2RequestExhaustedError(Stage2InfrastructureError):
    """A request exhausted its bounded transport or deadline budget."""


class Stage2ResponseValidationError(ValueError):
    """A completed model response remained semantically invalid after repairs."""


class _ExtractionCancelledError(RuntimeError):
    """Cooperatively stop sibling extraction tasks after one task fails."""


_SCALAR_EXTRACTION_RULES = (
    "Return one scalar value or null per feature; never return an object or array.",
    "For a continuous feature, return one JSON number whenever the record supplies the "
    "requested numeric measurement. If the record supplies only a documented categorical "
    "or threshold representation of that same measurement, return that one concise string "
    "instead of discarding it or inventing a number. From a composite such as 147/93, use "
    "only a component explicitly named by the feature; if the definition requests multiple "
    "components, return null rather than a ratio string or aggregate.",
)

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


_INFRASTRUCTURE_FAILURE_MARKERS = (
    "Stage2InfrastructureError",
    "Stage2RequestExhaustedError",
    "Stage 2 logical request deadline",
    "Stage 2 transport exhausted",
)
_INFRASTRUCTURE_AUDIT_FILENAMES = {
    "extraction_failure.json",
    "category_ontology_repair.json",
    "fallback.json",
}


def _records_infrastructure_failure(path: Path) -> bool:
    try:
        rendered = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return any(marker in rendered for marker in _INFRASTRUCTURE_FAILURE_MARKERS)


def _is_archived_audit_path(path: Path, *, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        return False
    for part in parts:
        normalized = part.lower()
        if "archive" in normalized or "backup" in normalized:
            return True
    return False


def infrastructure_failure_audit_paths(root: Path) -> tuple[Path, ...]:
    """Find legacy checkpoints that mislabeled request failure as model output."""

    root = Path(root)
    if not root.is_dir():
        return ()
    paths: list[Path] = []
    for current, directory_names, filenames in os.walk(root):
        directory_names[:] = [
            name
            for name in directory_names
            if "archive" not in name.lower() and "backup" not in name.lower()
        ]
        current_path = Path(current)
        candidates = set(filenames).intersection(_INFRASTRUCTURE_AUDIT_FILENAMES)
        if (
            "result.json" in filenames
            and current_path.name.startswith("feature_")
            and current_path.parent.name == "supervisor"
        ):
            candidates.add("result.json")
        for name in candidates:
            path = current_path / name
            if (
                not _is_archived_audit_path(path, root=root)
                and _records_infrastructure_failure(path)
            ):
                paths.append(path)
    return tuple(sorted(paths))


def _infrastructure_affected_directories(root: Path) -> set[Path]:
    root = Path(root)
    affected: set[Path] = set()
    for audit_path in infrastructure_failure_audit_paths(root):
        directory = audit_path.parent
        while directory == root or root in directory.parents:
            affected.add(directory)
            if directory == root:
                break
            directory = directory.parent
    return affected


def _supersede_infrastructure_checkpoint(directory: Path) -> None:
    """Retain a legacy bad leaf for audit while making it ineligible for reuse."""

    for name in (
        "complete.json",
        "result.json",
        "extraction_failure.json",
        "extraction_issues.json",
        "category_ontology_repair.json",
        "fallback.json",
    ):
        path = directory / name
        if path.is_file():
            os.replace(
                path,
                path.with_name(f"superseded_infrastructure_{name}"),
            )


def _ontology_refinement_limits(config: Any) -> tuple[int, int]:
    """Read refinement limits from current or pre-refinement config objects."""

    missing = [
        name
        for name in (
            "ontology_refinement_min_failure_patients",
            "max_ontology_refinement_rounds",
        )
        if not hasattr(config, name)
    ]
    if missing:
        LOGGER.warning(
            "Stage 2 received a pre-ontology-refinement config without %s; "
            "using compatibility defaults",
            ", ".join(missing),
        )
    return (
        int(
            getattr(
                config,
                "ontology_refinement_min_failure_patients",
                DEFAULT_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS,
            )
        ),
        int(
            getattr(
                config,
                "max_ontology_refinement_rounds",
                DEFAULT_MAX_ONTOLOGY_REFINEMENT_ROUNDS,
            )
        ),
    )


def _configured_extraction_feature_batch_size(config: Any) -> int:
    """Read the feature prompt cap from current or pre-batching configs."""

    if not hasattr(config, "extraction_feature_batch_size"):
        LOGGER.warning(
            "Stage 2 received a pre-feature-batching config; using the default "
            "extraction feature batch size of %s",
            DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE,
        )
    return int(
        getattr(
            config,
            "extraction_feature_batch_size",
            DEFAULT_EXTRACTION_FEATURE_BATCH_SIZE,
        )
    )


def _configured_serial_extraction(config: Any) -> dict[str, int]:
    """Read token-window settings from current or pre-serial configs."""

    defaults = {
        "chunk_size_tokens": DEFAULT_EXTRACTION_CHUNK_SIZE_TOKENS,
        "context_window_tokens": DEFAULT_EXTRACTION_CONTEXT_WINDOW_TOKENS,
        "max_output_tokens": DEFAULT_EXTRACTION_MAX_TOKENS,
        "context_margin_tokens": DEFAULT_EXTRACTION_CONTEXT_MARGIN_TOKENS,
        "deferred_retry_passes": 1,
    }
    names = {
        "chunk_size_tokens": "extraction_chunk_size_tokens",
        "context_window_tokens": "extraction_context_window_tokens",
        "max_output_tokens": "extraction_max_tokens",
        "context_margin_tokens": "extraction_context_margin_tokens",
        "deferred_retry_passes": "extraction_deferred_retry_passes",
    }
    settings = {
        key: int(getattr(config, attribute, defaults[key]))
        for key, attribute in names.items()
    }
    # Reserve room for a reasoning-enabled repair of the same lossless chunk.
    settings["max_output_tokens"] = max(
        settings["max_output_tokens"],
        int(getattr(config, "extraction_reasoning_max_tokens", None) or 0),
    )
    return settings


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

    minimum_failure_patients, maximum_refinement_rounds = (
        _ontology_refinement_limits(config)
    )
    return {
        "review_schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
        "review_convergence_schema_version": REVIEW_CONVERGENCE_SCHEMA_VERSION,
        "ontology_refinement_schema_version": (
            ONTOLOGY_REFINEMENT_CHECKPOINT_SCHEMA_VERSION
        ),
        "harmonization_schema_version": HARMONIZATION_CHECKPOINT_SCHEMA_VERSION,
        "max_review_rounds": int(getattr(config, "max_review_rounds", 1)),
        "ontology_refinement_min_failure_patients": int(
            minimum_failure_patients
        ),
        "max_ontology_refinement_rounds": int(maximum_refinement_rounds),
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
) -> list[str]:
    """Normalize and validate one extraction-ready closed ontology.

    This is deliberately domain-agnostic.  It validates the shape of the
    ontology, not which clinical labels should appear in it.
    """

    normalized_type = str(value_type or "ambiguous").strip().lower()
    if normalized_type not in {"binary", "categorical", "ordinal"}:
        raise ValueError(f"{source} is not a closed-ontology value type")
    categories = _normalized_category_values(
        value_type=normalized_type,
        values=values,
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
    return _normalized_category_values(
        value_type=definition.get("value_type"),
        values=definition.get("categories_or_unit") or [],
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
            f"{allowed} or null{suffix}"
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
    error_type: type[_ExtractionCategoryError] | type[_ExtractionValueError],
) -> _ExtractionCategoryError | _ExtractionValueError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, error_type):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
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


def _category_ontology_prompt(items: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Ask the configured Stage 2 LLM to normalize values without patient text."""

    body = {
        "job": "map_extracted_values_to_declared_category_ontology",
        "rules": [
            "Use only the feature definition, allowed categories, and prior extracted value.",
            "Do not perform clinical extraction and do not infer any new patient information.",
            "Map by semantic equivalence to exactly one allowed category.",
            "Return null when the prior value does not map unambiguously.",
            "Return every mapping_id exactly once and no additional mapping IDs.",
        ],
        "items": [dict(item) for item in items],
        "response": {
            "corrections": [
                {
                    "mapping_id": "one supplied mapping_id",
                    "value": "one exact allowed category or null",
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You normalize previously extracted categorical values to a closed ontology. "
                "Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _validate_category_ontology(
    value: Mapping[str, Any],
    *,
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
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
        if _is_missing_scalar(extracted) and _canonical_category(extracted, declared) is None:
            corrections[mapping_id] = None
            continue
        canonical = _canonical_category(extracted, declared)
        if canonical is None:
            allowed = json.dumps(declared, ensure_ascii=False, separators=(",", ":"))
            raise ValueError(
                f"category mapping {mapping_id!r} returned {extracted!r}; "
                f"allowed values are {allowed} or null"
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
    validate_response: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract rows, then recover closed-category failures without resending notes."""

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
            return validate_response(value)
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
            validated_response = request_json(
                messages,
                validate_candidate,
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
        _ExtractionCategoryError,
        _ExtractionValueError,
    ) as exc:
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
        corrections = request_json(
            _category_ontology_prompt(items),
            lambda value: _validate_category_ontology(value, items=items),
            # This is ontology judgment over aggregate invalid values, not raw
            # patient extraction, so it belongs to the primary supervisor.
            request_kind="interpretation",
        )
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


def _legacy_extraction_issue_audit(directory: Path) -> dict[str, Any]:
    """Reconstruct the issue ledger from pre-ledger extraction audit files."""

    events: list[dict[str, Any]] = []

    def read_mapping(filename: str) -> Mapping[str, Any] | None:
        path = directory / filename
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, Mapping) else None

    value_audit = read_mapping("invalid_feature_value_repair.json")
    if value_audit is not None:
        for issue in value_audit.get("issues") or []:
            if not isinstance(issue, Mapping):
                continue
            try:
                row_id = int(issue["row_id"])
            except (KeyError, TypeError, ValueError):
                continue
            events.append(
                {
                    "failure_kind": "invalid_scalar_or_value_type",
                    "row_id": row_id,
                    "feature_name": str(issue.get("feature_name") or ""),
                    "reason": str(issue.get("reason") or ""),
                    "prior_extracted_value": copy.deepcopy(issue.get("prior_extracted_value")),
                }
            )

    category_audit = read_mapping("category_ontology_repair.json")
    if category_audit is not None:
        items_by_id = {
            str(item.get("mapping_id") or ""): item
            for item in category_audit.get("items") or []
            if isinstance(item, Mapping) and str(item.get("mapping_id") or "")
        }
        targets_by_id = category_audit.get("targets") or {}
        if isinstance(targets_by_id, Mapping):
            for mapping_id, raw_targets in targets_by_id.items():
                item = items_by_id.get(str(mapping_id))
                if item is None or not isinstance(raw_targets, list):
                    continue
                for target in raw_targets:
                    if not isinstance(target, Mapping):
                        continue
                    try:
                        row_id = int(target["row_id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    events.append(
                        {
                            "failure_kind": "out_of_ontology_category",
                            "row_id": row_id,
                            "feature_name": str(
                                target.get("feature_name") or item.get("feature_name") or ""
                            ),
                            "reason": ("extracted value is outside the declared closed ontology"),
                            "prior_extracted_value": copy.deepcopy(
                                item.get("prior_extracted_value")
                            ),
                            "allowed_categories": list(item.get("allowed_categories") or []),
                        }
                    )

    structural_audit = read_mapping("extraction_failure.json")
    if structural_audit is not None:
        for raw_row_id in structural_audit.get("row_ids") or []:
            try:
                row_id = int(raw_row_id)
            except (TypeError, ValueError):
                continue
            events.append(
                {
                    "failure_kind": "structural_response_failure",
                    "row_id": row_id,
                    "feature_name": None,
                    "reason": str(structural_audit.get("validation_error") or ""),
                }
            )

    return {
        "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
        "completed_at": _now(),
        "reconstructed_from_legacy_audits": True,
        "events": events,
    }


def _ensure_extraction_issue_audit(directory: Path) -> dict[str, Any]:
    """Return an issue ledger, reconstructing it for a compatible old checkpoint."""

    path = directory / "extraction_issues.json"
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping) and isinstance(value.get("events"), list):
                return dict(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    reconstructed = _legacy_extraction_issue_audit(directory)
    _write_json(path, reconstructed)
    return reconstructed


def _extraction_prompt(
    *,
    definitions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if len(rows) != 1:
        raise ValueError("Stage 2 extraction prompts must contain exactly one patient's record")
    body = {
        "job": "extract_stage2_patient_variables",
        "rules": [
            "Use only the supplied clinical text for the patient in that row.",
            "Apply the measurement definition and missing-value rule literally.",
            "Consider every explicitly supported observation for a feature before "
            "selecting its one output value.",
            "When multiple supported observations remain, apply that feature's "
            "conflict_resolution policy literally. The conflict_resolution policy "
            "governs if prose in the measurement definition is ambiguous or inconsistent "
            "about how to choose among observations.",
            "For latest or earliest conflict resolution, prefer observations with an "
            "explicit governing date or time. If none are dated, use clinical-text source "
            "order and the declared source_order_tie_breaker; do not treat the first mention, "
            "diagnosis value, or demographics value as automatically authoritative.",
            "For a binary, categorical, or ordinal feature, return one declared category exactly.",
            "Do not substitute 0/1 or true/false for a declared category unless that "
            "exact value is declared.",
            *_SCALAR_EXTRACTION_RULES,
            "Return null when the record does not support a value.",
            "Return every row and every feature exactly once.",
        ],
        "features": _prompt_feature_definitions(definitions),
        "patients": list(rows),
        "response": {
            "rows": [
                {
                    "row_id": "one supplied integer row_id",
                    "values": {"every supplied feature name": "scalar value or null"},
                }
            ]
        },
    }
    # Keep clinical text as Unicode. ASCII escaping can expand multilingual
    # notes several-fold and consumes model tokens on literal ``\\uXXXX``
    # sequences without adding information.
    return [
        {
            "role": "system",
            "content": "You extract prespecified variables from supplied clinical text. Return JSON only.",
        },
        {
            "role": "user",
            "content": json.dumps(body, sort_keys=True, ensure_ascii=False),
        },
    ]


def _serial_extraction_prompt(
    *,
    definitions: Sequence[Mapping[str, Any]],
    row_id: int,
    chunk_text: str,
    prior_values: Mapping[str, Any],
    prior_feature_state: Mapping[str, Any],
    chunk_index: int,
    char_start: int,
    char_end: int,
    document_chars: int,
) -> list[dict[str, str]]:
    """Update one validated cumulative extraction with the next source chunk."""

    body = {
        "job": "update_stage2_patient_variables_serially",
        "rules": [
            "Process this clinical-text chunk after every earlier chunk and before every later chunk.",
            "prior_extraction contains the validated cumulative scalar values from all earlier contiguous chunks; it is state, not additional clinical text.",
            "prior_feature_state contains concise decision metadata retained from earlier chunks, such as the governing date and source order for latest/earliest or value counts for mode.",
            "Use only prior_extraction and the supplied current_chunk. Never infer evidence from a feature description or from chunk metadata.",
            "For each feature, combine supported evidence in the current chunk with the prior cumulative value and apply that feature's conflict_resolution policy literally.",
            "Preserve a nonnull prior value exactly when this chunk supplies no evidence that changes the policy-selected cumulative value.",
            "A null prior value means no supported cumulative value has been retained yet; it is not evidence of a negative clinical finding.",
            "For latest or earliest, compare explicit governing dates when available. Otherwise treat current_chunk as later in source order than prior_extraction and apply source_order_tie_breaker.",
            "For maximum, minimum, mode, any_positive, and single_or_null, update the cumulative value according to the named policy rather than automatically preferring the current chunk.",
            "Return carry_forward_state for every feature as a concise string or null. Preserve enough metadata to apply its conflict policy in later chunks, but do not quote or summarize unrelated record text.",
            f"Each carry_forward_state string must be at most {MAX_SERIAL_FEATURE_STATE_CHARS} characters.",
            "For a binary, categorical, or ordinal feature, return one declared category exactly.",
            "Do not substitute 0/1 or true/false for a declared category unless that exact value is declared.",
            *_SCALAR_EXTRACTION_RULES,
            "Return null only when the combined prior state and current chunk do not support a retained value under the feature policy.",
            "Return the row and every supplied feature exactly once.",
        ],
        "features": _prompt_feature_definitions(definitions),
        "patient": {
            "row_id": int(row_id),
            "prior_extraction": dict(prior_values),
            "prior_feature_state": dict(prior_feature_state),
            "current_chunk": chunk_text,
            "chunk": {
                "chunk_index": int(chunk_index),
                "char_start": int(char_start),
                "char_end": int(char_end),
                "document_chars": int(document_chars),
                "is_final_chunk": int(char_end) == int(document_chars),
            },
        },
        "response": {
            "rows": [
                {
                    "row_id": "the supplied integer row_id",
                    "values": {"every supplied feature name": "cumulative scalar value or null"},
                    "carry_forward_state": {
                        "every supplied feature name": "concise policy state string or null"
                    },
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You update a validated structured patient extraction from consecutive "
                "clinical-record chunks. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True, ensure_ascii=False)},
    ]


def _page_extraction_prompt(
    *,
    definitions: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Request every supported page observation with verifiable provenance."""

    body = {
        "job": "extract_stage2_patient_variable_observations",
        "rules": [
            "Use only the supplied clinical-text page for this patient.",
            "Return every distinct explicitly supported observation for every supplied feature; do not collapse conflicting or repeated longitudinal values.",
            "conflict_resolution is applied later by deterministic code. Do not apply it within this page and do not discard an otherwise supported observation because another value is newer, earlier, larger, smaller, or more frequent.",
            "Do not return an observation when the page does not support a nonmissing value.",
            "Each value must be one scalar. For closed ontologies, use one declared category exactly.",
            "For each observation, quote a short exact contiguous evidence substring from patient.text.",
            "evidence_start and evidence_end are zero-based Python-style character offsets into patient.text, with evidence_end exclusive.",
            "Set recorded_at only when a date or time explicitly governs that observation, such as an encounter, specimen, measurement, or result date.",
            "When recorded_at is set, normalize it to ISO-8601 and provide the exact source date text plus its offsets. Do not borrow an unrelated date.",
            "Use null for recorded_at and recorded_at_evidence when no governing date is explicit on this page.",
            "Return an empty observations array when no supplied feature has supported evidence on this page.",
        ],
        "features": _page_prompt_feature_definitions(definitions),
        "patient": dict(row),
        "response": {
            "rows": [
                {
                    "row_id": "the supplied integer row_id",
                    "observations": [
                        {
                            "feature_name": "one supplied feature name",
                            "value": "one supported scalar value",
                            "evidence": "exact quote from patient.text",
                            "evidence_start": "zero-based inclusive integer",
                            "evidence_end": "zero-based exclusive integer",
                            "recorded_at": "ISO-8601 date/time, year-month, year, or null",
                            "recorded_at_evidence": "exact source date/time quote or null",
                            "recorded_at_start": "inclusive integer or null",
                            "recorded_at_end": "exclusive integer or null",
                        }
                    ],
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You extract all supported clinical variable observations with exact "
                "source provenance. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True, ensure_ascii=False)},
    ]


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


def _validate_extraction(
    value: Mapping[str, Any],
    *,
    row_ids: Sequence[int],
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
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

    validated = _validate_extraction(
        value,
        row_ids=[row_id],
        definitions=definitions,
    )
    feature_names = [str(definition["name"]) for definition in definitions]
    row = validated["rows"][0]
    raw_state = row.get("carry_forward_state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("serial extraction row requires a carry_forward_state object")
    if set(map(str, raw_state)) != set(feature_names):
        raise ValueError(
            "serial extraction carry_forward_state must contain every supplied feature exactly"
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
            raise ValueError(
                f"serial carry_forward_state for {name!r} exceeds "
                f"{MAX_SERIAL_FEATURE_STATE_CHARS} characters"
            )
        state[name] = rendered or None
    row["carry_forward_state"] = state
    return validated


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


def _exact_quote_span(
    *,
    text: str,
    quote: Any,
    reported_start: Any,
    reported_end: Any,
    label: str,
) -> tuple[str, int, int, str]:
    """Resolve a model-provided exact quote to deterministic page offsets."""

    if not isinstance(quote, str) or not quote:
        raise ValueError(f"{label} must be a nonempty exact quote")
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
        raise ValueError(f"{label} is not an exact substring of the supplied page")
    if start is None:
        if len(matches) != 1:
            raise ValueError(
                f"{label} occurs more than once; exact offsets are required to prove provenance"
            )
        selected = matches[0]
        method = "unique_exact_match"
    else:
        ranked = sorted(matches, key=lambda candidate: (abs(candidate - start), candidate))
        if len(ranked) > 1 and abs(ranked[0] - start) == abs(ranked[1] - start):
            raise ValueError(
                f"{label} offsets are equidistant from repeated exact quotes; "
                "return the exact occurrence offsets"
            )
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


def _canonical_time_evidence(value: str) -> str:
    """Normalize an exact source date quote locally instead of trusting the model."""

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
    """Validate values and prove each observation against an exact page quote."""

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
    normalized: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for observation_index, raw in enumerate(raw_observations, start=1):
        feature_name = ""
        try:
            if not isinstance(raw, Mapping):
                raise ValueError("observation must be an object")
            feature_name = str(raw.get("feature_name") or "")
            if feature_name not in definition_by_name:
                raise ValueError("feature_name is not one of the supplied features")
            scalar_response = _validate_extraction(
                {
                    "rows": [
                        {
                            "row_id": row_id,
                            "values": {feature_name: raw.get("value")},
                        }
                    ]
                },
                row_ids=[row_id],
                definitions=[definition_by_name[feature_name]],
            )
            scalar = scalar_response["rows"][0]["values"][feature_name]
            if scalar is None:
                raise ValueError("a page observation must contain a supported nonmissing value")

            evidence, evidence_start, evidence_end, evidence_offset_resolution = (
                _exact_quote_span(
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
                ) = _exact_quote_span(
                    text=text,
                    quote=raw.get("recorded_at_evidence"),
                    reported_start=raw.get("recorded_at_start"),
                    reported_end=raw.get("recorded_at_end"),
                    label="recorded_at_evidence",
                )
                source_recorded_at = _canonical_time_evidence(recorded_at_evidence)
                if recorded_at != source_recorded_at:
                    raise ValueError(
                        "recorded_at does not match its exact recorded_at_evidence quote"
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

            identity_value = {
                "row_id": row_id,
                "feature_name": feature_name,
                "value": scalar,
                "source_start": page_char_start + evidence_start,
                "source_end": page_char_start + evidence_end,
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
                    "source_start": page_char_start + evidence_start,
                    "source_end": page_char_start + evidence_end,
                    "page_index": page_index,
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
                    key=lambda observation: (
                        int(observation["source_start"]),
                        str(observation["feature_name"]),
                        str(observation["observation_id"]),
                    ),
                ),
            }
        ]
    }
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
    """Request page observations, retaining valid provenance after exhausted repairs."""

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
                lambda candidate: _validate_page_observations(
                    candidate,
                    page=page,
                    definitions=definitions,
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
                    "failure_kind": "invalid_page_observation_provenance",
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
                    "resolution": "retain_valid_drop_unverifiable",
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
                "unverifiable observation(s) for row %s",
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
                    int(item[0]["source_end"]),
                    str(item[0]["observation_id"]),
                ),
            )
        else:
            selected, _timestamp = min(
                dated,
                key=lambda item: (
                    int(item[1]),
                    int(item[0]["source_start"]),
                    str(item[0]["observation_id"]),
                ),
            )
        return dict(selected), "verified_recorded_at"
    selected = (max if latest else min)(
        observations,
        key=lambda observation: (
            int(observation["source_end"] if latest else observation["source_start"]),
            str(observation["observation_id"]),
        ),
    )
    return dict(selected), "absolute_source_order"


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
            int(observation["source_start"]),
            int(observation["source_end"]),
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


def _partition_rows_for_prompt(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_prompt_chars: int,
    definition_batches: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[list[Mapping[str, Any]]], list[Mapping[str, Any]]]:
    """Create only singleton extraction requests, measured at exact prompt size.

    Rows that cannot fit even by themselves are returned separately for
    lossless page planning.  This avoids the old oversized-singleton hole in
    the approximate JSON-size partitioner.  Multi-patient prompts are forbidden
    by construction as well as by ``_extraction_prompt``.
    """

    batches: list[list[Mapping[str, Any]]] = []
    oversized: list[Mapping[str, Any]] = []
    for row in rows:
        prompt_sizes = (
            _prompt_chars(
                _extraction_prompt(
                    definitions=batch_definitions,
                    rows=[row],
                )
            )
            for batch_definitions in definition_batches
        )
        if any(size > int(max_prompt_chars) for size in prompt_sizes):
            oversized.append(row)
        else:
            batches.append([row])
    return batches, oversized


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
) -> list[dict[str, Any]]:
    """Split one note into the largest exact prompt-sized contiguous pages."""

    source = str(row.get("text") or "")
    row_id = int(row["row_id"])
    if not source:
        raise ValueError(
            "an empty Stage 2 row exceeded the prompt budget before note text was added; "
            "increase stage2.extraction_max_prompt_chars or shorten the feature definitions"
        )
    pages: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(source):
        low = cursor + 1
        high = len(source)
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
            prompt_sizes = (
                _prompt_chars(
                    _page_extraction_prompt(
                        definitions=batch_definitions,
                        row=candidate,
                    )
                )
                for batch_definitions in definition_batches
            )
            if all(size <= int(max_prompt_chars) for size in prompt_sizes):
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
            "source_tokens": _text_token_count(tokenizer, chunk_text),
            "prompt_tokens": prompt_token_count(tokenizer, messages),
            "prompt_chars": _prompt_chars(messages),
            "messages": messages,
        }

    def fits(value: Mapping[str, Any]) -> bool:
        return (
            int(value["source_tokens"]) <= int(chunk_size_tokens)
            and int(value["prompt_tokens"]) <= int(input_token_budget)
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
        if any(
            _records_infrastructure_failure(chunk_dir / name)
            for name in (
                "extraction_failure.json",
                "category_ontology_repair.json",
                "fallback.json",
            )
        ):
            _supersede_infrastructure_checkpoint(chunk_dir)
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
                "source_tokens": int(planned["source_tokens"]),
                "prompt_tokens": int(planned["prompt_tokens"]),
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
                    _ensure_extraction_issue_audit(chunk_dir)
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
                validate_response=lambda value: _validate_serial_extraction(
                    value,
                    row_id=row_id,
                    definitions=definitions,
                ),
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
                    "source_tokens": int(planned["source_tokens"]),
                    "prompt_tokens": int(planned["prompt_tokens"]),
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
                "source_tokens": int(planned["source_tokens"]),
                "prompt_tokens": int(planned["prompt_tokens"]),
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
    issue_files = sorted(output_dir.rglob("extraction_issues.json"))
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
                continue
            if feature_name not in definition_names:
                continue
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
) -> pd.DataFrame:
    """Extract one patient at a time, serializing long records across token chunks."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if (isinstance(deferred_retry_passes, bool) or not isinstance(deferred_retry_passes, int)
            or deferred_retry_passes < 0):
        raise ValueError("deferred_retry_passes must be a nonnegative integer")
    infrastructure_affected = _infrastructure_affected_directories(output_dir)
    if infrastructure_affected:
        LOGGER.warning(
            "superseding legacy Stage 2 infrastructure-failure checkpoints "
            "directories=%s root=%s",
            len(infrastructure_affected),
            output_dir,
        )
        _supersede_infrastructure_checkpoint(output_dir)
    cancellation = threading.Event()

    def guarded_request_json(
        messages: Sequence[Mapping[str, str]],
        validate: Callable[[Mapping[str, Any]], dict[str, Any]],
        *,
        request_kind: str = "interpretation",
    ) -> dict[str, Any]:
        if cancellation.is_set():
            raise _ExtractionCancelledError(
                "Stage 2 extraction cancelled after a sibling task failed"
            )
        return request_json(
            messages,
            validate,
            request_kind=request_kind,
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

    request_rows = [
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
        batches, oversized_rows = _partition_rows_for_prompt(
            request_rows,
            max_prompt_chars=int(max_prompt_chars),
            definition_batches=definition_batches,
        )

    page_requests: list[dict[str, Any]] = []
    for row in oversized_rows:
        page_requests.extend(
            _lossless_extraction_pages(
                row,
                definition_batches=definition_batches,
                max_prompt_chars=int(max_prompt_chars),
            )
        )

    if len(definition_batches) > 1:
        LOGGER.info(
            "Stage 2 extraction features=%s feature_batch_size=%s "
            "feature_batches_per_patient=%s patients=%s pages=%s",
            len(definitions),
            feature_batch_size,
            len(definition_batches),
            len(batches),
            len(page_requests),
        )

    # Snapshot every saved singleton result before concurrent workers begin
    # rewriting the newly numbered batch directories.  Removing an old
    # multi-patient batch shifts every later singleton's batch number, but its
    # row_id remains a stable identity that lets us safely relocate it.
    singleton_checkpoints: dict[int, list[dict[str, Any]]] = {}
    for saved_dir in sorted((output_dir / "batches").glob("batch_*")):
        if not saved_dir.is_dir() or saved_dir.is_symlink():
            continue
        if saved_dir in infrastructure_affected:
            _supersede_infrastructure_checkpoint(saved_dir)
            continue
        saved_complete_path = saved_dir / "complete.json"
        saved_result_path = saved_dir / "result.json"
        saved_manifest_path = saved_dir / "row_ids.json"
        saved_audit_path = saved_dir / "category_ontology_repair.json"
        saved_issue_path = saved_dir / "extraction_issues.json"
        if not (
            saved_complete_path.is_file()
            and saved_result_path.is_file()
            and saved_manifest_path.is_file()
        ):
            continue
        if _stale_category_ontology_audit(saved_audit_path) is not None:
            continue
        try:
            saved_completion = json.loads(saved_complete_path.read_text(encoding="utf-8"))
            saved_result = json.loads(saved_result_path.read_text(encoding="utf-8"))
            saved_row_ids = json.loads(saved_manifest_path.read_text(encoding="utf-8"))
            saved_audit = (
                json.loads(saved_audit_path.read_text(encoding="utf-8"))
                if saved_audit_path.is_file()
                else None
            )
            saved_issues = (
                json.loads(saved_issue_path.read_text(encoding="utf-8"))
                if saved_issue_path.is_file()
                else _legacy_extraction_issue_audit(saved_dir)
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            not isinstance(saved_completion, Mapping)
            or saved_completion.get("status") != "complete"
            or not isinstance(saved_result, Mapping)
            or not isinstance(saved_row_ids, list)
            or len(saved_row_ids) != 1
        ):
            continue
        try:
            saved_row_id = int(saved_row_ids[0])
        except (TypeError, ValueError):
            continue
        singleton_checkpoints.setdefault(saved_row_id, []).append(
            {
                "source_dir": saved_dir,
                "completion": dict(saved_completion),
                "result": dict(saved_result),
                "category_ontology_audit": saved_audit,
                "extraction_issues": saved_issues,
            }
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
            if feature_dir in infrastructure_affected:
                _supersede_infrastructure_checkpoint(feature_dir)
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
                        _ensure_extraction_issue_audit(feature_dir)
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
                use_serial = tokenizer is not None and _serial_extraction_required(
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
            if feature_dir in infrastructure_affected:
                _supersede_infrastructure_checkpoint(feature_dir)
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
                        _ensure_extraction_issue_audit(feature_dir)
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
        if batch_dir in infrastructure_affected:
            _supersede_infrastructure_checkpoint(batch_dir)
        result_path = batch_dir / "result.json"
        complete_path = batch_dir / "complete.json"
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
        candidates = list(singleton_checkpoints.get(row_ids[0], []))
        candidates.sort(
            key=lambda candidate: (
                (
                    0
                    if (
                        candidate["completion"].get("schema_version")
                        == EXTRACTION_CHECKPOINT_SCHEMA_VERSION
                        and candidate["completion"].get("input_fingerprint") == input_fingerprint
                    )
                    else 1
                ),
                str(candidate["source_dir"]),
            )
        )
        for candidate in candidates:
            completion = candidate["completion"]
            schema_version = completion.get("schema_version")
            if schema_version == EXTRACTION_CHECKPOINT_SCHEMA_VERSION:
                if completion.get("input_fingerprint") != input_fingerprint:
                    continue
            elif schema_version is not None:
                continue
            try:
                validated = _validate_extraction(
                    candidate["result"],
                    row_ids=row_ids,
                    definitions=definitions,
                )
            except (KeyError, TypeError, ValueError):
                continue
            source_dir = Path(candidate["source_dir"])
            legacy = schema_version is None
            relocated = source_dir != batch_dir
            if not legacy and not relocated and stale_audit is None:
                if isinstance(candidate.get("extraction_issues"), Mapping):
                    _write_json(
                        batch_dir / "extraction_issues.json",
                        candidate["extraction_issues"],
                    )
                return list(validated["rows"])

            batch_dir.mkdir(parents=True, exist_ok=True)
            _write_json(batch_dir / "row_ids.json", row_ids)
            _write_json(result_path, validated)
            source_audit = candidate.get("category_ontology_audit")
            if isinstance(source_audit, Mapping):
                _write_json(ontology_audit_path, source_audit)
            elif ontology_audit_path.is_file():
                try:
                    previous_audit = json.loads(ontology_audit_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    previous_audit = None
                _write_json(
                    ontology_audit_path,
                    {
                        "schema_version": "stage2_category_ontology_repair_v1",
                        "resolution": "superseded_by_single_patient_checkpoint_reindex",
                        "superseded_at": _now(),
                        "previous_audit": previous_audit,
                    },
                )
            source_issues = candidate.get("extraction_issues")
            if isinstance(source_issues, Mapping):
                _write_json(batch_dir / "extraction_issues.json", source_issues)
            _write_json(
                complete_path,
                {
                    "status": "complete",
                    "schema_version": EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
                    "input_fingerprint": input_fingerprint,
                    "completed_at": completion.get("completed_at") or _now(),
                    "rows": 1,
                    "adopted_legacy_single_patient_checkpoint": legacy,
                    "relocated_single_patient_checkpoint": relocated,
                    "checkpoint_source": str(source_dir),
                    "adopted_at": _now(),
                },
            )
            LOGGER.info(
                "%s single-patient Stage 2 extraction checkpoint: %s -> %s",
                "relocate" if relocated else "adopt legacy",
                source_dir,
                batch_dir,
            )
            return list(validated["rows"])

        if complete_path.is_file() or result_path.is_file():
            LOGGER.info("rerun incompatible Stage 2 extraction checkpoint: %s", batch_dir)
        if stale_audit is not None:
            LOGGER.info("retry stale Stage 2 category ontology batch: %s", batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        _write_json(batch_dir / "row_ids.json", row_ids)
        if len(definition_batches) == 1:
            use_serial = tokenizer is not None and _serial_extraction_required(
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
        if page_dir in infrastructure_affected:
            _supersede_infrastructure_checkpoint(page_dir)
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
                    _ensure_extraction_issue_audit(page_dir)
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

    completed: list[tuple[int, list[dict[str, Any]]]] = []
    completed_pages: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    tasks = [("batch", index, batch) for index, batch in enumerate(batches, start=1)]
    tasks += [("page", int(page["row_id"]), page) for page in page_requests]
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
                     else executor.submit(run_page, payload)): (kind, index, payload)
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
                        completed_pages.setdefault(index, []).append(result)
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
                active_task[1] if active_task and active_task[0] == "page" else None,
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
                    _ensure_extraction_issue_audit(reconciliation_dir)
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
                "reconciliation_method": "deterministic_provenance",
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
            "pages": len(page_requests),
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
        summary: dict[str, Any] = {
            "feature_id": str(feature["feature_id"]),
            "name": name,
            "rows": len(frame),
            "nonmissing": int(len(nonmissing)),
            "nonmissing_fraction": float(len(nonmissing) / row_count),
            "unique_nonmissing": int(nonmissing.nunique()),
            "dominant_value_fraction": dominant,
            "most_common_values": {str(key): int(count) for key, count in counts.head(8).items()},
        }
        if str(feature.get("value_type")) == "continuous":
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
    nonmissing = series.dropna()
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


def _harmonization_prompt(
    *,
    feature: Mapping[str, Any],
    observations: Mapping[str, Any],
    prior_plan: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    body = {
        "job": "harmonize_stage2_mixed_numeric_and_categorical_values",
        "information_boundary": (
            "The values and summaries come only from outer-training patients. "
            "No treatment, outcome, held-out text, or held-out values are supplied."
        ),
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
        "observed_training_representations": copy.deepcopy(observations),
        "prior_plan_to_extend_or_replace": copy.deepcopy(prior_plan),
        "rules": [
            "Choose one common modeling representation for every observed value.",
            "Use target_representation=continuous only when every nonnumeric token "
            "has an unambiguous exact numeric meaning in the feature's stated unit. "
            "Do not invent midpoints for ranges, inequalities, or qualitative labels.",
            "Otherwise use target_representation=categorical. Define clinically "
            "coherent canonical categories, map every exact observed text token, and "
            "supply ordered, exhaustive, nonoverlapping numeric bins.",
            "For categorical numeric bins, the first lower_bound and final upper_bound "
            "must be null. Adjacent bins must share a boundary with exactly one side inclusive.",
            "Map an unusable text token to null rather than guessing.",
            "This is generic value harmonization. Base the plan only on the supplied "
            "feature definition and observed representations.",
        ],
        "response_schema": {
            "target_representation": "continuous or categorical",
            "reason": "concise scientific rationale",
            "canonical_categories": ["empty for continuous; at least two strings for categorical"],
            "categorical_value_map": [
                {
                    "raw_value": "one exact observed text token",
                    "canonical_value": (
                        "finite number/null for continuous; canonical category/null "
                        "for categorical"
                    ),
                }
            ],
            "numeric_bin_rules": [
                {
                    "lower_bound": "number or null",
                    "lower_inclusive": "boolean",
                    "upper_bound": "number or null",
                    "upper_inclusive": "boolean",
                    "canonical_value": "canonical category",
                }
            ],
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You harmonize mixed representations of one clinical variable into "
                "a loss-aware, machine-readable ontology. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _harmonization_delta_prompt(
    *,
    feature: Mapping[str, Any],
    prior_plan: Mapping[str, Any],
    new_categorical_values: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    body = {
        "job": "extend_stage2_harmonization_map_for_new_text_values",
        "information_boundary": (
            "The values come only from outer-training patients. No treatment, outcome, "
            "held-out text, or held-out values are supplied."
        ),
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
        "frozen_harmonization_plan": {
            key: copy.deepcopy(prior_plan.get(key))
            for key in (
                "target_representation",
                "reason",
                "canonical_categories",
                "numeric_bin_rules",
                "unmapped_value_rule",
            )
        },
        "new_observed_training_text_values": copy.deepcopy(list(new_categorical_values)),
        "rules": [
            "Do not revise the frozen target representation, categories, or numeric bins.",
            "Return exactly one mapping for each supplied raw_value and no other raw values.",
            "Copy every raw_value exactly, including punctuation, spacing, and case.",
            "For a continuous target, use a finite number only when the exact text has "
            "an unambiguous value in the feature's stated unit.",
            "For a categorical target, use only a frozen canonical category.",
            "Map an unusable or ambiguous text token to null rather than guessing.",
        ],
        "response_schema": {
            "categorical_value_map": [
                {
                    "raw_value": "one exact supplied text token",
                    "canonical_value": (
                        "finite number/null for continuous; frozen canonical category/null "
                        "for categorical"
                    ),
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You extend one frozen clinical value map without revising its ontology. "
                "Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


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
            if lower is not None and upper is not None and lower >= upper:
                raise ValueError("numeric bins require lower_bound < upper_bound")
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
    if _records_infrastructure_failure(fallback_path):
        _supersede_infrastructure_checkpoint(output_dir)
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
            new_categorical_values=new_rows,
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
            delta = request_json(
                messages,
                lambda response: _validate_harmonization_delta(
                    response,
                    prior_plan=validated_prior,
                    new_raw_values=new_raw_values,
                ),
                request_kind="interpretation",
            )
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
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    newly_requested: list[str] = []
    fallbacks: list[dict[str, Any]] = []
    mapping_normalizations: list[dict[str, Any]] = []
    for raw_feature in definitions:
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
        updated.append(_normalized_feature_modeling_definition(feature))
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


class _FeatureEncoder:
    def __init__(self, definitions: Sequence[Mapping[str, Any]]) -> None:
        self.definitions = list(definitions)
        self.encodings: list[tuple[str, str, Any]] = []

    def fit(self, frame: pd.DataFrame) -> "_FeatureEncoder":
        self.encodings = []
        for feature in self.definitions:
            name = str(feature["name"])
            modeling_strategy = _feature_modeling_strategy(feature)
            series = frame[name] if name in frame else pd.Series([None] * len(frame))
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
            series = frame[name] if name in frame else pd.Series([None] * len(frame))
            if value_type == "continuous":
                median, scale = parameters
                numeric = pd.to_numeric(series, errors="coerce")
                missing = numeric.isna().to_numpy(dtype=float)
                values = (numeric.fillna(median).to_numpy(dtype=float) - median) / scale
                columns.extend([values, missing])
            elif value_type == "continuous_with_categorical_fallback":
                median, scale, categories = parameters
                numeric = pd.to_numeric(series, errors="coerce")
                raw_missing = series.isna()
                categorical_mask = series.notna() & numeric.isna()
                normalized = series.where(categorical_mask, "").astype(str)
                values = (numeric.fillna(median).to_numpy(dtype=float) - median) / scale
                columns.extend(
                    [
                        values,
                        numeric.notna().to_numpy(dtype=float),
                        raw_missing.to_numpy(dtype=float),
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
                    columns.append((normalized == category).to_numpy(dtype=float))
                if value_type == "categorical_with_other":
                    columns.append((~normalized.isin(parameters)).to_numpy(dtype=float))
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
    """Use persisted treatment/outcome supports with legacy-safe fallbacks."""

    if role not in {"treatment", "outcome"}:
        raise ValueError("nuisance role must be treatment or outcome")
    definitions = list(definitions)
    has_separate_supports = any(
        "nuisance_model_roles" in feature for feature in definitions
    )
    if not has_separate_supports:
        return (
            _definitions_for_roles(definitions, {"confounder"})
            if role == "treatment"
            else definitions
        )
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


def _fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> Any:
    if len(np.unique(y)) < 2 or x.shape[1] == 0:
        return _ConstantClassifier(float(np.mean(y)))
    model: Any = ElasticNetLogisticClassifier(random_state=seed, n_jobs=1)
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
) -> Any:
    if x.shape[1] == 0:
        return _ConstantRegressor(float(np.mean(y)))
    model: Any = ElasticNetRegressor(random_state=seed, n_jobs=1)
    model.fit(x, y)
    return model


@dataclass
class _OutcomeModels:
    control: Any
    treated: Any
    binary: bool


def _fit_outcome_models(
    x: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    binary: bool,
    seed: int,
) -> _OutcomeModels:
    models = []
    for arm in (0, 1):
        mask = treatment.astype(int) == arm
        if not mask.any():
            mask = np.ones(len(treatment), dtype=bool)
        if binary:
            models.append(
                _fit_classifier(
                    x[mask],
                    outcome[mask],
                    seed=seed + arm,
                )
            )
        else:
            models.append(
                _fit_regressor(
                    x[mask],
                    outcome[mask],
                    seed=seed + arm,
                )
            )
    return _OutcomeModels(control=models[0], treated=models[1], binary=binary)


def _predict_outcomes(models: _OutcomeModels, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if models.binary:
        return _predict_probability(models.control, x), _predict_probability(models.treated, x)
    return (
        np.asarray(models.control.predict(x), dtype=float),
        np.asarray(models.treated.predict(x), dtype=float),
    )


@dataclass
class _EffectModel:
    constant: float
    model: Any | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(x), self.constant, dtype=float)
        return np.asarray(self.model.predict(x), dtype=float)


def _fit_effect_model(
    x: np.ndarray,
    pseudo_outcome: np.ndarray,
    *,
    seed: int,
    trees: int | None,
) -> _EffectModel:
    finite = np.isfinite(pseudo_outcome)
    if not finite.any():
        return _EffectModel(constant=0.0)
    target = pseudo_outcome[finite]
    if len(target) >= 20:
        lower, upper = np.quantile(target, [0.01, 0.99])
        target = np.clip(target, lower, upper)
    constant = float(np.mean(target))
    if x.shape[1] == 0 or len(target) < 12:
        return _EffectModel(constant=constant)
    if trees is None:
        from sklearn.linear_model import Ridge

        model: Any = Ridge(alpha=2.0)
    else:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=int(trees),
            min_samples_leaf=max(2, min(20, len(target) // 10)),
            max_features="sqrt",
            n_jobs=1,
            random_state=seed,
        )
    model.fit(x[finite], target)
    return _EffectModel(constant=constant, model=model)


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


def _safe_auc(y: np.ndarray, probability: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, probability))


def _prediction_metrics(
    *,
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    observed_outcome: np.ndarray,
    binary_outcome: bool,
    r_loss: float,
) -> dict[str, Any]:
    from sklearn.metrics import log_loss, mean_squared_error

    e = np.clip(propensity, 1e-6, 1.0 - 1e-6)
    metrics: dict[str, Any] = {
        "treatment_log_loss": float(log_loss(treatment.astype(int), e, labels=[0, 1])),
        "treatment_brier": float(np.mean((treatment - e) ** 2)),
        "treatment_auc": _safe_auc(treatment, e),
        "r_loss": float(r_loss),
    }
    if binary_outcome:
        probability = np.clip(observed_outcome, 1e-6, 1.0 - 1e-6)
        metrics.update(
            {
                "outcome_log_loss": float(
                    log_loss(outcome.astype(int), probability, labels=[0, 1])
                ),
                "outcome_brier": float(np.mean((outcome - probability) ** 2)),
                "outcome_auc": _safe_auc(outcome, probability),
            }
        )
    else:
        rmse = float(math.sqrt(mean_squared_error(outcome, observed_outcome)))
        variance = float(np.var(outcome))
        metrics.update(
            {
                "outcome_rmse": rmse,
                "outcome_r2": (
                    float(1.0 - np.mean((outcome - observed_outcome) ** 2) / variance)
                    if variance > 0
                    else None
                ),
            }
        )
    return metrics


def _metric_improvements(
    baseline: Mapping[str, Any],
    enhanced: Mapping[str, Any],
) -> dict[str, Any]:
    """Orient every validation metric so positive values mean improvement."""

    improvements: dict[str, Any] = {}
    for key in sorted(set(baseline).intersection(enhanced)):
        if baseline[key] is None or enhanced[key] is None:
            improvements[key] = None
        elif key.endswith("auc") or key.endswith("r2"):
            improvements[key] = float(enhanced[key] - baseline[key])
        else:
            improvements[key] = float(baseline[key] - enhanced[key])
    return improvements


def _heldout_signal_diagnostic(
    performance: Mapping[str, Any],
    *,
    metric: str,
    minimum_positive_fold_fraction: float = 0.5,
) -> dict[str, Any]:
    fold_values = [
        row.get("improvement_positive_is_better", {}).get(metric)
        for row in performance.get("inner_fold_performance") or []
    ]
    finite = [float(value) for value in fold_values if value is not None and math.isfinite(value)]
    aggregate = performance.get("improvement_positive_is_better", {}).get(metric)
    aggregate = (
        float(aggregate) if aggregate is not None and math.isfinite(float(aggregate)) else None
    )
    positive_folds = sum(value > 0.0 for value in finite)
    positive_fraction = float(positive_folds / len(finite)) if finite else 0.0
    return {
        "metric": metric,
        "aggregate_improvement": aggregate,
        "fold_improvements": finite,
        "positive_folds": int(positive_folds),
        "evaluated_folds": int(len(finite)),
        "positive_fold_fraction": positive_fraction,
        "minimum_positive_fold_fraction": float(minimum_positive_fold_fraction),
        "supported": bool(
            aggregate is not None
            and aggregate > 0.0
            and positive_fraction >= float(minimum_positive_fold_fraction)
        ),
    }


def _feature_role_signal_diagnostics(
    feature: Mapping[str, Any],
    performance: Mapping[str, Any],
    *,
    binary_outcome: bool,
) -> dict[str, Any]:
    """Evaluate each claimed causal role using predictions on inner held-out rows."""

    outcome_metric = "outcome_brier" if binary_outcome else "outcome_rmse"
    treatment = _heldout_signal_diagnostic(performance, metric="treatment_brier")
    outcome = _heldout_signal_diagnostic(performance, metric=outcome_metric)
    effect = _heldout_signal_diagnostic(performance, metric="effect_model_r_loss")
    claimed_roles = list(map(str, feature.get("roles") or []))
    role_signals: dict[str, Any] = {}
    if "confounder" in claimed_roles:
        role_signals["confounder"] = {
            "supported": bool(treatment["supported"] and outcome["supported"]),
            "requires_treatment_and_outcome_signal": True,
            "treatment_signal": treatment,
            "outcome_signal": outcome,
        }
    if "prognostic" in claimed_roles:
        role_signals["prognostic"] = {
            "supported": bool(outcome["supported"]),
            "outcome_signal": outcome,
        }
    if "effect_modifier" in claimed_roles:
        role_signals["effect_modifier"] = {
            "supported": bool(effect["supported"]),
            "residual_effect_signal": effect,
        }
    return {
        "claimed_roles": claimed_roles,
        "role_signals": role_signals,
        "has_any_claimed_role_signal": any(
            bool(signal.get("supported")) for signal in role_signals.values()
        ),
    }


def _selection_representation_fingerprint(feature: Mapping[str, Any]) -> str:
    """Identify the measured representation to which selection votes apply."""

    fields = (
        "value_type",
        "categories_or_unit",
        "measurement_definition",
        "missing_value_rule",
        "modeling_strategy",
        "harmonization_plan",
    )
    representation = {field: copy.deepcopy(feature.get(field)) for field in fields}
    plan = representation.get("harmonization_plan")
    if isinstance(plan, dict):
        plan.pop("training_observations_fingerprint", None)
    return _value_fingerprint(representation)


def _stability_selection_policy(config: Any) -> dict[str, Any]:
    """Read role-stability settings from current or older config objects."""

    return {
        "minimum_evaluations": int(
            getattr(
                config,
                "stability_selection_rounds",
                DEFAULT_STABILITY_SELECTION_ROUNDS,
            )
        ),
        "selection_frequency": float(
            getattr(
                config,
                "stability_selection_frequency",
                DEFAULT_STABILITY_SELECTION_FREQUENCY,
            )
        ),
        "effect_modifier_negative_margin_fraction": float(
            getattr(
                config,
                "effect_modifier_negative_margin_fraction",
                DEFAULT_EFFECT_MODIFIER_NEGATIVE_MARGIN_FRACTION,
            )
        ),
        "effect_modifier_negative_fold_fraction": float(
            getattr(
                config,
                "effect_modifier_negative_fold_fraction",
                DEFAULT_EFFECT_MODIFIER_NEGATIVE_FOLD_FRACTION,
            )
        ),
    }


def _modifier_negative_vote(
    signal: Mapping[str, Any],
    *,
    margin_fraction: float,
    minimum_negative_fold_fraction: float,
) -> dict[str, Any]:
    role_signal = (signal.get("role_signals") or {}).get("effect_modifier") or {}
    diagnostic = role_signal.get("residual_effect_signal") or {}
    aggregate = diagnostic.get("aggregate_improvement")
    aggregate = (
        float(aggregate) if aggregate is not None and math.isfinite(float(aggregate)) else None
    )
    baseline = (signal.get("baseline") or {}).get("effect_model_r_loss")
    baseline = (
        abs(float(baseline)) if baseline is not None and math.isfinite(float(baseline)) else None
    )
    fold_values = [
        float(value)
        for value in diagnostic.get("fold_improvements") or []
        if value is not None and math.isfinite(float(value))
    ]
    negative_folds = sum(value < 0.0 for value in fold_values)
    negative_fraction = float(negative_folds / len(fold_values)) if fold_values else 0.0
    required_margin = (
        max(1e-12, float(margin_fraction) * baseline) if baseline is not None else None
    )
    vote = bool(
        aggregate is not None
        and required_margin is not None
        and aggregate <= -required_margin
        and negative_fraction >= float(minimum_negative_fold_fraction)
    )
    return {
        "aggregate_improvement": aggregate,
        "baseline_effect_model_r_loss": baseline,
        "required_negative_margin": required_margin,
        "negative_folds": int(negative_folds),
        "evaluated_folds": int(len(fold_values)),
        "negative_fold_fraction": negative_fraction,
        "minimum_negative_fold_fraction": float(minimum_negative_fold_fraction),
        "meaningfully_negative": vote,
    }


def _update_stability_selection(
    *,
    definitions: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    history: MutableMapping[str, list[dict[str, Any]]],
    evaluation_round: int,
    config: Any,
) -> dict[str, Any]:
    """Accumulate repeated forest-screen votes for each feature representation."""

    policy = _stability_selection_policy(config)
    signal_by_id = {
        str(row["feature_id"]): row for row in performance.get("individual_feature_signal") or []
    }
    features: list[dict[str, Any]] = []
    for feature in definitions:
        feature_id = str(feature["feature_id"])
        representation = _selection_representation_fingerprint(feature)
        signal = signal_by_id.get(feature_id)
        if signal is None:
            raise ValueError(
                "Stage 2 stability selection is missing inner-held-out diagnostics "
                f"for {feature_id!r}"
            )
        role_signals = signal.get("role_signals") or {}
        role_rows: dict[str, Any] = {}
        for role in map(str, feature.get("roles") or []):
            role_signal = role_signals.get(role) or {}
            history_key = f"{feature_id}:{representation}:{role}"
            observation: dict[str, Any] = {
                "evaluation_round": int(evaluation_round),
                "supported": bool(role_signal.get("supported")),
            }
            if role == "effect_modifier":
                observation["negative_margin_diagnostic"] = _modifier_negative_vote(
                    signal,
                    margin_fraction=policy["effect_modifier_negative_margin_fraction"],
                    minimum_negative_fold_fraction=policy["effect_modifier_negative_fold_fraction"],
                )
            observations = history.setdefault(history_key, [])
            if not any(
                int(row.get("evaluation_round") or -1) == int(evaluation_round)
                for row in observations
            ):
                observations.append(observation)
            evaluations = len(observations)
            support_votes = sum(bool(row.get("supported")) for row in observations)
            support_frequency = float(support_votes / evaluations)
            negative_votes = sum(
                bool((row.get("negative_margin_diagnostic") or {}).get("meaningfully_negative"))
                for row in observations
            )
            negative_frequency = float(negative_votes / evaluations)
            pending = evaluations < int(policy["minimum_evaluations"])
            stable_positive = bool(
                not pending and support_frequency >= float(policy["selection_frequency"])
            )
            stable_meaningfully_negative = bool(
                role == "effect_modifier"
                and not pending
                and negative_frequency >= float(policy["selection_frequency"])
            )
            role_rows[role] = {
                "evaluations": int(evaluations),
                "support_votes": int(support_votes),
                "support_frequency": support_frequency,
                "meaningfully_negative_votes": int(negative_votes),
                "meaningfully_negative_frequency": negative_frequency,
                "pending": pending,
                "stable_positive": stable_positive,
                "stable_meaningfully_negative": stable_meaningfully_negative,
                "observations": copy.deepcopy(observations),
            }
        features.append(
            {
                "feature_id": feature_id,
                "name": str(feature["name"]),
                "representation_fingerprint": representation,
                "roles": role_rows,
            }
        )
    return {
        "schema_version": "stage2_role_stability_selection_v1",
        "model_family": "elastic_net_nuisance_plus_random_forest_effect_model",
        "nuisance_model_family": "elastic_net",
        "effect_model_family": "random_forest",
        "evaluation_round": int(evaluation_round),
        "policy": policy,
        "features": features,
    }


def _stable_roles_for_feature(
    feature: Mapping[str, Any],
    stability_selection: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Return retained roles and explanations under asymmetric stability rules."""

    feature_id = str(feature["feature_id"])
    row = next(
        (
            item
            for item in stability_selection.get("features") or []
            if str(item.get("feature_id") or "") == feature_id
        ),
        None,
    )
    if row is None:
        raise ValueError(f"stability selection is missing feature {feature_id!r}")
    retained: list[str] = []
    reasons: list[str] = []
    role_rows = row.get("roles") or {}
    for role in map(str, feature.get("roles") or []):
        role_row = role_rows.get(role) or {}
        if bool(role_row.get("pending")):
            retained.append(role)
            reasons.append(f"{role}: pending repeated forest screens")
        elif role == "effect_modifier":
            if bool(role_row.get("stable_meaningfully_negative")):
                reasons.append("effect_modifier: pruned after stable, margin-negative R-loss")
            else:
                retained.append(role)
                reasons.append("effect_modifier: retained without stable, margin-negative R-loss")
        elif bool(role_row.get("stable_positive")):
            retained.append(role)
            reasons.append(f"{role}: retained by stability selection")
        else:
            reasons.append(f"{role}: lacked stable positive support")
    return retained, reasons


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


def evaluate_definitions(
    *,
    dataset: pd.DataFrame,
    extracted: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    inner_folds: int,
    seed: int,
    propensity_clip: float,
    forest_trees: int = DEFAULT_SCREENING_TREES,
    include_ablation: bool = True,
) -> dict[str, Any]:
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    extraction_by_id = extracted.set_index("_oci_row_id", drop=False)
    supplied = list(split.get("inner_splits") or [])
    inner = supplied or _fallback_inner_splits(
        fit_ids,
        folds=inner_folds,
        seed=seed,
    )
    predictions: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "t",
            "y",
            "base_e",
            "feature_e",
            "base_y",
            "feature_y",
            "base_r_residual",
            "feature_null_effect_r_residual",
            "feature_r_residual",
        )
    }
    all_defs = list(definitions)
    propensity_defs = _definitions_for_roles(all_defs, {"confounder"})
    effect_defs = _definitions_for_roles(all_defs, {"effect_modifier"})
    binary = str(outcome_type) == "binary"
    fold_performance: list[dict[str, Any]] = []

    for fold_index, fold in enumerate(inner, start=1):
        train_ids = [int(value) for value in fold["fit_row_ids"] if int(value) in fit_ids]
        valid_ids = [int(value) for value in fold["heldout_row_ids"] if int(value) in fit_ids]
        if not train_ids or not valid_ids:
            continue
        train_features = extraction_by_id.loc[train_ids].reset_index(drop=True)
        valid_features = extraction_by_id.loc[valid_ids].reset_index(drop=True)
        train_data = dataset.iloc[train_ids]
        valid_data = dataset.iloc[valid_ids]
        t_train = train_data[treatment_column].to_numpy(dtype=float)
        y_train = train_data[outcome_column].to_numpy(dtype=float)
        t_valid = valid_data[treatment_column].to_numpy(dtype=float)
        y_valid = valid_data[outcome_column].to_numpy(dtype=float)

        base_x_train = np.empty((len(train_ids), 0), dtype=float)
        base_x_valid = np.empty((len(valid_ids), 0), dtype=float)
        base_t_model = _fit_classifier(
            base_x_train,
            t_train,
            seed=seed + fold_index,
        )
        base_outcome = _fit_outcome_models(
            base_x_train,
            t_train,
            y_train,
            binary=binary,
            seed=seed + fold_index,
        )
        base_e_train = _predict_probability(base_t_model, base_x_train)
        base_e_valid = _predict_probability(base_t_model, base_x_valid)
        base_mu0_train, base_mu1_train = _predict_outcomes(base_outcome, base_x_train)
        base_mu0_valid, base_mu1_valid = _predict_outcomes(base_outcome, base_x_valid)
        base_m_train = base_e_train * base_mu1_train + (1 - base_e_train) * base_mu0_train
        base_m_valid = base_e_valid * base_mu1_valid + (1 - base_e_valid) * base_mu0_valid
        base_pseudo = (y_train - base_m_train) / np.where(
            np.abs(t_train - base_e_train) < propensity_clip,
            np.where(t_train - base_e_train < 0, -propensity_clip, propensity_clip),
            t_train - base_e_train,
        )
        base_effect = _fit_effect_model(
            np.empty((len(train_ids), 0)),
            base_pseudo,
            seed=seed + fold_index,
            trees=forest_trees,
        )
        base_tau = base_effect.predict(np.empty((len(valid_ids), 0)))

        t_encoder = _FeatureEncoder(propensity_defs).fit(train_features)
        x_t_train = t_encoder.transform(train_features)
        x_t_valid = t_encoder.transform(valid_features)
        y_encoder = _FeatureEncoder(all_defs).fit(train_features)
        x_y_train = y_encoder.transform(train_features)
        x_y_valid = y_encoder.transform(valid_features)
        effect_encoder = _FeatureEncoder(effect_defs).fit(train_features)
        x_effect_train = effect_encoder.transform(train_features)
        x_effect_valid = effect_encoder.transform(valid_features)
        feature_t_model = _fit_classifier(
            x_t_train,
            t_train,
            seed=seed + 100 + fold_index,
        )
        feature_outcome = _fit_outcome_models(
            x_y_train,
            t_train,
            y_train,
            binary=binary,
            seed=seed + 100 + fold_index,
        )
        feature_e_train = _predict_probability(feature_t_model, x_t_train)
        feature_e_valid = _predict_probability(feature_t_model, x_t_valid)
        feature_mu0_train, feature_mu1_train = _predict_outcomes(feature_outcome, x_y_train)
        feature_mu0_valid, feature_mu1_valid = _predict_outcomes(feature_outcome, x_y_valid)
        feature_m_train = (
            feature_e_train * feature_mu1_train + (1 - feature_e_train) * feature_mu0_train
        )
        feature_m_valid = (
            feature_e_valid * feature_mu1_valid + (1 - feature_e_valid) * feature_mu0_valid
        )
        feature_pseudo = (y_train - feature_m_train) / np.where(
            np.abs(t_train - feature_e_train) < propensity_clip,
            np.where(t_train - feature_e_train < 0, -propensity_clip, propensity_clip),
            t_train - feature_e_train,
        )
        feature_effect = _fit_effect_model(
            x_effect_train,
            feature_pseudo,
            seed=seed + 200 + fold_index,
            trees=forest_trees,
        )
        feature_null_effect = _fit_effect_model(
            np.empty((len(train_ids), 0)),
            feature_pseudo,
            seed=seed + 200 + fold_index,
            trees=forest_trees,
        )
        feature_tau = feature_effect.predict(x_effect_valid)
        feature_null_tau = feature_null_effect.predict(np.empty((len(valid_ids), 0)))

        predictions["t"].append(t_valid)
        predictions["y"].append(y_valid)
        predictions["base_e"].append(base_e_valid)
        predictions["feature_e"].append(feature_e_valid)
        predictions["base_y"].append(np.where(t_valid == 1, base_mu1_valid, base_mu0_valid))
        predictions["feature_y"].append(
            np.where(t_valid == 1, feature_mu1_valid, feature_mu0_valid)
        )
        base_r_residual = (y_valid - base_m_valid) - (t_valid - base_e_valid) * base_tau
        feature_r_residual = (y_valid - feature_m_valid) - (t_valid - feature_e_valid) * feature_tau
        feature_null_effect_r_residual = (y_valid - feature_m_valid) - (
            t_valid - feature_e_valid
        ) * feature_null_tau
        predictions["base_r_residual"].append(base_r_residual)
        predictions["feature_null_effect_r_residual"].append(feature_null_effect_r_residual)
        predictions["feature_r_residual"].append(feature_r_residual)
        fold_base = _prediction_metrics(
            treatment=t_valid,
            outcome=y_valid,
            propensity=base_e_valid,
            observed_outcome=np.where(t_valid == 1, base_mu1_valid, base_mu0_valid),
            binary_outcome=binary,
            r_loss=float(np.mean(base_r_residual**2)),
        )
        fold_enhanced = _prediction_metrics(
            treatment=t_valid,
            outcome=y_valid,
            propensity=feature_e_valid,
            observed_outcome=np.where(t_valid == 1, feature_mu1_valid, feature_mu0_valid),
            binary_outcome=binary,
            r_loss=float(np.mean(feature_r_residual**2)),
        )
        fold_base["effect_model_r_loss"] = float(np.mean(feature_null_effect_r_residual**2))
        fold_enhanced["effect_model_r_loss"] = float(np.mean(feature_r_residual**2))
        fold_performance.append(
            {
                "inner_fold": int(fold.get("inner_fold") or fold_index),
                "evaluation_rows": int(len(valid_ids)),
                "baseline": fold_base,
                "with_extracted_features": fold_enhanced,
                "improvement_positive_is_better": _metric_improvements(
                    fold_base,
                    fold_enhanced,
                ),
            }
        )
    if not predictions["t"]:
        raise ValueError("Stage 2 empirical review has no usable inner validation folds")
    joined = {key: np.concatenate(value) for key, value in predictions.items()}
    base = _prediction_metrics(
        treatment=joined["t"],
        outcome=joined["y"],
        propensity=joined["base_e"],
        observed_outcome=joined["base_y"],
        binary_outcome=binary,
        r_loss=float(np.mean(joined["base_r_residual"] ** 2)),
    )
    enhanced = _prediction_metrics(
        treatment=joined["t"],
        outcome=joined["y"],
        propensity=joined["feature_e"],
        observed_outcome=joined["feature_y"],
        binary_outcome=binary,
        r_loss=float(np.mean(joined["feature_r_residual"] ** 2)),
    )
    base["effect_model_r_loss"] = float(np.mean(joined["feature_null_effect_r_residual"] ** 2))
    enhanced["effect_model_r_loss"] = float(np.mean(joined["feature_r_residual"] ** 2))
    improvements = _metric_improvements(base, enhanced)
    result: dict[str, Any] = {
        "model_family": "elastic_net_nuisance_plus_random_forest_effect_model",
        "nuisance_model_family": "elastic_net",
        "effect_model_family": "random_forest",
        "forest_trees": int(forest_trees),
        "evaluation_rows": int(len(joined["t"])),
        "inner_folds": int(len(fold_performance)),
        "inner_fold_performance": fold_performance,
        "baseline": base,
        "with_extracted_features": enhanced,
        "improvement_positive_is_better": improvements,
    }
    if include_ablation and definitions:
        ablations = []
        individual_signals = []
        for feature in definitions:
            singleton_result = evaluate_definitions(
                dataset=dataset,
                extracted=extracted,
                definitions=[feature],
                split=split,
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                outcome_type=outcome_type,
                inner_folds=inner_folds,
                seed=seed,
                propensity_clip=propensity_clip,
                forest_trees=forest_trees,
                include_ablation=False,
            )
            individual_signals.append(
                {
                    "feature_id": str(feature["feature_id"]),
                    "name": str(feature["name"]),
                    **_feature_role_signal_diagnostics(
                        feature,
                        singleton_result,
                        binary_outcome=binary,
                    ),
                    "baseline": singleton_result["baseline"],
                    "with_feature": singleton_result["with_extracted_features"],
                    "improvement_positive_is_better": singleton_result[
                        "improvement_positive_is_better"
                    ],
                    "inner_fold_performance": singleton_result["inner_fold_performance"],
                }
            )
            without = [
                candidate
                for candidate in definitions
                if str(candidate["feature_id"]) != str(feature["feature_id"])
            ]
            without_result = evaluate_definitions(
                dataset=dataset,
                extracted=extracted,
                definitions=without,
                split=split,
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                outcome_type=outcome_type,
                inner_folds=inner_folds,
                seed=seed,
                propensity_clip=propensity_clip,
                forest_trees=forest_trees,
                include_ablation=False,
            )
            without_metrics = without_result["with_extracted_features"]
            contribution = _metric_improvements(without_metrics, enhanced)
            ablations.append(
                {
                    "feature_id": str(feature["feature_id"]),
                    "name": str(feature["name"]),
                    "metrics_without_feature": without_metrics,
                    "feature_contribution_positive_is_better": contribution,
                }
            )
        result["individual_feature_signal"] = individual_signals
        result["leave_one_feature_out"] = ablations
    return result


def _review_prompt(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    allow_measurement_revision: bool,
    min_nonmissing_fraction: float,
    max_dominant_fraction: float,
    feature_set_index: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    detailed_feature_ids = [str(feature["feature_id"]) for feature in definitions]
    configured_feature_ids = [
        str(feature["feature_id"])
        for feature in definitions
        if feature.get("configured_explicit_feature") is True
    ]
    body = {
        "job": "review_stage2_variables_against_training_fold_performance",
        "clinical_question": clinical_question,
        "information_boundary": (
            "All extraction summaries and performance metrics come only from the outer training fold. "
            "No outer-heldout outcomes are included."
        ),
        "allow_measurement_revision": allow_measurement_revision,
        "quality_guides": {
            "minimum_nonmissing_fraction": min_nonmissing_fraction,
            "maximum_dominant_value_fraction": max_dominant_fraction,
        },
        "review_scope": {
            "detailed_feature_ids": detailed_feature_ids,
            "configured_explicit_feature_ids": configured_feature_ids,
            "feature_count_in_entire_set": len(feature_set_index or definitions),
        },
        "rules": [
            "Give every detailed feature exactly one decision.",
            "Do not return decisions for index-only features; they are reviewed in other requests.",
            "Use the feature-set index to recognize related or potentially redundant variables.",
            "Keep a feature when extraction is usable and its scientific role remains plausible.",
            "Drop a feature when it is essentially unmeasured, invariant, or unsupported after extraction.",
            "Individual signals use random forests scored on inner-held-out rows.",
            "Stability selection requires ordinary roles to earn stable positive support; "
            "effect modifiers are removed only after stable, margin-negative R-loss. "
            "Do not bypass that asymmetric gate.",
            "Use leave-one-feature-out metrics to distinguish a feature's contribution from overall model performance.",
            "For every retained continuous feature, choose a modeling_strategy. Use "
            "continuous for numeric measurements and categorical for stable categories. "
            "Use a hybrid only when both remain and no harmonization_plan exists. Never "
            "invent a numeric value for a category or threshold.",
            *(
                [
                    "A harmonization_plan was learned from mixed outer-training values. "
                    "Keep its target_representation as modeling_strategy unless revising "
                    "the underlying measurement definition."
                ]
                if any(
                    isinstance(feature.get("harmonization_plan"), Mapping)
                    for feature in definitions
                )
                else []
            ),
            *(
                [
                    "A harmonization_fallback means no validated common plan was "
                    "available after bounded repairs. Keep the current safe modeling "
                    "strategy unless the training summaries support another declared "
                    "strategy; never invent a value mapping."
                ]
                if any(
                    isinstance(feature.get("harmonization_fallback"), Mapping)
                    for feature in definitions
                )
                else []
            ),
            "Use revise only to clarify how the same evidence-supported measurement is extracted.",
            "For a revised binary variable, provide exactly two distinct scalar ontology values as separate categories_or_unit array items.",
            "For a revised categorical or ordinal variable, provide at least two distinct scalar ontology values as separate categories_or_unit array items.",
            "Do not add a new feature, change a causal role, or change supporting evidence.",
            "A feature listed in configured_explicit_feature_ids was required by the investigator with a supplied ontology; return keep and do not revise or drop it.",
            "Predictive performance is diagnostic evidence, not permission to use a post-treatment variable.",
            (
                "Measurement revision is permitted and will be evaluated in another round."
                if allow_measurement_revision
                else "This is the final review round; choose only keep or drop."
            ),
        ],
        "feature_set_index": list(feature_set_index or []),
        "features": [
            {
                **dict(feature),
                **(
                    {
                        "harmonization_plan": {
                            key: copy.deepcopy(feature["harmonization_plan"].get(key))
                            for key in (
                                "target_representation",
                                "reason",
                                "canonical_categories",
                            )
                        }
                    }
                    if isinstance(feature.get("harmonization_plan"), Mapping)
                    else {}
                ),
                **(
                    {
                        "harmonization_fallback": {
                            key: copy.deepcopy(feature["harmonization_fallback"].get(key))
                            for key in (
                                "status",
                                "retained_prior_plan",
                                "unresolved_value_rule",
                                "modeling_strategy",
                            )
                        }
                    }
                    if isinstance(feature.get("harmonization_fallback"), Mapping)
                    else {}
                ),
            }
            for feature in definitions
        ],
        "extraction_summaries": list(summaries),
        "inner_validation_performance": dict(performance),
        "response": {
            "feature_decisions": [
                {
                    "feature_id": "one supplied feature_id",
                    "action": "keep|drop|revise",
                    "reason": "scientific and empirical reason",
                    "modeling_strategy": (
                        "required for every kept or revised continuous feature: "
                        "continuous|categorical|continuous_with_categorical_fallback"
                    ),
                    "value_type": "required only for revise",
                    "categories_or_unit": ["required only for revise"],
                    "measurement_definition": "required only for revise",
                    "missing_value_rule": "required only for revise",
                }
            ],
            "overall_assessment": "brief assessment of the feature set",
        },
    }
    return [
        {
            "role": "system",
            "content": "You review prespecified variables using training-fold evidence only. Return JSON only.",
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True)},
    ]


def _review_feature_set_index(
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain compact whole-set context in every partitioned review request."""

    return [
        {
            "feature_id": str(feature["feature_id"]),
            "name": str(feature["name"]),
            "description": str(feature.get("description") or ""),
            "roles": list(feature.get("roles") or []),
            "value_type": str(feature.get("value_type") or ""),
            "modeling_strategy": str(feature.get("modeling_strategy") or ""),
        }
        for feature in definitions
    ]


def _review_performance_for_features(
    performance: Mapping[str, Any],
    *,
    feature_ids: set[str],
) -> dict[str, Any]:
    """Return global metrics plus each detailed feature's own ablation metrics."""

    per_feature_keys = {
        "individual_feature_signal",
        "leave_one_feature_out",
        "stability_selection",
    }
    selected = {
        key: copy.deepcopy(value)
        for key, value in performance.items()
        if key not in per_feature_keys
    }
    selected["individual_feature_signal"] = [
        copy.deepcopy(row)
        for row in performance.get("individual_feature_signal") or []
        if str(row.get("feature_id") or "") in feature_ids
    ]
    selected["leave_one_feature_out"] = [
        copy.deepcopy(row)
        for row in performance.get("leave_one_feature_out") or []
        if str(row.get("feature_id") or "") in feature_ids
    ]
    stability = performance.get("stability_selection")
    if isinstance(stability, Mapping):
        compact_features: list[dict[str, Any]] = []
        for row in stability.get("features") or []:
            if str(row.get("feature_id") or "") not in feature_ids:
                continue
            compact_roles = {
                str(role): {
                    key: copy.deepcopy(role_row.get(key))
                    for key in (
                        "evaluations",
                        "support_frequency",
                        "pending",
                        "stable_positive",
                        "meaningfully_negative_frequency",
                        "stable_meaningfully_negative",
                    )
                }
                for role, role_row in (row.get("roles") or {}).items()
            }
            compact_features.append(
                {
                    "feature_id": str(row.get("feature_id") or ""),
                    "name": str(row.get("name") or ""),
                    "roles": compact_roles,
                }
            )
        selected["stability_selection"] = {
            "features": compact_features,
        }
    return selected


def _review_prompt_for_features(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    summaries_by_id: Mapping[str, Mapping[str, Any]],
    performance: Mapping[str, Any],
    feature_set_index: Sequence[Mapping[str, Any]],
    allow_measurement_revision: bool,
    min_nonmissing_fraction: float,
    max_dominant_fraction: float,
) -> list[dict[str, str]]:
    feature_ids = {str(feature["feature_id"]) for feature in definitions}
    return _review_prompt(
        clinical_question=clinical_question,
        definitions=definitions,
        summaries=[summaries_by_id[str(feature["feature_id"])] for feature in definitions],
        performance=_review_performance_for_features(
            performance,
            feature_ids=feature_ids,
        ),
        allow_measurement_revision=allow_measurement_revision,
        min_nonmissing_fraction=min_nonmissing_fraction,
        max_dominant_fraction=max_dominant_fraction,
        feature_set_index=feature_set_index,
    )


def _partition_review_features(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    allow_measurement_revision: bool,
    min_nonmissing_fraction: float,
    max_dominant_fraction: float,
    max_prompt_chars: int,
) -> list[list[dict[str, Any]]]:
    """Partition detailed review inputs while preserving every per-feature diagnostic.

    The soft limit leaves space for validation-repair turns. A single unusually
    verbose feature may exceed that soft limit, but it must still fit the configured
    hard transport limit.
    """

    hard_limit = int(max_prompt_chars)
    if hard_limit < 1:
        raise ValueError("Stage 2 max_prompt_chars must be positive")
    soft_limit = min(hard_limit, max(20_000, int(hard_limit * 0.6)))
    summaries_by_id = {str(row["feature_id"]): row for row in summaries}
    expected_ids = {str(feature["feature_id"]) for feature in definitions}
    missing_summaries = sorted(expected_ids - set(summaries_by_id))
    if missing_summaries:
        raise ValueError(
            "Stage 2 review is missing extraction summaries for feature ID(s): "
            f"{missing_summaries[:8]}"
        )
    feature_set_index = _review_feature_set_index(definitions)
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for raw_feature in definitions:
        feature = dict(raw_feature)
        proposed = [*current, feature]
        messages = _review_prompt_for_features(
            clinical_question=clinical_question,
            definitions=proposed,
            summaries_by_id=summaries_by_id,
            performance=performance,
            feature_set_index=feature_set_index,
            allow_measurement_revision=allow_measurement_revision,
            min_nonmissing_fraction=min_nonmissing_fraction,
            max_dominant_fraction=max_dominant_fraction,
        )
        prompt_chars = _prompt_chars(messages)
        if current and prompt_chars > soft_limit:
            batches.append(current)
            current = [feature]
            singleton_messages = _review_prompt_for_features(
                clinical_question=clinical_question,
                definitions=current,
                summaries_by_id=summaries_by_id,
                performance=performance,
                feature_set_index=feature_set_index,
                allow_measurement_revision=allow_measurement_revision,
                min_nonmissing_fraction=min_nonmissing_fraction,
                max_dominant_fraction=max_dominant_fraction,
            )
            prompt_chars = _prompt_chars(singleton_messages)
        else:
            current = proposed
        if prompt_chars > hard_limit:
            raise ValueError(
                "Stage 2 cannot review one complete feature within max_prompt_chars "
                f"({prompt_chars} > {hard_limit}); shorten that feature's metadata or "
                "increase the prompt budget"
            )
    if current:
        batches.append(current)
    return batches


def _validate_review(
    value: Mapping[str, Any],
    *,
    definitions: Sequence[Mapping[str, Any]],
    allow_measurement_revision: bool,
    summaries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    decisions = value.get("feature_decisions")
    if not isinstance(decisions, list):
        raise ValueError("review response requires feature_decisions")
    by_id = {str(feature["feature_id"]): dict(feature) for feature in definitions}
    clean: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("each feature decision must be an object")
        feature_id = str(decision.get("feature_id") or "")
        if feature_id not in by_id or feature_id in clean:
            raise ValueError("review named an unknown or duplicate feature_id")
        action = str(decision.get("action") or "")
        if action not in {"keep", "drop", "revise"}:
            raise ValueError("review action must be keep, drop, or revise")
        if by_id[feature_id].get("configured_explicit_feature") is True and action != "keep":
            raise ValueError(
                f"investigator-configured feature {feature_id!r} must be kept without revision"
            )
        if action == "revise" and not allow_measurement_revision:
            raise ValueError("measurement revision is not permitted in the final review round")
        row: dict[str, Any] = {
            "feature_id": feature_id,
            "action": action,
            "reason": str(decision.get("reason") or ""),
        }
        if action == "revise":
            value_type = str(decision.get("value_type") or "")
            categories = decision.get("categories_or_unit")
            if value_type not in {"binary", "categorical", "continuous", "ordinal"}:
                raise ValueError("a revised variable requires an operational value_type")
            if not isinstance(categories, list) or not categories:
                raise ValueError("a revised variable requires categories_or_unit")
            if value_type in {"binary", "categorical", "ordinal"}:
                categories = _validated_closed_category_values(
                    value_type=value_type,
                    values=categories,
                    source=f"revised feature {feature_id!r}",
                )
            for key in ("measurement_definition", "missing_value_rule"):
                if not str(decision.get(key) or "").strip():
                    raise ValueError(f"a revised variable requires {key}")
            row.update(
                {
                    "value_type": value_type,
                    "categories_or_unit": [str(item) for item in categories],
                    "measurement_definition": str(decision["measurement_definition"]),
                    "missing_value_rule": str(decision["missing_value_rule"]),
                }
            )
        resulting_value_type = str(
            row.get("value_type") or by_id[feature_id].get("value_type") or "ambiguous"
        )
        if action != "drop" and resulting_value_type == "continuous":
            modeling_strategy = str(decision.get("modeling_strategy") or "").strip().lower()
            if not modeling_strategy and summaries is None:
                modeling_strategy = _feature_modeling_strategy(by_id[feature_id])
            if modeling_strategy not in CONTINUOUS_MODELING_STRATEGIES:
                raise ValueError(
                    "a retained continuous variable requires modeling_strategy to be "
                    "continuous, categorical, or continuous_with_categorical_fallback"
                )
            row["modeling_strategy"] = modeling_strategy
        clean[feature_id] = row
    if set(clean) != set(by_id):
        raise ValueError("review must decide every supplied feature")
    return {
        "feature_decisions": [clean[feature_id] for feature_id in by_id],
        "overall_assessment": str(value.get("overall_assessment") or ""),
    }


def _request_partitioned_review(
    *,
    clinical_question: str,
    definitions: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    allow_measurement_revision: bool,
    min_nonmissing_fraction: float,
    max_dominant_fraction: float,
    max_prompt_chars: int,
    output_dir: Path,
    request_json: RequestJSON,
) -> dict[str, Any]:
    """Review every feature in resumable prompt-sized groups."""

    batches = _partition_review_features(
        clinical_question=clinical_question,
        definitions=definitions,
        summaries=summaries,
        performance=performance,
        allow_measurement_revision=allow_measurement_revision,
        min_nonmissing_fraction=min_nonmissing_fraction,
        max_dominant_fraction=max_dominant_fraction,
        max_prompt_chars=max_prompt_chars,
    )
    summaries_by_id = {str(row["feature_id"]): row for row in summaries}
    feature_set_index = _review_feature_set_index(definitions)
    prompt_sizes: list[int] = []
    decisions: list[dict[str, Any]] = []
    assessments: list[str] = []
    for batch_index, batch_definitions in enumerate(batches, start=1):
        messages = _review_prompt_for_features(
            clinical_question=clinical_question,
            definitions=batch_definitions,
            summaries_by_id=summaries_by_id,
            performance=performance,
            feature_set_index=feature_set_index,
            allow_measurement_revision=allow_measurement_revision,
            min_nonmissing_fraction=min_nonmissing_fraction,
            max_dominant_fraction=max_dominant_fraction,
        )
        prompt_chars = _prompt_chars(messages)
        prompt_sizes.append(prompt_chars)
        if prompt_chars > int(max_prompt_chars):  # pragma: no cover - planner invariant
            raise RuntimeError("Stage 2 review partition exceeded max_prompt_chars")
        batch_dir = output_dir / f"batch_{batch_index:05d}"
        input_value = {
            "schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
            "clinical_question": clinical_question,
            "allow_measurement_revision": allow_measurement_revision,
            "feature_set_index": feature_set_index,
            "definitions": list(batch_definitions),
            "summaries": [
                summaries_by_id[str(feature["feature_id"])] for feature in batch_definitions
            ],
            "performance": _review_performance_for_features(
                performance,
                feature_ids={str(feature["feature_id"]) for feature in batch_definitions},
            ),
            "quality_guides": {
                "minimum_nonmissing_fraction": min_nonmissing_fraction,
                "maximum_dominant_value_fraction": max_dominant_fraction,
            },
        }
        input_fingerprint = _value_fingerprint(input_value)
        result_path = batch_dir / "result.json"
        complete_path = batch_dir / "complete.json"
        batch_review: dict[str, Any] | None = None
        if result_path.is_file() and complete_path.is_file():
            try:
                completion = json.loads(complete_path.read_text(encoding="utf-8"))
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    completion.get("schema_version") == REVIEW_CHECKPOINT_SCHEMA_VERSION
                    and completion.get("input_fingerprint") == input_fingerprint
                ):
                    batch_review = _validate_review(
                        cached,
                        definitions=batch_definitions,
                        allow_measurement_revision=allow_measurement_revision,
                        summaries=[
                            summaries_by_id[str(feature["feature_id"])]
                            for feature in batch_definitions
                        ],
                    )
            except (
                KeyError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                batch_review = None
        if batch_review is None:
            batch_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                batch_dir / "input.json",
                {**input_value, "input_fingerprint": input_fingerprint},
            )
            batch_review = request_json(
                messages,
                lambda value, batch_definitions=batch_definitions: _validate_review(
                    value,
                    definitions=batch_definitions,
                    allow_measurement_revision=allow_measurement_revision,
                    summaries=[
                        summaries_by_id[str(feature["feature_id"])] for feature in batch_definitions
                    ],
                ),
            )
            _write_json(result_path, batch_review)
            _write_json(
                complete_path,
                {
                    "status": "complete",
                    "schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
                    "input_fingerprint": input_fingerprint,
                    "completed_at": _now(),
                    "batch_index": batch_index,
                    "batches": len(batches),
                    "features": len(batch_definitions),
                    "prompt_chars": prompt_chars,
                },
            )
        decisions.extend(batch_review["feature_decisions"])
        assessment = str(batch_review.get("overall_assessment") or "").strip()
        if assessment:
            assessments.append(f"Review group {batch_index}/{len(batches)}: {assessment}")
    LOGGER.info(
        "Stage 2 feature review groups=%s features=%s prompt_chars=%s",
        len(batches),
        len(definitions),
        prompt_sizes,
    )
    return _validate_review(
        {
            "feature_decisions": decisions,
            "overall_assessment": "\n".join(assessments),
        },
        definitions=definitions,
        allow_measurement_revision=allow_measurement_revision,
        summaries=summaries,
    )


def _review_drop_stability_guards(
    definitions: Sequence[Mapping[str, Any]],
    review: Mapping[str, Any],
    stability_selection: Mapping[str, Any],
) -> tuple[set[str], dict[str, Any]]:
    """Prevent one LLM review from bypassing repeated empirical selection."""

    decisions = {str(row["feature_id"]): row for row in review["feature_decisions"]}
    protected: set[str] = set()
    audit_rows: list[dict[str, Any]] = []
    for feature in definitions:
        feature_id = str(feature["feature_id"])
        if decisions[feature_id]["action"] != "drop":
            continue
        stable_roles, reasons = _stable_roles_for_feature(feature, stability_selection)
        is_protected = bool(stable_roles)
        if is_protected:
            protected.add(feature_id)
        audit_rows.append(
            {
                "feature_id": feature_id,
                "name": str(feature["name"]),
                "llm_action": "drop",
                "action": (
                    "override_drop_until_stability_rule_allows"
                    if is_protected
                    else "allow_drop_after_stability_rule"
                ),
                "roles_retained_by_stability_rule": stable_roles,
                "reasons": reasons,
            }
        )
    return protected, {
        "schema_version": "stage2_review_drop_stability_guard_v1",
        "llm_drop_decisions": len(audit_rows),
        "drop_decisions_overridden": len(protected),
        "decisions": audit_rows,
    }


def _apply_review(
    definitions: Sequence[Mapping[str, Any]],
    review: Mapping[str, Any],
    *,
    protected_drop_feature_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    decisions = {str(row["feature_id"]): row for row in review["feature_decisions"]}
    protected = set(protected_drop_feature_ids or set())
    revised: list[dict[str, Any]] = []
    measurement_changed = False
    for feature in definitions:
        feature_id = str(feature["feature_id"])
        decision = decisions[feature_id]
        if decision["action"] == "drop" and feature_id not in protected:
            continue
        updated = dict(feature)
        if decision["action"] == "revise":
            measurement_changed = True
            updated.pop("harmonization_plan", None)
            updated.pop("harmonization_fallback", None)
            for key in (
                "value_type",
                "categories_or_unit",
                "measurement_definition",
                "missing_value_rule",
            ):
                updated[key] = decision[key]
            updated = _refresh_conflict_resolution(updated)
        if "modeling_strategy" in decision:
            updated["modeling_strategy"] = decision["modeling_strategy"]
        revised.append(_normalized_feature_modeling_definition(updated))
    return revised, measurement_changed


def _changed_feature_representation_ids(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Identify retained features that need evaluation under a changed representation."""

    before_by_id = {str(feature["feature_id"]): feature for feature in before}
    fields = (
        "value_type",
        "categories_or_unit",
        "measurement_definition",
        "missing_value_rule",
        "modeling_strategy",
        "harmonization_plan",
    )
    changed: set[str] = set()
    for feature in after:
        feature_id = str(feature["feature_id"])
        previous = before_by_id.get(feature_id)
        if previous is None:
            changed.add(feature_id)
            continue
        prior_view = {key: copy.deepcopy(previous.get(key)) for key in fields}
        current_view = {key: copy.deepcopy(feature.get(key)) for key in fields}
        if _value_fingerprint(prior_view) != _value_fingerprint(current_view):
            changed.add(feature_id)
    return changed


def _apply_empirical_signal_pruning(
    definitions: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    *,
    defer_feature_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove unsupported causal roles using inner-held-out singleton models.

    A dual-role feature may remain under the role that validates even when its
    other proposed role does not. Investigator-configured features are audited
    but remain immutable.
    """

    deferred = set(defer_feature_ids or set())
    stability_selection = performance.get("stability_selection")
    use_stability_selection = isinstance(stability_selection, Mapping)
    signal_by_id = {
        str(row["feature_id"]): row for row in performance.get("individual_feature_signal") or []
    }
    retained: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for raw_feature in definitions:
        feature = dict(raw_feature)
        feature_id = str(feature["feature_id"])
        roles = list(map(str, feature.get("roles") or []))
        signal = signal_by_id.get(feature_id)
        if signal is None:
            raise ValueError(
                f"Stage 2 signal pruning is missing inner-held-out diagnostics for {feature_id!r}"
            )
        role_signals = signal.get("role_signals") or {}
        stability_reasons: list[str] = []
        if use_stability_selection:
            supported_roles, stability_reasons = _stable_roles_for_feature(
                feature,
                stability_selection,
            )
        else:
            supported_roles = [
                role for role in roles if bool((role_signals.get(role) or {}).get("supported"))
            ]
        if feature.get("configured_explicit_feature") is True:
            retained.append(feature)
            action = "keep_configured"
            retained_roles = roles
        elif feature_id in deferred:
            retained.append(feature)
            action = "defer_until_re_evaluated"
            retained_roles = roles
        elif supported_roles:
            feature["roles"] = supported_roles
            retained.append(feature)
            retained_roles = supported_roles
            action = "keep" if supported_roles == roles else "prune_unsupported_roles"
        else:
            retained_roles = []
            action = "drop_no_heldout_role_signal"
        decisions.append(
            {
                "feature_id": feature_id,
                "name": str(feature["name"]),
                "action": action,
                "claimed_roles": roles,
                "retained_roles": retained_roles,
                "role_signals": copy.deepcopy(role_signals),
                "stability_reasons": stability_reasons,
            }
        )
    selection_complete = True
    if use_stability_selection:
        for decision in decisions:
            if decision["action"] == "defer_until_re_evaluated":
                selection_complete = False
                break
            if decision["action"] == "keep_configured":
                continue
            feature_row = next(
                (
                    row
                    for row in stability_selection.get("features") or []
                    if str(row.get("feature_id") or "") == decision["feature_id"]
                ),
                None,
            )
            if feature_row is not None and any(
                bool((feature_row.get("roles") or {}).get(role, {}).get("pending"))
                for role in decision["retained_roles"]
            ):
                selection_complete = False
                break
    report = {
        "schema_version": (
            "stage2_inner_heldout_signal_pruning_v2_stability_selection"
            if use_stability_selection
            else "stage2_inner_heldout_signal_pruning_v1"
        ),
        "selection_complete": selection_complete,
        "stability_selection": (
            copy.deepcopy(stability_selection) if use_stability_selection else None
        ),
        "features_evaluated": len(definitions),
        "features_retained": len(retained),
        "features_dropped": sum(
            decision["action"] == "drop_no_heldout_role_signal" for decision in decisions
        ),
        "features_with_roles_pruned": sum(
            decision["action"] == "prune_unsupported_roles" for decision in decisions
        ),
        "features_deferred_for_re_evaluation": sum(
            decision["action"] == "defer_until_re_evaluated" for decision in decisions
        ),
        "decisions": decisions,
    }
    return retained, report


def _ontology_refinement_prompt(
    *,
    feature: Mapping[str, Any],
    failure_patterns: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Request a same-feature ontology repair from repeated training failures."""

    body = {
        "job": "refine_stage2_feature_ontology_from_repeated_extraction_failures",
        "information_boundary": (
            "These aggregate diagnostics come only from outer-training patients. "
            "No held-out patient text, treatment, or outcome is supplied."
        ),
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
        "repeated_failure_patterns": [
            {
                key: copy.deepcopy(pattern.get(key))
                for key in (
                    "failure_kind",
                    "reason",
                    "patient_count",
                    "example_values",
                    "allowed_categories",
                )
            }
            for pattern in failure_patterns
        ],
        "rules": [
            "Refine only the supplied feature's extraction ontology; do not rename, merge, split, add, or drop a feature and do not change its causal roles.",
            "The example values are prior model outputs that failed validation, not verified patient facts.",
            "Use revise only when the repeated failures identify a correctable mismatch in value type, closed categories or unit, measurement definition, or missing-value rule.",
            "Use keep when the current ontology is already appropriate and the failures do not justify a change.",
            "A revised ontology must still define exactly one reusable patient-level scalar measurement.",
            "Prefer a numeric continuous ontology when the named measurement is realistically extractable as one number; include one unit when applicable.",
            "For binary variables return exactly two distinct extractable scalar categories; for categorical or ordinal variables return at least two.",
            "Do not blindly add every failed output as a category; choose a stable, reproducible ontology and clarify how source documentation maps to it.",
            "Return JSON only.",
        ],
        "response": {
            "action": "keep|revise",
            "reason": "why the ontology is retained or changed",
            "description": "required for revise",
            "value_type": "binary|categorical|continuous|ordinal; required for revise",
            "categories_or_unit": ["required for revise; empty only for unitless continuous"],
            "measurement_definition": "required for revise",
            "missing_value_rule": "required for revise",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You refine one clinical extraction ontology from repeated validation "
                "failures on training patients. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True, ensure_ascii=False)},
    ]


def _validate_ontology_refinement(
    value: Mapping[str, Any],
    *,
    feature: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a bounded same-feature ontology decision."""

    if feature.get("configured_explicit_feature") is True:
        raise ValueError("investigator-configured feature ontology cannot be revised")
    action = str(value.get("action") or "").strip().lower()
    if action not in {"keep", "revise"}:
        raise ValueError("ontology refinement action must be keep or revise")
    decision: dict[str, Any] = {
        "feature_id": str(feature["feature_id"]),
        "feature_name": str(feature["name"]),
        "action": action,
        "reason": str(value.get("reason") or "").strip(),
    }
    if action == "keep":
        return decision

    description = str(value.get("description") or "").strip()
    value_type = str(value.get("value_type") or "").strip().lower()
    categories = value.get("categories_or_unit")
    measurement_definition = str(value.get("measurement_definition") or "").strip()
    missing_value_rule = str(value.get("missing_value_rule") or "").strip()
    if not description:
        raise ValueError("a revised ontology requires description")
    if value_type not in {"binary", "categorical", "continuous", "ordinal"}:
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
    elif len(clean_categories) > 1:
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


def _aggregate_ontology_supervisor_prompt(
    *,
    feature: Mapping[str, Any],
    summary: Mapping[str, Any],
    failure_patterns: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Ask the primary model to audit one small-model extraction schema."""

    body = {
        "job": "review_stage2_small_model_extraction_ontology",
        "information_boundary": (
            "Only aggregate extraction values and validation failures from outer-training "
            "patients are supplied. No patient text, treatment values, outcome values, "
            "causal-role evidence, model performance, or p-values are supplied."
        ),
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
        "aggregate_extraction_summary": copy.deepcopy(dict(summary)),
        "aggregate_validation_failures": [
            {
                key: copy.deepcopy(pattern.get(key))
                for key in (
                    "failure_kind",
                    "reason",
                    "patient_count",
                    "example_values",
                    "allowed_categories",
                )
            }
            for pattern in failure_patterns
        ],
        "rules": [
            "Return keep unless the aggregates demonstrate a correctable extraction-schema mismatch.",
            "You may revise only description, value_type, categories_or_unit, measurement_definition, and missing_value_rule.",
            "Never add, drop, split, merge, or rename a feature and never infer or change a causal role.",
            "A revision must remain one reusable pretreatment patient-level scalar variable.",
            "Do not optimize for association with treatment or outcome; neither is available.",
            "For binary variables return exactly two distinct scalar categories; for categorical or ordinal variables return at least two.",
            "Return JSON only.",
        ],
        "response": {
            "action": "keep|revise",
            "reason": "schema-quality rationale",
            "description": "required for revise",
            "value_type": "binary|categorical|continuous|ordinal; required for revise",
            "categories_or_unit": ["required for revise"],
            "measurement_definition": "required for revise",
            "missing_value_rule": "required for revise",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You supervise extraction ontologies using aggregate small-model outputs. "
                "You cannot select features or causal roles. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(body, sort_keys=True, ensure_ascii=False)},
    ]


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
        if _records_infrastructure_failure(result_path):
            _supersede_infrastructure_checkpoint(directory)
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
            decision = request_json(
                _aggregate_ontology_supervisor_prompt(
                    feature=feature,
                    summary=summary,
                    failure_patterns=failures,
                ),
                lambda value: _validate_ontology_refinement(value, feature=feature),
                request_kind="interpretation",
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
        if decision["action"] == "revise":
            before = {
                key: copy.deepcopy(feature.get(key))
                for key in (
                    "description",
                    "value_type",
                    "categories_or_unit",
                    "measurement_definition",
                    "missing_value_rule",
                )
            }
            for key in before:
                feature[key] = copy.deepcopy(decision[key])
            after = {key: copy.deepcopy(feature.get(key)) for key in before}
            if _value_fingerprint(before) != _value_fingerprint(after):
                feature.pop("harmonization_plan", None)
                feature.pop("harmonization_fallback", None)
                feature = _refresh_conflict_resolution(feature)
                changed_ids.append(str(feature["feature_id"]))
        updated.append(feature)

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
            if _records_infrastructure_failure(result_path):
                _supersede_infrastructure_checkpoint(feature_dir)
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
            decision = request_json(
                messages,
                lambda value: _validate_ontology_refinement(value, feature=feature),
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
        if decision is not None and decision["action"] == "revise":
            before = {
                key: copy.deepcopy(feature.get(key))
                for key in (
                    "description",
                    "value_type",
                    "categories_or_unit",
                    "measurement_definition",
                    "missing_value_rule",
                )
            }
            for key in (
                "description",
                "value_type",
                "categories_or_unit",
                "measurement_definition",
                "missing_value_rule",
            ):
                feature[key] = copy.deepcopy(decision[key])
            after = {key: copy.deepcopy(feature.get(key)) for key in before}
            if _value_fingerprint(before) != _value_fingerprint(after):
                feature.pop("harmonization_plan", None)
                feature.pop("harmonization_fallback", None)
                feature = _refresh_conflict_resolution(feature)
                changed_names.append(name)
        updated.append(feature)

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
    return {
        "schema_version": EXTRACTION_ISSUE_SCHEMA_VERSION,
        "completed_at": _now(),
        "issue_files": int(prior_summary.get("issue_files") or 0)
        + int(delta_summary.get("issue_files") or 0),
        "feature_failure_patterns": patterns,
        "structural_failure_patient_count": len(structural_rows),
        "structural_failure_patient_row_ids": sorted(structural_rows),
        "incremental_refinement": {
            "schema_version": INCREMENTAL_REFINEMENT_EXTRACTION_SCHEMA_VERSION,
            "reextracted_feature_names": sorted(changed),
            "reused_feature_names": sorted(current - changed),
            "structural_failure_policy": "union_reused_and_delta_patient_rows",
        },
    }


def _has_legacy_full_refinement_checkpoints(output_dir: Path) -> bool:
    """Detect a full re-extraction started by code predating feature deltas."""

    if (output_dir / "changed_features").exists():
        return False
    return any(
        (output_dir / name).exists()
        for name in (
            "batches",
            "pages",
            "extracted.csv",
            "failure_summary.json",
            "complete.json",
        )
    )


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

    # A long-running workflow may already have started a full pass under the
    # historical directory layout.  Resume that work rather than abandoning
    # valid checkpoints halfway through a patient cohort.
    if _has_legacy_full_refinement_checkpoints(output_dir):
        LOGGER.info(
            "resume legacy full Stage 2 refinement extraction before enabling "
            "feature-delta checkpoints: %s",
            output_dir,
        )
        extracted = extract_rows(
            dataset=dataset,
            row_ids=row_ids,
            text_column=text_column,
            definitions=current,
            output_dir=output_dir,
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
            (output_dir / "failure_summary.json").read_text(encoding="utf-8")
        )
        return extracted, summary

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
            workers=workers,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_ids = [int(value) for value in fit_ids]
    position = {row_id: index for index, row_id in enumerate(fit_ids)}
    e = np.full(len(fit_ids), np.nan)
    mu0 = np.full(len(fit_ids), np.nan)
    mu1 = np.full(len(fit_ids), np.nan)
    extracted_by_id = extracted.set_index("_oci_row_id", drop=False)
    propensity_defs = _definitions_for_nuisance_role(definitions, "treatment")
    outcome_defs = _definitions_for_nuisance_role(definitions, "outcome")
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
        treatment_model = _fit_classifier(
            x_t_train,
            t_train,
            seed=seed + fold_index,
        )
        outcome_models = _fit_outcome_models(
            x_y_train,
            t_train,
            y_train,
            binary=binary,
            seed=seed + fold_index,
        )
        fold_mu0, fold_mu1 = _predict_outcomes(outcome_models, x_y_valid)
        valid_positions = [position[row_id] for row_id in valid_ids]
        e[valid_positions] = _predict_probability(treatment_model, x_t_valid)
        mu0[valid_positions] = fold_mu0
        mu1[valid_positions] = fold_mu1
    if np.isnan(e).any() or np.isnan(mu0).any() or np.isnan(mu1).any():
        raise ValueError(
            "inner splits do not provide one nuisance prediction for every outer-fit row"
        )
    return e, mu0, mu1


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
) -> dict[str, Any]:
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
            "dataset_modeling_fingerprint": _frame_fingerprint(
                dataset[[unit_id_column, treatment_column, outcome_column]]
            ),
            "extracted_fit_fingerprint": _frame_fingerprint(extracted_fit),
            "extracted_heldout_fingerprint": _frame_fingerprint(extracted_heldout),
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
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    heldout_ids = [int(value) for value in split["heldout_row_ids"]]
    binary = str(outcome_type) == "binary"
    fit_data = dataset.iloc[fit_ids]
    heldout_data = dataset.iloc[heldout_ids]
    t_fit = fit_data[treatment_column].to_numpy(dtype=float)
    y_fit = fit_data[outcome_column].to_numpy(dtype=float)
    t_heldout = heldout_data[treatment_column].to_numpy(dtype=float)
    y_heldout = heldout_data[outcome_column].to_numpy(dtype=float)

    adjustment_defs = _definitions_for_roles(definitions, {"confounder"})
    propensity_defs = _definitions_for_nuisance_role(definitions, "treatment")
    outcome_defs = _definitions_for_nuisance_role(definitions, "outcome")
    effect_defs = _definitions_for_roles(definitions, {"effect_modifier"})
    effect_ids = {str(feature["feature_id"]) for feature in effect_defs}
    pure_confounder_defs = [
        feature
        for feature in adjustment_defs
        if str(feature["feature_id"]) not in effect_ids
    ]
    # Eligibility for effect fitting must not use in-sample propensity predictions.
    supplied_inner = list(split.get("inner_splits") or []) or _fallback_inner_splits(
        fit_ids, folds=inner_folds, seed=seed
    )
    fit_propensity, fit_mu0, fit_mu1 = _cross_fitted_nuisance(
        dataset=dataset,
        extracted=extracted_fit,
        definitions=definitions,
        fit_ids=fit_ids,
        inner_splits=supplied_inner,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        binary=binary,
        seed=seed + 30_000,
    )
    fit_keep = propensity_eligibility(fit_propensity, min_propensity, max_propensity)
    _write_frame(
        output_dir / "nuisance_fit_predictions.csv",
        pd.DataFrame(
            {
                "_oci_row_id": fit_ids,
                "treatment": t_fit,
                "outcome": y_fit,
                "propensity": fit_propensity,
                "mu0": fit_mu0,
                "mu1": fit_mu1,
                "effect_eligible": fit_keep,
            }
        ),
    )
    fit_overlap = overlap_diagnostics(fit_propensity, min_propensity, max_propensity)
    _write_json(output_dir / "propensity_overlap.json", {"fit": fit_overlap})
    if fit_keep.sum() < 4 or min(np.sum(t_fit[fit_keep] == arm) for arm in (0, 1)) < 2:
        raise ValueError(
            "propensity bounds leave insufficient training rows or treatment arms for causal forest"
        )
    t_encoder = _FeatureEncoder(propensity_defs).fit(extracted_fit)
    y_encoder = _FeatureEncoder(outcome_defs).fit(extracted_fit)
    eligible_fit = extracted_fit.iloc[np.flatnonzero(fit_keep)]
    effect_encoder = _FeatureEncoder(effect_defs).fit(eligible_fit)
    control_encoder = _FeatureEncoder(pure_confounder_defs).fit(eligible_fit)
    x_t_fit = t_encoder.transform(extracted_fit)
    x_t_heldout = t_encoder.transform(extracted_heldout)
    x_y_fit = y_encoder.transform(extracted_fit)
    x_y_heldout = y_encoder.transform(extracted_heldout)
    x_effect_fit = effect_encoder.transform(extracted_fit)
    x_effect_heldout = effect_encoder.transform(extracted_heldout)
    w_fit = control_encoder.transform(extracted_fit)
    w_heldout = control_encoder.transform(extracted_heldout)
    treatment_model = _fit_classifier(
        x_t_fit,
        t_fit,
        seed=seed + 10_000,
    )
    outcome_models = _fit_outcome_models(
        x_y_fit,
        t_fit,
        y_fit,
        binary=binary,
        seed=seed + 10_000,
    )
    propensity = _predict_probability(treatment_model, x_t_heldout)
    mu0, mu1 = _predict_outcomes(outcome_models, x_y_heldout)
    heldout_keep = propensity_eligibility(propensity, min_propensity, max_propensity)
    overlap = {
        "fit": fit_overlap,
        "heldout": overlap_diagnostics(propensity, min_propensity, max_propensity),
        "fit_eligibility_source": "inner_out_of_fold_external_propensity",
        "heldout_eligibility_source": "outer_training_only_external_propensity",
    }
    _write_json(output_dir / "propensity_overlap.json", overlap)
    if not heldout_keep.any():
        raise ValueError("propensity bounds leave no held-out patients for effect estimation")
    if x_effect_fit.shape[1] == 0:
        # EconML requires X for heterogeneous-effect prediction. A constant X
        # yields one fold-level treatment effect when no modifier survived.
        x_effect_fit = np.ones((len(extracted_fit), 1), dtype=float)
        x_effect_heldout = np.ones((len(extracted_heldout), 1), dtype=float)
        constant_effect_design = True
    else:
        constant_effect_design = False
    controls_fit = w_fit[fit_keep] if w_fit.shape[1] else None
    causal_forest = CausalForestHead(
        n_estimators=int(estimation_trees),
        max_depth=None,
        min_samples_leaf=10,
        max_features="sqrt",
        honest=True,
        inference=True,
        random_state=seed + 20_000,
        tune_model=False,
        subforest_size=next(
            size for size in (4, 3, 2, 1) if int(estimation_trees) % size == 0
        ),
        n_jobs=1,
        outcome_type=outcome_type,
    )
    causal_forest.fit(
        x_effect_fit[fit_keep],
        t_fit[fit_keep],
        y_fit[fit_keep],
        W=controls_fit,
    )
    causal_forest_predictions = causal_forest.predict(
        x_effect_heldout[heldout_keep],
        return_ci=True,
    )

    def scatter_prediction(key: str) -> np.ndarray:
        values = np.full(len(heldout_ids), np.nan)
        prediction = causal_forest_predictions.get(key)
        if prediction is not None:
            values[heldout_keep] = np.asarray(prediction, dtype=float).reshape(-1)
        return values

    cate = scatter_prediction("tau_pred")
    cate_lower_values = scatter_prediction("tau_lower")
    cate_upper_values = scatter_prediction("tau_upper")
    cate_std_values = scatter_prediction("tau_std")
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
            "estimated_cate": cate,
            "estimated_cate_lower_95": cate_lower_values,
            "estimated_cate_upper_95": cate_upper_values,
            "estimated_cate_standard_error": cate_std_values,
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
    diagnostics = {
        "model_family": "causal_forest_dml",
        "primary_ate_estimator": "outer_cross_fitted_aipw",
        "nuisance_model_family": "elastic_net",
        "binary_nuisance_model": ("oci.models.elastic_net_nuisance.ElasticNetLogisticClassifier"),
        "continuous_nuisance_model": ("oci.models.elastic_net_nuisance.ElasticNetRegressor"),
        "causal_forest_trees": int(estimation_trees),
        "causal_forest_honest": True,
        "causal_forest_inference": True,
        "causal_forest_tuned": False,
        "causal_forest_fit_audit": causal_forest.fit_audit(),
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
        "nuisance_calibration": {
            "fit_out_of_fold_treatment": calibration_diagnostics(
                t_fit, fit_propensity, binary=True
            ),
            "fit_out_of_fold_outcome_factual": calibration_diagnostics(
                y_fit, np.where(t_fit == 1, fit_mu1, fit_mu0), binary=binary
            ),
            "heldout_treatment": calibration_diagnostics(t_heldout, propensity, binary=True),
            "heldout_outcome_factual": calibration_diagnostics(
                y_heldout, np.where(t_heldout == 1, mu1, mu0), binary=binary
            ),
            "heldout_outcome_by_arm": {
                str(arm): calibration_diagnostics(
                    y_heldout[t_heldout == arm],
                    (mu1 if arm else mu0)[t_heldout == arm],
                    binary=binary,
                )
                for arm in (0, 1)
            },
            "eligible_heldout_treatment": calibration_diagnostics(
                t_heldout[heldout_keep], propensity[heldout_keep], binary=True
            ),
            "eligible_heldout_outcome_factual": calibration_diagnostics(
                y_heldout[heldout_keep],
                np.where(t_heldout == 1, mu1, mu0)[heldout_keep],
                binary=binary,
            ),
        },
        "features": len(definitions),
        "confounders": len(adjustment_defs),
        "treatment_nuisance_features": len(propensity_defs),
        "outcome_nuisance_features": len(outcome_defs),
        "effect_modifiers": len(effect_defs),
        "pure_confounders_in_w": len(pure_confounder_defs),
        "dual_role_features_in_x_only": len(
            [feature for feature in adjustment_defs if str(feature["feature_id"]) in effect_ids]
        ),
        "constant_effect_design": constant_effect_design,
        "ate_aipw": ate,
        "standard_error": standard_error,
        "confidence_interval_95": (
            [ate - 1.96 * standard_error, ate + 1.96 * standard_error]
            if standard_error is not None
            else None
        ),
        "mean_estimated_cate": float(np.nanmean(cate)) if np.isfinite(cate).any() else None,
        "mean_causal_forest_effect": float(np.nanmean(cate)) if np.isfinite(cate).any() else None,
        "mean_causal_forest_lower_95": (
            float(np.nanmean(cate_lower_values)) if np.isfinite(cate_lower_values).any() else None
        ),
        "mean_causal_forest_upper_95": (
            float(np.nanmean(cate_upper_values)) if np.isfinite(cate_upper_values).any() else None
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
            "outcome_model_contract": diagnostics["causal_forest_fit_audit"][
                "outcome_model_contract"
            ],
            "completed_at": _now(),
            "rows": len(heldout_ids),
        },
    )
    return diagnostics


def _run_fold_analysis_legacy(
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
    config: Any,
) -> dict[str, Any]:
    """Run bounded training-fold review and held-out causal estimation."""

    (
        ontology_refinement_min_failure_patients,
        max_ontology_refinement_rounds,
    ) = _ontology_refinement_limits(config)
    extraction_feature_batch_size = _configured_extraction_feature_batch_size(config)
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    heldout_ids = [int(value) for value in split["heldout_row_ids"]]
    current: list[dict[str, Any]] = []
    for raw_feature in definitions:
        feature = dict(raw_feature)
        value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
        if value_type in {"binary", "categorical", "ordinal"}:
            feature["categories_or_unit"] = _validated_closed_category_values(
                value_type=value_type,
                values=feature.get("categories_or_unit") or [],
                source=f"feature {feature.get('name')!r}",
            )
        current.append(_normalized_feature_modeling_definition(feature))
    final_fit_extraction: pd.DataFrame | None = None
    final_fit_definitions: list[dict[str, Any]] | None = None
    agent_review_rounds = 0
    evaluation_rounds = 0
    ontology_refinement_rounds = 0
    selection_history: dict[str, list[dict[str, Any]]] = {}
    review_converged = False
    final_round_definitions_changed: bool | None = None
    final_round_selection_complete: bool | None = None
    screening_trees = int(getattr(config, "screening_trees", DEFAULT_SCREENING_TREES))
    selection_policy = _stability_selection_policy(config)
    maximum_evaluation_rounds = int(
        getattr(
            config,
            "max_evaluation_rounds",
            DEFAULT_MAX_EVALUATION_ROUNDS,
        )
    )
    for round_index in range(1, maximum_evaluation_rounds + 1):
        evaluation_rounds = round_index
        round_dir = output_dir / "review" / f"round_{round_index:03d}"
        _write_json(round_dir / "definitions.json", {"features": current})
        extracted, current, feedback_rounds = _extract_training_with_ontology_feedback(
            dataset=dataset,
            row_ids=fit_ids,
            text_column=text_column,
            definitions=current,
            output_dir=round_dir / "extraction",
            feedback_dir=round_dir / "ontology_refinement",
            request_json=request_json,
            workers=config.workers,
            max_prompt_chars=config.extraction_max_prompt_chars,
            feature_batch_size=extraction_feature_batch_size,
            minimum_failure_patients=ontology_refinement_min_failure_patients,
            max_refinement_rounds=max_ontology_refinement_rounds,
        )
        ontology_refinement_rounds += feedback_rounds
        current = [_normalized_feature_modeling_definition(feature) for feature in current]
        _write_json(
            round_dir / "definitions_after_ontology_refinement.json",
            {"features": current, "ontology_refinement_rounds": feedback_rounds},
        )
        extracted, current, harmonization = _harmonize_training_extraction(
            extracted=extracted,
            definitions=current,
            output_dir=round_dir / "harmonization",
            request_json=request_json,
            max_prompt_chars=config.max_prompt_chars,
        )
        _write_json(
            round_dir / "definitions_after_harmonization.json",
            {"features": current, "harmonization": harmonization},
        )
        extraction_definitions = [dict(feature) for feature in current]
        summaries = feature_summaries(extracted, current)
        performance = evaluate_definitions(
            dataset=dataset,
            extracted=extracted,
            definitions=current,
            split=split,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=outcome_type,
            inner_folds=inner_folds,
            seed=seed + 1_000 * round_index,
            propensity_clip=config.propensity_clip,
            forest_trees=screening_trees,
        )
        performance["stability_selection"] = _update_stability_selection(
            definitions=current,
            performance=performance,
            history=selection_history,
            evaluation_round=round_index,
            config=config,
        )
        _write_json(round_dir / "extraction_summary.json", summaries)
        _write_json(round_dir / "performance.json", performance)
        review_path = round_dir / "review.json"
        complete_path = round_dir / "complete.json"
        review_performed = bool(current) and agent_review_rounds < int(config.max_review_rounds)
        review_input_fingerprint: str | None = None
        if review_performed:
            agent_review_rounds += 1
            allow_revision = agent_review_rounds < int(config.max_review_rounds)
            review_input_fingerprint = _value_fingerprint(
                {
                    "schema_version": REVIEW_CHECKPOINT_SCHEMA_VERSION,
                    "clinical_question": clinical_question,
                    "definitions": current,
                    "summaries": summaries,
                    "performance": performance,
                    "allow_measurement_revision": allow_revision,
                    "minimum_nonmissing_fraction": config.min_nonmissing_fraction,
                    "maximum_dominant_value_fraction": config.max_dominant_fraction,
                }
            )
            review: dict[str, Any] | None = None
            if complete_path.is_file() and review_path.is_file():
                try:
                    completion = json.loads(complete_path.read_text(encoding="utf-8"))
                    if (
                        completion.get("review_schema_version") == REVIEW_CHECKPOINT_SCHEMA_VERSION
                        and completion.get("review_input_fingerprint") == review_input_fingerprint
                    ):
                        review = _validate_review(
                            json.loads(review_path.read_text(encoding="utf-8")),
                            definitions=current,
                            allow_measurement_revision=allow_revision,
                            summaries=summaries,
                        )
                except (
                    AttributeError,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    review = None
            if review is None:
                review = _request_partitioned_review(
                    clinical_question=clinical_question,
                    definitions=current,
                    summaries=summaries,
                    performance=performance,
                    allow_measurement_revision=allow_revision,
                    min_nonmissing_fraction=config.min_nonmissing_fraction,
                    max_dominant_fraction=config.max_dominant_fraction,
                    max_prompt_chars=config.max_prompt_chars,
                    output_dir=round_dir / "review_batches",
                    request_json=request_json,
                )
                _write_json(review_path, review)
            protected_drop_ids, review_drop_guard = _review_drop_stability_guards(
                current,
                review,
                performance["stability_selection"],
            )
            _write_json(
                round_dir / "review_drop_stability_guard.json",
                review_drop_guard,
            )
            reviewed, _measurement_changed = _apply_review(
                current,
                review,
                protected_drop_feature_ids=protected_drop_ids,
            )
        else:
            review = {
                "feature_decisions": [],
                "overall_assessment": (
                    "Evaluation-only convergence round after the final allowed agent review."
                    if current
                    else "No retained features to review."
                ),
                "evaluation_only": True,
            }
            _write_json(review_path, review)
            _write_json(
                round_dir / "review_drop_stability_guard.json",
                {
                    "schema_version": "stage2_review_drop_stability_guard_v1",
                    "llm_drop_decisions": 0,
                    "drop_decisions_overridden": 0,
                    "decisions": [],
                },
            )
            reviewed = [dict(feature) for feature in current]
        representation_changed_ids = _changed_feature_representation_ids(current, reviewed)
        updated, signal_pruning = _apply_empirical_signal_pruning(
            reviewed,
            performance,
            defer_feature_ids=representation_changed_ids,
        )
        _write_json(round_dir / "signal_pruning.json", signal_pruning)
        definitions_changed = _value_fingerprint(current) != _value_fingerprint(updated)
        selection_complete = bool(signal_pruning.get("selection_complete", True))
        final_round_definitions_changed = definitions_changed
        final_round_selection_complete = selection_complete
        final_fit_extraction = extracted
        final_fit_definitions = extraction_definitions
        current = updated
        _write_json(
            complete_path,
            {
                "status": "complete",
                "completed_at": _now(),
                "evaluation_round": round_index,
                "agent_review_performed": review_performed,
                "agent_review_rounds": agent_review_rounds,
                "review_schema_version": (
                    REVIEW_CHECKPOINT_SCHEMA_VERSION if review_performed else None
                ),
                "review_input_fingerprint": review_input_fingerprint,
                "definitions_changed": definitions_changed,
                "selection_complete": selection_complete,
                "features_retained": len(current),
            },
        )
        if not definitions_changed and selection_complete:
            review_converged = True
            break
    pending_conditions = []
    if final_round_definitions_changed:
        pending_conditions.append("definitions_changed_in_final_round")
    if final_round_selection_complete is False:
        pending_conditions.append("stability_selection_incomplete")
    review_convergence = {
        "schema_version": REVIEW_CONVERGENCE_SCHEMA_VERSION,
        "status": "converged" if review_converged else "non_converged",
        "converged": review_converged,
        "recorded_at": _now(),
        "evaluation_rounds": evaluation_rounds,
        "max_evaluation_rounds": maximum_evaluation_rounds,
        "definitions_changed_in_final_round": final_round_definitions_changed,
        "selection_complete_in_final_round": final_round_selection_complete,
        "features_retained_at_review_exit": len(current),
        "reason": None if review_converged else "max_evaluation_rounds_reached",
        "pending_conditions": pending_conditions,
        "continued_after_non_convergence": not review_converged,
        "continuation_policy": (
            "use_converged_definitions"
            if review_converged
            else "use_latest_retained_definitions"
        ),
    }
    _write_json(output_dir / "review" / "convergence.json", review_convergence)
    if not review_converged:
        LOGGER.warning(
            "Stage 2 empirical feature pruning did not converge within "
            "max_evaluation_rounds=%s; continuing with the latest retained "
            "definitions and flagging the fold (pending=%s)",
            maximum_evaluation_rounds,
            pending_conditions or ["convergence_criteria_not_met"],
        )

    if final_fit_extraction is None or final_fit_definitions is None:
        raise RuntimeError("Stage 2 review did not produce a training-fold extraction")
    names = [str(feature["name"]) for feature in current]
    if _value_fingerprint(final_fit_definitions) == _value_fingerprint(current):
        fit_selected = final_fit_extraction[["_oci_row_id", *names]].copy()
        _write_frame(output_dir / "extraction" / "fit" / "extracted.csv", fit_selected)
        _write_frame(output_dir / "extraction" / "fit" / "harmonized.csv", fit_selected)
        _write_json(
            output_dir / "extraction" / "fit" / "complete.json",
            {
                "status": "complete",
                "completed_at": _now(),
                "rows": len(fit_selected),
                "features": len(current),
                "reused_from_evaluation_round": evaluation_rounds,
            },
        )
        fit_harmonization = {
            "schema_version": "stage2_applied_value_harmonization_v1",
            "scope": "outer_training_final",
            "rows": int(len(fit_selected)),
            "features_harmonized": sum(
                isinstance(feature.get("harmonization_plan"), Mapping) for feature in current
            ),
            "reused_already_harmonized_from_evaluation_round": evaluation_rounds,
        }
        _write_json(
            output_dir / "extraction" / "fit" / "harmonization.json",
            fit_harmonization,
        )
    else:
        LOGGER.info(
            "Stage 2 final feature set changed during review; re-extracting %s "
            "training rows against the %s retained definition(s)",
            len(fit_ids),
            len(current),
        )
        fit_selected, current, feedback_rounds = _extract_training_with_ontology_feedback(
            dataset=dataset,
            row_ids=fit_ids,
            text_column=text_column,
            definitions=current,
            output_dir=output_dir / "extraction" / "fit",
            feedback_dir=output_dir / "extraction" / "fit_ontology_refinement",
            request_json=request_json,
            workers=config.workers,
            max_prompt_chars=config.extraction_max_prompt_chars,
            feature_batch_size=extraction_feature_batch_size,
            minimum_failure_patients=ontology_refinement_min_failure_patients,
            max_refinement_rounds=max_ontology_refinement_rounds,
        )
        ontology_refinement_rounds += feedback_rounds
        fit_selected, current, _ = _harmonize_training_extraction(
            extracted=fit_selected,
            definitions=current,
            output_dir=output_dir / "extraction" / "fit" / "harmonization",
            request_json=request_json,
            max_prompt_chars=config.max_prompt_chars,
        )
        _write_frame(
            output_dir / "extraction" / "fit" / "harmonized.csv",
            fit_selected,
        )
        names = [str(feature["name"]) for feature in current]
    harmonization_validation_fallbacks = [
        {
            "feature_id": str(feature["feature_id"]),
            "name": str(feature["name"]),
            **copy.deepcopy(feature["harmonization_fallback"]),
        }
        for feature in current
        if isinstance(feature.get("harmonization_fallback"), Mapping)
    ]
    _write_json(
        output_dir / "final_definitions.json",
        {
            "features": current,
            "review_rounds": agent_review_rounds,
            "evaluation_rounds": evaluation_rounds,
            "review_converged": review_converged,
            "review_convergence": review_convergence,
            "ontology_refinement_rounds": ontology_refinement_rounds,
            "screening_model_family": (
                "elastic_net_nuisance_plus_random_forest_effect_model"
            ),
            "screening_trees": screening_trees,
            "stability_selection_policy": selection_policy,
            "harmonization_validation_fallbacks": harmonization_validation_fallbacks,
        },
    )
    _assert_extraction_health(
        fit_selected,
        current,
        scope="training",
        minimum_row_nonmissing_fraction=config.min_nonmissing_fraction,
        audit_path=output_dir / "extraction" / "fit_health.json",
    )
    heldout_raw_extraction = extract_rows(
        dataset=dataset,
        row_ids=heldout_ids,
        text_column=text_column,
        definitions=current,
        output_dir=output_dir / "extraction" / "heldout",
        request_json=request_json,
        workers=config.workers,
        max_prompt_chars=config.extraction_max_prompt_chars,
        feature_batch_size=extraction_feature_batch_size,
    )
    heldout_extraction, heldout_harmonization = _apply_harmonization_plans(
        heldout_raw_extraction,
        current,
        scope="outer_heldout",
    )
    _write_frame(
        output_dir / "extraction" / "heldout" / "harmonized.csv",
        heldout_extraction,
    )
    _write_json(
        output_dir / "extraction" / "heldout" / "harmonization.json",
        heldout_harmonization,
    )
    _assert_extraction_health(
        heldout_extraction,
        current,
        scope="heldout",
        minimum_row_nonmissing_fraction=config.min_nonmissing_fraction,
        audit_path=output_dir / "extraction" / "heldout_health.json",
    )
    combined = pd.concat([fit_selected, heldout_extraction], ignore_index=True).sort_values(
        "_oci_row_id"
    )
    _write_frame(output_dir / "extraction" / "extracted_features.csv", combined)
    diagnostics = estimate_outer_fold(
        dataset=dataset,
        extracted_fit=fit_selected,
        extracted_heldout=heldout_extraction,
        definitions=current,
        split=split,
        unit_id_column=unit_id_column,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        outcome_type=outcome_type,
        inner_folds=inner_folds,
        seed=seed,
        propensity_clip=config.propensity_clip,
        min_propensity=getattr(config, "min_propensity", None),
        max_propensity=getattr(config, "max_propensity", None),
        estimation_trees=config.estimation_trees,
        output_dir=output_dir / "estimation",
    )
    return {
        "features": current,
        "review_rounds": agent_review_rounds,
        "evaluation_rounds": evaluation_rounds,
        "review_converged": review_converged,
        "review_convergence": review_convergence,
        "ontology_refinement_rounds": ontology_refinement_rounds,
        "screening_model_family": (
            "elastic_net_nuisance_plus_random_forest_effect_model"
        ),
        "screening_trees": screening_trees,
        "stability_selection_policy": selection_policy,
        "harmonization_validation_fallbacks": harmonization_validation_fallbacks,
        "estimation": diagnostics,
    }


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


def _run_stage2_statistical_selection(
    selector_arguments: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[Any]]:
    """Run a large fold selector outside the thread-orchestration GIL."""

    arguments = dict(selector_arguments)
    candidate_count = len(arguments.get("definitions") or [])
    if candidate_count < STATISTICAL_SELECTION_PROCESS_ISOLATION_MIN_CANDIDATES:
        return select_stage2_features_elastic_net(**arguments)
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
            select_stage2_features_elastic_net,
            **arguments,
        ).result()


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

    del clinical_question  # Deliberately excluded from extraction-supervisor prompts.
    (
        ontology_refinement_min_failure_patients,
        max_ontology_refinement_rounds,
    ) = _ontology_refinement_limits(config)
    extraction_feature_batch_size = _configured_extraction_feature_batch_size(config)
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
        current: list[dict[str, Any]] = []
        for raw_feature in definitions:
            feature = dict(raw_feature)
            value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
            if value_type in {"binary", "categorical", "ordinal"}:
                feature["categories_or_unit"] = _validated_closed_category_values(
                    value_type=value_type,
                    values=feature.get("categories_or_unit") or [],
                    source=f"feature {feature.get('name')!r}",
                )
            current.append(_normalized_feature_modeling_definition(feature))
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
        final_fit_all, current, _ = _harmonize_training_extraction(
            extracted=final_fit_raw,
            definitions=current,
            output_dir=output_dir / "extraction" / "all_candidates_fit_harmonization",
            request_json=request_json,
            max_prompt_chars=int(config.max_prompt_chars),
        )
    else:
        _write_frame(
            output_dir / "extraction" / "all_candidates_fit" / "extracted.csv",
            final_fit_all,
        )

    selection_dir = output_dir / "selection"
    legacy_selection_path = selection_dir / "statistical_selection.json"
    if legacy_selection_path.exists():
        raise RuntimeError(
            "this outer fold contains a retired Stage 2 p-value-screen checkpoint. "
            "Preserve it for audit and use guarded reselection for the group-elastic-net "
            "selection schema."
        )
    if str(getattr(config, "input_temporal_scope", "")) != TEMPORAL_SCOPE:
        raise ValueError(
            f"Stage 2 requires input_temporal_scope={TEMPORAL_SCOPE!r}; it does not "
            "perform semantic temporal filtering"
        )
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
        "selection_consolidation_llm_model": str(getattr(config, "model", "")),
        "statistical_component_schema_version": (
            statistical_policy.multi_model.public_dict()["schema_version"]
            if statistical_policy.selection_mode == "multi_model"
            else ELASTIC_NET_COMPONENT_SCHEMA_VERSION
        ),
        "statistical_selection_policy": statistical_policy.public_dict(),
        "role_adjudication_policy": config.role_adjudication.public_dict(),
        "role_adjudication_llm_model": str(getattr(config, "model", "")),
    }
    if (
        statistical_policy.selection_mode == "multi_model"
        and statistical_policy.multi_model.modifier_count.enabled
    ):
        selection_input["modifier_count_estimation_trees"] = int(config.estimation_trees)
    selection_fingerprint = _value_fingerprint(selection_input)
    selection_report_path = selection_dir / "elastic_net_selection.json"
    selected_path = selection_dir / "selected_definitions.json"
    dependencies_path = selection_dir / "measurement_definitions.json"
    latent_states_path = selection_dir / "selected_latent_states.json"
    selection_complete_path = selection_dir / "complete.json"
    selection_input_path = selection_dir / "input.json"
    selection_report: dict[str, Any] | None = None
    selected: list[dict[str, Any]] | None = None
    measurement_definitions: list[dict[str, Any]] | None = None
    latent_states: list[dict[str, Any]] | None = None
    modeling_fit_frame: pd.DataFrame | None = None
    if (
        selection_report_path.is_file()
        and selected_path.is_file()
        and dependencies_path.is_file()
        and latent_states_path.is_file()
        and selection_complete_path.is_file()
        and selection_input_path.is_file()
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
                modeling_fit_frame = materialize_selected_latents(
                    frame=final_fit_all,
                    latent_states=latent_states,
                    measurement_definitions=measurement_definitions,
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
                **(
                    {"checkpoint_dir": selection_dir / "multi_model"}
                    if statistical_policy.selection_mode == "multi_model" else {}
                ),
            }
        )
        _write_json(
            selection_dir / "statistical_evidence.json",
            elastic_net_report,
        )
        nuisance_rows = elastic_net_report.get("cross_fitted_nuisance_models", {}).get(
            "predictions", []
        )
        if nuisance_rows:
            _write_frame(selection_dir / "nuisance_predictions.csv", pd.DataFrame(nuisance_rows))
        if consolidated_definitions and config.role_adjudication.enabled:
            if statistical_policy.selection_mode == "multi_model":
                elastic_net_report["adjudication_model_identity"] = str(getattr(config, "model", ""))
            selected, role_adjudication_report, _role_evidence = (
                adjudicate_stage2_roles(
                    definitions=consolidated_definitions,
                    statistical_report=elastic_net_report,
                    request_json=request_json,
                    output_dir=selection_dir / "role_adjudication",
                    policy=config.role_adjudication,
                )
            )
        else:
            selected = statistically_selected
            role_adjudication_report = {
                "schema_version": ROLE_ADJUDICATION_COMPONENT_SCHEMA_VERSION,
                "status": (
                    "complete_no_candidates"
                    if not consolidated_definitions
                    else "disabled_by_configuration"
                ),
                "retained_feature_ids": [
                    str(feature["feature_id"]) for feature in selected
                ],
            }
        modifier_count_report = None
        if (
            statistical_policy.selection_mode == "multi_model"
            and statistical_policy.multi_model.modifier_count.enabled
            and consolidated_definitions
        ):
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
            )
        measurement_definitions = measurement_definitions_for_selected(
            selected,
            current,
        )
        latent_states = latent_states_for_selected(selected, all_latent_entries)
        selection_report = {
            **elastic_net_report,
            "schema_version": STAGE2_ROLE_SELECTION_SCHEMA_VERSION,
            "elastic_net_component_schema_version": ELASTIC_NET_COMPONENT_SCHEMA_VERSION,
            "statistical_component_schema_version": elastic_net_report.get(
                "schema_version", ELASTIC_NET_COMPONENT_SCHEMA_VERSION
            ),
            "statistical_provisional_decisions": elastic_net_report.get(
                "decisions", []
            ),
            "role_adjudication_component_schema_version": (
                ROLE_ADJUDICATION_COMPONENT_SCHEMA_VERSION
            ),
            "role_adjudication": role_adjudication_report,
            "final_role_assignment": (
                "llm_confounders_and_nested_r_loss_modifiers"
                if modifier_count_report is not None
                else "llm_all_evidence_adjudication"
                if consolidated_definitions and config.role_adjudication.enabled
                else "statistical_provisional_roles"
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
    fit_selected = modeling_fit_frame[["_oci_row_id", *selected_names]].copy()
    _write_frame(output_dir / "extraction" / "fit" / "extracted.csv", fit_selected)
    _write_frame(output_dir / "extraction" / "fit" / "harmonized.csv", fit_selected)
    if selected:
        _assert_extraction_health(
            fit_selected,
            selected,
            scope="training",
            minimum_row_nonmissing_fraction=float(config.min_nonmissing_fraction),
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

    heldout_raw = _extract_outer_heldout_measurements(
        dataset=dataset,
        heldout_ids=heldout_ids,
        text_column=text_column,
        measurement_definitions=measurement_definitions,
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
        measurement_definitions,
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
        latent_states=latent_states,
        measurement_definitions=measurement_definitions,
    )
    heldout_extraction = heldout_with_latents[["_oci_row_id", *selected_names]].copy()
    _write_frame(
        output_dir / "extraction" / "heldout" / "harmonized.csv",
        heldout_extraction,
    )
    if selected:
        _assert_extraction_health(
            heldout_extraction,
            selected,
            scope="heldout",
            minimum_row_nonmissing_fraction=float(config.min_nonmissing_fraction),
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
    _write_json(
        output_dir / "final_definitions.json",
        {
            "schema_version": STAGE2_ROLE_SELECTION_SCHEMA_VERSION,
            "features": selected,
            "measurement_dependencies": measurement_definitions,
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
            "screening_model_family": (
                "multi_model_evidence_with_llm_theme_adjudication"
                if statistical_policy.selection_mode == "multi_model" else
                "all_evidence_univariable_and_elastic_net_with_llm_role_adjudication"
            ),
            "final_model_family": "causal_forest_dml",
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
        output_dir=output_dir / "estimation",
    )
    return {
        "features": selected,
        "review_rounds": review_rounds,
        "evaluation_rounds": review_rounds,
        "review_converged": review_converged,
        "review_convergence": review_convergence,
        "ontology_refinement_rounds": ontology_refinement_rounds,
        "screening_model_family": (
            "multi_model_evidence_with_llm_theme_adjudication"
            if statistical_policy.selection_mode == "multi_model" else
            "all_evidence_univariable_and_elastic_net_with_llm_role_adjudication"
        ),
        "selection": selection_report,
        "measurement_dependencies": measurement_definitions,
        "harmonization_validation_fallbacks": harmonization_validation_fallbacks,
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
    "infrastructure_failure_audit_paths",
    "run_fold_analysis",
]
