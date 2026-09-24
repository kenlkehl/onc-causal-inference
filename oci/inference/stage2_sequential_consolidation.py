"""Sequential equivalence consolidation before Stage 2 selection.

The consolidation pass is deliberately unsupervised with respect to treatment
and outcome.  It walks the original candidate order once.  For every candidate
that is still active, an embedding model retrieves the nearest currently active
features, mixed-type association evidence is computed on the outer-training
rows, and an LLM either leaves the cluster unchanged or replaces disjoint
sets of empirically concordant aliases with canonical, information-preserving
measurements.  Broader composites and general/specific rollups are deliberately
outside the scope of this pass.

Latents are immediately populated on the outer-training frame and enter the
active retrieval pool.  Their source columns remain physically available for
audit and held-out reconstruction, but only the active definitions proceed to
the group elastic-net selector.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
from . import stage2_clinical_prompts as clinical_prompts

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .stage2_agentic_selection import (
    LATENT_SCHEMA_VERSION,
    Stage2AgenticSelectionConfig,
    _apply_latent_state,
    _pairwise_evidence,
    _validate_rule_expression,
    fit_latent_state,
)

SCHEMA_VERSION = "stage2_sequential_equivalent_measurement_consolidation_v5_clinical_prompts"
SELECTION_SCHEMA_VERSION = (
    "stage2_all_evidence_llm_role_selection_v1"
)
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_NEIGHBOR_COUNT = 10
DEFAULT_MINIMUM_PAIRWISE_ASSOCIATION = 0.85


class RequestJSON(Protocol):
    def __call__(
        self,
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
    ) -> dict[str, Any]: ...


EmbeddingFunction = Callable[[Sequence[str], str, str], np.ndarray]
_ENCODING_LOCK = threading.Lock()


@dataclass(frozen=True)
class Stage2SequentialConsolidationConfig:
    """Policy for the pre-selection sequential consolidation pass."""

    # Off by default for backward compatibility with direct programmatic
    # callers.  The researcher-facing example configuration enables it.
    enabled: bool = False
    neighbor_count: int = DEFAULT_NEIGHBOR_COUNT
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = "cpu"
    max_latents_per_cluster: int = 2
    minimum_pairwise_complete_rows: int = 10
    minimum_pairwise_association: float = DEFAULT_MINIMUM_PAIRWISE_ASSOCIATION
    categorical_rare_level_min_count: int = 5
    latent_min_coverage: float = 0.05
    protect_explicit_features: bool = True
    max_prompt_chars: int = 100_000
    # Optional transport-only continuation route. These fields deliberately do
    # not participate in consolidation fingerprints, allowing an operator to
    # resume already accepted pivot decisions on a compatible replacement
    # server while recording the model transition separately.
    runtime_llm_endpoint: str = ""
    runtime_llm_model: str = ""
    runtime_llm_api_key: str = "EMPTY"

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("stage2.selection_consolidation.enabled must be boolean")
        for name in (
            "neighbor_count",
            "max_latents_per_cluster",
            "minimum_pairwise_complete_rows",
            "categorical_rare_level_min_count",
            "max_prompt_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"stage2.selection_consolidation.{name} must be a positive integer"
                )
        if self.max_prompt_chars < 4_000:
            raise ValueError(
                "stage2.selection_consolidation.max_prompt_chars must be at least 4000"
            )
        if self.max_latents_per_cluster > self.neighbor_count + 1:
            raise ValueError(
                "stage2.selection_consolidation.max_latents_per_cluster cannot exceed "
                "neighbor_count + 1"
            )
        if (
            isinstance(self.latent_min_coverage, bool)
            or not isinstance(self.latent_min_coverage, (int, float))
            or not math.isfinite(float(self.latent_min_coverage))
            or not 0.0 <= float(self.latent_min_coverage) <= 1.0
        ):
            raise ValueError(
                "stage2.selection_consolidation.latent_min_coverage must be in [0, 1]"
            )
        if (
            isinstance(self.minimum_pairwise_association, bool)
            or not isinstance(self.minimum_pairwise_association, (int, float))
            or not math.isfinite(float(self.minimum_pairwise_association))
            or not 0.0 < float(self.minimum_pairwise_association) <= 1.0
        ):
            raise ValueError(
                "stage2.selection_consolidation.minimum_pairwise_association "
                "must be in (0, 1]"
            )
        if not isinstance(self.protect_explicit_features, bool):
            raise ValueError(
                "stage2.selection_consolidation.protect_explicit_features must be boolean"
            )
        if not str(self.embedding_model).strip():
            raise ValueError(
                "stage2.selection_consolidation.embedding_model must be nonempty"
            )
        if not str(self.embedding_device).strip():
            raise ValueError(
                "stage2.selection_consolidation.embedding_device must be nonempty"
            )
        endpoint = str(self.runtime_llm_endpoint).strip()
        model = str(self.runtime_llm_model).strip()
        if bool(endpoint) != bool(model):
            raise ValueError(
                "stage2.selection_consolidation.runtime_llm_endpoint and "
                "runtime_llm_model must be configured together"
            )
        if endpoint and not re.match(r"^https?://[^/]+", endpoint):
            raise ValueError(
                "stage2.selection_consolidation.runtime_llm_endpoint must be HTTP(S)"
            )

    def public_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["runtime_llm_api_key"] = "<redacted>"
        return values

    def scientific_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in (
            "runtime_llm_endpoint",
            "runtime_llm_model",
            "runtime_llm_api_key",
        ):
            values.pop(key, None)
        return values


def sequential_consolidation_config_from_mapping(
    value: Mapping[str, Any] | None,
) -> Stage2SequentialConsolidationConfig:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("stage2.selection_consolidation must be an object")
    raw = dict(value or {})
    known = set(Stage2SequentialConsolidationConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            "stage2.selection_consolidation contains unsupported fields: " f"{unknown}"
        )
    config = Stage2SequentialConsolidationConfig(**raw)
    config.validate()
    return config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "columns": list(map(str, frame.columns)),
                "dtypes": [str(dtype) for dtype in frame.dtypes],
            }
        ).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _feature_key(feature: Mapping[str, Any]) -> str:
    value = str(feature.get("feature_id") or feature.get("candidate_id") or "").strip()
    if not value:
        raise ValueError("Stage 2 consolidation feature has no feature_id")
    return value


def _safe_slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return slug[:48] or "structured_concept"


def _embedding_text(feature: Mapping[str, Any]) -> str:
    fields = [
        feature.get("display_name"),
        feature.get("name"),
        feature.get("description"),
        feature.get("measurement_definition"),
    ]
    if feature.get("derived_structured_latent"):
        fields.extend(feature.get("source_feature_names") or [])
    rendered = [str(value).strip() for value in fields if str(value or "").strip()]
    return ". ".join(dict.fromkeys(rendered))


def _encode_texts(
    texts: Sequence[str],
    model_name: str,
    device: str,
) -> np.ndarray:
    from ..models.concept_embedding_cache import load_sentence_transformer

    resolved_device = None if str(device).strip().lower() == "auto" else str(device).strip()
    with _ENCODING_LOCK:
        encoder = load_sentence_transformer(str(model_name).strip(), device=resolved_device)
        values = encoder.encode(
            list(texts),
            batch_size=min(32, len(texts)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return np.asarray(values, dtype=np.float32)


def _normalized_embeddings(
    texts: Sequence[str],
    *,
    model_name: str,
    device: str,
    embedding_function: EmbeddingFunction | None,
) -> np.ndarray:
    encode = embedding_function or _encode_texts
    values = np.asarray(encode(texts, model_name, device), dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] != len(texts) or values.shape[1] < 1:
        raise RuntimeError(
            "selection consolidation embedding model returned unexpected shape "
            f"{values.shape} for {len(texts)} texts"
        )
    if not np.isfinite(values).all():
        raise RuntimeError("selection consolidation embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise RuntimeError("selection consolidation embedding model returned a zero vector")
    return values / norms


def _association_policy(
    policy: Stage2SequentialConsolidationConfig,
) -> Stage2AgenticSelectionConfig:
    return Stage2AgenticSelectionConfig(
        minimum_pairwise_complete_rows=int(policy.minimum_pairwise_complete_rows),
        categorical_rare_level_min_count=int(policy.categorical_rare_level_min_count),
    )


def _compact_association(row: Mapping[str, Any]) -> dict[str, Any]:
    missingness = dict(row.get("missingness") or {})
    details = dict(row.get("details") or {})
    compact_details: dict[str, Any] = {}
    for key in (
        "reason",
        "numeric_pairwise_rows",
        "raw_shape",
        "inferential_shape",
        "minimum_expected_count",
        "expected_cells_below_five",
        "levels",
        "level_counts",
    ):
        if key in details:
            compact_details[key] = details[key]
    if isinstance(details.get("inferential_table"), list):
        compact_details["inferential_table"] = list(details["inferential_table"])[:60]
        compact_details["inferential_table_truncated"] = (
            len(details["inferential_table"]) > 60
        )
    return {
        "left_feature_id": str(row["left_feature_id"]),
        "right_feature_id": str(row["right_feature_id"]),
        "n_pairwise_complete": int(row.get("n_pairwise_complete") or 0),
        "evaluable": bool(row.get("evaluable")),
        "association_kind": row.get("association_kind"),
        "association": row.get("association"),
        "signed_association": row.get("signed_association"),
        "missingness_absolute_phi": missingness.get("absolute_phi"),
        "missingness_jaccard": missingness.get("jaccard"),
        "details": compact_details,
    }


def _observed_summary(frame: pd.DataFrame, definition: Mapping[str, Any]) -> dict[str, Any]:
    name = str(definition["name"])
    series = frame[name] if name in frame else pd.Series([None] * len(frame))
    observed = series.dropna()
    counts = observed.astype(str).value_counts(dropna=False).head(12)
    numeric = pd.to_numeric(observed, errors="coerce")
    summary: dict[str, Any] = {
        "rows": int(len(series)),
        "nonmissing": int(series.notna().sum()),
        "nonmissing_fraction": float(series.notna().mean()) if len(series) else 0.0,
        "distinct_observed_values": int(observed.astype(str).nunique()),
        "most_common_values": [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ],
    }
    if len(numeric) and numeric.notna().all():
        summary["numeric_range"] = {
            "minimum": float(numeric.min()),
            "median": float(numeric.median()),
            "maximum": float(numeric.max()),
        }
    return summary


def _prompt_feature(
    feature: Mapping[str, Any],
    *,
    frame: pd.DataFrame,
    cosine_similarity: float,
    protected: bool,
) -> dict[str, Any]:
    return {
        "feature_id": _feature_key(feature),
        "name": str(feature.get("name") or ""),
        "display_name": str(feature.get("display_name") or feature.get("name") or ""),
        "description": str(feature.get("description") or ""),
        "value_type": str(feature.get("value_type") or "ambiguous"),
        "categories_or_unit": list(feature.get("categories_or_unit") or []),
        "measurement_definition": str(feature.get("measurement_definition") or ""),
        "missing_value_rule": str(feature.get("missing_value_rule") or ""),
        "derived_structured_latent": bool(feature.get("derived_structured_latent")),
        "direct_source_feature_ids": list(feature.get("source_feature_ids") or []),
        "protected_explicit_feature": protected,
        "embedding_cosine_similarity_to_pivot": float(cosine_similarity),
        "observed_outer_training_summary": _observed_summary(frame, feature),
    }


def _referenced_rule_features(expression: Mapping[str, Any]) -> set[str]:
    operation = str(expression.get("op") or "")
    if operation in {"count_present", "sum", "mean", "minimum", "maximum", "coalesce"}:
        return set(map(str, expression.get("feature_ids") or []))
    if operation in {"any", "all", "count_true"}:
        return {
            str(condition["feature_id"])
            for condition in expression.get("conditions") or []
        }
    if operation == "case":
        return {
            str(item["when"]["feature_id"])
            for item in expression.get("cases") or []
        }
    return set()


def _validate_ontology_text(value: Any, *, field_name: str, maximum: int) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError(f"latent {field_name} must be nonempty")
    return rendered[:maximum]


def _validate_categories(value: Any, *, output_type: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("latent categories_or_unit must be a list")
    categories = list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    if output_type == "continuous":
        if len(categories) != 1:
            raise ValueError("continuous latent requires exactly one unit description")
    elif output_type == "binary":
        if len(categories) != 2:
            raise ValueError("binary latent requires exactly two declared categories")
    elif len(categories) < 2:
        raise ValueError("categorical or ordinal latent requires at least two categories")
    return categories


def _normalized_category_token(value: Any) -> str:
    """Normalize harmless JSON/string numeric spelling differences for audits."""

    if isinstance(value, bool):
        return "true" if value else "false"
    rendered = str(value).strip()
    try:
        numeric = float(rendered)
    except (TypeError, ValueError):
        return rendered.casefold()
    if math.isfinite(numeric):
        return f"{numeric:.15g}"
    return rendered.casefold()


def _normalized_unit(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _validate_information_preserving_rule(
    proposal: Mapping[str, Any],
    *,
    definitions_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject broader, coarser, or missingness-destroying replacements."""

    source_ids = list(map(str, proposal["source_feature_ids"]))
    source_definitions = [definitions_by_id[source_id] for source_id in source_ids]
    source_types = {
        str(definition.get("value_type") or "").strip().lower()
        for definition in source_definitions
    }
    if len(source_types) != 1:
        raise ValueError(
            "equivalence consolidation requires every source to have the same "
            f"value_type; got {sorted(source_types)}"
        )
    source_type = next(iter(source_types))
    if source_type not in {"binary", "categorical", "ordinal", "continuous"}:
        raise ValueError(
            "equivalence consolidation requires a supported, unambiguous source "
            f"value_type; got {source_type!r}"
        )
    output_type = str(proposal["output_type"])
    if output_type != source_type:
        raise ValueError(
            "equivalence consolidation cannot change information granularity: "
            f"source value_type={source_type!r}, output_type={output_type!r}"
        )

    expression = dict(proposal["expression"])
    operation = str(expression["op"])
    if operation not in {"coalesce", "case"}:
        raise ValueError(
            "equivalence consolidation permits only coalesce or a lossless category "
            "recode; aggregation and composite rules are not allowed"
        )

    output_values = list(proposal["categories_or_unit"])
    if source_type == "continuous":
        if operation != "coalesce":
            raise ValueError(
                "continuous equivalent measurements may only be coalesced; they may "
                "not be thresholded, averaged, minimized, or maximized"
            )
        output_unit = _normalized_unit(output_values[0])
        source_units = [
            _normalized_unit((definition.get("categories_or_unit") or [""])[0])
            if len(definition.get("categories_or_unit") or []) == 1
            else ""
            for definition in source_definitions
        ]
        if not output_unit or any(unit != output_unit for unit in source_units):
            raise ValueError(
                "continuous alias coalescing requires exactly the same declared unit "
                "for every source and the output"
            )
        return

    source_categories = [
        list(definition.get("categories_or_unit") or [])
        for definition in source_definitions
    ]
    if any(len(categories) < 2 for categories in source_categories):
        raise ValueError(
            "categorical alias consolidation requires at least two declared categories "
            "for every source"
        )
    output_tokens = [_normalized_category_token(value) for value in output_values]
    if len(output_tokens) != len(set(output_tokens)):
        raise ValueError("latent output categories are not distinct after normalization")

    if operation == "coalesce":
        for source_id, categories in zip(source_ids, source_categories):
            tokens = [_normalized_category_token(value) for value in categories]
            categories_match = (
                tokens == output_tokens
                if source_type == "ordinal"
                else set(tokens) == set(output_tokens)
            )
            if not categories_match:
                raise ValueError(
                    "coalesce requires identical category vocabularies and granularity; "
                    f"source {source_id!r} differs from the output. Use a lossless case "
                    "recode only for synonymous labels"
                )
        return

    if expression.get("else") is not None:
        raise ValueError(
            "an equivalence recode must use else=null so all-source missingness remains "
            "missing rather than becoming a negative or reference category"
        )
    mappings: dict[str, dict[str, str]] = {source_id: {} for source_id in source_ids}
    output_token_set = set(output_tokens)
    for case in expression.get("cases") or []:
        condition = dict(case["when"])
        operator = str(condition["operator"])
        if operator not in {"eq", "in"}:
            raise ValueError(
                "an equivalence recode permits only eq/in category mappings; presence, "
                "missingness, and numeric-threshold rules are information-losing"
            )
        source_id = str(condition["feature_id"])
        input_values = (
            list(condition.get("values") or [])
            if operator == "in"
            else [condition.get("value")]
        )
        output_value = case.get("then")
        if output_value is None:
            raise ValueError(
                "an equivalence recode cannot map a documented source category to null"
            )
        output_token = _normalized_category_token(output_value)
        if output_token not in output_token_set:
            raise ValueError(
                f"case output {output_value!r} is not a declared latent category"
            )
        source_mapping = mappings[source_id]
        for input_value in input_values:
            input_token = _normalized_category_token(input_value)
            if input_token in source_mapping:
                raise ValueError(
                    f"source {source_id!r} category {input_value!r} is mapped more than once"
                )
            source_mapping[input_token] = output_token

    reachable_output_tokens: set[str] = set()
    for source_id, categories in zip(source_ids, source_categories):
        declared_tokens = {_normalized_category_token(value) for value in categories}
        mapping = mappings[source_id]
        unknown = sorted(set(mapping) - declared_tokens)
        missing = sorted(declared_tokens - set(mapping))
        if unknown or missing:
            raise ValueError(
                "a lossless equivalence recode must map every and only declared category "
                f"for source {source_id!r}; unknown={unknown}, missing={missing}"
            )
        mapped_outputs = list(mapping.values())
        if len(mapped_outputs) != len(set(mapped_outputs)):
            raise ValueError(
                "a lossless equivalence recode must map each source category one-to-one; "
                f"source {source_id!r} collapses distinct categories"
            )
        reachable_output_tokens.update(mapped_outputs)
        if source_type in {"binary", "ordinal"} and (
            len(mapping) != len(output_tokens)
            or set(mapped_outputs) != output_token_set
        ):
            raise ValueError(
                "binary and ordinal equivalence recodes must preserve scale cardinality "
                f"and map source {source_id!r} onto every output category"
            )
    if reachable_output_tokens != output_token_set:
        unused = sorted(output_token_set - reachable_output_tokens)
        raise ValueError(
            "a lossless categorical union may contain only categories reachable from at "
            f"least one source; unused_output_categories={unused}"
        )


def _validate_observed_alias_agreement(
    proposal: Mapping[str, Any],
    *,
    frame: pd.DataFrame,
    definitions_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject aliases that disagree where multiple sources are observed.

    A coalesce expression necessarily chooses one source when two aliases are
    populated.  That choice is information preserving only when the populated
    values are equal.  Structural ontology checks alone cannot establish that
    row-level property, so validate it on the outer-training measurements before
    accepting a replacement.
    """

    source_ids = list(map(str, proposal["source_feature_ids"]))
    expression = dict(proposal["expression"])
    operation = str(expression["op"])

    observed_by_source: dict[str, pd.Series] = {}
    for source_id in source_ids:
        name = str(definitions_by_id[source_id]["name"])
        if name not in frame:
            raise ValueError(
                f"alias agreement audit is missing source column {name!r}"
            )
        observed_by_source[source_id] = frame[name]

    comparable_by_source: dict[str, pd.Series] = {}
    if operation == "coalesce":
        continuous = str(proposal["output_type"]) == "continuous"
        for source_id, values in observed_by_source.items():
            if continuous:
                # Match application semantics: malformed continuous values are
                # excluded from the coalesce and therefore are not comparable.
                comparable_by_source[source_id] = pd.to_numeric(
                    values, errors="coerce"
                )
            else:
                comparable_by_source[source_id] = values.map(
                    lambda value: (
                        None
                        if pd.isna(value)
                        else _normalized_category_token(value)
                    )
                )
    elif operation == "case":
        mappings: dict[str, dict[str, str]] = {
            source_id: {} for source_id in source_ids
        }
        for case in expression.get("cases") or []:
            condition = dict(case["when"])
            source_id = str(condition["feature_id"])
            input_values = (
                list(condition.get("values") or [])
                if str(condition["operator"]) == "in"
                else [condition.get("value")]
            )
            output_token = _normalized_category_token(case.get("then"))
            for input_value in input_values:
                # _condition_mask applies eq/in with exact string spelling, so
                # use the same lookup semantics in this preflight audit.
                mappings[source_id][str(input_value)] = output_token
        for source_id, values in observed_by_source.items():
            mapping = mappings[source_id]
            comparable = values.map(
                lambda value: (
                    None
                    if pd.isna(value)
                    else mapping.get(str(value))
                )
            )
            unmapped = values.notna() & comparable.isna()
            if bool(unmapped.any()):
                examples = (
                    values.loc[unmapped]
                    .astype(str)
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                )
                raise ValueError(
                    "lossless categorical alias recode encountered observed values "
                    f"without a mapping for source {source_id!r}: {examples}"
                )
            comparable_by_source[source_id] = comparable
    else:  # Guarded by _validate_information_preserving_rule.
        raise ValueError(f"unsupported alias agreement operation {operation!r}")

    row_identity = (
        frame["_oci_row_id"]
        if "_oci_row_id" in frame
        else pd.Series(frame.index, index=frame.index)
    )
    for left_index, left_id in enumerate(source_ids):
        left = comparable_by_source[left_id]
        for right_id in source_ids[left_index + 1 :]:
            right = comparable_by_source[right_id]
            overlap = left.notna() & right.notna()
            conflicts = overlap & ~left.eq(right)
            if not bool(conflicts.any()):
                continue
            conflict_rows = row_identity.loc[conflicts].astype(str).head(10).tolist()
            raise ValueError(
                "lossless alias consolidation found conflicting overlapping values; "
                f"pair=({left_id!r}, {right_id!r}), "
                f"conflict_count={int(conflicts.sum())}, "
                f"example_row_ids={conflict_rows}"
            )


def _validate_pairwise_association_threshold(
    proposal: Mapping[str, Any],
    *,
    frame: pd.DataFrame,
    definitions_by_id: Mapping[str, Mapping[str, Any]],
    policy: Stage2SequentialConsolidationConfig,
) -> None:
    source_ids = list(map(str, proposal["source_feature_ids"]))
    threshold = float(policy.minimum_pairwise_association)
    association_policy = _association_policy(policy)
    for left_index in range(len(source_ids)):
        for right_index in range(left_index + 1, len(source_ids)):
            left_id = source_ids[left_index]
            right_id = source_ids[right_index]
            evidence = _pairwise_evidence(
                frame,
                definitions_by_id[left_id],
                definitions_by_id[right_id],
                policy=association_policy,
            )
            association = evidence.get("association")
            if not evidence.get("evaluable") or association is None:
                raise ValueError(
                    "equivalence consolidation requires evaluable association evidence "
                    f"for every source pair; pair=({left_id!r}, {right_id!r}) is not "
                    "evaluable"
                )
            association_value = float(association)
            if not math.isfinite(association_value) or association_value < threshold:
                raise ValueError(
                    "equivalence consolidation requires every source pair to meet "
                    f"minimum_pairwise_association={threshold:.3f}; pair=({left_id!r}, "
                    f"{right_id!r}) has association={association_value:.6f}"
                )


def _latent_definition(
    spec: Mapping[str, Any],
    definitions_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    direct_sources = list(map(str, spec["source_feature_ids"]))
    dependency_ids: list[str] = []
    for source_id in direct_sources:
        source = definitions_by_id[source_id]
        dependencies = (
            list(map(str, source.get("measurement_dependency_feature_ids") or []))
            if source.get("derived_structured_latent")
            else [source_id]
        )
        for dependency_id in dependencies:
            if dependency_id not in dependency_ids:
                dependency_ids.append(dependency_id)
    source_names = [str(definitions_by_id[source_id]["name"]) for source_id in direct_sources]
    value_type = str(spec["output_type"])
    return {
        "feature_id": str(spec["latent_id"]),
        "name": str(spec["name"]),
        "display_name": str(spec["label"]),
        "description": str(spec["description"]),
        "question": "Derived deterministically from structured Stage 2 measurements.",
        "value_type": value_type,
        "modeling_strategy": "continuous" if value_type == "continuous" else "categorical",
        "categories_or_unit": list(spec["categories_or_unit"]),
        "roles": [],
        "nuisance_model_roles": [],
        "configured_explicit_feature": False,
        "derived_structured_latent": True,
        "latent_schema_version": LATENT_SCHEMA_VERSION,
        "selection_consolidation_schema_version": SCHEMA_VERSION,
        "latent_spec": copy.deepcopy(dict(spec)),
        "source_feature_ids": direct_sources,
        "source_feature_names": source_names,
        "measurement_dependency_feature_ids": dependency_ids,
        "measurement_dependency_names": [
            str(definitions_by_id[feature_id]["name"])
            for feature_id in dependency_ids
        ],
        "measurement_definition": str(spec["measurement_definition"]),
        "missing_value_rule": str(spec["missing_value_rule"]),
        "temporal_scope": "pre_index_treatment",
    }


def _validate_latent_proposal(
    value: Mapping[str, Any],
    *,
    cluster_ids: set[str],
    protected_ids: set[str],
    definitions_by_id: Mapping[str, Mapping[str, Any]],
    step_index: int,
    pivot_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("each latent proposal must be an object")
    kind = str(value.get("kind") or "")
    if kind != "categorical_rule":
        raise ValueError(
            "equivalence consolidation requires kind='categorical_rule'; fitted "
            "components and broader composite latents are not allowed"
        )
    raw_sources = value.get("source_feature_ids")
    if not isinstance(raw_sources, list):
        raise ValueError("latent source_feature_ids must be a list")
    sources = list(dict.fromkeys(map(str, raw_sources)))
    if len(sources) < 2 or not set(sources) <= cluster_ids:
        raise ValueError("latent sources must be at least two distinct current-cluster IDs")
    locked = sorted(set(sources).intersection(protected_ids))
    if locked:
        raise ValueError(f"latent cannot consume protected explicit features: {locked}")
    label = _validate_ontology_text(value.get("label"), field_name="label", maximum=160)
    description = _validate_ontology_text(
        value.get("description"), field_name="description", maximum=2_000
    )
    rationale = _validate_ontology_text(
        value.get("rationale"), field_name="rationale", maximum=2_000
    )
    measurement_definition = _validate_ontology_text(
        value.get("measurement_definition"),
        field_name="measurement_definition",
        maximum=2_000,
    )
    missing_value_rule = _validate_ontology_text(
        value.get("missing_value_rule"), field_name="missing_value_rule", maximum=1_000
    )
    output_type = str(value.get("output_type") or "").strip().lower()
    if output_type not in {"binary", "categorical", "ordinal", "continuous"}:
        raise ValueError("categorical_rule output_type is unsupported")
    categories_or_unit = _validate_categories(
        value.get("categories_or_unit"), output_type=output_type
    )
    core: dict[str, Any] = {
        "schema_version": LATENT_SCHEMA_VERSION,
        "selection_consolidation_schema_version": SCHEMA_VERSION,
        "kind": kind,
        "source_feature_ids": sources,
        "label": label,
        "description": description,
        "rationale": rationale,
        "measurement_definition": measurement_definition,
        "missing_value_rule": missing_value_rule,
        "output_type": output_type,
        "categories_or_unit": categories_or_unit,
        "creation_step": int(step_index),
        "creation_pivot_feature_id": pivot_id,
    }
    expression = _validate_rule_expression(
        value.get("expression") or {}, allowed_feature_ids=set(sources)
    )
    if _referenced_rule_features(expression) != set(sources):
        raise ValueError(
            "categorical_rule expression must reference every and only declared source"
        )
    operation = str(expression["op"])
    if operation not in {"coalesce", "case"}:
        raise ValueError(
            "equivalence consolidation permits only coalesce or a lossless category "
            "recode; aggregation and composite rules are not allowed"
        )
    core["expression"] = expression
    latent_hash = _fingerprint(core)[:16]
    core["latent_id"] = f"s2latent_seq_{latent_hash}"
    core["name"] = f"s2_latent_{_safe_slug(label)}_{latent_hash[:8]}"
    return core


def _validate_and_fit_decision(
    value: Mapping[str, Any],
    *,
    frame: pd.DataFrame,
    definitions_by_id: Mapping[str, Mapping[str, Any]],
    cluster_ids: Sequence[str],
    protected_ids: set[str],
    pivot_id: str,
    step_index: int,
    policy: Stage2SequentialConsolidationConfig,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("consolidation response must be an object")
    if value.get("action") in {"keep", "merge"}:
        value = _clinical_alias_decision(value, definitions_by_id=definitions_by_id,
            cluster_ids=cluster_ids, protected_ids=protected_ids)
    action = str(value.get("action") or "")
    if action not in {"leave_unchanged", "replace_with_latents"}:
        raise ValueError("action must be leave_unchanged or replace_with_latents")
    rationale = _validate_ontology_text(
        value.get("rationale"), field_name="decision rationale", maximum=4_000
    )
    raw_latents = value.get("latents")
    if not isinstance(raw_latents, list):
        raise ValueError("latents must be a list")
    if action == "leave_unchanged":
        if raw_latents:
            raise ValueError("leave_unchanged requires an empty latents list")
        return {"action": action, "rationale": rationale, "latents": []}
    if not 1 <= len(raw_latents) <= int(policy.max_latents_per_cluster):
        raise ValueError(
            "replace_with_latents requires between one and max_latents_per_cluster latents"
        )
    for raw in raw_latents:
        if raw.get("output_type") == "continuous":
            for key in raw.get("source_feature_ids") or []:
                if key in definitions_by_id:
                    values = frame[str(definitions_by_id[key]["name"])]
                    if (values.notna() & pd.to_numeric(values, errors="coerce").isna()).any():
                        raise ValueError("keep numerical fields separate when coalescing would discard reported thresholds or other text values")
    proposals = [
        _validate_latent_proposal(
            item,
            cluster_ids=set(cluster_ids),
            protected_ids=protected_ids,
            definitions_by_id=definitions_by_id,
            step_index=step_index,
            pivot_id=pivot_id,
        )
        for item in raw_latents
    ]
    consumed: set[str] = set()
    for proposal in proposals:
        overlap = consumed.intersection(proposal["source_feature_ids"])
        if overlap:
            raise ValueError(f"latent source sets must be disjoint; overlap={sorted(overlap)}")
        consumed.update(proposal["source_feature_ids"])
    if pivot_id not in consumed:
        raise ValueError("a replacement decision must consume the current pivot feature")
    known_names = {str(feature.get("name") or "") for feature in definitions_by_id.values()}
    proposed_names: set[str] = set()
    for proposal in proposals:
        if proposal["name"] in known_names or proposal["name"] in proposed_names:
            raise ValueError(f"latent name collision: {proposal['name']!r}")
        proposed_names.add(proposal["name"])
        _validate_information_preserving_rule(
            proposal,
            definitions_by_id=definitions_by_id,
        )
        _validate_pairwise_association_threshold(
            proposal,
            frame=frame,
            definitions_by_id=definitions_by_id,
            policy=policy,
        )
        _validate_observed_alias_agreement(
            proposal,
            frame=frame,
            definitions_by_id=definitions_by_id,
        )
        definition = _latent_definition(proposal, definitions_by_id)
        state = fit_latent_state(frame, proposal, definitions_by_id)
        populated = _apply_latent_state(frame, state, definitions_by_id)
        coverage = float(populated.notna().mean()) if len(populated) else 0.0
        if coverage < float(policy.latent_min_coverage):
            raise ValueError(
                f"latent {proposal['latent_id']} population coverage {coverage:.4f} is below "
                f"latent_min_coverage={policy.latent_min_coverage:.4f}"
            )
        observed = populated.dropna()
        if observed.astype(str).nunique() < 2:
            raise ValueError(f"latent {proposal['latent_id']} is constant on outer-training rows")
        if str(proposal["output_type"]) == "continuous":
            numeric = pd.to_numeric(observed, errors="coerce")
            if numeric.isna().any():
                raise ValueError(
                    f"continuous latent {proposal['latent_id']} produces nonnumeric values"
                )
        else:
            declared = set(map(str, proposal["categories_or_unit"]))
            unexpected = sorted(set(observed.astype(str)) - declared)
            if unexpected:
                raise ValueError(
                    f"latent {proposal['latent_id']} produces undeclared categories: "
                    f"{unexpected[:20]}"
                )
        # Constructing the definition here also verifies flattened dependency lineage.
        if not definition["measurement_dependency_feature_ids"]:
            raise ValueError("latent has no original measurement dependencies")
    return {"action": action, "rationale": rationale, "latents": proposals}


def _rule_expression_examples(cluster_ids: Sequence[str]) -> dict[str, Any]:
    first, second = map(str, cluster_ids[:2])
    return {
        "coalesce_identically_encoded_aliases": {
            "op": "coalesce",
            "feature_ids": [first, second],
        },
        "lossless_synonymous_category_union_recode": {
            "op": "case",
            "cases": [
                {
                    "when": {
                        "feature_id": first,
                        "operator": "eq",
                        "value": "Source label A",
                    },
                    "then": "Canonical label A",
                },
                {
                    "when": {
                        "feature_id": first,
                        "operator": "eq",
                        "value": "Source label B",
                    },
                    "then": "Canonical label B",
                },
                {
                    "when": {
                        "feature_id": first,
                        "operator": "eq",
                        "value": "Source-only label C",
                    },
                    "then": "Canonical label C",
                },
                {
                    "when": {
                        "feature_id": second,
                        "operator": "eq",
                        "value": "Synonymous label A",
                    },
                    "then": "Canonical label A",
                },
                {
                    "when": {
                        "feature_id": second,
                        "operator": "eq",
                        "value": "Synonymous label B",
                    },
                    "then": "Canonical label B",
                },
            ],
            "else": None,
        },
    }


def _decision_response_schema(
    cluster_ids: Sequence[str],
    *,
    max_latents_per_cluster: int,
) -> dict[str, Any]:
    allowed_ids = list(map(str, cluster_ids))
    nonempty_text = {"type": "string", "minLength": 1}
    condition = {
        "type": "object",
        "required": ["feature_id", "operator"],
        "properties": {
            "feature_id": {"type": "string", "enum": allowed_ids},
            "operator": {
                "type": "string",
                "enum": ["eq", "in"],
            },
            "value": {"type": ["string", "number", "boolean", "null"]},
            "values": {
                "type": "array",
                "minItems": 1,
                "items": {"type": ["string", "number", "boolean", "null"]},
            },
        },
        "additionalProperties": False,
    }
    expression = {
        "oneOf": [
            {
                "type": "object",
                "required": ["op", "feature_ids"],
                "properties": {
                    "op": {
                        "const": "coalesce",
                    },
                    "feature_ids": {
                        "type": "array",
                        "minItems": 2,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": allowed_ids},
                    },
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": ["op", "cases", "else"],
                "properties": {
                    "op": {"const": "case"},
                    "cases": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["when", "then"],
                            "properties": {
                                "when": {"$ref": "#/$defs/condition"},
                                "then": {"type": ["string", "number", "boolean", "null"]},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "else": {"type": "null"},
                },
                "additionalProperties": False,
            },
        ]
    }
    common_latent_properties = {
        "source_feature_ids": {
            "type": "array",
            "minItems": 2,
            "uniqueItems": True,
            "items": {"type": "string", "enum": allowed_ids},
        },
        "label": nonempty_text,
        "description": nonempty_text,
        "rationale": nonempty_text,
        "measurement_definition": nonempty_text,
        "missing_value_rule": nonempty_text,
    }
    common_required = list(common_latent_properties)
    latent = {
        "type": "object",
        "required": [
            "kind",
            *common_required,
            "output_type",
            "categories_or_unit",
            "expression",
        ],
        "properties": {
            "kind": {"const": "categorical_rule"},
            **common_latent_properties,
            "output_type": {
                "type": "string",
                "enum": ["binary", "categorical", "ordinal", "continuous"],
            },
            "categories_or_unit": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "expression": {"$ref": "#/$defs/expression"},
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["action", "rationale", "latents"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["leave_unchanged", "replace_with_latents"],
            },
            "rationale": nonempty_text,
            "latents": {
                "type": "array",
                "maxItems": int(max_latents_per_cluster),
                "items": {"$ref": "#/$defs/latent"},
            },
        },
        "oneOf": [
            {
                "properties": {
                    "action": {"const": "leave_unchanged"},
                    "latents": {"maxItems": 0},
                }
            },
            {
                "properties": {
                    "action": {"const": "replace_with_latents"},
                    "latents": {
                        "minItems": 1,
                        "maxItems": int(max_latents_per_cluster),
                    },
                }
            },
        ],
        "additionalProperties": False,
        "$defs": {
            "condition": condition,
            "expression": expression,
            "latent": latent,
        },
    }


def _decision_messages(step_input: Mapping[str, Any], *, max_latents_per_cluster: int) -> list[dict[str, str]]:
    features = step_input["features"]
    by_id = {str(f["feature_id"]): f for f in features}
    text = clinical_prompts.definitions_text(features)
    protected = [clinical_prompts.label(f) for f in features if f.get("protected")]
    if protected:
        text += "\n\nProtected clinical variables: " + "; ".join(protected)
    text += "\n\nPaired measurements\n"
    for pair in step_input["pairwise_associations"]:
        left, right = pair["left_feature_id"], pair["right_feature_id"]
        text += f"\n{clinical_prompts.label(by_id[left])} and {clinical_prompts.label(by_id[right])}\n"
        text += clinical_prompts.readable({k: v for k, v in pair.items() if k not in {"left_feature_id", "right_feature_id"}}) + "\n"
    return clinical_prompts.messages("13_post_extraction_aliases", text)


def _clinical_alias_decision(value, *, definitions_by_id, cluster_ids, protected_ids):
    if value.get("action") == "keep":
        if set(value) != {"action", "reason"}:
            raise ValueError("keep requires only action and reason")
        return {"action": "leave_unchanged", "rationale": value["reason"], "latents": []}
    if set(value) != {"action", "reason", "members", "canonical_label", "category_equivalences"}:
        raise ValueError("merge requires reason, members, canonical_label, and category_equivalences")
    if not isinstance(value["members"], list):
        raise ValueError("members must list existing clinical variables")
    names = {clinical_prompts.label(definitions_by_id[k]): k for k in cluster_ids}
    sources = [clinical_prompts.resolve_label(x, names) for x in value["members"]]
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError("a merge requires at least two distinct clinical variables")
    # Protected measurements remain untouched by this optional consolidation.
    if set(sources) & protected_ids:
        return {"action": "leave_unchanged", "rationale": "Preserve the investigator-configured measurement and its identity.", "latents": []}
    definitions = [definitions_by_id[k] for k in sources]
    canonical = str(value["canonical_label"]).strip()
    base = next((f for f in definitions if clinical_prompts.label(f).casefold() == canonical.casefold()), definitions[0])
    value_type = str(base["value_type"])
    if len({f["value_type"] for f in definitions}) != 1:
        raise ValueError("equivalent measurements need the same value type")
    recodings = value["category_equivalences"]
    if not isinstance(recodings, list):
        raise ValueError("category_equivalences must be an array")
    if value_type == "continuous" and recodings:
        raise ValueError("numerical aliases cannot use category recodings")
    mappings = {}
    for row in recodings:
        if set(row) != {"source_feature", "source_category", "canonical_category"}:
            raise ValueError("each category equivalence requires source_feature, source_category, and canonical_category")
        key = clinical_prompts.resolve_label(row["source_feature"], names)
        category = row["source_category"]
        if key not in sources or category not in definitions_by_id[key]["categories_or_unit"] or (key, category) in mappings:
            raise ValueError("a category equivalence must refer to one declared member category once")
        if not isinstance(row["canonical_category"], str) or not row["canonical_category"].strip():
            raise ValueError("canonical categories must be nonempty text")
        mappings[key, category] = row["canonical_category"].strip()
    categories = list(base.get("categories_or_unit") or [])
    expression = {"op": "coalesce", "feature_ids": sources}
    if recodings or (value_type != "continuous" and any(f["categories_or_unit"] != categories for f in definitions)):
        cases, categories = [], []
        for key, definition in zip(sources, definitions):
            for category in definition["categories_or_unit"]:
                mapped = mappings.get((key, category), category)
                if mapped not in categories:
                    categories.append(mapped)
                cases.append({"when": {"feature_id": key, "operator": "eq", "value": category}, "then": mapped})
        expression = {"op": "case", "cases": cases, "else": None}
    return {"action": "replace_with_latents", "rationale": value["reason"], "latents": [{
        "kind": "categorical_rule", "source_feature_ids": sources, "label": canonical,
        "description": base["description"], "rationale": value["reason"],
        "measurement_definition": base["measurement_definition"],
        "missing_value_rule": "Return null when every equivalent measurement is missing.",
        "output_type": value_type, "categories_or_unit": categories, "expression": expression}]}




def _protected_ids(
    definitions: Sequence[Mapping[str, Any]],
    policy: Stage2SequentialConsolidationConfig,
) -> set[str]:
    if not policy.protect_explicit_features:
        return set()
    return {
        _feature_key(feature)
        for feature in definitions
        if bool(feature.get("configured_explicit_feature"))
    }


def _apply_decision(
    *,
    decision: Mapping[str, Any],
    frame: pd.DataFrame,
    definitions_by_id: dict[str, dict[str, Any]],
    active_ids: list[str],
    embeddings_by_id: dict[str, np.ndarray],
    policy: Stage2SequentialConsolidationConfig,
    embedding_function: EmbeddingFunction | None,
    step_index: int,
    latent_entries: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    if decision["action"] == "leave_unchanged":
        return active_ids, []
    created: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for proposal in decision["latents"]:
        source_ids = list(map(str, proposal["source_feature_ids"]))
        consumed.update(source_ids)
        definition = _latent_definition(proposal, definitions_by_id)
        state = fit_latent_state(frame, proposal, definitions_by_id)
        frame[str(definition["name"])] = _apply_latent_state(
            frame, state, definitions_by_id
        )
        latent_id = _feature_key(definition)
        definitions_by_id[latent_id] = definition
        vector = _normalized_embeddings(
            [_embedding_text(definition)],
            model_name=policy.embedding_model,
            device=policy.embedding_device,
            embedding_function=embedding_function,
        )[0]
        embeddings_by_id[latent_id] = vector
        entry = {
            "latent_id": latent_id,
            "name": str(definition["name"]),
            "creation_step": int(step_index),
            "definition": copy.deepcopy(definition),
            "state": copy.deepcopy(state),
        }
        latent_entries.append(entry)
        created.append(entry)
    active_ids = [feature_id for feature_id in active_ids if feature_id not in consumed]
    active_ids.extend(str(entry["latent_id"]) for entry in created)
    return active_ids, created


def _reconstruct_completed(
    *,
    frame: pd.DataFrame,
    original_definitions: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    result = frame.copy()
    definitions_by_id = {
        _feature_key(feature): copy.deepcopy(dict(feature))
        for feature in original_definitions
    }
    entries = [copy.deepcopy(dict(item)) for item in registry.get("latents") or []]
    for entry in entries:
        definition = copy.deepcopy(dict(entry["definition"]))
        latent_id = _feature_key(definition)
        result[str(definition["name"])] = _apply_latent_state(
            result, entry["state"], definitions_by_id
        )
        definitions_by_id[latent_id] = definition
    active_ids = list(map(str, registry.get("active_feature_ids") or []))
    if not active_ids or any(feature_id not in definitions_by_id for feature_id in active_ids):
        raise ValueError("completed consolidation registry has invalid active feature IDs")
    return result, [definitions_by_id[feature_id] for feature_id in active_ids], entries


def consolidate_stage2_candidates(
    *,
    extracted_fit: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    request_json: RequestJSON,
    policy: Stage2SequentialConsolidationConfig,
    output_dir: Path,
    request_model: str,
    request_runtime_identity: Mapping[str, Any] | None = None,
    embedding_function: EmbeddingFunction | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Consolidate candidates and return frame, active definitions, report, registry entries."""

    policy.validate()
    original = [copy.deepcopy(dict(feature)) for feature in definitions]
    if not policy.enabled or len(original) < 2:
        status = "disabled" if not policy.enabled else "complete_fewer_than_two_candidates"
        early_input = {
            "schema_version": SCHEMA_VERSION,
            "frame_fingerprint": _frame_fingerprint(extracted_fit),
            "definitions": original,
            "policy": policy.scientific_dict(),
            "request_model": str(request_model),
            "terminal_status": status,
        }
        early_fingerprint = _fingerprint(early_input)
        report = {
            "schema_version": SCHEMA_VERSION,
            "input_fingerprint": early_fingerprint,
            "status": status,
            "policy": policy.scientific_dict(),
            "original_candidates": len(original),
            "active_candidates": len(original),
            "latents_created": 0,
            "components_consumed": 0,
            "steps": [],
        }
        _write_json(
            output_dir / "input.json",
            {**early_input, "input_fingerprint": early_fingerprint},
        )
        _write_json(output_dir / "report.json", report)
        return (
            extracted_fit.copy(),
            original,
            report,
            [],
        )
    feature_ids = [_feature_key(feature) for feature in original]
    feature_names = [str(feature.get("name") or "") for feature in original]
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("selection consolidation requires unique feature IDs")
    if any(not name for name in feature_names) or len(feature_names) != len(set(feature_names)):
        raise ValueError("selection consolidation requires unique nonempty feature names")
    missing_columns = sorted(set(feature_names) - set(extracted_fit.columns))
    if missing_columns:
        raise ValueError(
            f"selection consolidation frame is missing feature columns: {missing_columns}"
        )
    root_input = {
        "schema_version": SCHEMA_VERSION,
        "frame_fingerprint": _frame_fingerprint(
            extracted_fit[[column for column in extracted_fit.columns if column == "_oci_row_id" or column in feature_names]]
        ),
        "definitions": original,
        "policy": policy.scientific_dict(),
        "request_model": str(request_model),
    }
    root_fingerprint = _fingerprint(root_input)
    input_path = output_dir / "input.json"
    report_path = output_dir / "report.json"
    registry_path = output_dir / "registry.json"
    complete_path = output_dir / "complete.json"
    if all(path.is_file() for path in (input_path, report_path, registry_path, complete_path)):
        try:
            prior_input = json.loads(input_path.read_text(encoding="utf-8"))
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            if (
                prior_input.get("input_fingerprint") == root_fingerprint
                and completion.get("input_fingerprint") == root_fingerprint
                and completion.get("schema_version") == SCHEMA_VERSION
                and report.get("schema_version") == SCHEMA_VERSION
                and report.get("input_fingerprint") == root_fingerprint
                and registry.get("schema_version") == SCHEMA_VERSION
            ):
                frame, active, entries = _reconstruct_completed(
                    frame=extracted_fit,
                    original_definitions=original,
                    registry=registry,
                )
                return frame, active, dict(report), entries
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            pass
    _write_json(input_path, {**root_input, "input_fingerprint": root_fingerprint})
    if request_runtime_identity is not None:
        existing_decisions = len(
            list((output_dir / "steps").glob("step_*/decision.json"))
        )
        _write_json(
            output_dir / "runtime_route.json",
            {
                "schema_version": SCHEMA_VERSION,
                "recorded_at": _now(),
                "scientific_request_model": str(request_model),
                "effective_runtime": dict(request_runtime_identity),
                "checkpointed_decisions_before_this_route": existing_decisions,
                "runtime_route_is_excluded_from_scientific_fingerprint": True,
            },
        )

    frame = extracted_fit.copy()
    definitions_by_id: dict[str, dict[str, Any]] = {
        _feature_key(feature): feature for feature in original
    }
    active_ids = list(feature_ids)
    protected_ids = _protected_ids(original, policy)
    initial_matrix = _normalized_embeddings(
        [_embedding_text(feature) for feature in original],
        model_name=policy.embedding_model,
        device=policy.embedding_device,
        embedding_function=embedding_function,
    )
    embeddings_by_id = {
        feature_id: initial_matrix[index]
        for index, feature_id in enumerate(feature_ids)
    }
    association_policy = _association_policy(policy)
    latent_entries: list[dict[str, Any]] = []
    step_reports: list[dict[str, Any]] = []
    components_consumed: set[str] = set()

    for step_index, pivot_id in enumerate(feature_ids, start=1):
        if pivot_id not in active_ids:
            step_reports.append(
                {
                    "step": step_index,
                    "pivot_feature_id": pivot_id,
                    "status": "skipped_consumed_by_prior_latent",
                }
            )
            continue
        if pivot_id in protected_ids:
            step_reports.append(
                {
                    "step": step_index,
                    "pivot_feature_id": pivot_id,
                    "status": "left_unchanged_protected_explicit_feature",
                }
            )
            continue
        pivot_vector = embeddings_by_id[pivot_id]
        active_positions = {feature_id: position for position, feature_id in enumerate(active_ids)}
        ranked = sorted(
            (
                (float(np.dot(pivot_vector, embeddings_by_id[feature_id])), feature_id)
                for feature_id in active_ids
                if feature_id != pivot_id
            ),
            key=lambda item: (-item[0], active_positions[item[1]], item[1]),
        )
        neighbors = ranked[: int(policy.neighbor_count)]
        if not neighbors:
            step_reports.append(
                {
                    "step": step_index,
                    "pivot_feature_id": pivot_id,
                    "status": "left_unchanged_no_active_neighbors",
                }
            )
            continue
        cluster_ids = [pivot_id, *[feature_id for _score, feature_id in neighbors]]
        cluster = [definitions_by_id[feature_id] for feature_id in cluster_ids]
        similarities = {pivot_id: 1.0, **{feature_id: score for score, feature_id in neighbors}}
        associations = [
            _compact_association(
                _pairwise_evidence(
                    frame,
                    cluster[left],
                    cluster[right],
                    policy=association_policy,
                )
            )
            for left in range(len(cluster))
            for right in range(left + 1, len(cluster))
        ]
        step_input = {
            "schema_version": SCHEMA_VERSION,
            "step": step_index,
            "pivot_feature_id": pivot_id,
            "active_candidate_count": len(active_ids),
            "neighbor_count": len(neighbors),
            "equivalence_policy": {
                "replacement_scope": "same_measurement_aliases_only",
                "minimum_pairwise_association": float(
                    policy.minimum_pairwise_association
                ),
                "require_evaluable_association_for_every_source_pair": True,
                "require_same_value_type_and_granularity": True,
                "allow_lossless_categorical_union_recode": True,
                "continuous_coalesce_skips_nonnumeric_source_values": True,
                "require_overlapping_source_agreement": True,
                "allow_general_specific_rollups": False,
                "allow_aggregation_or_composite_rules": False,
                "preserve_all_source_missingness_as_null": True,
            },
            "features": [
                _prompt_feature(
                    feature,
                    frame=frame,
                    cosine_similarity=similarities[_feature_key(feature)],
                    protected=_feature_key(feature) in protected_ids,
                )
                for feature in cluster
            ],
            "pairwise_associations": [
                {**pair, "paired_agreement": _paired_agreement_summary(
                    frame, definitions_by_id[pair["left_feature_id"]], definitions_by_id[pair["right_feature_id"]])}
                for pair in associations
            ],
        }
        step_fingerprint = _fingerprint(
            {
                "root_input_fingerprint": root_fingerprint,
                "prompt_version": clinical_prompts.PROMPT_VERSION,
                "active_feature_ids": active_ids,
                "step_input": step_input,
            }
        )
        step_dir = output_dir / "steps" / f"step_{step_index:04d}_{_safe_slug(pivot_id)}"
        step_input_path = step_dir / "input.json"
        step_decision_path = step_dir / "decision.json"
        decision: dict[str, Any] | None = None
        if step_input_path.is_file() and step_decision_path.is_file():
            try:
                prior_step = json.loads(step_input_path.read_text(encoding="utf-8"))
                cached_decision = json.loads(step_decision_path.read_text(encoding="utf-8"))
                if prior_step.get("input_fingerprint") == step_fingerprint:
                    decision = _validate_and_fit_decision(
                        cached_decision,
                        frame=frame,
                        definitions_by_id=definitions_by_id,
                        cluster_ids=cluster_ids,
                        protected_ids=protected_ids,
                        pivot_id=pivot_id,
                        step_index=step_index,
                        policy=policy,
                    )
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                decision = None
        if decision is None:
            _write_json(
                step_input_path,
                {**step_input, "input_fingerprint": step_fingerprint},
            )
            messages = _decision_messages(
                step_input,
                max_latents_per_cluster=int(policy.max_latents_per_cluster),
            )
            prompt_chars = sum(len(message["content"]) for message in messages)
            if prompt_chars > int(policy.max_prompt_chars):
                raise ValueError(
                    f"selection consolidation step {step_index} prompt has {prompt_chars} "
                    f"characters, exceeding max_prompt_chars={policy.max_prompt_chars}"
                )
            conservative_fallback = {
                "action": "leave_unchanged",
                "rationale": (
                    "Conservative schema-validation fallback: the consolidation model "
                    "repeatedly failed to return a valid structured latent, so all "
                    "current-cluster variables remain unchanged."
                ),
                "latents": [],
            }
            repair_dir = step_dir / "repair_attempts"
            existing_attempts = list(repair_dir.glob("attempt_*.json"))
            audit_sequence = len(existing_attempts)

            def record_validation_event(event: Mapping[str, Any]) -> None:
                nonlocal audit_sequence
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "recorded_at": _now(),
                    "step": step_index,
                    "pivot_feature_id": pivot_id,
                    "allowed_feature_ids": list(cluster_ids),
                    **dict(event),
                }
                if str(event.get("event") or "") == "invalid_response":
                    audit_sequence += 1
                    destination = repair_dir / f"attempt_{audit_sequence:04d}.json"
                else:
                    destination = repair_dir / "fallback.json"
                _write_json(destination, payload)

            decision = request_json(
                messages,
                lambda value: _validate_and_fit_decision(
                    value,
                    frame=frame,
                    definitions_by_id=definitions_by_id,
                    cluster_ids=cluster_ids,
                    protected_ids=protected_ids,
                    pivot_id=pivot_id,
                    step_index=step_index,
                    policy=policy,
                ),
                request_kind="interpretation",
                repair_context={
                    "allowed_feature_ids": list(cluster_ids),
                    "valid_expression_examples": _rule_expression_examples(cluster_ids),
                    "equivalence_constraints": step_input["equivalence_policy"],
                    "conservative_response": conservative_fallback,
                },
                validation_event_observer=record_validation_event,
                conservative_validation_fallback=conservative_fallback,
                fallback_after_same_error=3,
            )
            _write_json(step_decision_path, decision)
            if decision == conservative_fallback and not (repair_dir / "fallback.json").is_file():
                _write_json(
                    repair_dir / "fallback.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "recorded_at": _now(),
                        "event": "conservative_fallback",
                        "trigger": "model_returned_instructed_conservative_response",
                        "step": step_index,
                        "pivot_feature_id": pivot_id,
                        "allowed_feature_ids": list(cluster_ids),
                        "fallback_response": decision,
                    },
                )
        before = set(active_ids)
        active_ids, created = _apply_decision(
            decision=decision,
            frame=frame,
            definitions_by_id=definitions_by_id,
            active_ids=active_ids,
            embeddings_by_id=embeddings_by_id,
            policy=policy,
            embedding_function=embedding_function,
            step_index=step_index,
            latent_entries=latent_entries,
        )
        consumed_here = sorted(before - set(active_ids))
        components_consumed.update(consumed_here)
        step_report = {
            "step": step_index,
            "pivot_feature_id": pivot_id,
            "status": "replaced" if created else "left_unchanged",
            "cluster_feature_ids": cluster_ids,
            "decision_rationale": str(decision["rationale"]),
            "consumed_feature_ids": consumed_here,
            "created_latent_ids": [str(item["latent_id"]) for item in created],
            "active_candidate_count_after_step": len(active_ids),
            "input_fingerprint": step_fingerprint,
        }
        step_reports.append(step_report)
        _write_json(
            step_dir / "complete.json",
            {
                "status": "complete",
                "schema_version": SCHEMA_VERSION,
                "completed_at": _now(),
                **step_report,
            },
        )

    active_definitions = [definitions_by_id[feature_id] for feature_id in active_ids]
    report = {
        "schema_version": SCHEMA_VERSION,
        "input_fingerprint": root_fingerprint,
        "status": "complete",
        "policy": policy.scientific_dict(),
        "treatment_and_outcome_were_unavailable": True,
        "fit_scope": "outer_training_only",
        "iteration_queue": "original_candidate_order_once",
        "new_latents_enter_later_retrievals": True,
        "original_candidates": len(original),
        "active_candidates": len(active_definitions),
        "latents_created": len(latent_entries),
        "components_consumed": len(components_consumed),
        "protected_explicit_features": sorted(protected_ids),
        "active_feature_ids": active_ids,
        "steps": step_reports,
    }
    registry = {
        "schema_version": SCHEMA_VERSION,
        "input_fingerprint": root_fingerprint,
        "original_feature_ids": feature_ids,
        "active_feature_ids": active_ids,
        "latents": latent_entries,
    }
    _write_json(report_path, report)
    _write_json(registry_path, registry)
    _write_json(
        complete_path,
        {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "completed_at": _now(),
            "input_fingerprint": root_fingerprint,
            "original_candidates": len(original),
            "active_candidates": len(active_definitions),
            "latents_created": len(latent_entries),
        },
    )
    return frame, active_definitions, report, latent_entries


def measurement_definitions_for_selected(
    selected: Sequence[Mapping[str, Any]],
    original_definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten selected latent lineage to the original extraction ontology."""

    original_by_id = {
        _feature_key(feature): copy.deepcopy(dict(feature))
        for feature in original_definitions
    }
    required: set[str] = set()
    for feature in selected:
        if feature.get("derived_structured_latent"):
            required.update(
                map(str, feature.get("measurement_dependency_feature_ids") or [])
            )
        else:
            required.add(_feature_key(feature))
    unknown = sorted(required - set(original_by_id))
    if unknown:
        raise ValueError(f"selected latents have unavailable dependencies: {unknown}")
    return [
        original_by_id[_feature_key(feature)]
        for feature in original_definitions
        if _feature_key(feature) in required
    ]


def latent_states_for_selected(
    selected: Sequence[Mapping[str, Any]],
    latent_entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return selected latent ancestors in their fitted creation order."""

    entries_by_id = {
        str(entry["latent_id"]): copy.deepcopy(dict(entry)) for entry in latent_entries
    }
    required: set[str] = set()

    def visit(feature_id: str) -> None:
        if feature_id in required or feature_id not in entries_by_id:
            return
        entry = entries_by_id[feature_id]
        definition = entry["definition"]
        for source_id in map(str, definition.get("source_feature_ids") or []):
            visit(source_id)
        required.add(feature_id)

    for feature in selected:
        visit(_feature_key(feature))
    return [
        copy.deepcopy(dict(entry))
        for entry in latent_entries
        if str(entry["latent_id"]) in required
    ]


def materialize_selected_latents(
    *,
    frame: pd.DataFrame,
    latent_states: Sequence[Mapping[str, Any]],
    measurement_definitions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Apply outer-training-fitted latent states to original held-out measurements."""

    result = frame.copy()
    definitions_by_id: dict[str, dict[str, Any]] = {
        _feature_key(feature): copy.deepcopy(dict(feature))
        for feature in measurement_definitions
    }
    for entry in latent_states:
        definition = copy.deepcopy(dict(entry["definition"]))
        latent_id = _feature_key(definition)
        missing_sources = sorted(
            set(map(str, definition.get("source_feature_ids") or []))
            - set(definitions_by_id)
        )
        if missing_sources:
            raise ValueError(
                f"cannot materialize latent {latent_id}; missing ancestors {missing_sources}"
            )
        result[str(definition["name"])] = _apply_latent_state(
            result, entry["state"], definitions_by_id
        )
        definitions_by_id[latent_id] = definition
    return result


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MINIMUM_PAIRWISE_ASSOCIATION",
    "DEFAULT_NEIGHBOR_COUNT",
    "SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "Stage2SequentialConsolidationConfig",
    "consolidate_stage2_candidates",
    "latent_states_for_selected",
    "materialize_selected_latents",
    "measurement_definitions_for_selected",
    "sequential_consolidation_config_from_mapping",
]


def _paired_agreement_summary(frame, left, right):
    a, b = frame[str(left["name"])], frame[str(right["name"])]
    keep = a.notna() & b.notna()
    paired = pd.DataFrame({"left": a[keep].astype(str), "right": b[keep].astype(str)})
    exact = int((paired["left"] == paired["right"]).sum())
    result = {"paired_patients": int(keep.sum()), "exactly_equal_pairs": exact,
              "unequal_pairs": int(keep.sum()) - exact}
    if left["value_type"] == right["value_type"] == "continuous":
        x, y = pd.to_numeric(a[keep], errors="coerce"), pd.to_numeric(b[keep], errors="coerce")
        numeric = x.notna() & y.notna()
        result.update({"numeric_pairs": int(numeric.sum()),
                       "maximum_absolute_numeric_difference": float((x[numeric] - y[numeric]).abs().max()) if numeric.any() else None})
    else:
        pairs = paired.value_counts().rename("patients").reset_index()
        result["category_pairs"] = pairs.head(60).to_dict(orient="records")
        result["category_pairs_truncated"] = len(pairs) > 60
    return result
