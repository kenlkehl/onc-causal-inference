"""Prompt-bounding policy and shared JSON helpers for Stage 2 clinical review.

``Stage2RoleAdjudicationConfig`` bounds how candidate definitions are projected
into the optional modifier concept-review prompt. The projection is an explicit
allowlist: row-level values, outer-heldout outcomes, oracle columns, and
data-generation metadata cannot enter a prompt through this interface. No LLM
assigns causal roles; Stage 2 selection is numerical (multi-model).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)
ALLOWED_ROLES = ("confounder", "effect_modifier")
# Fields of the removed LLM role adjudicator; saved configurations that still
# carry them load with a warning.
RETIRED_FIELDS = ("enabled", "max_candidates_per_request")


@dataclass(frozen=True)
class Stage2RoleAdjudicationConfig:
    """Prompt-bounding policy for clinical review of candidate definitions."""

    max_description_chars: int = 1_200
    max_measurement_definition_chars: int = 1_600
    max_categories_per_feature: int = 40

    def validate(self) -> None:
        for name in (
            "max_description_chars",
            "max_measurement_definition_chars",
            "max_categories_per_feature",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"stage2.role_adjudication.{name} must be a positive integer"
                )

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def role_adjudication_config_from_mapping(
    value: Mapping[str, Any] | None,
) -> Stage2RoleAdjudicationConfig:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("stage2.role_adjudication must be an object")
    raw = dict(value or {})
    retired = sorted(name for name in RETIRED_FIELDS if name in raw)
    if retired:
        LOGGER.warning(
            "ignoring retired stage2.role_adjudication fields: %s", ", ".join(retired)
        )
        for name in retired:
            raw.pop(name)
    known = set(Stage2RoleAdjudicationConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            "stage2.role_adjudication contains unsupported fields: " f"{unknown}"
        )
    config = Stage2RoleAdjudicationConfig(**raw)
    config.validate()
    return config


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _feature_id(feature: Mapping[str, Any]) -> str:
    return str(feature.get("feature_id") or feature["name"])


def _bounded_text(value: Any, limit: int) -> str:
    rendered = str(value or "").strip()
    return rendered[: int(limit)]


def _prompt_safe_definition(
    feature: Mapping[str, Any],
    *,
    policy: Stage2RoleAdjudicationConfig,
) -> dict[str, Any]:
    """Project one definition through an explicit prompt allowlist."""

    raw_categories = feature.get("categories_or_unit") or []
    if isinstance(raw_categories, (str, bytes)):
        raw_categories = [raw_categories]
    categories = [
        _bounded_text(value, 240)
        for value in list(raw_categories)[: int(policy.max_categories_per_feature)]
    ]
    locked = feature.get("configured_explicit_feature") is True
    configured_roles = (
        [
            role
            for role in ALLOWED_ROLES
            if role in set(map(str, feature.get("roles") or []))
        ]
        if locked
        else []
    )
    return {
        "feature_id": _feature_id(feature),
        "name": _bounded_text(feature.get("name"), 300),
        **({"clinical_label": str(feature["clinical_label"])} if feature.get("clinical_label") else {}),
        "description": _bounded_text(
            feature.get("description"), policy.max_description_chars
        ),
        "value_type": _bounded_text(feature.get("value_type"), 80),
        "categories_or_unit": categories,
        "categories_or_unit_truncated": len(list(raw_categories)) > len(categories),
        "measurement_definition": _bounded_text(
            feature.get("measurement_definition"),
            policy.max_measurement_definition_chars,
        ),
        "missing_value_rule": _bounded_text(
            feature.get("missing_value_rule"), 800
        ),
        "supporting_architectures": [
            _bounded_text(value, 160)
            for value in list(feature.get("supporting_architectures") or [])[:20]
        ],
        "evidence_axes": [
            _bounded_text(value, 160)
            for value in list(feature.get("evidence_axes") or [])[:20]
        ],
        "derived_equivalent_measurement": bool(
            feature.get("derived_structured_latent")
        ),
        "source_feature_ids": [
            _bounded_text(value, 300)
            for value in list(feature.get("source_feature_ids") or [])[:40]
        ],
        "investigator_locked": locked,
        "configured_roles": configured_roles,
    }


__all__ = [
    "ALLOWED_ROLES",
    "Stage2RoleAdjudicationConfig",
    "role_adjudication_config_from_mapping",
]
