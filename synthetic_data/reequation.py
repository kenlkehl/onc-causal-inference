# synthetic_data/reequation.py
"""Re-equation: recompute treatment and outcome over an existing cohort's oracle covariates.

Generated notes depend only on patient characteristics and never mention the
treatment or the outcome, so one corpus can serve many data-generating
processes. :func:`reequation_cohort` keeps a source cohort's text and
``true_<variable>`` columns, assigns roles from a scenario, draws new
treatment and outcome equations in Python, rescales and calibrates them with
the generator's Monte Carlo procedure, and redraws treatment and outcome.

The equations depend on the source, the scenario, and the seed, not on
``size``; with one seed, smaller subsamples are nested in larger ones and every
patient keeps its unmeasured confounders, treatment, and outcome.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import generator as g
from .config import SyntheticDataConfig
from .dgp_terms import (
    ALL_ROLES,
    CONFOUNDER,
    EFFECT_MODIFIER,
    INSTRUMENT,
    MIXED_NONLINEARITY_SHAPES,
    NONLINEARITIES,
    PROGNOSTIC,
    ROLE_GROUPS,
    canonical_category,
    category_probabilities,
    draw_main_coefficients,
    equation_nonlinear_terms,
    feature_terms,
    outcome_nonlinear_terms,
    role_group,
    role_lists,
    roles_of,
)

LOGGER = logging.getLogger(__name__)

REEQUATION_CODE_VERSION = "synthetic_reequation_v1"
MANIFEST_SCHEMA = "reequation_manifest_v1"
MANIFEST_FILENAME = "reequation_manifest.json"
SOURCE_FILES = ("dataset.parquet", "metadata.json", "generation_config.json")
OUTPUT_FILES = ("dataset.parquet", "metadata.json", "generation_config.json", MANIFEST_FILENAME)
# Same number of Monte Carlo draws as the generator's rescaling and calibration.
MONTE_CARLO_DRAWS = 10000

DGP_MODES = ("parametric", "interacting", "subgroup", "nonparametric", "composite")
EFFECT_SCALES = ("log_or", "risk_difference")

SCENARIO_KEYS = frozenset({
    "name",
    "roles",
    "role_assignment",
    "modifier_main_effects",
    "nonlinearity",
    "treatment_logit_std",
    "outcome_logit_std",
    "treatment_effect",
    "modifier_strength",
    "unmeasured_confounders",
    "unmeasured_strength",
    "target_treatment_rate",
    "target_control_outcome_rate",
    "outcome_noise_std",
    # Complex & variable-dimensionality DGP controls (all optional; defaults preserve legacy S0-S6)
    "dgp_mode",
    "treatment_nonlinearity",
    "max_treatment_interactions",
    "outcome_interactions",
    "hte_interactions",
    "pure_epistatic_hte",
    "subgroup_rules",
    "rff_components",
    "rational_terms",
    "effect_scale",
    "component_weights",
    "balance_hte_scale",
})

# Columns recomputed from the scenario's equations; every other source column
# is copied unchanged.
TREATMENT_OUTCOME_COLUMNS = ("treatment_indicator", "outcome_indicator")
BINARY_ORACLE_COLUMNS = (
    "true_treatment_prob",
    "true_outcome_prob",
    "true_y0_prob",
    "true_y1_prob",
    "true_ite_prob",
    "true_y0",
    "true_y1",
    "true_ite",
)
CONTINUOUS_ORACLE_COLUMNS = ("true_treatment_prob", "true_y0", "true_y1", "true_ite")
RECOMPUTED_COLUMNS = TREATMENT_OUTCOME_COLUMNS + BINARY_ORACLE_COLUMNS

# Any of these marks a source whose text depends on its original treatment.
POST_TREATMENT_TEXT_COLUMN = "clinical_text_with_post_treatment"
POST_TREATMENT_ORACLE_PREFIX = "true_post_treatment_"
UNMEASURED_PREFIX = "u_"

# Named random streams, keyed by (seed, stream).
_STREAMS = {
    "subsample": 1,
    "roles": 2,
    "coefficients": 3,
    "monte_carlo": 4,
    "unmeasured": 5,
    "treatment": 6,
    "outcome": 7,
}


def _rng(seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), _STREAMS[stream]])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def code_sha256() -> Dict[str, str]:
    """SHA-256 of the generator modules a re-equation runs."""

    here = Path(__file__).resolve().parent
    return {
        name: _sha256(here / name)
        for name in ("reequation.py", "generator.py", "dgp_terms.py")
    }


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------


def _number(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"scenario {label} must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"scenario {label} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"scenario {label} must be non-negative")
    return value


def _rate(value: Any, label: str) -> float:
    value = _number(value, label)
    if not 0.0 < value < 1.0:
        raise ValueError(f"scenario {label} must be between 0 and 1 (exclusive)")
    return value


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"scenario {label} must be a non-negative integer")
    return int(value)


def _count_range(value: Any, label: str) -> Tuple[int, int]:
    """Validate an exact non-negative integer or a ``{"min": lo, "max": hi}`` count range."""
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {"min", "max"})
        if unknown or "min" not in value or "max" not in value:
            raise ValueError(
                f"scenario {label} range must be an object with exact keys 'min' and 'max'"
            )
        lo = _count(value["min"], f"{label}.min")
        hi = _count(value["max"], f"{label}.max")
        if lo > hi:
            raise ValueError(f"scenario {label}.min ({lo}) cannot exceed {label}.max ({hi})")
        return lo, hi
    exact = _count(value, label)
    return exact, exact


def _resolve_count_spec(value: Any, label: str, rng: np.random.Generator) -> int:
    lo, hi = _count_range(value, label)
    if lo == hi:
        return lo
    return int(rng.integers(lo, hi + 1))


def _validate_subgroup_rules(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("scenario subgroup_rules must be a JSON object or null")
    allowed = {"treatment", "outcome", "hte", "min_depth", "max_depth", "soft_slope"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown scenario subgroup_rules keys: {unknown}")
    min_depth = _count(value.get("min_depth", 2), "subgroup_rules.min_depth")
    max_depth = _count(value.get("max_depth", 3), "subgroup_rules.max_depth")
    if min_depth < 1 or max_depth < min_depth:
        raise ValueError("scenario subgroup_rules requires 1 <= min_depth <= max_depth")
    soft_slope = value.get("soft_slope", None)
    if soft_slope is not None:
        soft_slope = _number(soft_slope, "subgroup_rules.soft_slope", positive=True)
    return {
        "treatment": _count(value.get("treatment", 0), "subgroup_rules.treatment"),
        "outcome": _count(value.get("outcome", 0), "subgroup_rules.outcome"),
        "hte": _count(value.get("hte", 0), "subgroup_rules.hte"),
        "min_depth": min_depth,
        "max_depth": max_depth,
        "soft_slope": soft_slope,
    }


def _validate_rff_components(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("scenario rff_components must be a JSON object or null")
    allowed = {"treatment", "outcome", "hte", "lengthscale", "max_terms"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown scenario rff_components keys: {unknown}")
    max_terms = _count(value.get("max_terms", 16), "rff_components.max_terms")
    if max_terms < 1:
        raise ValueError("scenario rff_components.max_terms must be at least 1")
    return {
        "treatment": _count(value.get("treatment", 0), "rff_components.treatment"),
        "outcome": _count(value.get("outcome", 0), "rff_components.outcome"),
        "hte": _count(value.get("hte", 0), "rff_components.hte"),
        "lengthscale": _number(
            value.get("lengthscale", 1.15), "rff_components.lengthscale", positive=True
        ),
        "max_terms": max_terms,
    }


def _validate_rational_terms(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("scenario rational_terms must be a JSON object or null")
    allowed = {"treatment", "outcome", "hte"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown scenario rational_terms keys: {unknown}")
    return {
        "treatment": _count(value.get("treatment", 0), "rational_terms.treatment"),
        "outcome": _count(value.get("outcome", 0), "rational_terms.outcome"),
        "hte": _count(value.get("hte", 0), "rational_terms.hte"),
    }


def _validate_component_weights(value: Any) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("scenario component_weights must be a JSON object or null")
    allowed = {"linear", "interaction", "subgroup", "rff", "rational"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown scenario component_weights keys: {unknown}")
    return {
        key: _number(value.get(key, 1.0), f"component_weights.{key}", nonnegative=True)
        for key in ("linear", "interaction", "subgroup", "rff", "rational")
    }


def _dgp_mode_defaults(dgp_mode: str) -> Dict[str, Any]:
    if dgp_mode == "interacting":
        return {
            "outcome_interactions": 4,
            "hte_interactions": 6,
            "subgroup_rules": None,
            "rff_components": None,
            "rational_terms": None,
            "component_weights": {
                "linear": 1.0, "interaction": 1.0, "subgroup": 0.0, "rff": 0.0, "rational": 0.0
            },
        }
    if dgp_mode == "subgroup":
        return {
            "outcome_interactions": 0,
            "hte_interactions": 0,
            "subgroup_rules": {
                "treatment": 3, "outcome": 3, "hte": 4, "min_depth": 2, "max_depth": 3, "soft_slope": None
            },
            "rff_components": None,
            "rational_terms": None,
            "component_weights": {
                "linear": 0.8, "interaction": 0.5, "subgroup": 1.0, "rff": 0.0, "rational": 0.0
            },
        }
    if dgp_mode == "nonparametric":
        return {
            "outcome_interactions": 0,
            "hte_interactions": 0,
            "subgroup_rules": None,
            "rff_components": {
                "treatment": 32, "outcome": 32, "hte": 32, "lengthscale": 1.1, "max_terms": 16
            },
            "rational_terms": {"treatment": 2, "outcome": 2, "hte": 2},
            "component_weights": {
                "linear": 0.4, "interaction": 0.4, "subgroup": 0.0, "rff": 1.0, "rational": 0.6
            },
        }
    if dgp_mode == "composite":
        return {
            "outcome_interactions": 4,
            "hte_interactions": 5,
            "subgroup_rules": {
                "treatment": 2, "outcome": 2, "hte": 3, "min_depth": 2, "max_depth": 3, "soft_slope": None
            },
            "rff_components": {
                "treatment": 24, "outcome": 24, "hte": 24, "lengthscale": 1.2, "max_terms": 14
            },
            "rational_terms": {"treatment": 2, "outcome": 2, "hte": 2},
            "component_weights": {
                "linear": 0.7, "interaction": 0.8, "subgroup": 0.8, "rff": 0.9, "rational": 0.6
            },
        }
    return {
        "outcome_interactions": 0,
        "hte_interactions": 0,
        "subgroup_rules": None,
        "rff_components": None,
        "rational_terms": None,
        "component_weights": None,
    }


def resolve_scenario(
    scenario: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> Dict[str, Any]:
    """Validate a scenario and fill omitted keys with the generator's defaults.

    ``roles`` maps role groups (``confounder``, ``effect_modifier``,
    ``dual_role``, ``instrument``, ``prognostic``) to a count, a
    ``{"min": lo, "max": hi}`` count range, or a list of variable names.
    With ``role_assignment: "named"`` every value must be a list; with
    ``"random"`` (the default) lists are fixed first and counts (or ranged
    counts) are drawn, with a seeded RNG, from the remaining variables in
    group order. A scenario without ``roles`` keeps the source's roles.
    Proxies take no role.
    """

    if not isinstance(scenario, Mapping):
        raise ValueError("scenario must be a JSON object")
    unknown = sorted(set(scenario) - SCENARIO_KEYS)
    if unknown:
        raise ValueError(f"unknown scenario keys: {unknown}")
    name = scenario.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("scenario 'name' is required and must be a non-empty string")
    defaults = SyntheticDataConfig()
    assignment = scenario.get("role_assignment", "random")
    if assignment not in {"random", "named"}:
        raise ValueError("scenario role_assignment must be 'random' or 'named'")

    eligible = [feature["name"] for feature in features if not feature.get("proxy_of")]
    if "roles" in scenario:
        groups = _resolve_role_groups(scenario["roles"], assignment, eligible, features, seed)
        role_source = assignment
    else:
        groups = {group: [] for group in ROLE_GROUPS}
        for feature in features:
            group = role_group(feature)
            if group is not None:
                groups[group].append(feature["name"])
        role_source = "source"

    modifier_main_effects = scenario.get("modifier_main_effects", defaults.modifier_main_effects)
    if not isinstance(modifier_main_effects, bool):
        raise ValueError("scenario modifier_main_effects must be true or false")
    nonlinearity = scenario.get("nonlinearity", defaults.nonlinearity)
    if nonlinearity not in NONLINEARITIES:
        raise ValueError(f"scenario nonlinearity must be one of {list(NONLINEARITIES)}")
    dgp_mode = scenario.get("dgp_mode", "parametric")
    if dgp_mode not in DGP_MODES:
        raise ValueError(f"scenario dgp_mode must be one of {list(DGP_MODES)}")
    treatment_nonlinearity = scenario.get("treatment_nonlinearity", "none")
    if treatment_nonlinearity not in NONLINEARITIES:
        raise ValueError(f"scenario treatment_nonlinearity must be one of {list(NONLINEARITIES)}")
    effect_scale = scenario.get("effect_scale", "log_or")
    if effect_scale not in EFFECT_SCALES:
        raise ValueError(f"scenario effect_scale must be one of {list(EFFECT_SCALES)}")
    pure_epistatic_hte = scenario.get("pure_epistatic_hte", False)
    if not isinstance(pure_epistatic_hte, bool):
        raise ValueError("scenario pure_epistatic_hte must be true or false")
    max_tx_inter = scenario.get("max_treatment_interactions", None)
    if max_tx_inter is not None:
        max_tx_inter = _count(max_tx_inter, "max_treatment_interactions")
    balance_hte_scale = scenario.get("balance_hte_scale", None)
    if balance_hte_scale is not None and not isinstance(balance_hte_scale, bool):
        raise ValueError("scenario balance_hte_scale must be true, false, or null")

    mode_defaults = _dgp_mode_defaults(dgp_mode)
    count_rng = _rng(seed, "roles") if any(
        isinstance(scenario.get(k), Mapping) for k in ("outcome_interactions", "hte_interactions")
    ) else None
    raw_out_inter = scenario.get("outcome_interactions", mode_defaults["outcome_interactions"])
    raw_hte_inter = scenario.get("hte_interactions", mode_defaults["hte_interactions"])
    outcome_interactions = (
        _resolve_count_spec(raw_out_inter, "outcome_interactions", count_rng)
        if count_rng is not None
        else _count(raw_out_inter, "outcome_interactions")
    )
    hte_interactions = (
        _resolve_count_spec(raw_hte_inter, "hte_interactions", count_rng)
        if count_rng is not None
        else _count(raw_hte_inter, "hte_interactions")
    )
    subgroup_rules = _validate_subgroup_rules(
        scenario.get("subgroup_rules", mode_defaults["subgroup_rules"])
    )
    rff_components = _validate_rff_components(
        scenario.get("rff_components", mode_defaults["rff_components"])
    )
    rational_terms = _validate_rational_terms(
        scenario.get("rational_terms", mode_defaults["rational_terms"])
    )
    component_weights = _validate_component_weights(
        scenario.get("component_weights", mode_defaults["component_weights"])
    )

    treatment_std, outcome_std = defaults.logit_stds()
    resolved: Dict[str, Any] = {
        "name": name,
        "roles": groups,
        "role_assignment": role_source,
        "modifier_main_effects": modifier_main_effects,
        "nonlinearity": nonlinearity,
        "treatment_logit_std": _number(
            scenario.get("treatment_logit_std", treatment_std), "treatment_logit_std", positive=True
        ),
        "outcome_logit_std": _number(
            scenario.get("outcome_logit_std", outcome_std), "outcome_logit_std", positive=True
        ),
        "treatment_effect": _number(
            scenario.get("treatment_effect", defaults.treatment_effect_prob), "treatment_effect"
        ),
        "modifier_strength": _number(
            scenario.get("modifier_strength", defaults.modifier_strength),
            "modifier_strength",
            nonnegative=True,
        ),
        "unmeasured_confounders": _count(
            scenario.get("unmeasured_confounders", 0), "unmeasured_confounders"
        ),
        "unmeasured_strength": _number(
            scenario.get("unmeasured_strength", 1.0), "unmeasured_strength", nonnegative=True
        ),
        "target_treatment_rate": _rate(
            scenario.get("target_treatment_rate", defaults.target_treatment_rate),
            "target_treatment_rate",
        ),
        "target_control_outcome_rate": _rate(
            scenario.get("target_control_outcome_rate", defaults.target_control_outcome_rate),
            "target_control_outcome_rate",
        ),
        "outcome_noise_std": _number(
            scenario.get("outcome_noise_std", defaults.outcome_noise_std),
            "outcome_noise_std",
            positive=True,
        ),
    }

    # Include extended DGP settings only when used so legacy S0-S6 resolved dicts stay clean.
    extended = {
        "dgp_mode": dgp_mode if dgp_mode != "parametric" or "dgp_mode" in scenario else None,
        "treatment_nonlinearity": (
            treatment_nonlinearity
            if treatment_nonlinearity != "none" or "treatment_nonlinearity" in scenario
            else None
        ),
        "max_treatment_interactions": max_tx_inter,
        "outcome_interactions": outcome_interactions if outcome_interactions or "outcome_interactions" in scenario else None,
        "hte_interactions": hte_interactions if hte_interactions or "hte_interactions" in scenario else None,
        "pure_epistatic_hte": True if pure_epistatic_hte else (False if "pure_epistatic_hte" in scenario else None),
        "subgroup_rules": subgroup_rules,
        "rff_components": rff_components,
        "rational_terms": rational_terms,
        "effect_scale": effect_scale if effect_scale != "log_or" or "effect_scale" in scenario else None,
        "component_weights": component_weights,
        "balance_hte_scale": balance_hte_scale,
    }
    for key, val in extended.items():
        if val is not None:
            resolved[key] = val
    return resolved


def _resolve_role_groups(
    roles: Any,
    assignment: str,
    eligible: Sequence[str],
    features: Sequence[Mapping[str, Any]],
    seed: int,
) -> Dict[str, List[str]]:
    if not isinstance(roles, Mapping):
        raise ValueError("scenario roles must be an object mapping role groups to counts or names")
    unknown = sorted(set(roles) - set(ROLE_GROUPS))
    if unknown:
        raise ValueError(f"unknown scenario role groups: {unknown}; use {list(ROLE_GROUPS)}")
    proxies = {feature["name"] for feature in features if feature.get("proxy_of")}
    groups: Dict[str, List[str]] = {group: [] for group in ROLE_GROUPS}
    taken: Dict[str, str] = {}
    ranges: Dict[str, Tuple[int, int]] = {}
    has_range = False
    for group in ROLE_GROUPS:
        if group not in roles:
            continue
        value = roles[group]
        if isinstance(value, list):
            for name in value:
                if not isinstance(name, str):
                    raise ValueError(f"scenario roles.{group} names must be strings")
                if name in proxies:
                    raise ValueError(f"proxy variable {name!r} cannot take a role")
                if name not in eligible:
                    raise ValueError(f"scenario roles.{group} names unknown variable {name!r}")
                if name in taken:
                    raise ValueError(
                        f"variable {name!r} is in roles.{taken[name]} and roles.{group}; "
                        "role groups are disjoint (use dual_role for both roles)"
                    )
                taken[name] = group
                groups[group].append(name)
        elif assignment == "named":
            raise ValueError(
                f"role_assignment 'named' needs a list of names for roles.{group}, got {value!r}"
            )
        else:
            lo, hi = _count_range(value, f"roles.{group}")
            ranges[group] = (lo, hi)
            if isinstance(value, Mapping):
                has_range = True
    remaining = [name for name in eligible if name not in taken]
    min_requested = sum(lo for lo, _ in ranges.values())
    if min_requested > len(remaining):
        raise ValueError(
            f"scenario roles request at least {min_requested} randomly assigned variables but only "
            f"{len(remaining)} unassigned variables remain"
        )
    rng = _rng(seed, "roles")
    counts: Dict[str, int] = {}
    if not has_range:
        counts = {group: lo for group, (lo, _) in ranges.items()}
    else:
        for group in ROLE_GROUPS:
            if group not in ranges:
                continue
            lo, hi = ranges[group]
            counts[group] = lo if lo == hi else int(rng.integers(lo, hi + 1))
        # If sampled upper-end counts exceed available features, trim only the
        # slack above each group's minimum so every group stays inside [min, max].
        excess = sum(counts.values()) - len(remaining)
        if excess > 0:
            adjustable = [group for group in ROLE_GROUPS if group in ranges and counts[group] > ranges[group][0]]
            total_slack = sum(counts[g] - ranges[g][0] for g in adjustable)
            if total_slack > 0:
                trimmed = 0
                for group in adjustable:
                    slack = counts[group] - ranges[group][0]
                    cut = min(slack, int(math.floor(excess * (slack / total_slack))))
                    counts[group] -= cut
                    trimmed += cut
                # Trim any rounding remainder one-by-one from groups with remaining slack.
                while trimmed < excess:
                    progressed = False
                    for group in adjustable:
                        if trimmed >= excess:
                            break
                        if counts[group] > ranges[group][0]:
                            counts[group] -= 1
                            trimmed += 1
                            progressed = True
                    if not progressed:
                        break
    order = [remaining[index] for index in rng.permutation(len(remaining))]
    position = 0
    for group in ROLE_GROUPS:
        for name in order[position:position + counts.get(group, 0)]:
            groups[group].append(name)
        position += counts.get(group, 0)
    # Report names in source feature order within each group.
    rank = {name: index for index, name in enumerate(eligible)}
    return {group: sorted(names, key=rank.__getitem__) for group, names in groups.items()}


def _features_with_roles(
    features: Sequence[Mapping[str, Any]], groups: Mapping[str, Sequence[str]]
) -> List[Dict[str, Any]]:
    group_of = {name: group for group, names in groups.items() for name in names}
    output = []
    for feature in features:
        feature = dict(feature)
        group = group_of.get(feature["name"])
        feature["roles"] = list(ROLE_GROUPS[group]) if group else []
        output.append(feature)
    return output


# --------------------------------------------------------------------------
# Equations
# --------------------------------------------------------------------------


def _unmeasured_features(count: int, existing: Sequence[str]) -> List[Dict[str, Any]]:
    names = [f"{UNMEASURED_PREFIX}{index}" for index in range(1, count + 1)]
    clashes = sorted(set(names) & set(existing))
    if clashes:
        raise ValueError(f"source variables {clashes} collide with unmeasured-confounder names")
    return [{"name": name, "type": "continuous", "roles": []} for name in names]


def _reference_rms(coefficients: Mapping[str, float], terms: Sequence[str], fallback: float) -> float:
    values = [coefficients[term] for term in terms if term in coefficients]
    if not values:
        return fallback
    return float(math.sqrt(sum(value * value for value in values) / len(values)))


def _term_centers_map(
    features: Sequence[Mapping[str, Any]],
    summary_stats: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    """Expected marginal value of each equation term (0.0 for standardized continuous, p_k for dummies)."""
    centers: Dict[str, float] = {}
    stats = summary_stats or {}
    for feature in features:
        if feature["type"] == "continuous":
            centers[feature["name"]] = 0.0
        else:
            probs = category_probabilities(feature, stats)
            for idx, cat in enumerate(feature["categories"][1:], start=1):
                centers[f"{feature['name']}_{cat}"] = float(probs[idx])
    return centers


def _sample_distinct_interactions(
    term_pairs_pool: Sequence[Tuple[str, str]],
    count: int,
    rng: np.random.Generator,
    *,
    term_centers: Mapping[str, float],
    scale: float,
    allow_triplets: bool = True,
    center_factors: bool = True,
) -> List[Dict[str, Any]]:
    """Sample distinct cross-variable 2-way (and optional 3-way) centered product interactions."""
    if count <= 0 or not term_pairs_pool:
        return []
    by_owner: Dict[str, List[str]] = {}
    for owner, term in term_pairs_pool:
        by_owner.setdefault(owner, []).append(term)
    owners = list(by_owner.keys())
    if len(owners) < 2:
        return []

    # Enumerate candidate 2-way cross-variable pairs when pool is moderate; sample directly when huge.
    results: List[Dict[str, Any]] = []
    seen: set = set()
    n_triplets = int(round(0.25 * count)) if (allow_triplets and count >= 4 and len(owners) >= 3) else 0
    n_pairs = count - n_triplets

    # Draw 2-way pairs
    max_attempts = max(200, n_pairs * 20)
    attempts = 0
    while len(results) < n_pairs and attempts < max_attempts:
        attempts += 1
        chosen_owners = rng.choice(len(owners), size=2, replace=False)
        o1, o2 = owners[int(chosen_owners[0])], owners[int(chosen_owners[1])]
        t1 = by_owner[o1][int(rng.integers(len(by_owner[o1])))]
        t2 = by_owner[o2][int(rng.integers(len(by_owner[o2])))]
        key = tuple(sorted((t1, t2)))
        if key in seen:
            continue
        seen.add(key)
        entry: Dict[str, Any] = {
            "terms": [t1, t2],
            "coefficient": float(rng.uniform(-1.0, 1.0)) * scale,
        }
        if center_factors:
            c1, c2 = float(term_centers.get(t1, 0.0)), float(term_centers.get(t2, 0.0))
            if c1 != 0.0 or c2 != 0.0:
                entry["term_centers"] = [c1, c2]
        results.append(entry)

    # Draw 3-way triplets (only when n_triplets > 0 and >= 3 distinct owners exist)
    if n_triplets > 0 and len(owners) >= 3:
        target_total = len(results) + n_triplets
        attempts = 0
        max_trip_attempts = max(200, n_triplets * 25)
        while len(results) < target_total and attempts < max_trip_attempts:
            attempts += 1
            chosen_owners = rng.choice(len(owners), size=3, replace=False)
            picked = [
                by_owner[owners[int(idx)]][int(rng.integers(len(by_owner[owners[int(idx)]])))]
                for idx in chosen_owners
            ]
            key = tuple(sorted(picked))
            if key in seen:
                continue
            seen.add(key)
            entry = {
                "terms": picked,
                "coefficient": float(rng.uniform(-1.0, 1.0)) * scale * 0.85,
            }
            if center_factors:
                centers = [float(term_centers.get(t, 0.0)) for t in picked]
                if any(c != 0.0 for c in centers):
                    entry["term_centers"] = centers
            results.append(entry)

    return results


def _sample_subgroup_rules(
    eligible_features: Sequence[Mapping[str, Any]],
    count: int,
    rng: np.random.Generator,
    *,
    min_depth: int,
    max_depth: int,
    soft_slope: Optional[float],
    summary_stats: Optional[Mapping[str, Any]],
    scale: float,
    treatment_only: bool,
) -> List[Dict[str, Any]]:
    """Sample multi-way conjunctive decision-tree subgroup rules over eligible features."""
    if count <= 0 or not eligible_features:
        return []
    stats = summary_stats or {}
    eff_max = min(max_depth, len(eligible_features))
    eff_min = min(min_depth, eff_max)
    rules: List[Dict[str, Any]] = []
    for _ in range(count):
        depth = eff_min if eff_min == eff_max else int(rng.integers(eff_min, eff_max + 1))
        chosen_idx = rng.choice(len(eligible_features), size=depth, replace=False)
        conditions = []
        expected_prob = 1.0
        for idx in chosen_idx:
            feat = eligible_features[int(idx)]
            if feat["type"] == "continuous":
                cut = float(rng.uniform(-0.75, 0.75))
                op = ">" if rng.random() < 0.5 else "<"
                # Standard normal tail probability approximation for centering
                cdf = 0.5 * (1.0 + math.erf(cut / math.sqrt(2.0)))
                prob = (1.0 - cdf) if op == ">" else cdf
                conditions.append({"term": feat["name"], "op": op, "cut": round(cut, 4), "slope": soft_slope})
                expected_prob *= max(0.05, min(0.95, prob))
            else:
                probs = category_probabilities(feat, stats)
                dummies = [
                    (f"{feat['name']}_{cat}", float(probs[i]))
                    for i, cat in enumerate(feat["categories"][1:], start=1)
                ]
                preferred = [d for d in dummies if d[1] >= 0.12]
                pool = preferred if preferred else dummies
                term, p_cat = pool[int(rng.integers(len(pool)))]
                op = ">" if (p_cat >= 0.25 or rng.random() < 0.5) else "<"
                prob = p_cat if op == ">" else (1.0 - p_cat)
                conditions.append({"term": term, "op": op, "cut": 0.5, "slope": soft_slope})
                expected_prob *= max(0.05, min(0.95, prob))
        gate_sd = math.sqrt(max(0.04, expected_prob * (1.0 - expected_prob)))
        coef = float(rng.uniform(-1.0, 1.0)) * (scale / gate_sd)
        rules.append({
            "conditions": conditions,
            "coefficient": coef,
            "treatment_only": bool(treatment_only),
            "center": round(expected_prob, 6),
        })
    return rules


def _sample_rff_components(
    eligible_terms: Sequence[str],
    n_rff: int,
    rng: np.random.Generator,
    *,
    lengthscale: float,
    max_terms: int,
    scale: float,
    treatment_only: bool,
) -> List[Dict[str, Any]]:
    """Sample Bochner Random Fourier Feature GP surfaces covering ``eligible_terms``."""
    if n_rff <= 0 or not eligible_terms:
        return []
    terms = list(eligible_terms)
    n_blocks = max(1, min(5, math.ceil(len(terms) / max_terms)))
    block_scale = scale / math.sqrt(n_blocks)
    components: List[Dict[str, Any]] = []
    for block_idx in range(n_blocks):
        if len(terms) <= max_terms:
            sub_terms = list(terms)
        else:
            # Deterministic overlapping coverage so every active region participates
            order = rng.permutation(len(terms))
            sub_terms = [terms[int(i)] for i in sorted(order[:max_terms])]
        d = len(sub_terms)
        # Scale frequency variance by 1/(lengthscale^2 * d) so w^T z has SD = 1/lengthscale
        # regardless of dimensionality (5..50 features).
        freq_std = 1.0 / (float(lengthscale) * math.sqrt(max(1, d)))
        weights = rng.normal(0.0, freq_std, size=(n_rff, d))
        phases = rng.uniform(0.0, 2.0 * math.pi, size=n_rff)
        basis = rng.normal(0.0, 1.0, size=n_rff)
        norm = float(np.linalg.norm(basis))
        if norm > 0:
            basis = basis * (math.sqrt(n_rff / 2.0) / norm)
        components.append({
            "terms": sub_terms,
            "weights": [[round(float(x), 6) for x in row] for row in weights],
            "phases": [round(float(p), 6) for p in phases],
            "basis_weights": [round(float(b), 6) for b in basis],
            "coefficient": float(block_scale),
            "treatment_only": bool(treatment_only),
            "center": 0.0,
        })
    return components


def _sample_rational_terms(
    eligible_terms: Sequence[str],
    count: int,
    rng: np.random.Generator,
    *,
    scale: float,
    treatment_only: bool,
) -> List[Dict[str, Any]]:
    """Sample smooth bounded rational/ratio interaction terms ``(a^T z) / (1 + sum((b_j z_j)^2))``."""
    if count <= 0 or not eligible_terms:
        return []
    terms = list(eligible_terms)
    items: List[Dict[str, Any]] = []
    per_scale = scale / math.sqrt(max(1, count))
    for _ in range(count):
        n_num = min(len(terms), int(rng.integers(2, 5)) if len(terms) >= 2 else 1)
        n_den = min(len(terms), int(rng.integers(1, 4)))
        num_idx = rng.choice(len(terms), size=n_num, replace=False)
        num_terms = [terms[int(i)] for i in sorted(num_idx)]
        remaining_idx = [i for i in range(len(terms)) if i not in set(num_idx.tolist())]
        if len(remaining_idx) >= n_den:
            den_idx = rng.choice(remaining_idx, size=n_den, replace=False)
        else:
            den_idx = rng.choice(len(terms), size=n_den, replace=False)
        den_terms = [terms[int(i)] for i in sorted(den_idx)]
        num_w = rng.uniform(-1.0, 1.0, size=n_num)
        num_norm = float(np.linalg.norm(num_w)) or 1.0
        num_w = num_w / num_norm
        den_w = rng.uniform(0.4, 1.0, size=n_den)
        den_norm = float(np.linalg.norm(den_w)) or 1.0
        den_w = den_w / den_norm
        items.append({
            "num_terms": num_terms,
            "num_weights": [round(float(w), 6) for w in num_w],
            "den_terms": den_terms,
            "den_weights": [round(float(w), 6) for w in den_w],
            "den_bias": 1.0,
            "coefficient": float(rng.uniform(-1.0, 1.0)) * per_scale * 1.4,
            "treatment_only": bool(treatment_only),
            "center": 0.0,
        })
    return items


def augment_complex_equations(
    features: Sequence[Mapping[str, Any]],
    treatment_eq: Dict[str, Any],
    outcome_eq: Dict[str, Any],
    resolved: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    summary_stats: Optional[Mapping[str, Any]] = None,
    main_coefficient_scale: float = 0.3,
    interaction_coefficient_scale: float = 0.1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Augment treatment and outcome equations with complex nonlinear, interacting, subgroup, RFF, and rational blocks.

    Strictly preserves causal role boundaries:
    - treatment-equation terms use only ``confounder`` and ``instrument`` features;
    - baseline outcome terms use only ``confounder`` and ``prognostic`` features
      (plus ``effect_modifier`` main terms only when ``modifier_main_effects`` is True);
    - HTE terms use only ``effect_modifier`` features (including ``dual_role`` variables).
    """
    confounders = [f for f in features if CONFOUNDER in roles_of(f)]
    instruments = [f for f in features if INSTRUMENT in roles_of(f)]
    prognostic = [f for f in features if PROGNOSTIC in roles_of(f)]
    modifiers = [f for f in features if EFFECT_MODIFIER in roles_of(f)]

    tx_features = confounders + instruments
    out_base_features = confounders + prognostic
    if resolved.get("modifier_main_effects"):
        mod_only = [f for f in modifiers if roles_of(f) == [EFFECT_MODIFIER]]
        out_base_features = out_base_features + mod_only

    tx_terms = [t for f in tx_features for t in feature_terms(f)]
    out_base_terms_paired = [(f["name"], t) for f in out_base_features for t in feature_terms(f)]
    out_base_terms = [t for _, t in out_base_terms_paired]
    mod_terms_paired = [(f["name"], t) for f in modifiers for t in feature_terms(f)]
    mod_terms = [t for _, t in mod_terms_paired]

    weights = resolved.get("component_weights") or {
        "linear": 1.0, "interaction": 1.0, "subgroup": 1.0, "rff": 1.0, "rational": 1.0,
    }
    w_lin = float(weights.get("linear", 1.0))
    w_int = float(weights.get("interaction", 1.0))
    w_sub = float(weights.get("subgroup", 1.0))
    w_rff = float(weights.get("rff", 1.0))
    w_rat = float(weights.get("rational", 1.0))

    # Reference scale matched to the equation's baseline linear RMS so complex blocks
    # remain well-balanced whether there are 5 or 50 active variables.
    tx_ref = max(
        main_coefficient_scale / math.sqrt(3.0),
        float(math.sqrt(sum(v * v for v in treatment_eq.get("coefficients", {}).values()))),
    )
    out_ref = max(
        main_coefficient_scale / math.sqrt(3.0),
        float(math.sqrt(sum(v * v for v in outcome_eq.get("coefficients", {}).values()))),
    )
    modifier_strength = float(resolved.get("modifier_strength", 2.0))
    hte_ref = out_ref * modifier_strength * 0.45 if modifier_strength > 0.0 else 0.0

    # Apply linear weight multiplier if requested (< 1 lets non-parametric terms dominate)
    if w_lin != 1.0:
        treatment_eq["coefficients"] = {
            k: float(v * w_lin) for k, v in treatment_eq.get("coefficients", {}).items()
        }
        outcome_eq["coefficients"] = {
            k: float(v * w_lin) for k, v in outcome_eq.get("coefficients", {}).items()
        }
        outcome_eq["treatment_interactions"] = [
            {**item, "coefficient": float(item.get("coefficient", 0.0) * w_lin)}
            for item in outcome_eq.get("treatment_interactions", ())
        ]

    centers = _term_centers_map(features, summary_stats)

    # 1. Baseline outcome interactions (C x C, C x P, P x P, optional 3-way)
    n_out_inter = int(resolved.get("outcome_interactions") or 0)
    if n_out_inter > 0 and w_int > 0.0 and len(out_base_features) >= 2:
        inter_scale = (out_ref * 0.70 * w_int) / math.sqrt(max(1, n_out_inter))
        outcome_eq.setdefault("interactions", []).extend(
            _sample_distinct_interactions(
                out_base_terms_paired,
                n_out_inter,
                rng,
                term_centers=centers,
                scale=inter_scale,
                allow_triplets=True,
                center_factors=True,
            )
        )

    # 2. Higher-order HTE interactions (M x M, Dual x M, optional 3-way)
    pure_epistatic = bool(resolved.get("pure_epistatic_hte", False))
    if pure_epistatic or modifier_strength == 0.0:
        if pure_epistatic:
            outcome_eq["treatment_interactions"] = []
    n_hte_inter = int(resolved.get("hte_interactions") or 0)
    if n_hte_inter > 0 and modifier_strength > 0.0 and w_int > 0.0 and modifiers:
        hte_inter_scale = (hte_ref * 0.85 * w_int) / math.sqrt(max(1, n_hte_inter))
        hte_items = _sample_distinct_interactions(
            mod_terms_paired,
            n_hte_inter,
            rng,
            term_centers=centers,
            scale=hte_inter_scale,
            allow_triplets=True,
            center_factors=True,
        )
        if not hte_items and len(modifiers) == 1 and modifiers[0]["type"] == "continuous":
            # Single continuous modifier fallback: centered quadratic epistasis z^2 - 1
            t0 = modifiers[0]["name"]
            hte_items = [{
                "terms": [t0, t0],
                "center": 1.0,
                "coefficient": float(rng.uniform(-1.0, 1.0)) * hte_inter_scale,
            }]
        if hte_items:
            outcome_eq.setdefault("hte_interactions", []).extend(hte_items)

    # 3. Multi-way subgroup / decision-tree conjunction rules
    sub_cfg = resolved.get("subgroup_rules")
    if sub_cfg and w_sub > 0.0:
        min_d, max_d = int(sub_cfg["min_depth"]), int(sub_cfg["max_depth"])
        slope = sub_cfg.get("soft_slope")
        if sub_cfg["treatment"] > 0 and tx_features:
            tx_rules = _sample_subgroup_rules(
                tx_features,
                int(sub_cfg["treatment"]),
                rng,
                min_depth=min_d,
                max_depth=max_d,
                soft_slope=slope,
                summary_stats=summary_stats,
                scale=(tx_ref * 0.75 * w_sub) / math.sqrt(int(sub_cfg["treatment"])),
                treatment_only=False,
            )
            if tx_rules:
                treatment_eq.setdefault("subgroup_rules", []).extend(tx_rules)
        if sub_cfg["outcome"] > 0 and out_base_features:
            out_rules = _sample_subgroup_rules(
                out_base_features,
                int(sub_cfg["outcome"]),
                rng,
                min_depth=min_d,
                max_depth=max_d,
                soft_slope=slope,
                summary_stats=summary_stats,
                scale=(out_ref * 0.75 * w_sub) / math.sqrt(int(sub_cfg["outcome"])),
                treatment_only=False,
            )
            if out_rules:
                outcome_eq.setdefault("subgroup_rules", []).extend(out_rules)
        if sub_cfg["hte"] > 0 and modifier_strength > 0.0 and modifiers:
            hte_rules = _sample_subgroup_rules(
                modifiers,
                int(sub_cfg["hte"]),
                rng,
                min_depth=min_d,
                max_depth=max_d,
                soft_slope=slope,
                summary_stats=summary_stats,
                scale=(hte_ref * 0.90 * w_sub) / math.sqrt(int(sub_cfg["hte"])),
                treatment_only=True,
            )
            if hte_rules:
                outcome_eq.setdefault("subgroup_rules", []).extend(hte_rules)

    # 4. Random Fourier Feature GP-like non-parametric multivariate surfaces
    rff_cfg = resolved.get("rff_components")
    if rff_cfg and w_rff > 0.0:
        lengthscale = float(rff_cfg["lengthscale"])
        max_terms = int(rff_cfg["max_terms"])
        if rff_cfg["treatment"] > 0 and tx_terms:
            tx_rff = _sample_rff_components(
                tx_terms,
                int(rff_cfg["treatment"]),
                rng,
                lengthscale=lengthscale,
                max_terms=max_terms,
                scale=tx_ref * 0.90 * w_rff,
                treatment_only=False,
            )
            if tx_rff:
                treatment_eq.setdefault("rff_components", []).extend(tx_rff)
        if rff_cfg["outcome"] > 0 and out_base_terms:
            out_rff = _sample_rff_components(
                out_base_terms,
                int(rff_cfg["outcome"]),
                rng,
                lengthscale=lengthscale,
                max_terms=max_terms,
                scale=out_ref * 0.90 * w_rff,
                treatment_only=False,
            )
            if out_rff:
                outcome_eq.setdefault("rff_components", []).extend(out_rff)
        if rff_cfg["hte"] > 0 and modifier_strength > 0.0 and mod_terms:
            hte_rff = _sample_rff_components(
                mod_terms,
                int(rff_cfg["hte"]),
                rng,
                lengthscale=lengthscale,
                max_terms=max_terms,
                scale=hte_ref * 0.95 * w_rff,
                treatment_only=True,
            )
            if hte_rff:
                outcome_eq.setdefault("rff_components", []).extend(hte_rff)

    # 5. Bounded rational / ratio interactions
    rat_cfg = resolved.get("rational_terms")
    if rat_cfg and w_rat > 0.0:
        if rat_cfg["treatment"] > 0 and tx_terms:
            tx_rat = _sample_rational_terms(
                tx_terms,
                int(rat_cfg["treatment"]),
                rng,
                scale=tx_ref * 0.70 * w_rat,
                treatment_only=False,
            )
            if tx_rat:
                treatment_eq.setdefault("rational_terms", []).extend(tx_rat)
        if rat_cfg["outcome"] > 0 and out_base_terms:
            out_rat = _sample_rational_terms(
                out_base_terms,
                int(rat_cfg["outcome"]),
                rng,
                scale=out_ref * 0.70 * w_rat,
                treatment_only=False,
            )
            if out_rat:
                outcome_eq.setdefault("rational_terms", []).extend(out_rat)
        if rat_cfg["hte"] > 0 and modifier_strength > 0.0 and mod_terms:
            hte_rat = _sample_rational_terms(
                mod_terms,
                int(rat_cfg["hte"]),
                rng,
                scale=hte_ref * 0.80 * w_rat,
                treatment_only=True,
            )
            if hte_rat:
                outcome_eq.setdefault("rational_terms", []).extend(hte_rat)

    # 6. Treatment-side 1-D nonlinearities (when enabled)
    tx_nl = resolved.get("treatment_nonlinearity", "none")
    if tx_nl and tx_nl != "none":
        treatment_eq["nonlinearity"] = tx_nl
        tx_nl_terms = equation_nonlinear_terms(features, treatment_eq)
        treatment_eq["nonlinear_terms"] = tx_nl_terms
        if tx_nl == "mixed":
            treatment_eq["term_nonlinearities"] = {
                term: str(MIXED_NONLINEARITY_SHAPES[int(rng.integers(len(MIXED_NONLINEARITY_SHAPES)))])
                for term in tx_nl_terms
            }

    # 7. Effect scale metadata for binary outcomes
    effect_scale = resolved.get("effect_scale", "log_or")
    if effect_scale and effect_scale != "log_or":
        outcome_eq["effect_scale"] = effect_scale

    return treatment_eq, outcome_eq


def _has_complex_settings(resolved: Mapping[str, Any]) -> bool:
    return any([
        resolved.get("dgp_mode", "parametric") != "parametric",
        resolved.get("treatment_nonlinearity", "none") != "none",
        resolved.get("max_treatment_interactions") is not None,
        int(resolved.get("outcome_interactions") or 0) > 0,
        int(resolved.get("hte_interactions") or 0) > 0,
        bool(resolved.get("pure_epistatic_hte", False)),
        bool(resolved.get("subgroup_rules")),
        bool(resolved.get("rff_components")),
        bool(resolved.get("rational_terms")),
        resolved.get("effect_scale", "log_or") != "log_or",
        resolved.get("component_weights") is not None,
        resolved.get("balance_hte_scale") is True,
        resolved.get("nonlinearity", "none") == "mixed",
    ])


def draw_equations(
    features: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Any],
    seed: int,
    *,
    main_coefficient_scale: float,
    interaction_coefficient_scale: float,
    summary_stats: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    """Python-drawn treatment and outcome equations (before rescaling and calibration).

    Main coefficients are ``uniform(-1, 1) * main_coefficient_scale`` per term
    and interaction coefficients ``uniform(-1, 1) *
    interaction_coefficient_scale``, as the generator draws its Python terms:

    - treatment equation: confounder and instrument main terms, and pairwise
      interactions of confounder terms from different variables (bounded
      automatically when >20 confounder terms or ``max_treatment_interactions``
      is set);
    - outcome equation: confounder and prognostic main terms, effect-modifier
      main terms when ``modifier_main_effects`` is on, the treatment main
      effect, and treatment interactions for every effect-modifier term,
      scaled by ``modifier_strength``.

    When complex DGP controls are set (``dgp_mode``, ``outcome_interactions``,
    ``hte_interactions``, ``pure_epistatic_hte``, ``subgroup_rules``,
    ``rff_components``, ``rational_terms``, ``treatment_nonlinearity``,
    ``nonlinearity="mixed"``, ``effect_scale``), the corresponding blocks are
    added while keeping all causal role boundaries intact.

    Each unmeasured confounder ``u_k`` enters both equations linearly with
    coefficient ``±unmeasured_strength`` times the root mean square of that
    equation's measured confounder main coefficients (random sign per equation).
    Returns the two equations, the unmeasured-confounder pseudo-features, and
    the reference root mean squares.
    """

    rng = _rng(seed, "coefficients")
    confounders = [feature for feature in features if CONFOUNDER in roles_of(feature)]
    instruments = [feature for feature in features if INSTRUMENT in roles_of(feature)]
    prognostic = [feature for feature in features if PROGNOSTIC in roles_of(feature)]
    modifiers = [feature for feature in features if EFFECT_MODIFIER in roles_of(feature)]
    modifier_only = [feature for feature in modifiers if roles_of(feature) == [EFFECT_MODIFIER]]

    confounder_terms = [
        (feature["name"], term) for feature in confounders for term in feature_terms(feature)
    ]
    is_complex = _has_complex_settings(resolved)
    is_wide = len(confounder_terms) > 20 or len(confounders) > 10 or len(modifiers) > 10

    # Balance HTE vs baseline main scale automatically on wide or complex DGPs
    # so 50 confounders do not drown out 5 modifiers (or vice versa), while keeping
    # legacy S0-S6 (<=10 each, parametric) bit-identical.
    balance_setting = resolved.get("balance_hte_scale")
    balance_hte = bool(balance_setting) if balance_setting is not None else (is_complex or is_wide)

    treatment_eq: Dict[str, Any] = {"intercept": 0.0, "coefficients": {}, "interactions": []}
    treatment_eq["coefficients"].update(
        draw_main_coefficients(confounders, rng.uniform, main_coefficient_scale)
    )
    treatment_eq["coefficients"].update(
        draw_main_coefficients(instruments, rng.uniform, main_coefficient_scale)
    )

    max_tx_inter = resolved.get("max_treatment_interactions")
    if max_tx_inter is None and len(confounder_terms) <= 20:
        for index, (owner, term) in enumerate(confounder_terms):
            for other_owner, other in confounder_terms[index + 1:]:
                if owner != other_owner:
                    treatment_eq["interactions"].append({
                        "terms": [term, other],
                        "coefficient": float(rng.uniform(-1.0, 1.0)) * interaction_coefficient_scale,
                    })
    else:
        limit = (
            int(max_tx_inter)
            if max_tx_inter is not None
            else max(30, 2 * len(confounders))
        )
        if limit > 0 and len(confounders) >= 2:
            tx_pair_scale = (
                interaction_coefficient_scale
                / math.sqrt(max(1.0, limit / max(1, len(confounders))))
            )
            treatment_eq["interactions"] = _sample_distinct_interactions(
                confounder_terms,
                limit,
                rng,
                term_centers=_term_centers_map(confounders, summary_stats),
                scale=tx_pair_scale,
                allow_triplets=False,
                center_factors=False,
            )

    outcome_eq: Dict[str, Any] = {"intercept": 0.0, "coefficients": {}, "interactions": []}
    outcome_eq["coefficients"].update(
        draw_main_coefficients(confounders, rng.uniform, main_coefficient_scale)
    )
    outcome_eq["coefficients"].update(
        draw_main_coefficients(prognostic, rng.uniform, main_coefficient_scale)
    )
    if resolved["modifier_main_effects"]:
        outcome_eq["coefficients"].update(
            draw_main_coefficients(modifier_only, rng.uniform, main_coefficient_scale)
        )
    outcome_eq["treatment_coefficient"] = float(resolved["treatment_effect"])

    mod_term_list = [term for feature in modifiers for term in feature_terms(feature)]
    if balance_hte and mod_term_list and float(resolved["modifier_strength"]) > 0.0:
        n_out_main = max(1, len(outcome_eq["coefficients"]))
        balanced_mod_scale = (
            main_coefficient_scale
            * math.sqrt(n_out_main / len(mod_term_list))
            * 0.35
            * float(resolved["modifier_strength"])
        )
    else:
        balanced_mod_scale = interaction_coefficient_scale * float(resolved["modifier_strength"])

    outcome_eq["treatment_interactions"] = [
        {
            "term": term,
            "coefficient": float(rng.uniform(-1.0, 1.0)) * balanced_mod_scale,
        }
        for term in mod_term_list
    ]

    # Save unscaled reference RMS BEFORE complex component_weights shrink linear weights
    # so unmeasured confounder strength stays anchored cleanly to the measured confounder scale.
    unmeasured = _unmeasured_features(
        resolved["unmeasured_confounders"], [feature["name"] for feature in features]
    )
    terms = [term for _, term in confounder_terms]
    fallback = main_coefficient_scale / math.sqrt(3.0)
    reference = {
        "treatment": _reference_rms(treatment_eq["coefficients"], terms, fallback),
        "outcome": _reference_rms(outcome_eq["coefficients"], terms, fallback),
    }

    if is_complex:
        treatment_eq, outcome_eq = augment_complex_equations(
            features,
            treatment_eq,
            outcome_eq,
            resolved,
            rng,
            summary_stats=summary_stats,
            main_coefficient_scale=main_coefficient_scale,
            interaction_coefficient_scale=interaction_coefficient_scale,
        )

    strength = float(resolved["unmeasured_strength"])
    for feature in unmeasured:
        for key, equation in (("treatment", treatment_eq), ("outcome", outcome_eq)):
            sign = 1.0 if rng.random() < 0.5 else -1.0
            equation["coefficients"][feature["name"]] = sign * strength * reference[key]

    if resolved["nonlinearity"] != "none":
        outcome_eq["nonlinearity"] = resolved["nonlinearity"]
        nl_terms = outcome_nonlinear_terms(features, outcome_eq)
        outcome_eq["nonlinear_terms"] = nl_terms
        if resolved["nonlinearity"] == "mixed":
            outcome_eq["term_nonlinearities"] = {
                term: str(MIXED_NONLINEARITY_SHAPES[int(rng.integers(len(MIXED_NONLINEARITY_SHAPES)))])
                for term in nl_terms
            }
    return treatment_eq, outcome_eq, unmeasured, reference


# --------------------------------------------------------------------------
# Source checks
# --------------------------------------------------------------------------


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _refuse_post_treatment_source(source_dir: Path, metadata: Mapping[str, Any], columns: Sequence[str]) -> None:
    reasons = []
    if metadata.get("post_treatment"):
        reasons.append("metadata.json has a post_treatment block")
    if POST_TREATMENT_TEXT_COLUMN in columns:
        reasons.append(f"dataset.parquet has {POST_TREATMENT_TEXT_COLUMN}")
    leaked = [column for column in columns if str(column).startswith(POST_TREATMENT_ORACLE_PREFIX)]
    if leaked:
        reasons.append(f"dataset.parquet has {leaked[0]}")
    if (source_dir / g.NOTES_FILENAME).exists():
        reasons.append(f"the source directory holds {g.NOTES_FILENAME}")
    if reasons:
        raise ValueError(
            "re-equation refuses a source with post-treatment notes, whose text depends on the "
            f"original treatment ({'; '.join(reasons)})"
        )


def _source_characteristics(
    frame: pd.DataFrame, features: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Each patient's true characteristics, read from the ``true_<variable>`` columns."""

    missing = [feature["name"] for feature in features if f"true_{feature['name']}" not in frame.columns]
    if missing:
        raise ValueError(
            f"re-equation needs a true_<variable> column for every metadata feature; missing "
            f"{['true_' + name for name in missing]}"
        )
    values: Dict[str, List[Any]] = {}
    for feature in features:
        column = f"true_{feature['name']}"
        raw = frame[column]
        if feature["type"] == "continuous":
            numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise ValueError(f"{column} has missing or non-numeric values")
            values[feature["name"]] = numeric.tolist()
        else:
            canonical = []
            for value in raw.tolist():
                category = None if value is None else canonical_category(feature, value)
                if category is None:
                    raise ValueError(
                        f"{column} value {value!r} is not one of the categories "
                        f"{feature['categories']}"
                    )
                canonical.append(category)
            values[feature["name"]] = canonical
    names = [feature["name"] for feature in features]
    return [
        {name: values[name][row] for name in names}
        for row in range(len(frame))
    ]


def _outcome_type(generation_config: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    outcome_type = generation_config.get("outcome_type", "binary")
    recorded = (metadata.get("config") or {}).get("outcome_type", outcome_type)
    if outcome_type not in {"binary", "continuous"} or recorded != outcome_type:
        raise ValueError(
            f"source outcome type is inconsistent or unknown (generation_config.json: "
            f"{outcome_type!r}, metadata.json: {recorded!r})"
        )
    return outcome_type


# --------------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------------


def _write_atomic(path: Path, write) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    def write(temporary: Path) -> None:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, default=str)

    _write_atomic(path, write)


def _existing_output(output_dir: Path, identity: Mapping[str, Any]) -> bool:
    """True when ``output_dir`` already holds this exact re-equation."""

    if not output_dir.exists():
        return False
    if not output_dir.is_dir():
        raise ValueError(f"output directory {output_dir} is not a directory")
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("identity") != _canonical(identity):
            raise ValueError(
                f"{output_dir} holds a different re-equation (source files, scenario, size, "
                "seed, or code differ); use another --output-dir"
            )
        outputs = manifest.get("outputs") or {}
        for name, digest in outputs.items():
            path = output_dir / name
            if not path.exists() or _sha256(path) != digest:
                raise ValueError(f"{path} does not match {MANIFEST_FILENAME}; use another --output-dir")
        return True
    allowed = set(OUTPUT_FILES) | {name + ".tmp" for name in OUTPUT_FILES}
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in allowed)
    if unexpected:
        raise ValueError(
            f"output directory {output_dir} holds other files ({unexpected[:5]}); use an empty "
            "or new --output-dir"
        )
    return False


# --------------------------------------------------------------------------
# Re-equation
# --------------------------------------------------------------------------


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


def _linear_predictors(
    characteristics: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    summary_stats: Mapping[str, Any],
    equation: Mapping[str, Any],
    treatment: Optional[int],
    *,
    term_matrix: Optional[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = None,
) -> np.ndarray:
    return g._compute_logits_batch(
        list(characteristics),
        list(features),
        dict(summary_stats),
        dict(equation),
        treatment=treatment,
        term_matrix=term_matrix,
    )


def _with_unmeasured(
    characteristics: Sequence[Mapping[str, Any]], unmeasured: Sequence[Mapping[str, Any]], draws: np.ndarray
) -> List[Dict[str, Any]]:
    if not unmeasured:
        return list(characteristics)
    names = [feature["name"] for feature in unmeasured]
    return [
        {**chars, **{name: float(draws[row, column]) for column, name in enumerate(names)}}
        for row, chars in enumerate(characteristics)
    ]


def reequation_cohort(
    source_dir: Path,
    scenario: Mapping[str, Any],
    *,
    size: Optional[int] = None,
    seed: int,
    output_dir: Path,
) -> Path:
    """Write a re-equated cohort to ``output_dir`` and return its path.

    Reads the source's ``dataset.parquet``, ``metadata.json``, and
    ``generation_config.json``. Writes ``dataset.parquet`` (every source column,
    with treatment, outcome, and the true probability and effect columns
    recomputed, plus ``true_u_<k>`` for each unmeasured confounder),
    ``metadata.json``, ``reequation_manifest.json``, and a copy of
    ``generation_config.json``. ``size`` subsamples patients without
    replacement (deterministic in ``seed``); ``None`` keeps all of them.

    Fails closed on a source with post-treatment notes, a source without a
    ``true_<variable>`` column for every metadata feature, a re-equation
    output as source, an invalid scenario, and a size larger than the source.
    An output directory that already holds the same re-equation is returned
    as is; one that holds anything else is an error.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if size is not None and (isinstance(size, bool) or not isinstance(size, (int, np.integer)) or int(size) < 1):
        raise ValueError("size must be a positive integer or None")
    size = None if size is None else int(size)

    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir == source_dir:
        raise ValueError("output directory must differ from the source directory")
    paths = {name: source_dir / name for name in SOURCE_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"source directory {source_dir} lacks {missing}")
    source_hashes = {name: _sha256(path) for name, path in paths.items()}

    metadata = _read_json(paths["metadata.json"])
    generation_config = _read_json(paths["generation_config.json"])
    if metadata.get("scenario") is not None or metadata.get("unmeasured_confounders"):
        raise ValueError("the source is itself a re-equation output; re-equate its source corpus")
    import pyarrow.parquet as pq

    columns = list(pq.read_schema(paths["dataset.parquet"]).names)
    _refuse_post_treatment_source(source_dir, metadata, columns)
    features = g._validate_feature_roles(
        metadata.get("features") or [], allowed_roles=ALL_ROLES, allow_empty=True
    )
    if not features:
        raise ValueError("source metadata.json has no features")
    summary_stats = metadata.get("summary_statistics") or {}
    outcome_type = _outcome_type(generation_config, metadata)
    resolved = resolve_scenario(scenario, features, seed=seed)

    identity = {
        "code_version": REEQUATION_CODE_VERSION,
        "code_sha256": code_sha256(),
        "source_sha256": source_hashes,
        "scenario": _canonical(dict(scenario)),
        "size": size,
        "seed": seed,
    }
    if _existing_output(output_dir, identity):
        LOGGER.info("Reusing the identical re-equation in %s", output_dir)
        return output_dir

    frame = pd.read_parquet(paths["dataset.parquet"]).reset_index(drop=True)
    n_source = len(frame)
    if n_source == 0:
        raise ValueError("source dataset.parquet has no rows")
    if size is not None and size > n_source:
        raise ValueError(f"size {size} exceeds the source's {n_source} patients")
    characteristics = _source_characteristics(frame, features)

    assigned = _features_with_roles(features, resolved["roles"])
    defaults = SyntheticDataConfig()
    main_scale = float(generation_config.get("main_coefficient_scale", defaults.main_coefficient_scale))
    interaction_scale = float(
        generation_config.get("interaction_coefficient_scale", defaults.interaction_coefficient_scale)
    )
    treatment_eq, outcome_eq, unmeasured, reference = draw_equations(
        assigned,
        resolved,
        seed,
        main_coefficient_scale=main_scale,
        interaction_coefficient_scale=interaction_scale,
        summary_stats=summary_stats,
    )
    equation_features = assigned + unmeasured
    equation_stats = {
        **summary_stats,
        **{feature["name"]: {"type": "continuous", "mean": 0.0, "std": 1.0} for feature in unmeasured},
    }

    # The generator's Monte Carlo rescaling and calibration, with draws taken
    # from the source cohort's covariates (rows drawn with replacement) and
    # standard-normal unmeasured confounders instead of from the summary
    # statistics, so the targets hold for the corpus actually re-equated.
    mc_rng = _rng(seed, "monte_carlo")

    def monte_carlo_draws() -> List[Dict[str, Any]]:
        rows = mc_rng.integers(0, n_source, MONTE_CARLO_DRAWS)
        latent = mc_rng.standard_normal((MONTE_CARLO_DRAWS, len(unmeasured)))
        return _with_unmeasured([characteristics[row] for row in rows], unmeasured, latent)

    treatment_eq, outcome_eq = g._rescale_for_target_logit_std(
        confounders=equation_features,
        summary_stats=equation_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        treatment_logit_std=resolved["treatment_logit_std"],
        outcome_logit_std=resolved["outcome_logit_std"],
        sampled_chars=monte_carlo_draws(),
    )
    calibration_draws = monte_carlo_draws()
    treatment_eq, outcome_eq = g._calibrate_intercepts(
        confounders=equation_features,
        summary_stats=equation_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        target_treatment_rate=resolved["target_treatment_rate"],
        target_control_outcome_rate=resolved["target_control_outcome_rate"],
        sampled_chars=calibration_draws,
    )
    cal_matrix = g._build_term_matrix(calibration_draws, equation_features, equation_stats)
    achieved_treatment = float(_sigmoid(_linear_predictors(
        calibration_draws, equation_features, equation_stats, treatment_eq, None,
        term_matrix=cal_matrix,
    )).mean())
    achieved_control = float(_sigmoid(_linear_predictors(
        calibration_draws, equation_features, equation_stats, outcome_eq, 0,
        term_matrix=cal_matrix,
    )).mean())

    # Every source row gets its unmeasured confounders, treatment, and outcome
    # from streams indexed by source row, so they do not depend on ``size``.
    latent = _rng(seed, "unmeasured").standard_normal((n_source, len(unmeasured)))
    full = _with_unmeasured(characteristics, unmeasured, latent)
    full_matrix = g._build_term_matrix(full, equation_features, equation_stats)
    treatment_lp = _linear_predictors(
        full, equation_features, equation_stats, treatment_eq, None, term_matrix=full_matrix
    )
    lp0 = _linear_predictors(
        full, equation_features, equation_stats, outcome_eq, 0, term_matrix=full_matrix
    )
    lp1 = _linear_predictors(
        full, equation_features, equation_stats, outcome_eq, 1, term_matrix=full_matrix
    )
    propensity = _sigmoid(treatment_lp)
    treatment = (_rng(seed, "treatment").random(n_source) < propensity).astype("int64")
    lp_observed = np.where(treatment == 1, lp1, lp0)
    outcome_rng = _rng(seed, "outcome")
    if outcome_type == "continuous":
        outcome = lp_observed + outcome_rng.normal(0.0, resolved["outcome_noise_std"], n_source)
        oracle = {
            "true_treatment_prob": propensity,
            "true_y0": lp0,
            "true_y1": lp1,
            "true_ite": lp1 - lp0,
        }
    else:
        p0, p1 = g.potential_outcome_probabilities(lp0, lp1, outcome_eq)
        p_observed = np.where(treatment == 1, p1, p0)
        outcome = (outcome_rng.random(n_source) < p_observed).astype("int64")
        oracle = {
            "true_treatment_prob": propensity,
            "true_outcome_prob": p_observed,
            "true_y0_prob": p0,
            "true_y1_prob": p1,
            "true_ite_prob": p1 - p0,
            "true_y0": p0,
            "true_y1": p1,
            "true_ite": p1 - p0,
        }

    if size is None:
        rows = np.arange(n_source)
    else:
        rows = np.sort(_rng(seed, "subsample").permutation(n_source)[:size])

    output = frame.iloc[rows].copy()
    output["treatment_indicator"] = treatment[rows]
    output["outcome_indicator"] = outcome[rows]
    for column in RECOMPUTED_COLUMNS:
        if column not in TREATMENT_OUTCOME_COLUMNS and column not in oracle and column in output.columns:
            output = output.drop(columns=column)
    for column, values in oracle.items():
        output[column] = np.asarray(values, dtype=float)[rows]
    for index, feature in enumerate(unmeasured):
        column = f"true_{feature['name']}"
        if column in output.columns:
            raise ValueError(f"source already has a column {column!r}")
        output[column] = latent[rows, index]
    output = output.reset_index(drop=True)

    lists = role_lists(assigned)
    unmeasured_block = [
        {
            "name": feature["name"],
            "coefficients": {
                "treatment": float(treatment_eq["coefficients"][feature["name"]]),
                "outcome": float(outcome_eq["coefficients"][feature["name"]]),
            },
            "oracle_column": f"true_{feature['name']}",
            "distribution": "standard_normal",
        }
        for feature in unmeasured
    ]
    out_metadata = {
        "config": metadata.get("config"),
        "features": assigned,
        "confounders": lists["confounders"],
        "effect_modifiers": lists["effect_modifiers"],
        "instruments": lists["instruments"],
        "prognostic_variables": lists["prognostic_variables"],
        "proxies": [
            {"name": feature["name"], "proxy_of": feature["proxy_of"]}
            for feature in assigned
            if feature.get("proxy_of")
        ],
        "treatment_equation": treatment_eq,
        "outcome_equation": outcome_eq,
        "summary_statistics": summary_stats,
        "unmeasured_confounders": unmeasured_block,
        "scenario": resolved,
        "documentation": metadata.get("documentation"),
        "post_treatment": None,
        "reequation": {
            "source_dir": str(source_dir),
            "source_sha256": source_hashes,
            "size": size,
            "seed": seed,
            "source_patients": n_source,
            "outcome_type": outcome_type,
            "main_coefficient_scale": main_scale,
            "interaction_coefficient_scale": interaction_scale,
            "unmeasured_reference_rms": reference,
            "monte_carlo": {
                "draws": MONTE_CARLO_DRAWS,
                "sampled_from": "source cohort covariates, rows drawn with replacement",
                "treatment_rate": achieved_treatment,
                "control_outcome_rate": achieved_control,
            },
            "code_version": REEQUATION_CODE_VERSION,
        },
        "dataset_statistics": g.oracle_dataset_statistics(output),
    }
    if "longitudinal" in metadata:
        longitudinal = metadata["longitudinal"]
        out_metadata["longitudinal"] = {
            **longitudinal,
            "event_counts": {
                name: int(output[name].map(len).sum()) for name in longitudinal["streams"]
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        output_dir / "dataset.parquet", lambda path: output.to_parquet(path, index=False)
    )
    _write_json(output_dir / "metadata.json", out_metadata)
    _write_atomic(
        output_dir / "generation_config.json",
        lambda path: shutil.copyfile(paths["generation_config.json"], path),
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "identity": _canonical(identity),
        "source": {"path": str(source_dir), "sha256": source_hashes},
        "scenario": _canonical(dict(scenario)),
        "resolved_scenario": _canonical(resolved),
        "size": size,
        "seed": seed,
        "code_version": REEQUATION_CODE_VERSION,
        "code_sha256": identity["code_sha256"],
        "outputs": {
            name: _sha256(output_dir / name)
            for name in ("dataset.parquet", "metadata.json", "generation_config.json")
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / MANIFEST_FILENAME, manifest)
    LOGGER.info(
        "Re-equated %d of %d patients from %s into %s (scenario %r)",
        len(output), n_source, source_dir, output_dir, resolved["name"],
    )
    return output_dir


def reequation_main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m synthetic_data.cli reequation``."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m synthetic_data.cli reequation",
        description=(
            "Recompute treatment and outcome over an existing synthetic cohort's oracle "
            "covariates with a scenario's equations; the text is copied unchanged."
        ),
    )
    parser.add_argument("--source", required=True, help="Source cohort directory")
    parser.add_argument("--scenario", required=True, help="Scenario JSON file")
    parser.add_argument(
        "--size", type=int, default=None,
        help="Patients to keep, drawn without replacement (default: all)",
    )
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    with open(args.scenario, "r", encoding="utf-8") as handle:
        scenario = json.load(handle)
    path = reequation_cohort(
        Path(args.source), scenario, size=args.size, seed=args.seed, output_dir=Path(args.output_dir)
    )
    print(path)
    return 0
