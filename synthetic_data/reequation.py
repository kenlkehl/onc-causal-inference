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
    NONLINEARITIES,
    PROGNOSTIC,
    ROLE_GROUPS,
    canonical_category,
    draw_main_coefficients,
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


def resolve_scenario(
    scenario: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> Dict[str, Any]:
    """Validate a scenario and fill omitted keys with the generator's defaults.

    ``roles`` maps role groups (``confounder``, ``effect_modifier``,
    ``dual_role``, ``instrument``, ``prognostic``) to a count or a list of
    variable names. With ``role_assignment: "named"`` every value must be a
    list; with ``"random"`` (the default) lists are fixed first and counts are
    drawn, with a seeded RNG, from the remaining variables in group order. A
    scenario without ``roles`` keeps the source's roles. Proxies take no role.
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
    treatment_std, outcome_std = defaults.logit_stds()
    return {
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
    counts: Dict[str, int] = {}
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
            counts[group] = _count(value, f"roles.{group}")
    remaining = [name for name in eligible if name not in taken]
    requested = sum(counts.values())
    if requested > len(remaining):
        raise ValueError(
            f"scenario roles request {requested} randomly assigned variables but only "
            f"{len(remaining)} unassigned variables remain"
        )
    order = [remaining[index] for index in _rng(seed, "roles").permutation(len(remaining))]
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


def draw_equations(
    features: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Any],
    seed: int,
    *,
    main_coefficient_scale: float,
    interaction_coefficient_scale: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    """Python-drawn treatment and outcome equations (before rescaling and calibration).

    Main coefficients are ``uniform(-1, 1) * main_coefficient_scale`` per term
    and interaction coefficients ``uniform(-1, 1) *
    interaction_coefficient_scale``, as the generator draws its Python terms:

    - treatment equation: confounder and instrument main terms, and the
      pairwise interactions of confounder terms from different variables;
    - outcome equation: confounder and prognostic main terms, effect-modifier
      main terms when ``modifier_main_effects`` is on, the treatment main
      effect, and treatment interactions for every effect-modifier term,
      scaled by ``modifier_strength``.

    Each unmeasured confounder ``u_k`` enters both equations with coefficient
    ``±unmeasured_strength`` times the root mean square of that equation's
    measured confounder main coefficients (random sign per equation).
    Returns the two equations, the unmeasured-confounder pseudo-features, and
    the reference root mean squares.
    """

    rng = _rng(seed, "coefficients")
    confounders = [feature for feature in features if CONFOUNDER in roles_of(feature)]
    instruments = [feature for feature in features if INSTRUMENT in roles_of(feature)]
    prognostic = [feature for feature in features if PROGNOSTIC in roles_of(feature)]
    modifiers = [feature for feature in features if EFFECT_MODIFIER in roles_of(feature)]
    modifier_only = [feature for feature in modifiers if roles_of(feature) == [EFFECT_MODIFIER]]

    treatment_eq: Dict[str, Any] = {"intercept": 0.0, "coefficients": {}, "interactions": []}
    treatment_eq["coefficients"].update(
        draw_main_coefficients(confounders, rng.uniform, main_coefficient_scale)
    )
    treatment_eq["coefficients"].update(
        draw_main_coefficients(instruments, rng.uniform, main_coefficient_scale)
    )
    confounder_terms = [
        (feature["name"], term) for feature in confounders for term in feature_terms(feature)
    ]
    for index, (owner, term) in enumerate(confounder_terms):
        for other_owner, other in confounder_terms[index + 1:]:
            if owner != other_owner:
                treatment_eq["interactions"].append({
                    "terms": [term, other],
                    "coefficient": float(rng.uniform(-1.0, 1.0)) * interaction_coefficient_scale,
                })

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
    outcome_eq["treatment_interactions"] = [
        {
            "term": term,
            "coefficient": float(rng.uniform(-1.0, 1.0))
            * interaction_coefficient_scale
            * float(resolved["modifier_strength"]),
        }
        for feature in modifiers
        for term in feature_terms(feature)
    ]

    unmeasured = _unmeasured_features(
        resolved["unmeasured_confounders"], [feature["name"] for feature in features]
    )
    terms = [term for _, term in confounder_terms]
    fallback = main_coefficient_scale / math.sqrt(3.0)
    reference = {
        "treatment": _reference_rms(treatment_eq["coefficients"], terms, fallback),
        "outcome": _reference_rms(outcome_eq["coefficients"], terms, fallback),
    }
    strength = float(resolved["unmeasured_strength"])
    for feature in unmeasured:
        for key, equation in (("treatment", treatment_eq), ("outcome", outcome_eq)):
            sign = 1.0 if rng.random() < 0.5 else -1.0
            equation["coefficients"][feature["name"]] = sign * strength * reference[key]

    if resolved["nonlinearity"] != "none":
        outcome_eq["nonlinearity"] = resolved["nonlinearity"]
        outcome_eq["nonlinear_terms"] = outcome_nonlinear_terms(features, outcome_eq)
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
) -> np.ndarray:
    features = list(features)
    return np.array([
        g._compute_logit(chars, features, summary_stats, equation, treatment=treatment)
        for chars in characteristics
    ], dtype=float)


def _with_unmeasured(
    characteristics: Sequence[Mapping[str, Any]], unmeasured: Sequence[Mapping[str, Any]], draws: np.ndarray
) -> List[Dict[str, Any]]:
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
    achieved_treatment = float(_sigmoid(_linear_predictors(
        calibration_draws, equation_features, equation_stats, treatment_eq, None)).mean())
    achieved_control = float(_sigmoid(_linear_predictors(
        calibration_draws, equation_features, equation_stats, outcome_eq, 0)).mean())

    # Every source row gets its unmeasured confounders, treatment, and outcome
    # from streams indexed by source row, so they do not depend on ``size``.
    latent = _rng(seed, "unmeasured").standard_normal((n_source, len(unmeasured)))
    full = _with_unmeasured(characteristics, unmeasured, latent)
    treatment_lp = _linear_predictors(full, equation_features, equation_stats, treatment_eq, None)
    lp0 = _linear_predictors(full, equation_features, equation_stats, outcome_eq, 0)
    lp1 = _linear_predictors(full, equation_features, equation_stats, outcome_eq, 1)
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
        p0, p1, p_observed = _sigmoid(lp0), _sigmoid(lp1), _sigmoid(lp_observed)
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
