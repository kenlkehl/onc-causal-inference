# synthetic_data/dgp_terms.py
"""Causal roles, equation terms, and category matching shared by the generator and re-equation.

A variable's ``roles`` (in ``metadata.json`` ``features``) place it in the
equations:

- ``confounder``: main terms in both the treatment and the outcome equation;
- ``effect_modifier``: a treatment interaction in the outcome equation, plus an
  outcome main term when ``modifier_main_effects`` is on;
- ``instrument``: a main term in the treatment equation only;
- ``prognostic``: a main term in the outcome equation only.

A variable may carry ``confounder`` and ``effect_modifier`` together (a dual
role); ``instrument`` and ``prognostic`` stand alone. A variable without roles
(an unassigned variable, or a proxy) enters no equation but still appears in the
patient's text.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

CONFOUNDER = "confounder"
EFFECT_MODIFIER = "effect_modifier"
INSTRUMENT = "instrument"
PROGNOSTIC = "prognostic"

# Roles the design LLM is asked to tag; the others are assigned by Python.
LLM_ROLES = frozenset({CONFOUNDER, EFFECT_MODIFIER})
ALL_ROLES = frozenset({CONFOUNDER, EFFECT_MODIFIER, INSTRUMENT, PROGNOSTIC})

# Disjoint role groups of the re-equation scenario ``roles`` block, in the
# order in which counts are drawn.
ROLE_GROUPS: Dict[str, tuple] = {
    "confounder": (CONFOUNDER,),
    "effect_modifier": (EFFECT_MODIFIER,),
    "dual_role": (CONFOUNDER, EFFECT_MODIFIER),
    "instrument": (INSTRUMENT,),
    "prognostic": (PROGNOSTIC,),
}

# Metadata keys that list the features with each role.
ROLE_METADATA_KEYS = {
    CONFOUNDER: "confounders",
    EFFECT_MODIFIER: "effect_modifiers",
    INSTRUMENT: "instruments",
    PROGNOSTIC: "prognostic_variables",
}

NONLINEARITIES = (
    "none",
    "threshold",
    "quadratic",
    "hinge",
    "saturating",
    "sinusoidal",
    "band",
    "cubic",
    "mixed",
)
# Per-term shapes sampled when ``nonlinearity == "mixed"``.
MIXED_NONLINEARITY_SHAPES = (
    "none",
    "threshold",
    "quadratic",
    "hinge",
    "saturating",
    "sinusoidal",
    "band",
    "cubic",
)
_INV_SQRT_6 = 1.0 / math.sqrt(6.0)

# Hyphen/dash look-alikes that LLMs emit interchangeably (U+2010 hyphen,
# U+2011 non-breaking hyphen, U+2012 figure dash, U+2013 en dash, U+2014 em
# dash, U+2212 minus). NFKC does not fold these to ASCII "-".
_DASH_TRANSLATION = str.maketrans({ch: "-" for ch in "\u2010\u2011\u2012\u2013\u2014\u2212"})


def category_lookup_key(value: Any) -> str:
    """Spelling-insensitive key for matching category labels across LLM outputs.

    Feature definitions and summary statistics are produced by separate LLM
    calls, so the same category can arrive as "First\u2011line" in one and
    "First-line" in the other. Matching on this key keeps every declared
    category sampleable while the sampled value keeps the definition's label.
    """

    text = unicodedata.normalize("NFKC", str(value)).translate(_DASH_TRANSLATION)
    return re.sub(r"\s+", " ", text).strip().casefold()


def category_probabilities(feature: Mapping[str, Any], summary_stats: Mapping[str, Any]) -> List[float]:
    """Sampling probabilities of a categorical feature's levels, in category order.

    Proportions are matched to categories by :func:`category_lookup_key` and
    normalized; missing or all-zero proportions give a uniform distribution.
    """

    categories = list(feature["categories"])
    proportions = (summary_stats.get(feature["name"], {}) or {}).get("proportions", {})
    if not proportions:
        return [1.0 / len(categories) for _ in categories]
    by_key = {category_lookup_key(cat): value for cat, value in proportions.items()}
    probs = [by_key.get(category_lookup_key(cat), 0.0) for cat in categories]
    prob_sum = sum(probs)
    if prob_sum > 0:
        return [p / prob_sum for p in probs]
    return [1.0 / len(categories)] * len(categories)


def continuous_moments(feature: Mapping[str, Any], summary_stats: Mapping[str, Any]) -> tuple:
    """Summary mean and SD used to standardize a continuous feature (SD 0 -> 1)."""

    stats = summary_stats.get(feature["name"], {}) or {}
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0))
    if std == 0:
        std = 1.0
    return mean, std


def level_index_moments(probabilities: Sequence[float]) -> tuple:
    """Mean and SD of a categorical level index 0..K-1 under its probabilities."""

    mean = sum(index * p for index, p in enumerate(probabilities))
    variance = sum(((index - mean) ** 2) * p for index, p in enumerate(probabilities))
    return mean, (math.sqrt(variance) if variance > 0 else 1.0)


def standardized_value(
    feature: Mapping[str, Any], value: Any, summary_stats: Mapping[str, Any]
) -> float:
    """Standardized value of one characteristic.

    Continuous: ``(x - mean) / sd`` with the summary moments. Categorical: the
    level index in category order, standardized by its mean and SD under the
    sampling proportions.
    """

    if feature["type"] == "continuous":
        mean, std = continuous_moments(feature, summary_stats)
        return (float(value) - mean) / std
    categories = list(feature["categories"])
    index = categories.index(value)
    mean, std = level_index_moments(category_probabilities(feature, summary_stats))
    return (index - mean) / std


def canonical_category(feature: Mapping[str, Any], value: Any) -> Optional[str]:
    """The feature's category label that ``value`` spells, or None."""

    if value in feature["categories"]:
        return value
    key = category_lookup_key(value)
    for category in feature["categories"]:
        if category_lookup_key(category) == key:
            return category
    return None


def feature_terms(feature: Mapping[str, Any]) -> List[str]:
    """Equation term names of a feature: its name, or one dummy per non-reference level."""

    if feature["type"] == "continuous":
        return [feature["name"]]
    return [f"{feature['name']}_{category}" for category in feature["categories"][1:]]


def roles_of(feature: Mapping[str, Any]) -> List[str]:
    return [str(role) for role in feature.get("roles") or []]


def features_with_role(features: Iterable[Mapping[str, Any]], role: str) -> List[Dict[str, Any]]:
    return [dict(feature) for feature in features if role in roles_of(feature)]


def validate_role_combination(name: str, roles: Sequence[str], *, allow_empty: bool) -> List[str]:
    """Normalize one feature's roles; reject unknown roles and invalid combinations."""

    roles = list(dict.fromkeys(str(role) for role in roles or []))
    if not roles:
        if allow_empty:
            return []
        raise ValueError(
            f"Feature '{name}' is missing roles; expected one or both of "
            "['confounder', 'effect_modifier']"
        )
    invalid = set(roles) - ALL_ROLES
    if invalid:
        raise ValueError(f"Feature '{name}' has invalid roles: {sorted(invalid)}")
    standalone = set(roles) & {INSTRUMENT, PROGNOSTIC}
    if standalone and len(roles) > 1:
        raise ValueError(
            f"Feature '{name}' combines {sorted(standalone)} with other roles; instrument "
            "and prognostic roles stand alone"
        )
    return roles


def role_group(feature: Mapping[str, Any]) -> Optional[str]:
    """The scenario role group of a feature, or None when it has no role."""

    roles = set(roles_of(feature))
    for group, group_roles in ROLE_GROUPS.items():
        if roles == set(group_roles):
            return group
    return None


def role_lists(features: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """``confounders``, ``effect_modifiers``, ``instruments``, ``prognostic_variables``."""

    return {key: features_with_role(features, role) for role, key in ROLE_METADATA_KEYS.items()}


def resolve_term_nonlinearity(
    term: str,
    nonlinearity: str,
    term_nonlinearities: Optional[Mapping[str, str]] = None,
) -> str:
    """Effective 1-D nonlinear shape for ``term`` under an equation's nonlinearity settings."""

    if term_nonlinearities and term in term_nonlinearities:
        return str(term_nonlinearities[term])
    if nonlinearity == "mixed":
        return "none"
    return nonlinearity


def nonlinear_value(z: float, nonlinearity: str) -> float:
    """Shape of a standardized continuous term.

    Legacy shapes ``"none"`` (``z``), ``"threshold"`` (``1[z > 0]``), and
    ``"quadratic"`` (``z**2``) keep their exact historical definitions.
    Additional shapes support smooth/non-monotonic/piecewise responses:
    ``"hinge"`` (``max(0, z)``), ``"saturating"`` (``tanh(z)``),
    ``"sinusoidal"`` (``sin(1.5 * z)``), ``"band"`` (``exp(-0.5 * z**2)``),
    and ``"cubic"`` (clipped standardized Hermite ``(z**3 - 3*z) / sqrt(6)``).
    """

    if nonlinearity == "threshold":
        return 1.0 if z > 0 else 0.0
    if nonlinearity == "quadratic":
        return z * z
    if nonlinearity == "hinge":
        return z if z > 0.0 else 0.0
    if nonlinearity == "saturating":
        return math.tanh(z)
    if nonlinearity == "sinusoidal":
        return math.sin(1.5 * z)
    if nonlinearity == "band":
        return math.exp(-0.5 * z * z)
    if nonlinearity == "cubic":
        val = (z * z * z - 3.0 * z) * _INV_SQRT_6
        if val > 4.0:
            return 4.0
        if val < -4.0:
            return -4.0
        return val
    return z


def nonlinear_array(z: np.ndarray, nonlinearity: str) -> np.ndarray:
    """Vectorized NumPy equivalent of :func:`nonlinear_value`."""

    arr = np.asarray(z, dtype=float)
    if nonlinearity == "threshold":
        return np.where(arr > 0.0, 1.0, 0.0)
    if nonlinearity == "quadratic":
        return arr * arr
    if nonlinearity == "hinge":
        return np.where(arr > 0.0, arr, 0.0)
    if nonlinearity == "saturating":
        return np.tanh(arr)
    if nonlinearity == "sinusoidal":
        return np.sin(1.5 * arr)
    if nonlinearity == "band":
        return np.exp(-0.5 * arr * arr)
    if nonlinearity == "cubic":
        return np.clip((arr * arr * arr - 3.0 * arr) * _INV_SQRT_6, -4.0, 4.0)
    return arr


def equation_nonlinear_terms(
    features: Sequence[Mapping[str, Any]], equation: Mapping[str, Any]
) -> List[str]:
    """Continuous features that appear in an equation's main terms or treatment interactions."""

    continuous = [feature["name"] for feature in features if feature["type"] == "continuous"]
    used = set((equation.get("coefficients") or {}).keys())
    used.update(
        str(item.get("term", "")) for item in equation.get("treatment_interactions") or []
    )
    return [name for name in continuous if name in used]


def outcome_nonlinear_terms(
    features: Sequence[Mapping[str, Any]], outcome_equation: Mapping[str, Any]
) -> List[str]:
    """Continuous features that appear in the outcome main terms or treatment interactions."""

    return equation_nonlinear_terms(features, outcome_equation)


def draw_main_coefficients(
    features: Iterable[Mapping[str, Any]], uniform, scale: float
) -> Dict[str, float]:
    """Python-drawn main coefficients, ``uniform(-1, 1) * scale`` per term, in feature order.

    ``uniform`` is called as ``uniform(-1.0, 1.0)`` (for example
    ``numpy.random.uniform`` or a ``numpy.random.Generator.uniform``).
    """

    return {
        term: float(uniform(-1.0, 1.0)) * scale
        for feature in features
        for term in feature_terms(feature)
    }
