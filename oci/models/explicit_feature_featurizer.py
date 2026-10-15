# oci/models/explicit_feature_featurizer.py
"""Raw featurization of explicitly extracted features for causal forests.

For categorical features: k-1 dummy variables (reference coding)
For continuous features: z-score normalized value
For all features: binary missingness indicator
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import ExplicitFeatureSpec

logger = logging.getLogger(__name__)


def _category_match_key(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\u2265", ">=").replace("\u2264", "<=")
    text = re.sub(r"[\s_-]+", "", text)
    return text


def _categorical_value_map(spec: ExplicitFeatureSpec) -> Dict[str, str]:
    cats = spec.categories or []
    value_map = {_category_match_key(cat): str(cat) for cat in cats}
    by_category_key = {_category_match_key(cat): str(cat) for cat in cats}
    aliases = getattr(spec, "value_aliases", None) or {}
    if isinstance(aliases, dict):
        for raw_category, raw_aliases in aliases.items():
            category = by_category_key.get(_category_match_key(raw_category))
            if category is None:
                continue
            alias_values = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
            for alias in alias_values:
                value_map[_category_match_key(alias)] = category
    return value_map


def filter_specs_by_role(
    specs: List[ExplicitFeatureSpec],
    role: Optional[str],
) -> List[ExplicitFeatureSpec]:
    """Return specs matching a causal role; role=None returns all specs."""
    if role is None:
        return list(specs)
    return [spec for spec in specs if role in spec.roles]


def get_raw_explicit_features(
    feature_values_list: List[Dict[str, Any]],
    specs: List[ExplicitFeatureSpec],
    continuous_means: Optional[Dict[str, float]] = None,
    continuous_stds: Optional[Dict[str, float]] = None,
    role: Optional[str] = None,
) -> Tuple[List[List[float]], List[str]]:
    """Get raw explicit features for causal forest (no MLP projection).

    Returns one-hot encoded categoricals + normalized continuous + missingness indicators.
    This is used for causal forest models where we want interpretable raw features.

    Args:
        feature_values_list: List of dicts, one per sample
        specs: List of feature specifications
        continuous_means: Optional pre-computed means for normalization
        continuous_stds: Optional pre-computed stds for normalization
        role: Optional causal role filter ("confounder" or "effect_modifier")

    Returns:
        Tuple of (features_list, feature_names)
        features_list: List of feature vectors, one per sample
        feature_names: List of feature names for interpretability
    """
    specs = filter_specs_by_role(specs, role)
    if continuous_means is None:
        continuous_means = {}
    if continuous_stds is None:
        continuous_stds = {}

    if not specs:
        return [[] for _ in feature_values_list], []

    # Build feature names
    feature_names = []
    for spec in specs:
        if spec.type == "categorical":
            cats = spec.categories or []
            for cat in cats[1:]:  # Skip reference category
                feature_names.append(f"{spec.name}_{cat}")
            feature_names.append(f"{spec.name}_missing")
        else:
            feature_names.append(f"{spec.name}_normalized")
            feature_names.append(f"{spec.name}_missing")

    # Compute missing means/stds as needed.
    for spec in specs:
        if (
            spec.type == "continuous"
            and (spec.name not in continuous_means or spec.name not in continuous_stds)
        ):
            vals = []
            for values in feature_values_list:
                val = values.get(spec.name)
                missing = values.get(f"{spec.name}_missing", val is None)
                if not missing and val is not None:
                    vals.append(float(val))
            if vals:
                continuous_means[spec.name] = sum(vals) / len(vals)
                variance = sum((v - continuous_means[spec.name]) ** 2 for v in vals) / len(vals)
                continuous_stds[spec.name] = max(variance ** 0.5, 1e-6)
            else:
                continuous_means[spec.name] = 0.0
                continuous_stds[spec.name] = 1.0

    # Encode samples
    features_list = []
    for values in feature_values_list:
        features = []

        for spec in specs:
            name = spec.name
            val = values.get(name)
            missing = values.get(f"{name}_missing", val is None)

            if spec.type == "categorical":
                cats = spec.categories or []
                value_map = _categorical_value_map(spec)
                canonical_val = (
                    value_map.get(_category_match_key(val))
                    if val is not None
                    else None
                )
                # k-1 dummy coding
                for i, cat in enumerate(cats[1:], 1):
                    if not missing and canonical_val == cat:
                        features.append(1.0)
                    else:
                        features.append(0.0)
                features.append(1.0 if missing else 0.0)
            else:
                if not missing and val is not None:
                    mean = continuous_means.get(name, 0.0)
                    std = continuous_stds.get(name, 1.0)
                    features.append((float(val) - mean) / std)
                else:
                    features.append(0.0)
                features.append(1.0 if missing else 0.0)

        features_list.append(features)

    return features_list, feature_names
