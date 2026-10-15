"""Shared grouped elastic-net primitives and the statistical policy object."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import oci.inference.stage2_elastic_net_selection as selection_module
from oci.inference.stage2_elastic_net_selection import (
    Stage2ElasticNetSelectionConfig,
    _encode_design,
)


def _continuous(feature_id: str, name: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "description": name,
        "value_type": "continuous",
        "modeling_strategy": "continuous",
        "categories_or_unit": ["unitless"],
        "measurement_definition": name,
        "missing_value_rule": "null when absent",
    }


def _categorical(feature_id: str, name: str, levels: list[str]) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "description": name,
        "value_type": "categorical",
        "categories_or_unit": levels,
        "measurement_definition": name,
        "missing_value_rule": "null when absent",
    }


def test_ordinal_measurement_is_one_standardized_score_not_dummy_columns():
    definition = {
        **_categorical("ecog", "ecog", ["0", "1", "2", "3"]),
        "value_type": "ordinal",
    }
    train = pd.DataFrame({"ecog": [0, 1, 2, 3, 1, 2, None, 0]})
    valid = pd.DataFrame({"ecog": [3, None, 1]})

    design = _encode_design(
        train,
        valid,
        [definition],
        categorical_min_count=1,
    )

    assert design.column_names == ("ecog:ordered_score", "ecog:missing")
    assert design.column_feature_ids == ("ecog", "ecog")
    np.testing.assert_allclose(np.mean(design.train, axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(design.train, axis=0), 1.0, atol=1e-12)


def test_modifier_continuous_values_are_winsorized_from_training_only():
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"lab": rng.normal(7.5, 1.2, size=800)})
    valid = pd.DataFrame({"lab": [7.0, 7_200.0]})
    design = _encode_design(
        train,
        valid,
        [_continuous("lab", "lab")],
        categorical_min_count=5,
    )

    bounded = selection_module._winsorize_modifier_design(
        design,
        quantile=0.005,
    )

    assert bounded.valid[-1, 0] <= np.max(bounded.train[:, 0])
    assert bounded.valid[-1, 0] < design.valid[-1, 0]


def test_modifier_policy_values_are_validated():
    policy = Stage2ElasticNetSelectionConfig()
    policy.validate()
    assert "modifier_min_fold_r_loss_improvement" not in policy.public_dict()
    with pytest.raises(ValueError, match="modifier_ridge_alpha"):
        Stage2ElasticNetSelectionConfig(modifier_ridge_alpha=0.0).validate()
    with pytest.raises(ValueError, match="modifier_continuous_winsor_quantile"):
        Stage2ElasticNetSelectionConfig(
            modifier_continuous_winsor_quantile=0.25
        ).validate()
    with pytest.raises(ValueError, match="internal_cv_folds"):
        Stage2ElasticNetSelectionConfig(internal_cv_folds=1).validate()
