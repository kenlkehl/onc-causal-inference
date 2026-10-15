"""Nested interaction tests shared by the multi-model univariable evidence family."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oci.inference.stage2_statistical_selection import _modifier_test_chunk


def test_categorical_modifier_uses_all_level_interactions_in_one_omnibus_test():
    rng = np.random.default_rng(991)
    rows = 3_000
    levels = np.asarray(["A", "B", "C"] * (rows // 3))
    rng.shuffle(levels)
    treatment = rng.binomial(1, 0.5, size=rows).astype(float)
    interaction = np.where(levels == "B", 2.1, np.where(levels == "C", -2.1, 0.0))
    outcome_logit = -0.5 + 0.4 * treatment + treatment * interaction
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-outcome_logit))).astype(float)
    definition = {
        "feature_id": "marker",
        "name": "three_level_marker",
        "description": "Pretreatment three_level_marker.",
        "value_type": "categorical",
        "categories_or_unit": ["A", "B", "C"],
        "measurement_definition": "Extract three_level_marker before treatment.",
        "missing_value_rule": "Use null when undocumented.",
    }
    frame = pd.DataFrame({"three_level_marker": levels})
    reduced_with_treatment = np.column_stack([np.ones(rows), treatment])

    (row,) = _modifier_test_chunk(
        frame,
        treatment,
        outcome,
        reduced_with_treatment,
        [definition],
        binary_outcome=True,
        p_value_threshold=0.001,
    )

    assert row["feature_id"] == "marker" and row["vote"] is True
    test = row["interaction_test"]
    assert test["test"] == "likelihood_ratio_chi_square"
    assert test["degrees_of_freedom"] == 2
    assert test["categorical_reference_level"] == "A"
    assert test["candidate_interaction_columns"] == [
        "three_level_marker:level=B",
        "three_level_marker:level=C",
    ]
    assert test["tested_interaction_columns"] == [
        "three_level_marker:level=B",
        "three_level_marker:level=C",
    ]
