from __future__ import annotations

import numpy as np

from oci.config import (
    BoWViewConfig,
    TfidfNuisanceStackScientificConfig,
)
from oci.inference.tfidf_topic_discovery import (
    _strata,
    fit_joint_cross_fitted_nuisance_stacks,
)


def _inputs() -> dict:
    row_count = 48
    treatment = np.asarray(
        [index % 2 for index in range(row_count)],
        dtype=float,
    )
    outcome = np.asarray(
        [(index // 2) % 2 for index in range(row_count)],
        dtype=float,
    )
    long_prefix = "paddingword " * 1500
    texts = [
        (
            f"{long_prefix}sentinelafterfourteenthousand "
            f"patient_{index} arm_{int(treatment[index])} "
            f"outcome_{int(outcome[index])} cycle_{index % 6}"
        )
        for index in range(row_count)
    ]
    return {
        "texts": texts,
        "treatment": treatment,
        "outcome": outcome,
        "outcome_binary": True,
        "strata": _strata(
            treatment,
            outcome,
            outcome_binary=True,
        ),
        "views": [
            BoWViewConfig(
                name="linear_1_2",
                min_df=1,
                max_df=1.0,
                max_features=256,
                ngram_range_min=1,
                ngram_range_max=2,
                bow_model="linear",
            )
        ],
        "folds": 3,
        "random_state": 71,
        "nuisance_stack_config": (
            TfidfNuisanceStackScientificConfig()
        ),
    }


def test_serial_and_threaded_top_level_folds_are_exact():
    inputs = _inputs()
    serial = fit_joint_cross_fitted_nuisance_stacks(
        **inputs,
        tfidf_workers=1,
        owner_cpu_budget=1,
    )
    threaded = fit_joint_cross_fitted_nuisance_stacks(
        **inputs,
        tfidf_workers=3,
        owner_cpu_budget=3,
    )

    assert set(serial) == set(threaded) == {"treatment", "outcome", "mu0", "mu1"}
    for target in ("treatment", "outcome", "mu0", "mu1"):
        for key in ("base_oof", "stacked_oof", "fold_ids"):
            assert np.array_equal(serial[target][key], threaded[target][key])
        assert (
            serial[target]["fit_positions_by_row"]
            == threaded[target]["fit_positions_by_row"]
        )
        serial_external, serial_views = serial[target]["fitted"].predict(
            ["sentinelafterfourteenthousand heldout cycle_2"]
        )
        threaded_external, threaded_views = threaded[target]["fitted"].predict(
            ["sentinelafterfourteenthousand heldout cycle_2"]
        )
        assert np.array_equal(serial_external, threaded_external)
        assert serial_views.keys() == threaded_views.keys()
        for view_name in serial_views:
            assert np.array_equal(serial_views[view_name], threaded_views[view_name])
        for row, fit_positions in enumerate(threaded[target]["fit_positions_by_row"]):
            assert fit_positions
            assert row not in fit_positions
    # The complete note, not a prompt-sized prefix, reaches the vectorizer.
    vectorizer = threaded["treatment"]["fitted"].base_models[0][0]
    assert "sentinelafterfourteenthousand" in vectorizer.vocabulary_
