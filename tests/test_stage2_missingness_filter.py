"""Training-only measurement coverage and the strict 95% missing boundary."""

import json

import numpy as np
import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis


def test_missingness_boundary_preserves_negative_and_categorical_observations(tmp_path):
    rows = 800
    observed_values = {
        "below": [1] * 39,
        "boundary_zero": [0] * 40,
        "boundary_false": [False] * 40,
        "boundary_absent": ["Absent"] * 40,
        "above": [-1] * 41,
        "empty_locked": [],
    }
    frame = pd.DataFrame({"_oci_row_id": np.arange(rows) * 2, **{
        name: values + [None, np.nan, pd.NA] + [None] * (rows - len(values) - 3)
        for name, values in observed_values.items()
    }})
    before = frame.copy(deep=True)
    definitions = [{"name": name, "feature_id": f"f_{name}",
                    "roles": ["confounder", "effect_modifier"],
                    "configured_explicit_feature": name == "empty_locked"}
                   for name in observed_values]
    filtered, retained, summary = analysis._filter_sparse_training_candidates(
        frame, definitions, audit_path=tmp_path / "filter.json",
        health_audit_path=tmp_path / "health.json", minimum_row_nonmissing_fraction=0.05,
    )
    expected = ["boundary_zero", "boundary_false", "boundary_absent", "above"]
    assert [f["name"] for f in retained] == expected
    pd.testing.assert_frame_equal(filtered, before[["_oci_row_id", *expected]])
    pd.testing.assert_frame_equal(frame, before)
    assert summary["candidates_dropped"] == 2
    audit = json.loads((tmp_path / "filter.json").read_text())
    by_name = {f["name"]: f for f in audit["features"]}
    assert by_name["below"]["missing_rows"] == 761
    assert by_name["boundary_zero"]["missing_fraction"] == 0.95
    assert by_name["empty_locked"]["configured_explicit_feature"]
    assert by_name["empty_locked"]["action"] == "drop"


@pytest.mark.parametrize("disjoint_observations", [False, True])
def test_all_candidates_dropped_fails_even_when_union_coverage_is_adequate(
    tmp_path, disjoint_observations,
):
    frame = pd.DataFrame({"_oci_row_id": range(40), "a": np.nan, "b": np.nan, "c": np.nan})
    if disjoint_observations:
        for row, name in enumerate(["a", "b", "c"]):
            frame.loc[row, name] = 1
    with pytest.raises(ValueError, match="catastrophically sparse"):
        analysis._filter_sparse_training_candidates(
            frame, [{"name": name, "feature_id": name} for name in ["a", "b", "c"]],
            audit_path=tmp_path / "filter.json", health_audit_path=tmp_path / "health.json",
            minimum_row_nonmissing_fraction=0.05,
        )
    audit = json.loads((tmp_path / "filter.json").read_text())
    assert audit["status"] == "failed" and audit["candidates_dropped"] == 3
    health = json.loads((tmp_path / "health.json").read_text())
    assert health["status"] == "failed"
    assert health["rows_with_any_nonmissing"] == (3 if disjoint_observations else 0)
