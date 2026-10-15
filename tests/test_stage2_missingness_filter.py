"""Training-only, count-based candidate informativeness rule (B4 / S2A-003)."""

import json

import numpy as np
import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference.plain_handoff_stage2 import plain_stage2_config_from_mapping
from oci.inference.stage2_extraction_failures import extraction_failure_column


def _padded(values, rows):
    return list(values) + [None] * (rows - len(values))


def _filter(frame, definitions, tmp_path, **kwargs):
    return analysis._filter_sparse_training_candidates(
        frame, definitions, audit_path=tmp_path / "filter.json",
        health_audit_path=tmp_path / "health.json", **kwargs,
    )


def test_count_rule_keeps_rare_documented_and_drops_uninformative(tmp_path):
    rows = 800
    columns = {
        # Rare-but-documented presence (25/800 = 3%): the former 5% rule dropped it.
        "brain_metastases": _padded(["present"] * 25, rows),
        # Observed for everyone but nearly constant: only one level reaches K.
        "nearly_constant": ["no"] * 795 + ["yes"] * 5,
        # Every observed value identical and no "not documented" level.
        "all_same_category": ["no"] * rows,
        # Exactly K observed with two informative levels (observed vs not documented).
        "boundary_count": _padded([False] * 20, rows),
        "below_count": _padded(["present"] * 19, rows),
        "continuous_varied": _padded(list(np.linspace(1, 30, 30)), rows),
        "continuous_constant": _padded([7.0] * 300, rows),
        "continuous_sparse": _padded(list(range(19)), rows),
    }
    frame = pd.DataFrame({"_oci_row_id": np.arange(rows) * 2, **columns})
    before = frame.copy(deep=True)
    definitions = [
        {
            "name": name,
            "feature_id": f"f_{name}",
            "value_type": "continuous" if name.startswith("continuous") else "categorical",
            "roles": ["confounder"],
        }
        for name in columns
    ]
    filtered, retained, summary = _filter(frame, definitions, tmp_path)
    expected = ["brain_metastases", "boundary_count", "continuous_varied"]
    assert [f["name"] for f in retained] == expected
    pd.testing.assert_frame_equal(filtered, before[["_oci_row_id", *expected]])
    pd.testing.assert_frame_equal(frame, before)
    audit = json.loads((tmp_path / "filter.json").read_text())
    reasons = {f["name"]: f["reason"] for f in audit["features"]}
    assert reasons["nearly_constant"] == "fewer_than_two_levels_with_min_informative_patients"
    assert reasons["all_same_category"] == "fewer_than_two_levels_with_min_informative_patients"
    assert reasons["below_count"] == "fewer_than_min_informative_patients_observed"
    assert reasons["continuous_constant"] == "constant_observed_values"
    assert reasons["continuous_sparse"] == "fewer_than_min_informative_patients_observed"
    by_name = {f["name"]: f for f in audit["features"]}
    assert by_name["brain_metastases"]["level_counts"] == {
        "__not_documented__": 775, "present": 25,
    }
    assert summary["min_informative_patients"] == 20
    assert summary["dropped_by_reason"]["constant_observed_values"] == 1
    assert "not documented" in summary["description"]
    # A stricter K is honored, not a hard-coded fraction.
    _, stricter, _ = _filter(frame, definitions, tmp_path, min_informative_patients=26)
    assert [f["name"] for f in stricter] == ["continuous_varied"]


def test_rule_scales_with_counts_not_fractions(tmp_path):
    rows = 10_000
    frame = pd.DataFrame({"_oci_row_id": range(rows), "rare": _padded(["yes"] * 60, rows)})
    definition = [{"name": "rare", "feature_id": "rare", "value_type": "binary"}]
    # 0.6% observed would have been dropped by the former 5% rule.
    _, retained, _ = _filter(frame, definition, tmp_path)
    assert [f["name"] for f in retained] == ["rare"]


def test_extraction_failures_are_not_not_documented_and_unreliable_features_drop(tmp_path):
    rows = 200
    rng = np.random.default_rng(3)
    treatment = np.tile([0.0, 1.0], rows // 2)
    values = {
        "reliable": list(rng.normal(size=rows)),
        "unreliable": list(rng.normal(size=rows)),
        "category": _padded(["a"] * 60 + ["b"] * 60, rows),
    }
    frame = pd.DataFrame({"_oci_row_id": range(rows), **values})
    mask = pd.DataFrame({"_oci_row_id": range(rows), "reliable": 0, "unreliable": 0, "category": 0})
    # 5% failures in treated rows only for "reliable"; 15% for "unreliable".
    reliable_failed = np.flatnonzero(treatment == 1)[:10]
    frame.loc[reliable_failed, "reliable"] = np.nan
    mask.loc[reliable_failed, "reliable"] = 1
    frame.loc[:29, "unreliable"] = np.nan
    mask.loc[:29, "unreliable"] = 1
    # Failed cells among the undocumented tail of the categorical feature.
    frame.loc[150:169, "category"] = None
    mask.loc[150:169, "category"] = 1
    definitions = [
        {"name": "reliable", "feature_id": "reliable", "value_type": "continuous"},
        {"name": "unreliable", "feature_id": "unreliable", "value_type": "continuous"},
        {"name": "category", "feature_id": "category", "value_type": "categorical"},
    ]
    filtered, retained, summary = _filter(
        frame, definitions, tmp_path, failure_mask=mask, treatment=treatment,
    )
    assert [f["name"] for f in retained] == ["reliable", "category"]
    assert all(f["extraction_failure_indicator"] for f in retained)
    audit = {f["name"]: f for f in json.loads((tmp_path / "filter.json").read_text())["features"]}
    assert audit["unreliable"]["reason"] == "extraction_failure_fraction_above_maximum"
    assert audit["unreliable"]["extraction_failure_fraction"] == pytest.approx(0.15)
    # 80 undocumented rows minus 20 failures: failures are not "not documented".
    assert audit["category"]["not_documented_rows"] == 60
    assert audit["category"]["extraction_failed_rows"] == 20
    assert filtered[extraction_failure_column("reliable")].sum() == 10
    assert filtered[extraction_failure_column("category")].sum() == 20
    report = {r["name"]: r for r in summary["training_extraction_failure_report"]["per_feature"]}
    assert report["reliable"]["by_treatment_arm"]["1"]["failed_rows"] == 10
    assert report["reliable"]["by_treatment_arm"]["0"]["failed_rows"] == 0
    assert report["reliable"]["arm_difference_treated_minus_control"] == pytest.approx(0.1)
    # A permissive threshold keeps the unreliable feature, with its indicator.
    _, permissive, _ = _filter(
        frame, definitions, tmp_path, failure_mask=mask, max_extraction_failure_fraction=0.2,
    )
    assert [f["name"] for f in permissive] == ["reliable", "unreliable", "category"]


@pytest.mark.parametrize("disjoint_observations", [False, True])
def test_all_candidates_dropped_fails_even_when_union_coverage_is_adequate(
    tmp_path, disjoint_observations,
):
    frame = pd.DataFrame({"_oci_row_id": range(40), "a": np.nan, "b": np.nan, "c": np.nan})
    if disjoint_observations:
        for row, name in enumerate(["a", "b", "c"]):
            frame.loc[row, name] = 1
    with pytest.raises(ValueError, match="catastrophically sparse"):
        _filter(frame, [{"name": name, "feature_id": name} for name in ["a", "b", "c"]], tmp_path)
    audit = json.loads((tmp_path / "filter.json").read_text())
    assert audit["status"] == "failed" and audit["candidates_dropped"] == 3
    health = json.loads((tmp_path / "health.json").read_text())
    assert health["status"] == "failed"
    assert health["rows_with_any_nonmissing"] == (3 if disjoint_observations else 0)


def test_filter_rejects_invalid_thresholds(tmp_path):
    frame = pd.DataFrame({"_oci_row_id": range(4), "a": [1, 2, 3, 4]})
    for kwargs in ({"min_informative_patients": 0}, {"min_informative_patients": 2.5},
                   {"max_extraction_failure_fraction": 1.5}):
        with pytest.raises(ValueError, match="stage2"):
            _filter(frame, [{"name": "a", "feature_id": "a"}], tmp_path, **kwargs)


def test_stage2_config_wires_count_rule_and_rejects_retired_fraction_keys():
    base = {"endpoint": "http://localhost:8000/v1", "model": "m"}
    config = plain_stage2_config_from_mapping(base, default_workers=1)
    assert config.min_informative_patients == 20
    assert config.max_extraction_failure_fraction == pytest.approx(0.10)
    custom = plain_stage2_config_from_mapping(
        {**base, "min_informative_patients": 40, "max_extraction_failure_fraction": 0.2},
        default_workers=1,
    )
    assert custom.min_informative_patients == 40
    assert custom.public_dict()["max_extraction_failure_fraction"] == pytest.approx(0.2)
    for retired in ("min_nonmissing_fraction", "max_dominant_fraction"):
        with pytest.raises(ValueError, match="unsupported fields"):
            plain_stage2_config_from_mapping({**base, retired: 0.05}, default_workers=1)
    for bad in ({"min_informative_patients": 0}, {"min_informative_patients": True},
                {"max_extraction_failure_fraction": -0.1}):
        with pytest.raises(ValueError, match="stage2"):
            plain_stage2_config_from_mapping({**base, **bad}, default_workers=1)
