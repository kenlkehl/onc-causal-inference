"""Training-only, count-based candidate informativeness rule (B4 / S2A-003)."""

import json

import numpy as np
import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference.stage2_extraction_failures import extraction_failure_column


def _padded(values, rows):
    return list(values) + [None] * (rows - len(values))


def _filter(frame, definitions, tmp_path, **kwargs):
    return analysis._filter_sparse_training_candidates(
        frame, definitions, audit_path=tmp_path / "filter.json",
        health_audit_path=tmp_path / "health.json", **kwargs,
    )


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
