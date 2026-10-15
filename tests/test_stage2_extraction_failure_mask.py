"""Extraction failures after all repairs are masked, not read as "not documented" (S2A-004)."""

import json

import numpy as np
import pandas as pd

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_elastic_net_selection as linear
from oci.inference import stage2_extraction_failures as failures
from oci.inference.stage2_statistical_selection import _encode_feature


DEFINITIONS = [
    {"feature_id": "f_age", "name": "age", "value_type": "continuous",
     "categories_or_unit": ["years"]},
    {"feature_id": "f_bp", "name": "blood_pressure", "value_type": "continuous",
     "categories_or_unit": ["mmHg"]},
]


def _extract(tmp_path):
    dataset = pd.DataFrame({"clinical_text": ["Record R0.", "Record R1.", "Record R2."]})

    def request_json(messages, validate, *, request_kind="interpretation"):
        text = json.dumps(messages)
        if "R2" in text:
            raise analysis.Stage2ResponseValidationError(
                "Stage 2 response remained invalid after 10 repairs: Unterminated string"
            )
        row = 1 if "R1" in text else 0
        values = {"age": 60 + row, "blood_pressure": None}
        if row == 1:
            values["blood_pressure"] = {"systolic": 147, "diastolic": 93}  # invalid scalar
        return validate({"rows": [{"row_id": row, "values": values}]})

    output = tmp_path / "extraction"
    frame = analysis.extract_rows(
        dataset=dataset, row_ids=[0, 1, 2], text_column="clinical_text",
        definitions=DEFINITIONS, output_dir=output, request_json=request_json,
        workers=1, max_prompt_chars=10_000,
    )
    summary = json.loads((output / "failure_summary.json").read_text())
    return frame, summary


def test_terminal_failure_cells_and_mask_from_real_extraction(tmp_path):
    frame, summary = _extract(tmp_path)
    # Row 0's null blood pressure is "not documented"; rows 1-2 failed.
    assert pd.isna(frame.loc[0, "blood_pressure"]) and frame.loc[0, "age"] == 60
    assert summary["terminal_failure_cells"] == {"age": [2], "blood_pressure": [1, 2]}
    assert "after all repairs" in summary["terminal_failure_policy"]
    issues = json.loads(
        next((tmp_path / "extraction").rglob("batch_*/extraction_failure.json"))
        .with_name("extraction_issues.json").read_text()
    )
    structural = [e for e in issues["events"] if e["failure_kind"] == "structural_response_failure"]
    assert structural and structural[0]["feature_names"] == ["age", "blood_pressure"]
    mask = failures.failure_mask(frame, summary, DEFINITIONS)
    assert mask["age"].tolist() == [0, 0, 1]
    assert mask["blood_pressure"].tolist() == [0, 1, 1]
    # A cell named by an event but holding a final value is not a failure.
    recovered = frame.copy()
    recovered.loc[2, "age"] = 70.0
    assert failures.failure_mask(recovered, summary, DEFINITIONS)["age"].tolist() == [0, 0, 0]
    report = failures.failure_rate_report(mask, DEFINITIONS, scope="t", treatment=[0, 1, 1])
    by_name = {r["name"]: r for r in report["per_feature"]}
    assert by_name["blood_pressure"]["by_treatment_arm"]["1"]["failed_rows"] == 2
    assert by_name["blood_pressure"]["by_treatment_arm"]["0"]["failure_fraction"] == 0.0
    assert report["rows_with_any_failure"] == 2


def test_incremental_merge_replaces_changed_feature_failure_cells():
    prior = {"feature_failure_patterns": [], "terminal_failure_cells": {"a": [1, 2], "b": [3]}}
    delta = {"feature_failure_patterns": [], "terminal_failure_cells": {"b": [5]}}
    merged = analysis._merge_incremental_failure_summaries(
        prior_summary=prior, delta_summary=delta,
        current_feature_names=["a", "b", "c"], changed_feature_names=["b", "c"],
    )
    assert merged["terminal_failure_cells"] == {"a": [1, 2], "b": [5]}
    try:
        analysis._merge_incremental_failure_summaries(
            prior_summary=prior, delta_summary={"terminal_failure_cells": {"a": [9]}},
            current_feature_names=["a", "b"], changed_feature_names=["b"],
        )
    except ValueError as exc:
        assert "not re-extracted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("delta failures for reused features must be rejected")


def _frame_and_definitions():
    rows = 12
    values = pd.DataFrame({
        "_oci_row_id": range(rows),
        "lab": [1.0, 2.0, 3.0, np.nan, np.nan, 4.0, 5.0, 6.0, np.nan, 7.0, 8.0, 9.0],
        "stage": ["I", "II", None, None, "I", "II", "I", None, "II", "I", "II", "I"],
    })
    failed_lab = [3]  # row 3 failed; rows 4 and 8 are not documented
    failed_stage = [2]  # row 2 failed; rows 3 and 7 are not documented
    values[failures.extraction_failure_column("lab")] = [int(i in failed_lab) for i in range(rows)]
    values[failures.extraction_failure_column("stage")] = [
        int(i in failed_stage) for i in range(rows)
    ]
    definitions = [
        {"feature_id": "lab", "name": "lab", "value_type": "continuous",
         "extraction_failure_indicator": True},
        {"feature_id": "stage", "name": "stage", "value_type": "categorical",
         "categories_or_unit": ["I", "II"], "extraction_failure_indicator": True},
    ]
    return values, definitions


def test_feature_encoder_separates_failure_from_not_documented():
    frame, definitions = _frame_and_definitions()
    encoded = analysis._FeatureEncoder(definitions).fit(frame).transform(frame)
    # lab: value, missing, failed | stage: I, II, __missing__, failed
    assert encoded.shape[1] == 3 + 4
    lab_missing, lab_failed = encoded[:, 1], encoded[:, 2]
    assert lab_missing.tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    assert lab_failed.tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    stage_missing, stage_failed = encoded[:, 5], encoded[:, 6]
    assert stage_missing.tolist() == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    assert stage_failed.tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert encoded[2, 3:5].tolist() == [0.0, 0.0]  # no category level for a failed cell
    # Unflagged definitions keep the original encoding (failure read as missing).
    plain = [{k: v for k, v in d.items() if k != "extraction_failure_indicator"}
             for d in definitions]
    legacy = analysis._FeatureEncoder(plain).fit(frame).transform(frame)
    assert legacy.shape[1] == 2 + 3
    assert legacy[3, 1] == 1.0 and legacy[2, 4] == 1.0


def test_selection_designs_group_failure_indicator_with_parent_feature():
    frame, definitions = _frame_and_definitions()
    design = linear._encode_design(frame, frame, definitions, categorical_min_count=2)
    names = list(design.column_names)
    assert "lab:extraction_failed" in names and "stage:extraction_failed" in names
    for name in ("lab", "stage"):
        index = names.index(f"{name}:extraction_failed")
        assert design.column_feature_ids[index] == name  # grouped for penalties/permutation
        missing = design.train[:, names.index(f"{name}:missing")]
        failed = design.train[:, index]
        # Standardized columns: failure and not-documented rows never coincide.
        assert np.corrcoef(missing, failed)[0, 1] < 0
    # Groups stay contiguous by feature, as the group elastic net requires.
    linear._group_structure(design.column_feature_ids)
    univariable = _encode_feature(frame, definitions[0])
    assert univariable.main_names[-1] == "lab:extraction_failed"
    assert univariable.main[:, -1].tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # A training partition without failures drops the constant indicator column.
    no_failures = frame.drop(index=[2, 3]).reset_index(drop=True)
    design = linear._encode_design(no_failures, frame, definitions, categorical_min_count=2)
    assert not any(n.endswith(":extraction_failed") for n in design.column_names)


def test_attach_failure_columns_only_for_flagged_definitions():
    frame = pd.DataFrame({"_oci_row_id": [5, 6, 7], "a": [1, None, 2], "b": [None, 1, 1]})
    mask = pd.DataFrame({"_oci_row_id": [7, 6, 5], "a": [0, 1, 0], "b": [1, 0, 0]})
    definitions = [{"name": "a", "extraction_failure_indicator": True}, {"name": "b"}]
    result = failures.attach_failure_columns(frame, mask, definitions)
    assert result[failures.extraction_failure_column("a")].tolist() == [0, 1, 0]
    assert failures.extraction_failure_column("b") not in result
    assert failures.companion_columns(result, definitions) == ["a__oci_extraction_failed"]
