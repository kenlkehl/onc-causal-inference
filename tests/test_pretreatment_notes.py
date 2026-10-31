"""Dated-notes input: classification, headers, validation, artifacts, and resume."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference import pretreatment_notes as notes_module
from oci.inference import research_all_evidence_workflow as workflow
from oci.inference.pretreatment_notes import (
    NOTE_CATEGORIES,
    PATIENT_AUDIT_CATEGORIES,
    build_pretreatment_text,
    notes_config_from_mapping,
)
from tests.test_research_all_evidence_workflow import _inputs

CALENDAR_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _config(**overrides):
    raw = {"path": "notes.parquet", "index_date_column": "index_date", **overrides}
    return notes_config_from_mapping(
        raw,
        config_dir=Path("/tmp/oci-notes-test"),
        default_unit_id_column="patient_id",
    )


def _build(cohort, notes, **overrides):
    return build_pretreatment_text(
        cohort,
        notes,
        config=_config(**overrides),
        unit_id_column="patient_id",
        text_column="clinical_text",
    )


def _cohort(ids=("a", "b"), index_dates=("2024-03-10", "2024-03-10")):
    return pd.DataFrame(
        {
            "patient_id": list(ids),
            "index_date": list(index_dates),
            "treatment_indicator": [index % 2 for index in range(len(ids))],
            "outcome_indicator": [0] * len(ids),
        }
    )


def _notes(rows):
    return pd.DataFrame(rows, columns=["patient_id", "note_date", "note_text"])


# --------------------------------------------------------------------------
# Classification, ordering, and headers


def test_each_note_falls_in_exactly_one_category():
    notes = _notes(
        [
            ("a", "2024-03-01", "before"),
            ("a", "2024-03-10", "on index day"),
            ("a", "2024-03-11", "after"),
            ("a", None, "no date"),
            ("a", "not a date", "garbled date"),
            ("zz", "2024-03-01", "orphan"),
            ("a", "2024-03-02", "   "),
            ("a", "2024-03-03", None),
            ("b", "2024-03-12", "after but empty "),
        ]
    )
    result = _build(_cohort(), notes)

    assert result.note_categories == [
        "retained",
        "on_index_date",
        "after_index_date",
        "undated",
        "undated",
        "unmatched_patient",
        "empty_text",
        "empty_text",
        "after_index_date",
    ]
    assert result.note_counts == {
        "retained": 1,
        "on_index_date": 1,
        "after_index_date": 2,
        "before_lookback": 0,
        "undated": 2,
        "unmatched_patient": 1,
        "empty_text": 2,
        "total": 9,
    }
    assert sum(result.note_counts[name] for name in NOTE_CATEGORIES) == len(notes)
    assert result.texts == ["[Note 1 of 1 · 9 days before index date]\nbefore", ""]


def test_include_index_date_keeps_index_day_notes():
    notes = _notes([("a", "2024-03-10", "decision"), ("a", "2024-03-11", "after")])

    excluded = _build(_cohort(), notes)
    included = _build(_cohort(), notes, include_index_date=True)

    assert excluded.note_categories == ["on_index_date", "after_index_date"]
    assert excluded.texts[0] == ""
    assert included.note_categories == ["retained", "after_index_date"]
    assert included.texts[0] == "[Note 1 of 1 · on index date]\ndecision"


def test_headers_with_note_type_and_without_headers_never_show_calendar_dates():
    notes = pd.DataFrame(
        {
            "patient_id": ["a", "a", "a"],
            "note_date": ["2023-12-01", "2024-03-09", "2024-02-01"],
            "note_text": ["Baseline visit.", "Clinic note.", "Imaging."],
            "kind": ["Progress  note", None, "Radiology report"],
        }
    )
    plain = _build(_cohort(), notes)
    typed = _build(_cohort(), notes, columns={"note_type": "kind"})
    bare = _build(_cohort(), notes, columns={"note_type": "kind"}, note_header="none")

    assert plain.texts[0] == (
        "[Note 1 of 3 · 100 days before index date]\nBaseline visit.\n\n"
        "[Note 2 of 3 · 38 days before index date]\nImaging.\n\n"
        "[Note 3 of 3 · 1 day before index date]\nClinic note."
    )
    assert typed.texts[0] == (
        "[Note 1 of 3 · 100 days before index date · Progress note]\nBaseline visit.\n\n"
        "[Note 2 of 3 · 38 days before index date · Radiology report]\nImaging.\n\n"
        "[Note 3 of 3 · 1 day before index date]\nClinic note."
    )
    assert bare.texts[0] == "Baseline visit.\n\nImaging.\n\nClinic note."
    for text in (*plain.texts, *typed.texts, *bare.texts):
        assert not CALENDAR_DATE.search(text)


# --------------------------------------------------------------------------
# Fail-closed validation


@pytest.mark.parametrize(
    ("cohort", "message"),
    [(_cohort(index_dates=("2024-03-10", None)), "index date")],
)
def test_cohort_validation_fails_closed(cohort, message):
    notes = _notes([("a", "2024-03-01", "note")])

    with pytest.raises(ValueError, match=message):
        _build(cohort, notes)


@pytest.mark.parametrize(
    ("raw", "message"),
    [({"path": "n.csv", "index_date_column": "d", "lookback": 30}, "unsupported fields")],
)
def test_notes_block_validation_rejects_unknown_or_invalid_settings(tmp_path, raw, message):
    with pytest.raises(ValueError, match=message):
        notes_config_from_mapping(raw, config_dir=tmp_path, default_unit_id_column="patient_id")


# --------------------------------------------------------------------------
# Workflow fixtures


COHORT_IDS = ["a", "b", "c", "d", "e", "f"]


def _write_dated_notes(raw, tmp_path, *, notes_suffix=".parquet"):
    """Turn the shared workflow fixture into a dated-notes configuration."""

    pd.DataFrame(
        {
            "patient_id": COHORT_IDS,
            "index_date": ["2024-03-10"] * 6,
            "treatment_indicator": [0, 1, 0, 1, 0, 1],
            "outcome_indicator": [0, 0, 1, 1, 0, 1],
            "true_ite": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    ).to_parquet(tmp_path / "cohort.parquet", index=False)
    words = ["one", "two", "three", "four", "five", "six"]
    rows = []
    for patient, word in zip(COHORT_IDS, words):
        rows.extend(
            [
                (patient, "2024-03-01", f"{word} early", 0.9),
                (patient, "2024-03-08", f"{word} later", 0.9),
                (patient, "2024-03-10", f"{word} decision", 0.9),
                (patient, "2024-03-20", f"{word} response", 0.9),
            ]
        )
    rows.append(("ghost", "2024-03-01", "unmatched", 0.0))
    notes = pd.DataFrame(rows, columns=["patient_id", "note_date", "note_text", "true_ite"])
    notes["oracle_role"] = "confounder"
    path = tmp_path / f"notes{notes_suffix}"
    if notes_suffix == ".csv":
        notes.to_csv(path, index=False)
    else:
        notes.to_parquet(path, index=False)
    raw["notes"] = {"path": path.name, "index_date_column": "index_date"}
    return raw


def _notes_inputs(tmp_path, *, components=(), notes_suffix=".parquet"):
    raw, _config = _inputs(tmp_path, components=components)
    _write_dated_notes(raw, tmp_path, notes_suffix=notes_suffix)
    return raw, workflow.compile_config(raw, config_dir=tmp_path)


def _expected_text(word):
    return (
        f"[Note 1 of 2 · 9 days before index date]\n{word} early\n\n"
        f"[Note 2 of 2 · 2 days before index date]\n{word} later"
    )


def _input_files(output_dir):
    directory = output_dir / "input"
    return {
        path.name: (path.stat().st_mtime_ns, path.read_bytes())
        for path in sorted(directory.iterdir())
    }


# --------------------------------------------------------------------------
# Artifacts, loaders, and resume


@pytest.mark.parametrize("notes_suffix", [".parquet"])
def test_manifest_audit_and_analysis_cohort_contents(tmp_path, notes_suffix):
    _raw, config = _notes_inputs(tmp_path, notes_suffix=notes_suffix)

    dataset = workflow._load_analysis_dataset(config, materialize=True)

    input_dir = config.output_dir / "input"
    assert sorted(path.name for path in input_dir.iterdir()) == [
        "analysis_cohort.parquet",
        "pretreatment_notes_audit.csv",
        "pretreatment_notes_manifest.json",
    ]
    cohort = pd.read_parquet(tmp_path / "cohort.parquet")
    assert list(dataset.columns) == [*cohort.columns, "clinical_text"]
    pd.testing.assert_frame_equal(dataset.drop(columns="clinical_text"), cohort)
    assert dataset["clinical_text"].tolist() == [
        _expected_text(word) for word in ["one", "two", "three", "four", "five", "six"]
    ]
    # Oracle columns of the notes table are never read.
    assert "oracle_role" not in dataset.columns
    assert dataset["true_ite"].tolist() == cohort["true_ite"].tolist()

    manifest = json.loads((input_dir / "pretreatment_notes_manifest.json").read_text())
    assert manifest["schema_version"] == "pretreatment_notes_manifest_v1"
    assert manifest["note_counts"] == {
        "retained": 12,
        "on_index_date": 6,
        "after_index_date": 6,
        "before_lookback": 0,
        "undated": 0,
        "unmatched_patient": 1,
        "empty_text": 0,
        "total": 25,
    }
    assert sum(manifest["note_counts"][name] for name in NOTE_CATEGORIES) == 25
    assert manifest["patients"] == 6
    assert manifest["patients_without_retained_notes"] == 0
    assert manifest["retained_notes_per_patient"] == {"min": 2, "median": 2.0, "max": 2}
    lengths = [len(text) for text in dataset["clinical_text"]]
    assert manifest["characters_per_patient"] == {
        "min": min(lengths),
        "median": float(np.median(lengths)),
        "max": max(lengths),
    }
    assert manifest["sources"]["cohort"]["path"] == str(config.dataset)
    assert manifest["sources"]["cohort"]["sha256"] == notes_module.file_sha256(config.dataset)
    assert manifest["sources"]["notes"]["path"] == str(config.notes.path)
    assert manifest["sources"]["notes"]["sha256"] == notes_module.file_sha256(config.notes.path)
    assert manifest["notes_config"] == config.notes.as_dict()
    assert manifest["cohort_columns"] == {"unit_id": "patient_id", "text": "clinical_text"}
    analysis = input_dir / "analysis_cohort.parquet"
    assert manifest["analysis_cohort"]["sha256"] == notes_module.file_sha256(analysis)
    assert manifest["analysis_cohort"]["columns"] == list(dataset.columns)
    assert manifest["created_at"].endswith("Z")

    audit = pd.read_csv(input_dir / "pretreatment_notes_audit.csv")
    assert list(audit.columns) == [
        "unit_id",
        *PATIENT_AUDIT_CATEGORIES,
        "retained_characters",
        "earliest_retained_days_before_index",
        "latest_retained_days_before_index",
    ]
    assert audit["unit_id"].tolist() == COHORT_IDS
    assert not {"treatment_indicator", "outcome_indicator", "true_ite"} & set(audit.columns)
    assert audit[list(PATIENT_AUDIT_CATEGORIES)].to_numpy().sum() + 1 == 25
    assert audit["retained"].tolist() == [2] * 6
    assert audit["retained_characters"].tolist() == lengths
    assert audit["earliest_retained_days_before_index"].tolist() == [9] * 6
    assert audit["latest_retained_days_before_index"].tolist() == [2] * 6


def test_three_loaders_return_the_derived_cohort_and_reuse_it(tmp_path):
    _raw, config = _notes_inputs(tmp_path)

    stage1 = workflow.ResearchAllEvidenceWorkflow(config)._resolved_context().dataset
    before = _input_files(config.output_dir)
    stage2_config = replace(config, components=("stage2",))
    stage2 = workflow.ResearchAllEvidenceWorkflow(stage2_config)._stage2_only_context().dataset
    reselection = workflow._load_reselection_dataset(stage2_config)
    resumed = workflow.ResearchAllEvidenceWorkflow(config)._resolved_context().dataset

    for frame in (stage2, reselection, resumed):
        pd.testing.assert_frame_equal(frame, stage1)
    assert stage1["clinical_text"].tolist()[0] == _expected_text("one")
    # Reuse never rewrites the derived files, so Stage 1 checkpoint identities hold.
    assert _input_files(config.output_dir) == before

    analysis = config.output_dir / "input" / "analysis_cohort.parquet"
    assert config.analysis_dataset == analysis
    resolved = json.loads((config.output_dir / "resolved_stage1_model_config.json").read_text())
    assert resolved["dataset_path"] == str(analysis)
    saved = json.loads((config.output_dir / "run_config.json").read_text())
    assert saved["dataset"] == str(config.dataset)
    assert saved["notes"] == config.notes.as_dict()


@pytest.mark.parametrize("change", ["notes_file", "notes_config", "tampered"])
def test_resume_fails_closed_when_inputs_or_settings_change(tmp_path, change):
    raw, config = _notes_inputs(tmp_path)
    workflow._load_analysis_dataset(config, materialize=True)
    before = _input_files(config.output_dir)

    if change == "notes_file":
        notes = pd.read_parquet(config.notes.path)
        notes.loc[0, "note_text"] = "edited"
        notes.to_parquet(config.notes.path, index=False)
        message = "notes file contents differ"
    elif change == "cohort_file":
        cohort = pd.read_parquet(config.dataset)
        cohort.loc[0, "index_date"] = "2024-03-09"
        cohort.to_parquet(config.dataset, index=False)
        message = "cohort file contents differ"
    elif change == "notes_config":
        raw["notes"]["include_index_date"] = True
        config = workflow.compile_config(raw, config_dir=tmp_path)
        message = "notes configuration differs"
    else:
        analysis = config.output_dir / "input" / "analysis_cohort.parquet"
        frame = pd.read_parquet(analysis)
        frame.loc[0, "clinical_text"] = "tampered"
        frame.to_parquet(analysis, index=False)
        before = _input_files(config.output_dir)
        message = "derived analysis cohort differs"

    with pytest.raises(ValueError, match=message) as error:
        workflow._load_analysis_dataset(config, materialize=True)
    assert "fresh output directory" in str(error.value)
    assert _input_files(config.output_dir) == before


# --------------------------------------------------------------------------
# Configuration round trip and command line


# --------------------------------------------------------------------------
# Stage 2 consumers
