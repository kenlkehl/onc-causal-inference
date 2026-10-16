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


def test_lookback_drops_notes_older_than_the_window():
    notes = _notes(
        [
            ("a", "2024-02-29", "exactly ten days"),
            ("a", "2024-02-28", "eleven days"),
            ("a", "2024-03-09", "one day"),
        ]
    )
    result = _build(_cohort(), notes, lookback_days=10)

    assert result.note_categories == ["retained", "before_lookback", "retained"]
    assert result.texts[0] == (
        "[Note 1 of 2 · 10 days before index date]\nexactly ten days\n\n"
        "[Note 2 of 2 · 1 day before index date]\none day"
    )
    assert _build(_cohort(), notes).note_categories == ["retained"] * 3


def test_retained_notes_are_ordered_by_date_with_stable_ties():
    notes = _notes(
        [
            ("a", "2024-03-05", "third"),
            ("b", "2024-01-01", "other patient"),
            ("a", "2024-03-01 18:00", "first, evening"),
            ("a", "2024-03-01 08:00", "second, morning but later row"),
            ("a", "2024-03-09", "fourth"),
        ]
    )
    result = _build(_cohort(), notes, note_header="none")

    assert result.texts[0] == (
        "first, evening\n\nsecond, morning but later row\n\nthird\n\nfourth"
    )
    assert result.texts[1] == "other patient"


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


def test_timezone_aware_and_timestamp_dates_compare_by_calendar_day():
    cohort = _cohort(ids=("a", "b"), index_dates=("2024-03-02", "2024-03-10 15:45"))
    notes = _notes(
        [
            # 23:30 at UTC-5 is 04:30 UTC on 2024-03-02, the index day.
            ("a", "2024-03-01T23:30:00-05:00", "late evening, US Eastern"),
            ("a", "2024-03-01 23:59:59", "naive timestamp, day before"),
            ("b", "2024-03-10 01:00", "same day, earlier hour"),
            ("b", "2024-03-09 23:00", "day before"),
        ]
    )
    result = _build(cohort, notes)

    assert result.note_categories == [
        "on_index_date",
        "retained",
        "on_index_date",
        "retained",
    ]

    aware_cohort = _cohort(ids=("a",), index_dates=("2024-03-02",))
    aware_cohort["index_date"] = pd.to_datetime(["2024-03-02 03:00"]).tz_localize("UTC")
    aware_notes = _notes([("a", "2024-03-01T22:00:00-05:00", "03:00 UTC on the index day")])
    aware_notes["note_date"] = pd.to_datetime(aware_notes["note_date"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    assert _build(aware_cohort, aware_notes).note_categories == ["on_index_date"]


def test_notes_without_matching_patients_or_retained_notes_leave_empty_text():
    notes = _notes([("a", "2024-03-01", "kept"), ("a", "2024-04-01", "after")])
    result = _build(_cohort(ids=("a", "b", "c"), index_dates=("2024-03-10",) * 3), notes)

    assert result.texts[1:] == ["", ""]
    assert result.audit["retained"].tolist() == [1, 0, 0]
    assert result.audit["earliest_retained_days_before_index"].isna().tolist() == [
        False,
        True,
        True,
    ]


# --------------------------------------------------------------------------
# Fail-closed validation


@pytest.mark.parametrize(
    ("cohort", "message"),
    [
        (_cohort(ids=("a", "a")), "unique"),
        (_cohort(ids=("a", None)), "nonmissing"),
        (_cohort(index_dates=("2024-03-10", None)), "index date"),
        (_cohort(index_dates=("2024-03-10", "soon")), "index date"),
        (_cohort().assign(clinical_text=["x", "y"]), "already has a column"),
        (_cohort().drop(columns="index_date"), "missing columns"),
        (_cohort().assign(index_date=[20240310, 20240310]), "numbers, not dates"),
    ],
)
def test_cohort_validation_fails_closed(cohort, message):
    notes = _notes([("a", "2024-03-01", "note")])

    with pytest.raises(ValueError, match=message):
        _build(cohort, notes)


def test_missing_note_columns_and_unmatched_identifier_types_fail_closed(tmp_path):
    notes_path = tmp_path / "notes.csv"
    pd.DataFrame({"patient_id": ["a"], "note_text": ["x"]}).to_csv(notes_path, index=False)
    config = replace(_config(), path=notes_path)

    with pytest.raises(ValueError, match="missing configured columns: \\['note_date'\\]"):
        notes_module.read_notes_table(config)
    with pytest.raises(ValueError, match="matches a cohort"):
        _build(_cohort(ids=(1, 2)), _notes([("1", "2024-03-01", "string identifiers")]))


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"index_date_column": "index_date"}, "notes.path is required"),
        ({"path": "notes.csv"}, "index_date_column is required"),
        ({"path": "n.csv", "index_date_column": "d", "lookback": 30}, "unsupported fields"),
        ({"path": "n.csv", "index_date_column": "d", "columns": {"type": "x"}}, "unsupported"),
        ({"path": "n.csv", "index_date_column": "d", "include_index_date": "yes"}, "true or false"),
        ({"path": "n.csv", "index_date_column": "d", "lookback_days": 0}, "positive integer"),
        ({"path": "n.csv", "index_date_column": "d", "lookback_days": True}, "positive integer"),
        ({"path": "n.csv", "index_date_column": "d", "note_header": "dates"}, "note_header"),
        ({"path": "n.csv", "index_date_column": "d", "columns": {"text": "note_date"}}, "distinct"),
        ("notes.csv", "configuration object"),
    ],
)
def test_notes_block_validation_rejects_unknown_or_invalid_settings(tmp_path, raw, message):
    with pytest.raises(ValueError, match=message):
        notes_config_from_mapping(raw, config_dir=tmp_path, default_unit_id_column="patient_id")


def test_notes_block_defaults_and_relative_path(tmp_path):
    config = notes_config_from_mapping(
        {"path": "data/notes.parquet", "index_date_column": "index_date"},
        config_dir=tmp_path,
        default_unit_id_column="mrn",
    )

    assert config.path == (tmp_path / "data" / "notes.parquet").resolve()
    assert (config.unit_id_column, config.date_column, config.text_column) == (
        "mrn",
        "note_date",
        "note_text",
    )
    assert config.note_type_column is None
    assert config.include_index_date is False
    assert config.lookback_days is None
    assert config.note_header == "relative_days"


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


@pytest.mark.parametrize("notes_suffix", [".parquet", ".csv"])
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


def test_interrupted_dated_notes_run_resumes_on_the_same_analysis_cohort(tmp_path):
    _raw, config = _notes_inputs(tmp_path, components=("tfidf", "text_models"))
    seen = []

    def runner(name, *, fail=False):
        def run(context, component_dir):
            seen.append((name, context.dataset["clinical_text"].tolist()))
            (component_dir / "result.txt").write_text(name, encoding="utf-8")
            if fail:
                raise RuntimeError("interrupted model")

        return run

    with pytest.raises(RuntimeError, match="interrupted model"):
        workflow.ResearchAllEvidenceWorkflow(
            config,
            component_runners={"tfidf": runner("tfidf"), "text_models": runner("t", fail=True)},
        ).run()
    before = _input_files(config.output_dir)

    resumed = workflow.ResearchAllEvidenceWorkflow(
        config,
        component_runners={"tfidf": runner("tfidf"), "text_models": runner("text_models")},
    ).run()

    assert resumed["components"]["tfidf"]["status"] == "already_complete"
    assert [name for name, _texts in seen] == ["tfidf", "t", "text_models"]
    assert all(texts == seen[0][1] for _name, texts in seen)
    assert _input_files(config.output_dir) == before


def test_stage2_only_without_a_built_cohort_fails_without_writing(tmp_path):
    _raw, config = _notes_inputs(tmp_path)
    stage2_config = replace(config, components=("stage2",))

    with pytest.raises(FileNotFoundError, match="has not been built"):
        workflow._load_reselection_dataset(stage2_config)
    assert not (config.output_dir / "input").exists()


@pytest.mark.parametrize("change", ["notes_file", "cohort_file", "notes_config", "tampered"])
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


def test_text_column_change_on_resume_fails_closed(tmp_path):
    raw, config = _notes_inputs(tmp_path)
    workflow._load_analysis_dataset(config, materialize=True)
    raw["columns"] = {"text": "note_history"}
    changed = workflow.compile_config(raw, config_dir=tmp_path)

    with pytest.raises(ValueError, match="columns.unit_id or columns.text differs"):
        workflow._load_analysis_dataset(changed, materialize=True)


def test_dated_notes_and_single_text_runs_do_not_share_an_output_directory(tmp_path):
    raw, config = _notes_inputs(tmp_path)
    workflow._load_analysis_dataset(config, materialize=True)
    plain_raw = {key: value for key, value in raw.items() if key != "notes"}
    plain = workflow.compile_config(plain_raw, config_dir=tmp_path)

    with pytest.raises(ValueError, match="holds a dated-notes run"):
        workflow.ResearchAllEvidenceWorkflow(plain)
    with pytest.raises(ValueError, match="holds a dated-notes run"):
        workflow._load_reselection_dataset(plain)

    other = tmp_path / "other"
    other.mkdir()
    existing_raw, existing = _inputs(other)
    existing_output = existing.output_dir
    (existing_output / "components" / "tfidf").mkdir(parents=True)
    (existing_output / "components" / "tfidf" / "complete.json").write_text("{}")
    (existing_output / "run_config.json").write_text(
        json.dumps(existing.as_dict(), default=str)
    )
    _write_dated_notes(existing_raw, other)
    with_notes = workflow.compile_config(existing_raw, config_dir=other)

    with pytest.raises(ValueError, match="without a dated-notes manifest"):
        workflow.ResearchAllEvidenceWorkflow(with_notes)
    with pytest.raises(ValueError, match="without a dated-notes manifest"):
        workflow._load_analysis_dataset(with_notes, materialize=True)
    assert not (existing_output / "input").exists()


def test_empty_retained_text_is_allowed_with_a_warning(tmp_path, caplog):
    _raw, config = _notes_inputs(tmp_path)
    notes = pd.read_parquet(config.notes.path)
    notes = notes[~((notes["patient_id"] == "c") & (notes["note_date"] < "2024-03-10"))]
    notes.to_parquet(config.notes.path, index=False)

    with caplog.at_level("WARNING", logger=notes_module.__name__):
        dataset = workflow._load_analysis_dataset(config, materialize=True)

    assert dataset.loc[2, "clinical_text"] == ""
    manifest = json.loads(
        (config.output_dir / "input" / "pretreatment_notes_manifest.json").read_text()
    )
    assert manifest["patients_without_retained_notes"] == 1
    assert "1 of 6 cohort patients have no retained pretreatment notes" in caplog.text


# --------------------------------------------------------------------------
# Configuration round trip and command line


def test_notes_config_round_trips_through_run_config(tmp_path):
    raw, config = _notes_inputs(tmp_path)
    raw["notes"].update(
        {
            "columns": {"note_type": "kind"},
            "include_index_date": True,
            "lookback_days": 365,
            "note_header": "none",
        }
    )
    config = workflow.compile_config(raw, config_dir=tmp_path)
    saved = json.loads(json.dumps(config.as_dict(), default=workflow._json_default))

    reloaded = workflow.compile_config(saved, config_dir=tmp_path / "elsewhere")

    assert reloaded.notes == config.notes
    assert reloaded.analysis_dataset == config.analysis_dataset
    assert saved["notes"] == {
        "path": str(tmp_path / "notes.parquet"),
        "columns": {
            "unit_id": "patient_id",
            "date": "note_date",
            "text": "note_text",
            "note_type": "kind",
        },
        "index_date_column": "index_date",
        "include_index_date": True,
        "lookback_days": 365,
        "note_header": "none",
    }


def test_runs_without_notes_keep_their_run_config_and_dataset_path(tmp_path):
    _raw, config = _inputs(tmp_path)

    assert config.notes is None
    assert "notes" not in config.as_dict()
    assert config.analysis_dataset == config.dataset
    assert workflow._load_stage1_template(config)["dataset_path"] == str(config.dataset)


def test_set_override_reaches_the_notes_block(tmp_path):
    raw, _config = _notes_inputs(tmp_path)
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    args = workflow.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--set",
            "notes.include_index_date=true",
            "--set",
            "notes.lookback_days=30",
            "--set",
            "notes.columns.note_type=kind",
        ]
    )

    overridden, config_dir = workflow._raw_config_from_args(args)
    config = workflow.compile_config(overridden, config_dir=config_dir)

    assert config.notes.include_index_date is True
    assert config.notes.lookback_days == 30
    assert config.notes.note_type_column == "kind"
    bad = workflow.build_parser().parse_args(
        ["--config", str(config_path), "--set", "notes.lookback=30"]
    )
    overridden, config_dir = workflow._raw_config_from_args(bad)
    with pytest.raises(ValueError, match="unsupported fields: \\['lookback'\\]"):
        workflow.compile_config(overridden, config_dir=config_dir)


def test_status_does_not_build_the_analysis_cohort(tmp_path, capsys):
    raw, config = _notes_inputs(tmp_path)
    raw["notes"]["path"] = "missing_notes.parquet"
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "progress.json").write_text('{"status": "running"}\n')

    assert workflow.main(["--config", str(config_path), "--status"]) == 0

    assert '"running"' in capsys.readouterr().out
    assert not (config.output_dir / "input").exists()


# --------------------------------------------------------------------------
# Stage 2 consumers


def test_stage2_preflight_validates_a_dated_notes_run_read_only(tmp_path):
    from oci.inference import stage2_preflight as preflight
    from tests.test_stage2_launch_preflight import saved_fixture, tree

    config = saved_fixture(tmp_path, prepare_raw=_write_dated_notes)
    before = tree(tmp_path)

    report = preflight.preflight_stage2(config)

    assert report["status"] == "validated"
    assert report["dataset"] == str(config.output_dir / "input" / "analysis_cohort.parquet")
    assert report["source_dataset"] == str(config.dataset)
    assert tree(tmp_path) == before

    changed = replace(config, notes=replace(config.notes, include_index_date=True))
    with pytest.raises(RuntimeError, match="saved Stage 1 configuration mismatch: notes"):
        preflight.preflight_stage2(changed)
    assert tree(tmp_path) == before


def test_empty_text_patient_passes_stage2_extraction_routing(tmp_path, monkeypatch):
    from oci.inference import plain_handoff_stage2_analysis as analysis
    from tests.stage2_prompt_spy import install, prompt_inputs
    from tests.test_stage2_extraction_preparation import definition

    install(monkeypatch)
    mode = definition()
    scalar = dict(
        definition(1),
        measurement_definition="Latest documented state.",
        conflict_resolution={"strategy": "latest"},
    )
    calls = []

    def request(messages, validate, *, request_kind):
        body = prompt_inputs(messages)
        names = [feature["name"] for feature in body["features"]]
        if body["job"] == "extract_stage2_patient_variables":
            texts = [patient["text"] for patient in body["patients"]]
            calls.append((names, texts))
            if texts == [""]:
                return validate({name.replace("_", " "): None for name in names})
            assert names == ["state_1"]
            return validate({"state 1": "blue"})
        assert body["job"] == "extract_stage2_patient_variable_observations"
        assert body["patient"]["text"]
        calls.append((names, [body["patient"]["text"]]))
        return validate(
            {
                "observations": [
                    {
                        "feature": "state 0",
                        "value": "red",
                        "quote": "State: red.",
                        "governing_date_quote": None,
                    }
                ]
            }
        )

    frame = analysis.extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["", "State: red. State: blue."]}),
        row_ids=[0, 1],
        text_column="clinical_text",
        definitions=[mode, scalar],
        output_dir=tmp_path,
        request_json=request,
        workers=2,
        max_prompt_chars=100_000,
    )

    # Each strategy group sends the empty record through its ordinary prompt.
    assert sorted(calls) == [
        (["state_0"], [""]),
        (["state_0"], ["State: red. State: blue."]),
        (["state_1"], [""]),
        (["state_1"], ["State: red. State: blue."]),
    ]
    assert frame["_oci_row_id"].tolist() == [0, 1]
    assert frame.loc[0, ["state_0", "state_1"]].isna().all()
    assert frame.loc[1, "state_0"] == "red" and frame.loc[1, "state_1"] == "blue"
    summary = json.loads((tmp_path / "failure_summary.json").read_text())
    assert summary["feature_failure_patterns"] == []
    assert summary["structural_failure_patient_count"] == 0
