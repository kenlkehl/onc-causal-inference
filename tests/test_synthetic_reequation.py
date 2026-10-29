"""Re-equation of a synthetic cohort: contract API, CLI, roles, U, calibration, refusals (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_data import cli as synthetic_cli
from synthetic_data import generator as g
from synthetic_data import reequation as rq
from synthetic_data.config import SyntheticDataConfig
from synthetic_data.dgp_terms import feature_terms

FEATURES = [
    {"name": "age", "type": "continuous", "description": "Age in years", "roles": ["confounder"]},
    {"name": "ecog", "type": "categorical", "categories": ["0", "1", "2"],
     "description": "ECOG performance status", "roles": ["confounder"]},
    {"name": "creatinine_clearance", "type": "continuous", "description": "Creatinine clearance",
     "roles": ["confounder"]},
    {"name": "histology", "type": "categorical", "categories": ["Squamous", "Adenocarcinoma", "Other"],
     "description": "Histology", "roles": ["effect_modifier"]},
    {"name": "baseline_nlr", "type": "continuous", "description": "Neutrophil-to-lymphocyte ratio",
     "roles": ["effect_modifier"]},
    {"name": "hemoglobin", "type": "continuous", "description": "Hemoglobin",
     "roles": ["confounder", "effect_modifier"]},
    {"name": "prior_platinum", "type": "categorical", "categories": ["None", "First‑line", "Multiple"],
     "description": "Prior platinum", "roles": ["confounder"]},
    {"name": "albumin", "type": "continuous", "description": "Serum albumin",
     "roles": ["effect_modifier"]},
]
SUMMARY = {
    "age": {"type": "continuous", "mean": 66.0, "std": 10.0},
    "ecog": {"type": "categorical", "proportions": {"0": 0.3, "1": 0.5, "2": 0.2}},
    "creatinine_clearance": {"type": "continuous", "mean": 80.0, "std": 20.0},
    "histology": {"type": "categorical",
                  "proportions": {"Squamous": 0.3, "Adenocarcinoma": 0.6, "Other": 0.1}},
    "baseline_nlr": {"type": "continuous", "mean": 4.5, "std": 2.0},
    "hemoglobin": {"type": "continuous", "mean": 13.0, "std": 1.5},
    "prior_platinum": {"type": "categorical",
                       "proportions": {"None": 0.4, "First-line": 0.4, "Multiple": 0.2}},
    "albumin": {"type": "continuous", "mean": 3.8, "std": 0.5},
}
SCENARIO = {
    "name": "instruments_and_prognostic",
    "roles": {"confounder": 3, "effect_modifier": 2, "dual_role": 1, "instrument": 1, "prognostic": 1},
    "role_assignment": "random",
    "modifier_main_effects": True,
    "nonlinearity": "none",
    "treatment_logit_std": 1.5,
    "outcome_logit_std": 2.0,
    "treatment_effect": 0.2,
    "modifier_strength": 1.0,
    "unmeasured_confounders": 1,
    "unmeasured_strength": 1.0,
    "target_treatment_rate": 0.5,
    "target_control_outcome_rate": 0.5,
    "outcome_noise_std": 1.0,
}
NAMED = {
    "name": "named_roles",
    "role_assignment": "named",
    "roles": {
        "confounder": ["age", "ecog"],
        "effect_modifier": ["histology"],
        "dual_role": ["hemoglobin"],
        "instrument": ["creatinine_clearance"],
        "prognostic": ["prior_platinum"],
    },
    "unmeasured_confounders": 2,
}
TEXT_COLUMNS = ("clinical_text", "patient_prompt", "event_timeline")


def write_source(directory: Path, n: int = 300, *, outcome_type: str = "binary", seed: int = 0) -> Path:
    """A small cohort in the generator's output layout (text plus true_<variable> columns)."""

    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for patient in range(n):
        row = {"patient_id": patient}
        values = {}
        for feature in FEATURES:
            stats = SUMMARY[feature["name"]]
            if feature["type"] == "continuous":
                values[feature["name"]] = float(rng.normal(stats["mean"], stats["std"]))
            else:
                probabilities = list(stats["proportions"].values())
                values[feature["name"]] = str(rng.choice(feature["categories"], p=probabilities))
        row["patient_prompt"] = "\n".join(f"- {name}: {value}" for name, value in values.items())
        row["clinical_text"] = (
            f"Patient {patient} ‑ consult note.\n\n<new_note>\n\nCT chest: mass {patient % 7} cm. "
            "Unicode é–‑ kept."
        )
        row["treatment_indicator"] = int(rng.random() < 0.5)
        row["outcome_indicator"] = int(rng.random() < 0.5)
        if outcome_type == "binary":
            row.update(true_treatment_prob=0.5, true_outcome_prob=0.5, true_y0_prob=0.5,
                       true_y1_prob=0.5, true_ite_prob=0.0)
        else:
            row.update(true_treatment_prob=0.5, true_y0=0.0, true_y1=0.0, true_ite=0.0)
        for name, value in values.items():
            row[f"true_{name}"] = value
        row["event_timeline"] = f"<clinical_note> At age 60, visit {patient}."
        row["num_notes"] = 2
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(directory / "dataset.parquet", index=False)
    config = SyntheticDataConfig(outcome_type=outcome_type, continuous_covariate_distribution="normal")
    public = config.to_public_dict()
    metadata = {
        "config": public,
        "features": FEATURES,
        "confounders": [f for f in FEATURES if "confounder" in f["roles"]],
        "effect_modifiers": [f for f in FEATURES if "effect_modifier" in f["roles"]],
        "treatment_equation": {"intercept": 0.0, "coefficients": {"age": 0.1}},
        "outcome_equation": {"intercept": 0.0, "coefficients": {"age": 0.1}, "treatment_coefficient": 0.2},
        "summary_statistics": SUMMARY,
        "dataset_statistics": g.oracle_dataset_statistics(frame),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (directory / "generation_config.json").write_text(json.dumps(public, indent=2))
    return directory


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return write_source(tmp_path_factory.mktemp("source") / "cohort")


def _read(directory: Path):
    return (
        pd.read_parquet(directory / "dataset.parquet"),
        json.loads((directory / "metadata.json").read_text()),
    )


def _equation_inputs(metadata):
    unmeasured = [{"name": u["name"], "type": "continuous"} for u in metadata["unmeasured_confounders"]]
    features = metadata["features"] + unmeasured
    stats = {**metadata["summary_statistics"],
             **{u["name"]: {"mean": 0.0, "std": 1.0} for u in unmeasured}}
    return features, stats


def _recompute(frame, metadata):
    features, stats = _equation_inputs(metadata)
    out = {"treatment": [], "lp0": [], "lp1": []}
    for row in frame.to_dict("records"):
        chars = {feature["name"]: row[f"true_{feature['name']}"] for feature in features}
        out["treatment"].append(g._compute_logit(chars, features, stats, metadata["treatment_equation"]))
        out["lp0"].append(g._compute_logit(chars, features, stats, metadata["outcome_equation"], treatment=0))
        out["lp1"].append(g._compute_logit(chars, features, stats, metadata["outcome_equation"], treatment=1))
    return {key: np.array(value) for key, value in out.items()}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _terms(metadata, role):
    return {
        term
        for feature in metadata["features"]
        if role in feature["roles"]
        for term in feature_terms(feature)
    }


def test_equations_follow_declared_roles(source, tmp_path):
    out = rq.reequation_cohort(source, NAMED, size=None, seed=4, output_dir=tmp_path / "named")
    _, metadata = _read(out)
    by_name = {f["name"]: f["roles"] for f in metadata["features"]}
    assert by_name == {
        "age": ["confounder"], "ecog": ["confounder"], "creatinine_clearance": ["instrument"],
        "histology": ["effect_modifier"], "baseline_nlr": [], "hemoglobin": ["confounder", "effect_modifier"],
        "prior_platinum": ["prognostic"], "albumin": [],
    }
    treatment_eq, outcome_eq = metadata["treatment_equation"], metadata["outcome_equation"]
    treatment_terms = set(treatment_eq["coefficients"]) | {
        term for item in treatment_eq["interactions"] for term in item["terms"]
    }
    outcome_main = set(outcome_eq["coefficients"])
    interactions = {item["term"] for item in outcome_eq["treatment_interactions"]}
    instrument, prognostic = _terms(metadata, "instrument"), _terms(metadata, "prognostic")
    confounder, modifier = _terms(metadata, "confounder"), _terms(metadata, "effect_modifier")
    # An instrument is absent from the outcome equation, a prognostic variable
    # from the treatment equation; confounders are in both.
    assert instrument <= set(treatment_eq["coefficients"])
    assert not instrument & (outcome_main | interactions)
    assert prognostic <= outcome_main
    assert not prognostic & treatment_terms
    assert confounder <= set(treatment_eq["coefficients"]) and confounder <= outcome_main
    # Modifiers (including the dual-role variable) have treatment interactions;
    # without modifier_main_effects a modifier-only variable has no main term.
    assert interactions == modifier
    assert not (_terms(metadata, "effect_modifier") - confounder) & outcome_main
    # Unassigned variables enter no equation.
    for name in ("baseline_nlr", "albumin"):
        assert name not in treatment_terms | outcome_main | interactions
    # Pairwise treatment interactions join confounder terms only.
    assert {term for item in treatment_eq["interactions"] for term in item["terms"]} <= confounder
    # U enters both equations and is recorded in the metadata.
    names = [u["name"] for u in metadata["unmeasured_confounders"]]
    assert names == ["u_1", "u_2"]
    for u in metadata["unmeasured_confounders"]:
        assert u["coefficients"]["treatment"] == treatment_eq["coefficients"][u["name"]]
        assert u["coefficients"]["outcome"] == outcome_eq["coefficients"][u["name"]]
        assert u["oracle_column"] == f"true_{u['name']}"
    # Role lists keep their meaning, and a dual-role variable is in both.
    assert [f["name"] for f in metadata["confounders"]] == ["age", "ecog", "hemoglobin"]
    assert [f["name"] for f in metadata["effect_modifiers"]] == ["histology", "hemoglobin"]
    assert [f["name"] for f in metadata["instruments"]] == ["creatinine_clearance"]
    assert [f["name"] for f in metadata["prognostic_variables"]] == ["prior_platinum"]
    assert metadata["scenario"]["role_assignment"] == "named"
    assert metadata["post_treatment"] is None

    with_main = dict(NAMED, name="main_effects", modifier_main_effects=True)
    _, main_metadata = _read(rq.reequation_cohort(source, with_main, seed=4, output_dir=tmp_path / "main"))
    assert _terms(main_metadata, "effect_modifier") <= set(main_metadata["outcome_equation"]["coefficients"])


@pytest.mark.parametrize(
    ("nonlinearity", "outcome_type"), [("none", "binary"), ("quadratic", "continuous")]
)
def test_true_ite_matches_the_potential_outcome_equations(tmp_path, outcome_type, nonlinearity):
    src = write_source(tmp_path / "src", n=150, outcome_type=outcome_type)
    scenario = dict(SCENARIO, nonlinearity=nonlinearity)
    frame, metadata = _read(rq.reequation_cohort(src, scenario, seed=9, output_dir=tmp_path / "out"))
    lp = _recompute(frame, metadata)
    assert frame["true_treatment_prob"].to_numpy() == pytest.approx(_sigmoid(lp["treatment"]))
    if outcome_type == "binary":
        p0, p1 = _sigmoid(lp["lp0"]), _sigmoid(lp["lp1"])
        for column, expected in (("true_y0_prob", p0), ("true_y1_prob", p1), ("true_ite_prob", p1 - p0),
                                 ("true_y0", p0), ("true_y1", p1), ("true_ite", p1 - p0)):
            assert frame[column].to_numpy() == pytest.approx(expected)
        observed = np.where(frame["treatment_indicator"] == 1, p1, p0)
        assert frame["true_outcome_prob"].to_numpy() == pytest.approx(observed)
        assert set(frame["outcome_indicator"].unique()) <= {0, 1}
    else:
        assert frame["true_y0"].to_numpy() == pytest.approx(lp["lp0"])
        assert frame["true_y1"].to_numpy() == pytest.approx(lp["lp1"])
        assert frame["true_ite"].to_numpy() == pytest.approx(lp["lp1"] - lp["lp0"])
        for column in ("true_outcome_prob", "true_y0_prob", "true_y1_prob", "true_ite_prob"):
            assert column not in frame.columns
        noise = frame["outcome_indicator"] - np.where(frame["treatment_indicator"] == 1, lp["lp1"], lp["lp0"])
        assert 0.7 < noise.std() < 1.3
    outcome_eq = metadata["outcome_equation"]
    if nonlinearity == "none":
        assert "nonlinear_terms" not in outcome_eq
    else:
        continuous = {f["name"] for f in metadata["features"] if f["type"] == "continuous"}
        used = set(outcome_eq["coefficients"]) | {i["term"] for i in outcome_eq["treatment_interactions"]}
        assert outcome_eq["nonlinearity"] == nonlinearity
        assert set(outcome_eq["nonlinear_terms"]) == continuous & used
        assert not any(term.startswith("u_") for term in outcome_eq["nonlinear_terms"])


def test_text_columns_are_byte_identical_to_the_source(source, tmp_path):
    frame, metadata = _read(rq.reequation_cohort(source, SCENARIO, size=90, seed=1, output_dir=tmp_path / "t"))
    original = pd.read_parquet(source / "dataset.parquet").set_index("patient_id")
    selected = original.loc[frame["patient_id"]].reset_index()
    for column in TEXT_COLUMNS:
        assert [text.encode("utf-8") for text in frame[column]] == [
            text.encode("utf-8") for text in selected[column]
        ]
    # Every source column is kept; only treatment, outcome, and the oracle
    # probability and effect columns change, and U is added.
    kept = [c for c in original.reset_index().columns if c not in rq.RECOMPUTED_COLUMNS]
    pd.testing.assert_frame_equal(frame[kept], selected[kept])
    added = set(frame.columns) - set(original.reset_index().columns)
    assert added == {"true_y0", "true_y1", "true_ite", "true_u_1"}


def test_cli_reequation_subcommand(source, tmp_path, capsys):
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(NAMED))
    out = tmp_path / "cli"
    assert synthetic_cli.main([
        "reequation", "--source", str(source), "--scenario", str(scenario_path),
        "--size", "40", "--seed", "7", "--output-dir", str(out),
    ]) == 0
    assert capsys.readouterr().out.strip() == str(out.resolve())
    frame, metadata = _read(out)
    assert len(frame) == 40 and metadata["scenario"]["name"] == "named_roles"
    direct = rq.reequation_cohort(source, NAMED, size=40, seed=7, output_dir=tmp_path / "direct")
    pd.testing.assert_frame_equal(frame, _read(direct)[0])
    with pytest.raises(SystemExit):
        synthetic_cli.main(["reequation", "--source", str(source), "--scenario", str(scenario_path),
                            "--output-dir", str(tmp_path / "noseed")])


def test_refuses_sources_with_post_treatment_notes(source, tmp_path):
    def variant(name):
        copy = tmp_path / name
        copy.mkdir()
        for file in rq.SOURCE_FILES:
            (copy / file).write_bytes((source / file).read_bytes())
        return copy

    with_block = variant("block")
    metadata = json.loads((with_block / "metadata.json").read_text())
    metadata["post_treatment"] = {"rate": 1.0, "events": []}
    (with_block / "metadata.json").write_text(json.dumps(metadata))
    with_notes = variant("notes")
    (with_notes / "notes.parquet").write_bytes(b"")
    with_column = variant("column")
    frame = pd.read_parquet(with_column / "dataset.parquet")
    frame["true_post_treatment_toxicity"] = 0
    frame.to_parquet(with_column / "dataset.parquet", index=False)
    with_text = variant("text")
    frame = pd.read_parquet(with_text / "dataset.parquet")
    frame["clinical_text_with_post_treatment"] = frame["clinical_text"]
    frame.to_parquet(with_text / "dataset.parquet", index=False)
    for directory in (with_block, with_notes, with_column, with_text):
        with pytest.raises(ValueError, match="post-treatment notes"):
            rq.reequation_cohort(directory, SCENARIO, seed=1, output_dir=tmp_path / f"{directory.name}_out")
        assert not (tmp_path / f"{directory.name}_out").exists()
