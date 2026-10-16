"""Re-equation of a synthetic cohort: contract API, CLI, roles, U, calibration, refusals (no LLM)."""

from __future__ import annotations

import hashlib
import json
import math
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


def test_unmeasured_strength_is_relative_to_measured_confounders(source, tmp_path):
    for strength in (0.5, 2.0):
        scenario = dict(NAMED, name=f"u{strength}", unmeasured_confounders=1, unmeasured_strength=strength)
        _, metadata = _read(rq.reequation_cohort(source, scenario, seed=2, output_dir=tmp_path / str(strength)))
        confounder_terms = _terms(metadata, "confounder")
        for key in ("treatment", "outcome"):
            coefficients = metadata[f"{key}_equation"]["coefficients"]
            rms = math.sqrt(np.mean([coefficients[t] ** 2 for t in confounder_terms]))
            u = metadata["unmeasured_confounders"][0]["coefficients"][key]
            assert abs(u) == pytest.approx(strength * rms)


@pytest.mark.parametrize("outcome_type", ["binary", "continuous"])
@pytest.mark.parametrize("nonlinearity", ["none", "threshold", "quadratic"])
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


def test_threshold_and_quadratic_shape_the_continuous_terms():
    features = [{"name": "x", "type": "continuous", "roles": ["effect_modifier"]}]
    stats = {"x": {"mean": 0.0, "std": 1.0}}
    for shape, expected in (("threshold", lambda z: float(z > 0)), ("quadratic", lambda z: z * z)):
        equation = {"intercept": 0.0, "coefficients": {"x": 2.0}, "treatment_coefficient": 0.0,
                    "treatment_interactions": [{"term": "x", "coefficient": 3.0}],
                    "nonlinearity": shape, "nonlinear_terms": ["x"]}
        for z in (-1.5, 0.7, 2.0):
            assert g._compute_logit({"x": z}, features, stats, equation, treatment=1) == pytest.approx(
                5.0 * expected(z)
            )


def test_subsampling_and_redrawing_are_deterministic(source, tmp_path):
    a = rq.reequation_cohort(source, SCENARIO, size=120, seed=5, output_dir=tmp_path / "a")
    b = rq.reequation_cohort(source, SCENARIO, size=120, seed=5, output_dir=tmp_path / "b")
    frame_a, meta_a = _read(a)
    frame_b, meta_b = _read(b)
    pd.testing.assert_frame_equal(frame_a, frame_b)
    assert meta_a["treatment_equation"] == meta_b["treatment_equation"]
    assert meta_a["scenario"] == meta_b["scenario"]

    other, _ = _read(rq.reequation_cohort(source, SCENARIO, size=120, seed=6, output_dir=tmp_path / "c"))
    assert other["patient_id"].tolist() != frame_a["patient_id"].tolist()

    # Without replacement, in source order; a larger size with the same seed
    # is a superset with the same DGP and the same per-patient draws.
    ids = frame_a["patient_id"].tolist()
    assert len(set(ids)) == 120 and ids == sorted(ids)
    larger, meta_large = _read(rq.reequation_cohort(source, SCENARIO, size=200, seed=5, output_dir=tmp_path / "d"))
    assert set(ids) <= set(larger["patient_id"])
    assert meta_large["outcome_equation"] == meta_a["outcome_equation"]
    joined = larger.set_index("patient_id").loc[ids].reset_index()
    pd.testing.assert_frame_equal(joined[frame_a.columns], frame_a)

    everyone, _ = _read(rq.reequation_cohort(source, SCENARIO, size=None, seed=5, output_dir=tmp_path / "e"))
    assert everyone["patient_id"].tolist() == list(range(300))


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


def test_unmeasured_confounders_are_standard_normal_and_never_in_text(tmp_path):
    src = write_source(tmp_path / "src", n=2000)
    frame, metadata = _read(rq.reequation_cohort(
        src, dict(SCENARIO, unmeasured_confounders=2), seed=3, output_dir=tmp_path / "u"
    ))
    for u in metadata["unmeasured_confounders"]:
        values = frame[u["oracle_column"]]
        assert abs(values.mean()) < 0.1 and values.std() == pytest.approx(1.0, abs=0.08)
        assert u["name"].startswith("u_")
        for column in TEXT_COLUMNS:
            assert not frame[column].str.contains(u["name"], regex=False).any()
            assert not frame[column].str.contains(f"{values.iloc[0]:.6f}", regex=False).any()
    assert "u_1" not in {f["name"] for f in metadata["features"]}


def test_calibration_hits_the_target_rates_and_logit_sds(tmp_path):
    src = write_source(tmp_path / "src", n=4000, seed=11)
    scenario = dict(SCENARIO, target_treatment_rate=0.3, target_control_outcome_rate=0.65,
                    treatment_logit_std=1.2, outcome_logit_std=2.5)
    frame, metadata = _read(rq.reequation_cohort(src, scenario, seed=8, output_dir=tmp_path / "cal"))
    assert frame["true_treatment_prob"].mean() == pytest.approx(0.3, abs=0.02)
    assert frame["true_y0_prob"].mean() == pytest.approx(0.65, abs=0.02)
    assert frame["treatment_indicator"].mean() == pytest.approx(0.3, abs=0.04)
    lp = _recompute(frame, metadata)
    assert np.std(lp["treatment"]) == pytest.approx(1.2, rel=0.06)
    assert max(np.std(lp["lp0"]), np.std(lp["lp1"])) == pytest.approx(2.5, rel=0.06)
    monte_carlo = metadata["reequation"]["monte_carlo"]
    assert monte_carlo["draws"] == 10000
    assert monte_carlo["treatment_rate"] == pytest.approx(0.3, abs=1e-6)


def test_omitted_keys_take_the_generator_defaults_and_the_source_roles(source, tmp_path):
    _, metadata = _read(rq.reequation_cohort(source, {"name": "S0"}, seed=0, output_dir=tmp_path / "s0"))
    resolved = metadata["scenario"]
    defaults = SyntheticDataConfig()
    assert resolved["role_assignment"] == "source"
    assert [f["roles"] for f in metadata["features"]] == [f["roles"] for f in FEATURES]
    assert resolved["roles"]["dual_role"] == ["hemoglobin"]
    assert resolved["treatment_effect"] == defaults.treatment_effect_prob
    assert resolved["treatment_logit_std"] == resolved["outcome_logit_std"] == defaults.target_logit_std
    assert resolved["target_treatment_rate"] == defaults.target_treatment_rate
    assert resolved["modifier_main_effects"] is False and resolved["nonlinearity"] == "none"
    assert resolved["unmeasured_confounders"] == 0
    assert metadata["unmeasured_confounders"] == []
    assert metadata["outcome_equation"]["treatment_coefficient"] == defaults.treatment_effect_prob

    # Random assignment draws each group from the unassigned variables.
    random_roles = rq.resolve_scenario(SCENARIO, FEATURES, seed=3)["roles"]
    assigned = [name for names in random_roles.values() for name in names]
    assert len(assigned) == len(set(assigned)) == 8
    assert {group: len(names) for group, names in random_roles.items()} == SCENARIO["roles"]
    assert rq.resolve_scenario(SCENARIO, FEATURES, seed=3)["roles"] == random_roles


def test_manifest_records_the_inputs_and_an_identical_rerun_is_reused(source, tmp_path):
    out = rq.reequation_cohort(source, SCENARIO, size=50, seed=2, output_dir=tmp_path / "m")
    manifest = json.loads((out / rq.MANIFEST_FILENAME).read_text())
    assert manifest["source"]["path"] == str(source.resolve())
    for name in rq.SOURCE_FILES:
        assert manifest["source"]["sha256"][name] == hashlib.sha256((source / name).read_bytes()).hexdigest()
    assert manifest["scenario"] == SCENARIO
    assert (manifest["size"], manifest["seed"]) == (50, 2)
    assert manifest["code_version"] == rq.REEQUATION_CODE_VERSION
    assert set(manifest["code_sha256"]) == {"reequation.py", "generator.py", "dgp_terms.py"}
    assert (out / "generation_config.json").read_bytes() == (source / "generation_config.json").read_bytes()
    assert sorted(p.name for p in out.iterdir()) == sorted(rq.OUTPUT_FILES)

    before = (out / "dataset.parquet").stat().st_mtime_ns
    assert rq.reequation_cohort(source, SCENARIO, size=50, seed=2, output_dir=out) == out
    assert (out / "dataset.parquet").stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="different re-equation"):
        rq.reequation_cohort(source, SCENARIO, size=50, seed=3, output_dir=out)


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


def test_refuses_sources_without_every_oracle_covariate(source, tmp_path):
    copy = tmp_path / "missing"
    copy.mkdir()
    for file in rq.SOURCE_FILES:
        (copy / file).write_bytes((source / file).read_bytes())
    frame = pd.read_parquet(copy / "dataset.parquet").drop(columns="true_albumin")
    frame.to_parquet(copy / "dataset.parquet", index=False)
    with pytest.raises(ValueError, match="true_albumin"):
        rq.reequation_cohort(copy, SCENARIO, seed=1, output_dir=tmp_path / "out")
    frame = pd.read_parquet(source / "dataset.parquet")
    frame.loc[3, "true_histology"] = "Unlisted"
    frame.to_parquet(copy / "dataset.parquet", index=False)
    with pytest.raises(ValueError, match="not one of the categories"):
        rq.reequation_cohort(copy, SCENARIO, seed=1, output_dir=tmp_path / "out")
    (copy / "generation_config.json").unlink()
    with pytest.raises(ValueError, match="generation_config.json"):
        rq.reequation_cohort(copy, SCENARIO, seed=1, output_dir=tmp_path / "out")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"name": ""}, "'name' is required"),
        ({"temperature": 0.2}, "unknown scenario keys"),
        ({"roles": {"mediator": 1}}, "unknown scenario role groups"),
        ({"roles": {"confounder": 9}}, "only 8 unassigned"),
        ({"role_assignment": "named", "roles": {"confounder": 2}}, "needs a list of names"),
        ({"roles": {"confounder": ["age"], "instrument": ["age"]}}, "role groups are disjoint"),
        ({"roles": {"confounder": ["weight"]}}, "unknown variable"),
        ({"nonlinearity": "spline"}, "nonlinearity"),
        ({"target_treatment_rate": 1.0}, "target_treatment_rate"),
        ({"treatment_logit_std": 0}, "treatment_logit_std"),
        ({"unmeasured_confounders": -1}, "unmeasured_confounders"),
        ({"modifier_main_effects": "yes"}, "modifier_main_effects"),
    ],
)
def test_invalid_scenarios_fail_closed(source, tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        rq.reequation_cohort(source, {**SCENARIO, **change}, seed=1, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_invalid_sizes_seeds_and_outputs_fail_closed(source, tmp_path):
    with pytest.raises(ValueError, match="exceeds the source's 300 patients"):
        rq.reequation_cohort(source, SCENARIO, size=301, seed=1, output_dir=tmp_path / "big")
    for size in (0, -3, True):
        with pytest.raises(ValueError, match="size"):
            rq.reequation_cohort(source, SCENARIO, size=size, seed=1, output_dir=tmp_path / "s")
    with pytest.raises(ValueError, match="seed"):
        rq.reequation_cohort(source, SCENARIO, seed=-1, output_dir=tmp_path / "s")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "other.txt").write_text("x")
    with pytest.raises(ValueError, match="holds other files"):
        rq.reequation_cohort(source, SCENARIO, seed=1, output_dir=busy)
    # A re-equation output is not a re-equation source.
    out = rq.reequation_cohort(source, SCENARIO, seed=1, output_dir=tmp_path / "first")
    with pytest.raises(ValueError, match="itself a re-equation output"):
        rq.reequation_cohort(out, SCENARIO, seed=1, output_dir=tmp_path / "second")


def test_proxies_take_no_role(source, tmp_path):
    copy = tmp_path / "proxy"
    copy.mkdir()
    for file in rq.SOURCE_FILES:
        (copy / file).write_bytes((source / file).read_bytes())
    metadata = json.loads((copy / "metadata.json").read_text())
    metadata["features"].append({"name": "karnofsky", "type": "continuous", "description": "KPS",
                                 "roles": [], "proxy_of": "ecog"})
    metadata["summary_statistics"]["karnofsky"] = {"mean": 80.0, "std": 10.0}
    (copy / "metadata.json").write_text(json.dumps(metadata))
    frame = pd.read_parquet(copy / "dataset.parquet")
    frame["true_karnofsky"] = 80.0
    frame.to_parquet(copy / "dataset.parquet", index=False)
    with pytest.raises(ValueError, match="cannot take a role"):
        rq.reequation_cohort(copy, {"name": "p", "roles": {"confounder": ["karnofsky"]}}, seed=1,
                             output_dir=tmp_path / "out")
    all_random = {"name": "all", "roles": {"confounder": 8}}
    _, result = _read(rq.reequation_cohort(copy, all_random, seed=1, output_dir=tmp_path / "ok"))
    proxy = next(f for f in result["features"] if f["name"] == "karnofsky")
    assert proxy["roles"] == [] and proxy["proxy_of"] == "ecog"
    assert result["proxies"] == [{"name": "karnofsky", "proxy_of": "ecog"}]
    assert "karnofsky" not in result["treatment_equation"]["coefficients"]


BUNDLED = (
    Path(__file__).resolve().parents[1]
    / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured"
)


def test_bundled_five_confounder_cohort_is_a_re_equation_source(tmp_path):
    out = rq.reequation_cohort(BUNDLED, SCENARIO, size=None, seed=0, output_dir=tmp_path / "bundled")
    frame, metadata = _read(out)
    source = pd.read_parquet(BUNDLED / "dataset.parquet")
    assert len(frame) == 1000
    assert frame["clinical_text"].tolist() == source["clinical_text"].tolist()
    # The cohort predates matching U+2011 category labels: "First‑line" is
    # never observed, so its dummy is inert, and calibration uses the observed
    # covariates.
    assert "First‑line" not in set(source["true_prior_platinum_therapy"])
    assert frame["true_treatment_prob"].mean() == pytest.approx(0.5, abs=0.02)
    assert frame["true_y0_prob"].mean() == pytest.approx(0.5, abs=0.02)
    counts = {group: len(names) for group, names in metadata["scenario"]["roles"].items()}
    assert counts == SCENARIO["roles"]
