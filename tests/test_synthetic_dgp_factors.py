"""Generator roles, equation shape, and generation-time factors, with a fake LLM client."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference.pretreatment_notes import build_pretreatment_text, notes_config_from_mapping
from synthetic_data import cli as synthetic_cli
from synthetic_data import generation_factors as gf
from synthetic_data import generator as g
from synthetic_data import reequation as rq
from synthetic_data import saved_dgp
from synthetic_data.config import LLMConfig, SyntheticDataConfig
from synthetic_data.dgp_terms import feature_terms
from synthetic_data.prompts import DOCUMENTATION_TIMELINE_INSTRUCTION

SEPARATOR = "\n\n<new_note>\n\n"
FEATURES = [
    {"name": "age", "type": "continuous", "description": "Age in years", "roles": ["confounder"]},
    {"name": "ecog", "type": "categorical", "categories": ["0", "1", "2"],
     "description": "ECOG performance status", "roles": ["confounder", "effect_modifier"]},
    {"name": "creatinine_clearance", "type": "continuous",
     "description": "Creatinine clearance in mL per minute", "roles": ["confounder"]},
    {"name": "smoking", "type": "categorical", "categories": ["never", "former", "current"],
     "description": "Smoking history", "roles": ["confounder"]},
    {"name": "albumin", "type": "continuous", "description": "Serum albumin in g per dL",
     "roles": ["confounder"]},
    {"name": "baseline_nlr", "type": "continuous", "description": "Neutrophil-to-lymphocyte ratio",
     "roles": ["effect_modifier"]},
]
SUMMARY = {
    "age": {"mean": 65.0, "std": 10.0},
    "ecog": {"proportions": {"0": 0.4, "1": 0.4, "2": 0.2}},
    "creatinine_clearance": {"mean": 80.0, "std": 20.0},
    "smoking": {"proportions": {"never": 0.3, "former": 0.5, "current": 0.2}},
    "albumin": {"mean": 3.8, "std": 0.5},
    "baseline_nlr": {"mean": 4.5, "std": 2.0},
    "karnofsky_score": {"mean": 80.0, "std": 10.0},
}
TIMELINE = (
    "<demographics> The patient is a 64-year-old woman.\n"
    "<clinical_note> At age 64, consultation for lung cancer.\n"
    "<imaging_report> At age 65, CT shows a lung mass.\n"
    "<lab_result> At age 65, CBC: WBC 8.2 k/uL (normal).\n"
    "<systemic> At age 65, completed prior chemotherapy."
)


class FakeClient:
    """Fake design and text LLM; records every prompt."""

    calls: list = []
    features = FEATURES

    def __init__(self, config):
        self.config = config

    def generate_json(self, prompt, system_prompt=None, temperature=None, **kwargs):
        self.calls.append(("json", prompt, temperature))
        if prompt.startswith("Given the following comparative effectiveness"):
            return {"features": json.loads(json.dumps(self.features))}
        if prompt.startswith("Given the following role-tagged variables"):
            return {
                "treatment_equation": {"intercept": 0.0, "coefficients": {"age": 0.5, "ecog_1": 0.3, "ecog_2": 0.6}},
                "outcome_equation": {"intercept": 0.0, "coefficients": {"age": 0.2, "ecog_1": 0.4, "ecog_2": 0.8}},
            }
        if prompt.startswith("Given the following confounding variables"):
            return {"summary_statistics": json.loads(json.dumps(SUMMARY))}
        if prompt.startswith("Name related patient measurements"):
            source = next(f["name"] for f in FEATURES if f"- {f['name']} (" in prompt)
            return {"proxies": [{"source": source, "name": "karnofsky_score", "type": "continuous",
                                 "description": "Karnofsky performance score"}]}
        raise AssertionError(f"unexpected design prompt: {prompt[:60]}")

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
        self.calls.append(("text", prompt, temperature))
        if temperature == 0.7:
            return TIMELINE
        if "progress note from a follow-up visit" in prompt:
            return f"Follow-up progress note {len(self.calls)} on gemcitabine."
        return f"Clinical document {len(self.calls)} mentioning gemcitabine."


@pytest.fixture
def fake(monkeypatch, tmp_path):
    FakeClient.calls = []
    FakeClient.features = FEATURES
    monkeypatch.setattr(g, "LLMClient", FakeClient)

    def make(**changes):
        base = SyntheticDataConfig(
            clinical_question="Compare vinorelbine with gemcitabine for NSCLC.",
            dataset_size=12,
            min_events_per_patient=5,
            max_events_per_patient=6,
            output_dir=str(tmp_path / "cohort"),
            llm=LLMConfig(),
            drug_perturbation_prob=0.0,
            continuous_covariate_distribution="normal",
        )
        return replace(base, **changes)

    return make


def _prompts(kind, start=None):
    return [prompt for k, prompt, _ in FakeClient.calls if k == kind and (start is None or prompt.startswith(start))]


def _ite_matches(frame, metadata):
    features, stats = metadata["features"], metadata["summary_statistics"]
    for row in frame.to_dict("records"):
        chars = {f["name"]: row[f"true_{f['name']}"] for f in features}
        lp = [g._compute_logit(chars, features, stats, metadata["outcome_equation"], t) for t in (0, 1)]
        p = [1.0 / (1.0 + np.exp(-value)) for value in lp]
        assert row["true_ite"] == pytest.approx(p[1] - p[0])
        e = 1.0 / (1.0 + np.exp(-g._compute_logit(chars, features, stats, metadata["treatment_equation"])))
        assert row["true_treatment_prob"] == pytest.approx(e)


def test_generation_assigns_python_roles_and_shapes_the_equations(fake):
    config = fake(
        num_features=4, num_confounders=3, num_effect_modifiers=2, num_instruments=1, num_prognostic=1,
        modifier_main_effects=True, nonlinearity="quadratic", treatment_logit_std=1.0,
        outcome_logit_std=2.5,
    )
    frame, metadata = g.generate_synthetic_dataset(config, 1, False)

    # The LLM is asked for the extra variables as confounders, and for nothing else.
    feature_prompt = _prompts("json", "Given the following comparative effectiveness")[0]
    assert ("exactly 6 total features; exactly 5 features with the confounder role; "
            "exactly 2 features with the effect_modifier role") in feature_prompt
    assert "instrument" not in feature_prompt and "prognostic" not in feature_prompt
    roles = {f["name"]: f["roles"] for f in metadata["features"]}
    instruments = [n for n, r in roles.items() if r == ["instrument"]]
    prognostic = [n for n, r in roles.items() if r == ["prognostic"]]
    assert len(instruments) == len(prognostic) == 1
    assert set(instruments + prognostic) <= {"age", "creatinine_clearance", "smoking", "albumin"}
    assert sum("confounder" in r for r in roles.values()) == 3
    assert [f["name"] for f in metadata["instruments"]] == instruments
    assert [f["name"] for f in metadata["prognostic_variables"]] == prognostic

    equation_prompt = _prompts("json", "Given the following role-tagged variables")[0]
    summary_prompt = _prompts("json", "Given the following confounding variables")[0]
    for name in instruments + prognostic:
        assert f"- {name} (" not in equation_prompt
        assert f"- {name} (" in summary_prompt
    assert "instrument" not in summary_prompt and "prognostic" not in summary_prompt

    by_name = {f["name"]: f for f in metadata["features"]}
    treatment_eq, outcome_eq = metadata["treatment_equation"], metadata["outcome_equation"]
    treatment_terms = set(treatment_eq["coefficients"]) | {
        t for item in treatment_eq["interactions"] for t in item["terms"]
    }
    outcome_terms = set(outcome_eq["coefficients"]) | {i["term"] for i in outcome_eq["treatment_interactions"]}
    instrument_terms = set(feature_terms(by_name[instruments[0]]))
    prognostic_terms = set(feature_terms(by_name[prognostic[0]]))
    assert instrument_terms <= set(treatment_eq["coefficients"]) and not instrument_terms & outcome_terms
    assert prognostic_terms <= set(outcome_eq["coefficients"]) and not prognostic_terms & treatment_terms
    assert "baseline_nlr" in outcome_eq["coefficients"]  # modifier main effect
    assert outcome_eq["nonlinearity"] == "quadratic"
    continuous_used = {
        n for n, f in by_name.items() if f["type"] == "continuous" and (set(feature_terms(f)) & outcome_terms)
    }
    assert set(outcome_eq["nonlinear_terms"]) == continuous_used
    _ite_matches(frame, metadata)

    # Separate logit SDs hold on Monte Carlo draws from the DGP.
    np.random.seed(0)
    draws = [g._sample_patient_characteristics(metadata["features"], metadata["summary_statistics"],
                                               continuous_distribution="normal") for _ in range(6000)]
    stats = metadata["summary_statistics"]
    t_lp = [g._compute_logit(c, metadata["features"], stats, treatment_eq) for c in draws]
    o_sd = max(np.std([g._compute_logit(c, metadata["features"], stats, outcome_eq, t) for c in draws])
               for t in (0, 1))
    assert np.std(t_lp) == pytest.approx(1.0, rel=0.08)
    assert o_sd == pytest.approx(2.5, rel=0.08)
    for key, value in (("unmeasured_confounders", []), ("scenario", None), ("documentation", None),
                       ("post_treatment", None), ("proxies", [])):
        assert metadata[key] == value


def test_python_roles_need_enough_confounder_only_variables(fake):
    FakeClient.features = [dict(f, roles=["confounder", "effect_modifier"]) if f["roles"] == ["confounder"]
                           else f for f in FEATURES]
    with pytest.raises(ValueError, match="confounder-only"):
        g.generate_synthetic_dataset(fake(num_instruments=1), 1, False)


def test_covariate_correlation_is_achieved_in_samples():
    features = [
        {"name": "x1", "type": "continuous"},
        {"name": "x2", "type": "continuous"},
        {"name": "x3", "type": "continuous"},
        {"name": "level", "type": "categorical", "categories": ["low", "mid", "high"]},
        {"name": "marker", "type": "continuous"},
    ]
    stats = {"x1": {"mean": 0.0, "std": 1.0}, "x2": {"mean": 10.0, "std": 3.0}, "x3": {"mean": -2.0, "std": 0.5},
             "level": {"proportions": {"low": 0.2, "mid": 0.5, "high": 0.3}},
             "marker": {"mean": 4.5, "std": 2.5}}
    np.random.seed(1)
    draws = pd.DataFrame([
        g._sample_patient_characteristics(features, stats, continuous_distribution="positive_lognormal",
                                          covariate_correlation=0.4)
        for _ in range(8000)
    ])
    correlation = draws[["x1", "x2", "x3"]].corr().to_numpy()
    assert correlation[np.triu_indices(3, 1)] == pytest.approx([0.4] * 3, abs=0.04)
    # Categorical levels keep their proportions; positive means keep log-normal moments.
    shares = draws["level"].value_counts(normalize=True)
    assert [shares["low"], shares["mid"], shares["high"]] == pytest.approx([0.2, 0.5, 0.3], abs=0.02)
    assert draws["marker"].min() > 0
    assert draws["marker"].mean() == pytest.approx(4.5, rel=0.04)
    assert draws["marker"].std() == pytest.approx(2.5, rel=0.08)
    # The level index follows the shared latent factor.
    index = draws["level"].map({"low": 0, "mid": 1, "high": 2})
    assert np.corrcoef(index, draws["x1"])[0, 1] > 0.2
    # Zero correlation without proxies keeps the independent draw.
    np.random.seed(3)
    independent = g._sample_patient_characteristics(features, stats)
    np.random.seed(3)
    assert g._sample_patient_characteristics(features, stats, covariate_correlation=0.0) == independent


def test_proxy_correlation_holds_on_the_latent_scale():
    features = [
        {"name": "crcl", "type": "continuous", "roles": ["confounder"]},
        {"name": "other", "type": "continuous", "roles": ["confounder"]},
        {"name": "crcl_proxy", "type": "continuous", "roles": [], "proxy_of": "crcl"},
    ]
    stats = {"crcl": {"mean": 80.0, "std": 20.0}, "other": {"mean": 0.0, "std": 1.0},
             "crcl_proxy": {"mean": 1.0, "std": 0.3}}
    for rho in (0.0, 0.3):
        np.random.seed(2)
        draws = pd.DataFrame([
            g._sample_patient_characteristics(features, stats, continuous_distribution="normal",
                                              covariate_correlation=rho, proxy_correlation=0.9)
            for _ in range(6000)
        ])
        assert np.corrcoef(draws["crcl"], draws["crcl_proxy"])[0, 1] == pytest.approx(0.9, abs=0.015)
        assert np.corrcoef(draws["other"], draws["crcl_proxy"])[0, 1] == pytest.approx(0.9 * rho, abs=0.04)
        assert draws["crcl_proxy"].mean() == pytest.approx(1.0, abs=0.02)


def test_proxy_variables_are_designed_by_the_llm_and_enter_no_equation(fake):
    frame, metadata = g.generate_synthetic_dataset(fake(num_proxy_variables=1, proxy_correlation=0.8), 1, False)
    proxy = metadata["features"][-1]
    assert proxy["name"] == "karnofsky_score" and proxy["roles"] == [] and proxy["proxy_of"] in {
        "age", "ecog", "creatinine_clearance", "smoking", "albumin"
    }
    assert metadata["proxies"] == [{"name": "karnofsky_score", "proxy_of": proxy["proxy_of"]}]
    proxy_prompt = _prompts("json", "Name related patient measurements")[0]
    assert f"- {proxy['proxy_of']} (" in proxy_prompt
    assert "- karnofsky_score (" not in _prompts("json", "Given the following role-tagged variables")[0]
    assert "- karnofsky_score (" in _prompts("json", "Given the following confounding variables")[0]
    for equation in (metadata["treatment_equation"], metadata["outcome_equation"]):
        terms = set(equation["coefficients"]) | {i.get("term") for i in equation.get("treatment_interactions", [])}
        assert "karnofsky_score" not in terms
    assert "true_karnofsky_score" in frame.columns
    assert frame["patient_prompt"].str.contains("Karnofsky performance score").all()
    with pytest.raises(ValueError, match="duplicates another variable"):
        gf.validate_proxy_definitions(
            {"proxies": [{"source": "age", "name": "ecog", "type": "continuous"}]}, ["age"], ["age", "ecog"]
        )


def _documentation_scaffolds(mechanism, subset=None, n=3000):
    features = [
        {"name": "lab", "type": "continuous", "description": "Serum lab", "roles": ["confounder"]},
        {"name": "stage", "type": "categorical", "categories": ["I", "II", "III", "IV"],
         "description": "Disease stage", "roles": ["effect_modifier"]},
        {"name": "score", "type": "continuous", "description": "Frailty score", "roles": ["confounder"]},
    ]
    stats = {"lab": {"mean": 5.0, "std": 2.0},
             "stage": {"proportions": {"I": 0.1, "II": 0.2, "III": 0.3, "IV": 0.4}},
             "score": {"mean": 0.0, "std": 1.0}}
    config = SyntheticDataConfig(documentation_rate=0.6, documentation_mechanism=mechanism,
                                 documentation_mnar_slope=1.5, continuous_covariate_distribution="normal")
    np.random.seed(4)
    sampled = [g._sample_patient_characteristics(features, stats, continuous_distribution="normal")
               for _ in range(10000)]
    design = gf.documentation_design(
        rate=0.6, mechanism=mechanism, slope=1.5,
        variables=gf.documentation_variables(subset, features),
        features=features, summary_stats=stats, sampled_characteristics=sampled,
    )
    treatment_eq = {"intercept": 0.0, "coefficients": {"lab": 0.5}}
    outcome_eq = {"intercept": 0.0, "coefficients": {"lab": 0.2}, "treatment_coefficient": 0.2}
    records = pd.DataFrame([
        g._build_patient_scaffold_record(i, config, features, stats, treatment_eq, outcome_eq,
                                         factors={"documentation": design})
        for i in range(n)
    ])
    return records, features, design


@pytest.mark.parametrize("mechanism", ["mcar", "mnar"])
def test_documentation_rates_are_achieved(mechanism):
    records, features, design = _documentation_scaffolds(mechanism)
    for feature in features:
        column = gf.documented_oracle_column(feature["name"])
        assert set(records[column].unique()) <= {0, 1}
        assert records[column].mean() == pytest.approx(0.6, abs=0.03)
    high = records["true_lab"] > 5.0
    shown_high = records.loc[high, "true_lab__documented"].mean()
    shown_low = records.loc[~high, "true_lab__documented"].mean()
    if mechanism == "mnar":
        # Higher standardized values (and later levels) are documented more often.
        assert shown_high - shown_low > 0.2
        late = records["true_stage"].isin(["III", "IV"])
        assert records.loc[late, "true_stage__documented"].mean() > records.loc[~late, "true_stage__documented"].mean()
        assert all(value is not None for value in design["intercepts"].values())
    else:
        assert abs(shown_high - shown_low) < 0.06
    # The note writer's characteristics hold exactly the documented values.
    for row in records.head(200).to_dict("records"):
        for feature in features:
            line = f"- {feature['description']}:"
            assert (line in row["patient_prompt"]) == bool(row[gf.documented_oracle_column(feature["name"])])


def test_documented_variables_subset_limits_the_control():
    records, _, design = _documentation_scaffolds("mcar", subset=["stage"], n=800)
    assert design["variables"] == ["stage"]
    assert records["true_lab__documented"].eq(1).all() and records["true_score__documented"].eq(1).all()
    assert records["true_stage__documented"].mean() == pytest.approx(0.6, abs=0.05)
    with pytest.raises(ValueError, match="unknown variables"):
        gf.documentation_variables(["weight"], [{"name": "lab"}])


def test_generation_prompts_omit_hidden_variables(fake):
    frame, metadata = g.generate_synthetic_dataset(fake(documentation_rate=0.5, dataset_size=16), 1, False)
    block = metadata["documentation"]
    assert block["rate"] == 0.5 and block["mechanism"] == "mcar"
    assert block["variables"] == [f["name"] for f in FEATURES]
    assert block["oracle_columns"]["age"] == "true_age__documented"
    timelines = [prompt for kind, prompt, temperature in FakeClient.calls if kind == "text" and temperature == 0.7]
    assert len(timelines) == 16
    header = "Patient Characteristics (YOU MUST ACCURATELY REPRESENT THESE VALUES):\n"
    hidden_checked = 0
    for row, prompt in zip(frame.to_dict("records"), timelines):
        characteristics = prompt.split(header, 1)[1].split(DOCUMENTATION_TIMELINE_INSTRUCTION, 1)[0]
        assert characteristics == row["patient_prompt"]
        assert DOCUMENTATION_TIMELINE_INSTRUCTION in prompt
        assert "if the patient's age is listed above" in prompt
        for feature in FEATURES:
            line = f"- {feature['description']}:"
            if row[f"true_{feature['name']}__documented"]:
                assert line in prompt
            else:
                hidden_checked += 1
                assert line not in prompt
                if feature["type"] == "continuous":
                    assert f": {row['true_' + feature['name']]:.2f}" not in prompt
    assert hidden_checked > 10
    expansions = [p for kind, p, t in FakeClient.calls if kind == "text" and t == 0.5]
    assert expansions and all("that the list of events does not give" in p for p in expansions)
    assert all("you should imagine what they might be" not in p for p in expansions)


def test_post_treatment_notes_table_index_date_and_text(fake, tmp_path):
    config = fake(post_treatment_notes={"rate": 0.75, "events": ["toxicity", {"name": "early_response",
                                                                              "treatment_log_odds": 1.5}]},
                  dataset_size=20, drug_perturbation_prob=0.5)
    frame, metadata = g.generate_synthetic_dataset(config, 1, False)
    output = Path(config.output_dir)
    notes = pd.read_parquet(output / "notes.parquet")
    assert list(notes.columns) == ["patient_id", "note_date", "note_text", "note_type"]
    assert notes["patient_id"].dtype == "int64"
    assert set(notes["patient_id"]) <= set(frame["patient_id"])

    post_total = 0
    for row in frame.to_dict("records"):
        index = date.fromisoformat(row["index_date"])
        mine = notes[notes["patient_id"] == row["patient_id"]]
        dates = [date.fromisoformat(value) for value in mine["note_date"]]
        assert dates == sorted(dates)
        pre = mine[[d < index for d in dates]]
        post = mine[[d > index for d in dates]]
        assert len(pre) + len(post) == len(mine)
        assert SEPARATOR.join(pre["note_text"]) == row["clinical_text"]
        assert len(pre) == row["num_notes"]
        assert list(pre["note_type"]) == ["clinical_note", "imaging_report"]
        for value in post["note_date"]:
            assert 7 <= (date.fromisoformat(value) - index).days <= 90
        assert set(post["note_type"]) <= {"clinical_note"} and len(post) <= 2
        assert row["clinical_text_with_post_treatment"] == SEPARATOR.join([row["clinical_text"], *post["note_text"]])
        assert all("Follow-up progress note" not in text for text in pre["note_text"])
        post_total += len(post)
    assert 0 < post_total
    for column in ("true_post_treatment_toxicity", "true_post_treatment_early_response"):
        assert set(frame[column].unique()) <= {0, 1}

    block = metadata["post_treatment"]
    assert [event["name"] for event in block["events"]] == ["toxicity", "early_response"]
    assert block["events"][1]["treatment_log_odds"] == 1.5
    assert all(event["covariate"] for event in block["events"])
    assert block["post_treatment_note_count"] == post_total
    assert block["notes_file"] == "notes.parquet" and block["index_date_column"] == "index_date"

    # Post-index prompts carry the arm, the event statuses, and the regimen rule.
    post_prompts = [
        p for kind, p, _ in FakeClient.calls
        if kind == "text" and "progress note from a follow-up visit" in p
    ]
    assert len(post_prompts) == post_total
    assert any("Name the regimen" in p for p in post_prompts)
    assert any("do not name the drugs" in p for p in post_prompts)
    assert all(("first treatment named" in p) or ("second treatment named" in p) for p in post_prompts)

    # OCI's dated-notes input keeps exactly the pretreatment documents.
    notes_config = notes_config_from_mapping(
        {"path": str(output / "notes.parquet"), "index_date_column": "index_date", "note_header": "none"},
        config_dir=output, default_unit_id_column="patient_id",
    )
    derived = build_pretreatment_text(
        frame.drop(columns="clinical_text"), notes, config=notes_config,
        unit_id_column="patient_id", text_column="clinical_text",
    )
    assert derived.note_counts["after_index_date"] == post_total
    assert derived.note_counts["retained"] == len(notes) - post_total
    for text, row in zip(derived.texts, frame.to_dict("records")):
        assert text == "\n\n".join(part.strip() for part in row["clinical_text"].split(SEPARATOR))

    # Re-equation refuses the cohort, whose post-index notes depend on its treatment.
    config.to_json(str(output / "generation_config.json"))
    with pytest.raises(ValueError, match="post-treatment notes"):
        rq.reequation_cohort(output, {"name": "S0"}, seed=1, output_dir=tmp_path / "reeq")


def test_post_treatment_events_depend_on_treatment_and_one_covariate():
    features = [{"name": "lab", "type": "continuous", "roles": ["confounder"]}]
    stats = {"lab": {"mean": 0.0, "std": 1.0}}
    block = gf.validate_post_treatment_notes({"events": [{"name": "toxicity", "treatment_log_odds": 2.0,
                                                          "covariate_log_odds": 1.0}]})
    events = gf.resolve_post_treatment_events(block, features, seed=0)
    assert events[0]["covariate"] == "lab"
    rng = np.random.default_rng(0)
    rates = {}
    for treatment in (0, 1):
        for value in (-1.0, 1.0):
            draws = [gf.draw_post_treatment_events(events, {"lab": value}, treatment, features, stats, rng)
                     ["true_post_treatment_toxicity"] for _ in range(4000)]
            rates[treatment, value] = np.mean(draws)
            expected = 1 / (1 + np.exp(-(np.log(0.15 / 0.85) + 2.0 * treatment + value)))
            assert rates[treatment, value] == pytest.approx(expected, abs=0.025)
    assert rates[1, 1.0] > rates[0, 1.0] and rates[0, 1.0] > rates[0, -1.0]


def test_event_dates_and_post_index_note_plan():
    events = [
        {"event_type": "demographics", "event_text": "A 60-year-old man."},
        {"event_type": "clinical_note", "event_text": "At age 60, consult."},
        {"event_type": "imaging_report", "event_text": "At age 62, CT."},
        {"event_type": "clinical_note", "event_text": "Follow-up visit."},
    ]
    dates, index = gf.assign_event_dates(events, random.Random("x"))
    assert all(later > earlier for earlier, later in zip(dates, dates[1:]))
    assert 3 <= (dates[1] - dates[0]).days <= 45
    assert 2 * 365 - 90 <= (dates[2] - dates[1]).days <= 2 * 365 + 1
    assert index == dates[-1] + timedelta(days=1)
    assert gf.assign_event_dates(events, random.Random("x")) == (dates, index)
    block = gf.validate_post_treatment_notes({"rate": 1.0})
    plans = [gf.post_treatment_note_plan(block, random.Random(i)) for i in range(300)]
    assert all(1 <= len(plan) <= 2 for plan in plans)
    assert {len(plan) for plan in plans} == {1, 2}
    for plan in plans:
        days = [note["days_after_index"] for note in plan]
        assert days == sorted(set(days)) and all(7 <= day <= 90 for day in days)
    rare = gf.validate_post_treatment_notes({"rate": 0.2})
    share = np.mean([bool(gf.post_treatment_note_plan(rare, random.Random(i))) for i in range(2000)])
    assert share == pytest.approx(0.2, abs=0.03)


def test_resumed_generation_keeps_factor_draws(fake):
    config = fake(documentation_rate=0.5, documentation_mechanism="mnar",
                  post_treatment_notes={"rate": 1.0}, covariate_correlation=0.3, dataset_size=4)
    first, _ = g.generate_synthetic_dataset(config, 1, False)
    design_calls = len(_prompts("json"))
    first_notes = pd.read_parquet(Path(config.output_dir) / "notes.parquet")
    resumed, _ = g.generate_synthetic_dataset(replace(config, dataset_size=6), 2, False)
    assert len(_prompts("json")) == design_calls
    pd.testing.assert_frame_equal(resumed.iloc[:4].reset_index(drop=True), first)
    notes = pd.read_parquet(Path(config.output_dir) / "notes.parquet")
    pd.testing.assert_frame_equal(notes[notes["patient_id"] < 4].reset_index(drop=True), first_notes)
    with pytest.raises(ValueError, match="covariate_correlation"):
        g.generate_synthetic_dataset(replace(config, covariate_correlation=0.5), 1, False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"covariate_correlation": 1.0}, "covariate_correlation"),
        ({"documentation_rate": 1.5}, "documentation_rate"),
        ({"documentation_mechanism": "mar"}, "documentation_mechanism"),
        ({"nonlinearity": "cubic"}, "nonlinearity"),
        ({"proxy_correlation": 1.0}, "proxy_correlation"),
        ({"num_instruments": -1}, "num_instruments"),
        ({"treatment_logit_std": 0.0}, "treatment_logit_std"),
        ({"documented_variables_subset": []}, "documented_variables_subset"),
        ({"post_treatment_notes": {"rate": 0.0}}, "rate"),
        ({"post_treatment_notes": {"rate": 1.0, "window": 3}}, "unknown post_treatment_notes keys"),
        ({"post_treatment_notes": {"events": ["neuropathy"]}}, "not built in"),
        ({"post_treatment_notes": {"events": ["toxicity", "toxicity"]}}, "distinct"),
        ({"post_treatment_notes": {"events": ["Bad Name"]}}, "lowercase"),
    ],
)
def test_factor_settings_are_validated(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(SyntheticDataConfig(), **changes).validate()


def _run_cli(monkeypatch, tmp_path, *args):
    captured = {}

    def fake_generate(config, num_workers, show_progress, dgp_metadata_path):
        captured["config"] = config
        return pd.DataFrame({"treatment_indicator": [0, 1], "outcome_indicator": [0, 1]}), {}

    monkeypatch.setattr(synthetic_cli, "generate_synthetic_dataset", fake_generate)
    monkeypatch.setattr(sys, "argv", ["synthetic", "--output-dir", str(tmp_path / "out"), *args])
    synthetic_cli.main()
    return captured["config"]


def test_cli_factor_flags_set_the_config(monkeypatch, tmp_path):
    default = _run_cli(monkeypatch, tmp_path)
    for name, value in g.dgp_factor_defaults().items():
        assert getattr(default, name) == value
    config = _run_cli(
        monkeypatch, tmp_path,
        "--num-instruments", "2", "--num-prognostic", "1", "--modifier-main-effects",
        "--nonlinearity", "threshold", "--treatment-logit-std", "1.5", "--outcome-logit-std", "2.5",
        "--modifier-strength", "3", "--covariate-correlation", "0.3", "--documentation-rate", "0.7",
        "--documentation-mechanism", "mnar", "--documentation-mnar-slope", "0.5",
        "--documented-variables", "age, ecog", "--post-treatment-rate", "0.8",
        "--post-treatment-events", "toxicity", "--num-proxy-variables", "1", "--proxy-correlation", "0.85",
    )
    assert (config.num_instruments, config.num_prognostic, config.modifier_main_effects) == (2, 1, True)
    assert (config.nonlinearity, config.treatment_logit_std, config.outcome_logit_std) == ("threshold", 1.5, 2.5)
    assert (config.modifier_strength, config.covariate_correlation, config.documentation_rate) == (3.0, 0.3, 0.7)
    assert (config.documentation_mechanism, config.documentation_mnar_slope) == ("mnar", 0.5)
    assert config.documented_variables_subset == ["age", "ecog"]
    assert config.post_treatment_notes == {"rate": 0.8, "events": ["toxicity"]}
    assert (config.num_proxy_variables, config.proxy_correlation) == (1, 0.85)

    saved = SyntheticDataConfig(covariate_correlation=0.2, post_treatment_notes={"rate": 0.5})
    path = tmp_path / "saved.json"
    saved.to_json(str(path))
    kept = _run_cli(monkeypatch, tmp_path, "--config", str(path), "--post-treatment-events", "early_response")
    assert kept.covariate_correlation == 0.2
    assert kept.post_treatment_notes == {"rate": 0.5, "events": ["early_response"]}


BUNDLED_METADATA = (
    Path(__file__).resolve().parents[1]
    / "synthetic_data/example_synthetic_datasets"
    / "five_confounders_five_effect_modifiers_nsclc_with_structured/metadata.json"
)


def test_saved_dgp_expansion_fails_closed_on_unsupported_factors(tmp_path, monkeypatch):
    def refuse_client(config):
        raise AssertionError("no requests expected")

    monkeypatch.setattr(saved_dgp, "LLMClient", refuse_client)
    base = SyntheticDataConfig.from_dict(json.loads(BUNDLED_METADATA.read_text())["config"])
    base.dataset_size = 2
    base.output_dir = str(tmp_path / "expanded")
    for changes, message in (
        ({"documentation_rate": 0.5}, "documentation control"),
        ({"post_treatment_notes": {"rate": 1.0}}, "post-treatment notes"),
        ({"covariate_correlation": 0.2}, "covariate_correlation"),
        ({"num_instruments": 1}, "num_instruments"),
    ):
        with pytest.raises(ValueError, match=message):
            g.generate_synthetic_dataset(replace(base, **changes), 1, False, str(BUNDLED_METADATA))
    reequated = json.loads(BUNDLED_METADATA.read_text())
    reequated["scenario"] = {"name": "S0"}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(reequated))
    with pytest.raises(ValueError, match="re-equation output"):
        g.generate_synthetic_dataset(base, 1, False, str(path))
