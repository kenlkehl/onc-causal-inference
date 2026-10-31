"""Generator roles, equation shape, and generation-time factors, with a fake LLM client."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference.pretreatment_notes import build_pretreatment_text, notes_config_from_mapping
from synthetic_data import generator as g
from synthetic_data import reequation as rq
from synthetic_data.config import LLMConfig, SyntheticDataConfig
from synthetic_data.dgp_terms import feature_terms

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
