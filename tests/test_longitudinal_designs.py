import copy
import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.special import expit

from synthetic_data import cli
from synthetic_data.generator import _compute_logit
from synthetic_data.longitudinal import draw_longitudinal_dgp, sample_longitudinal_patient
from synthetic_data.study_designs import DESIGN_DIR, build_bank, load_bank, validate_bank


def candidates():
    return {
        "measurements": [
            {"name": "lab.platelets.change", "roles": ["unused", "confounder", "dual_role"]},
            {"name": "vital.pulse_bpm.last", "roles": ["instrument"]},
            {"name": "pro.fatigue.interference.mean", "roles": ["prognostic"]},
        ],
        "note_findings": [
            dict(
                name="peritoneal_implants",
                prevalence=0.4,
                roles=["prognostic", "effect_modifier"],
                present="Imaging shows peritoneal implants.",
                absent="No peritoneal implants on imaging.",
            )
        ],
    }


def test_llm_design_cache_and_validation(tmp_path):
    questions = json.loads((DESIGN_DIR / "questions.json").read_text())[1:3]
    path = tmp_path / "bank.json"

    class Client:
        config = SimpleNamespace(model_name="test-model")
        calls = 0

        def generate_json(self, prompt, temperature):
            self.calls += 1
            if self.calls == 1:
                return {"bad": 1}
            if self.calls == 3:
                raise RuntimeError("connection lost")
            body = candidates()
            if "For this variant" in prompt:
                menu = prompt.split("Available measurement names: ")[1].splitlines()[0].split(", ")
                body["measurements"] = [
                    dict(name=next(k for k in menu if k.endswith(".change")), roles=["confounder"]),
                    dict(
                        name=next(k for k in menu if k.startswith("pro.")),
                        roles=["effect_modifier"],
                    ),
                ]
            return body

    client = Client()
    with pytest.raises(RuntimeError, match="connection lost"):
        build_bank(questions, 1, 42, client, path)
    assert not path.exists()
    bank = build_bank(questions, 1, 42, client, path)
    assert client.calls == 4 and len(bank["designs"]) == 2
    assert load_bank(path) == build_bank(questions, 1, 42, client, path)
    assert client.calls == 4
    expanded = build_bank(questions, 2, 42, client, path, workers=2)
    assert expanded["designs"][::2] == bank["designs"] and client.calls == 6
    with pytest.raises(FileExistsError):
        build_bank(questions, 1, 42, client, path)
    bad = copy.deepcopy(bank)
    bad["designs"][1]["question"] = {**questions[0], "id": "other", "split": "held_out"}
    with pytest.raises(ValueError, match="different splits"):
        validate_bank(bad)
    item = bank["designs"][0]["candidates"]["note_findings"][0]
    item["name"] = "lab_platelets_change"
    with pytest.raises(ValueError, match="duplicate"):
        validate_bank(bank)
    item["name"] = "outcome_prob"
    with pytest.raises(ValueError, match="candidate"):
        validate_bank(bank)
    item["name"] = "baseline_implants"
    item["present"] = "The patient was assigned to the control arm."
    with pytest.raises(ValueError, match="assignment"):
        validate_bank(bank)


def test_studies_share_one_equation_and_replay_from_the_cli(tmp_path):
    question = json.loads((DESIGN_DIR / "questions.json").read_text())[1]
    design = dict(id="colon_variant", question=question, candidates=candidates())
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(dict(version=1, designs=[design])))
    dgps = [draw_longitudinal_dgp(i, "design", design=design) for i in range(12)]
    assert len({str(d.features[0]["roles"]) for d in dgps}) == 3
    assert len({str(d.outcome) for d in dgps}) == 12
    output = tmp_path / "cohort"
    cli.main(
        [
            "longitudinal",
            "--design-bank",
            str(bank),
            "--design-id",
            design["id"],
            "--seed",
            "31",
            "--dgp-seed",
            "0",
            "--dataset-size",
            "12",
            "--output-dir",
            str(output),
        ]
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["design"] == design
    assert metadata["outcome_equation"] == dgps[0].outcome
    for i, row in enumerate(pq.read_table(output / "dataset.parquet").to_pylist()):
        assert row == sample_longitudinal_patient(i, seed=31, dgp=dgps[0])
        sibling = sample_longitudinal_patient(i, seed=31, dgp=dgps[1])
        for stream in ("notes", "labs", "encounters", "vitals", "pros"):
            assert row[stream] == sibling[stream]
        assert row["true_y1"] != sibling["true_y1"]
        assert (
            question["population"] in row["clinical_text"]
            and "lung cancer" not in row["clinical_text"]
        )
        assert all(e["icd10cm"][0] == question["icd10cm"] for e in row["encounters"])
        platelets = [
            v["value"] for p in row["labs"] for v in p["results"] if v["component"] == "platelets"
        ]
        interference = [
            v["score"]
            for f in row["pros"]
            for v in f["responses"]
            if (v["symptom"], v["attribute"]) == ("fatigue", "interference")
            and v["score"] is not None
        ]
        fact = design["candidates"]["note_findings"][0]
        present = fact["present"] in row["clinical_text"]
        assert (fact["absent"] in row["clinical_text"]) != present
        x = dict(
            lab_platelets_change=platelets[-1] - platelets[0],
            vital_pulse_bpm_last=row["vitals"][-1]["pulse_bpm"],
            pro_fatigue_interference_mean=np.mean(interference),
            peritoneal_implants="present" if present else "absent",
        )
        assert all(row["true_" + name] == value for name, value in x.items())
        for target, equation, t in [
            ("true_treatment_prob", "treatment_equation", None),
            ("true_y0", "outcome_equation", 0),
            ("true_y1", "outcome_equation", 1),
        ]:
            assert row[target] == pytest.approx(
                expit(
                    _compute_logit(
                        x,
                        metadata["features"],
                        metadata["summary_statistics"],
                        metadata[equation],
                        t,
                    )
                )
            )


def test_wide_design_calibration_deduplication_and_llm_rendering_order(tmp_path):
    from synthetic_data.longitudinal import generate_longitudinal_cohort
    from synthetic_data.study_designs import DEFAULT_BANK

    bank = load_bank(DEFAULT_BANK)
    # bladder_q02_v001 has two findings with the exact same 'present' sentence in the frozen bank.
    colliding = next(d for d in bank["designs"] if d["id"] == "bladder_q02_v001")
    dgp_col = draw_longitudinal_dgp(17, "design", design=colliding)
    names = [f["name"] for f in dgp_col.features]
    assert not ("hypertension_history" in names and "history_of_hypertension" in names)

    # Wide design (100 candidates): active causal features and tx interactions are bounded,
    # and propensities/risks stay calibrated instead of saturating to 0/1.
    wide = next(
        d
        for d in bank["designs"]
        if len(d["candidates"]["measurements"]) + len(d["candidates"]["note_findings"]) == 100
    )
    dgp_wide = draw_longitudinal_dgp(42, "design", design=wide)
    active = [f for f in dgp_wide.features if f["roles"]]
    assert 1 <= len(active) <= 16
    assert len(dgp_wide.treatment["interactions"]) <= 15
    rows = [sample_longitudinal_patient(i, seed=7, dgp=dgp_wide) for i in range(60)]
    props = np.array([r["true_treatment_prob"] for r in rows])
    y0 = np.array([r["true_y0"] for r in rows])
    assert 0.15 < float(props.mean()) < 0.85
    assert 0.05 < float(y0.mean()) < 0.95
    assert float(((props < 0.01) | (props > 0.99)).mean()) < 0.25

    # LLM note rendering runs AFTER document_findings so causal chart findings enter the prompt,
    # and sex-specific cancers (e.g. prostate) do not fail _validate_note on pronouns.
    prostate = next(d for d in bank["designs"] if d["question"]["cancer"] == "prostate cancer")
    prompts = []

    class StubClient:
        config = SimpleNamespace(model_name="stub", max_tokens=512)

        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            return "He presents for oncology evaluation. " + prompt.split("\n\n")[-1]

        def close(self):
            pass

    dgp_p = draw_longitudinal_dgp(3, "design", design=prostate)
    row = sample_longitudinal_patient(
        0, seed=5, dgp=dgp_p, note_client=StubClient(), cache_dir=tmp_path / "cache"
    )
    finding = prostate["candidates"]["note_findings"][0]
    expected_stmt = finding[row["true_" + finding["name"]]]
    assert any(expected_stmt in p for p in prompts)
