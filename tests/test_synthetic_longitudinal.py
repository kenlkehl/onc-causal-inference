"""End-to-end checks of the multimodal simulator and its observational boundary."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.special import expit

from synthetic_data import cli, longitudinal as sim, reequation
from synthetic_data.generator import _compute_logit
from synthetic_data.longitudinal_history import DIFFERENTIAL, LEGACY_STREAMS, STREAMS
from synthetic_data.pro_ctcae import PRO_ATTRIBUTES


@pytest.fixture
def cohort(tmp_path):
    path = tmp_path / "cohort"
    assert cli.main(["longitudinal", "--dataset-size", "24", "--output-dir", str(path)]) == 0
    return path


def read(path):
    return pq.read_table(path / "dataset.parquet").to_pylist()


def test_observed_streams_and_oracle(cohort):
    rows = read(cohort)
    meta = json.loads((cohort / "metadata.json").read_text())
    seen_panels = set()
    saw_between_visits = saw_missing_score = saw_missed_week = False
    assert meta["config"]["version"] == 2
    assert meta["longitudinal"]["event_counts"]["pros"] == sum(len(r["pros"]) for r in rows)
    assert [r["patient_id"] for r in rows] == list(range(24))
    for row in rows:
        encounters = row["encounters"]
        visit_days = [event["day_from_index"] for event in encounters]
        for stream in STREAMS:
            events = row[stream]
            days = [e["day_from_index"] for e in events]
            assert days and days == sorted(days) and max(days) < 0
            if stream != "pros":
                assert len(events) >= 2 and set(days) <= set(visit_days)
            assert events[0]["delta_years"] is None
            np.testing.assert_allclose(
                [e["delta_years"] for e in events[1:]], np.diff(days) / 365.25
            )
        assert row["clinical_text"] == sim.NOTE_SEPARATOR.join(e["text"] for e in row["notes"])
        assert all(
            v["systolic_mmhg"] > v["diastolic_mmhg"] and 0 < v["spo2_pct"] <= 100
            for v in row["vitals"]
        )
        assert all(e["icd10cm"] and e["cpt"] for e in encounters)
        fatigue = []
        days = [form["day_from_index"] for form in row["pros"]]
        saw_between_visits |= bool(set(days) - set(visit_days))
        assert days[0] - 6 >= visit_days[0] and days[-1] == visit_days[-1]
        assert np.all(np.diff(days) % 7 == 0) and len(days) <= 12
        saw_missed_week |= bool(np.any(np.diff(days) > 7))
        for form in row["pros"]:
            assert form["instrument"] == "PRO-CTCAE" and form["recall_days"] == 7
            items = {(i["symptom"], i["attribute"]): i["score"] for i in form["responses"]}
            assert len(items) == len(form["responses"]) == 11
            for symptom, attributes in PRO_ATTRIBUTES.items():
                scores = [items[symptom, a] for a in attributes]
                assert all(s is None or s in range(5) for s in scores)
                saw_missing_score |= None in scores
                for earlier, later in zip(scores, scores[1:]):
                    if earlier in (None, 0):
                        assert later == earlier
            if items["fatigue", "interference"] is not None:
                fatigue.append(items["fatigue", "interference"])
        creatinines = []
        for panel in row["labs"]:
            seen_panels.add(panel["panel"])
            values = {r["component"]: r["value"] for r in panel["results"]}
            assert all(r["unit"] and np.isfinite(r["value"]) for r in panel["results"])
            if panel["panel"] == "CBC_DIFF":
                assert sum(values[cell + "_abs"] for cell in DIFFERENTIAL) == pytest.approx(
                    values["wbc"], abs=0.003
                )
                assert sum(values[cell + "_pct"] for cell in DIFFERENTIAL) == pytest.approx(
                    100, abs=0.03
                )
                last_nlr = values["neutrophils_abs"] / values["lymphocytes_abs"]
            else:
                expected = {
                    "sodium",
                    "potassium",
                    "chloride",
                    "co2",
                    "calcium",
                    "glucose",
                    "bun",
                    "creatinine",
                }
                if panel["panel"] == "CMP":
                    expected.update(("albumin", "total_protein", "bilirubin", "alp", "ast", "alt"))
                assert set(values) == expected
                creatinines.append((panel["day_from_index"], values["creatinine"]))
        (d0, c0), (d1, c1) = creatinines[0], creatinines[-1]
        v0, v1 = row["vitals"][0], row["vitals"][-1]
        years = (v1["day_from_index"] - v0["day_from_index"]) / 365.25
        observed = dict(
            mean_fatigue_interference=np.mean(fatigue),
            latest_ecog=float(row["notes"][-1]["text"].split()[-1].strip(".")),
            recent_ed_visits=sum(
                e["day_from_index"] >= -180 and "99284" in e["cpt"] for e in encounters
            ),
            log_nlr=np.log(last_nlr),
            creatinine_slope=(c1 - c0) / ((d1 - d0) / 365.25),
            weight_loss_per_year=(1 - v1["weight_kg"] / v0["weight_kg"]) * 100 / years,
        )
        for name, value in observed.items():
            assert row["true_" + name] == pytest.approx(value)
        for arm in (0, 1):
            lp = _compute_logit(
                observed,
                meta["features"],
                meta["summary_statistics"],
                meta["outcome_equation"],
                arm,
            )
            assert row[f"true_y{arm}"] == pytest.approx(expit(lp))
        assert row["true_ite"] == pytest.approx(row["true_y1"] - row["true_y0"])
        assert row["true_outcome_prob"] == row[f"true_y{row['treatment_indicator']}"]
        assert row["true_treatment_prob"] == pytest.approx(
            expit(
                _compute_logit(
                    observed,
                    meta["features"],
                    meta["summary_statistics"],
                    meta["treatment_equation"],
                )
            )
        )
    assert seen_panels == {"CBC_DIFF", "BMP", "CMP"}
    assert saw_between_visits and saw_missing_score and saw_missed_week


def test_previous_schema_is_replayable(cohort, tmp_path):
    path = sim.generate_longitudinal_cohort(tmp_path / "legacy", schema_version=1, dataset_size=2)
    meta = json.loads((path / "metadata.json").read_text())
    assert meta["longitudinal"]["streams"] == list(LEGACY_STREAMS)
    assert "mean_fatigue_interference" not in meta["summary_statistics"]
    for old, new in zip(read(path), read(cohort)):
        assert "pros" not in old
        assert all(old[key] == new[key] for key in LEGACY_STREAMS)


def test_patient_and_equation_seeds_are_independent(cohort, tmp_path):
    original = read(cohort)
    replay = read(sim.generate_longitudinal_cohort(tmp_path / "short", dataset_size=4))
    assert original[:4] == replay
    alternative = read(
        sim.generate_longitudinal_cohort(tmp_path / "other", dataset_size=4, dgp_seed=77)
    )
    for a, b in zip(replay, alternative):
        assert all(a[key] == b[key] for key in STREAMS)
        assert a["true_ite"] != b["true_ite"]
    others = read(sim.generate_longitudinal_cohort(tmp_path / "new", dataset_size=1, seed=11))
    assert others[0]["labs"] != original[0]["labs"]
    with pytest.raises(FileExistsError):
        sim.generate_longitudinal_cohort(cohort)


def test_llm_renderer_uses_pretreatment_facts_and_resumes(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="undocumented patient demographic"):
        sim._validate_note("A 41-year-old male with lung cancer.")
    prompts = []

    class StubClient:
        def __init__(self, config):
            self.config = config

        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            if "offline" in self.config.api_base_url and len(prompts) >= 2:
                raise RuntimeError("server unavailable")
            return "Clinical note. " + prompt.split("\n\n")[-1]

        def close(self):
            pass

    monkeypatch.setattr(sim, "LLMClient", StubClient)
    monkeypatch.setattr("synthetic_data.saved_dgp.time.sleep", lambda seconds: None)
    llm = sim.LLMConfig(api_base_url="offline")
    path = tmp_path / "rendered"
    with pytest.raises(RuntimeError, match="server unavailable"):
        sim.generate_longitudinal_cohort(path, dataset_size=1, llm_config=llm)
    assert not path.exists()
    first_prompt = prompts[0]
    llm.api_base_url = "restored"
    sim.generate_longitudinal_cohort(path, dataset_size=1, llm_config=llm)
    assert prompts.count(first_prompt) == 1
    assert all("true_" not in prompt and "outcome_indicator" not in prompt for prompt in prompts)
    rendered = read(path)[0]
    offline = read(sim.generate_longitudinal_cohort(tmp_path / "offline", dataset_size=1))[0]
    for key in offline.keys() - {"notes", "clinical_text"}:
        assert rendered[key] == offline[key]
    for old, new in zip(offline["notes"], rendered["notes"]):
        assert new == {**old, "text": "Clinical note. " + old["text"]}


def test_reequation_preserves_streams_and_updates_counts(cohort, tmp_path, monkeypatch):
    monkeypatch.setattr(reequation, "MONTE_CARLO_DRAWS", 200)
    path = reequation.reequation_cohort(
        cohort, {"name": "alternative"}, size=12, seed=8, output_dir=tmp_path / "reassigned"
    )
    original = {r["patient_id"]: r for r in read(cohort)}
    rows = read(path)
    counts = json.loads((path / "metadata.json").read_text())["longitudinal"]["event_counts"]
    for name in STREAMS:
        assert counts[name] == sum(len(row[name]) for row in rows)
        assert all(row[name] == original[row["patient_id"]][name] for row in rows)
