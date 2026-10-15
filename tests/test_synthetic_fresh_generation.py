"""Fresh (LLM-designed DGP) synthetic generation: key redaction, resume, and seeding."""

from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from synthetic_data import cli as synthetic_cli
from synthetic_data import generator as g
from synthetic_data.config import LLMConfig, SyntheticDataConfig

SECRET = "sk-sentinel-synthetic"
TIMELINE = (
    "<clinical_note> 01/01/2025: Evaluation before vinorelbine.\n"
    "<imaging_report> 01/02/2025: CT shows a lung mass."
)
FEATURES = {
    "features": [
        {"name": "age", "type": "continuous", "description": "Age in years",
         "roles": ["confounder"]},
        {
            "name": "ecog",
            "type": "categorical",
            "description": "ECOG performance status",
            "categories": ["0", "1", "2"],
            "roles": ["confounder", "effect_modifier"],
        },
        {"name": "baseline_nlr", "type": "continuous",
         "description": "Neutrophil-to-lymphocyte ratio", "roles": ["effect_modifier"]},
    ]
}
EQUATIONS = {
    "treatment_equation": {
        "intercept": 0.0,
        "coefficients": {"age": 0.5, "ecog_1": 0.3, "ecog_2": 0.6},
    },
    "outcome_equation": {
        "intercept": 0.0,
        "coefficients": {"age": 0.2, "ecog_1": 0.4, "ecog_2": 0.8},
    },
}
SUMMARY = {
    "summary_statistics": {
        "age": {"mean": 65.0, "std": 10.0},
        "ecog": {"proportions": {"0": 0.4, "1": 0.4, "2": 0.2}},
        "baseline_nlr": {"mean": 4.5, "std": 2.5},
    }
}


class FakeClient:
    design_calls = []
    text_calls = []
    note_counter = itertools.count()

    def __init__(self, config):
        self.config = config

    def generate_json(self, prompt, system_prompt=None, temperature=None, **kwargs):
        self.design_calls.append(temperature)
        if prompt.startswith("Given the following comparative effectiveness"):
            return json.loads(json.dumps(FEATURES))
        if prompt.startswith("Given the following role-tagged variables"):
            return json.loads(json.dumps(EQUATIONS))
        if prompt.startswith("Given the following confounding variables"):
            return json.loads(json.dumps(SUMMARY))
        raise AssertionError("unexpected design prompt")

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
        self.text_calls.append(temperature)
        if temperature == 0.7:
            return TIMELINE
        # Earlier requests finish later, so concurrent patients complete out of
        # order; text post-processing must not depend on that order.
        time.sleep(0.01 * (3 - next(self.note_counter) % 4))
        return "Plan: start gemcitabine; CT shows a lung mass."


@pytest.fixture
def config(tmp_path, monkeypatch):
    FakeClient.design_calls = []
    FakeClient.text_calls = []
    monkeypatch.setattr(g, "LLMClient", FakeClient)
    return SyntheticDataConfig(
        clinical_question="Compare vinorelbine with gemcitabine for NSCLC.",
        dataset_size=6,
        min_events_per_patient=5,
        max_events_per_patient=6,
        drug_perturbation_prob=0.5,
        output_dir=str(tmp_path / "fresh"),
        llm=LLMConfig(api_key=SECRET),
    )


def _scaffold_columns(frame):
    return frame[
        ["patient_id", "patient_prompt", "treatment_indicator", "outcome_indicator",
         "true_treatment_prob", "true_age", "true_ecog", "true_baseline_nlr"]
    ]


def test_fresh_generation_never_writes_the_api_key(config, tmp_path, monkeypatch):
    g.generate_synthetic_dataset(config, num_workers=2, show_progress=False)
    output = Path(config.output_dir)
    for path in output.iterdir():
        assert SECRET not in path.read_text(encoding="utf-8", errors="replace"), path
    assert json.loads((output / "metadata.json").read_text())["config"]["llm"]["api_key"] == ""

    # The CLI's generation_config.json is written before generation starts.
    monkeypatch.setattr(
        synthetic_cli,
        "generate_synthetic_dataset",
        lambda **kwargs: (pd.DataFrame({"treatment_indicator": [0], "outcome_indicator": [1]}), {}),
    )
    cli_output = tmp_path / "cli"
    monkeypatch.setattr(
        sys, "argv",
        ["synthetic", "--output-dir", str(cli_output), "--api-key", SECRET],
    )
    synthetic_cli.main()
    saved = (cli_output / "generation_config.json").read_text()
    assert SECRET not in saved
    assert json.loads(saved)["llm"]["api_key"] == ""


def test_resume_reuses_the_saved_dgp_and_its_checkpointed_patients(config):
    first_config = replace(config, dataset_size=3)
    first, first_metadata = g.generate_synthetic_dataset(first_config, 1, False)
    assert len(FakeClient.design_calls) == 3
    design = json.loads((Path(config.output_dir) / g.DGP_DESIGN_FILENAME).read_text())

    # A resumed (here: enlarged) run asks the LLM for no new DGP and keeps the
    # checkpointed patients, which were sampled from the same DGP.
    resumed, metadata = g.generate_synthetic_dataset(config, 3, False)
    assert len(FakeClient.design_calls) == 3
    for key in ("features", "treatment_equation", "outcome_equation", "summary_statistics"):
        assert metadata[key] == design[key] == first_metadata[key]
    pd.testing.assert_frame_equal(resumed.iloc[:3], first)
    assert resumed["patient_id"].tolist() == list(range(6))


def test_checkpoint_without_a_saved_dgp_fails_closed(config):
    output = Path(config.output_dir)
    output.mkdir(parents=True)
    (output / g.CHECKPOINT_FILENAME).write_text(json.dumps({"patients": []}))

    with pytest.raises(ValueError, match="DGP was not saved"):
        g.generate_synthetic_dataset(config, 1, False)
    assert not FakeClient.design_calls


def test_resume_with_different_design_settings_fails_closed(config):
    g.generate_synthetic_dataset(replace(config, dataset_size=2), 1, False)
    changed = replace(config, target_logit_std=1.0)

    with pytest.raises(ValueError, match="target_logit_std"):
        g.generate_synthetic_dataset(changed, 1, False)
    assert len(FakeClient.design_calls) == 3


def test_seeded_cohort_does_not_depend_on_worker_count(config):
    serial, _ = g.generate_synthetic_dataset(config, 1, False)
    parallel_config = replace(config, output_dir=str(Path(config.output_dir).with_name("parallel")))
    parallel, _ = g.generate_synthetic_dataset(parallel_config, 4, False)

    pd.testing.assert_frame_equal(serial, parallel)
    other_seed = replace(
        config, seed=7, output_dir=str(Path(config.output_dir).with_name("seed7"))
    )
    reseeded, _ = g.generate_synthetic_dataset(other_seed, 4, False)
    assert not _scaffold_columns(reseeded).equals(_scaffold_columns(serial))


def test_generation_steps_use_fixed_temperatures_and_the_cli_has_no_temperature_flag(
    config, capsys, monkeypatch
):
    g.generate_synthetic_dataset(replace(config, dataset_size=1), 1, False)
    assert FakeClient.design_calls == [0.7, 0.5, 0.5]
    assert set(FakeClient.text_calls) == {0.7, 0.5}

    monkeypatch.setattr(sys, "argv", ["synthetic", "--help"])
    with pytest.raises(SystemExit):
        synthetic_cli.main()
    assert "--temperature" not in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["synthetic", "--temperature", "0.2"])
    with pytest.raises(SystemExit):
        synthetic_cli.main()
