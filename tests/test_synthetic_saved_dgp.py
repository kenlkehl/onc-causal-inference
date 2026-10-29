import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_data.config import SyntheticDataConfig
from synthetic_data.generator import generate_synthetic_dataset, _compute_logit
from synthetic_data import saved_dgp


SOURCE = Path(__file__).resolve().parents[1] / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/metadata.json"
TIMELINE = "<clinical_note> 01/01/2025: Evaluation for vinorelbine.\n<imaging_report> 01/02/2025: CT shows a lung mass."


class FakeClient:
    calls = []
    fail_notes = False

    def __init__(self, config):
        assert config.reject_incomplete

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if "masked_event_timeline" in kwargs["prompt"]:
            raise AssertionError("Unformatted prompt")
        if kwargs["temperature"] == 0.7:
            return TIMELINE
        if self.fail_notes:
            raise RuntimeError("temporary note-generation failure")
        return "Patient evaluated for vinorelbine; CT shows a lung mass."

    def close(self):
        pass


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = SyntheticDataConfig.from_dict(json.loads(SOURCE.read_text())["config"])
    config.dataset_size = 4
    config.output_dir = str(tmp_path / "data")
    config.drug_perturbation_prob = 0.3
    FakeClient.calls = []
    FakeClient.fail_notes = False
    monkeypatch.setattr(saved_dgp, "LLMClient", FakeClient)
    monkeypatch.setattr(saved_dgp.time, "sleep", lambda _: None)
    return config


def test_frozen_equations_and_parallel_sampling_resume(setup):
    config = setup
    df, metadata = generate_synthetic_dataset(config, 4, False, str(SOURCE))
    source = json.loads(SOURCE.read_text())
    for key in ("features", "treatment_equation", "outcome_equation", "summary_statistics"):
        assert metadata[key] == source[key]
    assert len(FakeClient.calls) == 12  # timeline + two notes per patient
    for row in df.to_dict("records"):
        features = {f["name"]: row["true_" + f["name"]] for f in metadata["features"]}
        p = []
        for treatment in (0, 1):
            logit = _compute_logit(features, metadata["features"], metadata["summary_statistics"], metadata["outcome_equation"], treatment)
            p.append(1 / (1 + np.exp(-logit)))
        assert row["true_ite_prob"] == pytest.approx(p[1] - p[0])
    resumed, _ = generate_synthetic_dataset(config, 1, False, str(SOURCE))
    pd.testing.assert_frame_equal(df, resumed)
    assert len(FakeClient.calls) == 12
    independent = replace(config, output_dir=str(Path(config.output_dir).with_name("serial")))
    serial, _ = generate_synthetic_dataset(independent, 1, False, str(SOURCE))
    pd.testing.assert_frame_equal(df, serial)
