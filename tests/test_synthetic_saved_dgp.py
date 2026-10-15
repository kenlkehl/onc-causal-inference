import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from synthetic_data.config import SyntheticDataConfig, LLMConfig
from synthetic_data.generator import generate_synthetic_dataset, _compute_logit
from synthetic_data import saved_dgp, llm_client
from synthetic_data import generator as g


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


def test_failed_note_resumes_timeline_without_sampling_again(setup):
    setup.dataset_size = 1
    FakeClient.fail_notes = True
    with pytest.raises(RuntimeError, match="incomplete"):
        generate_synthetic_dataset(setup, 1, False, str(SOURCE))
    output = Path(setup.output_dir)
    original = (output / "patient_scaffold.parquet").read_bytes()
    assert not (output / "dataset.parquet").exists()
    assert sum(x["temperature"] == 0.7 for x in FakeClient.calls) == 1
    FakeClient.fail_notes = False
    df, _ = generate_synthetic_dataset(setup, 2, False, str(SOURCE))
    assert len(df) == 1 and df.num_notes.iloc[0] == 2
    assert sum(x["temperature"] == 0.7 for x in FakeClient.calls) == 1
    assert (output / "patient_scaffold.parquet").read_bytes() == original
    assert json.loads((output / "progress.json").read_text())["status"] == "complete"


def test_changed_model_and_corrupt_checkpoint_are_rejected(setup):
    generate_synthetic_dataset(setup, 1, False, str(SOURCE))
    changed = replace(setup, llm=replace(setup.llm, model_name="different-model"))
    with pytest.raises(ValueError, match="different generation"):
        generate_synthetic_dataset(changed, 1, False, str(SOURCE))
    path = Path(setup.output_dir) / "patients/patient_000000.json"
    payload = json.loads(path.read_text())
    payload["value"]["clinical_text"] = "corrupted"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="incomplete"):
        generate_synthetic_dataset(setup, 1, False, str(SOURCE))


def test_changed_source_settings_rejected_before_requests(setup):
    setup.enforce_positivity = True
    with pytest.raises(ValueError, match="enforce_positivity"):
        generate_synthetic_dataset(setup, 1, False, str(SOURCE))
    assert not FakeClient.calls


def test_note_format_instruction_and_reject_structured_wrappers(setup):
    generate_synthetic_dataset(setup, 1, False, str(SOURCE))
    for call in FakeClient.calls:
        assert (saved_dgp.NOTE_OUTPUT_FORMAT in call["system_prompt"]) == (call["temperature"] == 0.5)
    for text in ('{"content":"note"}', '```json\n{}\n```', '[{"note":"text"}]'):
        with pytest.raises(ValueError, match="structured/code wrapper"):
            saved_dgp._clinical_document(text)
    saved_dgp._clinical_document("RADIOLOGY\nFindings: Left lung mass.")


def test_replica_dispatch_kwargs_and_truncation(monkeypatch):
    calls = []
    finish = ["stop"]

    class FakeOpenAI:
        def __init__(self, base_url, **kwargs):
            self.base_url = base_url
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            calls.append((self.base_url, kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish[0], message=SimpleNamespace(content='{"ok":true}'))])

        def close(self):
            pass

    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    config = LLMConfig(api_base_urls=["http://one/v1", "http://two/v1"],
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}}, reject_incomplete=True)
    client = llm_client.LLMClient(config)
    client.generate("note")
    client.generate_json("json please")
    assert [x[0] for x in calls] == config.api_base_urls
    assert all(x[1]["extra_body"] == config.extra_body for x in calls)
    finish[0] = "length"
    with pytest.raises(ValueError, match="Incomplete generation"):
        client.generate("note")
    with pytest.raises(ValueError):
        client.generate_json("json please", max_retries=1)
    client.close()


def test_expansion_of_a_legacy_saved_dgp_uses_normal_covariate_draws(setup):
    # The bundled metadata config has no continuous_covariate_distribution.
    assert "continuous_covariate_distribution" not in json.loads(SOURCE.read_text())["config"]
    assert setup.continuous_covariate_distribution == "normal"
    lognormal = replace(setup, continuous_covariate_distribution="positive_lognormal")
    with pytest.raises(ValueError, match="continuous_covariate_distribution"):
        generate_synthetic_dataset(lognormal, 1, False, str(SOURCE))
    assert not FakeClient.calls

    df, metadata = generate_synthetic_dataset(setup, 2, False, str(SOURCE))
    normal_config = replace(setup, continuous_covariate_distribution="normal")
    np.random.seed(setup.seed)
    expected = [
        g._build_patient_scaffold_record(
            i, normal_config, metadata["features"], metadata["summary_statistics"],
            metadata["treatment_equation"], metadata["outcome_equation"],
        )
        for i in range(setup.dataset_size)
    ]
    normal = [
        {f"true_{f['name']}": row[f"true_{f['name']}"] for f in metadata["features"]}
        for row in expected
    ]
    observed = df[list(normal[0])].to_dict("records")
    assert observed == normal
