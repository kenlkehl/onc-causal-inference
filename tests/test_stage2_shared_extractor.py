from contextlib import contextmanager
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import plain_handoff_stage2_analysis as analysis


MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"
OLD_MODEL = "nvidia/Gemma-4-31B-IT-NVFP4"


@pytest.mark.parametrize("overlapping", [False, True])
def test_shared_model_launches_once_and_routes_both_roles_across_gpu_union(tmp_path, monkeypatch, overlapping):
    cfg = workflow.plain_stage2_config_from_mapping({
        "model": MODEL,
        "vllm": {"gpus": list(range(8)) if overlapping else list(range(4)), "gpus_per_server": 1},
        "extraction_llm": {"model": MODEL, "vllm": {
            "gpus": list(range(8)) if overlapping else list(range(4, 8)),
            "gpus_per_server": 1, "base_port": 8110,
        }},
    }, default_workers=32)
    root = tmp_path / "stage2"
    phase = root / "vllm_servers/model_phase.json"
    phase.parent.mkdir(parents=True)
    phase.write_text(json.dumps({"allocation_mode": "configured_split"}))
    lifecycle, routed = [], []

    @contextmanager
    def launch(**kwargs):
        lifecycle.append("start")
        assert kwargs["model"] == MODEL
        assert kwargs["output_dir"].name == "shared"
        assert set(kwargs["config"].gpus) == {f"cuda:{i}" for i in range(8)}
        assert kwargs["config"].server_count == 8
        try:
            yield tuple(f"http://127.0.0.1:{8110+i}/v1" for i in range(8))
        finally:
            lifecycle.append("stop")

    def complete(messages, config):
        routed.append((config.endpoint, config.model, config.runtime_request_kind))
        return "{}"

    def run(self, **kwargs):
        assert self.completion.completion is self.extraction_completion.completion
        for i in range(16):
            kind = "extraction" if i % 2 == 0 else "interpretation"
            router = self.extraction_completion if kind == "extraction" else self.completion
            request_cfg = self.extraction_request_config if kind == "extraction" else self.config
            router([], replace(request_cfg, runtime_request_kind=kind))
        return {"ok": True}

    monkeypatch.setattr(workflow, "launch_managed_vllm_servers", launch)
    monkeypatch.setattr(workflow, "_served_model_ids", lambda cfg: [cfg.model])
    monkeypatch.setattr(workflow, "_openai_completion", complete)
    monkeypatch.setattr(workflow.PlainHandoffStage2, "run", run)
    assert workflow.run_plain_handoff_stage2(
        handoff_path=tmp_path / "handoff", output_dir=root,
        clinical_question="test", config=cfg, dataset=object(),
    ) == {"ok": True}
    assert lifecycle == ["start", "stop"]
    assert [x[0] for x in routed] == [f"http://127.0.0.1:{8110+i%8}/v1" for i in range(16)]
    assert all(x[1] == MODEL for x in routed)
    assert json.loads(phase.read_text())["allocation_mode"] == "shared_model"


@pytest.mark.parametrize("kind", ["extraction", "interpretation"])
@pytest.mark.parametrize("off_repairs,failed_responses,expected_thinking", [
    (1, 1, [False, False]),
    (1, 2, [False, False, True]),
    (5, 1, [False, False]),
    (5, 5, [False, False, False, False, False, False]),
    (5, 6, [False, False, False, False, False, False, True]),
])
def test_wire_repairs_enable_thinking_after_configured_off_repairs(
    monkeypatch, kind, off_repairs, failed_responses, expected_thinking,
):
    import openai
    requests = []

    class Client:
        def __init__(self, **kwargs):
            self.chat = NS(completions=NS(create=self.create))

        def create(self, **kwargs):
            requests.append(kwargs)
            answer = {"ecog": [0, 1]} if len(requests) <= failed_responses else {"ecog": 1}
            return NS(id="test", usage=None, choices=[NS(
                finish_reason="stop", message=NS(content=json.dumps(answer)),
            )])

        def close(self):
            pass

    def validate(value):
        if not isinstance(value["ecog"], int):
            raise ValueError("ecog requires one scalar value; received list")
        return value

    monkeypatch.setattr(openai, "OpenAI", Client)
    cfg = workflow.PlainHandoffStage2Config(
        endpoint="http://test/v1", model=MODEL,
        thinking_after_response_repairs=off_repairs,
    )
    assert workflow._request_json(
        messages=[{"role": "user", "content": "ECOG 1"}], config=cfg,
        completion=workflow._openai_completion, validate=validate, request_kind=kind,
        initial_reasoning_effort="none" if kind == "interpretation" else None,
        prompt_token_counter=lambda messages: 100, context_window_tokens=262144,
    ) == {"ecog": 1}
    assert [r["extra_body"]["chat_template_kwargs"]["enable_thinking"] for r in requests] == expected_thinking
    for request in requests[1:]:
        assert request["messages"][-2]["role"] == "assistant"
        assert "ValueError: ecog requires one scalar value; received list" in request["messages"][-1]["content"]


def test_category_mapping_starts_without_thinking_and_includes_validation_error(tmp_path):
    definition = {"feature_id": "f", "name": "status", "value_type": "categorical",
                  "categories_or_unit": ["Negative", "Positive"],
                  "measurement_definition": "Extract status.", "missing_value_rule": "Null if missing."}
    calls = []

    def request(messages, validate, *, request_kind="interpretation", initial_reasoning_effort=None):
        calls.append((request_kind, initial_reasoning_effort, messages))
        if request_kind == "extraction":
            return validate({"rows": [{"row_id": 0, "values": {"status": "not detected"}}]})
        return validate({"corrections": [{"mapping_id": "category_mapping_0001", "value": "Negative"}]})

    result = analysis._request_validated_extraction(
        messages=[{"role": "user", "content": "SOURCE NOTE"}], row_ids=[0],
        definitions=[definition], request_json=request,
        ontology_audit_path=tmp_path / "category_ontology_repair.json",
        messages_for_definitions=lambda definitions: [],
    )
    assert result["rows"][0]["values"]["status"] == "Negative"
    assert calls[1][:2] == ("interpretation", "none")
    prompt = json.dumps(calls[1][2])
    assert "Validation error" in prompt and "_ExtractionCategoryError" in prompt
    assert "not detected" in prompt and "SOURCE NOTE" not in prompt


def test_frozen_definition_model_preserves_original_input_fingerprint():
    old = workflow.PlainHandoffStage2Config(endpoint="http://test/v1", model=OLD_MODEL)
    new = replace(old, model=MODEL, frozen_feature_definition_model=OLD_MODEL,
                  runtime_sampling_model=MODEL, runtime_model_family="gemma4")
    kwargs = dict(clinical_question="test", outer_fold=1, discovery_packets=[], seed=42)
    assert workflow._feature_definition_input_value(config=old, **kwargs) == (
        workflow._feature_definition_input_value(config=new, **kwargs)
    )
    assert workflow._feature_definition_input_value(config=replace(new, frozen_feature_definition_model=""), **kwargs) != (
        workflow._feature_definition_input_value(config=old, **kwargs)
    )


def test_frozen_definition_model_requires_completed_definitions(tmp_path):
    runner = workflow.PlainHandoffStage2(
        config=workflow.PlainHandoffStage2Config(
            endpoint="http://test/v1", model=MODEL, frozen_feature_definition_model=OLD_MODEL,
        ), clinical_question="test", completion=lambda *_: pytest.fail("unexpected model call"),
    )
    with pytest.raises(RuntimeError, match="requires completed feature definitions"):
        runner._run_outer_fold(outer_fold=1, packets=[], output_dir=tmp_path)


@pytest.mark.parametrize("value", [None, False, 123, " "])
def test_frozen_definition_model_rejects_invalid_provenance(value):
    with pytest.raises(ValueError, match="frozen_feature_definition_model"):
        workflow.PlainHandoffStage2Config(
            endpoint="http://test/v1", model=MODEL, frozen_feature_definition_model=value,
        ).validate()
