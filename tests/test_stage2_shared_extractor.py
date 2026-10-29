from contextlib import contextmanager
from dataclasses import replace
import json

import pytest

from oci.inference import plain_handoff_stage2 as workflow


MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"


@pytest.mark.parametrize("overlapping", [True])
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
