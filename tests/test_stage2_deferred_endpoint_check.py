"""A resumed Stage 2 start defers its endpoint check while the server is down."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

import oci.inference.plain_handoff_stage2 as workflow
from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    PlainHandoffStage2Config,
    plain_stage2_config_from_mapping,
    run_plain_handoff_stage2,
)
from oci.inference.plain_handoff_stage2_analysis import Stage2RequestExhaustedError


MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
ENDPOINT = "http://127.0.0.1:8000/v1"
MESSAGES = [{"role": "user", "content": "ping"}]


class _Server:
    """A fake OpenAI-compatible server whose /models answer can change."""

    def __init__(self, *, model: str = MODEL, max_model_len: int = 262144) -> None:
        self.up = True
        self.model = model
        self.max_model_len = max_model_len
        self.probes = 0
        self.requests: list[tuple[str, str]] = []

    def served_model_ids(self, config: PlainHandoffStage2Config) -> list[str]:
        self.probes += 1
        if not self.up:
            raise workflow._Stage2EndpointUnreachable(
                f"Stage 2 could not auto-discover a model from {config.endpoint}/models: "
                "APIConnectionError: Connection error."
            )
        return workflow._ServedModelIds([self.model], records=[{
            "id": self.model, "root": self.model, "parent": None, "revision": None,
            "max_model_len": self.max_model_len,
        }])

    def completion(self, _messages, config: PlainHandoffStage2Config) -> str:
        self.requests.append((config.endpoint, config.model))
        return "{}"


@pytest.fixture
def server(monkeypatch) -> _Server:
    fake = _Server()
    monkeypatch.setattr(workflow, "_served_model_ids", fake.served_model_ids)
    monkeypatch.setattr(workflow, "_openai_completion", fake.completion)
    return fake


def _config() -> PlainHandoffStage2Config:
    return plain_stage2_config_from_mapping({
        "endpoint": ENDPOINT,
        "model": MODEL,
        "extraction_llm": {
            "model": MODEL,
            "endpoints": [{"endpoint": ENDPOINT, "max_concurrency": 4}],
        },
    }, default_workers=4)


def _runner(output_dir: Path) -> PlainHandoffStage2:
    return PlainHandoffStage2(
        config=_config(),
        clinical_question="Identify confounders.",
        prior_model_identity=workflow._saved_model_identity(output_dir),
    )


def _verified_run(output_dir: Path) -> PlainHandoffStage2:
    """The interrupted run: started while the server was up."""

    runner = _runner(output_dir)
    runner._check_and_record_model_identity(output_dir)
    return runner


def _identity(output_dir: Path) -> dict:
    return json.loads((output_dir / "model_identity.json").read_text(encoding="utf-8"))


def test_fresh_start_still_fails_when_the_endpoint_is_unreachable(tmp_path, server):
    server.up = False
    with pytest.raises(workflow._Stage2EndpointUnreachable, match="Connection error"):
        _runner(tmp_path)
    assert server.requests == []


def test_resume_defers_the_check_and_configures_the_runner_as_verified(tmp_path, server):
    verified = _verified_run(tmp_path)
    server.up = False

    resumed = _runner(tmp_path)
    resumed._check_and_record_model_identity(tmp_path)

    assert resumed.deferred_endpoint_check is not None
    assert resumed.completion.endpoint_check is resumed.deferred_endpoint_check
    assert resumed.extraction_completion.endpoint_check is resumed.deferred_endpoint_check
    for field in ("runtime_model_family", "runtime_sampling_model", "runtime_context_window_tokens"):
        assert getattr(resumed.config, field) == getattr(verified.config, field)
        assert (getattr(resumed.extraction_request_config, field)
                == getattr(verified.extraction_request_config, field))
    assert resumed.config.runtime_context_window_tokens == 262144
    assert resumed.model_identity["effective_request_policies"] == (
        verified.model_identity["effective_request_policies"]
    )
    recorded = _identity(tmp_path)
    for role in ("primary", "extraction"):
        assert recorded[role]["live_endpoint_verified"] is False
        assert recorded[role]["endpoint_verification"] == "deferred_to_first_request"
        assert recorded[role]["endpoint_observations"] == []
        previous = recorded[role]["previous_live_endpoint_verification"]
        assert previous["live_endpoint_verified"] is True
        assert previous["endpoint_observations"] == (
            verified.model_identity[role]["endpoint_observations"]
        )
    assert recorded["scientific_identity"] == _identity_scientific(verified)
    assert server.requests == []


def _identity_scientific(runner: PlainHandoffStage2) -> dict:
    return workflow._scientific_model_identity(runner.model_identity)


def test_first_request_fails_as_exhausted_while_the_endpoint_is_still_down(tmp_path, server):
    _verified_run(tmp_path)
    server.up = False
    resumed = _runner(tmp_path)
    resumed._check_and_record_model_identity(tmp_path)

    for router, request_config in (
        (resumed.completion, resumed.config),
        (resumed.extraction_completion, resumed.extraction_request_config),
    ):
        with pytest.raises(Stage2RequestExhaustedError, match="unreachable at startup"):
            router(MESSAGES, request_config)
    assert server.requests == []
    assert resumed.deferred_endpoint_check.verified is False
    assert _identity(tmp_path)["primary"]["live_endpoint_verified"] is False


def test_first_request_verifies_and_records_the_endpoint_once_it_is_back(tmp_path, server):
    _verified_run(tmp_path)
    server.up = False
    resumed = _runner(tmp_path)
    resumed._check_and_record_model_identity(tmp_path)

    server.up = True
    probes_before = server.probes
    resumed.extraction_completion(MESSAGES, resumed.extraction_request_config)
    # One probe for each deferred role, then requests go straight through.
    assert server.probes == probes_before + 2
    resumed.completion(MESSAGES, resumed.config)
    resumed.extraction_completion(MESSAGES, resumed.extraction_request_config)
    assert server.probes == probes_before + 2
    assert server.requests == [(ENDPOINT, MODEL)] * 3
    recorded = _identity(tmp_path)
    for role in ("primary", "extraction"):
        assert recorded[role]["live_endpoint_verified"] is True
        assert "endpoint_verification" not in recorded[role]
        assert recorded[role]["endpoint_observations"][0]["selected_model_advertised"] is True


@pytest.mark.parametrize("change", ["context_window", "model"])
def test_a_returning_endpoint_that_would_configure_the_run_differently_stops_it(
    tmp_path, server, change,
):
    _verified_run(tmp_path)
    server.up = False
    resumed = _runner(tmp_path)
    resumed._check_and_record_model_identity(tmp_path)

    server.up = True
    if change == "context_window":
        server.max_model_len = 131072
        expected = "would configure the run differently"
    else:
        server.model = "Qwen/Qwen3.6-27B"
        expected = "not the selected model"
    with pytest.raises(RuntimeError, match=expected) as raised:
        resumed.completion(MESSAGES, resumed.config)
    assert not isinstance(raised.value, Stage2RequestExhaustedError)
    assert server.requests == []
    assert resumed.deferred_endpoint_check.verified is False


def test_a_second_deferred_restart_reuses_the_earlier_verification(tmp_path, server):
    verified = _verified_run(tmp_path)
    server.up = False
    _runner(tmp_path)._check_and_record_model_identity(tmp_path)

    again = _runner(tmp_path)
    again._check_and_record_model_identity(tmp_path)
    assert again.deferred_endpoint_check is not None
    assert again.config.runtime_context_window_tokens == (
        verified.config.runtime_context_window_tokens
    )
    previous = _identity(tmp_path)["primary"]["previous_live_endpoint_verification"]
    assert previous["endpoint_observations"] == (
        verified.model_identity["primary"]["endpoint_observations"]
    )


def test_resume_does_not_defer_for_other_endpoints_models_or_errors(tmp_path, server, monkeypatch):
    _verified_run(tmp_path)
    server.up = False
    prior = workflow._saved_model_identity(tmp_path)

    moved = plain_stage2_config_from_mapping({
        "endpoint": "http://127.0.0.1:8001/v1", "model": MODEL,
    }, default_workers=4)
    with pytest.raises(workflow._Stage2EndpointUnreachable):
        PlainHandoffStage2(config=moved, clinical_question="q", prior_model_identity=prior)

    other_model = plain_stage2_config_from_mapping({
        "endpoint": ENDPOINT, "model": "Qwen/Qwen3.6-27B",
    }, default_workers=4)
    with pytest.raises(workflow._Stage2EndpointUnreachable):
        PlainHandoffStage2(config=other_model, clinical_question="q", prior_model_identity=prior)

    def unauthorized(config):
        raise RuntimeError(f"Stage 2 could not auto-discover a model from {config.endpoint}/models")

    monkeypatch.setattr(workflow, "_served_model_ids", unauthorized)
    with pytest.raises(RuntimeError, match="auto-discover") as raised:
        _runner(tmp_path)
    assert not isinstance(raised.value, workflow._Stage2EndpointUnreachable)


def test_a_recorded_identity_that_does_not_cover_the_config_reports_the_outage(tmp_path, server):
    _verified_run(tmp_path)
    server.up = False
    prior = workflow._saved_model_identity(tmp_path)
    prior["primary"]["endpoint_observations"][0]["advertised_model_ids"] = ["another/model"]
    with pytest.raises(workflow._Stage2EndpointUnreachable, match="Connection error"):
        PlainHandoffStage2(config=_config(), clinical_question="q", prior_model_identity=prior)


def test_run_plain_handoff_stage2_passes_the_saved_identity_to_the_runner(
    tmp_path, server, monkeypatch,
):
    _verified_run(tmp_path)
    server.up = False
    seen = {}

    def run(self, **kwargs):
        self._check_and_record_model_identity(kwargs["output_dir"])
        seen["deferred"] = self.deferred_endpoint_check is not None
        return {"ok": True}

    monkeypatch.setattr(PlainHandoffStage2, "run", run)
    assert run_plain_handoff_stage2(
        handoff_path=tmp_path / "handoff", output_dir=tmp_path,
        clinical_question="q", config=_config(),
    ) == {"ok": True}
    assert seen == {"deferred": True}


def test_served_model_ids_reports_a_closed_port_as_unreachable():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = PlainHandoffStage2Config(
        endpoint=f"http://127.0.0.1:{port}/v1", model=MODEL, request_timeout=5.0,
    )
    with pytest.raises(workflow._Stage2EndpointUnreachable, match="APIConnectionError"):
        workflow._served_model_ids(config)
