from dataclasses import replace

import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import stage2_endpoint_pool as routing


def config(**overrides):
    raw = {
        "endpoint": "http://review.test/v1", "model": "review",
        "extraction_reasoning_effort": "none", "transport_retry_backoff": 0,
        "extraction_llm": {
            "model": "extract", "api_key": "test-secret",
            "endpoints": [
                {"endpoint": "http://a.test/v1/", "max_concurrency": 2},
                {"endpoint": "http://b.test/v1", "max_concurrency": 1},
            ],
        },
    }
    raw.update(overrides)
    return workflow.plain_stage2_config_from_mapping(raw, default_workers=4)


def runner(monkeypatch, *, extraction_workers=None):
    monkeypatch.setattr(workflow, "_served_model_ids", lambda cfg: [cfg.model])
    cfg = config()
    if extraction_workers is not None:
        cfg = replace(cfg, extraction_llm=replace(cfg.extraction_llm, workers=extraction_workers))
    result = workflow.PlainHandoffStage2(config=cfg, clinical_question="test", extraction_tokenizer=lambda _: {})
    result.extraction_completion.completion.pool._metrics_reader = None
    return result


@pytest.mark.parametrize("extraction", [
    {"endpoints": [{"endpoint": "http://a.test/v1", "workers": 2}]},
])
def test_external_pool_rejects_invalid_caps_duplicates_and_conflicting_routes(extraction):
    with pytest.raises(ValueError):
        config(extraction_llm=extraction)


def test_transport_retry_uses_healthy_server_and_releases_failed_capacity(monkeypatch):
    result = runner(monkeypatch)
    called = []

    def completion(_messages, cfg):
        called.append(cfg.endpoint)
        if cfg.endpoint == "http://a.test/v1":
            raise workflow._RetryableStage2ResponseError("connection interrupted")
        return "{}"

    monkeypatch.setattr(workflow, "_openai_completion", completion)
    assert workflow._request_json(messages=[], config=result.extraction_request_config,
        completion=result.extraction_completion, validate=lambda value: value, request_kind="extraction") == {}
    assert called == ["http://a.test/v1", "http://b.test/v1"]
    states = result.extraction_completion.completion.pool.snapshot()
    assert states[0]["cooldown_seconds"] > 0
    assert all(state["in_flight"] == 0 for state in states)


def pool(*, clock=None, **kwargs):
    arguments = {"metrics_reader": None, **kwargs}
    if clock is not None:
        arguments["clock"] = clock
    return routing.EndpointPool([routing.ExtractionEndpoint("http://a.test/v1", 2),
                                 routing.ExtractionEndpoint("http://b.test/v1", 2)], **arguments)
