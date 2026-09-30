from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading
import time

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


def test_external_pool_config_resolves_model_verifies_all_servers_and_roundtrips(monkeypatch):
    checked = []

    def models(cfg):
        checked.append(cfg.endpoint)
        return ["review" if cfg.endpoint == "http://review.test/v1" else "extract"]

    monkeypatch.setattr(workflow, "_served_model_ids", models)
    cfg = config(extraction_llm={"endpoints": [
        {"endpoint": "http://a.test/v1/", "max_concurrency": 2},
        {"endpoint": "http://b.test/v1", "max_concurrency": 1},
    ]})
    result = workflow.PlainHandoffStage2(config=cfg, clinical_question="test", extraction_tokenizer=lambda _: {})
    assert checked == ["http://a.test/v1", "http://review.test/v1", "http://a.test/v1", "http://b.test/v1"]
    assert result.config.extraction_llm.model == "extract"
    assert result.config.extraction_llm.workers == 3
    assert result.model_identity["extraction"]["live_endpoint_verified"]
    public = result.config.public_dict()
    assert public["extraction_llm"]["api_key"] == "<redacted>"
    parsed = workflow.plain_stage2_config_from_mapping(json.loads(json.dumps(public)), default_workers=4)
    assert parsed.extraction_llm.endpoints == result.config.extraction_llm.endpoints


@pytest.mark.parametrize("extraction", [
    {"endpoints": []},
    {"endpoints": [{"endpoint": "ftp://a.test", "max_concurrency": 1}]},
    {"endpoints": [{"endpoint": "http://a.test/v1", "max_concurrency": True}]},
    {"endpoints": [{"endpoint": "http://a.test/v1", "max_concurrency": 0}]},
    {"endpoints": [{"endpoint": "http://a.test/v1", "workers": 2}]},
    {"endpoints": ["http://a.test/v1", "http://a.test/v1/"]},
    {"endpoint": "http://c.test/v1", "endpoints": ["http://a.test/v1"]},
    {"model": "extract", "vllm": {"gpus": ["cuda:0"]}, "endpoints": ["http://a.test/v1"]},
])
def test_external_pool_rejects_invalid_caps_duplicates_and_conflicting_routes(extraction):
    with pytest.raises(ValueError):
        config(extraction_llm=extraction)


def test_external_pool_rejects_mismatched_model_before_any_extraction(monkeypatch):
    monkeypatch.setattr(workflow, "_served_model_ids", lambda cfg:
                        ["different"] if cfg.endpoint == "http://b.test/v1" else [cfg.model])
    with pytest.raises(RuntimeError, match="actual served model does not match"):
        workflow.PlainHandoffStage2(config=config(), clinical_question="test")


def test_global_and_per_server_caps_hold_under_concurrent_logical_requests(monkeypatch):
    result = runner(monkeypatch, extraction_workers=10)
    lock, saturated, release = threading.Lock(), threading.Event(), threading.Event()
    active, peaks = {}, {}

    def completion(_messages, cfg):
        with lock:
            active[cfg.endpoint] = active.get(cfg.endpoint, 0) + 1
            peaks[cfg.endpoint] = max(peaks.get(cfg.endpoint, 0), active[cfg.endpoint])
            if sum(active.values()) == 3:
                saturated.set()
        try:
            assert release.wait(5)
            return "{}"
        finally:
            with lock:
                active[cfg.endpoint] -= 1

    monkeypatch.setattr(workflow, "_openai_completion", completion)

    def request():
        return workflow._request_json(messages=[], config=result.extraction_request_config,
            completion=result.extraction_completion, validate=lambda value: value, request_kind="extraction")

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(request) for _ in range(12)]
        try:
            assert saturated.wait(5)
            assert peaks == {"http://a.test/v1": 2, "http://b.test/v1": 1}
        finally:
            release.set()
        assert all(future.result() == {} for future in futures)
    assert max(peaks.values()) == 2
    assert all(server["in_flight"] == 0 for server in result.extraction_completion.completion.pool.snapshot())


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


def test_shared_load_and_cache_pressure_redirect_work_and_stale_metrics_expire():
    now = [100.0]
    servers = pool(clock=lambda: now[0])
    servers._states[0].metrics = {"running": 20, "waiting": 10, "cache_fraction": .95}
    servers._states[0].metrics_at = now[0]
    first = servers.reserve()
    assert first == 1
    servers.release(first, duration_seconds=1)
    now[0] += 16
    second = servers.reserve()
    assert second == 0  # Expired observations cannot leave a server permanently disfavored.
    servers.release(second, duration_seconds=1)


def test_load_score_does_not_double_count_our_requests():
    servers = pool()
    a, b = servers.reserve(), servers.reserve()
    assert (a, b) == (0, 1)
    servers._states[0].metrics = {"running": 1, "waiting": 0}
    servers._states[0].metrics_at = servers._clock()
    servers._states[0].local_at_metrics = 1
    next_server = servers.reserve()
    assert next_server == 0
    for index in (a, b, next_server):
        servers.release(index, duration_seconds=1)


def test_recent_latency_prefers_faster_servers():
    servers = pool()
    first = servers.reserve()
    servers.release(first, duration_seconds=40)
    second = servers.reserve()
    assert second == 1
    servers.release(second, duration_seconds=1)
    faster = servers.reserve()
    assert faster == 1
    servers.release(faster, duration_seconds=1)


def test_failed_server_gets_one_probe_after_cooldown_then_recovers():
    now = [100.0]
    servers = pool(clock=lambda: now[0])
    a = servers.reserve()
    servers.release(a, duration_seconds=1, transport_failed=True, succeeded=False)
    b = servers.reserve()
    assert b == 1
    servers.release(b, duration_seconds=1)
    now[0] += 3
    probe = servers.reserve()
    assert probe == 0
    while_probe_runs = servers.reserve()
    assert while_probe_runs == 1
    servers.release(while_probe_runs, duration_seconds=1)
    servers.release(probe, duration_seconds=1)
    assert servers.snapshot()[0]["cooldown_seconds"] == 0


def test_metrics_fetch_never_blocks_routing_and_failure_falls_back_to_local_load():
    started, finish = threading.Event(), threading.Event()

    def blocked_metrics(_endpoint, _key):
        started.set()
        assert finish.wait(5)
        raise OSError("metrics unavailable")

    servers = pool(metrics_reader=blocked_metrics)
    try:
        index = servers.reserve()  # Must return while the metrics reader remains blocked.
        assert started.wait(5)
        assert servers.snapshot()[index]["in_flight"] == 1
        servers.release(index, duration_seconds=1)
    finally:
        finish.set()


def test_capacity_wait_respects_logical_deadline():
    servers = pool()
    held = [servers.reserve() for _ in range(servers.capacity)]
    try:
        with pytest.raises(TimeoutError, match="deadline expired"):
            servers.reserve(deadline=time.monotonic() + .02)
    finally:
        for index in held:
            servers.release(index, duration_seconds=1)


def test_vllm_metrics_parser_aggregates_engines_and_ignores_unrelated_samples(monkeypatch):
    from io import BytesIO
    raw = b'''# HELP irrelevant
vllm:num_requests_running{engine="0"} 3
vllm:num_requests_running{engine="1",model_name="model with spaces"} 5
vllm:num_requests_waiting{engine="0"} 2
vllm:kv_cache_usage_perc{engine="0"} 0.2
vllm:kv_cache_usage_perc{engine="1"} 0.9
vllm:num_requests_waiting_by_reason{reason="capacity"} 2
vllm:kv_cache_usage_perc{engine="broken"} NaN
'''
    seen = []

    def open_metrics(request, timeout):
        seen.append((request.full_url, request.headers, timeout))
        return BytesIO(raw)

    monkeypatch.setattr(routing, "urlopen", open_metrics)
    assert routing.read_vllm_load("http://a.test/v1", "secret") == {"running": 8, "waiting": 2, "cache_fraction": .9}
    assert seen[0][0] == "http://a.test/metrics"
    assert seen[0][1]["Authorization"] == "Bearer secret"


def test_pool_addresses_and_caps_do_not_change_definition_fingerprint():
    single = config(extraction_llm={"endpoint": "http://a.test/v1", "model": "extract", "workers": 3})
    multiple = config()
    arguments = dict(clinical_question="test", outer_fold=1, discovery_packets=[], seed=42)
    assert workflow._feature_definition_input_value(config=single, **arguments) == workflow._feature_definition_input_value(config=multiple, **arguments)


def test_cli_can_replace_saved_single_route_with_pool_and_back(tmp_path):
    from oci.inference import research_all_evidence_workflow as launch
    source = tmp_path / "run_config.json"
    source.write_text(json.dumps({"stage2": config(
        extraction_llm={"endpoint": "http://old.test/v1", "model": "extract"}
    ).public_dict()}))
    endpoints = [server.public_dict() for server in config().extraction_llm.endpoints]
    arguments = launch.build_parser().parse_args([
        "--config", str(source), "--stage2-only", "--stage2-extraction-endpoints", json.dumps(endpoints),
        "--stage2-extraction-workers", "3",
    ])
    raw, _ = launch._raw_config_from_args(arguments)
    assert "endpoint" not in raw["stage2"]["extraction_llm"]
    multiple = workflow.plain_stage2_config_from_mapping(raw["stage2"], default_workers=4)
    assert multiple.extraction_llm.endpoints == config().extraction_llm.endpoints
    source.write_text(json.dumps(raw))
    arguments = launch.build_parser().parse_args([
        "--config", str(source), "--stage2-extraction-endpoint", "http://new.test/v1",
    ])
    raw, _ = launch._raw_config_from_args(arguments)
    single = workflow.plain_stage2_config_from_mapping(raw["stage2"], default_workers=4)
    assert single.extraction_llm.endpoint == "http://new.test/v1"
    assert not single.extraction_llm.endpoints
