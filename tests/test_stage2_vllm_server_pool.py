from __future__ import annotations

import concurrent.futures
import json
import os
import signal
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

import oci.inference.plain_handoff_stage2 as stage2_workflow
import oci.inference.vllm_server_pool as server_pool
from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    PlainHandoffStage2Config,
    plain_stage2_config_from_mapping,
    run_plain_handoff_stage2,
)
from oci.inference.vllm_server_pool import (
    ManagedVLLMConfig,
    ManagedVLLMServerPool,
)
from oci.inference.research_all_evidence_workflow import (
    _raw_config_from_args,
    build_parser,
    compile_config,
)


def test_managed_gemma_defaults_use_request_scoped_reasoning():
    config = plain_stage2_config_from_mapping(
        {
            "model": "google/gemma-4-31B-it",
            "vllm": {
                "server_count": 2,
                "gpus": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                "download_dir": "/models/cache",
                "extra_args": ["--gpu-memory-utilization", "0.85"],
            },
        },
        default_workers=8,
    )

    assert config is not None
    assert config.endpoint == ""
    assert config.interpretation_reasoning_effort == "high"
    assert config.extraction_reasoning_effort == "none"
    assert config.vllm is not None
    assert config.vllm.reasoning_parser == "gemma4"
    assert config.vllm.language_model_only is True
    assert config.vllm.default_chat_template_kwargs is None
    assert config.vllm.gpu_groups() == (
        ("cuda:0", "cuda:1"),
        ("cuda:2", "cuda:3"),
    )

    command = server_pool._build_vllm_command(
        config.vllm,
        model=config.model,
        api_key="EMPTY",
        port=8010,
        tensor_parallel_size=2,
    )

    assert command[:5] == [
        server_pool.sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "google/gemma-4-31B-it",
    ]
    assert command[command.index("--reasoning-parser") + 1] == "gemma4"
    assert "--language-model-only" in command
    assert "--default-chat-template-kwargs" not in command
    assert command[command.index("--download-dir") + 1] == "/models/cache"
    assert command[-2:] == ["--gpu-memory-utilization", "0.85"]


def test_managed_qwen_defaults_can_be_overridden():
    default_config = plain_stage2_config_from_mapping(
        {
            "model": "Qwen/Qwen3.8-27B",
            "vllm": {"server_count": 2, "gpus": "0,1"},
        },
        default_workers=4,
    )
    overridden = plain_stage2_config_from_mapping(
        {
            "model": "Qwen/Qwen3.8-27B",
            "interpretation_reasoning_effort": "low",
            "vllm": {
                "server_count": 2,
                "gpus": [0, 1],
                "reasoning_parser": "custom-parser",
                "language_model_only": False,
                "default_chat_template_kwargs": {"enable_thinking": True},
            },
        },
        default_workers=4,
    )

    assert default_config is not None and default_config.vllm is not None
    assert default_config.vllm.reasoning_parser == "qwen3"
    assert default_config.vllm.language_model_only is True
    assert default_config.vllm.default_chat_template_kwargs is None
    assert default_config.interpretation_reasoning_effort == "high"
    assert default_config.extraction_reasoning_effort == "none"
    assert overridden is not None and overridden.vllm is not None
    assert overridden.vllm.reasoning_parser == "custom-parser"
    assert overridden.vllm.language_model_only is False
    assert overridden.interpretation_reasoning_effort == "low"


def test_dual_managed_pools_derive_independent_replica_layouts():
    config = plain_stage2_config_from_mapping(
        {
            "model": "Qwen/Qwen3.8-27B",
            "workers": 12,
            "vllm": {
                "gpus": [0, 1, 2, 3],
                "gpus_per_server": 2,
            },
            "extraction_llm": {
                "model": "LiquidAI/LFM2.5-2.6B",
                "workers": 40,
                "vllm": {
                    "gpus": [4, 5, 6],
                    "gpus_per_server": 1,
                },
            },
        },
        default_workers=4,
    )

    assert config is not None and config.vllm is not None
    assert config.endpoint == ""
    assert config.vllm.server_count == 2
    assert config.vllm.effective_gpus_per_server() == 2
    assert config.vllm.gpu_groups() == (
        ("cuda:0", "cuda:1"),
        ("cuda:2", "cuda:3"),
    )
    assert config.vllm.effective_ports() == (8010, 8011)
    assert config.vllm.internal_port_base == 20_000
    assert config.extraction_llm is not None
    assert config.extraction_llm.endpoint == ""
    assert config.extraction_llm.workers == 40
    assert config.extraction_llm.vllm is not None
    assert config.extraction_llm.vllm.server_count == 3
    assert config.extraction_llm.vllm.effective_gpus_per_server() == 1
    assert config.extraction_llm.vllm.gpu_groups() == (
        ("cuda:4",),
        ("cuda:5",),
        ("cuda:6",),
    )
    assert config.extraction_llm.vllm.effective_ports() == (8110, 8111, 8112)
    assert config.extraction_llm.vllm.internal_port_base == 40_000


def test_pre_extraction_pool_uses_union_with_primary_tensor_parallel_width():
    primary = ManagedVLLMConfig(
        server_count=1,
        gpus=("cuda:0", "cuda:1"),
        gpus_per_server=2,
        ports=(9010,),
    )
    extraction = ManagedVLLMConfig(
        server_count=2,
        gpus=("cuda:2", "cuda:3"),
        gpus_per_server=1,
    )

    combined = stage2_workflow._all_gpu_interpretation_vllm_config(
        primary,
        extraction,
    )

    assert combined.gpus == ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    assert combined.gpus_per_server == 2
    assert combined.gpu_groups() == (
        ("cuda:0", "cuda:1"),
        ("cuda:2", "cuda:3"),
    )
    assert combined.effective_ports() == (9010, 9011)


def test_pre_extraction_pool_widens_one_server_when_union_is_not_even():
    primary = ManagedVLLMConfig(
        server_count=1,
        gpus=("cuda:0", "cuda:1"),
        gpus_per_server=2,
    )
    extraction = ManagedVLLMConfig(
        server_count=1,
        gpus=("cuda:2",),
        gpus_per_server=1,
    )

    combined = stage2_workflow._all_gpu_interpretation_vllm_config(
        primary,
        extraction,
    )

    assert combined.server_count == 1
    assert combined.gpus_per_server == 3
    assert combined.gpu_groups() == (("cuda:0", "cuda:1", "cuda:2"),)
    assert combined.effective_ports() == (8010,)


def test_extraction_pool_uses_union_with_extractor_replica_width():
    primary = ManagedVLLMConfig(
        server_count=1,
        gpus=("cuda:0", "cuda:1"),
        gpus_per_server=2,
    )
    extraction = ManagedVLLMConfig(
        server_count=2,
        gpus=("cuda:2", "cuda:3"),
        gpus_per_server=1,
        ports=(9110, 9111),
    )

    combined = stage2_workflow._all_gpu_extraction_vllm_config(
        primary,
        extraction,
    )

    assert combined.gpus == ("cuda:2", "cuda:3", "cuda:0", "cuda:1")
    assert combined.gpus_per_server == 1
    assert combined.gpu_groups() == (
        ("cuda:2",),
        ("cuda:3",),
        ("cuda:0",),
        ("cuda:1",),
    )
    assert combined.effective_ports() == (9110, 9111, 9112, 9113)


def test_managed_switch_tracker_only_marks_intervals_below_the_cutoff_as_rapid(
    monkeypatch,
):
    monotonic_values = iter((100.0, 159.9, 230.0))
    monkeypatch.setattr(
        stage2_workflow.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    tracker = stage2_workflow._ManagedStage2SwitchTracker(
        rapid_switch_seconds=60.0
    )

    assert tracker.mark_switch() is None
    rapid_elapsed = tracker.mark_switch()
    assert rapid_elapsed == pytest.approx(59.9)
    assert tracker.is_rapid(rapid_elapsed)
    slow_elapsed = tracker.mark_switch()
    assert slow_elapsed == pytest.approx(70.1)
    assert not tracker.is_rapid(slow_elapsed)

    disabled = stage2_workflow._ManagedStage2SwitchTracker(
        rapid_switch_seconds=0.0
    )
    assert not disabled.is_rapid(0.0)


@pytest.mark.parametrize(
    ("vllm", "message"),
    [
        (
            {"server_count": 3, "gpus": [0, 1, 2, 3]},
            "divide evenly",
        ),
        (
            {"server_count": 3, "gpus": [0, 1]},
            "cannot exceed",
        ),
        (
            {"server_count": 2, "gpus": [0, 0]},
            "duplicate",
        ),
        (
            {"server_count": 1, "gpus": [0], "extra_args": ["--port=9000"]},
            "cannot override",
        ),
        (
            {"server_count": 2, "gpus": [0, 1, 2, 3], "gpus_per_server": 1},
            "disagree",
        ),
    ],
)
def test_managed_vllm_rejects_unsafe_or_ambiguous_layouts(vllm, message):
    with pytest.raises(ValueError, match=message):
        plain_stage2_config_from_mapping(
            {"model": "Qwen/Qwen3.8-27B", "vllm": vllm},
            default_workers=4,
        )


def test_external_endpoint_and_managed_pool_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either stage2.endpoint or stage2.vllm"):
        plain_stage2_config_from_mapping(
            {
                "endpoint": "http://stage2.test/v1",
                "model": "Qwen/Qwen3.8-27B",
                "vllm": {"server_count": 1, "gpus": [0]},
            },
            default_workers=4,
        )


def test_extraction_endpoint_and_managed_pool_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either stage2.extraction_llm.endpoint"):
        plain_stage2_config_from_mapping(
            {
                "endpoint": "http://stage2.test/v1",
                "model": "large-model",
                "extraction_llm": {
                    "endpoint": "http://extract.test/v1",
                    "model": "small-model",
                    "vllm": {"gpus": [0], "gpus_per_server": 1},
                },
            },
            default_workers=4,
        )


def test_managed_vllm_cli_options_compile_into_stage2_config(tmp_path):
    args = build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "cohort.parquet"),
            "--output-dir",
            str(tmp_path / "output"),
            "--stage2-only",
            "--stage2-model",
            "Qwen/Qwen3.8-27B",
            "--stage2-vllm-servers",
            "2",
            "--stage2-vllm-gpus",
            "cuda:0,cuda:1",
            "--stage2-vllm-gpus-per-server",
            "1",
            "--stage2-vllm-rapid-switch-seconds",
            "1200",
            "--stage2-vllm-base-port",
            "9100",
            "--stage2-vllm-download-dir",
            "/models/cache",
            "--stage2-vllm-reasoning-parser",
            "custom-qwen",
            "--no-stage2-vllm-language-model-only",
            "--stage2-vllm-default-chat-template-kwargs",
            '{"enable_thinking":true}',
            "--stage2-vllm-extra-arg=--max-model-len",
            "--stage2-vllm-extra-arg=65536",
        ]
    )
    raw, config_dir = _raw_config_from_args(args)
    config = compile_config(raw, config_dir=config_dir)

    assert config.mode == "stage2"
    assert config.stage2 is not None and config.stage2.vllm is not None
    assert config.stage2.vllm.server_count == 2
    assert config.stage2.vllm.gpus == ("cuda:0", "cuda:1")
    assert config.stage2.vllm.gpus_per_server == 1
    assert config.stage2.vllm_rapid_switch_seconds == 1_200.0
    assert config.stage2.vllm.effective_ports() == (9100, 9101)
    assert config.stage2.vllm.download_dir == "/models/cache"
    assert config.stage2.vllm.reasoning_parser == "custom-qwen"
    assert config.stage2.vllm.language_model_only is False
    assert config.stage2.vllm.default_chat_template_kwargs == {"enable_thinking": True}
    assert config.stage2.vllm.extra_args == ("--max-model-len", "65536")


def test_managed_extraction_vllm_cli_options_compile_into_nested_config(tmp_path):
    args = build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "cohort.parquet"),
            "--output-dir",
            str(tmp_path / "output"),
            "--stage2-only",
            "--stage2-endpoint",
            "http://review.test/v1",
            "--stage2-model",
            "large-reviewer",
            "--stage2-extraction-model",
            "LiquidAI/LFM2.5-2.6B",
            "--stage2-extraction-workers",
            "48",
            "--stage2-extraction-vllm-gpus",
            "cuda:4,cuda:5,cuda:6,cuda:7",
            "--stage2-extraction-vllm-gpus-per-server",
            "2",
            "--stage2-extraction-vllm-base-port",
            "9200",
            "--stage2-extraction-vllm-internal-port-base",
            "45000",
            "--stage2-extraction-vllm-download-dir",
            "/extractor/cache",
            "--stage2-extraction-vllm-extra-arg=--gpu-memory-utilization",
            "--stage2-extraction-vllm-extra-arg=0.75",
        ]
    )
    raw, config_dir = _raw_config_from_args(args)
    config = compile_config(raw, config_dir=config_dir)

    assert config.stage2 is not None
    extraction = config.stage2.extraction_llm
    assert extraction is not None and extraction.vllm is not None
    assert extraction.endpoint == ""
    assert extraction.model == "LiquidAI/LFM2.5-2.6B"
    assert extraction.workers == 48
    assert extraction.vllm.server_count == 2
    assert extraction.vllm.gpus_per_server == 2
    assert extraction.vllm.gpu_groups() == (
        ("cuda:4", "cuda:5"),
        ("cuda:6", "cuda:7"),
    )
    assert extraction.vllm.effective_ports() == (9200, 9201)
    assert extraction.vllm.internal_port_base == 45_000
    assert extraction.vllm.download_dir == "/extractor/cache"
    assert extraction.vllm.extra_args == ("--gpu-memory-utilization", "0.75")


def test_visible_gpu_mapping_respects_parent_cuda_remapping():
    assert server_pool._visible_gpu_tokens(
        ("cuda:0", "cuda:2"),
        parent_cuda_visible_devices="5,GPU-abcdef,7",
    ) == ("5", "7")

    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        server_pool._visible_gpu_tokens(
            ("cuda:3",),
            parent_cuda_visible_devices="5,7",
        )
    with pytest.raises(ValueError, match="hides all GPUs"):
        server_pool._visible_gpu_tokens(
            ("cuda:0",),
            parent_cuda_visible_devices="",
        )


def test_stage2_round_robins_requests_and_transport_retries_across_endpoints(monkeypatch):
    called_endpoints: list[str] = []

    def fake_openai_completion(_messages, config):
        called_endpoints.append(config.endpoint)
        if len(called_endpoints) == 1:
            raise stage2_workflow._RetryableStage2ResponseError("temporary failure")
        return "{}"

    monkeypatch.setattr(stage2_workflow, "_openai_completion", fake_openai_completion)
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda _config: ["test-model"],
    )
    config = PlainHandoffStage2Config(
        endpoint="http://127.0.0.1:8010/v1",
        model="test-model",
        runtime_endpoints=(
            "http://127.0.0.1:8010/v1",
            "http://127.0.0.1:8011/v1",
            "http://127.0.0.1:8012/v1",
        ),
        transport_retry_backoff=0.0,
    )
    runner = PlainHandoffStage2(config=config, clinical_question="test")

    assert stage2_workflow._completion_with_transport_retries(
        [],
        runner.config,
        runner.completion,
    ) == "{}"
    for _ in range(5):
        assert runner.completion([], runner.config) == "{}"

    assert called_endpoints == [
        "http://127.0.0.1:8010/v1",
        "http://127.0.0.1:8011/v1",
        "http://127.0.0.1:8012/v1",
        "http://127.0.0.1:8010/v1",
        "http://127.0.0.1:8011/v1",
        "http://127.0.0.1:8012/v1",
        "http://127.0.0.1:8010/v1",
    ]


@pytest.mark.parametrize("logical_request", [False, True])
def test_stage2_workers_bound_completion_concurrency_globally(logical_request):
    lock = threading.Lock()
    release = threading.Event()
    saturated = threading.Event()
    active = 0
    peak = 0

    def blocking_completion(_messages, _config):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                saturated.set()
        try:
            assert release.wait(timeout=2.0)
            return "{}"
        finally:
            with lock:
                active -= 1

    runner = PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            workers=2,
        ),
        clinical_question="test",
        completion=blocking_completion,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        def request():
            if logical_request:
                return stage2_workflow._request_json(
                    messages=[], config=runner.config,
                    completion=runner.completion, validate=dict,
                )
            return runner.completion([], runner.config)

        futures = [executor.submit(request) for _ in range(6)]
        try:
            assert saturated.wait(timeout=2.0)
            with lock:
                assert active == 2
                assert peak == 2
        finally:
            release.set()
        assert [future.result(timeout=2.0) for future in futures] == (
            [{}] * 6 if logical_request else ["{}"] * 6
        )


def test_run_wrapper_keeps_managed_servers_alive_for_stage2_and_cleans_up(
    tmp_path,
    monkeypatch,
):
    config = plain_stage2_config_from_mapping(
        {
            "model": "Qwen/Qwen3.8-27B",
            "vllm": {"server_count": 2, "gpus": [0, 1]},
        },
        default_workers=4,
    )
    assert config is not None
    lifecycle: list[str] = []
    captured = {}

    @contextmanager
    def fake_launch(**kwargs):
        lifecycle.append("started")
        assert kwargs["output_dir"] == tmp_path / "stage2" / "vllm_servers" / "orchestrator"
        try:
            yield ("http://127.0.0.1:8010/v1", "http://127.0.0.1:8011/v1")
        finally:
            lifecycle.append("stopped")

    def fake_run(self, **_kwargs):
        lifecycle.append("stage2")
        captured["config"] = self.config
        return {"ok": True}

    monkeypatch.setattr(stage2_workflow, "launch_managed_vllm_servers", fake_launch)
    monkeypatch.setattr(PlainHandoffStage2, "run", fake_run)
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda _config: ["Qwen/Qwen3.8-27B"],
    )

    result = run_plain_handoff_stage2(
        handoff_path=tmp_path / "handoff.jsonl",
        output_dir=tmp_path / "stage2",
        clinical_question="test",
        config=config,
    )

    assert result == {"ok": True}
    assert lifecycle == ["started", "stage2", "stopped"]
    assert captured["config"].runtime_endpoints == (
        "http://127.0.0.1:8010/v1",
        "http://127.0.0.1:8011/v1",
    )


def test_runtime_disabled_extractor_preserves_identity_without_launching_pool(
    tmp_path,
    monkeypatch,
):
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://primary.test/v1",
            "model": "large-reviewer",
            "runtime_disable_extraction": True,
            "extraction_llm": {
                "model": "small-extractor",
                "vllm": {"gpus": [1], "gpus_per_server": 1},
            },
        },
        default_workers=4,
    )
    assert config is not None
    assert config.runtime_disable_extraction is True
    assert "runtime_disable_extraction" not in config.public_dict()

    def unexpected_launch(**_kwargs):
        raise AssertionError("the runtime-disabled extractor must not be launched")

    def fake_run(self, **kwargs):
        assert kwargs["dataset"] is not None
        assert self.model_identity["extraction"]["selected_model"] == "small-extractor"
        assert self.extraction_request_config is not None
        assert self.extraction_completion is not None
        with pytest.raises(RuntimeError, match="frozen measurement caches"):
            self.extraction_completion([], self.extraction_request_config)
        return {"phase": "post_extraction_reselection"}

    monkeypatch.setattr(
        stage2_workflow,
        "launch_managed_vllm_servers",
        unexpected_launch,
    )
    monkeypatch.setattr(PlainHandoffStage2, "run", fake_run)
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda request_config: [request_config.model],
    )

    result = run_plain_handoff_stage2(
        handoff_path=tmp_path / "handoff.jsonl",
        output_dir=tmp_path / "stage2",
        clinical_question="test",
        config=config,
        dataset=object(),
    )

    assert result == {"phase": "post_extraction_reselection"}


def test_feature_definition_only_run_never_starts_managed_extractor(
    tmp_path,
    monkeypatch,
):
    config = plain_stage2_config_from_mapping(
        {
            "model": "large-reviewer",
            "vllm": {"gpus": [0], "gpus_per_server": 1},
            "extraction_llm": {
                "model": "small-extractor",
                "vllm": {"gpus": [1], "gpus_per_server": 1},
            },
        },
        default_workers=4,
    )
    assert config is not None
    lifecycle: list[str] = []

    @contextmanager
    def fake_launch(**kwargs):
        role = kwargs["output_dir"].name
        lifecycle.append(f"start-{role}")
        endpoints = tuple(
            f"http://127.0.0.1:{port}/v1"
            for port in kwargs["config"].effective_ports()
        )
        try:
            yield endpoints
        finally:
            lifecycle.append(f"stop-{role}")

    def fake_run(self, **kwargs):
        assert kwargs["dataset"] is None
        lifecycle.append("stage2")
        return {"phase": "feature_definitions"}

    monkeypatch.setattr(stage2_workflow, "launch_managed_vllm_servers", fake_launch)
    monkeypatch.setattr(PlainHandoffStage2, "run", fake_run)
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda request_config: [request_config.model],
    )

    result = run_plain_handoff_stage2(
        handoff_path=tmp_path / "handoff.jsonl",
        output_dir=tmp_path / "stage2",
        clinical_question="test",
        config=config,
    )

    assert result == {"phase": "feature_definitions"}
    assert lifecycle == [
        "start-orchestrator_all_gpus",
        "stage2",
        "stop-orchestrator_all_gpus",
    ]


def test_managed_resume_after_extraction_starts_loads_all_gpu_extractor_immediately(
    tmp_path,
    monkeypatch,
):
    config = plain_stage2_config_from_mapping(
        {
            "model": "large-reviewer",
            "vllm": {"gpus": [0], "gpus_per_server": 1},
            "extraction_llm": {
                "model": "small-extractor",
                "vllm": {"gpus": [1], "gpus_per_server": 1},
            },
        },
        default_workers=4,
    )
    assert config is not None
    output_dir = tmp_path / "stage2"
    (output_dir / "outer_001" / "extraction").mkdir(parents=True)
    lifecycle: list[str] = []

    @contextmanager
    def fake_launch(**kwargs):
        role = kwargs["output_dir"].name
        lifecycle.append(f"start-{role}")
        assert role == "extractor_all_gpus"
        assert kwargs["config"].gpus == ("cuda:1", "cuda:0")
        endpoints = tuple(
            f"http://127.0.0.1:{port}/v1"
            for port in kwargs["config"].effective_ports()
        )
        try:
            yield endpoints
        finally:
            lifecycle.append(f"stop-{role}")

    def fake_run(self, **kwargs):
        assert kwargs["dataset"] is not None
        assert isinstance(
            self.extraction_tokenizer,
            stage2_workflow._LazyStage2ExtractionTokenizer,
        )
        lifecycle.append("stage2")
        return {"phase": "causal_estimation"}

    monkeypatch.setattr(stage2_workflow, "launch_managed_vllm_servers", fake_launch)
    monkeypatch.setattr(PlainHandoffStage2, "run", fake_run)
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda request_config: [request_config.model],
    )

    result = run_plain_handoff_stage2(
        handoff_path=tmp_path / "handoff.jsonl",
        output_dir=output_dir,
        clinical_question="test",
        config=config,
        dataset=object(),
    )

    assert result == {"phase": "causal_estimation"}
    assert lifecycle == [
        "start-extractor_all_gpus",
        "stage2",
        "stop-extractor_all_gpus",
    ]


def test_managed_resume_honors_persisted_interpretation_switch(tmp_path, monkeypatch):
    config = plain_stage2_config_from_mapping(
        {
            "model": "large-reviewer",
            "vllm": {"gpus": [0], "gpus_per_server": 1},
            "extraction_llm": {
                "model": "small-extractor",
                "vllm": {"gpus": [1], "gpus_per_server": 1},
            },
        },
        default_workers=4,
    )
    assert config is not None
    output_dir = tmp_path / "stage2"
    (output_dir / "outer_001" / "extraction").mkdir(parents=True)
    phase_path = output_dir / "vllm_servers" / "model_phase.json"
    phase_path.parent.mkdir(parents=True)
    phase_path.write_text(
        json.dumps(
            {
                "schema_version": stage2_workflow.MANAGED_MODEL_PHASE_SCHEMA_VERSION,
                "status": "switch_required",
                "active_role": "interpretation",
            }
        ),
        encoding="utf-8",
    )
    lifecycle = []

    @contextmanager
    def fake_launch(**kwargs):
        role = kwargs["output_dir"].name
        lifecycle.append(f"start-{role}")
        assert role == "orchestrator_all_gpus"
        assert kwargs["config"].gpus == ("cuda:0", "cuda:1")
        try:
            yield ("http://127.0.0.1:8010/v1", "http://127.0.0.1:8011/v1")
        finally:
            lifecycle.append(f"stop-{role}")

    def fake_run(self, **kwargs):
        assert kwargs["dataset"] is not None
        lifecycle.append("stage2")
        return {"phase": "causal_estimation"}

    monkeypatch.setattr(stage2_workflow, "launch_managed_vllm_servers", fake_launch)
    monkeypatch.setattr(PlainHandoffStage2, "run", fake_run)
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda request_config: [request_config.model],
    )

    result = run_plain_handoff_stage2(
        handoff_path=tmp_path / "handoff.jsonl",
        output_dir=output_dir,
        clinical_question="test",
        config=config,
        dataset=object(),
    )

    assert result == {"phase": "causal_estimation"}
    assert lifecycle == [
        "start-orchestrator_all_gpus",
        "stage2",
        "stop-orchestrator_all_gpus",
    ]


def test_managed_resume_retains_persisted_configured_split(tmp_path, monkeypatch):
    config = plain_stage2_config_from_mapping(
        {
            "model": "large-reviewer",
            "vllm": {"gpus": [0], "gpus_per_server": 1},
            "extraction_llm": {
                "model": "small-extractor",
                "vllm": {"gpus": [1], "gpus_per_server": 1},
            },
        },
        default_workers=4,
    )
    assert config is not None
    output_dir = tmp_path / "stage2"
    (output_dir / "outer_001" / "extraction").mkdir(parents=True)
    phase_path = output_dir / "vllm_servers" / "model_phase.json"
    phase_path.parent.mkdir(parents=True)
    phase_path.write_text(
        json.dumps(
            {
                "schema_version": stage2_workflow.MANAGED_MODEL_PHASE_SCHEMA_VERSION,
                "status": "running_configured_split",
                "active_role": "interpretation",
                "allocation_mode": "configured_split",
                "transition": 4,
                "rapid_switch_trigger_elapsed_seconds": 42.0,
            }
        ),
        encoding="utf-8",
    )
    lifecycle: list[str] = []

    @contextmanager
    def fake_launch(**kwargs):
        role = kwargs["output_dir"].name
        lifecycle.append(f"start-{role}")
        if role == "orchestrator":
            assert kwargs["config"].gpus == ("cuda:0",)
            endpoints = ("http://127.0.0.1:8010/v1",)
        else:
            assert role == "extractor"
            assert kwargs["config"].gpus == ("cuda:1",)
            endpoints = ("http://127.0.0.1:8110/v1",)
        try:
            yield endpoints
        finally:
            lifecycle.append(f"stop-{role}")

    def fake_run(self, **kwargs):
        assert kwargs["dataset"] is not None
        assert self.config.endpoint == "http://127.0.0.1:8010/v1"
        assert self.extraction_request_config is not None
        assert self.extraction_request_config.endpoint == "http://127.0.0.1:8110/v1"
        lifecycle.append("stage2")
        return {"phase": "causal_estimation"}

    monkeypatch.setattr(stage2_workflow, "launch_managed_vllm_servers", fake_launch)
    monkeypatch.setattr(PlainHandoffStage2, "run", fake_run)
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda request_config: [request_config.model],
    )

    result = run_plain_handoff_stage2(
        handoff_path=tmp_path / "handoff.jsonl",
        output_dir=output_dir,
        clinical_question="test",
        config=config,
        dataset=object(),
    )

    assert result == {"phase": "causal_estimation"}
    assert lifecycle == [
        "start-orchestrator",
        "start-extractor",
        "stage2",
        "stop-extractor",
        "stop-orchestrator",
    ]
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    assert phase["status"] == "complete"
    assert phase["allocation_mode"] == "configured_split"
    assert phase["transition"] == 4
    assert phase["rapid_switch_trigger_elapsed_seconds"] == 42.0


def test_rapid_managed_switch_falls_back_to_concurrent_configured_pools(
    tmp_path,
    monkeypatch,
):
    config = plain_stage2_config_from_mapping(
        {
            "model": "large-reviewer",
            "vllm": {"gpus": [0, 1], "gpus_per_server": 1},
            "extraction_llm": {
                "model": "small-extractor",
                "workers": 8,
                "vllm": {"gpus": [2, 3, 4], "gpus_per_server": 1},
            },
            "vllm_rapid_switch_seconds": 60,
        },
        default_workers=4,
    )
    assert config is not None
    lifecycle: list[str] = []
    calls: list[tuple[str, str]] = []
    captured = {}

    @contextmanager
    def fake_launch(**kwargs):
        role = kwargs["output_dir"].name
        lifecycle.append(f"start-{role}")
        if role == "orchestrator_all_gpus":
            assert kwargs["model"] == "large-reviewer"
            assert kwargs["config"].gpus == (
                "cuda:0",
                "cuda:1",
                "cuda:2",
                "cuda:3",
                "cuda:4",
            )
            assert kwargs["config"].gpu_groups() == (
                ("cuda:0",),
                ("cuda:1",),
                ("cuda:2",),
                ("cuda:3",),
                ("cuda:4",),
            )
            endpoints = tuple(
                f"http://127.0.0.1:{port}/v1" for port in range(8010, 8015)
            )
        elif role == "extractor_all_gpus":
            assert kwargs["model"] == "small-extractor"
            assert kwargs["config"].internal_port_base == 40_000
            assert kwargs["config"].gpus == (
                "cuda:2",
                "cuda:3",
                "cuda:4",
                "cuda:0",
                "cuda:1",
            )
            endpoints = tuple(
                f"http://127.0.0.1:{port}/v1" for port in range(8110, 8115)
            )
        elif role == "orchestrator":
            assert kwargs["model"] == "large-reviewer"
            assert kwargs["config"].gpus == ("cuda:0", "cuda:1")
            endpoints = (
                "http://127.0.0.1:8010/v1",
                "http://127.0.0.1:8011/v1",
            )
        else:
            assert role == "extractor"
            assert kwargs["model"] == "small-extractor"
            assert kwargs["config"].gpus == ("cuda:2", "cuda:3", "cuda:4")
            endpoints = (
                "http://127.0.0.1:8110/v1",
                "http://127.0.0.1:8111/v1",
                "http://127.0.0.1:8112/v1",
            )
        try:
            yield endpoints
        finally:
            lifecycle.append(f"stop-{role}")

    def fake_completion(_messages, request_config):
        calls.append((request_config.model, request_config.endpoint))
        return "{}"

    def fake_run(self, **kwargs):
        assert self.extraction_completion is not None
        assert self.extraction_request_config is not None
        if kwargs["dataset"] is None:
            lifecycle.append("stage2-interpretation")
            captured["interpretation_config"] = self.config
            for _ in range(4):
                self.completion([], self.config)
            return {"phase": "feature_definitions"}
        if self.config.endpoint.startswith("http://127.0.0.1:8110"):
            lifecycle.append("stage2-extraction")
            captured["all_gpu_extraction_config"] = self.config
            for _ in range(5):
                self.extraction_completion([], self.extraction_request_config)
            self.completion([], self.config)
            raise AssertionError("inactive primary completion should request a model switch")
        if not self.extraction_request_config.endpoint:
            lifecycle.append("stage2-orchestration")
            for _ in range(4):
                self.completion([], self.config)
            self.extraction_completion([], self.extraction_request_config)
            raise AssertionError("inactive extraction completion should request a model switch")
        lifecycle.append("stage2-concurrent")
        captured["configured_split"] = self.config
        for _ in range(4):
            self.completion([], self.config)
        for _ in range(5):
            self.extraction_completion([], self.extraction_request_config)
        return {"ok": True}

    monkeypatch.setattr(stage2_workflow, "launch_managed_vllm_servers", fake_launch)
    monkeypatch.setattr(stage2_workflow, "_openai_completion", fake_completion)
    monkeypatch.setattr(PlainHandoffStage2, "run", fake_run)
    # The extraction-to-interpretation interval is long enough to preserve
    # all-GPU alternation. The next transition is rapid and selects the split.
    monotonic_values = iter((100.0, 200.0, 230.0))
    monkeypatch.setattr(
        stage2_workflow.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        stage2_workflow,
        "_served_model_ids",
        lambda request_config: [request_config.model],
    )

    result = run_plain_handoff_stage2(
        handoff_path=tmp_path / "handoff.jsonl",
        output_dir=tmp_path / "stage2",
        clinical_question="test",
        config=config,
        dataset=object(),
    )

    assert result == {"ok": True}
    assert lifecycle == [
        "start-orchestrator_all_gpus",
        "stage2-interpretation",
        "stop-orchestrator_all_gpus",
        "start-extractor_all_gpus",
        "stage2-extraction",
        "stop-extractor_all_gpus",
        "start-orchestrator_all_gpus",
        "stage2-orchestration",
        "stop-orchestrator_all_gpus",
        "start-orchestrator",
        "start-extractor",
        "stage2-concurrent",
        "stop-extractor",
        "stop-orchestrator",
    ]
    assert calls[:4] == [
        ("large-reviewer", "http://127.0.0.1:8010/v1"),
        ("large-reviewer", "http://127.0.0.1:8011/v1"),
        ("large-reviewer", "http://127.0.0.1:8012/v1"),
        ("large-reviewer", "http://127.0.0.1:8013/v1"),
    ]
    assert calls[4:9] == [
        ("small-extractor", "http://127.0.0.1:8110/v1"),
        ("small-extractor", "http://127.0.0.1:8111/v1"),
        ("small-extractor", "http://127.0.0.1:8112/v1"),
        ("small-extractor", "http://127.0.0.1:8113/v1"),
        ("small-extractor", "http://127.0.0.1:8114/v1"),
    ]
    assert calls[9:13] == [
        ("large-reviewer", "http://127.0.0.1:8010/v1"),
        ("large-reviewer", "http://127.0.0.1:8011/v1"),
        ("large-reviewer", "http://127.0.0.1:8012/v1"),
        ("large-reviewer", "http://127.0.0.1:8013/v1"),
    ]
    assert calls[13:17] == [
        ("large-reviewer", "http://127.0.0.1:8010/v1"),
        ("large-reviewer", "http://127.0.0.1:8011/v1"),
        ("large-reviewer", "http://127.0.0.1:8010/v1"),
        ("large-reviewer", "http://127.0.0.1:8011/v1"),
    ]
    assert calls[17:] == [
        ("small-extractor", "http://127.0.0.1:8110/v1"),
        ("small-extractor", "http://127.0.0.1:8111/v1"),
        ("small-extractor", "http://127.0.0.1:8112/v1"),
        ("small-extractor", "http://127.0.0.1:8110/v1"),
        ("small-extractor", "http://127.0.0.1:8111/v1"),
    ]
    runtime_extraction = captured["configured_split"].extraction_llm
    assert runtime_extraction is not None
    assert runtime_extraction.runtime_endpoints == (
        "http://127.0.0.1:8110/v1",
        "http://127.0.0.1:8111/v1",
        "http://127.0.0.1:8112/v1",
    )
    phase = json.loads(
        (tmp_path / "stage2" / "vllm_servers" / "model_phase.json").read_text(
            encoding="utf-8"
        )
    )
    assert phase["status"] == "complete"
    assert phase["allocation_mode"] == "configured_split"
    assert phase["rapid_switch_seconds"] == 60.0
    assert phase["last_switch_at"].endswith("Z")
    assert phase["previous_switch_elapsed_seconds"] == 30.0
    assert phase["rapid_switch_trigger_elapsed_seconds"] == 30.0
    assert phase["configured_gpu_allocations"] == {
        "interpretation": ["cuda:0", "cuda:1"],
        "extraction": ["cuda:2", "cuda:3", "cuda:4"],
    }


def test_managed_pool_launches_one_process_per_gpu_and_writes_a_redacted_manifest(
    tmp_path,
    monkeypatch,
):
    popen_calls = []

    class FakeProcess:
        next_pid = 10_000

        def __init__(self, command, **kwargs):
            self.command = list(command)
            self.kwargs = kwargs
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.returncode = None
            popen_calls.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -signal.SIGTERM

        def kill(self):
            self.returncode = -signal.SIGKILL

        def wait(self, timeout=None):
            assert timeout is not None
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    monkeypatch.setenv("VLLM_PORT", "37973")
    monkeypatch.setattr(server_pool, "_assert_port_available", lambda *_args: None)
    monkeypatch.setattr(
        server_pool,
        "_server_model_ids",
        lambda *_args, **_kwargs: ["Qwen/Qwen3.8-27B"],
    )
    monkeypatch.setattr(server_pool.subprocess, "Popen", FakeProcess)

    def fake_signal(server, sig):
        if server.process.poll() is not None:
            return
        if sig == signal.SIGTERM:
            server.process.terminate()
        else:
            server.process.kill()

    monkeypatch.setattr(
        ManagedVLLMServerPool,
        "_signal_process_group",
        staticmethod(fake_signal),
    )
    config = ManagedVLLMConfig(
        server_count=2,
        gpus=("cuda:0", "cuda:1"),
        reasoning_parser="qwen3",
        language_model_only=True,
    )
    pool = ManagedVLLMServerPool(
        config=config,
        model="Qwen/Qwen3.8-27B",
        api_key="super-secret",
        output_dir=tmp_path / "servers",
    )

    assert pool.start() == (
        "http://127.0.0.1:8010/v1",
        "http://127.0.0.1:8011/v1",
    )
    assert [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in popen_calls] == [
        "4",
        "6",
    ]
    assert [call.kwargs["env"]["VLLM_PORT"] for call in popen_calls] == [
        "20000",
        "20128",
    ]
    interpreter_bin = str(Path(server_pool.sys.executable).parent)
    assert all(
        call.kwargs["env"]["PATH"].split(os.pathsep)[0] == interpreter_bin
        for call in popen_calls
    )
    assert all("--tensor-parallel-size" in call.command for call in popen_calls)
    assert all(
        call.command[call.command.index("--tensor-parallel-size") + 1] == "1"
        for call in popen_calls
    )

    pool.stop()

    manifest_text = (tmp_path / "servers" / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["status"] == "stopped"
    assert [server["exit_code"] for server in manifest["servers"]] == [
        -signal.SIGTERM,
        -signal.SIGTERM,
    ]
    assert "super-secret" not in manifest_text
    assert "<redacted>" in manifest_text
    assert [server["vllm_internal_port_base"] for server in manifest["servers"]] == [
        20_000,
        20_128,
    ]
    assert all(call.kwargs["stdout"].closed for call in popen_calls)


def test_managed_pool_cleans_up_other_servers_when_one_exits_during_startup(
    tmp_path,
    monkeypatch,
):
    processes = []

    class FakeProcess:
        def __init__(self, _command, **kwargs):
            self.pid = 20_000 + len(processes)
            self.returncode = 17 if processes else None
            self.stdout_handle = kwargs["stdout"]
            processes.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -signal.SIGTERM

        def kill(self):
            self.returncode = -signal.SIGKILL

        def wait(self, timeout=None):
            assert timeout is not None
            return self.returncode

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(server_pool, "_assert_port_available", lambda *_args: None)
    monkeypatch.setattr(
        server_pool,
        "_server_model_ids",
        lambda *_args, **_kwargs: ["Qwen/Qwen3.8-27B"],
    )
    monkeypatch.setattr(server_pool.subprocess, "Popen", FakeProcess)

    def fake_startup_failure_signal(server, sig):
        if server.process.poll() is not None:
            return
        if sig == signal.SIGTERM:
            server.process.terminate()
        else:
            server.process.kill()

    monkeypatch.setattr(
        ManagedVLLMServerPool,
        "_signal_process_group",
        staticmethod(fake_startup_failure_signal),
    )
    pool = ManagedVLLMServerPool(
        config=ManagedVLLMConfig(
            server_count=2,
            gpus=("cuda:0", "cuda:1"),
            startup_poll_interval=0.01,
        ),
        model="Qwen/Qwen3.8-27B",
        api_key="EMPTY",
        output_dir=tmp_path / "servers",
    )

    with pytest.raises(RuntimeError, match="server 1 exited with code 17"):
        pool.start()

    assert [process.returncode for process in processes] == [-signal.SIGTERM, 17]
    assert all(process.stdout_handle.closed for process in processes)
    manifest = json.loads(
        (tmp_path / "servers" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert "server 1 exited with code 17" in manifest["error"]


def test_managed_pool_context_stops_before_propagating_termination_signal(
    tmp_path,
    monkeypatch,
):
    stops = []

    class FakePool:
        def __init__(self, **_kwargs):
            self.stopped = False

        def start(self):
            return ("http://127.0.0.1:8010/v1",)

        def stop(self, *, final_status="stopped"):
            if not self.stopped:
                stops.append(final_status)
                self.stopped = True

    monkeypatch.setattr(server_pool, "ManagedVLLMServerPool", FakePool)
    original_handler = signal.getsignal(signal.SIGTERM)
    context = server_pool.launch_managed_vllm_servers(
        config=ManagedVLLMConfig(server_count=1, gpus=("cuda:0",)),
        model="test-model",
        api_key="EMPTY",
        output_dir=tmp_path / "servers",
    )
    assert context.__enter__() == ("http://127.0.0.1:8010/v1",)
    installed_handler = signal.getsignal(signal.SIGTERM)
    assert callable(installed_handler)

    try:
        with pytest.raises(SystemExit) as exc_info:
            installed_handler(signal.SIGTERM, None)
        assert exc_info.value.code == 128 + signal.SIGTERM
    finally:
        context.__exit__(None, None, None)

    assert stops == ["interrupted"]
    assert signal.getsignal(signal.SIGTERM) is original_handler
