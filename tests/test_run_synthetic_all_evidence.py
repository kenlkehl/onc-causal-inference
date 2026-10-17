from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from oci.inference.research_all_evidence_workflow import (
    _raw_config_from_args,
    build_parser,
    compile_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "run_synthetic.sh"
DATASETS = REPO_ROOT / "synthetic_data" / "example_synthetic_datasets"
COHORT_DATASETS = {
    "one_conf_one_mod": DATASETS
    / "one_confounder_one_effect_modifier_nsclc_with_structured"
    / "dataset.parquet",
    "five_conf_five_mod": DATASETS
    / "five_confounders_five_effect_modifiers_nsclc_with_structured"
    / "dataset.parquet",
}
FAKE_PYTHON = f"""#!{sys.executable}
import json, os, sys
with open(os.environ["FAKE_PYTHON_INVOCATION_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if os.environ.get("FAKE_PYTHON_ENVIRONMENT_LOG"):
    with open(os.environ["FAKE_PYTHON_ENVIRONMENT_LOG"], "a") as log:
        names = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        log.write(json.dumps({{name: os.environ.get(name) for name in names}}) + "\\n")
if os.environ.get("FAKE_PYTHON_LIMITS_LOG"):
    import resource
    with open(os.environ["FAKE_PYTHON_LIMITS_LOG"], "a") as log:
        log.write(json.dumps(resource.getrlimit(resource.RLIMIT_NOFILE)) + "\\n")
if sys.argv[1].endswith("detect_all_evidence_hardware.py"):
    args = sys.argv[1:]
    workers = args[args.index("--stage2-workers") + 1] if "--stage2-workers" in args else "32"
    devices = ",".join(f"cuda:{{index}}" for index in range(8))
    print(f"8\\t{{devices}}\\t12\\t{{workers}}\\t12\\teight eligible GPUs")
"""


def _launch(tmp_path: Path, *arguments: str, check: bool = True, **variables: str):
    log = tmp_path / "invocations.jsonl"
    log.unlink(missing_ok=True)
    fake_python = tmp_path / "python"
    fake_python.write_text(FAKE_PYTHON, encoding="utf-8")
    fake_python.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("STAGE1_", "STAGE2_", "OCI_"))
        and key not in {"PRESET", "GPU_COUNT", "PHYSICAL_GPUS", "DISABLE_HTR"}
    }
    environment.update(
        OCI_PYTHON=str(fake_python),
        FAKE_PYTHON_INVOCATION_LOG=str(log),
        **variables,
    )
    completed = subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )
    invocations = (
        [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if log.is_file()
        else []
    )
    return completed, invocations


def _bundled_cell_config(tmp_path: Path, **stage2) -> Path:
    """A benchmark-style configuration file: dataset, output, science, and Stage 2."""

    config = json.loads((REPO_ROOT / "benchmarks" / "base_config.json").read_text(encoding="utf-8"))
    config["dataset"] = str(COHORT_DATASETS["five_conf_five_mod"])
    config["output_dir"] = str(tmp_path / "cell_output")
    config["science"]["seed"] = 101
    config["stage2"].update(stage2)
    path = tmp_path / "cell" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _workflow_args(invocations):
    workflow = invocations[-1]
    assert workflow[:2] == ["-m", "oci.inference.research_all_evidence_workflow"]
    return workflow[2:]


def _compile(arguments):
    raw, config_dir = _raw_config_from_args(build_parser().parse_args(arguments))
    return compile_config(raw, config_dir=config_dir)


@pytest.mark.parametrize("cohort", ["one_conf_one_mod", "five_conf_five_mod"])
@pytest.mark.parametrize("preset", ["external", "h100x8", "rtxpro6000x8"])
def test_launcher_emits_compilable_cohort_and_preset_defaults(tmp_path, cohort, preset):
    completed, invocations = _launch(tmp_path, cohort, PRESET=preset)
    arguments = _workflow_args(invocations)
    config = _compile(arguments)

    managed = preset != "external"
    detect = invocations[-2]
    assert detect[0].endswith("detect_all_evidence_hardware.py")
    assert detect[detect.index("--min-free-vram-gib") + 1] == "0"
    assert ("find_spec" in " ".join(invocations[1])) is managed
    suffix = "" if preset == "external" else f"_{preset}"
    assert config.output_dir == (
        REPO_ROOT / "artifacts" / "research_all_evidence" / f"{cohort}_nsclc{suffix}_full"
    )
    assert config.dataset == COHORT_DATASETS[cohort]
    assert config.components[-1] == "stage2"
    assert config.stage1_architectures is None
    assert config.outer_folds == config.inner_folds == 5 and config.seed == 42
    assert config.devices == tuple(f"cuda:{index}" for index in range(8))
    assert config.workers == 12
    stage2 = config.stage2
    extraction = stage2.extraction_llm
    assert stage2.consolidation_batch_size == 20
    assert stage2.consolidation_max_rounds == 55
    assert stage2.operationalization_max_prompt_chars == 640_000
    assert stage2.ontology_refinement_min_failure_patients == 3
    assert stage2.max_ontology_refinement_rounds == 2

    one_conf_external = cohort == "one_conf_one_mod" and preset == "external"
    expected_workers = 4 if one_conf_external else 32
    assert stage2.workers == expected_workers
    assert extraction.workers == expected_workers
    assert stage2.modeling_workers == (128 if preset == "rtxpro6000x8" else 1)
    if preset == "external":
        assert stage2.endpoint == "http://127.0.0.1:8010/v1"
        assert extraction.endpoint == "http://127.0.0.1:8020/v1"
        assert stage2.model == "RedHatAI/Gemma-4-31B-IT-FP8-Dynamic"
        assert extraction.model == "google/gemma-4-e4b-it"
        assert stage2.vllm is None and extraction.vllm is None
        assert "http://127.0.0.1:8010/v1" in completed.stdout
    if one_conf_external or managed:
        assert stage2.request_attempt_timeout == 1800
        assert stage2.request_timeout == 6000
        assert stage2.extraction_context_margin_tokens == 4096
    window = {"external": 128_000, "h100x8": 128_000, "rtxpro6000x8": 262_144}[preset]
    if one_conf_external or managed:
        assert stage2.extraction_context_window_tokens == window
    if not managed:
        return

    primary_model = {
        "h100x8": "RedHatAI/gemma-4-31B-it-FP8-dynamic",
        "rtxpro6000x8": "nvidia/Gemma-4-26B-A4B-NVFP4",
    }[preset]
    extraction_model = {
        "h100x8": "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic",
        "rtxpro6000x8": "nvidia/Gemma-4-26B-A4B-NVFP4",
    }[preset]
    assert stage2.endpoint == "" and extraction.endpoint == ""
    assert stage2.model == primary_model
    assert extraction.model == extraction_model
    assert stage2.vllm_rapid_switch_seconds == 900
    for pool, gpus, port in (
        (stage2.vllm, ("cuda:0", "cuda:1", "cuda:2", "cuda:3"), 8010),
        (extraction.vllm, ("cuda:4", "cuda:5", "cuda:6", "cuda:7"), 8110),
    ):
        assert pool.gpu_groups() == tuple((gpu,) for gpu in gpus)
        assert pool.server_count == 4
        assert pool.base_port == port
        assert list(pool.extra_args) == [
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", str(window),
            "--max-num-seqs", "32",
        ]
    shared = "one shared pool" in completed.stdout
    assert shared is (preset == "rtxpro6000x8")


def test_endpoint_overrides_extraction_pool_and_passthrough_options(tmp_path):
    endpoints = [
        {"endpoint": "http://a.test/v1", "max_concurrency": 64},
        {"endpoint": "http://b.test/v1", "max_concurrency": 32},
    ]
    _completed, invocations = _launch(
        tmp_path,
        "five_conf_five_mod",
        str(tmp_path / "custom_output"),
        "--set",
        "stage2.consolidation_max_rounds=12",
        STAGE2_ENDPOINT="http://primary.test/v1",
        STAGE2_MODEL="custom-primary",
        STAGE2_EXTRACTION_ENDPOINTS=json.dumps(endpoints),
        STAGE2_EXTRACTION_MODEL="custom-extractor",
        STAGE2_WORKERS="8",
        STAGE2_REQUEST_TIMEOUT="3600",
        STAGE1_ARCHITECTURES="bow_nuisance,tfidf_topics",
    )
    arguments = _workflow_args(invocations)
    assert arguments[-2:] == ["--set", "stage2.consolidation_max_rounds=12"]
    assert "--stage2-extraction-endpoint" not in arguments
    config = _compile(arguments)

    assert config.output_dir == tmp_path / "custom_output"
    assert config.stage1_architectures == ("bow_nuisance", "tfidf_topics")
    assert config.stage2.endpoint == "http://primary.test/v1"
    assert config.stage2.model == "custom-primary"
    assert config.stage2.workers == 8
    assert config.stage2.request_timeout == 3600
    assert config.stage2.consolidation_max_rounds == 12
    extraction = config.stage2.extraction_llm
    assert extraction.model == "custom-extractor"
    assert [server.public_dict() for server in extraction.endpoints] == endpoints
    # Without an explicit limit the pool uses the sum of its per-server caps.
    assert "stage2.extraction_llm.workers=8" not in arguments


def test_managed_primary_from_environment_uses_detected_gpus(tmp_path):
    completed, invocations = _launch(
        tmp_path,
        "five_conf_five_mod",
        STAGE2_VLLM_SERVERS="2",
        STAGE2_MODEL="Qwen/Qwen3.8-27B",
        STAGE2_EXTRACTION_ENDPOINT="http://small.test/v1",
        STAGE2_EXTRACTION_MODEL="small-extractor",
    )
    config = _compile(_workflow_args(invocations))

    assert any("find_spec" in " ".join(invocation) for invocation in invocations)
    assert config.stage2.endpoint == ""
    assert config.stage2.vllm.server_count == 2
    assert config.stage2.vllm.gpus == tuple(f"cuda:{index}" for index in range(8))
    assert config.stage2.extraction_llm.endpoint == "http://small.test/v1"
    assert config.stage2.extraction_llm.vllm is None
    assert "managed vLLM Qwen/Qwen3.8-27B" in completed.stdout


def test_empty_stage2_endpoint_runs_stage1_only(tmp_path):
    completed, invocations = _launch(
        tmp_path, "one_conf_one_mod", STAGE2_ENDPOINT="", DISABLE_HTR="1"
    )
    arguments = _workflow_args(invocations)
    config = _compile(arguments)

    assert "--stage1-only" in arguments
    assert not any(value.startswith("stage2.") for value in arguments)
    assert config.mode == "stage1" and config.stage2 is None
    assert config.htr_enabled is False
    assert config.output_dir.name == "one_conf_one_mod_nsclc_full_no_htr"
    assert "disabled (STAGE2_ENDPOINT is empty)" in completed.stdout


@pytest.fixture
def saved_run(tmp_path):
    output_dir = tmp_path / "saved_output"
    raw = {
        "dataset": str(COHORT_DATASETS["one_conf_one_mod"]),
        "output_dir": str(output_dir),
        "stage2": {
            "endpoint": "http://saved.test/v1",
            "model": "saved-primary",
            "workers": 3,
            "extraction_llm": {"endpoint": "http://saved-small.test/v1", "model": "saved-small"},
        },
    }
    config = compile_config(raw, config_dir=tmp_path)
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text(
        json.dumps(config.as_dict(), default=str), encoding="utf-8"
    )
    return config


def test_saved_run_relaunch_preflights_then_runs_with_runtime_overrides_only(
    tmp_path, saved_run
):
    endpoints = [{"endpoint": "http://pool.test/v1", "max_concurrency": 4}]
    completed, invocations = _launch(
        tmp_path,
        "one_conf_one_mod",
        str(saved_run.output_dir),
        STAGE2_ONLY="1",
        PRESET="rtxpro6000x8",
        STAGE2_WORKERS="8",
        STAGE2_EXTRACTION_ENDPOINTS=json.dumps(endpoints),
    )

    assert len(invocations) == 2
    preflight, run = invocations
    assert preflight == [*run, "--preflight-only"]
    arguments = _workflow_args([run])
    assert arguments[:3] == [
        "--config", str(saved_run.output_dir / "run_config.json"), "--stage2-only",
    ]
    config = _compile(arguments)
    assert config.components == ("stage2",)
    # Presets and cohort defaults never override the saved run's settings.
    assert config.stage2.model == "saved-primary"
    assert config.stage2.vllm is None
    assert config.stage2.workers == 8
    extraction = config.stage2.extraction_llm
    assert extraction.model == "saved-small"
    assert extraction.endpoint == ""
    assert [server.public_dict() for server in extraction.endpoints] == endpoints
    assert "Saved run:" in completed.stdout


def test_saved_run_preflight_only_reselection_and_explicit_config(tmp_path, saved_run):
    _completed, invocations = _launch(
        tmp_path,
        "one_conf_one_mod",
        STAGE2_ONLY="1",
        STAGE2_RESELECT="1",
        OCI_PREFLIGHT_ONLY="1",
        OCI_RUN_CONFIG=str(saved_run.output_dir / "run_config.json"),
    )

    assert len(invocations) == 1
    arguments = _workflow_args(invocations)
    assert arguments == [
        "--config", str(saved_run.output_dir / "run_config.json"),
        "--stage2-only", "--stage2-reselect", "--preflight-only",
    ]


@pytest.mark.parametrize(
    "arguments,variables,message",
    [
        (["three_conf"], {}, "Usage:"),
        (["one_conf_one_mod"], {"PRESET": "a100"}, "Unknown PRESET=a100"),
        (["one_conf_one_mod"], {"STAGE2_SELECTION_MODE": "multi_model"}, "STAGE2_SELECTION_MODE"),
        (["one_conf_one_mod"], {"STAGE2_RESELECT": "1"}, "requires STAGE2_ONLY=1"),
        (["one_conf_one_mod"], {"OCI_PREFLIGHT_ONLY": "1"}, "requires STAGE2_ONLY=1"),
        (
            ["one_conf_one_mod"],
            {"STAGE2_ONLY": "1", "STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS": "1"},
            "STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS",
        ),
        (
            ["one_conf_one_mod", "/tmp/output"],
            {"STAGE2_ONLY": "1", "OCI_RUN_CONFIG": "/tmp/run_config.json"},
            "either OCI_RUN_CONFIG or OUTPUT_DIR",
        ),
        (["/nonexistent/config.json"], {}, "Configuration file not found"),
    ],
)
def test_launcher_rejects_invalid_requests_before_python(tmp_path, arguments, variables, message):
    completed, invocations = _launch(tmp_path, *arguments, check=False, **variables)

    assert completed.returncode == 2
    assert message in completed.stderr
    assert invocations == []


def test_config_file_mode_adds_only_machine_settings(tmp_path):
    config_path = _bundled_cell_config(tmp_path, consolidation_max_rounds=7)
    completed, invocations = _launch(
        tmp_path, str(config_path), "--set", "stage2.max_review_rounds=1",
        PRESET="rtxpro6000x8",
    )
    arguments = _workflow_args(invocations)
    assert arguments[:6] == [
        "--config", str(config_path),
        "--devices", ",".join(f"cuda:{index}" for index in range(8)),
        "--workers", "12",
    ]
    assert arguments[-2:] == ["--set", "stage2.max_review_rounds=1"]
    assert "--dataset" not in arguments and "--seed" not in arguments
    config = _compile(arguments)

    # The file's cohort, output, science, and Stage 2 settings stand.
    assert config.dataset == COHORT_DATASETS["five_conf_five_mod"]
    assert config.output_dir == tmp_path / "cell_output"
    assert config.seed == 101
    assert config.stage2.consolidation_max_rounds == 7
    assert config.stage2.max_review_rounds == 1
    # The preset supplies the serving settings.
    assert config.stage2.model == config.stage2.extraction_llm.model == "nvidia/Gemma-4-26B-A4B-NVFP4"
    assert config.stage2.vllm.server_count == config.stage2.extraction_llm.vllm.server_count == 4
    assert config.stage2.modeling_workers == 128
    assert f"Configuration:  {config_path}" in completed.stdout


def test_config_file_mode_with_external_servers_and_no_cohort_defaults(tmp_path):
    config_path = _bundled_cell_config(tmp_path)
    _completed, invocations = _launch(tmp_path, str(config_path))
    config = _compile(_workflow_args(invocations))

    assert config.stage2.endpoint == "http://127.0.0.1:8010/v1"
    assert config.stage2.extraction_llm.endpoint == "http://127.0.0.1:8020/v1"
    assert config.stage2.workers == config.stage2.extraction_llm.workers == 32
    assert config.stage2.consolidation_max_rounds == 55


def test_config_file_mode_rejects_stage2_only(tmp_path):
    completed, invocations = _launch(
        tmp_path, str(_bundled_cell_config(tmp_path)), check=False, STAGE2_ONLY="1"
    )

    assert completed.returncode == 2
    assert "OCI_RUN_CONFIG" in completed.stderr
    assert invocations == []


def test_dgx_station_preset_selects_the_gb300_and_one_shared_server(tmp_path):
    environment_log = tmp_path / "environment.jsonl"
    completed, invocations = _launch(
        tmp_path, "five_conf_five_mod", PRESET="dgx_station_gb300",
        FAKE_PYTHON_ENVIRONMENT_LOG=str(environment_log),
    )
    detect = invocations[-2]
    assert detect[detect.index("--min-free-vram-gib") + 1] == "0"
    assert detect[detect.index("--min-total-vram-gib") + 1] == "150"
    environments = [json.loads(line) for line in environment_log.read_text().splitlines()]
    assert environments[-1] == {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    config = _compile(_workflow_args(invocations))

    stage2 = config.stage2
    extraction = stage2.extraction_llm
    assert stage2.model == extraction.model == "nvidia/Gemma-4-26B-A4B-NVFP4"
    for pool in (stage2.vllm, extraction.vllm):
        assert pool.gpus == ("cuda:0",) and pool.server_count == 1
        assert list(pool.extra_args) == [
            "--gpu-memory-utilization", "0.90", "--max-model-len", "262144",
            "--max-num-seqs", "64",
        ]
    assert stage2.workers == extraction.workers == 32
    assert stage2.modeling_workers == 64
    assert stage2.extraction_context_window_tokens == 262_144
    assert "one shared pool" in completed.stdout


def test_dgx_station_preset_keeps_explicit_gpu_choices(tmp_path):
    environment_log = tmp_path / "environment.jsonl"
    _completed, invocations = _launch(
        tmp_path, "five_conf_five_mod", PRESET="dgx_station_gb300",
        PHYSICAL_GPUS="0", CUDA_DEVICE_ORDER="FASTEST_FIRST", MIN_TOTAL_GPU_GB="20",
        FAKE_PYTHON_ENVIRONMENT_LOG=str(environment_log),
    )
    detect = invocations[-2]
    assert detect[detect.index("--min-total-vram-gib") + 1] == "20"
    last = json.loads(environment_log.read_text().splitlines()[-1])
    assert last == {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}


def test_benchmark_driver_llm_url_environment_reaches_the_launcher(tmp_path):
    from oci.evaluation import benchmark_driver as driver

    environ = {key: value for key, value in os.environ.items()
               if not key.startswith(("STAGE1_", "STAGE2_", "OCI_", "CUDA_"))
               and key not in {"PRESET", "GPU_COUNT", "PHYSICAL_GPUS", "MIN_TOTAL_GPU_GB"}}
    servers = driver.LLMServers(("http://gb300:8000/v1",), "served-model", 100_000, 32)
    env = driver.external_run_environment(
        environ, driver.preset_environment("dgx_station_gb300", environ), servers
    )
    variables = {key: value for key, value in env.items() if key not in environ}
    environment_log = tmp_path / "environment.jsonl"
    _completed, invocations = _launch(
        tmp_path, str(_bundled_cell_config(tmp_path)),
        FAKE_PYTHON_ENVIRONMENT_LOG=str(environment_log), **variables,
    )
    detect = invocations[-2]
    assert detect[detect.index("--min-total-vram-gib") + 1] == "150"
    last = json.loads(environment_log.read_text().splitlines()[-1])
    assert last == {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    config = _compile(_workflow_args(invocations))

    stage2 = config.stage2
    extraction = stage2.extraction_llm
    assert stage2.vllm is None and extraction.vllm is None
    assert stage2.endpoint == "http://gb300:8000/v1" and stage2.model == "served-model"
    assert stage2.workers == 8
    assert [server.public_dict() for server in extraction.endpoints] == [
        {"endpoint": "http://gb300:8000/v1", "max_concurrency": 24},
    ]
    assert extraction.model == "served-model"
    assert stage2.extraction_context_window_tokens == 100_000
    assert stage2.modeling_workers == 64
    assert stage2.request_attempt_timeout == 1800


def test_launcher_raises_the_open_file_limit_for_many_concurrent_requests(tmp_path):
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limits_log = tmp_path / "limits.jsonl"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("STAGE1_", "STAGE2_", "OCI_"))}
    fake_python = tmp_path / "python"
    fake_python.write_text(FAKE_PYTHON, encoding="utf-8")
    fake_python.chmod(0o755)
    environment.update(OCI_PYTHON=str(fake_python), FAKE_PYTHON_LIMITS_LOG=str(limits_log),
                       FAKE_PYTHON_INVOCATION_LOG=str(tmp_path / "invocations.jsonl"))
    lowered = min(soft, 512)
    subprocess.run(
        ["bash", "-c", f'ulimit -Sn {lowered} && exec bash "$0" "$@"', str(LAUNCHER), "one_conf_one_mod"],
        cwd=REPO_ROOT, env=environment, check=True, capture_output=True, text=True,
    )
    limits = [json.loads(line) for line in limits_log.read_text().splitlines()]
    expected = hard if hard != resource.RLIM_INFINITY else 1048576
    assert limits and all(limit[0] == min(expected, hard) for limit in limits)
