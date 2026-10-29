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


@pytest.mark.parametrize("cohort", ["one_conf_one_mod"])
@pytest.mark.parametrize("preset", ["external"])
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
