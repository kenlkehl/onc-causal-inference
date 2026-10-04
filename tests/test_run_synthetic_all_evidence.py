from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("launcher", [
    "run_one_conf_one_mod.sh",
    "run_five_conf_five_mod.sh",
    "run_one_conf_one_mod_h100x8.sh",
    "run_one_conf_one_mod_rtxpro6000x8.sh",
    "run_five_conf_five_mod_rtxpro6000x8.sh",
])
@pytest.mark.parametrize("saved,overrides", [
    (False, {}),
    (False, {"STAGE2_DECISION_EXTRACTION": "0"}),
    (False, {
        "STAGE2_DECISION_EXTRACTION": "0",
        "STAGE2_VLLM_EXTRA_ARGS_JSON": '["--max-model-len","196608"]',
        "STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON": '["--max-model-len","131072"]',
        "STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS": "131072",
    }),
    (False, {
        "STAGE2_MODEL": "custom-gemma",
        "STAGE2_EXTRACTION_MODEL": "custom-plumb",
        "STAGE2_VLLM_GPUS": "cuda:0,cuda:1",
        "STAGE2_EXTRACTION_VLLM_GPUS": "cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7",
        "STAGE2_EXTRACTION_WORKERS": "24",
        "STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON": '["--max-model-len","3072","--gpu-memory-utilization","0.35"]',
    }),
    (True, {}),
])
def test_wrappers_select_backend_and_preserve_saved_settings(
    tmp_path: Path, launcher, saved, overrides,
):
    repo_root = Path(__file__).resolve().parents[1]
    invocation_log = tmp_path / "invocations.jsonl"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['FAKE_PYTHON_INVOCATION_LOG'], 'a') as log:\n"
        "    log.write(json.dumps({'args': sys.argv[1:], 'env': "
        "{k: v for k, v in os.environ.items() if k.startswith('STAGE2_')}}) + '\\n')\n"
        "if sys.argv[1].endswith('detect_all_evidence_hardware.py'):\n"
        "    print('8\\tcuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
        "\\t12\\t32\\t12\\teight eligible GPUs')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("STAGE2_", "OCI_"))
        and k not in {"GPU_COUNT", "PHYSICAL_GPUS", "DISABLE_HTR", "STAGE1_ARCHITECTURES"}
    }
    environment.update(
        OCI_PYTHON=str(fake_python),
        FAKE_PYTHON_INVOCATION_LOG=str(invocation_log),
        **overrides,
    )
    if saved:
        environment.update(STAGE2_ONLY="1", OCI_RUN_CONFIG=str(tmp_path / "saved.json"))
    completed = subprocess.run(
        ["bash", str(repo_root / launcher), str(tmp_path / "output")],
        cwd=repo_root, env=environment, check=True, capture_output=True, text=True,
    )
    invocations = [json.loads(line) for line in invocation_log.read_text().splitlines()]
    args = invocations[-1]["args"]
    if saved:
        assert len(invocations) == 1
        assert args[0].endswith("launch_saved_stage2.py")
        assert invocations[0]["env"] == {"STAGE2_ONLY": "1"}
        return

    assert "oci.inference.research_all_evidence_workflow" in args
    decision = overrides.get("STAGE2_DECISION_EXTRACTION", "1") == "1"
    rtx = "rtxpro6000" in launcher
    h100 = "h100" in launcher
    managed = rtx or h100
    primary = (
        "nvidia/Gemma-4-26B-A4B-NVFP4" if rtx else
        "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic" if h100 and decision else
        "RedHatAI/gemma-4-31B-it-FP8-dynamic" if h100 else
        "RedHatAI/Gemma-4-31B-IT-FP8-Dynamic"
    )
    legacy = (
        "nvidia/Gemma-4-26B-A4B-NVFP4" if rtx else
        "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic" if h100 else
        "google/gemma-4-e4b-it"
    )
    assert args[args.index("--stage2-model") + 1] == overrides.get("STAGE2_MODEL", primary)
    assert args[args.index("--stage2-extraction-model") + 1] == overrides.get(
        "STAGE2_EXTRACTION_MODEL", "crh225/plumb-4b" if decision else legacy,
    )
    assert args[args.index("--stage2-extraction-context-strategy") + 1] == "colbert"
    assert args[args.index("--stage2-colbert-cache-dir") + 1] == str(repo_root / ".oci_cache/colbert")
    settings = dict(args[i + 1].split("=", 1) for i, arg in enumerate(args) if arg == "--set")
    assert json.loads(settings["stage2.decision_extraction.enabled"]) is decision
    # Compile the actual shell-emitted CLI, catching invalid combinations of
    # backend, retrieval, primary model, and extraction server settings.
    from oci.inference.research_all_evidence_workflow import (
        _raw_config_from_args, build_parser, compile_config,
    )
    raw, config_dir = _raw_config_from_args(build_parser().parse_args(args[2:]))
    config = compile_config(raw, config_dir=config_dir)
    assert config.stage2.decision_extraction.enabled is decision
    assert config.stage2.decision_extraction.max_prompt_tokens == 3000
    assert config.stage2.extraction_context_strategy == "colbert"
    if not managed:
        assert config.stage2.extraction_llm.vllm is None
        return

    assert ("separate resident primary and decision pools" in completed.stdout) is decision
    primary_gpus = overrides.get(
        "STAGE2_VLLM_GPUS", "cuda:0" if decision else "cuda:0,cuda:1,cuda:2,cuda:3",
    ).split(",")
    extraction_gpus = overrides.get(
        "STAGE2_EXTRACTION_VLLM_GPUS",
        "cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7" if decision else "cuda:4,cuda:5,cuda:6,cuda:7",
    ).split(",")
    for pool, gpus in ((config.stage2.vllm, primary_gpus), (config.stage2.extraction_llm.vllm, extraction_gpus)):
        assert pool.server_count == len(gpus)
        assert pool.gpu_groups() == tuple((gpu,) for gpu in gpus)
    assert not set(primary_gpus) & set(extraction_gpus)
    assert set(primary_gpus + extraction_gpus) == {f"cuda:{i}" for i in range(8)}
    assert config.stage2.extraction_llm.workers == int(overrides.get(
        "STAGE2_EXTRACTION_WORKERS", "128" if decision else "32",
    ))
    for variable, setting in (
        ("STAGE2_VLLM_EXTRA_ARGS_JSON", "stage2.vllm.extra_args"),
        ("STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON", "stage2.extraction_llm.vllm.extra_args"),
    ):
        extra_args = json.loads(settings[setting])
        expected = json.loads(overrides[variable]) if variable in overrides else [
            "--gpu-memory-utilization", "0.90", "--max-model-len", "262144" if rtx else "128000",
            "--max-num-seqs", "32",
        ]
        if decision and variable == "STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON" and variable not in overrides:
            assert extra_args[extra_args.index("--max-model-len") + 1] == "3072"
            assert extra_args[extra_args.index("--gpu-memory-utilization") + 1] == "0.28"
            assert extra_args[extra_args.index("--revision") + 1] == config.stage2.decision_extraction.tokenizer_revision
            assert "--enforce-eager" in extra_args
            assert "--convert" not in extra_args  # Added by the backend, once.
            continue
        assert extra_args == expected
    context_flag = args.index("--stage2-extraction-context-window-tokens")
    assert args[context_flag + 1] == overrides.get(
        "STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS", "262144" if rtx else "128000",
    )


@pytest.mark.parametrize("value", ["true", "2", "invalid"])
def test_invalid_decision_backend_flag_fails_before_setup(tmp_path, value):
    repo_root = Path(__file__).resolve().parents[1]
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("STAGE2_", "OCI_"))}
    environment.update(STAGE2_DECISION_EXTRACTION=value, OCI_PYTHON="/does/not/exist")
    completed = subprocess.run(
        ["bash", str(repo_root / "run_one_conf_one_mod.sh"), str(tmp_path / "output")],
        env=environment, capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert "STAGE2_DECISION_EXTRACTION must be 0 or 1" in completed.stderr
    assert "Synchronizing" not in completed.stdout


@pytest.mark.parametrize("runtime_overrides", [
    {},
    {"STAGE2_WORKERS": "8", "STAGE2_EXTRACTION_WORKERS": "12",
     "STAGE2_REQUEST_TIMEOUT": "3600",
     "STAGE2_REQUEST_ATTEMPT_TIMEOUT": "1200"},
    {"STAGE2_EXTRACTION_ENDPOINTS": json.dumps([
        {"endpoint": "http://a.test/v1", "max_concurrency": 64},
        {"endpoint": "http://b.test/v1", "max_concurrency": 32},
    ]), "STAGE2_EXTRACTION_WORKERS": "96"},
])
def test_example_wrappers_run_both_stages_by_default(tmp_path: Path, runtime_overrides):
    repo_root = Path(__file__).resolve().parents[1]

    for launcher in ("run_one_conf_one_mod.sh", "run_five_conf_five_mod.sh"):
        output_dir = tmp_path / launcher.removesuffix(".sh")
        invocation_log = tmp_path / f"{launcher}.invocations"
        fake_python = tmp_path / f"{launcher}.python"
        fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${FAKE_PYTHON_INVOCATION_LOG}"
printf '\n' >> "${FAKE_PYTHON_INVOCATION_LOG}"
if [[ "${1:-}" == *detect_all_evidence_hardware.py ]]; then
    printf '2\tcuda:0,cuda:1\t12\t32\t12\ttwo eligible GPUs\n'
fi
""",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        environment = os.environ.copy()
        for key in tuple(environment):
            if key.startswith("STAGE2_"):
                environment.pop(key)
        environment.update(
            {
                "FAKE_PYTHON_INVOCATION_LOG": str(invocation_log),
                "GPU_COUNT": "2",
                "OCI_PYTHON": str(fake_python),
                "PHYSICAL_GPUS": "",
            }
        )
        environment.update(runtime_overrides)
        completed = subprocess.run(
            ["bash", str(repo_root / launcher), str(output_dir)],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        invocations = invocation_log.read_text(encoding="utf-8").splitlines()
        workflow = invocations[-1]
        if "STAGE2_EXTRACTION_ENDPOINTS" in runtime_overrides:
            assert "--stage2-extraction-endpoints" in workflow
            assert "--stage2-extraction-workers 96" in workflow
            assert "--stage2-extraction-endpoint " not in workflow
            assert "load-aware external extractor pool" in completed.stdout
        elif runtime_overrides:
            assert "--stage2-workers 8" in invocations[-2]
            assert "--stage2-extraction-workers 12" in workflow
            assert "stage2.request_attempt_timeout=1200" in workflow
            assert "stage2.request_timeout=3600" in workflow
        elif launcher == "run_one_conf_one_mod.sh":
            assert "--stage2-workers 4" in invocations[-2]
            assert "--stage2-extraction-workers 4" in workflow
            assert "stage2.request_attempt_timeout=1800" in workflow
            assert "stage2.request_timeout=6000" in workflow
        assert "research_all_evidence_workflow" in workflow
        assert "--stage2-endpoint http://127.0.0.1:8010/v1" in workflow
        if "STAGE2_EXTRACTION_ENDPOINTS" not in runtime_overrides:
            assert "--stage2-extraction-endpoint http://127.0.0.1:8020/v1" in workflow
            assert "extractor http://127.0.0.1:8020/v1" in completed.stdout
        assert "--stage1-only" not in workflow
        assert "Stage 2:        http://127.0.0.1:8010/v1" in completed.stdout


def test_completed_handoff_uses_stage2_only_without_gpu_probe(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "output"
    handoff_dir = output_dir / "handoff"
    handoff_dir.mkdir(parents=True)
    (handoff_dir / "evidence.jsonl").write_text("{}\n", encoding="utf-8")
    (handoff_dir / "complete.json").write_text("{}\n", encoding="utf-8")

    invocation_log = tmp_path / "python_invocations.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${FAKE_PYTHON_INVOCATION_LOG}"
printf '\n' >> "${FAKE_PYTHON_INVOCATION_LOG}"
if [[ "${1:-}" == *detect_all_evidence_hardware.py ]]; then
    printf '0\tcpu\t12\t32\t12\tnot inspected (endpoint-backed Stage 2 only)\n'
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_PYTHON_INVOCATION_LOG": str(invocation_log),
            "GPU_COUNT": "not-a-number",
            "OCI_PYTHON": str(fake_python),
            "PHYSICAL_GPUS": "also-not-a-number",
            "STAGE2_ENDPOINT": "http://stage2.test/v1",
            "STAGE2_EXTRACTION_ENDPOINT": "http://small-stage2.test/v1",
            "STAGE2_EXTRACTION_FEATURE_BATCH_SIZE": "7",
            "STAGE2_CLUSTER_SIMILARITY_THRESHOLD": "0.7",
            "STAGE2_CLUSTER_CONSENSUS_FRACTION": "0.8",
            "STAGE2_MAX_TOKENS": "150000",
            "STAGE2_REQUEST_TIMEOUT": "3600",
            "STAGE2_REQUEST_ATTEMPT_TIMEOUT": "1200",
            "STAGE2_EXTRACTION_MAX_TOKENS": "70000",
            "STAGE2_WORKERS": "",
            "STAGE2_VLLM_SERVERS": "0",
        }
    )
    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "run_synthetic_all_evidence.sh"),
            (
                "synthetic_data/example_synthetic_datasets/"
                "five_confounders_five_effect_modifiers_nsclc_with_structured/"
                "dataset.parquet"
            ),
            "test_output",
            str(output_dir),
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 2
    assert "detect_all_evidence_hardware.py" in invocations[0]
    assert "--stage2-only" in invocations[0]
    assert "--stage2-workers 32" in invocations[0]
    assert "-c" not in invocations[0].split()
    assert "research_all_evidence_workflow" in invocations[1]
    assert "--stage2-only" in invocations[1]
    assert "--devices cpu" in invocations[1]
    assert "stage2.workers=32" in invocations[1]
    assert "--stage2-extraction-endpoint http://small-stage2.test/v1" in invocations[1]
    assert "--stage2-extraction-feature-batch-size 7" in invocations[1]
    assert "--stage2-cluster-similarity-threshold 0.7" in invocations[1]
    assert "--stage2-cluster-consensus-fraction 0.8" in invocations[1]
    assert "--stage2-max-tokens 150000" in invocations[1]
    assert "stage2.request_timeout=3600" in invocations[1]
    assert "stage2.request_attempt_timeout=1200" in invocations[1]
    assert "--stage2-extraction-max-tokens 70000" in invocations[1]
    assert (
        "CUDA devices:   not required for endpoint-backed Stage 2" in completed.stdout
    )
    assert "HTR modeling:   not run during Stage 2-only resume" in completed.stdout


def test_managed_vllm_keeps_gpu_detection_and_pool_arguments(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "output"
    invocation_log = tmp_path / "python_invocations.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${FAKE_PYTHON_INVOCATION_LOG}"
printf '\n' >> "${FAKE_PYTHON_INVOCATION_LOG}"
if [[ "${1:-}" == *detect_all_evidence_hardware.py ]]; then
    printf '2\tcuda:0,cuda:1\t12\t32\t12\ttwo eligible GPUs\n'
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_PYTHON_INVOCATION_LOG": str(invocation_log),
            "GPU_COUNT": "2",
            "OCI_PYTHON": str(fake_python),
            "PHYSICAL_GPUS": "",
            "STAGE2_ENDPOINT": "",
            "STAGE2_EXTRACTION_ENDPOINT": "http://small-stage2.test/v1",
            "STAGE2_EXTRACTION_MODEL": "small-extractor",
            "STAGE2_EXTRACTION_FEATURE_BATCH_SIZE": "",
            "STAGE2_MODEL": "Qwen/Qwen3.8-27B",
            "STAGE2_VLLM_DOWNLOAD_DIR": "",
            "STAGE2_VLLM_EXTRA_ARGS_JSON": "",
            "STAGE2_VLLM_GPUS": "",
            "STAGE2_VLLM_SERVERS": "2",
            "STAGE2_WORKERS": "",
        }
    )
    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "run_synthetic_all_evidence.sh"),
            (
                "synthetic_data/example_synthetic_datasets/"
                "five_confounders_five_effect_modifiers_nsclc_with_structured/"
                "dataset.parquet"
            ),
            "test_output",
            str(output_dir),
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 4
    assert "sentence_transformers" in invocations[0]
    assert "find_spec" in invocations[1] and "vllm" in invocations[1]
    assert "detect_all_evidence_hardware.py" in invocations[2]
    assert "--gpu-count 2" in invocations[2]
    assert "--stage2-workers 32" in invocations[2]
    assert "--stage2-only" not in invocations[2]
    assert "research_all_evidence_workflow" in invocations[3]
    assert "--stage2-vllm-servers 2" in invocations[3]
    assert r"--stage2-vllm-gpus cuda:0\,cuda:1" in invocations[3]
    assert "--stage2-model Qwen/Qwen3.8-27B" in invocations[3]
    assert "--stage2-extraction-endpoint http://small-stage2.test/v1" in invocations[3]
    assert "--stage2-extraction-model small-extractor" in invocations[3]
    assert "stage2.workers=32" in invocations[3]
    assert "--stage2-only" not in invocations[3]
    assert (
        "Stage 2:        managed orchestrator vLLM: 2 servers on cuda:0,cuda:1"
        in completed.stdout
    )
    assert "extractor http://small-stage2.test/v1" in completed.stdout


def test_dual_managed_vllm_pools_receive_independent_gpu_layouts(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "output"
    invocation_log = tmp_path / "python_invocations.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${FAKE_PYTHON_INVOCATION_LOG}"
printf '\n' >> "${FAKE_PYTHON_INVOCATION_LOG}"
if [[ "${1:-}" == *detect_all_evidence_hardware.py ]]; then
    printf '4\tcuda:0,cuda:1,cuda:2,cuda:3\t12\t32\t12\tfour eligible GPUs\n'
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_PYTHON_INVOCATION_LOG": str(invocation_log),
            "GPU_COUNT": "4",
            "OCI_PYTHON": str(fake_python),
            "PHYSICAL_GPUS": "",
            "STAGE2_MODEL": "Qwen/Qwen3.8-27B",
            "STAGE2_VLLM_SERVERS": "1",
            "STAGE2_VLLM_GPUS": "cuda:0,cuda:1",
            "STAGE2_VLLM_GPUS_PER_SERVER": "2",
            "STAGE2_VLLM_RAPID_SWITCH_SECONDS": "1200",
            "STAGE2_VLLM_BASE_PORT": "9010",
            "STAGE2_EXTRACTION_MODEL": "LiquidAI/LFM2.5-2.6B",
            "STAGE2_EXTRACTION_VLLM_SERVERS": "2",
            "STAGE2_EXTRACTION_VLLM_GPUS": "cuda:2,cuda:3",
            "STAGE2_EXTRACTION_VLLM_GPUS_PER_SERVER": "1",
            "STAGE2_EXTRACTION_VLLM_BASE_PORT": "9020",
            "STAGE2_EXTRACTION_WORKERS": "64",
            "STAGE2_VLLM_DOWNLOAD_DIR": "",
            "STAGE2_VLLM_EXTRA_ARGS_JSON": "",
            "STAGE2_EXTRACTION_VLLM_DOWNLOAD_DIR": "",
            "STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON": "",
            "STAGE2_WORKERS": "32",
        }
    )
    environment.pop("STAGE2_ENDPOINT", None)
    environment.pop("STAGE2_EXTRACTION_ENDPOINT", None)
    subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "run_synthetic_all_evidence.sh"),
            (
                "synthetic_data/example_synthetic_datasets/"
                "five_confounders_five_effect_modifiers_nsclc_with_structured/"
                "dataset.parquet"
            ),
            "test_output",
            str(output_dir),
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    workflow = invocations[-1]
    assert "--stage2-endpoint" not in workflow
    assert "--stage2-model Qwen/Qwen3.8-27B" in workflow
    assert "--stage2-vllm-servers 1" in workflow
    assert r"--stage2-vllm-gpus cuda:0\,cuda:1" in workflow
    assert "--stage2-vllm-gpus-per-server 2" in workflow
    assert "--stage2-vllm-rapid-switch-seconds 1200" in workflow
    assert "--stage2-vllm-base-port 9010" in workflow
    assert "--stage2-extraction-model LiquidAI/LFM2.5-2.6B" in workflow
    assert "--stage2-extraction-vllm-servers 2" in workflow
    assert r"--stage2-extraction-vllm-gpus cuda:2\,cuda:3" in workflow
    assert "--stage2-extraction-vllm-gpus-per-server 1" in workflow
    assert "--stage2-extraction-vllm-base-port 9020" in workflow
    assert "--stage2-extraction-workers 64" in workflow
    assert "--stage2-extraction-endpoint" not in workflow
