from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("runtime_overrides", [
    {},
    {"STAGE2_WORKERS": "8", "STAGE2_REQUEST_TIMEOUT": "3600",
     "STAGE2_REQUEST_ATTEMPT_TIMEOUT": "1200"},
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
        if runtime_overrides:
            assert "--stage2-workers 8" in invocations[-2]
            assert "stage2.request_attempt_timeout=1200" in workflow
            assert "stage2.request_timeout=3600" in workflow
        elif launcher == "run_one_conf_one_mod.sh":
            assert "--stage2-workers 4" in invocations[-2]
            assert "stage2.request_attempt_timeout=900" in workflow
            assert "stage2.request_timeout=2700" in workflow
        assert "research_all_evidence_workflow" in workflow
        assert "--stage2-endpoint http://127.0.0.1:8010/v1" in workflow
        assert "--stage2-extraction-endpoint http://127.0.0.1:8020/v1" in workflow
        assert "--stage1-only" not in workflow
        assert "Stage 2:        http://127.0.0.1:8010/v1" in completed.stdout
        assert "extractor http://127.0.0.1:8020/v1" in completed.stdout


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
