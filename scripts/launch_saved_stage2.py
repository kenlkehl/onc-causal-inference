#!/usr/bin/env python3
"""Configuration-preserving adapter for the two synthetic cohort wrappers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oci.inference import research_all_evidence_workflow as workflow
from oci.inference.stage2_preflight import preflight_stage2, restore_stage2_archive


def command(dataset: Path, output: str, env: dict[str, str]) -> list[str]:
    config_name = env.get("OCI_RUN_CONFIG")
    if not config_name:
        if not output:
            raise ValueError(
                "saved Stage 2 launch requires OCI_RUN_CONFIG or an existing output directory"
            )
        config_name = str(Path(output) / "run_config.json")
    config_path = Path(config_name).expanduser().resolve(strict=True)
    args = ["--config", str(config_path), "--stage2-only"]
    if env.get("STAGE2_RESELECT", "0") == "1":
        args.append("--stage2-reselect")
    mode = env.get("STAGE2_SELECTION_MODE", "")
    if mode:
        args += ["--set", "stage2.statistical_selection.selection_mode=" + mode]
    # Saved science wins. Runtime changes are explicit and still go through the
    # core config and model-identity checks. All other overrides fail visibly.
    runtime = {
        "STAGE2_ENDPOINT": "--stage2-endpoint",
        "STAGE2_MODEL": "--stage2-model",
        "STAGE2_EXTRACTION_ENDPOINT": "--stage2-extraction-endpoint",
        "STAGE2_EXTRACTION_MODEL": "--stage2-extraction-model",
        "STAGE2_EXTRACTION_WORKERS": "--stage2-extraction-workers",
    }
    for prefix, flag in (
        ("STAGE2_VLLM_", "--stage2-vllm-"),
        ("STAGE2_EXTRACTION_VLLM_", "--stage2-extraction-vllm-"),
    ):
        for suffix in (
            "SERVERS",
            "GPUS",
            "GPUS_PER_SERVER",
            "BASE_PORT",
            "INTERNAL_PORT_BASE",
            "DOWNLOAD_DIR",
        ):
            runtime[prefix + suffix] = flag + suffix.lower().replace("_", "-")
    settings = {
        "STAGE2_WORKERS": "stage2.workers",
        "STAGE2_REQUEST_TIMEOUT": "stage2.request_timeout",
        "STAGE2_REQUEST_ATTEMPT_TIMEOUT": "stage2.request_attempt_timeout",
    }
    controls = {"STAGE2_ONLY", "STAGE2_RESELECT", "STAGE2_SELECTION_MODE"}
    for name, value in env.items():
        if name in runtime:
            args += [runtime[name], value]
        elif name in settings:
            args += ["--set", settings[name] + "=" + value]
        elif name.startswith("STAGE2_") and name not in controls:
            raise ValueError(
                f"{name} is not a supported saved-run runtime override; edit OCI_RUN_CONFIG explicitly"
            )
    if "DISABLE_HTR" in env or "STAGE1_ARCHITECTURES" in env:
        raise ValueError(
            "saved-run Stage 1 settings come from OCI_RUN_CONFIG; omit DISABLE_HTR/STAGE1_ARCHITECTURES"
        )
    parsed = workflow.build_parser().parse_args(args)
    raw, directory = workflow._raw_config_from_args(parsed)
    config = workflow.compile_config(raw, config_dir=directory)
    if config.dataset != dataset.resolve():
        raise ValueError(
            "saved cohort dataset does not match this wrapper; retain the original absolute dataset path"
        )
    if output and config.output_dir != Path(output).expanduser().resolve():
        raise ValueError("requested output directory does not match OCI_RUN_CONFIG")
    return args


def main() -> int:
    try:
        args = command(Path(sys.argv[1]), sys.argv[2], dict(os.environ))
        parsed = workflow.build_parser().parse_args(args)
        raw, directory = workflow._raw_config_from_args(parsed)
        config = workflow.compile_config(raw, config_dir=directory)
        source = os.environ.get("OCI_STAGE2_SOURCE")
        if source and not parsed.stage2_reselect:
            raise ValueError("OCI_STAGE2_SOURCE requires STAGE2_RESELECT=1")
        if source and (config.output_dir / "stage2").exists():
            raise ValueError(
                "stage2 already exists; omit OCI_STAGE2_SOURCE and resume the active run"
            )
        result = preflight_stage2(
            config,
            reselect=parsed.stage2_reselect,
            source_dir=Path(source).resolve(strict=True) if source else None,
        )
        print(json.dumps(result, indent=2), flush=True)
        if os.environ.get("OCI_PREFLIGHT_ONLY") == "1":
            return 0
        if source:
            restore_stage2_archive(config, Path(source))
        return workflow.main(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Stage 2 launch refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
