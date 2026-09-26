#!/usr/bin/env python3
"""Run fresh Stage 2 against preserved Stage 1 evidence and split provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    plain_stage2_config_from_mapping,
)
from oci.inference.plain_handoff_stage2_analysis import prompt_token_count
from oci.inference.stage2_concurrency import LiveRequestLimits


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    raw = json.loads(args.config.read_text())
    directory = Path(raw["report_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    def status(phase, **fields):
        write(directory / "status.json", {"at": datetime.now(timezone.utc).isoformat(),
              "pid": os.getpid(), "phase": phase, **fields})
    try:
        status("preflight", output_dir=raw["output_dir"])
        config = plain_stage2_config_from_mapping(raw["stage2"], default_workers=8)
        runner = PlainHandoffStage2(config=config, clinical_question=raw["clinical_question"])
        if raw.get("concurrency_control_path"):
            extraction_workers = config.extraction_llm.workers if config.extraction_llm else config.workers
            limits = LiveRequestLimits(Path(raw["concurrency_control_path"]), ceilings={
                "interpretation": config.workers, "extraction": extraction_workers,
                "total": config.workers + extraction_workers,
            })
            runner.completion.admission_gate = lambda: limits.slot("interpretation")
            if runner.extraction_completion is not None:
                runner.extraction_completion.admission_gate = lambda: limits.slot("extraction")
            if runner.selection_consolidation_completion is not None:
                runner.selection_consolidation_completion.admission_gate = lambda: limits.slot("interpretation")
        if raw.get("tokenizer_cache_dir"):
            runner.extraction_tokenizer.cache_dir = raw["tokenizer_cache_dir"]
        tokenizer_tokens = prompt_token_count(runner.extraction_tokenizer,
            [{"role": "user", "content": "Token budget check."}])
        columns = [raw[key] for key in ("unit_id_column", "text_column", "treatment_column", "outcome_column")]
        dataset = pd.read_parquet(raw["dataset"], columns=columns)
        sources = {}
        for key in ("dataset", "handoff_path", "split_provenance_path"):
            source = Path(raw[key])
            stat = source.stat()
            identity = {"path": str(source), "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns}
            if key == "handoff_path":
                # Stage 2 hashes the full handoff before checking its compilation
                # cache. Avoid another complete read during launcher preflight.
                identity["sha256_recorded_in"] = str(
                    Path(raw["output_dir"]) / "evidence_compilation" / "compile_complete.json")
            else:
                identity["sha256"] = sha256(source)
            sources[key] = identity
        manifest = {"config_sha256": sha256(args.config), "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "sources": sources,
            "rows": len(dataset), "loaded_columns": columns, "tokenizer_smoke_prompt_tokens": tokenizer_tokens,
            "model_identity": runner.model_identity}
        write(directory / "preflight.json", manifest)
        if args.preflight:
            status("preflight_complete", output_dir=raw["output_dir"])
            print(json.dumps({"status": "preflight_complete", "rows": len(dataset),
                  "tokenizer_tokens": tokenizer_tokens, "model_identity": manifest["model_identity"]}), flush=True)
            return 0
        logging.info("Preflight complete; Stage 2 will fingerprint the handoff and load or compile evidence cards")
        status("running", output_dir=raw["output_dir"], git_commit=manifest["git_commit"])
        result = runner.run(handoff_path=Path(raw["handoff_path"]), output_dir=Path(raw["output_dir"]),
            dataset=dataset, split_provenance_path=Path(raw["split_provenance_path"]),
            **{key: raw[key] for key in (
                "unit_id_column", "text_column", "treatment_column", "outcome_column", "outcome_type", "inner_folds", "seed")})
        write(directory / "result.json", dict(result))
        status("complete", output_dir=raw["output_dir"])
        return 0
    except BaseException as exc:
        status("failed", error_type=type(exc).__name__, error=str(exc))
        logging.exception("Stage 2 run failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
