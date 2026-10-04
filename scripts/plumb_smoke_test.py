#!/usr/bin/env python3
"""Standalone Plumb feasibility experiment; does not change Stage 2 extraction.

Use the isolated environment and downloaded model described in docs/plumb_feasibility.md.
Gold labels are in a separate local fixture and are never passed to the model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/plumb_feasibility/model")
    parser.add_argument("--questions", type=Path, default=ROOT / "experiments/plumb_feasibility/questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/plumb_feasibility/results")
    parser.add_argument("--gpu", default="1", help="Physical GPU index or UUID; visible as cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graphs", action="store_true", help="Capture only the input lengths needed here")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    # Set visibility before importing torch or the official runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "artifacts/plumb_feasibility/triton_cache"))

    import numpy as np
    import torch
    from jevk5 import JevK5, decision_options

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.questions.read_text())
    write_json(args.output / "questions.json", cases)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique")
    print(f"Loading {args.model} on physical GPU {args.gpu}", flush=True)
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise RuntimeError(f"Only {free / 1024**3:.1f} GiB free; need at least 12 GiB")
    load_start = time.perf_counter()
    model = JevK5(str(args.model), device="cuda:0", graphs=False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    print(f"Model loaded in {load_seconds:.1f}s; temperature {model.temperature}", flush=True)
    max_tokens = 0
    for case in cases:
        question = case["question"]
        options = decision_options(question)
        if not 2 <= len(options) <= 16:
            raise ValueError(f"Invalid option count for {case['id']}")
        tokens = len(model.encode(case["state"], question["instructions"], [text for _, text in options]))
        if tokens > 4096:
            raise ValueError("This short-input experiment must not exceed 4096 tokens")
        max_tokens = max(max_tokens, tokens)
    graph_start = time.perf_counter()
    if args.graphs:
        print("Capturing CUDA graph for the required short-input length", flush=True)
        model.capture(lengths=(int(np.ceil(max_tokens / 64)) * 64,))
    graph_seconds = time.perf_counter() - graph_start
    warmup_start = time.perf_counter()
    print("Warming up the model and input shapes", flush=True)
    for _ in range(3):
        model.decide("A blue ball is on the table.", {"type": "noul", "instructions": "Is the ball blue?"})
    # Warm every original shape so per-case compilation is not counted as steady-state latency.
    for case in cases:
        model.decide(case["state"], case["question"])
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start
    print(f"Warm-up complete in {warmup_seconds:.1f}s; starting scored questions", flush=True)
    torch.cuda.reset_peak_memory_stats()
    rows = []

    def run(case: dict, variant: str) -> dict:
        question = case["question"]
        times, answers = [], []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            answer = model.decide(case["state"], question)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            answers.append(answer)
        answer = answers[0]
        if question["type"] == "noul":
            probs = {"true": answer["noul"], "false": 1.0 - answer["noul"]}
            predicted = answer["noul"] >= 0.5
        else:
            probs = answer["probabilities"]
            predicted = max(probs, key=probs.get)
            if question["type"] == "score":
                predicted = int(predicted)
        valid = (
            set(probs) == {key for key, _ in decision_options(question)}
            and all(np.isfinite(p) and 0 <= p <= 1 for p in probs.values())
            and abs(sum(probs.values()) - 1) < 1e-5
            and abs(answer["confidence"] - max(probs.values())) < 1e-5
        )
        if question["type"] == "score":
            valid = valid and abs(answer["score"] - sum(int(k) * v for k, v in probs.items())) < 1e-5
        if not valid:
            raise ValueError(f"Invalid distribution for {case['id']}: {answer}")
        row = {
            **case, "variant": variant, "answer": answer, "probabilities": probs,
            "predicted": predicted, "correct_argmax": predicted == case["expected"],
            "valid_distribution": valid, "repeat_answers_identical": all(a == answer for a in answers),
            "latency_ms": times, "median_latency_ms": statistics.median(times),
        }
        if question["type"] == "score":
            row["expected_level_absolute_error"] = abs(answer["score"] - case["expected"])
            # Readout sensitivity only: same distribution retempered to the model card's v5.2 T.
            adjusted = np.array(list(probs.values()), dtype=np.float64) ** (model.temperature / 1.2)
            adjusted /= adjusted.sum()
            row["score_temperature_1_2"] = {
                "probabilities": dict(zip(probs, adjusted.tolist(), strict=True)),
                "score": sum(int(k) * p for k, p in zip(probs, adjusted, strict=True)),
            }
        print(json.dumps({"id": case["id"], "variant": variant, "expected": case["expected"],
                          "predicted": predicted, "answer": answer,
                          "median_latency_ms": row["median_latency_ms"]}), flush=True)
        return row

    with (args.output / "results.jsonl").open("w") as handle:
        for case in cases:
            row = run(case, "original")
            rows.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
        for original in cases:
            if original["question"]["type"] != "choice":
                continue
            case = copy.deepcopy(original)
            crit = case["question"]["criteria"]
            case["question"]["criteria"] = list(reversed(crit)) if isinstance(crit, list) else dict(reversed(list(crit.items())))
            model.decide(case["state"], case["question"])
            row = run(case, "reversed_choice_options")
            rows.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
    originals = [row for row in rows if row["variant"] == "original"]
    reversals = [row for row in rows if row["variant"] != "original"]
    original_by_id = {row["id"]: row for row in originals}
    score_rows = [row for row in originals if row["question"]["type"] == "score"]
    latencies = [latency for row in originals for latency in row["latency_ms"]]
    packages = {}
    for name in ("torch", "transformers", "jevk5", "accelerate", "huggingface-hub", "flash-linear-attention", "triton", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    revision_file = args.model.parent / "model_revision.json"
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": json.loads(revision_file.read_text()) if revision_file.exists() else str(args.model),
        "runtime_revision": "85238d7be5527370c43206fe54cd752eb3134c1b",
        "runtime_source_sha256": hashlib.sha256(Path(inspect.getfile(JevK5)).read_bytes()).hexdigest(),
        "python": platform.python_version(), "python_executable": sys.executable,
        "python_path": sys.path, "packages": packages,
        "gpu": {"physical_selector": args.gpu, "name": torch.cuda.get_device_name(),
                "cuda_runtime": torch.version.cuda, "free_before_load_gib": free / 1024**3,
                "total_gib": total / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3},
        "settings": {"dtype": "bfloat16", "temperature": model.temperature, "noul_commit": False,
                     "graphs": args.graphs, "captured_lengths": list(model.graphs),
                     "batch_size": 1, "repeats": args.repeats, "generated_tokens": 0},
        "fixture_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "load_seconds": load_seconds, "graph_capture_seconds": graph_seconds,
        "warmup_seconds": warmup_seconds,
        "correct_argmax": sum(row["correct_argmax"] for row in originals), "n": len(originals),
        "by_type": {kind: {"n": sum(row["question"]["type"] == kind for row in originals),
                           "correct_argmax": sum(row["correct_argmax"] for row in originals if row["question"]["type"] == kind)}
                    for kind in ("choice", "noul", "score")},
        "all_distributions_valid": all(row["valid_distribution"] for row in rows),
        "all_repeats_identical": all(row["repeat_answers_identical"] for row in rows),
        "score_expected_level_mae": statistics.mean(row["expected_level_absolute_error"] for row in score_rows),
        "score_expected_level_mae_at_temperature_1_2": statistics.mean(abs(row["score_temperature_1_2"]["score"] - row["expected"]) for row in score_rows),
        "choice_reversal": {"n": len(reversals), "correct_argmax": sum(row["correct_argmax"] for row in reversals),
                            "changed_predictions": sum(row["predicted"] != original_by_id[row["id"]]["predicted"] for row in reversals)},
        "latency_ms": {"p50": float(np.percentile(latencies, 50)), "p95": float(np.percentile(latencies, 95)),
                       "min": min(latencies), "max": max(latencies)},
        "input_tokens": {"min": min(row["answer"]["input_tokens"] for row in originals),
                         "max": max(row["answer"]["input_tokens"] for row in originals)},
        "limitations": ["Hand-written small smoke test, not a random sample or clinical validation.",
                        "GPU shared with an existing vLLM worker; timing is not an isolated benchmark.",
                        "Scores are expected zero-based ordinal indices, not extracted continuous numbers.",
                        "No Stage 2 integration, real patient data, or generative baseline comparison."],
    }
    write_json(args.output / "summary.json", summary)
    # Include libraries inherited read-only through a .pth file, with the same
    # first-on-sys.path precedence used by Python imports.
    installed = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            installed.setdefault(name.lower().replace("_", "-"), dist.version)
    (args.output / "environment.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(installed.items()))
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
