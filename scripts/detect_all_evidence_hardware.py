#!/usr/bin/env python3
"""Select CUDA devices and conservative workflow concurrency from live hardware."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

DEFAULT_STAGE2_WORKERS = 32


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    free_gib: float
    total_gib: float


def _positive_int_or_auto(value: str) -> str:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected 'auto' or a positive integer")
    return str(parsed)


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        return max(1, process_cpu_count() or 1)
    return max(1, os.cpu_count() or 1)


def _visible_gpus() -> list[GPU]:
    import torch

    output: list[GPU] = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        properties = torch.cuda.get_device_properties(index)
        output.append(
            GPU(
                index=index,
                name=str(properties.name).replace("\t", " ").replace("\n", " "),
                free_gib=free_bytes / (1024**3),
                total_gib=total_bytes / (1024**3),
            )
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-count",
        type=_positive_int_or_auto,
        default="auto",
        help="number of eligible GPUs to select, or auto for all of them",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int_or_auto,
        default="auto",
        help="Stage 1 CPU workers, or auto to size from CPUs, folds, and GPU memory",
    )
    parser.add_argument(
        "--stage2-workers",
        type=_positive_int_or_auto,
        default=str(DEFAULT_STAGE2_WORKERS),
        help=(
            f"Stage 2 request concurrency (default: {DEFAULT_STAGE2_WORKERS}); "
            "auto caps it at the CPU count"
        ),
    )
    parser.add_argument("--outer-folds", type=int, required=True, help="outer folds")
    parser.add_argument("--inner-folds", type=int, required=True, help="inner folds")
    parser.add_argument(
        "--min-free-vram-gib",
        type=float,
        default=20.0,
        help="minimum free memory for a GPU to be eligible",
    )
    parser.add_argument(
        "--stage1-vram-gib-per-worker",
        type=float,
        default=8.0,
        help="GPU memory assumed per concurrent Stage 1 context",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.outer_folds < 1 or args.inner_folds < 1:
        raise SystemExit("fold counts must be positive")
    if args.min_free_vram_gib < 0:
        raise SystemExit("--min-free-vram-gib must be nonnegative")
    if args.stage1_vram_gib_per_worker <= 0:
        raise SystemExit("VRAM-per-worker estimates must be positive")

    cpu_count = _available_cpu_count()
    context_capacity = args.outer_folds * (args.inner_folds + 1)
    visible = _visible_gpus()
    if not visible:
        raise SystemExit("no CUDA GPUs are visible to PyTorch")
    eligible = [gpu for gpu in visible if gpu.free_gib >= args.min_free_vram_gib]
    if not eligible:
        details = "; ".join(
            f"cuda:{gpu.index} {gpu.free_gib:.1f}/{gpu.total_gib:.1f} GiB free" for gpu in visible
        )
        raise SystemExit(
            f"no visible GPU has the required {args.min_free_vram_gib:.1f} GiB free; {details}"
        )

    if args.gpu_count == "auto":
        selected = eligible
    else:
        requested = int(args.gpu_count)
        if requested > len(eligible):
            raise SystemExit(
                f"GPU_COUNT={requested} requested, but only {len(eligible)} visible GPU(s) "
                f"have at least {args.min_free_vram_gib:.1f} GiB free"
            )
        selected = sorted(eligible, key=lambda gpu: (-gpu.free_gib, gpu.index))[:requested]
        selected.sort(key=lambda gpu: gpu.index)

    gpu_worker_capacity = sum(
        max(1, min(args.inner_folds, int(gpu.free_gib // args.stage1_vram_gib_per_worker)))
        for gpu in selected
    )
    automatic_workers = min(cpu_count, max(context_capacity, gpu_worker_capacity))
    worker_count = automatic_workers if args.workers == "auto" else int(args.workers)

    # Stage 2 workers are primarily blocked on endpoint I/O. vLLM performs its
    # own memory-aware continuous batching, so request concurrency should not
    # be inferred from a one-process-per-VRAM estimate.
    automatic_stage2_workers = min(cpu_count, DEFAULT_STAGE2_WORKERS)
    stage2_workers = (
        automatic_stage2_workers if args.stage2_workers == "auto" else int(args.stage2_workers)
    )

    devices = ",".join(f"cuda:{gpu.index}" for gpu in selected)
    summary = "; ".join(
        f"cuda:{gpu.index} {gpu.name} {gpu.free_gib:.1f}/{gpu.total_gib:.1f} GiB free"
        for gpu in selected
    )
    print(
        "\t".join(
            [
                str(len(selected)),
                devices,
                str(worker_count),
                str(stage2_workers),
                str(cpu_count),
                summary,
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
