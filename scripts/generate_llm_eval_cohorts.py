#!/usr/bin/env python3
"""Materialize a stratified held-out suite of realistic full-LLM longitudinal cohorts.

Selects held-out designs spanning seen cancers + all unseen cancers and low/medium/high
candidate widths, renders every patient's dated notes through an OpenAI-compatible vLLM
endpoint (default: Nemotron on sn4622130540:8000) using ``--realistic-notes``, and skips
already-completed cohorts on re-run.
"""

import argparse
import os
from pathlib import Path
import time

from synthetic_data.config import LLMConfig
from synthetic_data.longitudinal import generate_longitudinal_cohort
from synthetic_data.paraphrase_bank import DEFAULT_API_URL, DEFAULT_MODEL
from synthetic_data.study_designs import DEFAULT_BANK, load_bank


def select_stratified_held_out(bank: dict, count: int) -> list[dict]:
    held = [d for d in bank["designs"] if d["question"]["split"] == "held_out"]
    train_cancers = {
        d["question"]["cancer"] for d in bank["designs"] if d["question"]["split"] == "train"
    }
    unseen = [d for d in held if d["question"]["cancer"] not in train_cancers]
    seen = [d for d in held if d["question"]["cancer"] in train_cancers]

    def width(d):
        return len(d["candidates"]["measurements"]) + len(d["candidates"]["note_findings"])

    def pick_across_widths(pool, k):
        if not pool or k <= 0:
            return []
        ordered = sorted(pool, key=lambda d: (width(d), d["id"]))
        if len(ordered) <= k:
            return ordered
        step = (len(ordered) - 1) / max(k - 1, 1)
        chosen = []
        used_ids = set()
        for i in range(k):
            cand = ordered[round(i * step)]
            if cand["id"] not in used_ids:
                chosen.append(cand)
                used_ids.add(cand["id"])
        for cand in ordered:
            if len(chosen) >= k:
                break
            if cand["id"] not in used_ids:
                chosen.append(cand)
                used_ids.add(cand["id"])
        return chosen

    n_unseen = min(len(unseen), count // 2)
    n_seen = count - n_unseen
    return pick_across_widths(unseen, n_unseen) + pick_across_widths(seen, n_seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/llm_eval_cohorts"),
    )
    parser.add_argument("--num-designs", type=int, default=16)
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--omit-absent-prob", type=float, default=0.25)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    selected = select_stratified_held_out(bank, args.num_designs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    llm_cfg = LLMConfig(
        api_base_url=args.api_url,
        model_name=args.model,
        max_tokens=args.max_tokens,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "",
        reject_incomplete=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 40},
    )
    print(
        f"Generating {len(selected)} realistic full-LLM held-out cohorts "
        f"(N={args.dataset_size}, workers={args.workers}, model={args.model})",
        flush=True,
    )
    t0 = time.monotonic()
    for idx, design in enumerate(selected, start=1):
        out_dir = args.output_root / design["id"]
        if (out_dir / "dataset.parquet").is_file():
            print(f"[{idx}/{len(selected)}] Skipping completed {design['id']}", flush=True)
            continue
        t_one = time.monotonic()
        w = len(design["candidates"]["measurements"]) + len(design["candidates"]["note_findings"])
        print(
            f"[{idx}/{len(selected)}] Rendering {design['id']} "
            f"(cancer={design['question']['cancer']}, width={w}, N={args.dataset_size})...",
            flush=True,
        )
        generate_longitudinal_cohort(
            out_dir,
            dataset_size=args.dataset_size,
            seed=args.seed + 2 * idx + 1,
            dgp_seed=args.seed + 1000 + 2 * idx + 1,
            llm_config=llm_cfg,
            num_workers=args.workers,
            prior="design",
            design=design,
            omit_absent_prob=args.omit_absent_prob,
            realistic_notes=True,
        )
        print(
            f"[{idx}/{len(selected)}] Finished {design['id']} in "
            f"{(time.monotonic() - t_one) / 60.0:.1f}m "
            f"(total {(time.monotonic() - t0) / 60.0:.1f}m)",
            flush=True,
        )


if __name__ == "__main__":
    main()
