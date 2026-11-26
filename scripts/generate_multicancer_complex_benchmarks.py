#!/usr/bin/env python3
"""Resumable multi-cancer Nemotron-120B benchmark corpus generator (strictly held-out).

Generates two complementary multi-disease oncology evaluation suites (never used
for training or fine-tuning) against a vLLM Nemotron-3.5 endpoint using NVIDIA's
recommended sampling profile (temp=1.0, top_p=0.95, top_k=0, min_p=0.0,
enable_thinking=False):

1. ``part1_longitudinal``: Six held-out wide (P=99-100) oncology cohorts from
   ``nemotron_5000.json.gz`` across multiple myeloma, malignant glioma, bladder,
   endometrial, breast, and cervical cancer (completing all 8 held-out cancer
   sites at P=99-100 alongside existing HCC and gastric cohorts), then
   re-equating each into ``artifacts/llm_eval_cohorts_complex``.
2. ``part2_wide120``: Four 120-feature full 2-stage LLM clinical prose corpora
   (N=1,000 each) across Multiple Myeloma, HCC, mCRC, and ER+ Breast cancer for
   end-to-end text benchmarking of S7-S12b (5-50 confounders and 5-50 modifiers).

Every note, request, and cohort is atomically checkpointed; rerunning skips
completed cohorts and resumes in-progress ones cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic_data.config import LLMConfig, SyntheticDataConfig
from synthetic_data.generator import generate_synthetic_dataset
from synthetic_data.llm_client import is_nemotron_3_5, nemotron_3_5_extra_body
from synthetic_data.longitudinal import generate_longitudinal_cohort
from synthetic_data.paraphrase_bank import DEFAULT_API_URL, DEFAULT_MODEL
from synthetic_data.reequation import reequation_cohort
from synthetic_data.study_designs import DEFAULT_BANK, load_bank
from synthetic_data.wide_corpus import ONCOLOGY_CONTEXT_QUESTIONS, generate_wide_source_cohort

WIDE_HELDOUT_DESIGN_IDS = (
    "myeloma_q03_v043",      # P=99, multiple myeloma
    "glioma_q05_v028",       # P=99, malignant glioma
    "bladder_q03_v027",      # P=100, bladder cancer
    "endometrium_q05_v031",  # P=100, endometrial cancer
    "breast_q02_v039",       # P=100, breast cancer
    "cervix_q04_v029",       # P=100, cervical cancer
)

TWO_STAGE_WIDE_CONTEXTS = ("myeloma", "hcc", "mcrc", "breast_er_pos")


def _llm_config(api_url: str, model: str, *, max_tokens: int, timeout: float) -> LLMConfig:
    if is_nemotron_3_5(model):
        temp = 1.0
        extra = nemotron_3_5_extra_body(enable_thinking=False)
    else:
        temp = 0.7
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
    return LLMConfig(
        api_base_url=api_url,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "",
        model_name=model,
        temperature=temp,
        max_tokens=max_tokens,
        timeout=timeout,
        reject_incomplete=True,
        extra_body=extra,
    )


def run_part1_longitudinal(
    *,
    api_url: str,
    model: str,
    output_root: Path,
    complex_root: Path,
    dataset_size: int,
    workers: int,
    timeout: float,
    seed: int,
) -> None:
    from scripts.materialize_complex_llm_eval_cohorts import build_scenarios

    bank = load_bank(DEFAULT_BANK)
    by_id = {d["id"]: d for d in bank["designs"]}
    llm_cfg = _llm_config(api_url, model, max_tokens=1536, timeout=timeout)
    output_root.mkdir(parents=True, exist_ok=True)

    for idx, design_id in enumerate(WIDE_HELDOUT_DESIGN_IDS, start=1):
        design = by_id[design_id]
        assert design["question"]["split"] == "held_out", f"{design_id} is not held_out!"
        out_dir = output_root / design_id
        w = len(design["candidates"]["measurements"]) + len(design["candidates"]["note_findings"])
        if (out_dir / "dataset.parquet").is_file():
            print(f"[Part 1 {idx}/{len(WIDE_HELDOUT_DESIGN_IDS)}] Already complete: {design_id} (P={w})", flush=True)
        else:
            t0 = time.monotonic()
            print(
                f"[Part 1 {idx}/{len(WIDE_HELDOUT_DESIGN_IDS)}] Generating {design_id} "
                f"(cancer={design['question']['cancer']}, P={w}, N={dataset_size}, workers={workers})...",
                flush=True,
            )
            generate_longitudinal_cohort(
                out_dir,
                dataset_size=dataset_size,
                seed=seed + 100 + 2 * idx,
                dgp_seed=seed + 2100 + 2 * idx,
                llm_config=llm_cfg,
                num_workers=workers,
                prior="design",
                design=design,
                omit_absent_prob=0.25,
                realistic_notes=True,
            )
            print(
                f"[Part 1 {idx}/{len(WIDE_HELDOUT_DESIGN_IDS)}] Finished {design_id} "
                f"in {(time.monotonic() - t0) / 60.0:.1f}m",
                flush=True,
            )

        # Immediately re-equate into complex benchmark suite
        meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        num_features = len(meta.get("features") or [])
        for s_idx, (tag, scen) in enumerate(build_scenarios(num_features)):
            target_dir = complex_root / tag / design_id
            if (target_dir / "dataset.parquet").is_file():
                continue
            scen_seed = 101 + s_idx * 17 + (sum(ord(ch) for ch in design_id) % 97)
            reequation_cohort(out_dir, scen, size=None, seed=scen_seed, output_dir=target_dir)
            print(f"  -> Re-equated {tag}/{design_id}", flush=True)


def run_part2_wide120(
    *,
    api_url: str,
    model: str,
    corpora_root: Path,
    dataset_size: int,
    workers: int,
    timeout: float,
    seed: int,
) -> None:
    corpora_root.mkdir(parents=True, exist_ok=True)
    llm_cfg = _llm_config(api_url, model, max_tokens=8192, timeout=timeout)

    for idx, context in enumerate(TWO_STAGE_WIDE_CONTEXTS, start=1):
        seed_dir = corpora_root / f"wide_120_{context}_dgp_seed"
        out_dir = corpora_root / f"wide_120_llm_{context}_n{dataset_size}"
        if (out_dir / "dataset.parquet").is_file():
            meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            if meta.get("generation_status") == "complete":
                print(f"[Part 2 {idx}/{len(TWO_STAGE_WIDE_CONTEXTS)}] Already complete: {context}", flush=True)
                continue

        if not (seed_dir / "metadata.json").is_file():
            generate_wide_source_cohort(
                seed_dir,
                num_patients=20,
                num_features=120,
                clinical_context=context,
                clinical_question=ONCOLOGY_CONTEXT_QUESTIONS[context],
                seed=seed + idx,
                render_text=False,
                structured_data_enabled=True,
                min_events_per_patient=15,
                max_events_per_patient=22,
            )

        cfg = SyntheticDataConfig.from_json(str(seed_dir / "generation_config.json"))
        cfg.dataset_size = dataset_size
        cfg.seed = seed + idx
        cfg.output_dir = str(out_dir)
        cfg.llm = llm_cfg

        t0 = time.monotonic()
        print(
            f"[Part 2 {idx}/{len(TWO_STAGE_WIDE_CONTEXTS)}] Generating 120-feature 2-stage LLM corpus "
            f"for {context} (N={dataset_size}, workers={workers}) -> {out_dir}",
            flush=True,
        )
        generate_synthetic_dataset(
            config=cfg,
            num_workers=workers,
            show_progress=False,
            dgp_metadata_path=str(seed_dir / "metadata.json"),
        )
        print(
            f"[Part 2 {idx}/{len(TWO_STAGE_WIDE_CONTEXTS)}] Finished {context} "
            f"in {(time.monotonic() - t0) / 60.0:.1f}m",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--parts",
        choices=("both", "part1", "part2"),
        default="both",
        help="Run Part 1 (P=99-100 longitudinal), Part 2 (P=120 2-stage), or both (default: both)",
    )
    parser.add_argument(
        "--longitudinal-out",
        type=Path,
        default=Path("artifacts/llm_eval_cohorts"),
    )
    parser.add_argument(
        "--complex-out",
        type=Path,
        default=Path("artifacts/llm_eval_cohorts_complex"),
    )
    parser.add_argument(
        "--corpora-out",
        type=Path,
        default=Path("/data1/ken/pcori_dev/oci_bench_complex/corpora"),
    )
    args = parser.parse_args()

    if args.parts in ("both", "part1"):
        run_part1_longitudinal(
            api_url=args.api_url,
            model=args.model,
            output_root=args.longitudinal_out,
            complex_root=args.complex_out,
            dataset_size=args.dataset_size,
            workers=args.workers,
            timeout=args.timeout,
            seed=args.seed,
        )
    if args.parts in ("both", "part2"):
        run_part2_wide120(
            api_url=args.api_url,
            model=args.model,
            corpora_root=args.corpora_out,
            dataset_size=args.dataset_size,
            workers=args.workers,
            timeout=args.timeout,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
