"""Train, resume, evaluate, or query a longitudinal causal study model."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from synthetic_data.study_designs import load_bank

from .events import binary_observations, read_patient_table
from .model import BASE_MODEL, ModelConfig
from .training import (
    OBSERVED,
    _autocast,
    evaluate,
    evaluate_cohorts,
    load_run,
    load_run_design_bank,
    run_training,
)

OVERLAP_MARGIN = 0.05


def _chunks(table, size):
    for start in range(0, table.num_rows, size):
        yield table.slice(start, size).to_pylist()


def _unique_ids(table):
    ids = table["patient_id"].to_pylist()
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("Patient IDs must be present and unique within each cohort")
    return set(ids)


@torch.inference_mode()
def predict(
    model,
    context_file,
    query_file,
    output,
    *,
    batch_size=16,
    context_size=None,
    seed=42,
    precision="auto",
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    context = read_patient_table(context_file, model.patient.streams, OBSERVED)
    context_ids = _unique_ids(context)
    if query_file is None:
        if context_size is None or not 0 < context_size < len(context):
            raise ValueError("For a single cohort, set context_size between 1 and N-1")
        chosen = np.zeros(len(context), dtype=bool)
        chosen[np.random.default_rng(seed).choice(len(chosen), context_size, replace=False)] = True
        queries = context.drop(OBSERVED).filter(~chosen)
        context = context.filter(chosen)
    else:
        queries = read_patient_table(query_file, model.patient.streams)
        if context_size is not None:
            raise ValueError("Explicit context and query files already define the split")
        if context_ids & _unique_ids(queries):
            raise ValueError("Context and query patient IDs must be disjoint")
    if not len(context) or not len(queries):
        raise ValueError("Need at least one context and one query patient")
    model.eval()
    device = model.centers.device
    with _autocast(device, precision):
        embeddings = torch.cat(
            [model.patient(rows) for rows in _chunks(context.drop(OBSERVED), batch_size)]
        )
        t, y = binary_observations(context.select(OBSERVED).to_pylist(), device)
        memories, scale = model.condition(embeddings, t, y)
    n0, n1 = int(scale.counts[0].item()), int(scale.counts[1].item())
    if min(n0, n1) < 5:
        print(
            f"Warning: sparse arm support in context (control={n0}, treated={n1}); "
            "counterfactual estimates may rely heavily on prior shrinkage.",
            file=sys.stderr,
        )

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Use query schema for patient_id so separate context/query files with differing ID types work.
    schema = pa.schema(
        [
            queries.schema.field("patient_id"),
            ("risk_control", pa.float32()),
            ("risk_treated", pa.float32()),
            ("risk_difference", pa.float32()),
            ("risk_control_sd", pa.float32()),
            ("risk_treated_sd", pa.float32()),
            ("propensity_score", pa.float32()),
            ("outside_overlap", pa.bool_()),
        ]
    )
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / "predictions.parquet"
        with pq.ParquetWriter(temporary, schema) as writer:
            for rows in _chunks(queries, batch_size):
                with _autocast(device, precision):
                    q_emb = model.patient(rows)
                    logits = model.query_logits(q_emb, memories, scale)
                    prop_logits = model.query_propensity_logits(q_emb, memories, scale)
                mean, sd = model.risk_stats(logits, scale)
                p0, p1 = mean.cpu().unbind(dim=1)
                s0, s1 = sd.cpu().unbind(dim=1)
                prop = prop_logits.float().sigmoid().cpu()
                outside = ((prop < OVERLAP_MARGIN) | (prop > 1.0 - OVERLAP_MARGIN)).numpy()
                writer.write_table(
                    pa.Table.from_arrays(
                        [
                            [r["patient_id"] for r in rows],
                            p0.numpy(),
                            p1.numpy(),
                            (p1 - p0).numpy(),
                            s0.numpy(),
                            s1.numpy(),
                            prop.numpy(),
                            outside,
                        ],
                        schema=schema,
                    )
                )
        # Claim destination exclusively without hard-linking (portable on network/FUSE mounts).
        fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        try:
            os.replace(temporary, output)
        except Exception:
            output.unlink(missing_ok=True)
            raise
    return len(queries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser(
        "train", help="Train on newly simulated studies or supplied synthetic cohorts."
    )
    resume = sub.add_parser(
        "resume", help="Continue an existing training run to a total step count."
    )
    scoring = sub.add_parser(
        "evaluate", help="Evaluate against oracle risks from held-out DGPs or materialized cohorts."
    )
    inference = sub.add_parser("predict", help="Estimate both risks for unlabeled query patients.")
    for p in (train, resume, scoring, inference):
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for p in (train, scoring, inference):
        p.add_argument(
            "--precision",
            choices=("auto", "fp32", "bf16"),
            default="auto",
            help="Mixed precision mode (auto uses bf16 on supported CUDA devices, fp32 on CPU).",
        )
    for p in (train, resume):
        p.add_argument(
            "--steps", type=int, required=True, help="Total optimizer steps."
        )
        p.add_argument("--save-every", type=int, default=50)
    for p in (train, scoring):
        p.add_argument("--context-size", type=int, default=64)
        p.add_argument("--query-size", type=int, default=16)
    for p in (train, scoring, inference):
        p.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--min-context-size",
        type=int,
        default=None,
        help="Optional lower bound to sample variable context sizes up to --context-size.",
    )
    train.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of independent studies accumulated per optimizer update.",
    )
    train.add_argument(
        "--note-model", default=BASE_MODEL, help="Released base model ID or a local snapshot."
    )
    train.add_argument("--note-lr", type=float, default=1e-5)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--backbone-weight-decay", type=float, default=0.0)
    train.add_argument("--warmup-steps", type=int, default=0)
    train.add_argument(
        "--lr-scheduler",
        choices=("constant", "linear", "cosine"),
        default="constant",
        help="Learning rate schedule over --steps.",
    )
    train.add_argument(
        "--augment-notes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Vary note section framing and positive-only charting on design-bank training episodes.",
    )
    train.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate on held-out studies every N steps and save best_checkpoint.pt (0 disables).",
    )
    train.add_argument("--eval-studies", type=int, default=16)
    prior = train.add_mutually_exclusive_group()
    prior.add_argument(
        "--design-bank", type=Path, help="LLM-designed study prior (default: bundled bank)."
    )
    prior.add_argument(
        "--mixed-prior", action="store_true", help="Use the original six-variable study prior."
    )
    prior.add_argument(
        "--train-cohort",
        type=Path,
        action="append",
        default=[],
        help="Repeat for materialized study Parquets with oracle risk labels.",
    )
    defaults = ModelConfig()
    for key, value in vars(defaults).items():
        arg_type = float if isinstance(value, float) else int
        train.add_argument("--" + key.replace("_", "-"), type=arg_type, default=value)
    scoring.add_argument("--studies", type=int, default=16)
    scoring.add_argument(
        "--eval-cohort",
        type=Path,
        action="append",
        default=[],
        help="Evaluate on materialized Parquet cohort(s) with oracle risk columns.",
    )
    scoring.add_argument(
        "--design-bank",
        type=Path,
        default=None,
        help="Override the checkpoint's design bank during evaluation.",
    )
    scoring.add_argument(
        "--note-batch-size",
        type=int,
        default=None,
        help="Override the checkpoint's note microbatch size for evaluation.",
    )
    inference.add_argument("--context", type=Path, required=True)
    inference.add_argument("--queries", type=Path, help="Separate Parquet of query histories.")
    inference.add_argument(
        "--context-size", type=int, help="Split a single input cohort reproducibly."
    )
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--batch-size", type=int, default=16)
    inference.add_argument(
        "--note-batch-size",
        type=int,
        default=None,
        help="Override the checkpoint's note microbatch size for inference.",
    )
    args = parser.parse_args(argv)
    if args.command == "train":
        cfg = ModelConfig(**{k: getattr(args, k) for k in vars(defaults)})
        settings = {
            k: getattr(args, k)
            for k in (
                "note_model",
                "note_lr",
                "lr",
                "weight_decay",
                "backbone_weight_decay",
                "warmup_steps",
                "lr_scheduler",
                "context_size",
                "min_context_size",
                "query_size",
                "grad_accum_steps",
                "precision",
                "augment_notes",
                "eval_every",
                "eval_studies",
                "seed",
            )
        }
        settings["cohorts"] = [str(p.resolve()) for p in args.train_cohort]
        settings["mixed_prior"] = bool(args.mixed_prior)
        if args.design_bank:
            settings["design_bank"] = load_bank(args.design_bank)
        elif args.mixed_prior:
            settings["design_bank"] = None
        run_training(
            args.run_dir, args.steps, args.device, args.save_every, settings=settings, cfg=cfg
        )
    elif args.command == "resume":
        run_training(args.run_dir, args.steps, args.device, args.save_every)
    else:
        model, saved = load_run(
            args.run_dir, args.device, note_batch_size=args.note_batch_size
        )
        if args.command == "evaluate":
            if args.eval_cohort:
                del saved
                metrics = evaluate_cohorts(
                    model,
                    args.eval_cohort,
                    args.seed,
                    args.studies,
                    args.context_size,
                    args.query_size,
                    precision=args.precision,
                )
            else:
                bank = (
                    load_bank(args.design_bank)
                    if args.design_bank is not None
                    else load_run_design_bank(args.run_dir, saved)
                )
                del saved
                metrics = evaluate(
                    model,
                    args.seed,
                    args.studies,
                    args.context_size,
                    args.query_size,
                    bank=bank,
                    precision=args.precision,
                )
            print(json.dumps(metrics, indent=2))
        else:
            del saved
            n = predict(
                model,
                args.context,
                args.queries,
                args.output,
                batch_size=args.batch_size,
                context_size=args.context_size,
                seed=args.seed,
                precision=args.precision,
            )
            print(
                f"Wrote arm-specific risks, uncertainty, propensities, and risk differences "
                f"for {n} patients to {args.output}"
            )


if __name__ == "__main__":
    main()
