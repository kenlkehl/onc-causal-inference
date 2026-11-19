"""Train, resume, evaluate, or query a longitudinal causal study model."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .events import STREAMS, binary_observations
from .model import BASE_MODEL, ModelConfig
from .training import OBSERVED, evaluate, load_run, run_training


def _chunks(table, size):
    for start in range(0, table.num_rows, size):
        yield table.slice(start, size).to_pylist()


def _unique_ids(table):
    ids = table["patient_id"].to_pylist()
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("Patient IDs must be present and unique within each cohort")
    return set(ids)


@torch.inference_mode()
def predict(model, context_file, query_file, output, *, batch_size=16, context_size=None, seed=42):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    fields = ["patient_id", *STREAMS]
    context = pq.read_table(context_file, columns=[*fields, *OBSERVED])
    context_ids = _unique_ids(context)
    if query_file is None:
        if context_size is None or not 0 < context_size < len(context):
            raise ValueError("For a single cohort, set context_size between 1 and N-1")
        chosen = np.zeros(len(context), dtype=bool)
        chosen[np.random.default_rng(seed).choice(len(chosen), context_size, replace=False)] = True
        queries = context.select(fields).filter(~chosen)
        context = context.filter(chosen)
    else:
        queries = pq.read_table(query_file, columns=fields)
        if context_size is not None:
            raise ValueError("Explicit context and query files already define the split")
        if context_ids & _unique_ids(queries):
            raise ValueError("Context and query patient IDs must be disjoint")
    if not len(context) or not len(queries):
        raise ValueError("Need at least one context and one query patient")
    model.eval()
    embeddings = torch.cat(
        [model.patient(rows) for rows in _chunks(context.select(fields), batch_size)]
    )
    t, y = binary_observations(context.select(OBSERVED).to_pylist(), model.centers.device)
    memories, scale = model.condition(embeddings, t, y)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            context.schema.field("patient_id"),
            ("risk_control", pa.float32()),
            ("risk_treated", pa.float32()),
            ("risk_difference", pa.float32()),
        ]
    )
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / "predictions.parquet"
        with pq.ParquetWriter(temporary, schema) as writer:
            for rows in _chunks(queries, batch_size):
                logits = model.query_logits(model.patient(rows), memories)
                p0, p1 = model.risks(logits, scale).cpu().unbind(dim=1)
                writer.write_table(
                    pa.Table.from_arrays(
                        [
                            [r["patient_id"] for r in rows],
                            p0.numpy(),
                            p1.numpy(),
                            (p1 - p0).numpy(),
                        ],
                        schema=schema,
                    )
                )
        os.link(temporary, output)  # publish atomically without replacing an existing result
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
        "evaluate", help="Evaluate against oracle risks from fresh, held-out DGPs."
    )
    inference = sub.add_parser("predict", help="Estimate both risks for unlabeled query patients.")
    for p in (train, resume, scoring, inference):
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for p in (train, resume):
        p.add_argument(
            "--steps", type=int, required=True, help="Total optimizer steps, one study per step."
        )
        p.add_argument("--save-every", type=int, default=50)
    for p in (train, scoring):
        p.add_argument("--context-size", type=int, default=32)
        p.add_argument("--query-size", type=int, default=8)
    for p in (train, scoring, inference):
        p.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--note-model", default=BASE_MODEL, help="Released base model ID or a local snapshot."
    )
    train.add_argument("--note-lr", type=float, default=1e-5)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument(
        "--train-cohort",
        type=Path,
        action="append",
        default=[],
        help="Repeat for materialized study Parquets with oracle risk labels.",
    )
    defaults = ModelConfig()
    for key, value in vars(defaults).items():
        train.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    scoring.add_argument("--studies", type=int, default=16)
    inference.add_argument("--context", type=Path, required=True)
    inference.add_argument("--queries", type=Path, help="Separate Parquet of query histories.")
    inference.add_argument(
        "--context-size", type=int, help="Split a single input cohort reproducibly."
    )
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "train":
        cfg = ModelConfig(**{k: getattr(args, k) for k in vars(defaults)})
        settings = {
            k: getattr(args, k)
            for k in ("note_model", "note_lr", "lr", "context_size", "query_size", "seed")
        }
        settings["cohorts"] = [str(p.resolve()) for p in args.train_cohort]
        run_training(
            args.run_dir, args.steps, args.device, args.save_every, settings=settings, cfg=cfg
        )
    elif args.command == "resume":
        run_training(args.run_dir, args.steps, args.device, args.save_every)
    else:
        model, saved = load_run(args.run_dir, args.device)
        del saved
        if args.command == "evaluate":
            print(
                json.dumps(
                    evaluate(model, args.seed, args.studies, args.context_size, args.query_size),
                    indent=2,
                )
            )
        else:
            n = predict(
                model,
                args.context,
                args.queries,
                args.output,
                batch_size=args.batch_size,
                context_size=args.context_size,
                seed=args.seed,
            )
            print(
                f"Wrote arm-specific risks and risk differences for {n} patients to {args.output}"
            )


if __name__ == "__main__":
    main()
