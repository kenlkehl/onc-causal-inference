"""CLI for training, evaluating, and running zero-shot tabular causal inference
with SparseTabularCausalPFN on high-dimensional (up to P = 2,000+) candidate tables.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .model import SparseTabularCausalPFN, TabularModelConfig, format_role_summary
from .preprocessing import TabularPreprocessor, read_tabular_file
from .prior import DGP_FAMILIES, ROLE_NAMES, PriorConfig
from .training import evaluate_tabular, load_tabular_run, run_tabular_training

OVERLAP_MARGIN = 0.05


@torch.inference_mode()
def _score_queries_batched(
    model: SparseTabularCausalPFN,
    X_q: np.ndarray,
    M_q: np.ndarray,
    mem,
    scale,
    *,
    batch_size: int = 512,
):
    device = model.centers.device
    n_q = len(X_q)
    p0s, p1s, s0s, s1s, props = [], [], [], [], []
    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        out_logits, prop_logits = model.query(
            torch.from_numpy(X_q[start:end]).to(device),
            mem,
            scale,
            M_q=torch.from_numpy(M_q[start:end]).to(device),
        )
        mean, sd = model.risk_stats(out_logits, scale)
        p0, p1 = mean.float().cpu().unbind(dim=1)
        s0, s1 = sd.float().cpu().unbind(dim=1)
        prop = prop_logits.float().sigmoid().cpu()
        p0s.append(p0.numpy())
        p1s.append(p1.numpy())
        s0s.append(s0.numpy())
        s1s.append(s1.numpy())
        props.append(prop.numpy())
    return (
        np.concatenate(p0s),
        np.concatenate(p1s),
        np.concatenate(s0s),
        np.concatenate(s1s),
        np.concatenate(props),
    )


@torch.inference_mode()
def predict_tabular(
    model: SparseTabularCausalPFN,
    context_file: str | Path,
    query_file: str | Path | None,
    output: str | Path,
    *,
    id_col: str = "patient_id",
    treatment_col: str = "treatment_indicator",
    outcome_col: str = "outcome_indicator",
    feature_cols: list[str] | None = None,
    context_size: int | None = None,
    cv_folds: int | None = None,
    query_batch_size: int = 512,
    role_report_path: str | Path | None = None,
    print_roles: bool = False,
    role_summary_top_k: int = 15,
    calibrate_roles: bool = True,
    seed: int = 42,
):
    """Estimate arm risks, uncertainty SDs, propensities, CATEs, and optional column roles.

    Supports three modes:
      1. Explicit ``--context`` + ``--queries`` tables.
      2. Single cohort with ``--context-size K`` (random K context, N-K query split).
      3. Single cohort with ``--cv-folds K`` (K-fold out-of-fold zero-shot predictions for all N rows).
    """
    model.eval()
    device = model.centers.device
    ctx_tbl = read_tabular_file(context_file)
    exclude = {id_col, treatment_col, outcome_col}
    if feature_cols is None:
        feature_cols = [
            c
            for c in ctx_tbl.column_names
            if c not in exclude and not c.startswith("true_")
        ]

    need_roles = role_report_path is not None or print_roles
    report = None
    if cv_folds is not None:
        if query_file is not None or context_size is not None:
            raise ValueError("Use --cv-folds without --queries or --context-size")
        if cv_folds < 2 or cv_folds > ctx_tbl.num_rows:
            raise ValueError("--cv-folds must be between 2 and N")
        n_total = ctx_tbl.num_rows
        t_all = np.asarray(ctx_tbl[treatment_col].to_pylist(), dtype=np.float32)
        rng = np.random.default_rng(seed)
        # Stratified fold assignment by treatment arm
        fold_id = np.empty(n_total, dtype=np.int64)
        for arm_val in (0, 1):
            idx_arm = np.flatnonzero((t_all > 0.5) == bool(arm_val))
            rng.shuffle(idx_arm)
            fold_id[idx_arm] = np.arange(len(idx_arm)) % cv_folds

        p0_arr = np.empty(n_total, dtype=np.float32)
        p1_arr = np.empty(n_total, dtype=np.float32)
        s0_arr = np.empty(n_total, dtype=np.float32)
        s1_arr = np.empty(n_total, dtype=np.float32)
        prop_arr = np.empty(n_total, dtype=np.float32)
        role_acc: dict[str, dict] = {}
        role_counts: dict[str, int] = {}

        for f in range(cv_folds):
            val_mask = fold_id == f
            tr_tbl = ctx_tbl.filter(pa.array(~val_mask))
            va_tbl = ctx_tbl.filter(pa.array(val_mask))
            pre = TabularPreprocessor()
            X_tr, M_tr = pre.fit_transform(tr_tbl, feature_cols)
            X_va, M_va = pre.transform(va_tbl)
            t_tr = np.asarray(tr_tbl[treatment_col].to_pylist(), dtype=np.float32)
            y_tr = np.asarray(tr_tbl[outcome_col].to_pylist(), dtype=np.float32)
            mem, scale = model.condition(
                torch.from_numpy(X_tr).to(device),
                torch.from_numpy(t_tr).to(device),
                torch.from_numpy(y_tr).to(device),
                M_ctx=torch.from_numpy(M_tr).to(device),
            )
            p0, p1, s0, s1, prop = _score_queries_batched(
                model, X_va, M_va, mem, scale, batch_size=query_batch_size
            )
            p0_arr[val_mask] = p0
            p1_arr[val_mask] = p1
            s0_arr[val_mask] = s0
            s1_arr[val_mask] = s1
            prop_arr[val_mask] = prop
            if need_roles:
                for row in model.column_role_report(
                    mem, pre.feature_names, calibrated=calibrate_roles
                ):
                    fn = row["feature"]
                    if fn not in role_acc:
                        role_acc[fn] = {k: v for k, v in row.items() if isinstance(v, (int, float))}
                        role_counts[fn] = 1
                    else:
                        for k, v in row.items():
                            if isinstance(v, (int, float)):
                                role_acc[fn][k] += v
                        role_counts[fn] += 1

        q_tbl = ctx_tbl
        if need_roles:
            report = []
            for fn, sums in role_acc.items():
                cnt = role_counts[fn]
                avg = {k: float(v / cnt) for k, v in sums.items()}
                p_sig = float(1.0 - avg.get("p_noise", 1.0))
                avg["p_signal"] = p_sig
                avg["p_any_confounder"] = float(
                    avg.get("p_confounder", 0.0) + avg.get("p_dual_role", 0.0)
                )
                avg["p_any_modifier"] = float(
                    avg.get("p_modifier", 0.0) + avg.get("p_dual_role", 0.0)
                )
                best_role = (
                    max(ROLE_NAMES[1:], key=lambda r: avg.get(f"p_{r}", 0.0))
                    if p_sig >= 0.5
                    else "noise"
                )
                report.append({"feature": fn, "predicted_role": best_role, **avg})
            report.sort(
                key=lambda r: (
                    r["p_signal"],
                    max(r.get("gate_outcome", 0.0), r.get("gate_cate", 0.0)),
                ),
                reverse=True,
            )
    else:
        if query_file is None:
            if context_size is None or not 0 < context_size < ctx_tbl.num_rows:
                raise ValueError("For a single cohort, pass --context-size (1..N-1) or --cv-folds")
            chosen = np.zeros(ctx_tbl.num_rows, dtype=bool)
            chosen[np.random.default_rng(seed).choice(ctx_tbl.num_rows, context_size, replace=False)] = True
            q_tbl = ctx_tbl.filter(pa.array(~chosen))
            ctx_tbl = ctx_tbl.filter(pa.array(chosen))
        else:
            q_tbl = read_tabular_file(query_file)
            if context_size is not None:
                raise ValueError("Explicit --context and --queries already define the split")

        pre = TabularPreprocessor()
        X_ctx, M_ctx = pre.fit_transform(ctx_tbl, feature_cols)
        X_q, M_q = pre.transform(q_tbl)
        t_ctx = np.asarray(ctx_tbl[treatment_col].to_pylist(), dtype=np.float32)
        y_ctx = np.asarray(ctx_tbl[outcome_col].to_pylist(), dtype=np.float32)

        mem, scale = model.condition(
            torch.from_numpy(X_ctx).to(device),
            torch.from_numpy(t_ctx).to(device),
            torch.from_numpy(y_ctx).to(device),
            M_ctx=torch.from_numpy(M_ctx).to(device),
        )
        p0_arr, p1_arr, s0_arr, s1_arr, prop_arr = _score_queries_batched(
            model, X_q, M_q, mem, scale, batch_size=query_batch_size
        )
        if need_roles:
            report = model.column_role_report(
                mem, pre.feature_names, calibrated=calibrate_roles
            )

    outside = (prop_arr < OVERLAP_MARGIN) | (prop_arr > 1.0 - OVERLAP_MARGIN)

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    id_field = q_tbl.schema.field(id_col) if id_col in q_tbl.column_names else pa.field("patient_id", pa.int64())
    q_ids = q_tbl[id_col].to_pylist() if id_col in q_tbl.column_names else list(range(q_tbl.num_rows))
    schema = pa.schema(
        [
            id_field,
            ("risk_control", pa.float32()),
            ("risk_treated", pa.float32()),
            ("risk_difference", pa.float32()),
            ("risk_control_sd", pa.float32()),
            ("risk_treated_sd", pa.float32()),
            ("propensity_score", pa.float32()),
            ("outside_overlap", pa.bool_()),
        ]
    )
    table = pa.Table.from_arrays(
        [
            q_ids,
            p0_arr,
            p1_arr,
            (p1_arr - p0_arr),
            s0_arr,
            s1_arr,
            prop_arr,
            outside,
        ],
        schema=schema,
    )
    with tempfile.TemporaryDirectory(dir=output.parent) as tmpdir:
        tmp_path = Path(tmpdir) / "predictions.parquet"
        pq.write_table(table, tmp_path)
        fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        try:
            os.replace(tmp_path, output)
        except Exception:
            output.unlink(missing_ok=True)
            raise

    if role_report_path is not None and report is not None:
        rpath = Path(role_report_path)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        if rpath.suffix.lower() == ".csv":
            fieldnames = list(report[0].keys()) if report else ["feature", "predicted_role"]
            with rpath.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(report)
        else:
            rpath.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if need_roles and report is not None and role_summary_top_k > 0:
        print(format_role_summary(report, top_k=role_summary_top_k))

    return len(q_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train", help="Train SparseTabularCausalPFN on high-P sparse synthetic prior.")
    p_resume = sub.add_parser("resume", help="Resume an existing SparseTabularCausalPFN run.")
    p_eval = sub.add_parser("evaluate", help="Evaluate on held-out sparse high-P synthetic studies.")
    p_pred = sub.add_parser("predict", help="Zero-shot causal predictions + optional column role report.")

    for p in (p_train, p_resume, p_eval, p_pred):
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--seed", type=int, default=42)

    for p in (p_train, p_resume):
        p.add_argument("--steps", type=int, required=True)
        p.add_argument("--save-every", type=int, default=250)
        p.add_argument("--eval-every", type=int, default=250)
        p.add_argument("--eval-studies", type=int, default=48)
        p.add_argument("--eval-context-size", type=int, default=800)
        p.add_argument("--eval-num-features", type=int, default=500)
        p.add_argument("--min-context-size", type=int, default=80)
        p.add_argument("--context-size", type=int, default=3000)
        p.add_argument("--query-size", type=int, default=64)
        p.add_argument("--grad-accum-steps", type=int, default=8)
        p.add_argument("--lr", type=float, default=5e-4)
        p.add_argument("--weight-decay", type=float, default=1e-2)
        p.add_argument("--warmup-steps", type=int, default=100)
        p.add_argument("--num-workers", type=int, default=0)

    # Prior & model config args on train
    p_train.add_argument("--min-features", type=int, default=20)
    p_train.add_argument("--max-features", type=int, default=2000)
    p_train.add_argument("--null-hte-prob", type=float, default=0.25)
    p_train.add_argument("--instrument-prob", type=float, default=0.70)
    p_train.add_argument("--unmeasured-conf-prob", type=float, default=0.25)
    p_train.add_argument("--extraction-noise-prob", type=float, default=0.35)
    p_train.add_argument("--mnar-prob", type=float, default=0.50)
    p_train.add_argument("--rd-scale-prob", type=float, default=0.25)
    p_train.add_argument("--rct-prob", type=float, default=0.08)
    p_train.add_argument("--design-bank", type=str, default=None)
    defaults = TabularModelConfig()
    for key, value in vars(defaults).items():
        flag = "--" + key.replace("_", "-")
        arg_type = float if isinstance(value, float) else int
        p_train.add_argument(flag, type=arg_type, default=value)

    p_eval.add_argument("--studies", type=int, default=64)
    p_eval.add_argument("--context-size", type=int, default=800)
    p_eval.add_argument("--query-size", type=int, default=128)
    p_eval.add_argument("--num-features", type=int, default=None)
    p_eval.add_argument("--min-features", type=int, default=20)
    p_eval.add_argument("--max-features", type=int, default=2000)
    p_eval.add_argument("--null-hte-prob", type=float, default=0.25)
    p_eval.add_argument("--family", choices=("all", *DGP_FAMILIES), default="all")
    p_eval.add_argument("--design-bank", type=str, default=None)
    p_eval.add_argument("--num-workers", type=int, default=0)
    p_eval.add_argument("--no-baselines", action="store_true")

    p_pred.add_argument("--context", type=Path, required=True)
    p_pred.add_argument("--queries", type=Path, default=None)
    p_pred.add_argument("--context-size", type=int, default=None)
    p_pred.add_argument("--cv-folds", type=int, default=None)
    p_pred.add_argument("--query-batch-size", type=int, default=512)
    p_pred.add_argument("--output", type=Path, required=True)
    p_pred.add_argument("--role-report", type=Path, default=None, help="Output .json or .csv per-column role report.")
    p_pred.add_argument("--print-roles", action="store_true", help="Print top confounders, modifiers, and instruments to stdout.")
    p_pred.add_argument("--role-summary-top-k", type=int, default=15, help="Rows per section in stdout role summary (0 disables).")
    p_pred.add_argument("--uncalibrated-roles", action="store_true", help="Disable empirical-sparsity EM calibration on role probabilities.")
    p_pred.add_argument("--id-col", default="patient_id")
    p_pred.add_argument("--treatment-col", default="treatment_indicator")
    p_pred.add_argument("--outcome-col", default="outcome_indicator")

    args = parser.parse_args(argv)
    if args.command in ("train", "resume"):
        cfg = (
            TabularModelConfig(**{k: getattr(args, k) for k in vars(defaults)})
            if args.command == "train"
            else TabularModelConfig()
        )
        prior = (
            PriorConfig(
                min_features=args.min_features,
                max_features=args.max_features,
                null_hte_prob=args.null_hte_prob,
                instrument_prob=args.instrument_prob,
                unmeasured_conf_prob=args.unmeasured_conf_prob,
                extraction_noise_prob=args.extraction_noise_prob,
                mnar_prob=args.mnar_prob,
                rd_scale_prob=args.rd_scale_prob,
                rct_prob=args.rct_prob,
                design_bank_path=args.design_bank,
                design_split="train",
            )
            if args.command == "train"
            else PriorConfig()
        )
        run_tabular_training(
            args.run_dir,
            args.steps,
            args.device,
            cfg=cfg,
            prior=prior,
            min_context_size=args.min_context_size,
            context_size=args.context_size,
            query_size=args.query_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            save_every=args.save_every,
            eval_every=args.eval_every,
            eval_studies=args.eval_studies,
            eval_context_size=args.eval_context_size,
            eval_num_features=args.eval_num_features,
            num_workers=args.num_workers,
            seed=args.seed,
            resume=(args.command == "resume"),
        )
    elif args.command == "evaluate":
        model, _ = load_tabular_run(args.run_dir, args.device)
        prior = PriorConfig(
            min_features=args.min_features,
            max_features=args.max_features,
            null_hte_prob=args.null_hte_prob,
            design_bank_path=args.design_bank,
            design_split="held_out",
        )
        metrics = evaluate_tabular(
            model,
            studies=args.studies,
            context_size=args.context_size,
            query_size=args.query_size,
            num_features=args.num_features,
            prior=prior,
            family=None if args.family == "all" else args.family,
            seed=args.seed,
            compute_baselines=not args.no_baselines,
            num_workers=args.num_workers,
        )
        print(json.dumps(metrics, indent=2))
    else:
        model, _ = load_tabular_run(args.run_dir, args.device)
        n = predict_tabular(
            model,
            args.context,
            args.queries,
            args.output,
            id_col=args.id_col,
            treatment_col=args.treatment_col,
            outcome_col=args.outcome_col,
            context_size=args.context_size,
            cv_folds=args.cv_folds,
            query_batch_size=args.query_batch_size,
            role_report_path=args.role_report,
            print_roles=args.print_roles,
            role_summary_top_k=args.role_summary_top_k,
            calibrate_roles=not args.uncalibrated_roles,
            seed=args.seed,
        )
        print(f"Wrote tabular causal predictions for {n} patients to {args.output}")


if __name__ == "__main__":
    main()
