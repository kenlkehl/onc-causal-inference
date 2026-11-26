#!/usr/bin/env python3
"""Production Option A (and Option B head-to-head) evaluation suite for finished Tabular v4.

Evaluates:
  Suite 1: Real OCI Stage-2 extracted all-evidence catalogs (S0/n1000/seed101 & seed102, raw unmasked)
  Suite 2: All 16 Full-LLM realistic cohorts (artifacts/llm_eval_cohorts/, N_ctx = 800 and N_ctx = 200)
  Suite 3: All 44 complex out-of-family Full-LLM cohorts (artifacts/llm_eval_cohorts_complex/:
           epistatic_s8, nonparametric_s10, composite_var_width, dense_50_50; N_ctx = 800)
  Suite 4: Complex tabular oracle stress cohorts (/data1/ken/pcori_dev/oci_bench_complex/:
           S7..S12b + 5 non-NSCLC oncology contexts under multi_context/)

Compares finished SparseTabularCausalPFN v4 (zero-shot Option A feature presets) head-to-head
against Longitudinal Run 5 (step 1200) and optionally a trained Option B (v6) checkpoint
(--option-b-run-dir). Saves structured JSON and a Markdown report.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from oci.longitudinal.option_b import (
    OPTION_A_PRESETS,
    assemble_option_a_features_from_precomputed,
    load_option_b_run,
    precompute_cohort_raw_features,
)
from oci.longitudinal.training import load_run as load_longitudinal_run
from oci.tabular_pfn.model import format_role_summary
from oci.tabular_pfn.preprocessing import TabularPreprocessor
from oci.tabular_pfn.prior import ROLE_NAMES
from oci.tabular_pfn.training import load_tabular_run

_META_TRUE_COLS = frozenset(
    {
        "true_treatment_prob",
        "true_outcome_prob",
        "true_y0_prob",
        "true_y1_prob",
        "true_ite_prob",
        "true_y0",
        "true_y1",
        "true_ite",
    }
)


def summarize_multimodal_records(records: list[dict]) -> dict:
    """Aggregate CATE/risk/propensity/ATE metrics across a list of cohort evaluation dicts."""
    if not records:
        return {}
    p0 = np.concatenate([r["p0"] for r in records])
    p1 = np.concatenate([r["p1"] for r in records])
    t0 = np.concatenate([r["t0"] for r in records])
    t1 = np.concatenate([r["t1"] for r in records])
    cate = p1 - p0
    tcate = t1 - t0
    e = np.concatenate([r["e"] for r in records])
    te = np.concatenate([r["te"] for r in records])
    within_rs, ate_errs = [], []
    for r in records:
        c = r["p1"] - r["p0"]
        tc = r["t1"] - r["t0"]
        ate_errs.append(float(c.mean() - tc.mean()))
        if len(c) > 1 and c.std() > 1e-8 and tc.std() > 1e-8:
            within_rs.append(float(np.corrcoef(c, tc)[0, 1]))
    pool_r = (
        float(np.corrcoef(cate, tcate)[0, 1])
        if len(cate) > 1 and cate.std() > 1e-8 and tcate.std() > 1e-8
        else 0.0
    )
    return {
        "cohorts": len(records),
        "query_patients": int(len(cate)),
        "cate_rmse": float(np.sqrt(np.mean((cate - tcate) ** 2))),
        "within_r": float(np.mean(within_rs)) if within_rs else 0.0,
        "pool_r": pool_r,
        "risk_rmse": float(np.sqrt(0.5 * (np.mean((p0 - t0) ** 2) + np.mean((p1 - t1) ** 2)))),
        "prop_rmse": float(np.sqrt(np.mean((e - te) ** 2))),
        "ate_rmse": float(np.sqrt(np.mean(np.square(ate_errs)))),
        "zero_cate_rmse": float(np.sqrt(np.mean(tcate**2))),
    }


def _stage2_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    mask = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[mask], truth[mask]
    return {
        "n": int(len(p)),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "mae": float(np.mean(np.abs(p - t))),
        "ate_bias": float(np.mean(p) - np.mean(t)),
        "pred_sd": float(np.std(p)),
        "pearson": float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-8 and np.std(t) > 1e-8 else 0.0,
        "spearman": float(spearmanr(p, t).statistic) if np.std(p) > 1e-8 and np.std(t) > 1e-8 else 0.0,
    }


@torch.inference_mode()
def _score_tabular_model(model, X_ctx_t, t_ctx_t, y_ctx_t, X_q_t, *, M_ctx_t=None, M_q_t=None):
    ol, pl, sc, _ = model(X_ctx_t, t_ctx_t, y_ctx_t, X_q_t, M_ctx=M_ctx_t, M_q=M_q_t)
    mr, _ = model.risk_stats(ol, sc)
    p0_np, p1_np = mr.float().cpu().unbind(dim=1)
    return p0_np.numpy(), p1_np.numpy(), pl.float().sigmoid().cpu().numpy()


# ==============================================================================
# SUITE 1: REAL OCI STAGE-2 EXTRACTED ALL-EVIDENCE CATALOGS (RAW, UNMASKED)
# ==============================================================================
@torch.inference_mode()
def run_stage2_suite(
    tabular_models: list[tuple[str, object]],
    stage2_dirs: list[Path],
    device: str,
) -> dict:
    out: dict = {}
    for seed_dir in stage2_dirs:
        if not seed_dir.is_dir():
            print(f"[suite1] Skipping missing Stage-2 dir: {seed_dir}", flush=True)
            continue
        stage2_dir = seed_dir / "runs" / "default" / "output" / "stage2"
        pred_csv = stage2_dir / "posthoc_predictions_with_oracle_ite.csv"
        if not pred_csv.is_file():
            print(f"[suite1] Skipping missing {pred_csv}", flush=True)
            continue

        pred_df = pd.read_csv(pred_csv).sort_values("_oci_row_id").reset_index(drop=True)
        meta = json.loads((seed_dir / "data" / "metadata.json").read_text())
        true_confs = [x["name"] for x in meta.get("confounders", [])]
        true_mods = [x["name"] for x in meta.get("effect_modifiers", [])]

        ds = pd.read_parquet(seed_dir / "data" / "dataset.parquet")
        truth_ite = pred_df["true_ite"].to_numpy(dtype=np.float64)
        row_ids = pred_df["_oci_row_id"].to_numpy(dtype=np.int64)
        folds = pred_df["outer_fold"].to_numpy(dtype=np.int64)
        t_all = pred_df["treatment"].to_numpy(dtype=np.float32)
        y_all = pred_df["outcome"].to_numpy(dtype=np.float32)
        tp = ds["true_treatment_prob"].to_numpy(dtype=np.float64)[row_ids]

        # Precompute fold-wise preprocessed matrices once for all models
        fold_data = []
        for f in sorted(np.unique(folds)):
            csv_path = stage2_dir / f"outer_{f:03d}" / "extraction" / "extracted_features.csv"
            feat_df = pd.read_csv(csv_path, low_memory=False)
            feat_df = feat_df.set_index("_oci_row_id").loc[row_ids].reset_index()
            feat_cols = [
                c for c in feat_df.columns if c != "_oci_row_id" and not c.startswith("true_")
            ]
            val_m = folds == f
            tr_m = ~val_m
            tr_tbl = pa.Table.from_pandas(feat_df.loc[tr_m, feat_cols], preserve_index=False)
            va_tbl = pa.Table.from_pandas(feat_df.loc[val_m, feat_cols], preserve_index=False)
            pre = TabularPreprocessor()
            X_tr, M_tr = pre.fit_transform(tr_tbl, feat_cols)
            X_va, M_va = pre.transform(va_tbl)
            fold_data.append((val_m, tr_m, X_tr, M_tr, X_va, M_va, list(pre.feature_names)))

        m_s2 = _stage2_metrics(pred_df["estimated_cate"].to_numpy(dtype=np.float64), truth_ite)
        m_s2["prop_rmse"] = float(
            np.sqrt(np.mean((pred_df["propensity"].to_numpy(dtype=np.float64) - tp) ** 2))
        )
        key = str(seed_dir.name)
        if seed_dir.parent.name.startswith("n"):
            key = f"{seed_dir.parent.parent.name}/{seed_dir.parent.name}/{seed_dir.name}"

        seed_res = {
            "true_confounders": true_confs,
            "true_modifiers": true_mods,
            "methods": {"OCI Stage-2 Ensemble": m_s2},
            "role_summaries": {},
        }
        print(f"\n{'=' * 96}\nSUITE 1 — REAL STAGE-2 CATALOG (RAW UNMASKED): {key}")
        print(f"  True confounders: {true_confs} | True modifiers: {true_mods}\n{'=' * 96}")
        print(
            f"  {'OCI Stage-2 Ensemble':<28s} RMSE={m_s2['rmse']:.4f} MAE={m_s2['mae']:.4f} "
            f"ATEbias={m_s2['ate_bias']:+.4f} r={m_s2['pearson']:+.3f} rho={m_s2['spearman']:+.3f} "
            f"PropRMSE={m_s2['prop_rmse']:.4f}"
        )

        for label, tab_model in tabular_models:
            pfn_cate = np.empty(len(pred_df), dtype=np.float64)
            pfn_prop = np.empty(len(pred_df), dtype=np.float64)
            fold_role_acc: dict[str, dict[str, float]] = {}
            fold_role_cnt: dict[str, int] = {}
            t0 = time.perf_counter()

            for val_m, tr_m, X_tr, M_tr, X_va, M_va, fnames in fold_data:
                mem, scale = tab_model.condition(
                    torch.from_numpy(X_tr).to(device),
                    torch.from_numpy(t_all[tr_m]).to(device),
                    torch.from_numpy(y_all[tr_m]).to(device),
                    M_ctx=torch.from_numpy(M_tr).to(device),
                )
                ol, pl = tab_model.query(
                    torch.from_numpy(X_va).to(device),
                    mem,
                    scale,
                    M_q=torch.from_numpy(M_va).to(device),
                )
                mr, _ = tab_model.risk_stats(ol, scale)
                p0, p1 = mr.float().cpu().unbind(dim=1)
                pfn_cate[val_m] = (p1 - p0).numpy()
                pfn_prop[val_m] = pl.float().sigmoid().cpu().numpy()

                rep = tab_model.column_role_report(mem, fnames, calibrated=True)
                for row in rep:
                    fn = row["feature"]
                    if fn not in fold_role_acc:
                        fold_role_acc[fn] = {
                            k: float(v) for k, v in row.items() if isinstance(v, (int, float))
                        }
                        fold_role_cnt[fn] = 1
                    else:
                        for k, v in row.items():
                            if isinstance(v, (int, float)):
                                fold_role_acc[fn][k] += float(v)
                        fold_role_cnt[fn] += 1

            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
            m = _stage2_metrics(pfn_cate, truth_ite)
            m["prop_rmse"] = float(np.sqrt(np.mean((pfn_prop - tp) ** 2)))
            m["runtime_s"] = round(dt, 2)
            seed_res["methods"][label] = m
            print(
                f"  {label:<28s} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} "
                f"ATEbias={m['ate_bias']:+.4f} r={m['pearson']:+.3f} rho={m['spearman']:+.3f} "
                f"PredSD={m['pred_sd']:.4f} PropRMSE={m['prop_rmse']:.4f} ({dt:.2f}s)"
            )

            n_folds = float(len(fold_data))
            avg_report = []
            for fn, sums in fold_role_acc.items():
                cnt = fold_role_cnt[fn]
                avg = {k: float(v / n_folds) for k, v in sums.items()}
                p_sig = float(avg.get("p_signal", 0.0))
                avg["p_signal"] = p_sig
                avg["p_any_confounder"] = float(
                    avg.get("p_confounder", 0.0) + avg.get("p_dual_role", 0.0)
                )
                avg["p_any_modifier"] = float(
                    avg.get("p_modifier", 0.0) + avg.get("p_dual_role", 0.0)
                )
                best_r = (
                    max(ROLE_NAMES[1:], key=lambda r: avg.get(f"p_{r}", 0.0))
                    if p_sig >= 0.5
                    else "noise"
                )
                avg_report.append({"feature": f"{fn} ({cnt}/{int(n_folds)})", "predicted_role": best_r, **avg})
            role_txt = format_role_summary(avg_report, top_k=8)
            seed_res["role_summaries"][label] = role_txt
            print(role_txt, flush=True)

        out[key] = seed_res
    return out


# ==============================================================================
# SUITE 2 & 3: FULL-LLM MULTIMODAL COHORT BENCHMARKS (16 REALISTIC + 44 COMPLEX)
# ==============================================================================
@torch.inference_mode()
def run_multimodal_cohort_suite(
    suite_name: str,
    cohort_items: list[tuple[str, str, Path]],  # (group_tag, cohort_name, parquet_path)
    tab_v4,
    v4_label: str,
    backbone,
    tokenizer,
    run5_model,
    run5_label: str,
    option_b_model,
    option_b_label: str,
    presets: list[str],
    n_ctx_list: tuple[int, ...],
    device: str,
) -> dict:
    if not cohort_items:
        return {}

    method_names = [f"{v4_label} [{p}]" for p in presets]
    if run5_model is not None:
        method_names.append(run5_label)
    if option_b_model is not None:
        method_names.append(option_b_label)

    # Track overall and per-group records
    overall: dict[int, dict[str, list[dict]]] = {
        n_ctx: {m: [] for m in method_names} for n_ctx in n_ctx_list
    }
    by_group: dict[str, dict[int, dict[str, list[dict]]]] = {}
    t0 = time.monotonic()
    use_bf16 = str(device).startswith("cuda") and torch.cuda.is_bf16_supported()

    for idx, (tag, cname, cpath) in enumerate(cohort_items, 1):
        rows = pq.read_table(cpath).to_pylist()
        n_total = len(rows)
        q_start = min(800, max(n_total - 200, int(0.8 * n_total)))
        q_end = n_total

        pre = precompute_cohort_raw_features(
            rows,
            device,
            backbone=backbone,
            tokenizer=tokenizer,
            need_nadir_peak=True,
            need_icd=True,
            need_lex=True,
            need_sent=True,
            need_latest_sent=True,
        )
        t_all = np.asarray([r["treatment_indicator"] for r in rows], dtype=np.float32)
        y_all = np.asarray([r["outcome_indicator"] for r in rows], dtype=np.float32)
        t0_q = np.asarray([r["true_y0"] for r in rows[q_start:q_end]], dtype=np.float32)
        t1_q = np.asarray([r["true_y1"] for r in rows[q_start:q_end]], dtype=np.float32)
        te_q = np.asarray([r["true_treatment_prob"] for r in rows[q_start:q_end]], dtype=np.float32)

        grp_bucket = by_group.setdefault(
            tag, {n_ctx: {m: [] for m in method_names} for n_ctx in n_ctx_list}
        )

        for n_ctx in n_ctx_list:
            eff_ctx = min(n_ctx, q_start)
            t_c = torch.from_numpy(t_all[:eff_ctx]).to(device)
            y_c = torch.from_numpy(y_all[:eff_ctx]).to(device)

            cached_explicit: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for preset in presets:
                fcfg = OPTION_A_PRESETS[preset]
                Xc, Xq, _ = assemble_option_a_features_from_precomputed(
                    pre, 0, eff_ctx, q_start, q_end, fcfg
                )
                cached_explicit[preset] = (Xc, Xq)
                p0, p1, ep = _score_tabular_model(tab_v4, Xc, t_c, y_c, Xq)
                rec = {"p0": p0, "p1": p1, "e": ep, "t0": t0_q, "t1": t1_q, "te": te_q}
                mname = f"{v4_label} [{preset}]"
                overall[n_ctx][mname].append(rec)
                grp_bucket[n_ctx][mname].append(rec)

            if run5_model is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    ol5, pl5, sc5 = run5_model(rows[:eff_ctx], rows[q_start:q_end])
                m5, _ = run5_model.risk_stats(ol5, sc5)
                p0_5, p1_5 = m5.float().cpu().unbind(dim=1)
                rec5 = {
                    "p0": p0_5.numpy(),
                    "p1": p1_5.numpy(),
                    "e": pl5.float().sigmoid().cpu().numpy(),
                    "t0": t0_q,
                    "t1": t1_q,
                    "te": te_q,
                }
                overall[n_ctx][run5_label].append(rec5)
                grp_bucket[n_ctx][run5_label].append(rec5)

            if option_b_model is not None:
                # Reuse precomputed explicit Option-A features matching Option B's feature_cfg!
                ob_Xc, ob_Xq, _ = assemble_option_a_features_from_precomputed(
                    pre, 0, eff_ctx, q_start, q_end, option_b_model.feature_cfg
                )
                ol_b, pl_b, sc_b, _ = option_b_model.forward_multimodal_from_explicit(
                    ob_Xc,
                    ob_Xq,
                    rows[:eff_ctx],
                    rows[q_start:q_end],
                    t_ctx=t_c,
                    y_ctx=y_c,
                )
                mb, _ = option_b_model.risk_stats(ol_b, sc_b)
                p0_b, p1_b = mb.float().cpu().unbind(dim=1)
                recb = {
                    "p0": p0_b.numpy(),
                    "p1": p1_b.numpy(),
                    "e": pl_b.float().sigmoid().cpu().numpy(),
                    "t0": t0_q,
                    "t1": t1_q,
                    "te": te_q,
                }
                overall[n_ctx][option_b_label].append(recb)
                grp_bucket[n_ctx][option_b_label].append(recb)

        if idx % 4 == 0 or idx == len(cohort_items):
            print(
                f"  [{suite_name} {idx:2d}/{len(cohort_items)}] {tag}/{cname} "
                f"({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    summary = {
        "overall": {
            str(n_ctx): {m: summarize_multimodal_records(overall[n_ctx][m]) for m in method_names}
            for n_ctx in n_ctx_list
        },
        "by_group": {
            tag: {
                str(n_ctx): {
                    m: summarize_multimodal_records(by_group[tag][n_ctx][m]) for m in method_names
                }
                for n_ctx in n_ctx_list
            }
            for tag in sorted(by_group)
        },
    }

    for n_ctx in n_ctx_list:
        print(f"\n{'=' * 106}")
        print(f"{suite_name} — OVERALL (N_ctx = {n_ctx}, {len(cohort_items)} cohorts)")
        print(f"{'=' * 106}")
        print(f"{'Method':<52s} {'cate':>7s} {'within':>7s} {'pool':>7s} {'risk':>7s} {'prop':>7s} {'ate':>7s}")
        print("-" * 106)
        for m in method_names:
            s = summary["overall"][str(n_ctx)][m]
            print(
                f"{m:<52s} {s['cate_rmse']:7.4f} {s['within_r']:+7.3f} {s['pool_r']:+7.3f} "
                f"{s['risk_rmse']:7.4f} {s['prop_rmse']:7.4f} {s['ate_rmse']:7.4f}"
            )
    if len(by_group) > 1:
        for tag in sorted(by_group):
            n_ctx = n_ctx_list[0]
            print(f"\n--- {suite_name} subgroup: {tag} (N_ctx = {n_ctx}) ---")
            for m in method_names:
                s = summary["by_group"][tag][str(n_ctx)][m]
                print(
                    f"  {m:<50s} cate={s['cate_rmse']:.4f} within={s['within_r']:+.3f} "
                    f"pool={s['pool_r']:+.3f} prop={s['prop_rmse']:.4f} ate={s['ate_rmse']:.4f}"
                )
    return summary


# ==============================================================================
# SUITE 4: COMPLEX OUT-OF-FAMILY TABULAR ORACLE BENCHMARK (S7-S12b + MULTI-CONTEXT)
# ==============================================================================
@torch.inference_mode()
def run_complex_tabular_suite(
    tabular_models: list[tuple[str, object]],
    bench_root: Path,
    device: str,
) -> dict:
    if not bench_root.is_dir():
        print(f"[suite4] Skipping missing complex tabular root: {bench_root}", flush=True)
        return {}

    cells: list[tuple[str, Path]] = []
    # Core S7..S12b at n1000
    for scen in ("S7", "S8", "S9", "S10", "S11", "S12a", "S12b"):
        for pq_path in sorted(bench_root.glob(f"{scen}/n1000/seed*/data/dataset.parquet")):
            cells.append((scen, pq_path))
    # Multi-context oncology S12a/S12b at n1000
    mc_root = bench_root / "multi_context"
    if mc_root.is_dir():
        for ctx_dir in sorted(p for p in mc_root.iterdir() if p.is_dir() and p.name != "corpora"):
            for scen in ("S12a", "S12b"):
                for pq_path in sorted(ctx_dir.glob(f"{scen}/n1000/seed*/data/dataset.parquet")):
                    cells.append((f"mc_{ctx_dir.name}_{scen}", pq_path))

    if not cells:
        return {}

    by_group: dict[str, dict[str, list[dict]]] = {}
    overall: dict[str, list[dict]] = {lbl: [] for lbl, _ in tabular_models}
    t0 = time.monotonic()

    for idx, (group, pq_path) in enumerate(cells, 1):
        tbl = pq.read_table(pq_path)
        n = tbl.num_rows
        n_ctx = min(800, int(0.8 * n))
        feat_cols = [
            c
            for c in tbl.column_names
            if c.startswith("true_") and c not in _META_TRUE_COLS
        ]
        if not feat_cols:
            continue
        ctx_tbl = tbl.slice(0, n_ctx)
        q_tbl = tbl.slice(n_ctx, n - n_ctx)
        pre = TabularPreprocessor()
        X_ctx, M_ctx = pre.fit_transform(ctx_tbl, feat_cols)
        X_q, M_q = pre.transform(q_tbl)

        t_ctx = torch.from_numpy(ctx_tbl["treatment_indicator"].to_numpy(zero_copy_only=False).astype(np.float32)).to(device)
        y_ctx = torch.from_numpy(ctx_tbl["outcome_indicator"].to_numpy(zero_copy_only=False).astype(np.float32)).to(device)
        X_ctx_t = torch.from_numpy(X_ctx).to(device)
        M_ctx_t = torch.from_numpy(M_ctx).to(device)
        X_q_t = torch.from_numpy(X_q).to(device)
        M_q_t = torch.from_numpy(M_q).to(device)

        t0_q = q_tbl["true_y0"].to_numpy(zero_copy_only=False).astype(np.float32)
        t1_q = q_tbl["true_y1"].to_numpy(zero_copy_only=False).astype(np.float32)
        te_q = q_tbl["true_treatment_prob"].to_numpy(zero_copy_only=False).astype(np.float32)

        grp_rec = by_group.setdefault(group, {lbl: [] for lbl, _ in tabular_models})
        for lbl, model in tabular_models:
            p0, p1, ep = _score_tabular_model(
                model, X_ctx_t, t_ctx, y_ctx, X_q_t, M_ctx_t=M_ctx_t, M_q_t=M_q_t
            )
            rec = {"p0": p0, "p1": p1, "e": ep, "t0": t0_q, "t1": t1_q, "te": te_q}
            overall[lbl].append(rec)
            grp_rec[lbl].append(rec)

        if idx % 15 == 0 or idx == len(cells):
            print(
                f"  [suite4 {idx:3d}/{len(cells)}] {group} P={X_ctx.shape[1]} "
                f"({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    summary = {
        "overall": {lbl: summarize_multimodal_records(overall[lbl]) for lbl, _ in tabular_models},
        "by_group": {
            grp: {lbl: summarize_multimodal_records(by_group[grp][lbl]) for lbl, _ in tabular_models}
            for grp in sorted(by_group)
        },
    }
    print(f"\n{'=' * 102}")
    print(f"SUITE 4 — COMPLEX OUT-OF-FAMILY TABULAR BENCHMARK ({len(cells)} cells, N_ctx=800)")
    print(f"{'=' * 102}")
    for lbl, _ in tabular_models:
        s = summary["overall"][lbl]
        print(
            f"  OVERALL {lbl:<32s} cate={s['cate_rmse']:.4f} within={s['within_r']:+.3f} "
            f"pool={s['pool_r']:+.3f} prop={s['prop_rmse']:.4f} ate={s['ate_rmse']:.4f}"
        )
    for grp in sorted(by_group):
        parts = []
        for lbl, _ in tabular_models:
            s = summary["by_group"][grp][lbl]
            parts.append(f"{lbl}: cate={s['cate_rmse']:.4f} wr={s['within_r']:+.3f}")
        print(f"  {grp:<26s} | " + " | ".join(parts))
    return summary


# ==============================================================================
# MARKDOWN REPORT BUILDER
# ==============================================================================
def build_markdown_summary(payload: dict) -> str:
    lines = [
        "# Option A / Tabular v4 Evaluation Summary",
        "",
        f"- **Tabular v4 checkpoint:** `{payload['meta']['tabular_v4_dir']}` (step `{payload['meta']['tabular_v4_step']}`)",
    ]
    if payload["meta"].get("run5_step") is not None:
        lines.append(
            f"- **Longitudinal Run 5 checkpoint:** `{payload['meta']['run5_dir']}` (step `{payload['meta']['run5_step']}`)"
        )
    if payload["meta"].get("option_b_step") is not None:
        lines.append(
            f"- **Option B (v6) checkpoint:** `{payload['meta']['option_b_dir']}` (step `{payload['meta']['option_b_step']}`)"
        )
    lines.append("")

    if payload.get("suite1_stage2"):
        lines.append("## Suite 1: Real OCI Stage-2 Extracted Catalogs (Raw, Unmasked)")
        lines.append("")
        for key, sres in payload["suite1_stage2"].items():
            lines.append(f"### {key}")
            lines.append("| Method | RMSE | MAE | ATE Bias | Pearson r | Spearman rho | Prop RMSE |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for m, v in sres["methods"].items():
                lines.append(
                    f"| {m} | {v['rmse']:.4f} | {v['mae']:.4f} | {v['ate_bias']:+.4f} | "
                    f"{v['pearson']:+.3f} | {v['spearman']:+.3f} | {v['prop_rmse']:.4f} |"
                )
            lines.append("")

    for key, title in (
        ("suite2_llm16", "Suite 2: 16 Full-LLM Realistic Cohorts"),
        ("suite3_llm_complex", "Suite 3: 44 Complex Out-of-Family Full-LLM Cohorts"),
    ):
        suite = payload.get(key)
        if not suite:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for n_ctx, methods in suite["overall"].items():
            lines.append(f"### Overall (`N_ctx = {n_ctx}`)")
            lines.append("| Method | CATE RMSE | Within-r | Pool-r | Risk RMSE | Prop RMSE | ATE RMSE |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for m, s in methods.items():
                lines.append(
                    f"| {m} | {s['cate_rmse']:.4f} | {s['within_r']:+.3f} | {s['pool_r']:+.3f} | "
                    f"{s['risk_rmse']:.4f} | {s['prop_rmse']:.4f} | {s['ate_rmse']:.4f} |"
                )
            lines.append("")
        if len(suite.get("by_group", {})) > 1:
            n_ctx0 = next(iter(suite["overall"]))
            lines.append(f"### Subgroup Breakdown (`N_ctx = {n_ctx0}`)")
            lines.append("| Subgroup | Method | CATE RMSE | Within-r | Pool-r | Prop RMSE | ATE RMSE |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for grp, gdata in suite["by_group"].items():
                for m, s in gdata[n_ctx0].items():
                    lines.append(
                        f"| `{grp}` | {m} | {s['cate_rmse']:.4f} | {s['within_r']:+.3f} | "
                        f"{s['pool_r']:+.3f} | {s['prop_rmse']:.4f} | {s['ate_rmse']:.4f} |"
                    )
            lines.append("")

    if payload.get("suite4_complex_tabular"):
        s4 = payload["suite4_complex_tabular"]
        lines.append("## Suite 4: Complex Out-of-Family Tabular Stress Cohorts (`S7`–`S12b` + 5 Oncology Contexts)")
        lines.append("")
        lines.append("| Group | Method | CATE RMSE | Within-r | Pool-r | Prop RMSE | ATE RMSE |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for m, s in s4["overall"].items():
            lines.append(
                f"| **OVERALL** | {m} | {s['cate_rmse']:.4f} | {s['within_r']:+.3f} | "
                f"{s['pool_r']:+.3f} | {s['prop_rmse']:.4f} | {s['ate_rmse']:.4f} |"
            )
        for grp, gdata in s4["by_group"].items():
            for m, s in gdata.items():
                lines.append(
                    f"| `{grp}` | {m} | {s['cate_rmse']:.4f} | {s['within_r']:+.3f} | "
                    f"{s['pool_r']:+.3f} | {s['prop_rmse']:.4f} | {s['ate_rmse']:.4f} |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Option A / Tabular v4 complete benchmark suite")
    p.add_argument(
        "--tabular-v4-dir",
        type=Path,
        default=REPO / "artifacts/tabular_causal_pfn_v4_h200",
    )
    p.add_argument(
        "--backbone-path",
        type=Path,
        default=REPO / "artifacts/tinyoncbert_6l_384d",
    )
    p.add_argument(
        "--run5-dir",
        type=Path,
        default=REPO / "artifacts/longitudinal_tinyoncbert_v5",
    )
    p.add_argument("--option-b-run-dir", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "artifacts/eval_option_a_v4",
    )
    p.add_argument(
        "--stage2-roots",
        nargs="*",
        type=Path,
        default=[
            Path("/data1/ken/pcori_dev/oci_bench/S0/n1000/seed101"),
            Path("/data1/ken/pcori_dev/oci_bench/S0/n1000/seed102"),
        ],
    )
    p.add_argument(
        "--llm16-dir",
        type=Path,
        default=REPO / "artifacts/llm_eval_cohorts",
    )
    p.add_argument(
        "--llm-complex-dir",
        type=Path,
        default=REPO / "artifacts/llm_eval_cohorts_complex",
    )
    p.add_argument(
        "--complex-tabular-dir",
        type=Path,
        default=Path("/data1/ken/pcori_dev/oci_bench_complex"),
    )
    p.add_argument(
        "--presets",
        nargs="+",
        choices=sorted(OPTION_A_PRESETS),
        default=["struct_only", "sent_256", "sent_256_lex_128", "option_a_temp128", "option_a_full"],
    )
    p.add_argument(
        "--complex-presets",
        nargs="+",
        choices=sorted(OPTION_A_PRESETS),
        default=["sent_256", "option_a_temp128", "option_a_full"],
    )
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-llm16", action="store_true")
    p.add_argument("--skip-llm-complex", action="store_true")
    p.add_argument("--skip-complex-tabular", action="store_true")
    p.add_argument("--skip-run5", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev = args.device
    t0 = time.monotonic()

    tab_v4, v4_meta = load_tabular_run(args.tabular_v4_dir, dev, prefer_best=True)
    v4_step = int(v4_meta.get("step", -1))
    v4_label = f"PFN-v4 (s{v4_step})"
    print(f"Loaded Tabular v4 (step={v4_step}) on {dev} in {time.monotonic() - t0:.1f}s", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path)
    backbone = AutoModel.from_pretrained(args.backbone_path).to(dev).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    run5_model = None
    run5_step = None
    run5_label = ""
    if not args.skip_run5 and args.run5_dir.is_dir():
        ck5_path = args.run5_dir / "best_checkpoint.pt"
        if ck5_path.is_file():
            ck5 = torch.load(ck5_path, map_location="cpu", weights_only=False)
            run5_step = int(ck5["step"])
            run5_model, _ = load_longitudinal_run(args.run5_dir, dev, note_batch_size=128)
            run5_model.load_state_dict(ck5["weights"], strict=True)
            run5_model.eval()
            run5_label = f"Longitudinal Run 5 (s{run5_step})"
            print(f"Loaded {run5_label}", flush=True)

    option_b_model = None
    option_b_step = None
    option_b_label = ""
    if args.option_b_run_dir is not None and args.option_b_run_dir.is_dir():
        option_b_model, ob_meta = load_option_b_run(
            args.option_b_run_dir,
            device=dev,
            prefer_best=True,
            backbone_override=args.backbone_path,
        )
        option_b_step = int(ob_meta.get("step", -1))
        option_b_label = f"Option-B v6 (s{option_b_step})"
        print(f"Loaded {option_b_label}", flush=True)

    tabular_models: list[tuple[str, object]] = [(v4_label, tab_v4)]
    if option_b_model is not None:
        tabular_models.append((f"{option_b_label} [tab-head]", option_b_model.tabular_head))

    payload: dict = {
        "meta": {
            "device": dev,
            "tabular_v4_dir": str(args.tabular_v4_dir),
            "tabular_v4_step": v4_step,
            "backbone_path": str(args.backbone_path),
            "run5_dir": str(args.run5_dir),
            "run5_step": run5_step,
            "option_b_dir": str(args.option_b_run_dir) if args.option_b_run_dir else None,
            "option_b_step": option_b_step,
        }
    }

    if not args.skip_stage2:
        payload["suite1_stage2"] = run_stage2_suite(tabular_models, args.stage2_roots, dev)

    if not args.skip_llm16 and args.llm16_dir.is_dir():
        cpaths = sorted(Path(p) for p in glob.glob(str(args.llm16_dir / "*/dataset.parquet")))
        items = [("realistic_llm16", p.parent.name, p) for p in cpaths]
        payload["suite2_llm16"] = run_multimodal_cohort_suite(
            "SUITE 2 — 16 FULL-LLM REALISTIC COHORTS",
            items,
            tab_v4,
            v4_label,
            backbone,
            tokenizer,
            run5_model,
            run5_label,
            option_b_model,
            option_b_label,
            args.presets,
            (800, 200),
            dev,
        )

    if not args.skip_llm_complex and args.llm_complex_dir.is_dir():
        cpaths = sorted(
            Path(p) for p in glob.glob(str(args.llm_complex_dir / "*/*/dataset.parquet"))
        )
        items = [(p.parent.parent.name, p.parent.name, p) for p in cpaths]
        payload["suite3_llm_complex"] = run_multimodal_cohort_suite(
            "SUITE 3 — 44 COMPLEX FULL-LLM COHORTS",
            items,
            tab_v4,
            v4_label,
            backbone,
            tokenizer,
            run5_model,
            run5_label,
            option_b_model,
            option_b_label,
            args.complex_presets,
            (800,),
            dev,
        )

    if not args.skip_complex_tabular:
        payload["suite4_complex_tabular"] = run_complex_tabular_suite(
            tabular_models, args.complex_tabular_dir, dev
        )

    payload["meta"]["total_elapsed_s"] = round(time.monotonic() - t0, 2)
    json_path = args.output_dir / "results.json"
    md_path = args.output_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown_summary(payload), encoding="utf-8")
    print(f"\nSaved Option A evaluation outputs:\n  JSON: {json_path}\n  MD:   {md_path}", flush=True)


if __name__ == "__main__":
    main()
