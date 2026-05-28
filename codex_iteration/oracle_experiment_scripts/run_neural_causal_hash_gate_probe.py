#!/usr/bin/env python
"""Probe neural causal-purity gates over anonymous token-hash features."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.models.causal_purity_hash_extractor import build_token_hash_matrix
from oci.models.neural_causal_hash_gate import (
    NeuralCausalHashGate,
    NeuralCausalHashGateConfig,
    extract_neural_causal_hash_features,
    train_neural_causal_hash_gate,
)
from run_oracle_experiments import _resolve_parquet_file, compute_metrics


def _candidate_columns(X_train, treatment, min_count: int, min_arm_count: int) -> np.ndarray:
    treatment = np.asarray(treatment, dtype=int)
    treated = treatment == 1
    control = ~treated
    counts = np.asarray(X_train.sum(axis=0)).ravel()
    present_treated = np.asarray(X_train[treated].sum(axis=0)).ravel()
    present_control = np.asarray(X_train[control].sum(axis=0)).ravel()
    min_arm = np.minimum.reduce([
        present_treated,
        present_control,
        int(treated.sum()) - present_treated,
        int(control.sum()) - present_control,
    ])
    return np.flatnonzero(
        (counts >= min_count)
        & (counts <= X_train.shape[0] - min_count)
        & (min_arm >= min_arm_count)
    )


def _fit_predict_nuisance(W_train, W_test, treatment_train, outcome_train, seed: int):
    propensity_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    propensity_model.fit(W_train, treatment_train)
    propensity = propensity_model.predict_proba(W_test)[:, 1]

    pred_y = []
    for arm in (0, 1):
        mask = treatment_train == arm
        outcome_model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=seed + 100 + arm,
            n_jobs=-1,
        )
        outcome_model.fit(W_train[mask], outcome_train[mask])
        pred_y.append(np.clip(outcome_model.predict(W_test), 1e-3, 1 - 1e-3))

    return np.clip(propensity, 1e-3, 1 - 1e-3), pred_y[0], pred_y[1]


def _run_cf_fold(X_train, X_test, W_train, W_test, train_df, test_df, fold: int):
    treatment_train = train_df["treatment_indicator"].to_numpy(dtype=int)
    outcome_train = train_df["outcome_indicator"].to_numpy(dtype=float)
    cf = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=10 + fold,
            n_jobs=-1,
        ),
        model_t=RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=20 + fold,
            n_jobs=-1,
        ),
        discrete_treatment=True,
        n_estimators=400,
        min_samples_leaf=10,
        random_state=30 + fold,
        n_jobs=-1,
        cv=3,
    )
    cf.fit(Y=outcome_train, T=treatment_train, X=X_train, W=W_train)
    pred_tau = cf.effect(X_test).reshape(-1)
    propensity, pred_y0, pred_y1 = _fit_predict_nuisance(
        W_train,
        W_test,
        treatment_train,
        outcome_train,
        seed=40 + fold,
    )
    fold_preds = test_df.copy()
    fold_preds["pred_ite_prob"] = pred_tau
    fold_preds["pred_tau"] = pred_tau
    fold_preds["pred_propensity"] = propensity
    fold_preds["pred_y0_prob"] = pred_y0
    fold_preds["pred_y1_prob"] = pred_y1
    fold_preds["cv_fold"] = fold
    return fold_preds


def _metrics_from_predictions(predictions: List[pd.DataFrame]) -> Dict[str, float]:
    results_df = pd.concat(predictions).sort_index()
    return compute_metrics(
        pred_ite=results_df["pred_ite_prob"].values,
        true_ite=results_df["true_ite_prob"].values,
        pred_propensity=results_df["pred_propensity"].values,
        true_treatment=results_df["treatment_indicator"].values,
        pred_y0=results_df["pred_y0_prob"].values,
        pred_y1=results_df["pred_y1_prob"].values,
        true_y0=results_df["true_y0_prob"].values,
        true_y1=results_df["true_y1_prob"].values,
        true_outcome=results_df["outcome_indicator"].values,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--max-length", type=int, default=50000)
    parser.add_argument("--document-window", default="tail", choices=["head", "tail", "head_tail"])
    parser.add_argument("--hash-matrix", default=None)
    parser.add_argument("--original-hashes", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--n-gates", type=int, default=8)
    parser.add_argument("--w-dim", type=int, default=16)
    parser.add_argument("--lambda-purity", type=float, default=20.0)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--min-arm-count", type=int, default=5)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = _resolve_parquet_file(args.dataset)
    if parquet_file is None:
        raise FileNotFoundError(f"No parquet dataset found under {args.dataset}")
    df = pd.read_parquet(parquet_file).reset_index(drop=True)

    if args.hash_matrix and args.original_hashes:
        X_hash = load_npz(args.hash_matrix).astype(np.float32).tocsr()
        original_hashes = np.load(args.original_hashes).astype(np.int64)
    else:
        X_hash, original_hashes = build_token_hash_matrix(
            df["clinical_text"].tolist(),
            model_name=args.model_name,
            max_length=args.max_length,
            document_window=args.document_window,
            min_count=1,
        )

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    predictions = {"neural_w": [], "svd_w": [], "combined_w": []}
    diagnostics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        treatment_train = train_df["treatment_indicator"].to_numpy(dtype=int)
        outcome_train = train_df["outcome_indicator"].to_numpy(dtype=float)
        candidates = _candidate_columns(
            X_hash[train_idx],
            treatment_train,
            args.min_count,
            args.min_arm_count,
        )
        if candidates.size == 0:
            raise RuntimeError(f"No candidate hash columns for fold {fold}")

        X_gate_train = X_hash[train_idx][:, candidates].toarray().astype(np.float32)
        X_gate_test = X_hash[test_idx][:, candidates].toarray().astype(np.float32)
        config = NeuralCausalHashGateConfig(
            n_gates=args.n_gates,
            w_dim=args.w_dim,
            lambda_purity=args.lambda_purity,
        )
        model = NeuralCausalHashGate(X_gate_train.shape[1], config=config)
        train_info = train_neural_causal_hash_gate(
            model,
            X_gate_train,
            treatment=treatment_train,
            outcome=outcome_train,
            steps=args.steps,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=20_000 + fold,
        )
        train_features = extract_neural_causal_hash_features(
            model,
            X_gate_train,
            threshold=args.gate_threshold,
            device=args.device,
        )
        test_features = extract_neural_causal_hash_features(
            model,
            X_gate_test,
            threshold=args.gate_threshold,
            device=args.device,
        )

        selectors = train_features["selectors"]
        top_local = np.argmax(selectors, axis=1)
        top_hashes = original_hashes[candidates[top_local]].astype(int).tolist()
        top_probs = selectors[np.arange(selectors.shape[0]), top_local].astype(float).tolist()
        diagnostics.append({
            "fold": fold,
            "candidate_count": int(candidates.size),
            "top_hashes": top_hashes,
            "top_probs": top_probs,
            **train_info,
        })
        print(
            f"fold={fold} candidates={candidates.size} top_hashes={top_hashes} "
            f"best_purity={train_info['best_purity']:.4f}",
            flush=True,
        )

        svd = TruncatedSVD(n_components=64, random_state=9000 + fold)
        W_svd_train = svd.fit_transform(X_hash[train_idx]).astype(np.float32)
        W_svd_test = svd.transform(X_hash[test_idx]).astype(np.float32)
        W_neural_train = train_features["W"]
        W_neural_test = test_features["W"]

        fold_inputs = {
            "neural_w": (W_neural_train, W_neural_test),
            "svd_w": (W_svd_train, W_svd_test),
            "combined_w": (
                np.concatenate([W_neural_train, W_svd_train], axis=1),
                np.concatenate([W_neural_test, W_svd_test], axis=1),
            ),
        }
        for mode, (W_train, W_test) in fold_inputs.items():
            predictions[mode].append(
                _run_cf_fold(
                    train_features["X"],
                    test_features["X"],
                    W_train,
                    W_test,
                    train_df,
                    test_df,
                    fold,
                )
            )

    results = {
        "config": vars(args),
        "model_config": asdict(NeuralCausalHashGateConfig(
            n_gates=args.n_gates,
            w_dim=args.w_dim,
            lambda_purity=args.lambda_purity,
        )),
        "diagnostics": diagnostics,
        "metrics": {
            mode: _metrics_from_predictions(preds)
            for mode, preds in predictions.items()
        },
    }
    (output_dir / "result.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame([
        {"w_mode": mode, **metrics}
        for mode, metrics in results["metrics"].items()
    ]).to_csv(output_dir / "summary.csv", index=False)
    print(json.dumps(results["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
