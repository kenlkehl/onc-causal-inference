#!/usr/bin/env python
"""Run CausalForestDML with candidate-variable X and unsupervised text-SVD W."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from oci.models.candidate_variable_text_svd_extractor import (  # noqa: E402
    CandidateVariableTextSVDConfig,
    CandidateVariableTextSVDExtractor,
)
from run_oracle_experiments import _resolve_parquet_file, compute_metrics  # noqa: E402


def _fit_predict_nuisance(
    W_train: np.ndarray,
    W_test: np.ndarray,
    treatment_train: np.ndarray,
    outcome_train: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    propensity_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    propensity_model.fit(W_train, treatment_train)
    propensity = propensity_model.predict_proba(W_test)[:, 1]

    pred_y: List[np.ndarray] = []
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

    return (
        np.clip(propensity, 1e-3, 1 - 1e-3),
        pred_y[0],
        pred_y[1],
    )


def _metrics_from_predictions(results_df: pd.DataFrame) -> Dict[str, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-prefix", default="llm_extracted_")
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--w-dim", type=int, default=64)
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--cf-n-estimators", type=int, default=400)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = _resolve_parquet_file(args.dataset)
    if parquet_file is None:
        raise FileNotFoundError(f"No parquet dataset found under {args.dataset}")
    df = pd.read_parquet(parquet_file).reset_index(drop=True)

    extractor_config = CandidateVariableTextSVDConfig(
        candidate_prefix=args.candidate_prefix,
        text_column=args.text_column,
        w_dim=args.w_dim,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )

    kf = KFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.random_state,
    )
    predictions: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        treatment_train = train_df["treatment_indicator"].to_numpy(dtype=int)
        outcome_train = train_df["outcome_indicator"].to_numpy(dtype=float)

        extractor = CandidateVariableTextSVDExtractor(
            extractor_config,
            random_state=args.random_state + fold,
        )
        extractor.fit(train_df)
        train_features = extractor.transform(train_df)
        test_features = extractor.transform(test_df)

        cf_seed = args.random_state * 1000 + fold
        cf = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=20,
                random_state=cf_seed + 1,
                n_jobs=-1,
            ),
            model_t=RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=20,
                random_state=cf_seed + 2,
                n_jobs=-1,
            ),
            discrete_treatment=True,
            n_estimators=args.cf_n_estimators,
            min_samples_leaf=args.cf_min_samples_leaf,
            random_state=cf_seed + 3,
            n_jobs=-1,
            cv=3,
        )
        cf.fit(
            Y=outcome_train,
            T=treatment_train,
            X=train_features["X"],
            W=train_features["W"],
        )
        pred_tau = cf.effect(test_features["X"]).reshape(-1)
        propensity, pred_y0, pred_y1 = _fit_predict_nuisance(
            W_train=train_features["W"],
            W_test=test_features["W"],
            treatment_train=treatment_train,
            outcome_train=outcome_train,
            seed=cf_seed + 100,
        )

        fold_preds = test_df.copy()
        fold_preds["pred_ite_prob"] = pred_tau
        fold_preds["pred_tau"] = pred_tau
        fold_preds["pred_propensity"] = propensity
        fold_preds["pred_y0_prob"] = pred_y0
        fold_preds["pred_y1_prob"] = pred_y1
        fold_preds["cv_fold"] = fold
        predictions.append(fold_preds)

        fold_diag = {"fold": fold, **extractor.diagnostics()}
        diagnostics.append(fold_diag)
        print(
            f"fold={fold} X={fold_diag['x_dim']} W={fold_diag['w_dim']} "
            f"candidates={len(fold_diag['candidate_columns'])}",
            flush=True,
        )

    results_df = pd.concat(predictions).sort_index()
    metrics = _metrics_from_predictions(results_df)
    result = {
        "config": vars(args),
        "extractor_config": asdict(extractor_config),
        "parquet_file": str(parquet_file),
        "diagnostics": diagnostics,
        "metrics": metrics,
    }

    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame([metrics]).to_csv(output_dir / "summary.csv", index=False)
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
