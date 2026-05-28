#!/usr/bin/env python
"""Residual text modifier features after prespecified structured variables.

This experiment simulates the applied workflow where a user prespecifies known
confounders/effect modifiers, an extractor provides structured values, and a
neural text model is asked to find only residual treatment-effect
heterogeneity. In the synthetic setting we progressively leave out true effect
modifiers from the structured X set and test whether learned residual features
recover any lost CATE signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from econml.dml import CausalForestDML
from scipy import sparse, stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_oracle_experiments import _resolve_parquet_file, compute_metrics  # noqa: E402


@dataclass
class ResidualNetConfig:
    hidden_dim: int = 128
    repr_dim: int = 16
    epochs: int = 250
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


class ResidualModifierNet(nn.Module):
    """Small MLP that outputs scalar residual tau and hidden modifier features."""

    def __init__(self, input_dim: int, config: ResidualNetConfig):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.repr_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(config.repr_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.body(x)
        tau = self.head(z).squeeze(-1)
        return tau, z


def _load_metadata(dataset_path: str) -> Tuple[List[dict], List[dict]]:
    metadata_file = Path(dataset_path) / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.json not found under {dataset_path}")
    metadata = json.loads(metadata_file.read_text())
    return metadata["confounders"], metadata["effect_modifiers"]


def _true_col(feature: dict) -> str:
    return f"true_{feature['name']}"


def _fit_transform_structured(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[dict],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not features:
        return (
            np.zeros((len(train_df), 0), dtype=np.float32),
            np.zeros((len(test_df), 0), dtype=np.float32),
            [],
        )

    numeric = [_true_col(feature) for feature in features if feature["type"] == "continuous"]
    categorical = [_true_col(feature) for feature in features if feature["type"] == "categorical"]
    used_cols = numeric + categorical
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers, sparse_threshold=0.0)
    X_train = preprocessor.fit_transform(train_df[used_cols]).astype(np.float32)
    X_test = preprocessor.transform(test_df[used_cols]).astype(np.float32)
    names = list(preprocessor.get_feature_names_out())
    return X_train, X_test, names


def _hstack(*arrays: np.ndarray) -> np.ndarray:
    present = [array for array in arrays if array is not None and array.shape[1] > 0]
    if not present:
        raise ValueError("At least one feature block is required")
    return np.concatenate(present, axis=1).astype(np.float32)


def _fit_text_svd(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    n_components: int,
    seed: int,
    analyzer: str = "word",
    max_features: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(1, 2) if analyzer == "word" else (3, 5),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    dim = min(int(n_components), train_matrix.shape[1] - 1, len(train_texts) - 2)
    if dim < 1:
        raise ValueError("Not enough text features to fit SVD")
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    train_svd = svd.fit_transform(train_matrix).astype(np.float32)
    test_svd = svd.transform(test_matrix).astype(np.float32)
    scaler = StandardScaler().fit(train_svd)
    return (
        scaler.transform(train_svd).astype(np.float32),
        scaler.transform(test_svd).astype(np.float32),
    )


def _make_cf(seed: int, n_estimators: int, min_samples_leaf: int) -> CausalForestDML:
    return CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=seed + 1,
            n_jobs=-1,
        ),
        model_t=RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=seed + 2,
            n_jobs=-1,
        ),
        discrete_treatment=True,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=seed + 3,
        n_jobs=-1,
        cv=3,
    )


def _fit_predict_nuisance(
    Z_train: np.ndarray,
    Z_pred: np.ndarray,
    treatment_train: np.ndarray,
    outcome_train: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    e_model = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    e_model.fit(Z_train, treatment_train)
    e_hat = np.clip(e_model.predict_proba(Z_pred)[:, 1], 0.03, 0.97)

    m_model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=20,
        random_state=seed + 10,
        n_jobs=-1,
    )
    m_model.fit(Z_train, outcome_train)
    m_hat = np.clip(m_model.predict(Z_pred), 0.01, 0.99)
    return e_hat, m_hat


def _crossfit_residual_targets(
    Y: np.ndarray,
    T: np.ndarray,
    X_base: np.ndarray,
    W: np.ndarray,
    n_splits: int,
    seed: int,
    cf_n_estimators: int,
    cf_min_samples_leaf: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(Y)
    e_hat = np.zeros(n, dtype=np.float32)
    m_hat = np.zeros(n, dtype=np.float32)
    tau_base = np.zeros(n, dtype=np.float32)
    Z = _hstack(X_base, W)
    inner = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for inner_fold, (fit_idx, pred_idx) in enumerate(inner.split(X_base), start=1):
        e, m = _fit_predict_nuisance(
            Z[fit_idx],
            Z[pred_idx],
            T[fit_idx],
            Y[fit_idx],
            seed=seed * 100 + inner_fold,
        )
        e_hat[pred_idx] = e
        m_hat[pred_idx] = m

        cf = _make_cf(
            seed=seed * 1000 + inner_fold,
            n_estimators=cf_n_estimators,
            min_samples_leaf=cf_min_samples_leaf,
        )
        cf.fit(Y=Y[fit_idx], T=T[fit_idx], X=X_base[fit_idx], W=W[fit_idx])
        tau_base[pred_idx] = cf.effect(X_base[pred_idx]).reshape(-1)
    return e_hat, m_hat, tau_base


def _train_residual_net(
    text_features: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    tau_base: np.ndarray,
    config: ResidualNetConfig,
    device: str,
) -> ResidualModifierNet:
    torch.manual_seed(config.seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        torch_device = torch.device("cpu")

    x = torch.as_tensor(text_features, dtype=torch.float32, device=torch_device)
    y_res = torch.as_tensor(Y - m_hat, dtype=torch.float32, device=torch_device)
    t_res = torch.as_tensor(T - e_hat, dtype=torch.float32, device=torch_device)
    base = torch.as_tensor(tau_base, dtype=torch.float32, device=torch_device)
    weights = torch.clamp(t_res.pow(2), min=1e-3)

    model = ResidualModifierNet(text_features.shape[1], config).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    n = len(Y)
    best_state = None
    best_loss = float("inf")

    for _epoch in range(config.epochs):
        order = torch.randperm(n, generator=generator).numpy()
        for start in range(0, n, config.batch_size):
            idx = torch.as_tensor(order[start: start + config.batch_size], device=torch_device)
            tau_res, _z = model(x[idx])
            pred = t_res[idx] * (base[idx] + tau_res)
            loss = (weights[idx] * (y_res[idx] - pred).pow(2)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        with torch.no_grad():
            tau_res, _z = model(x)
            pred = t_res * (base + tau_res)
            full_loss = float((weights * (y_res - pred).pow(2)).mean().detach().cpu())
        if full_loss < best_loss:
            best_loss = full_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(torch_device)
    model.eval()
    return model


@torch.no_grad()
def _predict_residual_net(
    model: ResidualModifierNet,
    text_features: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        torch_device = torch.device("cpu")
    model.to(torch_device)
    x = torch.as_tensor(text_features, dtype=torch.float32, device=torch_device)
    tau, z = model(x)
    return (
        tau.cpu().numpy().reshape(-1).astype(np.float32),
        z.cpu().numpy().astype(np.float32),
    )


def _reporting_outcomes(
    W_train: np.ndarray,
    W_test: np.ndarray,
    T_train: np.ndarray,
    Y_train: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    e_model.fit(W_train, T_train)
    propensity = np.clip(e_model.predict_proba(W_test)[:, 1], 1e-3, 1 - 1e-3)

    pred_y = []
    for arm in (0, 1):
        mask = T_train == arm
        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=seed + 100 + arm,
            n_jobs=-1,
        )
        model.fit(W_train[mask], Y_train[mask])
        pred_y.append(np.clip(model.predict(W_test), 1e-3, 1 - 1e-3))
    return propensity, pred_y[0], pred_y[1]


def _metrics_for_mode(results_df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    return compute_metrics(
        pred_ite=results_df[pred_col].values,
        true_ite=results_df["true_ite_prob"].values,
        pred_propensity=results_df["pred_propensity"].values,
        true_treatment=results_df["treatment_indicator"].values,
        pred_y0=results_df["pred_y0_prob"].values,
        pred_y1=results_df["pred_y1_prob"].values,
        true_y0=results_df["true_y0_prob"].values,
        true_y1=results_df["true_y1_prob"].values,
        true_outcome=results_df["outcome_indicator"].values,
    )


def _direct_ite_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    return {
        "ite_mse": float(mean_squared_error(true, pred)),
        "ite_mae": float(mean_absolute_error(true, pred)),
        "ite_corr": float(stats.pearsonr(pred, true)[0]),
        "ite_spearman_corr": float(stats.spearmanr(pred, true)[0]),
        "ate_bias": float(abs(np.mean(pred) - np.mean(true))),
    }


def _run_setting(
    df: pd.DataFrame,
    dataset_path: str,
    confounders: List[dict],
    effect_modifiers: List[dict],
    omit_count: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    n_effects = len(effect_modifiers)
    keep_count = max(n_effects - omit_count, 0)
    kept_effects = effect_modifiers[:keep_count]
    omitted_effects = effect_modifiers[keep_count:]
    texts = df[args.text_column].astype(str).tolist()
    predictions: List[pd.DataFrame] = []
    diagnostics = []

    outer = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
    for fold, (train_idx, test_idx) in enumerate(outer.split(df), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        Y_train = train_df["outcome_indicator"].to_numpy(dtype=float)
        T_train = train_df["treatment_indicator"].to_numpy(dtype=int)

        X_base_train, X_base_test, x_names = _fit_transform_structured(
            train_df,
            test_df,
            kept_effects,
        )
        if X_base_train.shape[1] == 0:
            X_base_train = np.zeros((len(train_df), 1), dtype=np.float32)
            X_base_test = np.zeros((len(test_df), 1), dtype=np.float32)
            x_names = ["constant_no_known_effect_modifiers"]

        W_struct_train, W_struct_test, w_names = _fit_transform_structured(
            train_df,
            test_df,
            confounders,
        )
        W_text_train, W_text_test = _fit_text_svd(
            [texts[idx] for idx in train_idx],
            [texts[idx] for idx in test_idx],
            n_components=args.w_text_dim,
            seed=args.random_state + 100 + fold,
            analyzer=args.w_text_analyzer,
            max_features=args.w_text_max_features,
        )
        W_train = _hstack(W_struct_train, W_text_train)
        W_test = _hstack(W_struct_test, W_text_test)

        R_text_train, R_text_test = _fit_text_svd(
            [texts[idx] for idx in train_idx],
            [texts[idx] for idx in test_idx],
            n_components=args.residual_text_dim,
            seed=args.random_state + 200 + fold,
            analyzer=args.residual_text_analyzer,
            max_features=args.residual_text_max_features,
        )

        e_oof, m_oof, tau_base_oof = _crossfit_residual_targets(
            Y=Y_train,
            T=T_train,
            X_base=X_base_train,
            W=W_train,
            n_splits=args.inner_folds,
            seed=args.random_state + 300 + fold,
            cf_n_estimators=args.inner_cf_n_estimators,
            cf_min_samples_leaf=args.cf_min_samples_leaf,
        )

        residual_config = ResidualNetConfig(
            hidden_dim=args.residual_hidden_dim,
            repr_dim=args.residual_repr_dim,
            epochs=args.residual_epochs,
            batch_size=args.residual_batch_size,
            learning_rate=args.residual_lr,
            weight_decay=args.residual_weight_decay,
            seed=args.random_state + 400 + fold,
        )
        residual_net = _train_residual_net(
            R_text_train,
            Y_train,
            T_train,
            e_oof,
            m_oof,
            tau_base_oof,
            residual_config,
            args.device,
        )
        tau_res_train, Z_res_train = _predict_residual_net(
            residual_net,
            R_text_train,
            args.device,
        )
        tau_res_test, Z_res_test = _predict_residual_net(
            residual_net,
            R_text_test,
            args.device,
        )
        z_scaler = StandardScaler().fit(Z_res_train)
        Z_res_train = z_scaler.transform(Z_res_train).astype(np.float32)
        Z_res_test = z_scaler.transform(Z_res_test).astype(np.float32)

        cf_seed = args.random_state * 1000 + omit_count * 100 + fold
        baseline_cf = _make_cf(
            cf_seed,
            n_estimators=args.cf_n_estimators,
            min_samples_leaf=args.cf_min_samples_leaf,
        )
        baseline_cf.fit(Y=Y_train, T=T_train, X=X_base_train, W=W_train)
        tau_base_test = baseline_cf.effect(X_base_test).reshape(-1)

        X_option_b_train = _hstack(X_base_train, Z_res_train)
        X_option_b_test = _hstack(X_base_test, Z_res_test)
        option_b_cf = _make_cf(
            cf_seed + 50,
            n_estimators=args.cf_n_estimators,
            min_samples_leaf=args.cf_min_samples_leaf,
        )
        option_b_cf.fit(Y=Y_train, T=T_train, X=X_option_b_train, W=W_train)
        tau_option_b = option_b_cf.effect(X_option_b_test).reshape(-1)
        tau_option_a = tau_base_test + tau_res_test

        propensity, pred_y0, pred_y1 = _reporting_outcomes(
            W_train,
            W_test,
            T_train,
            Y_train,
            seed=cf_seed + 100,
        )
        fold_preds = test_df.copy()
        fold_preds["omit_count"] = omit_count
        fold_preds["pred_tau_baseline"] = tau_base_test
        fold_preds["pred_tau_option_a"] = tau_option_a
        fold_preds["pred_tau_option_b"] = tau_option_b
        fold_preds["pred_tau_residual_scalar"] = tau_res_test
        fold_preds["pred_propensity"] = propensity
        fold_preds["pred_y0_prob"] = pred_y0
        fold_preds["pred_y1_prob"] = pred_y1
        fold_preds["cv_fold"] = fold
        predictions.append(fold_preds)

        diagnostics.append(
            {
                "fold": fold,
                "x_base_dim": int(X_base_train.shape[1]),
                "w_dim": int(W_train.shape[1]),
                "residual_text_dim": int(R_text_train.shape[1]),
                "residual_repr_dim": int(Z_res_train.shape[1]),
                "tau_base_oof_corr": float(np.corrcoef(tau_base_oof, df.iloc[train_idx]["true_ite_prob"].values)[0, 1]),
                "tau_res_train_std": float(np.std(tau_res_train)),
            }
        )
        print(
            f"omit={omit_count} fold={fold} X_base={X_base_train.shape[1]} "
            f"Z_res={Z_res_train.shape[1]} W={W_train.shape[1]}",
            flush=True,
        )

    results_df = pd.concat(predictions).sort_index()
    metrics = {
        "baseline": _metrics_for_mode(results_df, "pred_tau_baseline"),
        "option_a_additive": _metrics_for_mode(results_df, "pred_tau_option_a"),
        "option_b_residual_x": _metrics_for_mode(results_df, "pred_tau_option_b"),
        "residual_scalar_direct": _direct_ite_metrics(
            results_df["pred_tau_residual_scalar"].values,
            results_df["true_ite_prob"].values,
        ),
    }
    return {
        "omit_count": omit_count,
        "kept_effect_modifiers": [feature["name"] for feature in kept_effects],
        "omitted_effect_modifiers": [feature["name"] for feature in omitted_effects],
        "metrics": metrics,
        "diagnostics": diagnostics,
        "predictions": results_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--omit-counts", default="0,1,2,3,4,5")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-patients", type=int, default=None)

    parser.add_argument("--w-text-dim", type=int, default=64)
    parser.add_argument("--w-text-analyzer", choices=["word", "char_wb"], default="word")
    parser.add_argument("--w-text-max-features", type=int, default=50000)
    parser.add_argument("--residual-text-dim", type=int, default=128)
    parser.add_argument("--residual-text-analyzer", choices=["word", "char_wb"], default="char_wb")
    parser.add_argument("--residual-text-max-features", type=int, default=60000)

    parser.add_argument("--residual-hidden-dim", type=int, default=128)
    parser.add_argument("--residual-repr-dim", type=int, default=16)
    parser.add_argument("--residual-epochs", type=int, default=250)
    parser.add_argument("--residual-batch-size", type=int, default=128)
    parser.add_argument("--residual-lr", type=float, default=1e-3)
    parser.add_argument("--residual-weight-decay", type=float, default=1e-4)

    parser.add_argument("--cf-n-estimators", type=int, default=300)
    parser.add_argument("--inner-cf-n-estimators", type=int, default=160)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = _resolve_parquet_file(args.dataset)
    if parquet_file is None:
        raise FileNotFoundError(f"No parquet file found under {args.dataset}")
    df = pd.read_parquet(parquet_file).reset_index(drop=True)
    if args.max_patients is not None:
        df = df.iloc[: int(args.max_patients)].reset_index(drop=True)
    confounders, effect_modifiers = _load_metadata(args.dataset)
    omit_counts = [int(value) for value in args.omit_counts.split(",") if value.strip()]

    all_results = []
    all_summaries = []
    for omit_count in omit_counts:
        setting = _run_setting(
            df=df,
            dataset_path=args.dataset,
            confounders=confounders,
            effect_modifiers=effect_modifiers,
            omit_count=omit_count,
            args=args,
        )
        predictions = setting.pop("predictions")
        predictions.to_csv(output_dir / f"predictions_omit{omit_count}.csv", index=False)
        all_results.append(setting)
        for mode, metrics in setting["metrics"].items():
            all_summaries.append(
                {
                    "omit_count": omit_count,
                    "mode": mode,
                    "kept_effect_modifiers": ",".join(setting["kept_effect_modifiers"]),
                    "omitted_effect_modifiers": ",".join(setting["omitted_effect_modifiers"]),
                    **metrics,
                }
            )

    result = {
        "config": vars(args),
        "parquet_file": str(parquet_file),
        "confounders": [feature["name"] for feature in confounders],
        "effect_modifiers": [feature["name"] for feature in effect_modifiers],
        "settings": all_results,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
