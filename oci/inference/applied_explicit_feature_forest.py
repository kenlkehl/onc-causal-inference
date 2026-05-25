"""Explicit-feature-only causal forest inference pipeline."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from ..config import AppliedInferenceConfig, ExplicitFeatureForestConfig, ExplicitFeatureSpec
from ..models.causal_forest_head import CausalForestHead
from ..models.explicit_feature_featurizer import get_raw_explicit_features


logger = logging.getLogger(__name__)


def run_applied_inference_explicit_feature_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    num_workers: int = 1,
    verbose: bool = True,
    explicit_feature_columns: Optional[List[str]] = None,
) -> None:
    """Run causal forest using only role-tagged explicit features."""
    specs = _get_feature_specs(config)
    if not explicit_feature_columns or not specs:
        raise ValueError(
            "model_type='explicit_feature_forest' requires explicit_features.enabled=True "
            "with at least one role-tagged feature."
        )

    logger.info("=" * 80)
    logger.info("APPLIED CAUSAL INFERENCE (EXPLICIT FEATURE CAUSAL FOREST)")
    logger.info("=" * 80)
    logger.info(
        "Using explicit effect-modifier features as X and explicit confounder "
        "features as W for CausalForestDML"
    )

    if config.cv_folds > 1:
        _run_cv_inference(dataset, config, output_path, specs)
    else:
        _run_fixed_split_inference(dataset, config, output_path, specs)


def _get_feature_specs(config: AppliedInferenceConfig) -> List[ExplicitFeatureSpec]:
    if hasattr(config, "explicit_features") and config.explicit_features.features:
        return config.explicit_features.features
    return []


def _get_forest_config(config: AppliedInferenceConfig) -> ExplicitFeatureForestConfig:
    return getattr(config.architecture, "explicit_feature_forest", ExplicitFeatureForestConfig())


def _columns_to_feature_dicts(
    df: pd.DataFrame,
    specs: List[ExplicitFeatureSpec],
) -> List[Dict[str, Any]]:
    """Convert explicit_feat_* DataFrame columns to list-of-dicts format."""
    result = []
    for _, row in df.iterrows():
        values = {}
        for spec in specs:
            col = f"explicit_feat_{spec.name}"
            legacy_col = f"explicit_conf_{spec.name}"
            source_col = col if col in df.columns else legacy_col
            miss_col = f"{source_col}_missing"
            val = row.get(source_col)
            values[spec.name] = val
            values[f"{spec.name}_missing"] = bool(row.get(miss_col, pd.isna(val)))
        result.append(values)
    return result


def _matrix_or_none(values: List[List[float]]) -> Optional[np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return None
    return matrix


def _hstack_present(*matrices: Optional[np.ndarray]) -> Optional[np.ndarray]:
    present = [m for m in matrices if m is not None and m.shape[1] > 0]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return np.hstack(present)


def _build_features(
    df: pd.DataFrame,
    specs: List[ExplicitFeatureSpec],
    continuous_means: Optional[Dict[str, float]] = None,
    continuous_stds: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[str], Dict[str, float], Dict[str, float]]:
    """Build role-split X/W matrices from explicit feature columns."""
    feature_dicts = _columns_to_feature_dicts(df, specs)
    means = {} if continuous_means is None else continuous_means
    stds = {} if continuous_stds is None else continuous_stds

    w_list, w_names = get_raw_explicit_features(
        feature_dicts,
        specs,
        continuous_means=means,
        continuous_stds=stds,
        role="confounder",
    )
    x_list, x_names = get_raw_explicit_features(
        feature_dicts,
        specs,
        continuous_means=means,
        continuous_stds=stds,
        role="effect_modifier",
    )

    return _matrix_or_none(x_list), _matrix_or_none(w_list), x_names, w_names, means, stds


def _fit_predict_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: AppliedInferenceConfig,
    specs: List[ExplicitFeatureSpec],
    cf_config: ExplicitFeatureForestConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    train_T = train_df[config.treatment_column].values
    train_Y = train_df[config.outcome_column].values

    X_train, W_train, x_names, w_names, means, stds = _build_features(train_df, specs)
    X_test, W_test, _, _, _, _ = _build_features(test_df, specs, means, stds)
    nuisance_train = _hstack_present(X_train, W_train)
    nuisance_test = _hstack_present(X_test, W_test)

    if X_train is None and W_train is None:
        raise ValueError("No usable explicit feature columns were found for explicit_feature_forest.")
    if nuisance_train is None or nuisance_test is None:
        raise ValueError("Unable to build nuisance feature matrices for explicit_feature_forest.")

    logger.info(
        f"  Explicit feature matrices: X={0 if X_train is None else X_train.shape[1]}, "
        f"W={0 if W_train is None else W_train.shape[1]}"
    )

    forest = CausalForestHead(
        n_estimators=cf_config.n_estimators,
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        max_features=cf_config.max_features,
        honest=cf_config.honest,
        inference=cf_config.inference,
        random_state=42,
    )
    forest.fit(X_train, train_T, train_Y, W=W_train)
    cf_preds = forest.predict(X_test, return_ci=True)
    tau = cf_preds["tau_pred"]

    outcome_type = getattr(config, "outcome_type", "binary")
    prop_rf = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )
    prop_rf.fit(nuisance_train, train_T)
    propensity = prop_rf.predict_proba(nuisance_test)[:, 1]

    if outcome_type == "continuous":
        outcome_rf = RandomForestRegressor(
            n_estimators=max(50, cf_config.n_estimators // 2),
            max_depth=cf_config.max_depth,
            min_samples_leaf=cf_config.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        outcome_rf.fit(nuisance_train, train_Y)
        outcome_pred = outcome_rf.predict(nuisance_test)
    else:
        outcome_rf = RandomForestClassifier(
            n_estimators=max(50, cf_config.n_estimators // 2),
            max_depth=cf_config.max_depth,
            min_samples_leaf=cf_config.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        outcome_rf.fit(nuisance_train, train_Y)
        outcome_pred = outcome_rf.predict_proba(nuisance_test)[:, 1]

    y0_prob = outcome_pred - propensity * tau
    y1_prob = outcome_pred + (1 - propensity) * tau
    if outcome_type == "binary":
        y0_prob = np.clip(y0_prob, 0, 1)
        y1_prob = np.clip(y1_prob, 0, 1)

    preds = {
        "tau": tau,
        "y0": y0_prob,
        "y1": y1_prob,
        "propensity": propensity,
    }
    if "tau_lower" in cf_preds:
        preds["tau_lower"] = cf_preds["tau_lower"]
        preds["tau_upper"] = cf_preds["tau_upper"]

    metrics = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_x_features": 0 if X_train is None else X_train.shape[1],
        "n_w_features": 0 if W_train is None else W_train.shape[1],
        "n_explicit_features": len(specs),
        "ate_estimate": float(np.mean(tau)),
        "x_feature_names": x_names,
        "w_feature_names": w_names,
    }

    test_T = test_df[config.treatment_column].values
    try:
        metrics["propensity_auroc"] = float(roc_auc_score(test_T, propensity))
    except ValueError:
        metrics["propensity_auroc"] = None

    return preds, metrics


def _run_cv_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    specs: List[ExplicitFeatureSpec],
) -> None:
    k = config.cv_folds
    cf_config = _get_forest_config(config)
    logger.info(f"Starting {k}-Fold Cross-Validation on {len(dataset)} samples")

    dataset = dataset.reset_index(drop=True)
    splits = list(KFold(n_splits=k, shuffle=True, random_state=42).split(dataset))

    results = Parallel(n_jobs=k, verbose=10)(
        delayed(_process_fold)(fold, train_idx, test_idx, dataset, config, specs, cf_config)
        for fold, (train_idx, test_idx) in enumerate(splits)
    )

    all_predictions = [r[0] for r in results]
    all_fold_metrics = [r[1] for r in results]

    results_df = pd.concat(all_predictions).sort_index()
    _save_and_summarize(results_df, output_path)

    log_path = output_path.parent / "training_log.csv"
    metrics_for_csv = [{k: v for k, v in m.items() if not isinstance(v, list)} for m in all_fold_metrics]
    pd.DataFrame(metrics_for_csv).to_csv(log_path, index=False)
    logger.info(f"Fold metrics saved to: {log_path}")


def _process_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    specs: List[ExplicitFeatureSpec],
    cf_config: ExplicitFeatureForestConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    train_df = dataset.iloc[train_idx]
    test_df = dataset.iloc[test_idx]
    logger.info(f"  Fold {fold + 1}: Train={len(train_df)}, Test={len(test_df)}")

    preds, metrics = _fit_predict_split(train_df, test_df, config, specs, cf_config)

    preds_df = test_df.copy()
    preds_df["pred_ite_prob"] = preds["tau"]
    preds_df["pred_y0_prob"] = preds["y0"]
    preds_df["pred_y1_prob"] = preds["y1"]
    preds_df["pred_propensity_prob"] = preds["propensity"]
    preds_df["cv_fold"] = fold + 1
    if "tau_lower" in preds:
        preds_df["pred_ite_lower"] = preds["tau_lower"]
        preds_df["pred_ite_upper"] = preds["tau_upper"]

    metrics["fold"] = fold + 1
    logger.info(f"  ATE estimate: {metrics['ate_estimate']:.4f}")
    return preds_df, metrics


def _run_fixed_split_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    specs: List[ExplicitFeatureSpec],
) -> None:
    cf_config = _get_forest_config(config)
    logger.info("Running Fixed Split Inference (Train/Val/Test)")

    train_df = dataset[dataset[config.split_column] == "train"].copy()
    val_df = dataset[dataset[config.split_column] == "val"].copy()
    test_df = dataset[dataset[config.split_column] == "test"].copy()
    combined_df = pd.concat([train_df, val_df])

    logger.info(f"  Train+Val: {len(combined_df)}, Test: {len(test_df)}")
    preds, metrics = _fit_predict_split(combined_df, test_df, config, specs, cf_config)

    results_df = test_df.copy()
    results_df["pred_ite_prob"] = preds["tau"]
    results_df["pred_y0_prob"] = preds["y0"]
    results_df["pred_y1_prob"] = preds["y1"]
    results_df["pred_propensity_prob"] = preds["propensity"]
    if "tau_lower" in preds:
        results_df["pred_ite_lower"] = preds["tau_lower"]
        results_df["pred_ite_upper"] = preds["tau_upper"]

    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame([{k: v for k, v in metrics.items() if not isinstance(v, list)}]).to_csv(
        log_path, index=False
    )
    logger.info(f"Training log saved to: {log_path}")

    _save_and_summarize(results_df, output_path)


def _save_and_summarize(results_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)

    logger.info(f"\nPredictions saved to: {output_path}")
    logger.info("\nPrediction Summary (Explicit Feature Causal Forest):")
    logger.info(f"  Samples: {len(results_df)}")
    logger.info("  Predicted ITE:")
    logger.info(f"    Mean (ATE): {results_df['pred_ite_prob'].mean():.4f}")
    logger.info(f"    Std: {results_df['pred_ite_prob'].std():.4f}")
    if "pred_ite_lower" in results_df.columns:
        significant = (results_df["pred_ite_lower"] > 0) | (results_df["pred_ite_upper"] < 0)
        logger.info(
            f"    Significant effects (CI excludes 0): "
            f"{significant.sum()} ({significant.mean() * 100:.1f}%)"
        )
    logger.info(f"  Mean predicted propensity: {results_df['pred_propensity_prob'].mean():.4f}")
