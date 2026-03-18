# oci/inference/applied_confounder_forest.py
"""Confounders-Only Causal Forest inference pipeline.

A non-neural pathway that uses only LLM-extracted confounder features with
CausalForestDML for treatment effect estimation. No text processing, no GPU,
no training epochs.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

from ..config import AppliedInferenceConfig, ConfounderForestConfig, ExplicitConfounderSpec
from ..models.causal_forest_head import CausalForestHead
from ..models.explicit_confounder_featurizer import get_raw_confounder_features


logger = logging.getLogger(__name__)


def _columns_to_confounder_dicts(
    df: pd.DataFrame,
    specs: List[ExplicitConfounderSpec]
) -> List[Dict[str, Any]]:
    """Convert DataFrame confounder columns to list-of-dicts format.

    Expects columns named 'explicit_conf_{name}' and 'explicit_conf_{name}_missing'.
    """
    result = []
    for _, row in df.iterrows():
        d = {}
        for spec in specs:
            col = f"explicit_conf_{spec.name}"
            miss_col = f"explicit_conf_{spec.name}_missing"
            d[spec.name] = row.get(col)
            d[f"{spec.name}_missing"] = bool(row.get(miss_col, d[spec.name] is None))
        result.append(d)
    return result


def run_applied_inference_confounder_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    num_workers: int = 1,
    verbose: bool = True,
    explicit_confounder_columns: Optional[List[str]] = None
) -> None:
    """
    Run Confounders-Only Causal Forest inference.

    No neural network, no text features, no GPU. Uses only explicit confounder
    features with CausalForestDML.

    Args:
        dataset: DataFrame with confounder columns, outcomes, and treatments
        config: Configuration for applied inference
        output_path: Path to save predictions
        device: Ignored (no GPU needed)
        num_workers: Ignored (single-process)
        verbose: Print detailed logs
        explicit_confounder_columns: Columns with pre-extracted confounders (required)
    """
    if not explicit_confounder_columns:
        raise ValueError(
            "model_type='confounder_forest' requires explicit_confounders to be enabled "
            "with at least one confounder specified. No confounder columns found."
        )

    specs = _get_confounder_specs(config)
    if not specs:
        raise ValueError(
            "model_type='confounder_forest' requires explicit_confounders.confounders "
            "to be specified in the config."
        )

    logger.info("=" * 80)
    logger.info("APPLIED CAUSAL INFERENCE (CONFOUNDERS-ONLY CAUSAL FOREST)")
    logger.info("=" * 80)
    logger.info(f"No neural network -- using {len(specs)} explicit confounders + CausalForestDML")

    if config.cv_folds > 1:
        _run_cv_inference_confounder(
            dataset, config, output_path, specs, explicit_confounder_columns
        )
    else:
        _run_fixed_split_inference_confounder(
            dataset, config, output_path, specs, explicit_confounder_columns
        )


def _get_confounder_specs(config: AppliedInferenceConfig) -> List[ExplicitConfounderSpec]:
    """Get confounder specs from config."""
    if hasattr(config, 'explicit_confounders') and config.explicit_confounders.confounders:
        return config.explicit_confounders.confounders
    return []


def _get_confounder_forest_config(config: AppliedInferenceConfig) -> ConfounderForestConfig:
    """Get ConfounderForestConfig from the architecture config."""
    return getattr(config.architecture, 'confounder_forest', ConfounderForestConfig())


def _build_features(
    df: pd.DataFrame,
    specs: List[ExplicitConfounderSpec],
    continuous_means: Optional[Dict[str, float]] = None,
    continuous_stds: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, List[str], Dict[str, float], Dict[str, float]]:
    """Build feature matrix from confounder columns.

    Returns:
        X: Feature matrix (n_samples, n_features)
        feature_names: Feature name list
        means: Continuous variable means (for test set normalization)
        stds: Continuous variable stds (for test set normalization)
    """
    confounder_dicts = _columns_to_confounder_dicts(df, specs)
    means = continuous_means or {}
    stds = continuous_stds or {}
    features_list, feature_names = get_raw_confounder_features(
        confounder_dicts, specs,
        continuous_means=means, continuous_stds=stds
    )
    # After call, means/stds are populated if they were empty
    return np.array(features_list, dtype=np.float32), feature_names, means, stds


def _run_cv_inference_confounder(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    specs: List[ExplicitConfounderSpec],
    explicit_confounder_columns: List[str]
) -> None:
    """K-fold CV for Confounders-Only Causal Forest."""
    k = config.cv_folds
    logger.info(f"Starting {k}-Fold Cross-Validation on {len(dataset)} samples")

    dataset = dataset.reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = list(kf.split(dataset))

    results = Parallel(n_jobs=k, verbose=10)(
        delayed(_process_fold_confounder)(
            fold, train_idx, test_idx, dataset, config, specs
        )
        for fold, (train_idx, test_idx) in enumerate(splits)
    )

    all_predictions = [r[0] for r in results]
    all_fold_metrics = [r[1] for r in results]

    results_df = pd.concat(all_predictions).sort_index()
    _save_and_summarize(results_df, output_path)

    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(all_fold_metrics).to_csv(log_path, index=False)
    logger.info(f"Fold metrics saved to: {log_path}")


def _process_fold_confounder(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    specs: List[ExplicitConfounderSpec]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process a single CV fold with Confounders-Only Causal Forest."""
    cf_config = _get_confounder_forest_config(config)

    train_df = dataset.iloc[train_idx]
    test_df = dataset.iloc[test_idx]

    logger.info(f"  Fold {fold + 1}: Train={len(train_df)}, Test={len(test_df)}")

    train_T = train_df[config.treatment_column].values
    train_Y = train_df[config.outcome_column].values

    # Build features from confounders (fit normalization on train, apply to test)
    X_train, feature_names, means, stds = _build_features(train_df, specs)
    X_test, _, _, _ = _build_features(test_df, specs, means, stds)

    logger.info(f"  Confounder features: {X_train.shape[1]} ({len(specs)} confounders)")

    # Fit CausalForestHead
    forest = CausalForestHead(
        n_estimators=cf_config.n_estimators,
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        max_features=cf_config.max_features,
        honest=cf_config.honest,
        inference=cf_config.inference,
        random_state=42
    )
    forest.fit(X_train, train_T, train_Y)

    cf_preds = forest.predict(X_test, return_ci=True)
    tau = cf_preds['tau_pred']

    # Nuisance models for y0/y1 derivation
    outcome_type = getattr(config, 'outcome_type', 'binary')

    prop_rf = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=42, n_jobs=-1
    )
    prop_rf.fit(X_train, train_T)
    propensity = prop_rf.predict_proba(X_test)[:, 1]

    if outcome_type == "continuous":
        outcome_rf = RandomForestRegressor(
            n_estimators=max(50, cf_config.n_estimators // 2),
            max_depth=cf_config.max_depth,
            min_samples_leaf=cf_config.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        outcome_rf.fit(X_train, train_Y)
        outcome_pred = outcome_rf.predict(X_test)
    else:
        outcome_rf = RandomForestClassifier(
            n_estimators=max(50, cf_config.n_estimators // 2),
            max_depth=cf_config.max_depth,
            min_samples_leaf=cf_config.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        outcome_rf.fit(X_train, train_Y)
        outcome_pred = outcome_rf.predict_proba(X_test)[:, 1]

    # Derive y0/y1: y0 = m - e*tau, y1 = m + (1-e)*tau
    y0_prob = outcome_pred - propensity * tau
    y1_prob = outcome_pred + (1 - propensity) * tau
    if outcome_type == "binary":
        y0_prob = np.clip(y0_prob, 0, 1)
        y1_prob = np.clip(y1_prob, 0, 1)

    preds_df = test_df.copy()
    preds_df['pred_ite_prob'] = tau
    preds_df['pred_y0_prob'] = y0_prob
    preds_df['pred_y1_prob'] = y1_prob
    preds_df['pred_propensity_prob'] = propensity
    preds_df['cv_fold'] = fold + 1

    if 'tau_lower' in cf_preds:
        preds_df['pred_ite_lower'] = cf_preds['tau_lower']
        preds_df['pred_ite_upper'] = cf_preds['tau_upper']

    fold_metrics = {
        'fold': fold + 1,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_features': X_train.shape[1],
        'n_confounders': len(specs),
        'ate_estimate': float(np.mean(tau)),
    }

    test_T = test_df[config.treatment_column].values
    try:
        fold_metrics['propensity_auroc'] = float(roc_auc_score(test_T, propensity))
    except ValueError:
        fold_metrics['propensity_auroc'] = None

    logger.info(f"  ATE estimate: {fold_metrics['ate_estimate']:.4f}")

    return preds_df, fold_metrics


def _run_fixed_split_inference_confounder(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    specs: List[ExplicitConfounderSpec],
    explicit_confounder_columns: List[str]
) -> None:
    """Fixed split inference for Confounders-Only Causal Forest."""
    cf_config = _get_confounder_forest_config(config)
    logger.info("Running Fixed Split Inference (Train/Val/Test)")

    train_df = dataset[dataset[config.split_column] == 'train'].copy()
    val_df = dataset[dataset[config.split_column] == 'val'].copy()
    test_df = dataset[dataset[config.split_column] == 'test'].copy()

    combined_df = pd.concat([train_df, val_df])
    combined_T = combined_df[config.treatment_column].values
    combined_Y = combined_df[config.outcome_column].values

    X_combined, feature_names, means, stds = _build_features(combined_df, specs)
    X_test, _, _, _ = _build_features(test_df, specs, means, stds)

    logger.info(f"  Train+Val: {len(combined_df)}, Test: {len(test_df)}")
    logger.info(f"  Confounder features: {X_combined.shape[1]}")

    forest = CausalForestHead(
        n_estimators=cf_config.n_estimators,
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        max_features=cf_config.max_features,
        honest=cf_config.honest,
        inference=cf_config.inference,
        random_state=42
    )
    forest.fit(X_combined, combined_T, combined_Y)

    cf_preds = forest.predict(X_test, return_ci=True)
    tau = cf_preds['tau_pred']

    outcome_type = getattr(config, 'outcome_type', 'binary')

    prop_rf = RandomForestClassifier(
        n_estimators=max(50, cf_config.n_estimators // 2),
        max_depth=cf_config.max_depth,
        min_samples_leaf=cf_config.min_samples_leaf,
        random_state=42, n_jobs=-1
    )
    prop_rf.fit(X_combined, combined_T)
    propensity = prop_rf.predict_proba(X_test)[:, 1]

    if outcome_type == "continuous":
        outcome_rf = RandomForestRegressor(
            n_estimators=max(50, cf_config.n_estimators // 2),
            max_depth=cf_config.max_depth,
            min_samples_leaf=cf_config.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        outcome_rf.fit(X_combined, combined_Y)
        outcome_pred = outcome_rf.predict(X_test)
    else:
        outcome_rf = RandomForestClassifier(
            n_estimators=max(50, cf_config.n_estimators // 2),
            max_depth=cf_config.max_depth,
            min_samples_leaf=cf_config.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        outcome_rf.fit(X_combined, combined_Y)
        outcome_pred = outcome_rf.predict_proba(X_test)[:, 1]

    y0_prob = outcome_pred - propensity * tau
    y1_prob = outcome_pred + (1 - propensity) * tau
    if outcome_type == "binary":
        y0_prob = np.clip(y0_prob, 0, 1)
        y1_prob = np.clip(y1_prob, 0, 1)

    results_df = test_df.copy()
    results_df['pred_ite_prob'] = tau
    results_df['pred_y0_prob'] = y0_prob
    results_df['pred_y1_prob'] = y1_prob
    results_df['pred_propensity_prob'] = propensity

    if 'tau_lower' in cf_preds:
        results_df['pred_ite_lower'] = cf_preds['tau_lower']
        results_df['pred_ite_upper'] = cf_preds['tau_upper']

    _save_and_summarize(results_df, output_path)


def _save_and_summarize(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save results and print summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)

    logger.info(f"\nPredictions saved to: {output_path}")
    logger.info("\nPrediction Summary (Confounders-Only Causal Forest):")
    logger.info(f"  Samples: {len(results_df)}")

    ite_col = 'pred_ite_prob'
    logger.info(f"  Predicted ITE:")
    logger.info(f"    Mean (ATE): {results_df[ite_col].mean():.4f}")
    logger.info(f"    Std: {results_df[ite_col].std():.4f}")

    if 'pred_ite_lower' in results_df.columns:
        significant = (results_df['pred_ite_lower'] > 0) | (results_df['pred_ite_upper'] < 0)
        logger.info(f"    Significant effects (CI excludes 0): {significant.sum()} ({significant.mean()*100:.1f}%)")

    logger.info(f"  Mean predicted propensity: {results_df['pred_propensity_prob'].mean():.4f}")
