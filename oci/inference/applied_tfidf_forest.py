# oci/inference/applied_tfidf_forest.py
"""TF-IDF + Causal Forest baseline inference pipeline.

A non-neural baseline that uses TF-IDF text features directly with
CausalForestDML for treatment effect estimation. No GPU required.
"""

import gc
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

from ..config import AppliedInferenceConfig, ExplicitFeatureSpec, TfidfForestConfig
from ..models.causal_forest_head import CausalForestHead
from ..models.explicit_feature_featurizer import get_raw_explicit_features
from sklearn.ensemble import RandomForestRegressor


logger = logging.getLogger(__name__)


def run_applied_inference_tfidf_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device=None,
    num_workers: int = 1,
    verbose: bool = True,
    explicit_feature_columns: Optional[List[str]] = None
) -> None:
    """
    Run TF-IDF + Causal Forest baseline inference.

    No neural network, no GPU. Pure sklearn pipeline.

    Args:
        dataset: DataFrame with clinical text, outcomes, and treatments
        config: Configuration for applied inference
        output_path: Path to save predictions
        device: Ignored (no GPU needed)
        num_workers: Ignored (single-process)
        verbose: Print detailed logs
        explicit_feature_columns: Optional columns with pre-extracted explicit features
    """
    logger.info("=" * 80)
    logger.info("APPLIED CAUSAL INFERENCE (TF-IDF + CAUSAL FOREST BASELINE)")
    logger.info("=" * 80)
    logger.info("No neural network -- pure TF-IDF + CausalForestDML")

    if config.cv_folds > 1:
        _run_cv_inference_tfidf(
            dataset, config, output_path, verbose, explicit_feature_columns
        )
    else:
        _run_fixed_split_inference_tfidf(
            dataset, config, output_path, verbose, explicit_feature_columns
        )


def _get_tfidf_config(config: AppliedInferenceConfig) -> TfidfForestConfig:
    """Get TfidfForestConfig from the architecture config."""
    return getattr(config.architecture, 'tfidf_forest', TfidfForestConfig())


def _get_explicit_feature_specs(config: AppliedInferenceConfig) -> List[ExplicitFeatureSpec]:
    if hasattr(config, "explicit_features") and config.explicit_features.features:
        return config.explicit_features.features
    return []


def _columns_to_feature_dicts(
    df: pd.DataFrame,
    specs: List[ExplicitFeatureSpec],
) -> List[Dict[str, Any]]:
    values = []
    for _, row in df.iterrows():
        item = {}
        for spec in specs:
            col = f"explicit_feat_{spec.name}"
            legacy_col = f"explicit_conf_{spec.name}"
            source_col = col if col in df.columns else legacy_col
            val = row.get(source_col)
            item[spec.name] = val
            item[f"{spec.name}_missing"] = bool(row.get(f"{source_col}_missing", pd.isna(val)))
        values.append(item)
    return values


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


def _build_role_features(
    df: pd.DataFrame,
    specs: List[ExplicitFeatureSpec],
    means: Optional[Dict[str, float]] = None,
    stds: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, float], Dict[str, float]]:
    means = {} if means is None else means
    stds = {} if stds is None else stds
    if not specs:
        return None, None, means, stds
    feature_dicts = _columns_to_feature_dicts(df, specs)
    w_list, _ = get_raw_explicit_features(
        feature_dicts, specs, continuous_means=means, continuous_stds=stds, role="confounder"
    )
    x_list, _ = get_raw_explicit_features(
        feature_dicts, specs, continuous_means=means, continuous_stds=stds, role="effect_modifier"
    )
    return _matrix_or_none(x_list), _matrix_or_none(w_list), means, stds


def _run_cv_inference_tfidf(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    verbose: bool = True,
    explicit_feature_columns: Optional[List[str]] = None
) -> None:
    """K-fold CV for TF-IDF + Causal Forest. Folds run in parallel by default."""
    k = config.cv_folds
    logger.info(f"Starting {k}-Fold Cross-Validation on {len(dataset)} samples (parallel)")

    dataset = dataset.reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = list(kf.split(dataset))

    # Run all folds in parallel (no GPU, so safe to parallelize)
    results = Parallel(n_jobs=k, verbose=10)(
        delayed(_process_fold_tfidf)(
            fold, train_idx, test_idx, dataset, config, explicit_feature_columns
        )
        for fold, (train_idx, test_idx) in enumerate(splits)
    )

    all_predictions = [r[0] for r in results]
    all_fold_metrics = [r[1] for r in results]

    results_df = pd.concat(all_predictions).sort_index()
    _save_and_summarize_tfidf(results_df, output_path)

    # Save fold metrics as training log (for consistency with other pipelines)
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(all_fold_metrics).to_csv(log_path, index=False)
    logger.info(f"Fold metrics saved to: {log_path}")


def _process_fold_tfidf(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    explicit_feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process a single CV fold with TF-IDF + Causal Forest."""
    tfidf_config = _get_tfidf_config(config)

    train_df = dataset.iloc[train_idx]
    test_df = dataset.iloc[test_idx]

    logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    train_texts = train_df[config.text_column].tolist()
    test_texts = test_df[config.text_column].tolist()
    train_T = train_df[config.treatment_column].values
    train_Y = train_df[config.outcome_column].values

    # Step 1: TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=tfidf_config.max_features,
        ngram_range=(tfidf_config.ngram_range_min, tfidf_config.ngram_range_max),
        min_df=tfidf_config.min_df,
        max_df=tfidf_config.max_df,
        sublinear_tf=tfidf_config.sublinear_tf,
        dtype=np.float32
    )
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    W_train = None
    W_test = None
    specs = _get_explicit_feature_specs(config) if explicit_feature_columns else []
    if specs:
        raw_x_train, W_train, means, stds = _build_role_features(train_df, specs)
        raw_x_test, W_test, _, _ = _build_role_features(test_df, specs, means, stds)
        X_train = _hstack_present(X_train, raw_x_train)
        X_test = _hstack_present(X_test, raw_x_test)

    logger.info(f"  TF-IDF features: {X_train.shape[1]} (vocab: {len(vectorizer.vocabulary_)})")

    # Step 2: Fit CausalForestHead
    forest = CausalForestHead(
        n_estimators=tfidf_config.n_estimators,
        max_depth=tfidf_config.max_depth,
        min_samples_leaf=tfidf_config.min_samples_leaf,
        max_features=tfidf_config.max_features_forest,
        honest=tfidf_config.honest,
        inference=tfidf_config.inference,
        random_state=42
    )
    forest.fit(X_train, train_T, train_Y, W=W_train)

    # Step 3: Predict tau with CIs
    cf_preds = forest.predict(X_test, return_ci=True)
    tau = cf_preds['tau_pred']

    # Step 4: Fit separate propensity and outcome models for y0/y1 derivation
    outcome_type = getattr(config, 'outcome_type', 'binary')

    prop_rf = RandomForestClassifier(
        n_estimators=max(50, tfidf_config.n_estimators // 2),
        max_depth=tfidf_config.max_depth,
        min_samples_leaf=tfidf_config.min_samples_leaf,
        random_state=42, n_jobs=-1
    )
    nuisance_train = _hstack_present(X_train, W_train)
    nuisance_test = _hstack_present(X_test, W_test)
    prop_rf.fit(nuisance_train, train_T)
    propensity = prop_rf.predict_proba(nuisance_test)[:, 1]

    if outcome_type == "continuous":
        outcome_rf = RandomForestRegressor(
            n_estimators=max(50, tfidf_config.n_estimators // 2),
            max_depth=tfidf_config.max_depth,
            min_samples_leaf=tfidf_config.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        outcome_rf.fit(nuisance_train, train_Y)
        outcome_pred = outcome_rf.predict(nuisance_test)
    else:
        outcome_rf = RandomForestClassifier(
            n_estimators=max(50, tfidf_config.n_estimators // 2),
            max_depth=tfidf_config.max_depth,
            min_samples_leaf=tfidf_config.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        outcome_rf.fit(nuisance_train, train_Y)
        outcome_pred = outcome_rf.predict_proba(nuisance_test)[:, 1]

    # Step 5: Derive y0/y1 from tau, propensity, outcome
    # From m = e*y1 + (1-e)*y0 and tau = y1 - y0:
    #   y0 = m - e*tau, y1 = m + (1-e)*tau
    y0_prob = outcome_pred - propensity * tau
    y1_prob = outcome_pred + (1 - propensity) * tau
    if outcome_type == "binary":
        y0_prob = np.clip(y0_prob, 0, 1)
        y1_prob = np.clip(y1_prob, 0, 1)

    # Step 6: Build predictions DataFrame
    preds_df = test_df.copy()
    preds_df['pred_ite_prob'] = tau
    preds_df['pred_y0_prob'] = y0_prob
    preds_df['pred_y1_prob'] = y1_prob
    preds_df['pred_propensity_prob'] = propensity
    preds_df['cv_fold'] = fold + 1

    if 'tau_lower' in cf_preds:
        preds_df['pred_ite_lower'] = cf_preds['tau_lower']
        preds_df['pred_ite_upper'] = cf_preds['tau_upper']

    # Fold-level metrics
    fold_metrics = {
        'fold': fold + 1,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_features': X_train.shape[1],
        'vocab_size': len(vectorizer.vocabulary_),
        'ate_estimate': float(np.mean(tau)),
    }

    test_T = test_df[config.treatment_column].values
    test_Y = test_df[config.outcome_column].values
    try:
        fold_metrics['propensity_auroc'] = float(roc_auc_score(test_T, propensity))
    except ValueError:
        fold_metrics['propensity_auroc'] = None

    if outcome_type == "continuous":
        from sklearn.metrics import r2_score, mean_squared_error
        try:
            fold_metrics['outcome_r2'] = float(r2_score(test_Y, outcome_pred))
            fold_metrics['outcome_rmse'] = float(np.sqrt(mean_squared_error(test_Y, outcome_pred)))
        except:
            fold_metrics['outcome_r2'] = None
            fold_metrics['outcome_rmse'] = None
    else:
        try:
            fold_metrics['outcome_auroc'] = float(roc_auc_score(test_Y, outcome_pred))
        except ValueError:
            fold_metrics['outcome_auroc'] = None

    logger.info(f"  ATE estimate: {fold_metrics['ate_estimate']:.4f}")
    if fold_metrics['propensity_auroc'] is not None:
        logger.info(f"  Propensity AUROC: {fold_metrics['propensity_auroc']:.4f}")
    if fold_metrics.get('outcome_auroc') is not None:
        logger.info(f"  Outcome AUROC: {fold_metrics['outcome_auroc']:.4f}")
    if fold_metrics.get('outcome_r2') is not None:
        logger.info(f"  Outcome R²: {fold_metrics['outcome_r2']:.4f}, RMSE: {fold_metrics['outcome_rmse']:.4f}")

    return preds_df, fold_metrics


def _run_fixed_split_inference_tfidf(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    verbose: bool = True,
    explicit_feature_columns: Optional[List[str]] = None
) -> None:
    """Fixed split inference for TF-IDF + Causal Forest."""
    logger.info("Running Fixed Split Inference (Train/Val/Test)")
    tfidf_config = _get_tfidf_config(config)

    train_df = dataset[dataset[config.split_column] == 'train'].copy()
    val_df = dataset[dataset[config.split_column] == 'val'].copy()
    test_df = dataset[dataset[config.split_column] == 'test'].copy()

    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Combine train + val for fitting
    combined_df = pd.concat([train_df, val_df])
    combined_texts = combined_df[config.text_column].tolist()
    test_texts = test_df[config.text_column].tolist()
    combined_T = combined_df[config.treatment_column].values
    combined_Y = combined_df[config.outcome_column].values

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=tfidf_config.max_features,
        ngram_range=(tfidf_config.ngram_range_min, tfidf_config.ngram_range_max),
        min_df=tfidf_config.min_df,
        max_df=tfidf_config.max_df,
        sublinear_tf=tfidf_config.sublinear_tf,
        dtype=np.float32
    )
    X_combined = vectorizer.fit_transform(combined_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    W_combined = None
    W_test = None
    specs = _get_explicit_feature_specs(config) if explicit_feature_columns else []
    if specs:
        raw_x_combined, W_combined, means, stds = _build_role_features(combined_df, specs)
        raw_x_test, W_test, _, _ = _build_role_features(test_df, specs, means, stds)
        X_combined = _hstack_present(X_combined, raw_x_combined)
        X_test = _hstack_present(X_test, raw_x_test)

    logger.info(f"  TF-IDF features: {X_combined.shape[1]} (vocab: {len(vectorizer.vocabulary_)})")

    # Fit CausalForestHead
    forest = CausalForestHead(
        n_estimators=tfidf_config.n_estimators,
        max_depth=tfidf_config.max_depth,
        min_samples_leaf=tfidf_config.min_samples_leaf,
        max_features=tfidf_config.max_features_forest,
        honest=tfidf_config.honest,
        inference=tfidf_config.inference,
        random_state=42
    )
    forest.fit(X_combined, combined_T, combined_Y, W=W_combined)

    cf_preds = forest.predict(X_test, return_ci=True)
    tau = cf_preds['tau_pred']

    # Nuisance models
    outcome_type = getattr(config, 'outcome_type', 'binary')

    prop_rf = RandomForestClassifier(
        n_estimators=max(50, tfidf_config.n_estimators // 2),
        max_depth=tfidf_config.max_depth,
        min_samples_leaf=tfidf_config.min_samples_leaf,
        random_state=42, n_jobs=-1
    )
    nuisance_combined = _hstack_present(X_combined, W_combined)
    nuisance_test = _hstack_present(X_test, W_test)
    prop_rf.fit(nuisance_combined, combined_T)
    propensity = prop_rf.predict_proba(nuisance_test)[:, 1]

    if outcome_type == "continuous":
        outcome_rf = RandomForestRegressor(
            n_estimators=max(50, tfidf_config.n_estimators // 2),
            max_depth=tfidf_config.max_depth,
            min_samples_leaf=tfidf_config.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        outcome_rf.fit(nuisance_combined, combined_Y)
        outcome_pred = outcome_rf.predict(nuisance_test)
    else:
        outcome_rf = RandomForestClassifier(
            n_estimators=max(50, tfidf_config.n_estimators // 2),
            max_depth=tfidf_config.max_depth,
            min_samples_leaf=tfidf_config.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        outcome_rf.fit(nuisance_combined, combined_Y)
        outcome_pred = outcome_rf.predict_proba(nuisance_test)[:, 1]

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

    _save_and_summarize_tfidf(results_df, output_path)


def _save_and_summarize_tfidf(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save results and print summary for TF-IDF + Causal Forest baseline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)

    logger.info(f"\nPredictions saved to: {output_path}")
    logger.info("\nPrediction Summary (TF-IDF + Causal Forest Baseline):")
    logger.info(f"  Samples: {len(results_df)}")

    # ITE column name varies by outcome type
    ite_col = 'pred_ite_prob' if 'pred_ite_prob' in results_df.columns else 'pred_ite'
    scale_label = "probability scale" if ite_col == 'pred_ite_prob' else "predicted"
    logger.info(f"  Predicted ITE ({scale_label}):")
    logger.info(f"    Mean (ATE): {results_df[ite_col].mean():.4f}")
    logger.info(f"    Std: {results_df[ite_col].std():.4f}")
    logger.info(f"    Min: {results_df[ite_col].min():.4f}")
    logger.info(f"    Max: {results_df[ite_col].max():.4f}")

    if 'pred_ite_lower' in results_df.columns:
        significant = (results_df['pred_ite_lower'] > 0) | (results_df['pred_ite_upper'] < 0)
        logger.info(f"    Significant effects (CI excludes 0): {significant.sum()} ({significant.mean()*100:.1f}%)")

    logger.info(f"  Mean predicted propensity: {results_df['pred_propensity_prob'].mean():.4f}")
