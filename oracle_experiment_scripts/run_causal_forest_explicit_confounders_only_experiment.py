#!/usr/bin/env python
"""Oracle causal forest experiment using ONLY explicit confounders (no text).

Establishes a performance upper bound by running CausalForestDML directly on
known confounder features — no neural network, no text extraction. This answers:
"How good can causal inference get when we already know which confounders matter?"

Two value sources are tested:
  - "true": ground truth confounder values (true_{name} columns) — perfect oracle
  - "llm_extracted": LLM-extracted values (llm_extracted_{name} columns) — extraction oracle

Uses econml CausalForestDML directly (no CausalTextForest, no GPU needed).

Usage:
    # Quick test on 1K dataset, 5 experiments
    python oracle_experiment_scripts/run_causal_forest_explicit_confounders_only_experiment.py \
        --output-dir /tmp/test_explicit_only \
        --datasets ten_confounders \
        --max-experiments 5 \
        --workers 4

    # Full grid on 1K dataset
    python oracle_experiment_scripts/run_causal_forest_explicit_confounders_only_experiment.py \
        --output-dir ../pcori_experiments/oracle_explicit_confounders_only \
        --datasets ten_confounders \
        --workers 8

    # Resume from checkpoint
    python oracle_experiment_scripts/run_causal_forest_explicit_confounders_only_experiment.py \
        --output-dir ../pcori_experiments/oracle_explicit_confounders_only \
        --resume
"""

import argparse
import hashlib
import itertools
import json
import logging
import os
import queue
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdt.config import ExplicitConfounderSpec
from cdt.models.explicit_confounder_featurizer import get_raw_confounder_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Configuration for a single oracle experiment."""
    # Dataset
    dataset_path: str
    dataset_name: str

    # Value source: "true" or "llm_extracted"
    value_source: str

    # Causal forest hyperparameters
    cf_n_estimators: int = 200
    cf_min_samples_leaf: int = 5
    cf_max_depth: Optional[int] = None
    cf_max_features: Optional[str] = "sqrt"

    # Fixed parameters
    n_folds: int = 5
    epochs: int = 0  # Unused, kept for output compatibility

    def config_hash(self) -> str:
        """Generate unique hash for this config."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load_confounder_specs_from_metadata(dataset_path: str) -> List[ExplicitConfounderSpec]:
    """Load confounder specifications from a dataset's metadata.json."""
    metadata_file = Path(dataset_path) / "metadata.json"
    if not metadata_file.exists():
        logger.warning(f"metadata.json not found at {metadata_file}")
        return []

    with open(metadata_file) as f:
        metadata = json.load(f)

    specs = []
    for conf in metadata.get("confounders", []):
        specs.append(ExplicitConfounderSpec(
            name=conf["name"],
            type=conf["type"],
            categories=conf.get("categories"),
            description=conf.get("description"),
        ))

    return specs


def build_confounder_values_from_columns(
    df: pd.DataFrame,
    spec_names: List[str],
    prefix: str
) -> List[Dict[str, Any]]:
    """Build explicit_confounder_values list from dataframe columns.

    Generalizes build_confounder_values to work with any column prefix
    (e.g., "true" for true_{name} columns, "llm_extracted" for llm_extracted_{name}).

    Args:
        df: DataFrame with {prefix}_{name} columns
        spec_names: List of confounder names to include
        prefix: Column prefix ("true" or "llm_extracted")

    Returns:
        List of dicts, one per row
    """
    values_list = []
    for _, row in df.iterrows():
        values = {}
        for name in spec_names:
            col = f"{prefix}_{name}"
            val = row.get(col, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                values[name] = val
                values[f"{name}_missing"] = False
            else:
                values[name] = None
                values[f"{name}_missing"] = True
        values_list.append(values)
    return values_list


def compute_metrics(
    pred_ite: np.ndarray,
    true_ite: np.ndarray,
    pred_propensity: np.ndarray,
    true_treatment: np.ndarray,
    pred_y0: np.ndarray,
    pred_y1: np.ndarray,
    true_y0: np.ndarray,
    true_y1: np.ndarray,
    true_outcome: np.ndarray,
    tau_lower: Optional[np.ndarray] = None,
    tau_upper: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    metrics = {}

    # ITE metrics
    metrics['ite_mse'] = float(mean_squared_error(true_ite, pred_ite))
    metrics['ite_mae'] = float(mean_absolute_error(true_ite, pred_ite))
    try:
        metrics['ite_corr'] = float(stats.pearsonr(pred_ite, true_ite)[0])
    except Exception:
        metrics['ite_corr'] = np.nan
    try:
        metrics['ite_spearman_corr'] = float(stats.spearmanr(pred_ite, true_ite)[0])
    except Exception:
        metrics['ite_spearman_corr'] = np.nan
    metrics['ate_bias'] = float(abs(np.mean(pred_ite) - np.mean(true_ite)))
    metrics['ate_pred'] = float(np.mean(pred_ite))
    metrics['ate_true'] = float(np.mean(true_ite))

    # Propensity metrics
    try:
        metrics['propensity_auroc'] = float(roc_auc_score(true_treatment, pred_propensity))
    except ValueError:
        metrics['propensity_auroc'] = np.nan

    # Outcome metrics
    metrics['y0_mse'] = float(mean_squared_error(true_y0, pred_y0))
    metrics['y1_mse'] = float(mean_squared_error(true_y1, pred_y1))

    # Confidence interval coverage
    if tau_lower is not None and tau_upper is not None:
        coverage = np.mean((true_ite >= tau_lower) & (true_ite <= tau_upper))
        metrics['ci_coverage'] = float(coverage)
        metrics['mean_ci_width'] = float(np.mean(tau_upper - tau_lower))

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Single experiment
# ──────────────────────────────────────────────────────────────────────

def run_single_experiment(config: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
    """Run a single oracle experiment with K-fold CV.

    No neural network — uses CausalForestDML directly on confounder features.
    """
    from econml.dml import CausalForestDML

    dataset_path = Path(config.dataset_path)
    parquet_file = dataset_path / "dataset_with_extraction.parquet"
    if not parquet_file.exists():
        return {'error': f"Dataset not found: {parquet_file}", 'skipped': True}

    df = pd.read_parquet(parquet_file)

    # Load confounder specs
    all_specs = load_confounder_specs_from_metadata(config.dataset_path)
    if not all_specs:
        return {'error': f"No confounder specs found in {config.dataset_path}", 'skipped': True}

    spec_names = [s.name for s in all_specs]

    # Validate that required columns exist
    prefix = config.value_source
    missing_cols = [
        name for name in spec_names
        if f"{prefix}_{name}" not in df.columns
    ]
    if missing_cols:
        return {
            'error': f"Missing columns for value_source='{prefix}': {missing_cols}",
            'skipped': True
        }

    logger.info(
        f"Running: dataset={config.dataset_name}, value_source={config.value_source}, "
        f"n_estimators={config.cf_n_estimators}, min_samples_leaf={config.cf_min_samples_leaf}, "
        f"max_depth={config.cf_max_depth}, max_features={config.cf_max_features}"
    )

    # K-fold cross-validation
    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)

    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Build confounder values
        train_conf_values = build_confounder_values_from_columns(
            train_df, spec_names, prefix
        )
        test_conf_values = build_confounder_values_from_columns(
            test_df, spec_names, prefix
        )

        # Get raw features using the reusable function
        # Compute normalization stats from train fold
        train_features, feature_names = get_raw_confounder_features(
            train_conf_values, all_specs
        )

        # Compute train fold stats for normalizing test fold
        train_array = np.array(train_features)
        continuous_means = {}
        continuous_stds = {}
        for spec in all_specs:
            if spec.type == "continuous":
                vals = []
                for values in train_conf_values:
                    val = values.get(spec.name)
                    missing = values.get(f"{spec.name}_missing", val is None)
                    if not missing and val is not None:
                        vals.append(float(val))
                if vals:
                    continuous_means[spec.name] = sum(vals) / len(vals)
                    variance = sum((v - continuous_means[spec.name]) ** 2 for v in vals) / len(vals)
                    continuous_stds[spec.name] = max(variance ** 0.5, 1e-6)
                else:
                    continuous_means[spec.name] = 0.0
                    continuous_stds[spec.name] = 1.0

        # Re-encode train with stored stats (for consistency)
        train_features, feature_names = get_raw_confounder_features(
            train_conf_values, all_specs,
            continuous_means=continuous_means,
            continuous_stds=continuous_stds
        )
        test_features, _ = get_raw_confounder_features(
            test_conf_values, all_specs,
            continuous_means=continuous_means,
            continuous_stds=continuous_stds
        )

        X_train = np.array(train_features)
        X_test = np.array(test_features)
        T_train = train_df['treatment_indicator'].values.astype(float)
        T_test = test_df['treatment_indicator'].values.astype(float)
        Y_train = train_df['outcome_indicator'].values.astype(float)
        Y_test = test_df['outcome_indicator'].values.astype(float)

        # Fit CausalForestDML
        cf = CausalForestDML(
            n_estimators=config.cf_n_estimators,
            min_samples_leaf=config.cf_min_samples_leaf,
            max_depth=config.cf_max_depth,
            max_features=config.cf_max_features,
            honest=True,
            inference=True,
            random_state=42,
        )
        cf.fit(Y_train, T_train, X=X_train)

        # Predict treatment effects on test fold
        tau_pred = cf.effect(X_test).flatten()
        tau_interval = cf.effect_interval(X_test, alpha=0.05)
        tau_lower = tau_interval[0].flatten()
        tau_upper = tau_interval[1].flatten()

        # Train sklearn random forests on same features for propensity & outcome
        rf_propensity = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        rf_propensity.fit(X_train, T_train.astype(int))
        pred_propensity = rf_propensity.predict_proba(X_test)[:, 1]

        rf_outcome = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        rf_outcome.fit(X_train, Y_train.astype(int))
        pred_outcome_prob = rf_outcome.predict_proba(X_test)[:, 1]

        # Approximate y0/y1 from tau and outcome model
        # y0 ≈ E[Y|X] - e(X)*tau, y1 ≈ E[Y|X] + (1-e(X))*tau
        pred_y0 = pred_outcome_prob - pred_propensity * tau_pred
        pred_y1 = pred_outcome_prob + (1 - pred_propensity) * tau_pred
        pred_y0 = np.clip(pred_y0, 0, 1)
        pred_y1 = np.clip(pred_y1, 0, 1)

        # Store predictions
        fold_preds = test_df.copy()
        fold_preds['pred_y0_prob'] = pred_y0
        fold_preds['pred_y1_prob'] = pred_y1
        fold_preds['pred_ite_prob'] = tau_pred
        fold_preds['pred_propensity'] = pred_propensity
        fold_preds['pred_tau'] = tau_pred
        fold_preds['pred_tau_lower'] = tau_lower
        fold_preds['pred_tau_upper'] = tau_upper
        fold_preds['cv_fold'] = fold + 1

        all_predictions.append(fold_preds)

    # Combine predictions
    results_df = pd.concat(all_predictions).sort_index()

    # Compute metrics
    metrics = compute_metrics(
        pred_ite=results_df['pred_ite_prob'].values,
        true_ite=results_df['true_ite_prob'].values,
        pred_propensity=results_df['pred_propensity'].values,
        true_treatment=results_df['treatment_indicator'].values,
        pred_y0=results_df['pred_y0_prob'].values,
        pred_y1=results_df['pred_y1_prob'].values,
        true_y0=results_df['true_y0_prob'].values,
        true_y1=results_df['true_y1_prob'].values,
        true_outcome=results_df['outcome_indicator'].values,
        tau_lower=results_df['pred_tau_lower'].values,
        tau_upper=results_df['pred_tau_upper'].values
    )

    return {
        'config': asdict(config),
        'metrics': metrics,
        'n_samples': len(results_df),
        'skipped': False,
        'error': None
    }


# ──────────────────────────────────────────────────────────────────────
# Grid generation
# ──────────────────────────────────────────────────────────────────────

def generate_experiment_grid(
    filter_datasets: Optional[List[str]] = None,
) -> List[ExperimentConfig]:
    """Generate all experiment configurations with shuffled order."""

    script_dir = Path(__file__).parent
    datasets = [
        (str(script_dir.parent / "example_synthetic_data_ten_confounders"), "ten_confounders"),
        (str(script_dir.parent.parent / "example_synthetic_data_ten_confounders_50K_rows"), "ten_confounders_50K"),
    ]

    if filter_datasets:
        datasets = [(p, n) for p, n in datasets if n in filter_datasets]

    value_sources = ["true", "llm_extracted"]

    # Causal forest hyperparameter grid
    n_estimators_options = [100, 200, 500]
    min_samples_leaf_options = [3, 5, 10, 20]
    max_depth_options = [None, 10, 20]
    max_features_options = ["sqrt", "log2", None]

    configs = []
    for (dataset_path, dataset_name), value_source in itertools.product(
        datasets, value_sources
    ):
        # Check dataset exists
        if not Path(dataset_path).exists():
            logger.warning(f"Dataset path does not exist, skipping: {dataset_path}")
            continue

        # Check dataset has enough rows
        parquet_file = Path(dataset_path) / "dataset_with_extraction.parquet"
        if parquet_file.exists():
            try:
                df_check = pd.read_parquet(parquet_file)
                if len(df_check) < 50:
                    logger.warning(
                        f"Dataset '{dataset_name}' has only {len(df_check)} rows "
                        f"(need ≥50 for 5-fold CV), skipping"
                    )
                    continue
            except Exception as e:
                logger.warning(f"Could not read {parquet_file}: {e}")
                continue

        for n_est, min_leaf, max_dep, max_feat in itertools.product(
            n_estimators_options, min_samples_leaf_options,
            max_depth_options, max_features_options
        ):
            configs.append(ExperimentConfig(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                value_source=value_source,
                cf_n_estimators=n_est,
                cf_min_samples_leaf=min_leaf,
                cf_max_depth=max_dep,
                cf_max_features=max_feat,
            ))

    # Shuffle so patterns emerge before full grid completes
    random.Random(42).shuffle(configs)

    return configs


# ──────────────────────────────────────────────────────────────────────
# Worker thread
# ──────────────────────────────────────────────────────────────────────

def worker_thread(
    job_queue: queue.Queue,
    results_dict: Dict[str, Any],
    output_dir: Path,
    lock: threading.Lock,
    progress_bar: tqdm
):
    """Worker thread to process experiments (CPU-only, no GPU contention)."""
    while True:
        try:
            config = job_queue.get(timeout=1)
        except queue.Empty:
            break

        config_hash = config.config_hash()

        try:
            result = run_single_experiment(config, output_dir)

            with lock:
                results_dict[config_hash] = result

                # Save individual result
                result_file = output_dir / "results" / f"{config_hash}.json"
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)

                progress_bar.update(1)
                if result.get('skipped'):
                    progress_bar.set_postfix_str(
                        f"Skipped: {result.get('error', 'unknown')[:40]}"
                    )
                else:
                    metrics = result.get('metrics', {})
                    progress_bar.set_postfix_str(
                        f"ITE corr: {metrics.get('ite_corr', 'N/A'):.3f} "
                        f"src={config.value_source}"
                    )

        except Exception as e:
            with lock:
                results_dict[config_hash] = {
                    'config': asdict(config),
                    'error': str(e),
                    'skipped': True
                }
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Error: {str(e)[:40]}")

        finally:
            job_queue.task_done()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Oracle causal forest using only explicit confounders (no text)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="../pcori_experiments/oracle_explicit_confounders_only",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Filter datasets (ten_confounders, ten_confounders_50K)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads (CPU-only)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Unused, kept for output compatibility"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate grid
    configs = generate_experiment_grid(filter_datasets=args.datasets)

    # Update n_folds from args
    for config in configs:
        config.n_folds = args.n_folds
        config.epochs = args.epochs

    logger.info(f"Generated {len(configs)} experiment configurations")

    # Count by value_source
    source_counts = {}
    for c in configs:
        key = (c.dataset_name, c.value_source)
        source_counts[key] = source_counts.get(key, 0) + 1
    for key, count in sorted(source_counts.items()):
        logger.info(f"  {key[0]} / {key[1]}: {count} configs")

    # Load existing results if resuming
    completed_hashes = set()
    results_dict = {}
    if args.resume:
        results_dir = output_dir / "results"
        if results_dir.exists():
            for result_file in results_dir.glob("*.json"):
                config_hash = result_file.stem
                completed_hashes.add(config_hash)
                with open(result_file) as f:
                    results_dict[config_hash] = json.load(f)
            logger.info(f"Resuming: found {len(completed_hashes)} completed experiments")

    # Filter out completed experiments
    pending_configs = [c for c in configs if c.config_hash() not in completed_hashes]

    if args.max_experiments:
        pending_configs = pending_configs[:args.max_experiments]

    logger.info(f"Running {len(pending_configs)} experiments with {args.workers} workers")

    if not pending_configs:
        logger.info("No experiments to run")
        return

    # Create job queue
    job_queue = queue.Queue()
    for config in pending_configs:
        job_queue.put(config)

    # Create worker threads
    lock = threading.Lock()
    progress_bar = tqdm(total=len(pending_configs), desc="Experiments")

    threads = []
    for worker_idx in range(args.workers):
        t = threading.Thread(
            target=worker_thread,
            args=(job_queue, results_dict, output_dir, lock, progress_bar),
            name=f"worker-{worker_idx}"
        )
        t.start()
        threads.append(t)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    progress_bar.close()

    # Aggregate results
    logger.info("Aggregating results...")

    all_results = []
    for config_hash, result in results_dict.items():
        if not result.get('skipped'):
            row = {**result.get('config', {}), **result.get('metrics', {})}
            all_results.append(row)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / "all_results.csv", index=False)
        results_df.to_parquet(output_dir / "all_results.parquet", index=False)

        # Summary statistics
        group_cols = ['dataset_name', 'value_source']
        summary = results_df.groupby(group_cols).agg({
            'ite_corr': ['mean', 'std', 'max'],
            'ite_spearman_corr': ['mean', 'std', 'max'],
            'ite_mse': ['mean', 'std', 'min'],
            'ate_bias': ['mean', 'std', 'min'],
            'ci_coverage': ['mean', 'std'],
            'mean_ci_width': ['mean', 'std'],
        }).round(4)

        summary.to_csv(output_dir / "summary_by_condition.csv")

        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"Total successful experiments: {len(all_results)}")
        logger.info(f"Total skipped: {len(results_dict) - len(all_results)}")

        # Print best configurations per value_source
        for vs in results_df['value_source'].unique():
            subset = results_df[results_df['value_source'] == vs]
            if 'ite_corr' in subset.columns and len(subset) > 0:
                best = subset.nlargest(5, 'ite_corr')[
                    ['dataset_name', 'value_source',
                     'cf_n_estimators', 'cf_min_samples_leaf',
                     'cf_max_depth', 'cf_max_features',
                     'ite_corr', 'ate_bias', 'ci_coverage']
                ]
                logger.info(
                    f"\nTop 5 configs (value_source={vs}) by ITE correlation:\n"
                    f"{best.to_string()}"
                )
    else:
        logger.warning("No successful experiments completed")

    # Save experiment metadata
    metadata = {
        'total_configs': len(configs),
        'completed': len(results_dict),
        'successful': len(all_results) if all_results else 0,
        'workers': args.workers,
        'n_folds': args.n_folds,
        'description': 'Oracle causal forest using only explicit confounders (no text)',
        'value_sources': ['true', 'llm_extracted'],
        'source_counts': {f"{k[0]}_{k[1]}": v for k, v in source_counts.items()},
    }
    with open(output_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
