#!/usr/bin/env python
"""Matched Pair ITE Estimation experiment on synthetic clinical text.

This script runs experiments using the matched pair ITE estimation approach
with 5-fold cross-validation:

For each fold (80% train, 20% test):
  Stage 1: Train propensity model on full training fold (no inner validation split)
  Stage 2: Match treated/control patients by propensity score OR embedding similarity
  Stage 3: Train outcome/tau model on matched pairs only
  Stage 4: Evaluate on held-out test fold

Key design choices:
- Tau head predicts ITE from untreated patient's embedding only
- Target: log-odds difference between matched pair outcomes
- Representation is frozen after propensity training to preserve covariate balance
- No inner train/val split: trains for fixed epochs on full 80% training data

Experimental Conditions (grid search):
1. Matching method: propensity score vs cosine embedding similarity
2. Matching algorithm: nearest neighbor (greedy) vs optimal (Hungarian)
3. Learning rate: 1e-4, 5e-5
4. Propensity epochs: 25, 50
5. Outcome epochs: 25, 50

Usage:
    python oracle_experiment_scripts/run_matched_pair_ite_experiment.py \
        --dataset example_synthetic_data_one_confounder/dataset.parquet \
        --output-dir ../pcori_experiments/matched_pair_experiment_results \
        --gpu-ids 0 1

To run a quick test:
    python oracle_experiment_scripts/run_matched_pair_ite_experiment.py \
        --dataset example_synthetic_data_one_confounder/dataset.parquet \
        --output-dir ../pcori_experiments/matched_pair_test \
        --gpu-ids 0 \
        --quick-test
"""

import argparse
import gc
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdt.models.matched_pair_ite import (
    PropensityMatchingModel,
    MatchedPairOutcomeModel
)
from cdt.training.matched_pair_training import (
    train_propensity_model,
    train_matched_pair_outcome_model,
    extract_all_representations,
    extract_propensity_scores,
    MatchedPairDataset
)
from cdt.matching import PropensityMatcher, match_by_cosine_similarity
from cdt.config import MatchedPairConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentCondition:
    """Configuration for a single experimental condition."""
    name: str
    matching_method: str  # "propensity" or "embedding"
    matching_algorithm: str  # "nearest" or "optimal"
    propensity_lr: float
    outcome_lr: float
    propensity_epochs: int
    outcome_epochs: int
    caliper: float = 0.2
    representation_dim: int = 256
    hidden_outcome_dim: int = 128
    batch_size: int = 32
    text_column: str = "clinical_text"


def generate_experiment_grid(quick_test: bool = False) -> List[ExperimentCondition]:
    """Generate grid of experimental conditions."""
    conditions = []

    if quick_test:
        # Minimal test configuration
        matching_methods = ["propensity"]
        matching_algorithms = ["optimal"]
        lrs = [1e-4]
        prop_epochs_list = [5]
        outcome_epochs_list = [5]
    else:
        # Full grid
        matching_methods = ["propensity", "embedding"]
        matching_algorithms = ["nearest", "optimal"]
        lrs = [1e-4, 5e-5]
        prop_epochs_list = [25, 50]
        outcome_epochs_list = [25, 50]

    idx = 0
    for matching_method in matching_methods:
        for matching_algorithm in matching_algorithms:
            for lr in lrs:
                for prop_epochs in prop_epochs_list:
                    for outcome_epochs in outcome_epochs_list:
                        idx += 1
                        name = f"{idx:02d}_{matching_method}_{matching_algorithm}_lr{lr}_pe{prop_epochs}_oe{outcome_epochs}"
                        conditions.append(ExperimentCondition(
                            name=name,
                            matching_method=matching_method,
                            matching_algorithm=matching_algorithm,
                            propensity_lr=lr,
                            outcome_lr=lr,
                            propensity_epochs=prop_epochs,
                            outcome_epochs=outcome_epochs,
                        ))

    return conditions


class TextDataset(Dataset):
    """Simple dataset for text + labels."""

    def __init__(
        self,
        texts: List[str],
        treatments: np.ndarray,
        outcomes: np.ndarray
    ):
        self.texts = texts
        self.treatments = torch.tensor(treatments, dtype=torch.float32)
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'texts': self.texts[idx],
            'treatment': self.treatments[idx],
            'outcome': self.outcomes[idx]
        }


def collate_text_batch(batch):
    """Collate function for text batches."""
    texts = [b['texts'] for b in batch]
    treatments = torch.stack([b['treatment'] for b in batch])
    outcomes = torch.stack([b['outcome'] for b in batch])

    return {
        'texts': texts,
        'treatment': treatments,
        'outcome': outcomes
    }


def compute_metrics(
    pred_ite: np.ndarray,
    true_ite: np.ndarray,
    pred_propensity: np.ndarray,
    true_treatment: np.ndarray,
    pred_y0: np.ndarray,
    pred_y1: np.ndarray,
    true_y0: np.ndarray,
    true_y1: np.ndarray,
    true_outcome: np.ndarray
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    metrics = {}

    # ITE metrics
    metrics['ite_mse'] = float(mean_squared_error(true_ite, pred_ite))
    metrics['ite_mae'] = float(mean_absolute_error(true_ite, pred_ite))
    corr, pval = stats.pearsonr(pred_ite.flatten(), true_ite.flatten())
    metrics['ite_corr'] = float(corr)
    metrics['ite_corr_pval'] = float(pval)
    metrics['ate_bias'] = float(abs(np.mean(pred_ite) - np.mean(true_ite)))
    metrics['ate_pred'] = float(np.mean(pred_ite))
    metrics['ate_true'] = float(np.mean(true_ite))

    # Propensity metrics
    try:
        metrics['propensity_auroc'] = float(roc_auc_score(true_treatment, pred_propensity))
    except ValueError:
        metrics['propensity_auroc'] = np.nan

    # Outcome metrics (MSE against ground truth probabilities)
    metrics['y0_mse'] = float(mean_squared_error(true_y0, pred_y0))
    metrics['y1_mse'] = float(mean_squared_error(true_y1, pred_y1))

    # Outcome AUROC on factual outcomes
    untreated_mask = true_treatment == 0
    if untreated_mask.sum() > 0:
        try:
            metrics['y0_auroc'] = float(roc_auc_score(
                true_outcome[untreated_mask],
                pred_y0[untreated_mask]
            ))
        except ValueError:
            metrics['y0_auroc'] = np.nan
    else:
        metrics['y0_auroc'] = np.nan

    treated_mask = true_treatment == 1
    if treated_mask.sum() > 0:
        try:
            metrics['y1_auroc'] = float(roc_auc_score(
                true_outcome[treated_mask],
                pred_y1[treated_mask]
            ))
        except ValueError:
            metrics['y1_auroc'] = np.nan
    else:
        metrics['y1_auroc'] = np.nan

    return metrics


def run_single_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    condition: ExperimentCondition,
    device: torch.device
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Run a single fold of matched pair experiment.

    Uses full training fold (80%) for both propensity and outcome model training.
    No inner validation split - trains for fixed epochs on full training data.
    """
    logger.info(f"  Fold {fold + 1}: Starting on {device}")

    train_df = dataset.iloc[train_idx].reset_index(drop=True)
    test_df = dataset.iloc[test_idx].copy()

    text_col = condition.text_column

    # Create MatchedPairConfig
    mp_config = MatchedPairConfig(
        text_column=text_col,
        treatment_column='treatment_indicator',
        outcome_column='outcome_indicator',
        propensity_epochs=condition.propensity_epochs,
        propensity_lr=condition.propensity_lr,
        propensity_batch_size=condition.batch_size,
        propensity_early_stopping_patience=10,
        hier_transformer_sentence_model="prajjwal1/bert-tiny",
        hier_transformer_freeze_sentence_encoder=False,
        hier_transformer_max_sentences=100,
        hier_transformer_max_sentence_length=128,
        hier_transformer_num_layers=2,
        hier_transformer_num_heads=4,
        hier_transformer_dim=256,
        representation_dim=condition.representation_dim,
        matching_method=condition.matching_method,
        caliper=condition.caliper,
        caliper_scale="std",
        matching_algorithm=condition.matching_algorithm,
        outcome_epochs=condition.outcome_epochs,
        outcome_lr=condition.outcome_lr,
        outcome_batch_size=condition.batch_size,
        hidden_outcome_dim=condition.hidden_outcome_dim,
        dropout=0.0,
        alpha_outcome=1.0,
        beta_tau=1.0,
    )

    # Stage 1: Train propensity model
    logger.info(f"  Fold {fold + 1}: Training propensity model")
    propensity_model = PropensityMatchingModel(
        sentence_model=mp_config.hier_transformer_sentence_model,
        freeze_sentence_encoder=mp_config.hier_transformer_freeze_sentence_encoder,
        max_sentences=mp_config.hier_transformer_max_sentences,
        max_sentence_length=mp_config.hier_transformer_max_sentence_length,
        transformer_dim=mp_config.hier_transformer_dim,
        num_transformer_layers=mp_config.hier_transformer_num_layers,
        num_attention_heads=mp_config.hier_transformer_num_heads,
        representation_dim=mp_config.representation_dim,
        device=str(device)
    ).to(device)

    propensity_model.fit_tokenizer(train_df[text_col].tolist())

    # Train on full training fold (no validation split, fixed epochs)
    propensity_model, prop_history = train_propensity_model(
        propensity_model, train_df, None, mp_config, device
    )

    # Stage 2: Extract representations and match
    logger.info(f"  Fold {fold + 1}: Extracting representations and matching")
    train_texts = train_df[text_col].tolist()

    propensity_model.eval()
    with torch.no_grad():
        train_repr = extract_all_representations(
            propensity_model, train_texts,
            batch_size=mp_config.propensity_batch_size,
            device=device
        )
        train_propensity = extract_propensity_scores(
            propensity_model, train_texts,
            batch_size=mp_config.propensity_batch_size,
            device=device
        )

    treatment = train_df['treatment_indicator'].values

    if condition.matching_method == "embedding":
        match_result = match_by_cosine_similarity(
            train_repr.numpy(), treatment,
            caliper=condition.caliper,
            method=condition.matching_algorithm
        )
    else:
        matcher = PropensityMatcher(
            method=condition.matching_algorithm,
            caliper=condition.caliper,
            caliper_scale=mp_config.caliper_scale
        )
        match_result = matcher.match(train_propensity, treatment)

    match_stats = {
        'fold': fold + 1,
        'n_train': len(train_df),
        'n_treated': int(match_result.n_treated),
        'n_control': int(match_result.n_control),
        'n_matched': len(match_result.matched_pairs),
        'match_rate': len(match_result.matched_pairs) / min(match_result.n_treated, match_result.n_control)
            if min(match_result.n_treated, match_result.n_control) > 0 else 0.0,
        'mean_distance': float(match_result.distances.mean()) if len(match_result.distances) > 0 else None
    }
    logger.info(f"  Fold {fold + 1}: Matched {match_stats['n_matched']} pairs ({match_stats['match_rate']:.1%})")

    if len(match_result.matched_pairs) < 10:
        logger.warning(f"  Fold {fold + 1}: Very few matched pairs!")

    # Stage 3: Train outcome/tau model
    logger.info(f"  Fold {fold + 1}: Training outcome/tau model on {len(match_result.matched_pairs)} pairs")
    propensity_model.freeze_representation()

    outcome_model, outcome_history = train_matched_pair_outcome_model(
        propensity_model, train_df, match_result.matched_pairs,
        mp_config, device
    )

    # Stage 4: Predict on test set
    logger.info(f"  Fold {fold + 1}: Predicting on test set")
    test_texts = test_df[text_col].tolist()

    propensity_model.eval()
    outcome_model.eval()

    with torch.no_grad():
        test_repr = extract_all_representations(
            propensity_model, test_texts,
            batch_size=mp_config.propensity_batch_size,
            device=device
        )
        test_propensity = extract_propensity_scores(
            propensity_model, test_texts,
            batch_size=mp_config.propensity_batch_size,
            device=device
        )

        test_repr_device = test_repr.to(device)
        y0_prob, y1_prob, ite_prob = outcome_model.predict_potential_outcomes(test_repr_device)

    # Create predictions DataFrame
    preds_df = test_df.copy()
    preds_df['pred_propensity_prob'] = test_propensity
    preds_df['pred_y0_prob'] = y0_prob.cpu().numpy().flatten()
    preds_df['pred_y1_prob'] = y1_prob.cpu().numpy().flatten()
    preds_df['pred_ite_prob'] = ite_prob.cpu().numpy().flatten()
    preds_df['cv_fold'] = fold + 1

    # Compute metrics
    metrics = compute_metrics(
        pred_ite=preds_df['pred_ite_prob'].values,
        true_ite=preds_df['true_ite_prob'].values,
        pred_propensity=preds_df['pred_propensity_prob'].values,
        true_treatment=preds_df['treatment_indicator'].values,
        pred_y0=preds_df['pred_y0_prob'].values,
        pred_y1=preds_df['pred_y1_prob'].values,
        true_y0=preds_df['true_y0_prob'].values,
        true_y1=preds_df['true_y1_prob'].values,
        true_outcome=preds_df['outcome_indicator'].values
    )
    metrics['fold'] = fold + 1
    metrics.update(match_stats)

    logger.info(f"  Fold {fold + 1}: ITE corr={metrics['ite_corr']:.4f}, ATE bias={metrics['ate_bias']:.4f}")

    # Cleanup
    propensity_model.cpu()
    outcome_model.cpu()
    del propensity_model, outcome_model, train_repr, test_repr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return training history summary (using train metrics since no validation split)
    history_summary = {
        'propensity_final_train_loss': prop_history[-1]['train_loss'] if prop_history else None,
        'propensity_final_train_auroc': prop_history[-1].get('train_auroc') if prop_history else None,
        'outcome_final_loss': outcome_history[-1]['loss'] if outcome_history else None,
        'outcome_final_tau_loss': outcome_history[-1].get('tau_loss') if outcome_history else None,
    }

    return preds_df, metrics, history_summary


def run_experiment_condition(
    condition: ExperimentCondition,
    dataset: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    n_folds: int = 5
) -> Dict[str, Any]:
    """Run a single experimental condition with K-fold CV."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running condition: {condition.name}")
    logger.info(f"  Matching: {condition.matching_method} / {condition.matching_algorithm}")
    logger.info(f"  LR: {condition.propensity_lr}, Prop epochs: {condition.propensity_epochs}, Out epochs: {condition.outcome_epochs}")
    logger.info(f"  Device: {device}")
    logger.info(f"{'='*80}")

    condition_dir = output_dir / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Save condition config
    with open(condition_dir / "config.json", 'w') as f:
        json.dump(asdict(condition), f, indent=2)

    # K-fold CV
    dataset = dataset.reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_predictions = []
    all_metrics = []
    all_histories = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        preds_df, metrics, history = run_single_fold(
            fold, train_idx, test_idx, dataset, condition, device
        )
        all_predictions.append(preds_df)
        all_metrics.append(metrics)
        all_histories.append(history)

    # Combine predictions
    results_df = pd.concat(all_predictions).sort_index()
    results_df.to_parquet(condition_dir / "predictions.parquet")

    # Aggregate metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(condition_dir / "fold_metrics.csv", index=False)

    # Compute summary statistics
    summary = {
        'condition': condition.name,
        'matching_method': condition.matching_method,
        'matching_algorithm': condition.matching_algorithm,
        'propensity_lr': condition.propensity_lr,
        'propensity_epochs': condition.propensity_epochs,
        'outcome_epochs': condition.outcome_epochs,
    }

    for col in ['ite_mse', 'ite_mae', 'ite_corr', 'ate_bias', 'propensity_auroc',
                'y0_mse', 'y1_mse', 'y0_auroc', 'y1_auroc', 'match_rate']:
        if col in metrics_df.columns:
            values = metrics_df[col].dropna()
            summary[f'{col}_mean'] = float(values.mean()) if len(values) > 0 else np.nan
            summary[f'{col}_std'] = float(values.std()) if len(values) > 0 else np.nan

    summary['ate_pred_mean'] = float(metrics_df['ate_pred'].mean())
    summary['ate_true_mean'] = float(metrics_df['ate_true'].mean())

    with open(condition_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nCondition {condition.name} complete:")
    logger.info(f"  ITE corr: {summary['ite_corr_mean']:.4f} +/- {summary['ite_corr_std']:.4f}")
    logger.info(f"  ATE bias: {summary['ate_bias_mean']:.4f} +/- {summary['ate_bias_std']:.4f}")
    logger.info(f"  Match rate: {summary['match_rate_mean']:.1%} +/- {summary['match_rate_std']:.1%}")

    return summary


def run_condition_worker(args):
    """Worker function for parallel execution."""
    condition, dataset_path, output_dir, gpu_id, n_folds = args

    # Set CUDA device for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')

    # Load dataset in worker
    dataset = pd.read_parquet(dataset_path)

    return run_experiment_condition(condition, dataset, output_dir, device, n_folds)


def main():
    parser = argparse.ArgumentParser(
        description="Matched Pair ITE Estimation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset', '-d',
        required=True,
        help='Path to dataset parquet file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=[0],
        help='GPU IDs to use for parallel execution (default: [0])'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run minimal test configuration'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run conditions sequentially (no parallelization)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment grid
    conditions = generate_experiment_grid(quick_test=args.quick_test)
    logger.info(f"Generated {len(conditions)} experimental conditions")

    # Save experiment config
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'n_folds': args.n_folds,
            'gpu_ids': args.gpu_ids,
            'quick_test': args.quick_test,
            'n_conditions': len(conditions),
            'conditions': [asdict(c) for c in conditions]
        }, f, indent=2)

    # Load dataset for info
    dataset = pd.read_parquet(args.dataset)
    logger.info(f"Dataset: {len(dataset)} samples")
    logger.info(f"Treatment: {dataset['treatment_indicator'].sum()} treated, {(1 - dataset['treatment_indicator']).sum()} control")

    all_summaries = []

    if args.sequential or len(args.gpu_ids) == 1:
        # Sequential execution
        device = torch.device(f'cuda:{args.gpu_ids[0]}')
        for condition in tqdm(conditions, desc="Conditions"):
            summary = run_experiment_condition(
                condition, dataset, output_dir, device, args.n_folds
            )
            all_summaries.append(summary)
    else:
        # Parallel execution across GPUs
        logger.info(f"Running {len(conditions)} conditions in parallel across GPUs: {args.gpu_ids}")

        # Assign conditions to GPUs round-robin
        work_items = []
        for i, condition in enumerate(conditions):
            gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
            work_items.append((condition, args.dataset, output_dir, gpu_id, args.n_folds))

        # Use process pool for true parallelism (GPU isolation)
        # Note: Each worker will reload the dataset
        n_workers = min(len(args.gpu_ids), len(conditions))

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as executor:
            futures = {executor.submit(run_condition_worker, item): item[0].name
                       for item in work_items}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Conditions"):
                condition_name = futures[future]
                try:
                    summary = future.result()
                    all_summaries.append(summary)
                    logger.info(f"Completed: {condition_name}")
                except Exception as e:
                    logger.error(f"Failed: {condition_name} - {e}")
                    all_summaries.append({
                        'condition': condition_name,
                        'error': str(e)
                    })

    # Save all summaries
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / "all_results.csv", index=False)

    # Print final leaderboard
    logger.info("\n" + "=" * 100)
    logger.info("EXPERIMENT COMPLETE - LEADERBOARD")
    logger.info("=" * 100)

    if 'ite_corr_mean' in summary_df.columns:
        # Sort by ITE correlation (higher is better)
        leaderboard = summary_df.sort_values('ite_corr_mean', ascending=False)

        print("\nTop conditions by ITE correlation:")
        print(leaderboard[['condition', 'matching_method', 'matching_algorithm',
                          'ite_corr_mean', 'ite_corr_std', 'ate_bias_mean', 'match_rate_mean']].head(10).to_string())

        print("\n\nTop conditions by ATE bias (lower is better):")
        leaderboard_ate = summary_df.sort_values('ate_bias_mean', ascending=True)
        print(leaderboard_ate[['condition', 'matching_method', 'matching_algorithm',
                               'ate_bias_mean', 'ate_bias_std', 'ite_corr_mean']].head(10).to_string())

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
