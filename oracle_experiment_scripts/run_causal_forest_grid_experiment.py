#!/usr/bin/env python
"""Comprehensive grid experiment for causal forest with GRU-Pool.

This script runs a full factorial experiment over:
- 3 datasets: one_confounder, ten_confounders, ten_confounders_50K
- 3 R-learner modes: none, shared, dual
- 3 text input types: patient_prompt, clinical_text, llm_structured_text
- 2 CLAM settings: enabled/disabled
- 2 explicit confounder settings: with/without
- Full GRU-Pool hyperparameter grid

Usage:
    # Run full grid with both GPUs
    python oracle_experiment_scripts/run_causal_forest_grid_experiment.py \
        --output-dir ../pcori_experiments/causal_forest_experiments_2-5-26 \
        --devices cuda:0 cuda:1

    # Run subset for testing
    python oracle_experiment_scripts/run_causal_forest_grid_experiment.py \
        --output-dir ../pcori_experiments/causal_forest_test \
        --devices cuda:0 \
        --max-experiments 10

    # Resume from checkpoint
    python oracle_experiment_scripts/run_causal_forest_grid_experiment.py \
        --output-dir ../pcori_experiments/causal_forest_experiments_2-5-26 \
        --devices cuda:0 cuda:1 \
        --resume
"""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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

from cdt.models.causal_text_forest import CausalTextForest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    # Dataset
    dataset_path: str
    dataset_name: str

    # R-learner mode: "none", "shared", "dual"
    rlearner_mode: str

    # Text input column
    text_column: str

    # CLAM
    clam_enabled: bool

    # Explicit confounders
    use_explicit_confounders: bool

    # GRU-Pool hyperparameters
    embedding_dim: int
    gru_hidden_dim: int
    gru_num_layers: int
    transformer_layers: int
    transformer_heads: int

    # Fixed parameters
    transformer_dim: int = 256
    gated_attention_dim: int = 128
    projection_dim: int = 128
    chunk_size: int = 128
    chunk_overlap: int = 32
    max_chunks: int = 100
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-4
    n_folds: int = 5
    cf_n_estimators: int = 200
    cf_min_samples_leaf: int = 5
    clam_instance_weight: float = 0.5
    clam_num_instances: int = 5
    gamma_rlearner: float = 1.0

    def config_hash(self) -> str:
        """Generate unique hash for this config."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


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
    except:
        metrics['ite_corr'] = np.nan
    try:
        metrics['ite_spearman_corr'] = float(stats.spearmanr(pred_ite, true_ite)[0])
    except:
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


def create_llm_structured_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create structured text from LLM extracted values."""
    df = df.copy()

    if 'llm_extracted_metastatic_sites' in df.columns:
        df['llm_structured_text'] = df['llm_extracted_metastatic_sites'].apply(
            lambda x: f"Number of metastatic sites: {x}. This patient has {x} sites of metastatic disease."
        )
    else:
        # Check for other extraction columns (ten_confounders format)
        extraction_cols = [c for c in df.columns if c.startswith('llm_extracted_')]
        if extraction_cols:
            def build_text(row):
                parts = []
                for col in extraction_cols:
                    name = col.replace('llm_extracted_', '').replace('_', ' ')
                    parts.append(f"{name}: {row[col]}")
                return ". ".join(parts)
            df['llm_structured_text'] = df.apply(build_text, axis=1)

    return df


def run_single_experiment(
    config: ExperimentConfig,
    device: str,
    output_dir: Path
) -> Dict[str, Any]:
    """Run a single experiment configuration with 5-fold CV."""
    device = torch.device(device)

    # Determine dataset file
    dataset_path = Path(config.dataset_path)
    if config.use_explicit_confounders or config.text_column == 'llm_structured_text':
        parquet_file = dataset_path / "dataset_with_extraction.parquet"
        if not parquet_file.exists():
            return {'error': f"Dataset with extraction not found: {parquet_file}", 'skipped': True}
    else:
        parquet_file = dataset_path / "dataset.parquet"
        if not parquet_file.exists():
            parquet_file = dataset_path / "dataset_with_extraction.parquet"

    if not parquet_file.exists():
        return {'error': f"Dataset not found: {parquet_file}", 'skipped': True}

    # Load dataset
    df = pd.read_parquet(parquet_file)

    # Create LLM structured text if needed
    if config.text_column == 'llm_structured_text':
        df = create_llm_structured_text(df)
        if 'llm_structured_text' not in df.columns:
            return {'error': 'Could not create llm_structured_text column', 'skipped': True}

    # Check text column exists
    if config.text_column not in df.columns:
        return {'error': f"Text column '{config.text_column}' not found", 'skipped': True}

    # Parse R-learner mode
    use_rlearner = config.rlearner_mode in ("shared", "dual")
    rlearner_dual = config.rlearner_mode == "dual"

    # Adjust batch size for large datasets
    batch_size = config.batch_size
    if len(df) > 10000:
        batch_size = max(4, batch_size // 2)

    # K-fold cross-validation
    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)

    all_predictions = []
    fold_histories = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Create model
        model = CausalTextForest(
            feature_extractor_type="gru_pool",
            gru_pool_embedding_dim=config.embedding_dim,
            gru_pool_gru_hidden_dim=config.gru_hidden_dim,
            gru_pool_gru_num_layers=config.gru_num_layers,
            gru_pool_gru_bidirectional=True,
            gru_pool_gru_dropout=0.1,
            gru_pool_max_chunks=config.max_chunks,
            gru_pool_chunk_size=config.chunk_size,
            gru_pool_chunk_overlap=config.chunk_overlap,
            gru_pool_transformer_layers=config.transformer_layers,
            gru_pool_transformer_heads=config.transformer_heads,
            gru_pool_transformer_dim=config.transformer_dim,
            gru_pool_gated_attention_dim=config.gated_attention_dim,
            gru_pool_projection_dim=config.projection_dim,
            gru_pool_max_vocab=50000,
            gru_pool_min_word_freq=2,
            representation_dim=128,
            hidden_dim=64,
            dropout=0.2,
            cf_n_estimators=config.cf_n_estimators,
            cf_min_samples_leaf=config.cf_min_samples_leaf,
            cf_honest=True,
            cf_inference=True,
            cf_use_rlearner_representation=use_rlearner,
            cf_gamma_rlearner=config.gamma_rlearner,
            cf_rlearner_dual_extractors=rlearner_dual,
            clam_enabled=config.clam_enabled,
            clam_num_instances=config.clam_num_instances,
            clam_instance_hidden_dim=64,
            device=str(device)
        )

        # Fit tokenizer
        train_texts = train_df[config.text_column].tolist()
        model.fit_tokenizer(train_texts)

        # Create datasets
        train_dataset = TextDataset(
            texts=train_texts,
            treatments=train_df['treatment_indicator'].values,
            outcomes=train_df['outcome_indicator'].values
        )
        test_dataset = TextDataset(
            texts=test_df[config.text_column].tolist(),
            treatments=test_df['treatment_indicator'].values,
            outcomes=test_df['outcome_indicator'].values
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_text_batch
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_text_batch
        )

        # Training
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        best_val_loss = float('inf')
        best_state = None
        history = []

        effective_gamma = config.gamma_rlearner if use_rlearner else 0.0
        effective_clam = config.clam_instance_weight if config.clam_enabled else 0.0

        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)

                optimizer.zero_grad()
                losses = model.train_representation_step(
                    batch,
                    alpha_propensity=1.0,
                    gamma_rlearner=effective_gamma,
                    clam_instance_weight=effective_clam
                )
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += losses['loss'].item()

            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch['treatment'] = batch['treatment'].to(device)
                    batch['outcome'] = batch['outcome'].to(device)
                    losses = model.train_representation_step(
                        batch,
                        alpha_propensity=1.0,
                        gamma_rlearner=effective_gamma,
                        clam_instance_weight=effective_clam
                    )
                    val_loss += losses['loss'].item()

            train_loss /= len(train_loader)
            val_loss /= len(test_loader)

            history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
            model.to(device)

        fold_histories.append(history)

        # Train causal forest on combined train + test (for this fold's features)
        combined_df = pd.concat([train_df, test_df])
        combined_texts = combined_df[config.text_column].tolist()
        combined_T = combined_df['treatment_indicator'].values
        combined_Y = combined_df['outcome_indicator'].values

        model.train_causal_forest(combined_texts, combined_T, combined_Y, batch_size=batch_size)

        # Predictions on test set
        test_texts = test_df[config.text_column].tolist()
        preds = model.predict(test_texts, batch_size=batch_size, return_ci=True)

        # Store predictions
        fold_preds = test_df.copy()
        fold_preds['pred_y0_prob'] = preds['pred_y0_prob']
        fold_preds['pred_y1_prob'] = preds['pred_y1_prob']
        fold_preds['pred_ite_prob'] = preds['pred_ite_prob']
        fold_preds['pred_propensity'] = preds['propensity_prob']
        fold_preds['pred_tau'] = preds['tau_pred']
        fold_preds['cv_fold'] = fold + 1
        if 'tau_lower' in preds:
            fold_preds['pred_tau_lower'] = preds['tau_lower']
            fold_preds['pred_tau_upper'] = preds['tau_upper']

        all_predictions.append(fold_preds)

        # Cleanup
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
        tau_lower=results_df.get('pred_tau_lower', pd.Series([None])).values if 'pred_tau_lower' in results_df.columns else None,
        tau_upper=results_df.get('pred_tau_upper', pd.Series([None])).values if 'pred_tau_upper' in results_df.columns else None
    )

    return {
        'config': asdict(config),
        'metrics': metrics,
        'n_samples': len(results_df),
        'skipped': False,
        'error': None
    }


def generate_experiment_grid(
    filter_datasets: Optional[List[str]] = None,
    filter_rlearner_modes: Optional[List[str]] = None,
    filter_text_columns: Optional[List[str]] = None
) -> List[ExperimentConfig]:
    """Generate all experiment configurations."""

    datasets = [
        ("example_synthetic_data_one_confounder", "one_confounder"),
        ("example_synthetic_data_ten_confounders", "ten_confounders"),
        ("../example_synthetic_data_ten_confounders_50K_rows", "ten_confounders_50K"),
    ]

    if filter_datasets:
        datasets = [(p, n) for p, n in datasets if n in filter_datasets]

    rlearner_modes = ["none", "shared", "dual"]
    if filter_rlearner_modes:
        rlearner_modes = [m for m in rlearner_modes if m in filter_rlearner_modes]

    text_columns = ["patient_prompt", "clinical_text", "llm_structured_text"]
    if filter_text_columns:
        text_columns = [t for t in text_columns if t in filter_text_columns]

    clam_options = [False, True]
    explicit_confounder_options = [False, True]

    # Hyperparameter grid
    embedding_dims = [64, 128, 256]
    gru_hidden_dims = [64, 128, 256]
    gru_num_layers_options = [1, 2]
    transformer_layers_options = [1, 2, 4]
    transformer_heads_options = [2, 4, 8]

    configs = []
    for (dataset_path, dataset_name), rlearner_mode, text_col, clam, explicit_conf in itertools.product(
        datasets, rlearner_modes, text_columns, clam_options, explicit_confounder_options
    ):
        for emb_dim, gru_hid, gru_layers, trans_layers, trans_heads in itertools.product(
            embedding_dims, gru_hidden_dims, gru_num_layers_options,
            transformer_layers_options, transformer_heads_options
        ):
            configs.append(ExperimentConfig(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                rlearner_mode=rlearner_mode,
                text_column=text_col,
                clam_enabled=clam,
                use_explicit_confounders=explicit_conf,
                embedding_dim=emb_dim,
                gru_hidden_dim=gru_hid,
                gru_num_layers=gru_layers,
                transformer_layers=trans_layers,
                transformer_heads=trans_heads,
            ))

    return configs


def worker_thread(
    device: str,
    job_queue: queue.Queue,
    results_dict: Dict[str, Any],
    output_dir: Path,
    lock: threading.Lock,
    progress_bar: tqdm
):
    """Worker thread to process experiments on a single GPU."""
    while True:
        try:
            config = job_queue.get(timeout=1)
        except queue.Empty:
            break

        config_hash = config.config_hash()

        try:
            result = run_single_experiment(config, device, output_dir)

            with lock:
                results_dict[config_hash] = result

                # Save individual result
                result_file = output_dir / "results" / f"{config_hash}.json"
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)

                progress_bar.update(1)
                if result.get('skipped'):
                    progress_bar.set_postfix_str(f"Skipped: {result.get('error', 'unknown')[:30]}")
                else:
                    metrics = result.get('metrics', {})
                    progress_bar.set_postfix_str(f"ITE corr: {metrics.get('ite_corr', 'N/A'):.3f}")

        except Exception as e:
            with lock:
                results_dict[config_hash] = {
                    'config': asdict(config),
                    'error': str(e),
                    'skipped': True
                }
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Error: {str(e)[:30]}")

        finally:
            job_queue.task_done()

        # Clear GPU memory between experiments
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive grid experiment for causal forest with GRU-Pool"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="../pcori_experiments/causal_forest_experiments",
        help="Output directory for results"
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0"],
        help="GPU devices to use (e.g., cuda:0 cuda:1)"
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
        help="Filter datasets (one_confounder, ten_confounders, ten_confounders_50K)"
    )
    parser.add_argument(
        "--rlearner-modes",
        type=str,
        nargs="+",
        default=None,
        help="Filter R-learner modes (none, shared, dual)"
    )
    parser.add_argument(
        "--text-columns",
        type=str,
        nargs="+",
        default=None,
        help="Filter text columns (patient_prompt, clinical_text, llm_structured_text)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
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
    configs = generate_experiment_grid(
        filter_datasets=args.datasets,
        filter_rlearner_modes=args.rlearner_modes,
        filter_text_columns=args.text_columns
    )

    # Update epochs and folds from args
    for config in configs:
        config.epochs = args.epochs
        config.n_folds = args.n_folds

    logger.info(f"Generated {len(configs)} experiment configurations")

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

    logger.info(f"Running {len(pending_configs)} experiments on {len(args.devices)} GPU(s)")

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
    for device in args.devices:
        t = threading.Thread(
            target=worker_thread,
            args=(device, job_queue, results_dict, output_dir, lock, progress_bar)
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
        summary = results_df.groupby(
            ['dataset_name', 'rlearner_mode', 'text_column', 'clam_enabled', 'use_explicit_confounders']
        ).agg({
            'ite_corr': ['mean', 'std', 'max'],
            'ite_spearman_corr': ['mean', 'std', 'max'],
            'ite_mse': ['mean', 'std', 'min'],
            'ate_bias': ['mean', 'std', 'min']
        }).round(4)

        summary.to_csv(output_dir / "summary_by_condition.csv")

        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"Total successful experiments: {len(all_results)}")
        logger.info(f"Total skipped: {len(results_dict) - len(all_results)}")

        # Print best configurations
        if 'ite_corr' in results_df.columns:
            best = results_df.nlargest(5, 'ite_corr')[
                ['dataset_name', 'rlearner_mode', 'text_column', 'clam_enabled',
                 'embedding_dim', 'transformer_layers', 'ite_corr', 'ate_bias']
            ]
            logger.info(f"\nTop 5 configurations by ITE correlation:\n{best.to_string()}")
    else:
        logger.warning("No successful experiments completed")

    # Save experiment metadata
    metadata = {
        'total_configs': len(configs),
        'completed': len(results_dict),
        'successful': len(all_results),
        'devices': args.devices,
        'epochs': args.epochs,
        'n_folds': args.n_folds,
    }
    with open(output_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
