#!/usr/bin/env python
"""Causal forest grid experiment with Conv1d-Transformer Hybrid extractor.

Tests the conv1d_transformer_hybrid feature extractor (dilated 1D convolutions +
stride-based downsampling + transformer over full document) in the causal forest
pipeline. Key difference from conv_pool: no chunking - processes full documents
up to 8192 tokens with learned downsampling.

Output is compatible with analyze_results.py.

Usage:
    # Run full grid with both GPUs
    python oracle_experiment_scripts/run_causal_forest_conv1d_transformer_hybrid.py \
        --output-dir ../pcori_experiments/causal_text_forest_c1d_hybrid \
        --devices cuda:2 cuda:3 --workers-per-device 25

    # Run subset for testing
    python oracle_experiment_scripts/run_causal_forest_conv1d_transformer_hybrid.py \
        --output-dir ../pcori_experiments/causal_text_forest_c1d_hybrid \
        --devices cuda:0 \
        --max-experiments 10

    # Resume from checkpoint
    python oracle_experiment_scripts/run_causal_forest_conv1d_transformer_hybrid.py \
        --output-dir ../pcori_experiments/causal_text_forest_c1d_hybrid \
        --devices cuda:0 cuda:1 --workers-per-device 20 \
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
import random
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

from cdt.config import ExplicitConfounderSpec
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

    # CLAM
    clam_enabled: bool

    # Explicit confounders
    use_explicit_confounders: bool
    sampled_confounder_names: List[str] = field(default_factory=list)
    confounder_sample_seed: int = 0

    # Conv1d-Transformer Hybrid hyperparameters (grid search)
    conv_dim: int = 256
    num_blocks: int = 4
    kernel_size: int = 3
    max_length: int = 8192
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_dim: int = 256
    projection_dim: int = 128

    # Fixed parameters
    embedding_dim: int = 128
    conv_dropout: float = 0.1
    pool_stride: int = 2
    gated_attention_dim: int = 128
    max_vocab: int = 50000
    min_word_freq: int = 2
    epochs: int = 30
    batch_size: int = 4
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


def build_confounder_values(df: pd.DataFrame, spec_names: List[str]) -> List[Dict[str, Any]]:
    """Build explicit_confounder_values list from dataframe columns."""
    values_list = []
    for _, row in df.iterrows():
        values = {}
        for name in spec_names:
            col = f"llm_extracted_{name}"
            val = row.get(col, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                values[name] = val
                values[f"{name}_missing"] = False
            else:
                values[name] = None
                values[f"{name}_missing"] = True
        values_list.append(values)
    return values_list


class TextDataset(Dataset):
    """Dataset for text + labels + optional explicit confounder values."""

    def __init__(
        self,
        texts: List[str],
        treatments: np.ndarray,
        outcomes: np.ndarray,
        confounder_values: Optional[List[Dict[str, Any]]] = None
    ):
        self.texts = texts
        self.treatments = torch.tensor(treatments, dtype=torch.float32)
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32)
        self.confounder_values = confounder_values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            'texts': self.texts[idx],
            'treatment': self.treatments[idx],
            'outcome': self.outcomes[idx]
        }
        if self.confounder_values is not None:
            item['explicit_confounder_values'] = self.confounder_values[idx]
        return item


def collate_text_batch(batch):
    """Collate function for text batches with optional confounder values."""
    texts = [b['texts'] for b in batch]
    treatments = torch.stack([b['treatment'] for b in batch])
    outcomes = torch.stack([b['outcome'] for b in batch])
    result = {
        'texts': texts,
        'treatment': treatments,
        'outcome': outcomes
    }
    if 'explicit_confounder_values' in batch[0]:
        result['explicit_confounder_values'] = [
            b['explicit_confounder_values'] for b in batch
        ]
    return result


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


def run_single_experiment(
    config: ExperimentConfig,
    device: str,
    output_dir: Path
) -> Dict[str, Any]:
    """Run a single experiment configuration with K-fold CV."""
    device = torch.device(device)

    # Always use dataset_with_extraction.parquet (has all columns including llm_extracted_*)
    dataset_path = Path(config.dataset_path)
    parquet_file = dataset_path / "dataset_with_extraction.parquet"
    if not parquet_file.exists():
        parquet_file = dataset_path / "dataset.parquet"
    if not parquet_file.exists():
        return {'error': f"Dataset not found: {parquet_file}", 'skipped': True}

    # Load dataset
    df = pd.read_parquet(parquet_file)

    # Always use clinical_text
    text_column = 'clinical_text'
    if text_column not in df.columns:
        return {'error': f"Text column '{text_column}' not found", 'skipped': True}

    # Build confounder specs and values if using explicit confounders
    confounder_specs = None
    if config.use_explicit_confounders and config.sampled_confounder_names:
        all_specs = load_confounder_specs_from_metadata(config.dataset_path)
        spec_by_name = {s.name: s for s in all_specs}
        confounder_specs = [
            spec_by_name[name] for name in config.sampled_confounder_names
            if name in spec_by_name
        ]
        if not confounder_specs:
            return {
                'error': f"No valid confounder specs found for {config.sampled_confounder_names}",
                'skipped': True
            }
        logger.info(f"Using {len(confounder_specs)} sampled confounders: "
                    f"{[s.name for s in confounder_specs]}")

    # Parse R-learner mode
    use_rlearner = config.rlearner_mode in ("shared", "dual")
    rlearner_dual = config.rlearner_mode == "dual"

    # Batch size (smaller due to full-length sequences)
    batch_size = config.batch_size

    # K-fold cross-validation
    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)

    all_predictions = []
    fold_histories = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Create model with conv1d_transformer_hybrid extractor
        model = CausalTextForest(
            feature_extractor_type="conv1d_transformer_hybrid",
            c1d_hybrid_embedding_dim=config.embedding_dim,
            c1d_hybrid_conv_dim=config.conv_dim,
            c1d_hybrid_kernel_size=config.kernel_size,
            c1d_hybrid_num_blocks=config.num_blocks,
            c1d_hybrid_conv_dropout=config.conv_dropout,
            c1d_hybrid_pool_stride=config.pool_stride,
            c1d_hybrid_max_length=config.max_length,
            c1d_hybrid_transformer_layers=config.transformer_layers,
            c1d_hybrid_transformer_heads=config.transformer_heads,
            c1d_hybrid_transformer_dim=config.transformer_dim,
            c1d_hybrid_transformer_dropout=0.1,
            c1d_hybrid_gated_attention_dim=config.gated_attention_dim,
            c1d_hybrid_projection_dim=config.projection_dim,
            c1d_hybrid_max_vocab=config.max_vocab,
            c1d_hybrid_min_word_freq=config.min_word_freq,
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
            explicit_confounder_specs=confounder_specs,
            clam_enabled=config.clam_enabled,
            clam_num_instances=config.clam_num_instances,
            clam_instance_hidden_dim=64,
            device=str(device)
        )

        # fit_tokenizer is required (learns vocabulary from scratch)
        train_texts = train_df[text_column].tolist()
        model.fit_tokenizer(train_texts)

        # Build confounder values if needed
        train_conf_values = None
        test_conf_values = None
        if confounder_specs:
            sampled_names = [s.name for s in confounder_specs]
            train_conf_values = build_confounder_values(train_df, sampled_names)
            test_conf_values = build_confounder_values(test_df, sampled_names)

            # Fit confounder normalization stats
            model.fit_explicit_confounders(train_conf_values)
            model.fit_explicit_confounder_featurizer(train_conf_values)

        # Create datasets
        train_dataset = TextDataset(
            texts=train_texts,
            treatments=train_df['treatment_indicator'].values,
            outcomes=train_df['outcome_indicator'].values,
            confounder_values=train_conf_values,
        )
        test_dataset = TextDataset(
            texts=test_df[text_column].tolist(),
            treatments=test_df['treatment_indicator'].values,
            outcomes=test_df['outcome_indicator'].values,
            confounder_values=test_conf_values,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_text_batch
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_text_batch
        )

        # Training with AdamW + linear LR decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

        # Linear decay schedule
        num_training_steps = config.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_training_steps
        )

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
                scheduler.step()
                train_loss += losses['loss'].item()

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
        combined_texts = combined_df[text_column].tolist()
        combined_T = combined_df['treatment_indicator'].values
        combined_Y = combined_df['outcome_indicator'].values

        combined_conf_values = None
        if confounder_specs:
            sampled_names = [s.name for s in confounder_specs]
            combined_conf_values = build_confounder_values(combined_df, sampled_names)

        model.train_causal_forest(
            combined_texts, combined_T, combined_Y,
            batch_size=batch_size,
            explicit_confounder_values=combined_conf_values
        )

        # Predictions on test set
        test_texts = test_df[text_column].tolist()
        preds = model.predict(
            test_texts, batch_size=batch_size, return_ci=True,
            explicit_confounder_values=test_conf_values
        )

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
        tau_lower=results_df['pred_tau_lower'].values if 'pred_tau_lower' in results_df.columns else None,
        tau_upper=results_df['pred_tau_upper'].values if 'pred_tau_upper' in results_df.columns else None
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
) -> List[ExperimentConfig]:
    """Generate all experiment configurations with shuffled order."""

    datasets = [
        ("example_synthetic_data_one_confounder", "one_confounder"),
        ("example_synthetic_data_ten_confounders", "ten_confounders"),
        ("../example_synthetic_data_ten_confounders_50K_rows", "ten_confounders_50K"),
    ]

    if filter_datasets:
        datasets = [(p, n) for p, n in datasets if n in filter_datasets]

    rlearner_modes = ["shared"]  # shared mode is consistently best
    if filter_rlearner_modes:
        rlearner_modes = [m for m in rlearner_modes if m in filter_rlearner_modes]

    clam_options = [False, True]
    explicit_confounder_options = [False, True]

    # Conv1d-Transformer Hybrid hyperparameter grid
    conv_dim_options = [128, 256]
    num_blocks_options = [3, 4]
    kernel_size_options = [3, 5]
    max_length_options = [4096, 8192]
    transformer_layers_options = [1, 2, 4]
    transformer_heads_options = [2, 4, 8]
    transformer_dim_options = [128, 256]
    projection_dim_options = [64, 128, 256]

    # Pre-load confounder specs for each dataset
    dataset_specs = {}
    for dataset_path, dataset_name in datasets:
        specs = load_confounder_specs_from_metadata(dataset_path)
        dataset_specs[dataset_name] = specs
        logger.info(f"Dataset '{dataset_name}': {len(specs)} confounders available "
                   f"({[s.name for s in specs]})")

    configs = []
    sample_counter = 0

    for (dataset_path, dataset_name), rlearner_mode, clam, explicit_conf in itertools.product(
        datasets, rlearner_modes, clam_options, explicit_confounder_options
    ):
        for c_dim, n_blocks, k_size, max_len, trans_layers, trans_heads, trans_dim, proj_dim in itertools.product(
            conv_dim_options, num_blocks_options, kernel_size_options,
            max_length_options, transformer_layers_options, transformer_heads_options,
            transformer_dim_options, projection_dim_options
        ):
            # Ensure transformer_dim is divisible by transformer_heads
            if trans_dim % trans_heads != 0:
                continue

            sampled_names = []
            sample_seed = 0

            if explicit_conf:
                all_specs = dataset_specs.get(dataset_name, [])
                if not all_specs:
                    continue

                # Deterministic seed from the other grid params
                seed_str = (f"{dataset_name}_{rlearner_mode}_{clam}_{c_dim}_"
                           f"{n_blocks}_{k_size}_{max_len}_{trans_layers}_{trans_heads}_"
                           f"{trans_dim}_{proj_dim}_{sample_counter}")
                sample_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
                rng = random.Random(sample_seed)

                # Sample random number of confounders (1 to N)
                n_available = len(all_specs)
                k = rng.randint(1, n_available)
                sampled = rng.sample(all_specs, k)
                sampled_names = sorted([s.name for s in sampled])

                sample_counter += 1

            configs.append(ExperimentConfig(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                rlearner_mode=rlearner_mode,
                clam_enabled=clam,
                use_explicit_confounders=explicit_conf,
                sampled_confounder_names=sampled_names,
                confounder_sample_seed=sample_seed,
                conv_dim=c_dim,
                num_blocks=n_blocks,
                kernel_size=k_size,
                max_length=max_len,
                transformer_layers=trans_layers,
                transformer_heads=trans_heads,
                transformer_dim=trans_dim,
                projection_dim=proj_dim,
            ))

    # Shuffle experiment order so patterns emerge early
    random.Random(42).shuffle(configs)

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
                    conf_info = ""
                    if config.sampled_confounder_names:
                        conf_info = f" conf={len(config.sampled_confounder_names)}"
                    arch_info = f"d{config.conv_dim}b{config.num_blocks}k{config.kernel_size}L{config.max_length}"
                    progress_bar.set_postfix_str(
                        f"ITE corr: {metrics.get('ite_corr', 'N/A'):.3f} "
                        f"[{arch_info}]{conf_info}"
                    )

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
        description="Causal forest grid experiment with Conv1d-Transformer Hybrid extractor"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="../pcori_experiments/causal_text_forest_c1d_hybrid",
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
        "--workers-per-device",
        type=int,
        default=5,
        help="Number of concurrent experiments per GPU device (default: 5)"
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
    )

    # Update epochs and folds from args
    for config in configs:
        config.epochs = args.epochs
        config.n_folds = args.n_folds

    logger.info(f"Generated {len(configs)} experiment configurations")

    # Log architecture distribution
    arch_counts = {}
    for c in configs:
        key = f"d{c.conv_dim}_b{c.num_blocks}_k{c.kernel_size}_L{c.max_length}"
        arch_counts[key] = arch_counts.get(key, 0) + 1
    logger.info(f"Architecture distribution: {arch_counts}")

    # Log confounder sampling summary
    conf_counts = {}
    for c in configs:
        if c.use_explicit_confounders:
            k = len(c.sampled_confounder_names)
            conf_counts[k] = conf_counts.get(k, 0) + 1
    if conf_counts:
        logger.info(f"Confounder sample size distribution: {dict(sorted(conf_counts.items()))}")

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

    total_workers = len(args.devices) * args.workers_per_device
    logger.info(f"Running {len(pending_configs)} experiments on {len(args.devices)} GPU(s) "
               f"with {args.workers_per_device} workers each ({total_workers} total workers)")

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
        for worker_idx in range(args.workers_per_device):
            t = threading.Thread(
                target=worker_thread,
                args=(device, job_queue, results_dict, output_dir, lock, progress_bar),
                name=f"worker-{device}-{worker_idx}"
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
            # Convert sampled_confounder_names list to string for CSV compatibility
            if 'sampled_confounder_names' in row:
                row['num_sampled_confounders'] = len(row['sampled_confounder_names'])
                row['sampled_confounder_names'] = ','.join(row['sampled_confounder_names'])
            all_results.append(row)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / "all_results.csv", index=False)
        results_df.to_parquet(output_dir / "all_results.parquet", index=False)

        # Summary statistics
        group_cols = [
            'dataset_name', 'rlearner_mode', 'clam_enabled',
            'use_explicit_confounders',
        ]
        # Add num_sampled_confounders if present
        if 'num_sampled_confounders' in results_df.columns:
            group_cols.append('num_sampled_confounders')

        summary = results_df.groupby(group_cols).agg({
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
                ['dataset_name', 'rlearner_mode', 'clam_enabled',
                 'use_explicit_confounders',
                 'conv_dim', 'num_blocks', 'kernel_size', 'max_length',
                 'transformer_layers', 'transformer_dim', 'projection_dim',
                 'ite_corr', 'ate_bias']
            ]
            logger.info(f"\nTop 5 configurations by ITE correlation:\n{best.to_string()}")

        # Print architecture comparison
        if 'conv_dim' in results_df.columns:
            arch_summary = results_df.groupby(['conv_dim', 'num_blocks', 'kernel_size', 'max_length']).agg({
                'ite_corr': ['mean', 'std', 'count'],
                'ate_bias': ['mean', 'std'],
            }).round(4)
            logger.info(f"\nArchitecture comparison (conv_dim x num_blocks x kernel_size x max_length):\n"
                       f"{arch_summary.to_string()}")
    else:
        logger.warning("No successful experiments completed")

    # Save experiment metadata
    metadata = {
        'total_configs': len(configs),
        'completed': len(results_dict),
        'successful': len(all_results) if all_results else 0,
        'devices': args.devices,
        'workers_per_device': args.workers_per_device,
        'epochs': args.epochs,
        'n_folds': args.n_folds,
        'text_column': 'clinical_text',
        'feature_extractor': 'conv1d_transformer_hybrid',
        'lr_scheduler': 'linear_decay',
        'architecture_distribution': arch_counts,
        'confounder_sample_distribution': conf_counts,
    }
    with open(output_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
