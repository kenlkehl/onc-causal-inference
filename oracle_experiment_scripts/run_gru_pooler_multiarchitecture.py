#!/usr/bin/env python
"""Multi-architecture experiment with GRU Pool extractor.

Compares causal_forest, rlearner, and dragonnet model types using the
gru_pool feature extractor across multiple datasets and hyperparameters.

Tokenizers are pre-fitted once per (dataset, fold) and reused across all
experiments sharing that split, avoiding redundant vocabulary construction.

Output is compatible with analyze_results.py.

Usage:
    # Run full grid on 2 GPUs
    python oracle_experiment_scripts/run_gru_pooler_multiarchitecture.py \
        --output-dir ../pcori_experiments/gru_pool_multi_architecture \
        --devices cuda:0 cuda:1

    # Run subset for testing
    python oracle_experiment_scripts/run_gru_pooler_multiarchitecture.py \
        --output-dir ../pcori_experiments/gru_pool_multi_architecture \
        --devices cuda:0 \
        --max-experiments 1 --epochs 3 --n-folds 2

    # Resume from checkpoint
    python oracle_experiment_scripts/run_gru_pooler_multiarchitecture.py \
        --output-dir ../pcori_experiments/gru_pool_multi_architecture \
        --resume
"""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import queue
import random
import resource
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdt.config import ExplicitConfounderSpec
from cdt.data import ClinicalTextDataset, collate_batch, create_collator
from cdt.models.causal_text import CausalText
from cdt.models.causal_text_forest import CausalTextForest
from cdt.models.gru_pool_extractor import GRUPoolExtractor

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

    # Model type: "causal_forest", "rlearner", "dragonnet"
    model_type: str

    # Explicit confounders (use all when True)
    use_explicit_confounders: bool

    # GRU-Pool hyperparameters (gridded)
    embedding_dim: int = 128
    gru_hidden_dim: int = 128
    transformer_layers: int = 2

    # Fixed GRU-Pool parameters
    gru_num_layers: int = 1
    transformer_heads: int = 4
    transformer_dim: int = 256
    gated_attention_dim: int = 128
    projection_dim: int = 128
    chunk_size: int = 128
    chunk_overlap: int = 32
    max_chunks: int = 100

    # Training
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    n_folds: int = 5

    # Causal forest specific
    cf_n_estimators: int = 200
    cf_min_samples_leaf: int = 5

    # Loss weights
    gamma_rlearner: float = 1.0
    beta_targreg: float = 0.1

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


def _rename_confounder_columns(df: pd.DataFrame, confounder_specs: List[ExplicitConfounderSpec]) -> pd.DataFrame:
    """Rename llm_extracted_* columns to explicit_conf_* for ClinicalTextDataset."""
    rename_map = {}
    for s in confounder_specs:
        rename_map[f"llm_extracted_{s.name}"] = f"explicit_conf_{s.name}"
        src_miss = f"llm_extracted_{s.name}_missing"
        if src_miss in df.columns:
            rename_map[src_miss] = f"explicit_conf_{s.name}_missing"
    return df.rename(columns=rename_map)


def _resolve_parquet_file(dataset_path: str) -> Optional[Path]:
    """Resolve the parquet file path for a dataset."""
    dp = Path(dataset_path)
    parquet_file = dp / "dataset_with_extraction.parquet"
    if not parquet_file.exists():
        parquet_file = dp / "dataset.parquet"
    if not parquet_file.exists():
        return None
    return parquet_file


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

    # Confidence interval coverage (causal forest only)
    if tau_lower is not None and tau_upper is not None:
        coverage = np.mean((true_ite >= tau_lower) & (true_ite <= tau_upper))
        metrics['ci_coverage'] = float(coverage)
        metrics['mean_ci_width'] = float(np.mean(tau_upper - tau_lower))

    return metrics


def _fit_single_tokenizer(
    args: tuple,
) -> tuple:
    """Fit a single tokenizer for one (dataset, fold). Used by ThreadPoolExecutor."""
    dataset_name, fold, train_texts, max_vocab, min_word_freq = args

    extractor = GRUPoolExtractor(
        embedding_dim=64,  # Doesn't matter — only tokenizer state is kept
        max_vocab_size=max_vocab,
        min_word_freq=min_word_freq,
        device='cpu',
    )
    extractor.fit_tokenizer(train_texts)
    state = extractor.get_tokenizer_state()
    vocab_size = extractor.vocab_size
    del extractor

    logger.info(f"Pre-fitted tokenizer: {dataset_name} fold {fold} (vocab={vocab_size})")
    return (dataset_name, fold), state


def prefit_tokenizers(
    datasets: List[tuple],
    n_folds: int,
    max_vocab: int = 50000,
    min_word_freq: int = 2,
) -> Dict[tuple, Dict[str, Any]]:
    """Pre-fit tokenizers in parallel, once per (dataset, fold).

    Args:
        datasets: List of (dataset_path, dataset_name) tuples
        n_folds: Number of CV folds
        max_vocab: Maximum vocabulary size
        min_word_freq: Minimum word frequency for vocabulary inclusion

    Returns:
        Dict mapping (dataset_name, fold_index) -> tokenizer state dict
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Build list of (dataset_name, fold, train_texts, ...) jobs
    jobs = []
    for dataset_path, dataset_name in datasets:
        parquet_file = _resolve_parquet_file(dataset_path)
        if parquet_file is None:
            logger.warning(f"Dataset not found for {dataset_name}, skipping tokenizer pre-fit")
            continue

        df = pd.read_parquet(parquet_file)
        text_column = 'clinical_text'
        if text_column not in df.columns:
            logger.warning(f"Text column not found in {dataset_name}, skipping tokenizer pre-fit")
            continue

        df = df.reset_index(drop=True)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, _test_idx) in enumerate(kf.split(df)):
            train_texts = df.iloc[train_idx][text_column].tolist()
            jobs.append((dataset_name, fold, train_texts, max_vocab, min_word_freq))

    # Fit all tokenizers in parallel
    tokenizer_states = {}
    with ThreadPoolExecutor(max_workers=len(jobs) or 1) as executor:
        futures = {executor.submit(_fit_single_tokenizer, job): job for job in jobs}
        for future in as_completed(futures):
            key, state = future.result()
            tokenizer_states[key] = state

    logger.info(f"Pre-fitted {len(tokenizer_states)} tokenizers in parallel")
    return tokenizer_states


def _build_gru_pool_kwargs(config: ExperimentConfig) -> dict:
    """Build common GRU pool extractor kwargs from config."""
    return dict(
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
    )


def run_causal_forest_experiment(
    config: ExperimentConfig,
    device: torch.device,
    df: pd.DataFrame,
    confounder_specs,
    confounder_cols,
    tokenizer_states: Dict[tuple, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run a causal forest experiment with K-fold CV."""
    text_column = 'clinical_text'
    batch_size = config.batch_size

    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)

    all_predictions = []
    fold_histories = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        gru_kwargs = _build_gru_pool_kwargs(config)
        gru_kwargs.update(dict(
            representation_dim=128,
            hidden_dim=64,
            dropout=0.2,
            cf_n_estimators=config.cf_n_estimators,
            cf_min_samples_leaf=config.cf_min_samples_leaf,
            cf_honest=True,
            cf_inference=True,
            cf_use_rlearner_representation=True,
            cf_gamma_rlearner=config.gamma_rlearner,
            cf_rlearner_dual_extractors=False,
            explicit_confounder_specs=confounder_specs,
            device=str(device),
        ))

        model = CausalTextForest(**gru_kwargs)

        # Load pre-fitted tokenizer state
        tok_state = tokenizer_states.get((config.dataset_name, fold))
        if tok_state is not None:
            model.feature_extractor.load_tokenizer_state(tok_state)
        else:
            model.fit_tokenizer(train_df[text_column].tolist())

        # Rename confounder columns
        if confounder_specs:
            train_df = _rename_confounder_columns(train_df, confounder_specs)
            test_df = _rename_confounder_columns(test_df, confounder_specs)

        # Create datasets
        confounder_cols_local = [f"explicit_conf_{s.name}" for s in confounder_specs] if confounder_specs else None
        train_dataset = ClinicalTextDataset(
            data=train_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            explicit_confounder_columns=confounder_cols_local,
        )
        test_dataset = ClinicalTextDataset(
            data=test_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            explicit_confounder_columns=confounder_cols_local,
        )

        # Fit confounder normalization
        if confounder_specs and train_dataset.explicit_confounder_values:
            model.fit_explicit_confounders(train_dataset.explicit_confounder_values)
            model.fit_explicit_confounder_featurizer(train_dataset.explicit_confounder_values)

        # Create collator (uses fitted tokenizer vocab)
        collator = create_collator(model.feature_extractor)
        collate_fn = collator if collator is not None else collate_batch

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=2,
            persistent_workers=True, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2,
            persistent_workers=True, pin_memory=True
        )

        # Training
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        best_val_loss = float('inf')
        best_state = None
        history = []

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
                    gamma_rlearner=config.gamma_rlearner,
                )
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += losses['loss'].item()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch['treatment'] = batch['treatment'].to(device)
                    batch['outcome'] = batch['outcome'].to(device)
                    losses = model.train_representation_step(
                        batch,
                        alpha_propensity=1.0,
                        gamma_rlearner=config.gamma_rlearner,
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

        # Train causal forest on combined data
        combined_df = pd.concat([train_df, test_df])
        combined_T = combined_df['treatment_indicator'].values
        combined_Y = combined_df['outcome_indicator'].values

        combined_dataset = ClinicalTextDataset(
            data=combined_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            explicit_confounder_columns=confounder_cols_local,
        )
        combined_loader = DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2, pin_memory=True
        )

        model.train_causal_forest(combined_loader, combined_T, combined_Y)

        # Predictions on test set
        preds = model.predict(test_loader, return_ci=True)

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

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.concat(all_predictions).sort_index()

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

    return {'metrics': metrics, 'n_samples': len(results_df)}


def run_neural_experiment(
    config: ExperimentConfig,
    device: torch.device,
    df: pd.DataFrame,
    confounder_specs,
    confounder_cols,
    tokenizer_states: Dict[tuple, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run an rlearner or dragonnet experiment with K-fold CV."""
    text_column = 'clinical_text'
    batch_size = config.batch_size

    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)

    all_predictions = []
    fold_histories = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        gru_kwargs = _build_gru_pool_kwargs(config)
        gru_kwargs.update(dict(
            model_type=config.model_type,
            causal_head_representation_dim=128,
            causal_head_hidden_outcome_dim=64,
            causal_head_dropout=0.2,
            explicit_confounder_specs=confounder_specs,
            device=str(device),
        ))

        model = CausalText(**gru_kwargs)

        # Load pre-fitted tokenizer state
        tok_state = tokenizer_states.get((config.dataset_name, fold))
        if tok_state is not None:
            model.feature_extractor.load_tokenizer_state(tok_state)
        else:
            model.fit_tokenizer(train_df[text_column].tolist())

        # Rename confounder columns
        if confounder_specs:
            train_df = _rename_confounder_columns(train_df, confounder_specs)
            test_df = _rename_confounder_columns(test_df, confounder_specs)

        # Create datasets
        confounder_cols_local = [f"explicit_conf_{s.name}" for s in confounder_specs] if confounder_specs else None
        train_dataset = ClinicalTextDataset(
            data=train_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            explicit_confounder_columns=confounder_cols_local,
        )
        test_dataset = ClinicalTextDataset(
            data=test_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            explicit_confounder_columns=confounder_cols_local,
        )

        # Fit confounder normalization
        if confounder_specs and train_dataset.explicit_confounder_values:
            model.fit_explicit_confounders(train_dataset.explicit_confounder_values)

        # Create collator (uses fitted tokenizer vocab)
        collator = create_collator(model.feature_extractor)
        collate_fn = collator if collator is not None else collate_batch

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=2,
            persistent_workers=True, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2,
            persistent_workers=True, pin_memory=True
        )

        # Training
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        best_val_loss = float('inf')
        best_state = None
        history = []

        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)

                optimizer.zero_grad()
                if config.model_type == "rlearner":
                    losses = model.train_step(
                        batch,
                        alpha_propensity=1.0,
                        gamma_rlearner=config.gamma_rlearner,
                    )
                else:  # dragonnet
                    losses = model.train_step(
                        batch,
                        alpha_propensity=1.0,
                        beta_targreg=config.beta_targreg,
                    )
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += losses['loss'].item()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch['treatment'] = batch['treatment'].to(device)
                    batch['outcome'] = batch['outcome'].to(device)
                    if config.model_type == "rlearner":
                        losses = model.train_step(
                            batch,
                            alpha_propensity=1.0,
                            gamma_rlearner=config.gamma_rlearner,
                        )
                    else:
                        losses = model.train_step(
                            batch,
                            alpha_propensity=1.0,
                            beta_targreg=config.beta_targreg,
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

        # Predict using DataLoader batches
        model.eval()
        all_y0 = []
        all_y1 = []
        all_prop = []

        with torch.no_grad():
            for batch in test_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)

                preds = model.predict(batch)
                all_y0.append(preds['y0_prob'].cpu().numpy())
                all_y1.append(preds['y1_prob'].cpu().numpy())
                all_prop.append(preds['propensity'].cpu().numpy())

        pred_y0 = np.concatenate(all_y0)
        pred_y1 = np.concatenate(all_y1)
        pred_prop = np.concatenate(all_prop)
        pred_ite = pred_y1 - pred_y0

        fold_preds = test_df.copy()
        fold_preds['pred_y0_prob'] = pred_y0
        fold_preds['pred_y1_prob'] = pred_y1
        fold_preds['pred_ite_prob'] = pred_ite
        fold_preds['pred_propensity'] = pred_prop
        fold_preds['cv_fold'] = fold + 1

        all_predictions.append(fold_preds)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.concat(all_predictions).sort_index()

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
    )

    return {'metrics': metrics, 'n_samples': len(results_df)}


def run_single_experiment(
    config: ExperimentConfig,
    device: str,
    output_dir: Path,
    tokenizer_states: Dict[tuple, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    device = torch.device(device)

    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        return {'error': f"Dataset not found in {config.dataset_path}", 'skipped': True}

    df = pd.read_parquet(parquet_file)

    text_column = 'clinical_text'
    if text_column not in df.columns:
        return {'error': f"Text column '{text_column}' not found", 'skipped': True}

    # Build confounder specs if using explicit confounders
    confounder_specs = None
    confounder_cols = None
    if config.use_explicit_confounders:
        confounder_specs = load_confounder_specs_from_metadata(config.dataset_path)
        if not confounder_specs:
            return {
                'error': f"No confounder specs found in {config.dataset_path}",
                'skipped': True
            }
        logger.info(f"Using {len(confounder_specs)} explicit confounders: "
                    f"{[s.name for s in confounder_specs]}")
        confounder_cols = [f"explicit_conf_{s.name}" for s in confounder_specs]

    # Dispatch to model-specific runner
    if config.model_type == "causal_forest":
        result = run_causal_forest_experiment(
            config, device, df, confounder_specs, confounder_cols,
            tokenizer_states,
        )
    else:
        result = run_neural_experiment(
            config, device, df, confounder_specs, confounder_cols,
            tokenizer_states,
        )

    return {
        'config': asdict(config),
        'metrics': result['metrics'],
        'n_samples': result['n_samples'],
        'skipped': False,
        'error': None
    }


def generate_experiment_grid(
    filter_datasets: Optional[List[str]] = None,
    filter_model_types: Optional[List[str]] = None,
) -> tuple:
    """Generate all experiment configurations.

    Returns:
        (configs, datasets) tuple where datasets is the list of (path, name) tuples
    """

    datasets = [
        ("example_synthetic_data_one_confounder_twostage", "one_confounder_twostage"),
        ("example_synthetic_data_ten_confounders_twostage", "ten_confounders_twostage"),
    ]

    model_types = ["causal_forest", "rlearner", "dragonnet"]
    explicit_confounder_options = [False, True]

    # GRU hyperparameter grid
    embedding_dims = [64, 128, 256]
    gru_hidden_dims = [64, 128, 256]
    transformer_layers_options = [1, 2, 4]

    if filter_datasets:
        datasets = [(p, n) for p, n in datasets if n in filter_datasets]
    if filter_model_types:
        model_types = [m for m in model_types if m in filter_model_types]

    configs = []

    for (dataset_path, dataset_name), model_type, explicit_conf, emb_dim, gru_hid, trans_layers in itertools.product(
        datasets, model_types, explicit_confounder_options,
        embedding_dims, gru_hidden_dims, transformer_layers_options
    ):
        configs.append(ExperimentConfig(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_type=model_type,
            use_explicit_confounders=explicit_conf,
            embedding_dim=emb_dim,
            gru_hidden_dim=gru_hid,
            transformer_layers=trans_layers,
        ))

    # Shuffle so patterns emerge early
    random.Random(42).shuffle(configs)

    return configs, datasets


def worker_thread(
    device: str,
    job_queue: queue.Queue,
    results_dict: Dict[str, Any],
    output_dir: Path,
    lock: threading.Lock,
    progress_bar: tqdm,
    tokenizer_states: Dict[tuple, Dict[str, Any]],
):
    """Worker thread to process experiments on a single GPU."""
    while True:
        try:
            config = job_queue.get(timeout=1)
        except queue.Empty:
            break

        config_hash = config.config_hash()

        try:
            result = run_single_experiment(config, device, output_dir, tokenizer_states)

            with lock:
                results_dict[config_hash] = result

                result_file = output_dir / "results" / f"{config_hash}.json"
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)

                progress_bar.update(1)
                if result.get('skipped'):
                    progress_bar.set_postfix_str(f"Skipped: {result.get('error', 'unknown')[:30]}")
                else:
                    metrics = result.get('metrics', {})
                    progress_bar.set_postfix_str(
                        f"{config.model_type} ITE corr: {metrics.get('ite_corr', 'N/A'):.3f}"
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

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _raise_fd_limit(target: int = 65536) -> None:
    """Raise the soft file-descriptor limit to avoid OSError 24."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(target, hard)
    if new_soft > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        logger.info(f"Raised file descriptor limit: {soft} -> {new_soft} (hard={hard})")


def main():
    _raise_fd_limit()

    parser = argparse.ArgumentParser(
        description="Multi-architecture experiment with GRU Pool extractor"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="../pcori_experiments/gru_pool_multi_architecture",
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
        help="Filter datasets (one_confounder_twostage, ten_confounders_twostage)"
    )
    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        default=None,
        help="Filter model types (causal_forest, rlearner, dragonnet)"
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs, datasets = generate_experiment_grid(
        filter_datasets=args.datasets,
        filter_model_types=args.model_types,
    )

    for config in configs:
        config.epochs = args.epochs
        config.n_folds = args.n_folds

    logger.info(f"Generated {len(configs)} experiment configurations")

    # Log grid summary
    model_type_counts = {}
    for c in configs:
        model_type_counts[c.model_type] = model_type_counts.get(c.model_type, 0) + 1
    logger.info(f"Model type distribution: {model_type_counts}")

    # Pre-fit tokenizers for all (dataset, fold) combinations
    logger.info("Pre-fitting tokenizers...")
    tokenizer_states = prefit_tokenizers(datasets, n_folds=args.n_folds)

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
                args=(device, job_queue, results_dict, output_dir, lock, progress_bar,
                      tokenizer_states),
                name=f"worker-{device}-{worker_idx}"
            )
            t.start()
            threads.append(t)

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

        summary = results_df.groupby(
            ['dataset_name', 'model_type', 'use_explicit_confounders']
        ).agg({
            'ite_corr': ['mean', 'std', 'max'],
            'ite_spearman_corr': ['mean', 'std', 'max'],
            'ate_bias': ['mean', 'std', 'min'],
            'propensity_auroc': ['mean', 'std'],
        }).round(4)

        summary.to_csv(output_dir / "summary.csv")
        logger.info(f"\n{summary}")

        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"Total successful experiments: {len(all_results)}")
        logger.info(f"Total skipped: {len(results_dict) - len(all_results)}")

        # Print best configurations
        if 'ite_corr' in results_df.columns:
            best = results_df.nlargest(5, 'ite_corr')[
                ['dataset_name', 'model_type', 'use_explicit_confounders',
                 'embedding_dim', 'gru_hidden_dim', 'transformer_layers',
                 'ite_corr', 'ate_bias']
            ]
            logger.info(f"\nTop 5 configurations by ITE correlation:\n{best.to_string()}")
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
        'feature_extractor': 'gru_pool',
        'lr_scheduler': 'cosine_annealing',
    }
    with open(output_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
