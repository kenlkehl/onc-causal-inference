#!/usr/bin/env python
"""Multi-architecture experiment with Frozen LLM Pooler extractor.

Compares causal_forest, rlearner, and dragonnet model types using the
frozen_llm_pooler feature extractor across multiple datasets and max
sequence lengths.

Output is compatible with analyze_results.py.

Usage:
    # Run full grid on 4 GPUs (default)
    python oracle_experiment_scripts/run_frozen_llm_multi_architecture.py \
        --output-dir ../pcori_experiments/frozen_llm_multi_architecture

    # Run on specific GPUs
    python oracle_experiment_scripts/run_frozen_llm_multi_architecture.py \
        --output-dir ../pcori_experiments/frozen_llm_multi_architecture \
        --devices cuda:0 cuda:1

    # Run subset for testing
    python oracle_experiment_scripts/run_frozen_llm_multi_architecture.py \
        --output-dir ../pcori_experiments/frozen_llm_multi_architecture \
        --devices cuda:0 \
        --max-experiments 3 --epochs 3 --n-folds 2

    # Resume from checkpoint
    python oracle_experiment_scripts/run_frozen_llm_multi_architecture.py \
        --output-dir ../pcori_experiments/frozen_llm_multi_architecture \
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
from cdt.data import ClinicalTextDataset, collate_batch
from cdt.data import CachedHiddenStateDataset, collate_cached_batch, prepare_cached_batch
from cdt.models.causal_text import CausalText
from cdt.models.causal_text_forest import CausalTextForest
from cdt.models.hidden_state_cache import HiddenStateCache
from cdt.models.gpu_hidden_state_store import GPUHiddenStateStore

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

    # Frozen LLM Pooler hyperparameters
    flp_max_length: int = 10000
    flp_freeze_llm: bool = True
    flp_projection_dim: int = 128
    flp_gated_attention_dim: int = 128
    flp_random_projection_dim: Optional[int] = None  # None = full hidden size

    # Fixed parameters
    flp_model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    flp_dropout: float = 0.1
    flp_gradient_checkpointing: bool = True
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 1e-4
    n_folds: int = 5
    gamma_rlearner: float = 1.0
    beta_targreg: float = 0.1

    # Causal forest specific
    cf_n_estimators: int = 200
    cf_min_samples_leaf: int = 5

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


def _resolve_parquet_file(dataset_path: str) -> Optional[Path]:
    """Resolve the parquet file path for a dataset."""
    dp = Path(dataset_path)
    parquet_file = dp / "dataset_with_extraction.parquet"
    if not parquet_file.exists():
        parquet_file = dp / "dataset.parquet"
    if not parquet_file.exists():
        return None
    return parquet_file


def precompute_caches(
    configs: List[ExperimentConfig],
    devices: List[str],
) -> Dict[str, HiddenStateCache]:
    """Pre-compute hidden state caches for all unique (dataset, model, max_length) combos.

    Distributes precomputation across available GPUs. Returns a dict mapping
    cache_hash -> opened HiddenStateCache, ready for workers to read from.
    """
    unique_keys = {}
    for config in configs:
        if not config.flp_freeze_llm:
            continue
        parquet_file = _resolve_parquet_file(config.dataset_path)
        if parquet_file is None:
            continue
        rp_dim = config.flp_random_projection_dim
        cache_hash = HiddenStateCache.compute_cache_hash(
            config.flp_model_name, config.flp_max_length, str(parquet_file), rp_dim
        )
        if cache_hash not in unique_keys:
            unique_keys[cache_hash] = (
                parquet_file, config.flp_model_name,
                config.flp_max_length, config.batch_size,
                rp_dim,
            )

    if not unique_keys:
        logger.info("No hidden state caches to precompute")
        return {}

    logger.info(f"Found {len(unique_keys)} unique hidden state cache(s) to prepare")

    caches_to_compute = []
    ready_caches = {}

    for cache_hash, (parquet_file, model_name, max_length, batch_size, rp_dim) in unique_keys.items():
        cache_dir = str(parquet_file.parent / '.cdt_cache')
        cache = HiddenStateCache(
            cache_dir=cache_dir,
            model_name=model_name,
            max_length=max_length,
            dataset_path=str(parquet_file),
            random_projection_dim=rp_dim,
        )
        df = pd.read_parquet(parquet_file)
        if cache.is_valid(len(df)):
            logger.info(f"  Cache {cache_hash}: valid, reusing")
            cache.open()
            cache.preload_to_ram()
            ready_caches[cache_hash] = cache
        else:
            logger.info(f"  Cache {cache_hash}: needs precomputation "
                        f"({model_name}, max_len={max_length}, {len(df)} samples)")
            caches_to_compute.append((cache_hash, cache, parquet_file, batch_size))

    if not caches_to_compute:
        logger.info("All caches already valid")
        return ready_caches

    gpu_devices = [torch.device(d) for d in devices]
    logger.info(f"Pre-computing {len(caches_to_compute)} cache(s) sequentially, "
                f"each using {len(gpu_devices)} GPU(s)...")

    for i, (ch, cache, pf, bs) in enumerate(caches_to_compute):
        logger.info(f"  Cache {i+1}/{len(caches_to_compute)} ({ch}): "
                     f"precomputing on {len(gpu_devices)} GPU(s)...")
        df = pd.read_parquet(pf)
        all_texts = df['clinical_text'].tolist()
        cache.precompute_multi_gpu(all_texts, gpu_devices, batch_size=bs)
        cache.open()
        cache.preload_to_ram()
        ready_caches[ch] = cache
        logger.info(f"  Cache {ch}: precomputation complete")

    logger.info(f"All {len(ready_caches)} caches ready")
    return ready_caches


def precompute_gpu_stores(
    configs: List[ExperimentConfig],
    device: str,
    cache_registry: Optional[Dict[str, HiddenStateCache]] = None,
) -> Dict[str, GPUHiddenStateStore]:
    """Pre-compute GPU-resident hidden state stores for a single device."""
    unique_keys = {}
    for config in configs:
        if not config.flp_freeze_llm:
            continue
        parquet_file = _resolve_parquet_file(config.dataset_path)
        if parquet_file is None:
            continue
        rp_dim = config.flp_random_projection_dim
        cache_hash = HiddenStateCache.compute_cache_hash(
            config.flp_model_name, config.flp_max_length, str(parquet_file), rp_dim
        )
        if cache_hash not in unique_keys:
            unique_keys[cache_hash] = (
                parquet_file, config.flp_model_name,
                config.flp_max_length, config.batch_size,
            )

    if not unique_keys:
        return {}

    gpu_device = torch.device(device)
    stores = {}

    for cache_hash, (parquet_file, model_name, max_length, batch_size) in unique_keys.items():
        df = pd.read_parquet(parquet_file)
        all_texts = df['clinical_text'].tolist()

        estimated_gb = GPUHiddenStateStore.estimate_vram_gb(all_texts, model_name, max_length)
        free_vram_gb = torch.cuda.mem_get_info(gpu_device)[0] / 1e9

        if estimated_gb < free_vram_gb * 0.8:
            logger.info(f"  GPU store {cache_hash} on {device}: ~{estimated_gb:.1f} GB "
                        f"(free: {free_vram_gb:.1f} GB) — loading...")

            disk_cache = cache_registry.get(cache_hash) if cache_registry else None
            if disk_cache is not None:
                logger.info(f"  Loading GPU store from disk cache...")
                store = GPUHiddenStateStore()
                store.load_from_disk_cache(disk_cache, gpu_device)
                stores[cache_hash] = store
                logger.info(f"  GPU store {cache_hash}: loaded from disk, "
                            f"actual VRAM: {store.estimated_vram_gb:.2f} GB")
            else:
                store = GPUHiddenStateStore()
                store.precompute(all_texts, model_name, max_length, gpu_device, batch_size=batch_size)
                stores[cache_hash] = store
                logger.info(f"  GPU store {cache_hash}: done, "
                            f"actual VRAM: {store.estimated_vram_gb:.2f} GB")
        else:
            logger.warning(f"  GPU store {cache_hash} on {device}: needs ~{estimated_gb:.1f} GB "
                           f"but only {free_vram_gb:.1f} GB free — will use disk cache")

    return stores


def _create_datasets_and_loaders(
    train_df, test_df, train_idx, test_idx,
    text_column, confounder_cols, batch_size,
    hidden_state_cache, gpu_store,
):
    """Create train/test datasets and DataLoaders with appropriate caching."""
    use_cache = hidden_state_cache is not None

    if gpu_store is not None:
        train_dataset = CachedHiddenStateDataset(
            data=train_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            dataset_indices=np.array(train_idx),
            explicit_confounder_columns=confounder_cols,
        )
        test_dataset = CachedHiddenStateDataset(
            data=test_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            dataset_indices=np.array(test_idx),
            explicit_confounder_columns=confounder_cols,
        )
        collate_fn = collate_cached_batch
    elif use_cache:
        train_dataset = CachedHiddenStateDataset(
            data=train_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            dataset_indices=np.array(train_idx),
            explicit_confounder_columns=confounder_cols,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
        )
        test_dataset = CachedHiddenStateDataset(
            data=test_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            dataset_indices=np.array(test_idx),
            explicit_confounder_columns=confounder_cols,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
        )
        collate_fn = collate_cached_batch
    else:
        train_dataset = ClinicalTextDataset(
            data=train_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            explicit_confounder_columns=confounder_cols,
        )
        test_dataset = ClinicalTextDataset(
            data=test_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            explicit_confounder_columns=confounder_cols,
        )
        collate_fn = collate_batch

    if gpu_store is not None:
        dl_kwargs = dict(num_workers=0)
    elif use_cache:
        dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True)
    else:
        dl_kwargs = dict(num_workers=1, persistent_workers=True, pin_memory=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, **dl_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, **dl_kwargs
    )

    return train_dataset, test_dataset, train_loader, test_loader, collate_fn, dl_kwargs


def _get_cache_info(config, parquet_file, cache_registry, gpu_store_registry):
    """Look up GPU store or disk cache for a config."""
    gpu_store = None
    hidden_state_cache = None

    if config.flp_freeze_llm:
        cache_hash = HiddenStateCache.compute_cache_hash(
            config.flp_model_name, config.flp_max_length, str(parquet_file),
            config.flp_random_projection_dim,
        )
        if gpu_store_registry is not None:
            gpu_store = gpu_store_registry.get(cache_hash)
        if gpu_store is None and cache_registry is not None:
            hidden_state_cache = cache_registry.get(cache_hash)

    return gpu_store, hidden_state_cache


def _common_model_kwargs(config, gpu_store, hidden_state_cache, confounder_specs, device):
    """Build common model kwargs for frozen_llm_pooler extractor."""
    kwargs = dict(
        feature_extractor_type="frozen_llm_pooler",
        flp_model_name=config.flp_model_name,
        flp_max_length=config.flp_max_length,
        flp_freeze_llm=config.flp_freeze_llm,
        flp_gated_attention_dim=config.flp_gated_attention_dim,
        flp_projection_dim=config.flp_projection_dim,
        flp_dropout=config.flp_dropout,
        flp_gradient_checkpointing=config.flp_gradient_checkpointing,
        device=str(device),
    )

    # Enable cached mode (skip loading the LLM)
    if gpu_store is not None:
        kwargs['flp_skip_llm'] = True
        kwargs['flp_cached_hidden_size'] = gpu_store.hidden_size
    elif hidden_state_cache is not None:
        kwargs['flp_skip_llm'] = True
        kwargs['flp_cached_hidden_size'] = hidden_state_cache.hidden_size

    return kwargs


def run_causal_forest_experiment(
    config: ExperimentConfig,
    device: torch.device,
    df: pd.DataFrame,
    confounder_specs,
    confounder_cols,
    gpu_store,
    hidden_state_cache,
    cache_registry,
    gpu_store_registry,
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

        model_kwargs = _common_model_kwargs(config, gpu_store, hidden_state_cache, confounder_specs, device)
        model_kwargs.update(dict(
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
        ))

        model = CausalTextForest(**model_kwargs)

        train_dataset, test_dataset, train_loader, test_loader, collate_fn, dl_kwargs = \
            _create_datasets_and_loaders(
                train_df, test_df, train_idx, test_idx,
                text_column, confounder_cols, batch_size,
                hidden_state_cache, gpu_store,
            )

        if confounder_specs and train_dataset.explicit_confounder_values:
            model.fit_explicit_confounders(train_dataset.explicit_confounder_values)
            model.fit_explicit_confounder_featurizer(train_dataset.explicit_confounder_values)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate, weight_decay=0.01
        )
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
                prepare_cached_batch(batch, device, gpu_store=gpu_store)

                optimizer.zero_grad()
                losses = model.train_representation_step(
                    batch,
                    alpha_propensity=1.0,
                    gamma_rlearner=config.gamma_rlearner,
                )
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()
                train_loss += losses['loss'].item()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch['treatment'] = batch['treatment'].to(device)
                    batch['outcome'] = batch['outcome'].to(device)
                    prepare_cached_batch(batch, device, gpu_store=gpu_store)
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

        if gpu_store is not None:
            combined_indices = np.concatenate([train_idx, test_idx])
            combined_dataset = CachedHiddenStateDataset(
                data=combined_df, text_column=text_column,
                outcome_column='outcome_indicator', treatment_column='treatment_indicator',
                dataset_indices=combined_indices,
                explicit_confounder_columns=confounder_cols,
            )
            combined_collate = collate_cached_batch
        elif hidden_state_cache is not None:
            combined_indices = np.concatenate([train_idx, test_idx])
            combined_dataset = CachedHiddenStateDataset(
                data=combined_df, text_column=text_column,
                outcome_column='outcome_indicator', treatment_column='treatment_indicator',
                dataset_indices=combined_indices,
                explicit_confounder_columns=confounder_cols,
                cache_hidden_states=hidden_state_cache.hidden_states_array,
                cache_attention_masks=hidden_state_cache.attention_mask_array,
            )
            combined_collate = collate_cached_batch
        else:
            combined_dataset = ClinicalTextDataset(
                data=combined_df, text_column=text_column,
                outcome_column='outcome_indicator', treatment_column='treatment_indicator',
                explicit_confounder_columns=confounder_cols,
            )
            combined_collate = collate_batch

        combined_loader = DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=combined_collate, **dl_kwargs
        )

        model.train_causal_forest(combined_loader, combined_T, combined_Y, gpu_store=gpu_store)
        preds = model.predict(test_loader, return_ci=True, gpu_store=gpu_store)

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
    gpu_store,
    hidden_state_cache,
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

        model_kwargs = _common_model_kwargs(config, gpu_store, hidden_state_cache, confounder_specs, device)
        model_kwargs.update(dict(
            model_type=config.model_type,
            causal_head_representation_dim=128,
            causal_head_hidden_outcome_dim=64,
            causal_head_dropout=0.2,
            explicit_confounder_specs=confounder_specs,
        ))

        model = CausalText(**model_kwargs)

        train_dataset, test_dataset, train_loader, test_loader, collate_fn, dl_kwargs = \
            _create_datasets_and_loaders(
                train_df, test_df, train_idx, test_idx,
                text_column, confounder_cols, batch_size,
                hidden_state_cache, gpu_store,
            )

        if confounder_specs and train_dataset.explicit_confounder_values:
            model.fit_explicit_confounders(train_dataset.explicit_confounder_values)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate, weight_decay=0.01
        )
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
                prepare_cached_batch(batch, device, gpu_store=gpu_store)

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
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()
                train_loss += losses['loss'].item()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch['treatment'] = batch['treatment'].to(device)
                    batch['outcome'] = batch['outcome'].to(device)
                    prepare_cached_batch(batch, device, gpu_store=gpu_store)
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

        # Predict using DataLoader batches (supports cached hidden states)
        model.eval()
        all_y0 = []
        all_y1 = []
        all_prop = []
        all_ite = []

        with torch.no_grad():
            for batch in test_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)
                prepare_cached_batch(batch, device, gpu_store=gpu_store)

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
    cache_registry: Optional[Dict[str, HiddenStateCache]] = None,
    gpu_store_registry: Optional[Dict[str, GPUHiddenStateStore]] = None,
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
        df = _rename_confounder_columns(df, confounder_specs)
        confounder_cols = [f"explicit_conf_{s.name}" for s in confounder_specs]

    # Look up cache
    gpu_store, hidden_state_cache = _get_cache_info(
        config, parquet_file, cache_registry, gpu_store_registry
    )

    # Dispatch to model-specific runner
    if config.model_type == "causal_forest":
        result = run_causal_forest_experiment(
            config, device, df, confounder_specs, confounder_cols,
            gpu_store, hidden_state_cache, cache_registry, gpu_store_registry,
        )
    else:
        result = run_neural_experiment(
            config, device, df, confounder_specs, confounder_cols,
            gpu_store, hidden_state_cache,
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
    filter_max_lengths: Optional[List[int]] = None,
) -> List[ExperimentConfig]:
    """Generate all experiment configurations."""

    datasets = [
        ("example_synthetic_data_one_confounder_twostage", "one_confounder_twostage"),
        ("example_synthetic_data_ten_confounders_twostage", "ten_confounders_twostage"),
        #("../example_synthetic_data_ten_confounders_50K_twostage", "ten_confounders_50K_twostage"),
    ]

    model_types = ["causal_forest", "rlearner", "dragonnet"]
    max_lengths = [5000, 10000, 25000, 50000]
    explicit_confounder_options = [False, True]
    random_projection_dims = [None, 256, 512]

    if filter_datasets:
        datasets = [(p, n) for p, n in datasets if n in filter_datasets]
    if filter_model_types:
        model_types = [m for m in model_types if m in filter_model_types]
    if filter_max_lengths:
        max_lengths = [m for m in max_lengths if m in filter_max_lengths]

    configs = []

    for (dataset_path, dataset_name), model_type, max_len, explicit_conf, rp_dim in itertools.product(
        datasets, model_types, max_lengths, explicit_confounder_options, random_projection_dims
    ):
        configs.append(ExperimentConfig(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_type=model_type,
            use_explicit_confounders=explicit_conf,
            flp_max_length=max_len,
            flp_random_projection_dim=rp_dim,
        ))

    # Shuffle so patterns emerge early
    random.Random(42).shuffle(configs)

    return configs


def worker_thread(
    device: str,
    job_queue: queue.Queue,
    results_dict: Dict[str, Any],
    output_dir: Path,
    lock: threading.Lock,
    progress_bar: tqdm,
    cache_registry: Optional[Dict[str, HiddenStateCache]] = None,
    gpu_store_registry: Optional[Dict[str, GPUHiddenStateStore]] = None,
):
    """Worker thread to process experiments on a single GPU."""
    while True:
        try:
            config = job_queue.get(timeout=1)
        except queue.Empty:
            break

        config_hash = config.config_hash()

        try:
            result = run_single_experiment(config, device, output_dir, cache_registry, gpu_store_registry)

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


def main():
    parser = argparse.ArgumentParser(
        description="Multi-architecture experiment with Frozen LLM Pooler extractor"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="../pcori_experiments/frozen_llm_multi_architecture",
        help="Output directory for results"
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="GPU devices to use (default: cuda:0 cuda:1 cuda:2 cuda:3)"
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
        help="Filter datasets (one_confounder_twostage, ten_confounders_twostage, ten_confounders_50K_twostage)"
    )
    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        default=None,
        help="Filter model types (causal_forest, rlearner, dragonnet)"
    )
    parser.add_argument(
        "--max-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Filter max lengths (5000, 10000, 25000, 50000, 100000)"
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
    parser.add_argument(
        "--gpu-cache",
        action="store_true",
        help="Keep pre-computed hidden states in GPU VRAM instead of disk cache"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = generate_experiment_grid(
        filter_datasets=args.datasets,
        filter_model_types=args.model_types,
        filter_max_lengths=args.max_lengths,
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

    logger.info(f"Running {len(pending_configs)} experiments on {len(args.devices)} GPU(s) "
               f"(1 worker per GPU, {len(args.devices)} total workers)")

    if not pending_configs:
        logger.info("No experiments to run")
        return

    # Pre-compute disk caches (uses all GPUs in parallel)
    cache_registry = precompute_caches(pending_configs, args.devices)

    # Pre-compute GPU stores if requested
    gpu_store_registries = {}
    if args.gpu_cache:
        logger.info("GPU cache mode: loading hidden states to each device...")
        for device_str in args.devices:
            stores = precompute_gpu_stores(pending_configs, device_str, cache_registry)
            if stores:
                gpu_store_registries[device_str] = stores
                logger.info(f"  {device_str}: {len(stores)} GPU store(s) ready")

    # Create job queue and workers (1 worker per GPU)
    job_queue = queue.Queue()
    for config in pending_configs:
        job_queue.put(config)

    lock = threading.Lock()
    progress_bar = tqdm(total=len(pending_configs), desc="Experiments")

    threads = []
    for device in args.devices:
        device_gpu_stores = gpu_store_registries.get(device, {})
        t = threading.Thread(
            target=worker_thread,
            args=(device, job_queue, results_dict, output_dir, lock, progress_bar,
                  cache_registry, device_gpu_stores),
            name=f"worker-{device}"
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Free GPU stores
    for device_stores in gpu_store_registries.values():
        for store in device_stores.values():
            store.free()

    # Close all disk caches
    for cache in cache_registry.values():
        cache.close()

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
            ['dataset_name', 'model_type', 'flp_max_length', 'use_explicit_confounders']
        ).agg({
            'ite_corr': ['mean', 'std', 'max'],
            'ite_spearman_corr': ['mean', 'std', 'max'],
            'ate_bias': ['mean', 'std', 'min'],
            'propensity_auroc': ['mean', 'std'],
        })
        logger.info(f"\n{summary}")

        summary.to_csv(output_dir / "summary.csv")

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Total experiments: {len(results_dict)}, "
               f"Successful: {len(all_results)}, "
               f"Skipped/Failed: {len(results_dict) - len(all_results)}")


if __name__ == "__main__":
    main()
