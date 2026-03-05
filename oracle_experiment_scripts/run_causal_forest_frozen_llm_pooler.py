#!/usr/bin/env python
"""Causal forest grid experiment with Frozen LLM Pooler extractor.

Tests the frozen_llm_pooler feature extractor (pretrained LLM + gated attention
pooling) in the causal forest pipeline. Based on run_causal_forest_gru_pooler.py
but adapted for the LLM-based extractor which uses a pretrained tokenizer and
does not require fit_tokenizer().

Output is compatible with analyze_results.py.

Usage:
    # Run full grid with both GPUs
    python oracle_experiment_scripts/run_causal_forest_frozen_llm_pooler.py \
        --output-dir ../pcori_experiments/causal_text_forest_frozen_llm_pooler \
        --devices cuda:0 cuda:1 --workers-per-device 1

    # Run subset for testing
    python oracle_experiment_scripts/run_causal_forest_frozen_llm_pooler.py \
        --output-dir ../pcori_experiments/causal_text_forest_frozen_llm_pooler \
        --devices cuda:0 \
        --max-experiments 2 --epochs 3 --n-folds 2

    # Resume from checkpoint
    python oracle_experiment_scripts/run_causal_forest_frozen_llm_pooler.py \
        --output-dir ../pcori_experiments/causal_text_forest_frozen_llm_pooler \
        --devices cuda:0 cuda:1 --workers-per-device 1 \
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

    # R-learner mode: "none", "shared", "dual"
    rlearner_mode: str

    # Explicit confounders
    use_explicit_confounders: bool
    sampled_confounder_names: List[str] = field(default_factory=list)
    confounder_sample_seed: int = 0

    # Frozen LLM Pooler hyperparameters
    flp_freeze_llm: bool = True
    flp_projection_dim: int = 128
    flp_gated_attention_dim: int = 128
    flp_max_length: int = 200000 # int = 8192

    # Fixed parameters
    flp_model_name: str = "Qwen/Qwen3.5-0.8B-Base" #"Qwen/Qwen3-0.6B-Base"
    flp_dropout: float = 0.1
    flp_gradient_checkpointing: bool = True
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 1e-4
    n_folds: int = 5
    cf_n_estimators: int = 200
    cf_min_samples_leaf: int = 5
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

    # Confidence interval coverage
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

    Args:
        configs: All experiment configs (to discover unique dataset/model combos).
        devices: Available GPU devices.

    Returns:
        Dict mapping cache_hash -> HiddenStateCache (opened for reading).
    """
    # Collect unique cache keys: (dataset_path, model_name, max_length)
    unique_keys = {}  # cache_hash -> (parquet_file, model_name, max_length, batch_size)
    for config in configs:
        if not config.flp_freeze_llm:
            continue
        parquet_file = _resolve_parquet_file(config.dataset_path)
        if parquet_file is None:
            continue
        cache_hash = HiddenStateCache.compute_cache_hash(
            config.flp_model_name, config.flp_max_length, str(parquet_file)
        )
        if cache_hash not in unique_keys:
            unique_keys[cache_hash] = (
                parquet_file, config.flp_model_name,
                config.flp_max_length, config.batch_size,
            )

    if not unique_keys:
        logger.info("No hidden state caches to precompute")
        return {}

    logger.info(f"Found {len(unique_keys)} unique hidden state cache(s) to prepare")

    # Build HiddenStateCache objects and check validity
    caches_to_compute = []  # list of (cache_hash, cache_obj, parquet_file, batch_size)
    ready_caches = {}       # cache_hash -> HiddenStateCache

    for cache_hash, (parquet_file, model_name, max_length, batch_size) in unique_keys.items():
        cache_dir = str(parquet_file.parent / '.cdt_cache')
        cache = HiddenStateCache(
            cache_dir=cache_dir,
            model_name=model_name,
            max_length=max_length,
            dataset_path=str(parquet_file),
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

    # Distribute precomputation across GPUs (round-robin)
    # Since loading an LLM is expensive, we precompute one at a time per GPU
    logger.info(f"Pre-computing {len(caches_to_compute)} cache(s) across "
                f"{len(devices)} device(s)...")

    def _precompute_one(cache_hash, cache, parquet_file, batch_size, device):
        """Precompute a single cache on a given device."""
        df = pd.read_parquet(parquet_file)
        all_texts = df['clinical_text'].tolist()
        gpu_device = torch.device(device)
        cache.precompute(all_texts, gpu_device, batch_size=batch_size)
        cache.open()
        cache.preload_to_ram()
        return cache_hash, cache

    if len(caches_to_compute) == 1:
        # Single cache: precompute directly on first device
        ch, cache, pf, bs = caches_to_compute[0]
        _, cache = _precompute_one(ch, cache, pf, bs, devices[0])
        ready_caches[ch] = cache
    else:
        # Multiple caches: distribute across GPUs using threads
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(devices), len(caches_to_compute))
        ) as executor:
            futures = {}
            for i, (ch, cache, pf, bs) in enumerate(caches_to_compute):
                device = devices[i % len(devices)]
                fut = executor.submit(_precompute_one, ch, cache, pf, bs, device)
                futures[fut] = ch
            for fut in concurrent.futures.as_completed(futures):
                ch, cache = fut.result()
                ready_caches[ch] = cache
                logger.info(f"  Cache {ch}: precomputation complete")

    logger.info(f"All {len(ready_caches)} caches ready")
    return ready_caches


def precompute_gpu_stores(
    configs: List[ExperimentConfig],
    device: str,
) -> Dict[str, GPUHiddenStateStore]:
    """Pre-compute GPU-resident hidden state stores for a single device.

    Creates one GPUHiddenStateStore per unique (dataset, model, max_length)
    combo, with VRAM check. Returns empty dict for combos that don't fit.

    Args:
        configs: All experiment configs.
        device: GPU device string (e.g., "cuda:0").

    Returns:
        Dict mapping cache_hash -> GPUHiddenStateStore on this device.
    """
    unique_keys = {}  # cache_hash -> (parquet_file, model_name, max_length, batch_size)
    for config in configs:
        if not config.flp_freeze_llm:
            continue
        parquet_file = _resolve_parquet_file(config.dataset_path)
        if parquet_file is None:
            continue
        cache_hash = HiddenStateCache.compute_cache_hash(
            config.flp_model_name, config.flp_max_length, str(parquet_file)
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
                        f"(free: {free_vram_gb:.1f} GB) — precomputing...")
            store = GPUHiddenStateStore()
            store.precompute(all_texts, model_name, max_length, gpu_device, batch_size=batch_size)
            stores[cache_hash] = store
            logger.info(f"  GPU store {cache_hash}: done, "
                        f"actual VRAM: {store.estimated_vram_gb:.2f} GB")
        else:
            logger.warning(f"  GPU store {cache_hash} on {device}: needs ~{estimated_gb:.1f} GB "
                           f"but only {free_vram_gb:.1f} GB free — will use disk cache")

    return stores


def run_single_experiment(
    config: ExperimentConfig,
    device: str,
    output_dir: Path,
    cache_registry: Optional[Dict[str, HiddenStateCache]] = None,
    gpu_store_registry: Optional[Dict[str, GPUHiddenStateStore]] = None,
) -> Dict[str, Any]:
    """Run a single experiment configuration with K-fold CV.

    Args:
        config: Experiment configuration.
        device: GPU device string (e.g., "cuda:0").
        output_dir: Output directory for results.
        cache_registry: Dict mapping cache_hash -> pre-computed HiddenStateCache.
            Populated by precompute_caches() in main() before workers start.
        gpu_store_registry: Dict mapping cache_hash -> GPUHiddenStateStore on this device.
    """
    device = torch.device(device)

    # Always use dataset_with_extraction.parquet (has all columns including llm_extracted_*)
    parquet_file = _resolve_parquet_file(config.dataset_path)
    if parquet_file is None:
        return {'error': f"Dataset not found in {config.dataset_path}", 'skipped': True}

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

    batch_size = config.batch_size

    # Rename confounder columns once on the full dataframe (before any fold splitting)
    confounder_cols = None
    if confounder_specs:
        df = _rename_confounder_columns(df, confounder_specs)
        confounder_cols = [f"explicit_conf_{s.name}" for s in confounder_specs]

    # K-fold cross-validation
    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)

    # --- Look up GPU store or disk cache ---
    gpu_store = None
    hidden_state_cache = None

    if config.flp_freeze_llm:
        cache_hash = HiddenStateCache.compute_cache_hash(
            config.flp_model_name, config.flp_max_length, str(parquet_file)
        )
        # Prefer GPU store over disk cache
        if gpu_store_registry is not None:
            gpu_store = gpu_store_registry.get(cache_hash)
        if gpu_store is None and cache_registry is not None:
            hidden_state_cache = cache_registry.get(cache_hash)

    use_cache = hidden_state_cache is not None

    all_predictions = []
    fold_histories = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Create model (skip LLM when using cache)
        model_kwargs = dict(
            feature_extractor_type="frozen_llm_pooler",
            flp_model_name=config.flp_model_name,
            flp_max_length=config.flp_max_length,
            flp_freeze_llm=config.flp_freeze_llm,
            flp_gated_attention_dim=config.flp_gated_attention_dim,
            flp_projection_dim=config.flp_projection_dim,
            flp_dropout=config.flp_dropout,
            flp_gradient_checkpointing=config.flp_gradient_checkpointing,
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
            device=str(device),
        )
        if gpu_store is not None:
            model_kwargs['flp_skip_llm'] = True
            model_kwargs['flp_cached_hidden_size'] = gpu_store.hidden_size
        elif use_cache and hidden_state_cache is not None:
            model_kwargs['flp_skip_llm'] = True
            model_kwargs['flp_cached_hidden_size'] = hidden_state_cache.hidden_size

        model = CausalTextForest(**model_kwargs)

        # No fit_tokenizer needed - uses pretrained HF tokenizer

        # Create datasets: GPU store > disk cache > no cache
        if gpu_store is not None:
            # GPU store: cache_index mode (no inline loading, indices only)
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
        elif use_cache and hidden_state_cache is not None:
            # Disk cache: pass cache arrays for DataLoader workers
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

        # Fit confounder normalization stats from dataset's parsed values
        if confounder_specs and train_dataset.explicit_confounder_values:
            model.fit_explicit_confounders(train_dataset.explicit_confounder_values)
            model.fit_explicit_confounder_featurizer(train_dataset.explicit_confounder_values)

        # DataLoader config: GPU store requires num_workers=0 (GPU tensors not fork-safe)
        if gpu_store is not None:
            dl_kwargs = dict(num_workers=0)
        elif use_cache and hidden_state_cache is not None:
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

        # Training
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        best_val_loss = float('inf')
        best_state = None
        history = []

        effective_gamma = config.gamma_rlearner if use_rlearner else 0.0

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
                    gamma_rlearner=effective_gamma,
                )
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
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
                    prepare_cached_batch(batch, device, gpu_store=gpu_store)

                    losses = model.train_representation_step(
                        batch,
                        alpha_propensity=1.0,
                        gamma_rlearner=effective_gamma,
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
        elif use_cache and hidden_state_cache is not None:
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

        # Hidden states are already in DataLoader batches (loaded by workers).
        # train_causal_forest/predict handle prepare_cached_batch internally
        # via _get_extractor_input which checks for 'cached_hidden_states'.
        model.train_causal_forest(combined_loader, combined_T, combined_Y, gpu_store=gpu_store)
        preds = model.predict(test_loader, return_ci=True, gpu_store=gpu_store)

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

    # Note: cache is shared and managed by main(), not closed here

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
        #("example_synthetic_data_one_confounder", "one_confounder"),
        #("example_synthetic_data_ten_confounders", "ten_confounders"),
        ("example_synthetic_data_one_confounder_twostage", "one_confounder_twostage"),
        ("example_synthetic_data_ten_confounders_twostage", "ten_confounders_twostage")
       
        #("../example_synthetic_data_ten_confounders_50K_rows", "ten_confounders_50K"),
    ]

    if filter_datasets:
        datasets = [(p, n) for p, n in datasets if n in filter_datasets]

    rlearner_modes = ["shared"]
    if filter_rlearner_modes:
        rlearner_modes = [m for m in rlearner_modes if m in filter_rlearner_modes]

    explicit_confounder_options = [False, True]

    # Hyperparameter grid
    freeze_llm_options = [True]
    projection_dim_options = [64, 128, 256]
    gated_attention_dim_options = [64, 128]
    max_length_options = [15000]

    # Pre-load confounder specs for each dataset
    dataset_specs = {}
    for dataset_path, dataset_name in datasets:
        specs = load_confounder_specs_from_metadata(dataset_path)
        dataset_specs[dataset_name] = specs
        logger.info(f"Dataset '{dataset_name}': {len(specs)} confounders available "
                   f"({[s.name for s in specs]})")

    configs = []
    sample_counter = 0

    for (dataset_path, dataset_name), rlearner_mode, explicit_conf in itertools.product(
        datasets, rlearner_modes, explicit_confounder_options
    ):
        for freeze_llm, proj_dim, attn_dim, max_len in itertools.product(
            freeze_llm_options, projection_dim_options,
            gated_attention_dim_options, max_length_options
        ):
            sampled_names = []
            sample_seed = 0

            if explicit_conf:
                all_specs = dataset_specs.get(dataset_name, [])
                if not all_specs:
                    continue

                seed_str = f"{dataset_name}_{rlearner_mode}_{freeze_llm}_{proj_dim}_{attn_dim}_{max_len}_{sample_counter}"
                sample_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
                rng = random.Random(sample_seed)

                n_available = len(all_specs)
                k = rng.randint(1, n_available)
                sampled = rng.sample(all_specs, k)
                sampled_names = sorted([s.name for s in sampled])

                sample_counter += 1

            configs.append(ExperimentConfig(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                rlearner_mode=rlearner_mode,
                use_explicit_confounders=explicit_conf,
                sampled_confounder_names=sampled_names,
                confounder_sample_seed=sample_seed,
                flp_freeze_llm=freeze_llm,
                flp_projection_dim=proj_dim,
                flp_gated_attention_dim=attn_dim,
                flp_max_length=max_len,
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
                    progress_bar.set_postfix_str(
                        f"ITE corr: {metrics.get('ite_corr', 'N/A'):.3f}{conf_info}"
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
        description="Causal forest grid experiment with Frozen LLM Pooler extractor"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="../pcori_experiments/causal_text_forest_frozen_llm_pooler",
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
        default=1,
        help="Number of concurrent experiments per GPU device (default: 1, LLM is memory-intensive)"
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
    parser.add_argument(
        "--gpu-cache",
        action="store_true",
        help="Keep pre-computed hidden states in GPU VRAM instead of disk cache "
             "(auto-fallback to disk if insufficient VRAM)"
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

    # Pre-compute GPU stores per device if --gpu-cache is set.
    # Falls back to disk cache for combos that don't fit in VRAM.
    gpu_store_registries = {}  # device_str -> Dict[cache_hash -> GPUHiddenStateStore]
    if args.gpu_cache:
        logger.info("GPU cache mode: pre-computing hidden states on each device...")
        for device_str in args.devices:
            stores = precompute_gpu_stores(pending_configs, device_str)
            if stores:
                gpu_store_registries[device_str] = stores
                logger.info(f"  {device_str}: {len(stores)} GPU store(s) ready")

    # Pre-compute disk caches as fallback (reuses existing caches if valid).
    cache_registry = precompute_caches(pending_configs, args.devices)

    # Create job queue
    job_queue = queue.Queue()
    for config in pending_configs:
        job_queue.put(config)

    # Create worker threads
    lock = threading.Lock()
    progress_bar = tqdm(total=len(pending_configs), desc="Experiments")

    threads = []
    for device in args.devices:
        device_gpu_stores = gpu_store_registries.get(device, {})
        for worker_idx in range(args.workers_per_device):
            t = threading.Thread(
                target=worker_thread,
                args=(device, job_queue, results_dict, output_dir, lock, progress_bar,
                      cache_registry, device_gpu_stores),
                name=f"worker-{device}-{worker_idx}"
            )
            t.start()
            threads.append(t)

    # Wait for all threads to complete
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
            if 'sampled_confounder_names' in row:
                row['num_sampled_confounders'] = len(row['sampled_confounder_names'])
                row['sampled_confounder_names'] = ','.join(row['sampled_confounder_names'])
            all_results.append(row)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / "all_results.csv", index=False)
        results_df.to_parquet(output_dir / "all_results.parquet", index=False)

        # Summary statistics
        summary = results_df.groupby(
            ['dataset_name', 'rlearner_mode',
             'use_explicit_confounders']
        ).agg({
            'ite_corr': ['mean', 'std', 'max'],
            'ite_spearman_corr': ['mean', 'std', 'max'],
            'ate_bias': ['mean', 'std', 'min'],
            'propensity_auroc': ['mean', 'std'],
        })
        logger.info(f"\n{summary}")

        # Save summary
        summary.to_csv(output_dir / "summary.csv")

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Total experiments: {len(results_dict)}, "
               f"Successful: {len(all_results)}, "
               f"Skipped/Failed: {len(results_dict) - len(all_results)}")


if __name__ == "__main__":
    main()
