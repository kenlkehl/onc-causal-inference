#!/usr/bin/env python
"""Oracle experiment runner for CDT.

Compares causal_forest, rlearner, dragonnet, and best_attainable model types
using the frozen_llm_pooler feature extractor across multiple datasets and
max sequence lengths.  Each experiment configuration is repeated N times
(--n-repeats, default 10) with different random seeds so that summary
statistics report mean +/- std across repeats.

The "best_attainable" experiment type uses ground-truth confounder values
(true_{name} columns) to train a CausalForestDML, providing an upper-bound
reference for the neural-network-based methods.

By default, uses live LLM forward pass per batch (no pre-caching).
Pass --cache to opt-in to hidden state pre-computation with frozen
downprojection for large-scale runs.

Output is compatible with analyze_results.py.

Usage:
    # Run full grid on 4 GPUs (default)
    python oracle_experiment_scripts/run_oracle_experiments.py \
        --datasets ../example_synthetic_data_ten_confounders_50K_twostage \
        --output-dir ../pcori_experiments/oracle_experiments

    # Multiple datasets
    python oracle_experiment_scripts/run_oracle_experiments.py \
        --datasets path/to/dataset1 path/to/dataset2 \
        --output-dir ../pcori_experiments/oracle_experiments

    # Run on specific GPUs
    python oracle_experiment_scripts/run_oracle_experiments.py \
        --datasets ../example_synthetic_data_ten_confounders_50K_twostage \
        --output-dir ../pcori_experiments/oracle_experiments \
        --devices cuda:0 cuda:1

    # Run subset for testing
    python oracle_experiment_scripts/run_oracle_experiments.py \
        --datasets ../example_synthetic_data_ten_confounders_50K_twostage \
        --output-dir ../pcori_experiments/oracle_experiments \
        --devices cuda:1 \
        --max-experiments 1 --epochs 3 --n-folds 5

    # Run with 5 repeats instead of default 10
    python oracle_experiment_scripts/run_oracle_experiments.py \
        --datasets ../example_synthetic_data_ten_confounders_50K_twostage \
        --output-dir ../pcori_experiments/oracle_experiments \
        --n-repeats 5

    # Resume from checkpoint
    python oracle_experiment_scripts/run_oracle_experiments.py \
        --datasets ../example_synthetic_data_ten_confounders_50K_twostage \
        --output-dir ../pcori_experiments/oracle_experiments \
        --resume

    # Run multiple experiments per GPU (cached mode only)
    python oracle_experiment_scripts/run_oracle_experiments.py \
        --datasets ../example_synthetic_data_ten_confounders_50K_twostage \
        --output-dir ../pcori_experiments/oracle_experiments \
        --cache --gpu-cache --workers-per-gpu auto

    # Fixed 4 workers per GPU
    python oracle_experiment_scripts/run_oracle_experiments.py \
        --datasets ../example_synthetic_data_ten_confounders_50K_twostage \
        --output-dir ../pcori_experiments/oracle_experiments \
        --cache --workers-per-gpu 4
"""

import argparse
import concurrent.futures
import gc
import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import threading
import traceback
from copy import deepcopy
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

from oci.config import ExplicitConfounderSpec
from oci.data import ClinicalTextDataset, collate_batch
from oci.data import CachedHiddenStateDataset, collate_cached_batch, prepare_cached_batch
from oci.models.causal_text import CausalText
from oci.models.causal_text_forest import CausalTextForest
from oci.models.hidden_state_cache import HiddenStateCache
from oci.models.gpu_hidden_state_store import GPUHiddenStateStore

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

    # Model type: "causal_forest", "rlearner", "dragonnet", "best_attainable"
    model_type: str

    # Explicit confounders (use all when True)
    use_explicit_confounders: bool

    # Feature extractor type
    feature_extractor_type: str = "frozen_llm_pooler"

    # Repeat index for N-repeat capability (different random seed per repeat)
    repeat_index: int = 0

    # Frozen LLM Pooler hyperparameters
    flp_max_length: int = 10000
    flp_freeze_llm: bool = True
    flp_projection_dim: int = 128
    flp_gated_attention_dim: int = 128
    flp_downprojection_dim: Optional[int] = 256  # Frozen downprojection dim applied during caching (None = full hidden size)
    flp_cache_hidden_states: bool = False  # If True, pre-cache hidden states to disk
    hlm_cache_hidden_states: bool = False  # If True, pre-cache hidden states to disk (hierarchical LLM)
    flp_chat_template_prompt: Optional[str] = None  # Chat template prompt for instruct models

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

    # Hierarchical LLM hyperparameters
    hlm_model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    hlm_chunk_size: int = 12000
    hlm_chunk_overlap: int = 256
    hlm_max_chunks: int = 16
    hlm_downprojection_dim: Optional[int] = 256
    hlm_freeze_llm: bool = True
    hlm_chat_template_prompt: Optional[str] = None

    # Hierarchical CNN hyperparameters
    hcnn_embedding_dim: int = 256
    hcnn_conv_dim: int = 256
    hcnn_kernel_size: int = 5
    hcnn_num_conv_blocks: int = 4
    hcnn_chunk_size: int = 12000
    hcnn_chunk_overlap: int = 64
    hcnn_max_chunks: int = 32
    hcnn_vocab_size: int = 50000
    hcnn_projection_dim: int = 128
    hcnn_dropout: float = 0.1

    # Hierarchical GRU hyperparameters
    hgru_embedding_dim: int = 256
    hgru_gru_hidden_dim: int = 256
    hgru_num_gru_layers: int = 2
    hgru_chunk_size: int = 12000
    hgru_chunk_overlap: int = 64
    hgru_max_chunks: int = 32
    hgru_vocab_size: int = 50000
    hgru_projection_dim: int = 128
    hgru_dropout: float = 0.1

    # Simple CNN hyperparameters
    scnn_embedding_dim: int = 256
    scnn_conv_dim: int = 256
    scnn_kernel_size: int = 5
    scnn_num_conv_blocks: int = 4
    scnn_max_length: int = 20000
    scnn_vocab_size: int = 50000
    scnn_projection_dim: int = 128
    scnn_dropout: float = 0.1

    # Extractor-specific field prefixes -- used by config_hash() to exclude
    # parameters that belong to *other* extractors so adding a new extractor
    # type never invalidates existing result hashes.
    _EXTRACTOR_PREFIXES = {
        "frozen_llm_pooler": {"flp_"},
        "hierarchical_llm": {"hlm_"},
        "hierarchical_cnn": {"hcnn_"},
        "hierarchical_gru": {"hgru_"},
        "simple_cnn": {"scnn_"},
    }
    _ALL_EXTRACTOR_PREFIXES = set().union(*_EXTRACTOR_PREFIXES.values())

    def config_hash(self) -> str:
        """Generate unique hash for this config.

        Only includes fields relevant to the experiment's extractor type,
        so adding params for new extractors doesn't invalidate existing hashes.
        """
        d = asdict(self)

        # Determine which extractor prefixes to keep
        if self.model_type == "best_attainable":
            # No neural extractor -- exclude all extractor-specific fields
            keep_prefixes = set()
        else:
            keep_prefixes = self._EXTRACTOR_PREFIXES.get(
                self.feature_extractor_type, set()
            )

        # Remove fields belonging to OTHER extractors
        remove_prefixes = self._ALL_EXTRACTOR_PREFIXES - keep_prefixes
        d = {
            k: v for k, v in d.items()
            if not any(k.startswith(p) for p in remove_prefixes)
        }

        config_str = json.dumps(d, sort_keys=True)
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


def group_configs_by_cache_key(
    configs: List[ExperimentConfig],
    use_cache: bool,
) -> List[tuple]:
    """Group configs by their cache key for sequential processing.

    Returns list of (cache_hash, cache_info_dict, configs) tuples,
    sorted by (max_length, dataset_name, dp_dim) so smaller caches run first.
    Non-cached mode returns a single group.

    Trainable extractor configs (no caching) are grouped separately with
    a unique key so they skip the caching step even when --cache is used.
    """
    from oci.config import CACHEABLE_EXTRACTOR_TYPES

    if not use_cache:
        return [("__no_cache__", {}, configs)]

    # Separate cacheable from non-cacheable configs
    cacheable_configs = []
    non_cacheable_configs: Dict[str, list] = {}  # ext_type -> [configs]
    for config in configs:
        if config.feature_extractor_type in CACHEABLE_EXTRACTOR_TYPES:
            cacheable_configs.append(config)
        else:
            key = f"__no_cache__{config.feature_extractor_type}"
            non_cacheable_configs.setdefault(key, []).append(config)

    groups: Dict[str, tuple] = {}  # cache_hash -> (cache_info, [configs])
    for config in cacheable_configs:
        parquet_file = _resolve_parquet_file(config.dataset_path)
        if parquet_file is None:
            continue

        # Resolve cache params based on extractor type
        if config.feature_extractor_type == "hierarchical_llm":
            _model_name = config.hlm_model_name
            _max_length = config.hlm_chunk_size * config.hlm_max_chunks
            _dp_dim = config.hlm_downprojection_dim
            _ctp = config.hlm_chat_template_prompt
        else:  # frozen_llm_pooler
            _model_name = config.flp_model_name
            _max_length = config.flp_max_length
            _dp_dim = config.flp_downprojection_dim
            _ctp = config.flp_chat_template_prompt

        _cs = config.hlm_chunk_size if config.feature_extractor_type == "hierarchical_llm" else None
        _co = config.hlm_chunk_overlap if config.feature_extractor_type == "hierarchical_llm" else None
        _mc = config.hlm_max_chunks if config.feature_extractor_type == "hierarchical_llm" else None

        cache_hash = HiddenStateCache.compute_cache_hash(
            _model_name, _max_length, str(parquet_file), None,
            downprojection_dim=_dp_dim,
            chat_template_prompt=_ctp,
            chunk_size=_cs, chunk_overlap=_co, max_chunks=_mc,
        )
        if cache_hash not in groups:
            cache_info = dict(
                parquet_file=parquet_file,
                model_name=_model_name,
                max_length=_max_length,
                batch_size=config.batch_size,
                downprojection_dim=_dp_dim,
                dataset_name=config.dataset_name,
                chat_template_prompt=_ctp,
                chunk_size=_cs,
                chunk_overlap=_co,
                max_chunks=_mc,
            )
            groups[cache_hash] = (cache_info, [])
        groups[cache_hash][1].append(config)

    # Sort by (max_length, dataset_name, dp_dim) so smaller/faster caches run first
    result = []
    for cache_hash, (cache_info, cfgs) in groups.items():
        result.append((cache_hash, cache_info, cfgs))
    result.sort(key=lambda x: (
        x[1]['max_length'],
        x[1]['dataset_name'],
        (x[1]['downprojection_dim'] is None, x[1]['downprojection_dim'] or 0),
    ))

    # Append non-cacheable groups (trainable extractors skip cache precomputation)
    for key, cfgs in non_cacheable_configs.items():
        result.append((key, {}, cfgs))

    return result


def precompute_single_cache(
    cache_info: dict,
    devices: List[str],
    cache_base_dir: Optional[str] = None,
) -> HiddenStateCache:
    """Compute (if needed) and open a single hidden state cache.

    Returns an opened HiddenStateCache with data preloaded to RAM.

    Args:
        cache_base_dir: Directory to store caches in. Defaults to dataset dir/.oci_cache.
    """
    parquet_file = cache_info['parquet_file']
    model_name = cache_info['model_name']
    max_length = cache_info['max_length']
    batch_size = cache_info['batch_size']
    dp_dim = cache_info['downprojection_dim']

    ctp = cache_info.get('chat_template_prompt', None)
    cs = cache_info.get('chunk_size', None)
    co = cache_info.get('chunk_overlap', None)
    mc = cache_info.get('max_chunks', None)

    cache_dir = cache_base_dir if cache_base_dir else str(parquet_file.parent / '.oci_cache')
    cache = HiddenStateCache(
        cache_dir=cache_dir,
        model_name=model_name,
        max_length=max_length,
        dataset_path=str(parquet_file),
        random_projection_dim=None,
        downprojection_dim=dp_dim,
        chat_template_prompt=ctp,
        chunk_size=cs,
        chunk_overlap=co,
        max_chunks=mc,
    )

    df = pd.read_parquet(parquet_file)
    if cache.is_valid(len(df)):
        logger.info(f"  Cache valid, reusing from disk")
    else:
        gpu_devices = [torch.device(d) for d in devices]
        logger.info(f"  Precomputing on {len(gpu_devices)} GPU(s) "
                    f"({model_name}, max_len={max_length}, {len(df)} samples)...")
        all_texts = df['clinical_text'].tolist()
        if cs is not None:
            cache.precompute_chunked_multi_gpu(all_texts, gpu_devices, batch_size=batch_size)
        else:
            cache.precompute_multi_gpu(all_texts, gpu_devices, batch_size=batch_size)
        logger.info(f"  Precomputation complete")

    cache.open()
    cache.preload_to_ram()
    return cache


def load_single_gpu_store(
    cache: HiddenStateCache,
    cache_info: dict,
    device: str,
) -> Optional[GPUHiddenStateStore]:
    """Load a single cache to GPU VRAM if it fits.

    Returns GPUHiddenStateStore or None if insufficient VRAM.
    """
    parquet_file = cache_info['parquet_file']
    dp_dim = cache_info['downprojection_dim']
    max_length = cache_info['max_length']
    model_name = cache_info['model_name']
    gpu_device = torch.device(device)

    df = pd.read_parquet(parquet_file)
    all_texts = df['clinical_text'].tolist()

    estimated_gb = GPUHiddenStateStore.estimate_vram_gb(
        all_texts, model_name, max_length, downprojection_dim=dp_dim,
    )
    free_vram_gb = torch.cuda.mem_get_info(gpu_device)[0] / 1e9

    if estimated_gb < free_vram_gb * 0.8:
        logger.info(f"  GPU store on {device}: ~{estimated_gb:.1f} GB "
                    f"(free: {free_vram_gb:.1f} GB) — loading...")
        store = GPUHiddenStateStore()
        store.load_from_disk_cache(cache, gpu_device)
        logger.info(f"  GPU store on {device}: loaded, "
                    f"actual VRAM: {store.estimated_vram_gb:.2f} GB")
        return store
    else:
        logger.warning(f"  GPU store on {device}: needs ~{estimated_gb:.1f} GB "
                       f"but only {free_vram_gb:.1f} GB free — will use disk cache")
        return None


def _create_datasets_and_loaders(
    train_df, test_df, train_idx, test_idx,
    text_column, confounder_cols, batch_size,
    hidden_state_cache, gpu_store,
):
    """Create train/test datasets and DataLoaders with appropriate caching."""
    use_cache = hidden_state_cache is not None
    if use_cache:
        chunk_counts = hidden_state_cache.chunk_counts
    elif gpu_store is not None:
        chunk_counts = gpu_store.chunk_counts
    else:
        chunk_counts = None

    if gpu_store is not None:
        train_dataset = CachedHiddenStateDataset(
            data=train_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            dataset_indices=np.array(train_idx),
            explicit_confounder_columns=confounder_cols,
            cache_chunk_counts=chunk_counts,
        )
        test_dataset = CachedHiddenStateDataset(
            data=test_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            dataset_indices=np.array(test_idx),
            explicit_confounder_columns=confounder_cols,
            cache_chunk_counts=chunk_counts,
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
            cache_chunk_counts=chunk_counts,
        )
        test_dataset = CachedHiddenStateDataset(
            data=test_df, text_column=text_column,
            outcome_column='outcome_indicator', treatment_column='treatment_indicator',
            dataset_indices=np.array(test_idx),
            explicit_confounder_columns=confounder_cols,
            cache_hidden_states=hidden_state_cache.hidden_states_array,
            cache_attention_masks=hidden_state_cache.attention_mask_array,
            cache_chunk_counts=chunk_counts,
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
        # Live FLP mode: prefetch batches to keep GPU fed during LLM forward passes
        dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=2)

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
    """Look up GPU store or disk cache for a config.

    Only frozen_llm_pooler and hierarchical_llm support caching.
    Trainable extractors (hierarchical_cnn, hierarchical_gru, simple_cnn) return (None, None).
    """
    from oci.config import CACHEABLE_EXTRACTOR_TYPES
    gpu_store = None
    hidden_state_cache = None

    ext_type = config.feature_extractor_type

    if ext_type not in CACHEABLE_EXTRACTOR_TYPES:
        return gpu_store, hidden_state_cache

    if ext_type == "frozen_llm_pooler" and config.flp_freeze_llm and config.flp_cache_hidden_states:
        cache_hash = HiddenStateCache.compute_cache_hash(
            config.flp_model_name, config.flp_max_length, str(parquet_file),
            None, downprojection_dim=config.flp_downprojection_dim,
            chat_template_prompt=config.flp_chat_template_prompt,
        )
        if gpu_store_registry is not None:
            gpu_store = gpu_store_registry.get(cache_hash)
        if gpu_store is None and cache_registry is not None:
            hidden_state_cache = cache_registry.get(cache_hash)
    elif ext_type == "hierarchical_llm" and config.hlm_freeze_llm and config.hlm_cache_hidden_states:
        _max_length = config.hlm_chunk_size * config.hlm_max_chunks
        cache_hash = HiddenStateCache.compute_cache_hash(
            config.hlm_model_name, _max_length, str(parquet_file),
            None, downprojection_dim=config.hlm_downprojection_dim,
            chat_template_prompt=config.hlm_chat_template_prompt,
            chunk_size=config.hlm_chunk_size,
            chunk_overlap=config.hlm_chunk_overlap,
            max_chunks=config.hlm_max_chunks,
        )
        if gpu_store_registry is not None:
            gpu_store = gpu_store_registry.get(cache_hash)
        if gpu_store is None and cache_registry is not None:
            hidden_state_cache = cache_registry.get(cache_hash)

    return gpu_store, hidden_state_cache


def _common_model_kwargs(config, gpu_store, hidden_state_cache, confounder_specs, device):
    """Build common model kwargs for any extractor type."""
    ext_type = config.feature_extractor_type
    use_cache = gpu_store is not None or hidden_state_cache is not None

    kwargs = dict(
        feature_extractor_type=ext_type,
        device=str(device),
    )

    if ext_type == "frozen_llm_pooler":
        kwargs.update(
            flp_model_name=config.flp_model_name,
            flp_max_length=config.flp_max_length,
            flp_freeze_llm=config.flp_freeze_llm,
            flp_gated_attention_dim=config.flp_gated_attention_dim,
            flp_projection_dim=config.flp_projection_dim,
            flp_dropout=config.flp_dropout,
            flp_gradient_checkpointing=config.flp_gradient_checkpointing,
            flp_downprojection_dim=None if use_cache else config.flp_downprojection_dim,
            flp_chat_template_prompt=config.flp_chat_template_prompt,
        )
        if gpu_store is not None:
            kwargs['flp_skip_llm'] = True
            kwargs['flp_cached_hidden_size'] = gpu_store.hidden_size
        elif hidden_state_cache is not None:
            kwargs['flp_skip_llm'] = True
            kwargs['flp_cached_hidden_size'] = hidden_state_cache.hidden_size
    elif ext_type == "hierarchical_llm":
        kwargs.update(
            hlm_model_name=config.hlm_model_name,
            hlm_chunk_size=config.hlm_chunk_size,
            hlm_chunk_overlap=config.hlm_chunk_overlap,
            hlm_max_chunks=config.hlm_max_chunks,
            hlm_freeze_llm=config.hlm_freeze_llm,
            hlm_gated_attention_dim=getattr(config, 'hlm_gated_attention_dim', 128),
            hlm_projection_dim=getattr(config, 'hlm_projection_dim', 128),
            hlm_dropout=getattr(config, 'hlm_dropout', 0.1),
            hlm_gradient_checkpointing=getattr(config, 'hlm_gradient_checkpointing', True),
            hlm_downprojection_dim=None if use_cache else config.hlm_downprojection_dim,
            hlm_chat_template_prompt=config.hlm_chat_template_prompt,
        )
        if gpu_store is not None:
            kwargs['hlm_skip_llm'] = True
            kwargs['hlm_cached_hidden_size'] = gpu_store.hidden_size
        elif hidden_state_cache is not None:
            kwargs['hlm_skip_llm'] = True
            kwargs['hlm_cached_hidden_size'] = hidden_state_cache.hidden_size
    elif ext_type == "hierarchical_cnn":
        kwargs.update(
            hcnn_embedding_dim=config.hcnn_embedding_dim,
            hcnn_conv_dim=config.hcnn_conv_dim,
            hcnn_kernel_size=config.hcnn_kernel_size,
            hcnn_num_conv_blocks=config.hcnn_num_conv_blocks,
            hcnn_chunk_size=config.hcnn_chunk_size,
            hcnn_chunk_overlap=config.hcnn_chunk_overlap,
            hcnn_max_chunks=config.hcnn_max_chunks,
            hcnn_vocab_size=config.hcnn_vocab_size,
            hcnn_projection_dim=config.hcnn_projection_dim,
            hcnn_dropout=config.hcnn_dropout,
        )
    elif ext_type == "hierarchical_gru":
        kwargs.update(
            hgru_embedding_dim=config.hgru_embedding_dim,
            hgru_gru_hidden_dim=config.hgru_gru_hidden_dim,
            hgru_num_gru_layers=config.hgru_num_gru_layers,
            hgru_chunk_size=config.hgru_chunk_size,
            hgru_chunk_overlap=config.hgru_chunk_overlap,
            hgru_max_chunks=config.hgru_max_chunks,
            hgru_vocab_size=config.hgru_vocab_size,
            hgru_projection_dim=config.hgru_projection_dim,
            hgru_dropout=config.hgru_dropout,
        )
    elif ext_type == "simple_cnn":
        kwargs.update(
            scnn_embedding_dim=config.scnn_embedding_dim,
            scnn_conv_dim=config.scnn_conv_dim,
            scnn_kernel_size=config.scnn_kernel_size,
            scnn_num_conv_blocks=config.scnn_num_conv_blocks,
            scnn_max_length=config.scnn_max_length,
            scnn_vocab_size=config.scnn_vocab_size,
            scnn_projection_dim=config.scnn_projection_dim,
            scnn_dropout=config.scnn_dropout,
        )

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
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42 + config.repeat_index)

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
            explicit_confounder_specs=confounder_specs,
        ))

        model = CausalTextForest(**model_kwargs)

        from oci.config import TRAINABLE_EXTRACTOR_TYPES
        if config.feature_extractor_type in TRAINABLE_EXTRACTOR_TYPES:
            model.fit_tokenizer(train_df[text_column].tolist())

        # Verify all parameters are float32 (diagnose dtype leakage)
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                logger.warning(f"Parameter {name} has unexpected dtype {param.dtype}, casting to float32")
                param.data = param.data.float()

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

        use_cached = gpu_store is not None or hidden_state_cache is not None

        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)
                if use_cached:
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
                    if use_cached:
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

        if hidden_state_cache is not None:
            chunk_counts = hidden_state_cache.chunk_counts
        elif gpu_store is not None:
            chunk_counts = gpu_store.chunk_counts
        else:
            chunk_counts = None
        if gpu_store is not None:
            combined_indices = np.concatenate([train_idx, test_idx])
            combined_dataset = CachedHiddenStateDataset(
                data=combined_df, text_column=text_column,
                outcome_column='outcome_indicator', treatment_column='treatment_indicator',
                dataset_indices=combined_indices,
                explicit_confounder_columns=confounder_cols,
                cache_chunk_counts=chunk_counts,
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
                cache_chunk_counts=chunk_counts,
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

        cf_kwargs = dict(gpu_store=gpu_store) if use_cached else {}
        model.train_causal_forest(combined_loader, combined_T, combined_Y, **cf_kwargs)
        preds = model.predict(test_loader, return_ci=True, **cf_kwargs)

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
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42 + config.repeat_index)

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

        from oci.config import TRAINABLE_EXTRACTOR_TYPES
        if config.feature_extractor_type in TRAINABLE_EXTRACTOR_TYPES:
            model.fit_tokenizer(train_df[text_column].tolist())

        # Verify all parameters are float32 (diagnose dtype leakage)
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                logger.warning(f"Parameter {name} has unexpected dtype {param.dtype}, casting to float32")
                param.data = param.data.float()

        train_dataset, test_dataset, train_loader, test_loader, collate_fn, dl_kwargs = \
            _create_datasets_and_loaders(
                train_df, test_df, train_idx, test_idx,
                text_column, confounder_cols, batch_size,
                hidden_state_cache, gpu_store,
            )

        if confounder_specs and train_dataset.explicit_confounder_values:
            model.fit_explicit_confounder_featurizer(train_dataset.explicit_confounder_values)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        use_cached = gpu_store is not None or hidden_state_cache is not None

        best_val_loss = float('inf')
        best_state = None
        history = []

        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)
                if use_cached:
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
                    if use_cached:
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
                if use_cached:
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


def build_confounder_values_from_columns(
    df: pd.DataFrame,
    spec_names: List[str],
    prefix: str,
) -> List[Dict[str, Any]]:
    """Read {prefix}_{name} columns from a dataframe and return a list of dicts.

    For each row, builds a dict like {"name": value, "name_missing": bool, ...}
    suitable for passing to get_raw_confounder_features().
    """
    result = []
    for _, row in df.iterrows():
        values = {}
        for name in spec_names:
            col = f"{prefix}_{name}"
            if col in df.columns:
                val = row[col]
                if pd.isna(val):
                    values[name] = None
                    values[f"{name}_missing"] = True
                else:
                    values[name] = val
                    values[f"{name}_missing"] = False
            else:
                values[name] = None
                values[f"{name}_missing"] = True
        result.append(values)
    return result


def run_best_attainable_experiment(
    config: ExperimentConfig,
    df: pd.DataFrame,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Run a best-attainable experiment using ground-truth confounder values.

    Uses true_{name} columns from the dataset to train a CausalForestDML,
    providing an upper-bound reference for neural-network-based methods.
    No GPU required.
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from oci.models.causal_forest_head import tune_causal_forest_model
    from oci.models.explicit_confounder_featurizer import get_raw_confounder_features

    confounder_specs = load_confounder_specs_from_metadata(config.dataset_path)
    if not confounder_specs:
        return {'error': 'No confounder specs in metadata.json', 'skipped': True}

    spec_names = [s.name for s in confounder_specs]

    # Check that true_{name} columns exist
    missing = [f"true_{n}" for n in spec_names if f"true_{n}" not in df.columns]
    if missing:
        return {'error': f"Missing ground-truth columns: {missing}", 'skipped': True}

    df = df.reset_index(drop=True)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42 + config.repeat_index)

    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Build confounder value dicts from true_{name} columns
        train_values = build_confounder_values_from_columns(train_df, spec_names, "true")
        test_values = build_confounder_values_from_columns(test_df, spec_names, "true")

        # Compute normalization stats from train fold
        train_features, feature_names = get_raw_confounder_features(
            train_values, confounder_specs,
        )
        # Extract the computed means/stds by re-running (they are computed internally)
        # We need to pass them explicitly for test set normalization
        continuous_means = {}
        continuous_stds = {}
        for spec in confounder_specs:
            if spec.type == "continuous":
                vals = []
                for v in train_values:
                    val = v.get(spec.name)
                    miss = v.get(f"{spec.name}_missing", val is None)
                    if not miss and val is not None:
                        vals.append(float(val))
                if vals:
                    continuous_means[spec.name] = sum(vals) / len(vals)
                    variance = sum((x - continuous_means[spec.name]) ** 2 for x in vals) / len(vals)
                    continuous_stds[spec.name] = max(variance ** 0.5, 1e-6)
                else:
                    continuous_means[spec.name] = 0.0
                    continuous_stds[spec.name] = 1.0

        # Re-compute train features with explicit stats, compute test features
        train_features, feature_names = get_raw_confounder_features(
            train_values, confounder_specs,
            continuous_means=continuous_means,
            continuous_stds=continuous_stds,
        )
        test_features, _ = get_raw_confounder_features(
            test_values, confounder_specs,
            continuous_means=continuous_means,
            continuous_stds=continuous_stds,
        )

        X_train = np.array(train_features, dtype=np.float64)
        X_test = np.array(test_features, dtype=np.float64)
        T_train = train_df['treatment_indicator'].values.astype(np.float64)
        Y_train = train_df['outcome_indicator'].values.astype(np.float64)

        # Train CausalForestDML with flexible nuisance models
        # (DGP has interaction terms that linear defaults can't capture)
        def create_causal_forest():
            return CausalForestDML(
                model_t=RandomForestClassifier(
                    n_estimators=max(50, config.cf_n_estimators // 2),
                    min_samples_leaf=config.cf_min_samples_leaf,
                    random_state=42 + config.repeat_index,
                    n_jobs=n_jobs,
                ),
                model_y=RandomForestRegressor(
                    n_estimators=max(50, config.cf_n_estimators // 2),
                    min_samples_leaf=config.cf_min_samples_leaf,
                    random_state=42 + config.repeat_index,
                    n_jobs=n_jobs,
                ),
                discrete_treatment=True,
                n_estimators=config.cf_n_estimators,
                min_samples_leaf=config.cf_min_samples_leaf,
                max_depth=None,
                honest=True,
                inference=True,
                random_state=42 + config.repeat_index,
                n_jobs=n_jobs,
            )

        cf = create_causal_forest()
        if not tune_causal_forest_model(cf, Y=Y_train, T=T_train, X=X_train):
            cf = create_causal_forest()
        cf.fit(Y_train, T_train, X=X_train)

        # Predict on test fold
        tau_pred = cf.effect(X_test).flatten()
        tau_lower, tau_upper = cf.effect_interval(X_test, alpha=0.05)
        tau_lower = tau_lower.flatten()
        tau_upper = tau_upper.flatten()

        # Train RandomForest for propensity and outcome predictions (for metrics)
        rf_prop = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_prop.fit(X_train, T_train.astype(int))
        pred_propensity = rf_prop.predict_proba(X_test)[:, 1]

        rf_out = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_out.fit(X_train, Y_train.astype(int))
        pred_outcome = rf_out.predict_proba(X_test)[:, 1]

        # Use tau_pred for ITE; approximate y0/y1 from outcome + tau
        pred_y0 = pred_outcome - tau_pred * pred_propensity
        pred_y1 = pred_y0 + tau_pred

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
        tau_lower=results_df['pred_tau_lower'].values,
        tau_upper=results_df['pred_tau_upper'].values,
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
        missing_cols = [c for c in confounder_cols if c not in df.columns]
        if missing_cols:
            return {
                'error': (f"Confounder columns missing from dataset: {missing_cols}. "
                          f"Run LLM extraction first to create llm_extracted_* columns."),
                'skipped': True,
            }

    # Look up cache
    gpu_store, hidden_state_cache = _get_cache_info(
        config, parquet_file, cache_registry, gpu_store_registry
    )

    # Dispatch to model-specific runner
    if config.model_type == "best_attainable":
        result = run_best_attainable_experiment(config, df)
    elif config.model_type == "causal_forest":
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
    dataset_paths: List[str],
    filter_model_types: Optional[List[str]] = None,
    filter_max_lengths: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
    chat_template_prompt: Optional[str] = None,
    filter_extractor_types: Optional[List[str]] = None,
    filter_downprojection_dims: Optional[List[Optional[int]]] = None,
) -> List[ExperimentConfig]:
    """Generate all experiment configurations.

    When chat_template_prompt is provided, both None (raw text) and the
    prompt string are included as a grid dimension so that experiments
    run with and without the chat template for comparison.

    When filter_extractor_types is provided, only those extractor types are
    included.  Each extractor type generates its own relevant hyperparameter
    sub-grid.

    When multiple model_names are provided, each frozen LLM model name
    becomes a dimension of the experiment grid for LLM-based extractors
    (frozen_llm_pooler, hierarchical_llm).
    """

    if model_names is None:
        model_names = ["Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-0.8B", "google/medgemma-1.5-4b-it"]

    datasets = [(p, Path(p).name) for p in dataset_paths]

    # best_attainable is CPU-only and doesn't use LLM params -- added separately below
    model_types = ["causal_forest", "rlearner", "dragonnet"]
    explicit_confounder_options = [False, True]

    if filter_model_types:
        model_types = [m for m in model_types if m in filter_model_types
                       and m != "best_attainable"]

    # Determine which extractor types to include
    all_extractor_types = ["frozen_llm_pooler", "hierarchical_llm",
                           "hierarchical_cnn", "hierarchical_gru", "simple_cnn"]

    extractor_types = all_extractor_types
    if filter_extractor_types:
        extractor_types = [e for e in all_extractor_types if e in filter_extractor_types]

    # Training hyperparameters shared across all neural extractors
    learning_rates = [1e-5, 1e-4]
    epoch_counts = [5, 10, 25, 50]

    configs = []

    for ext_type in extractor_types:
        if ext_type == "frozen_llm_pooler":
            # frozen_llm_pooler grid: max_lengths x downprojection_dims x confounders x chat_template x model_names x lr x epochs
            max_lengths = [5000, 10000, 25000, 50000, 75000, 100000]
            downprojection_dims = [None, 128, 256, 512]  # None = no downprojection (pool on full hidden_size)

            chat_template_options = [None]
            if chat_template_prompt is not None:
                chat_template_options = [None, chat_template_prompt]
            if filter_max_lengths:
                max_lengths = [m for m in max_lengths if m in filter_max_lengths]
            if filter_downprojection_dims is not None:
                downprojection_dims = [d for d in downprojection_dims if d in filter_downprojection_dims]

            for (dataset_path, dataset_name), model_type, max_len, explicit_conf, dp_dim, ctp, mn, lr, ep in itertools.product(
                datasets, model_types, max_lengths, explicit_confounder_options, downprojection_dims, chat_template_options, model_names, learning_rates, epoch_counts
            ):
                configs.append(ExperimentConfig(
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    model_type=model_type,
                    use_explicit_confounders=explicit_conf,
                    feature_extractor_type="frozen_llm_pooler",
                    flp_max_length=max_len,
                    flp_downprojection_dim=dp_dim,
                    flp_model_name=mn,
                    flp_chat_template_prompt=ctp,
                    learning_rate=lr,
                    epochs=ep,
                ))

        elif ext_type == "hierarchical_llm":
            # hierarchical_llm grid: effective_lengths (chunk_size*max_chunks) x downprojection_dims x confounders
            chunk_size = 2048
            chunk_overlap = 256
            max_chunks_options = [16]  # effective: 2048*16=32K
            downprojection_dims = [None, 128, 256, 512]  # None = no downprojection (pool on full hidden_size)
            if filter_downprojection_dims is not None:
                downprojection_dims = [d for d in downprojection_dims if d in filter_downprojection_dims]

            for (dataset_path, dataset_name), model_type, n_chunks, explicit_conf, dp_dim, mn, lr, ep in itertools.product(
                datasets, model_types, max_chunks_options, explicit_confounder_options, downprojection_dims, model_names, learning_rates, epoch_counts
            ):
                configs.append(ExperimentConfig(
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    model_type=model_type,
                    use_explicit_confounders=explicit_conf,
                    feature_extractor_type="hierarchical_llm",
                    hlm_model_name=mn,
                    hlm_chunk_size=chunk_size,
                    hlm_chunk_overlap=chunk_overlap,
                    hlm_max_chunks=n_chunks,
                    hlm_downprojection_dim=dp_dim,
                    learning_rate=lr,
                    epochs=ep,
                ))

        elif ext_type == "hierarchical_cnn":
            # hierarchical_cnn grid: chunk_sizes x confounders
            chunk_sizes = [256, 512]

            for (dataset_path, dataset_name), model_type, explicit_conf, cs, lr, ep in itertools.product(
                datasets, model_types, explicit_confounder_options, chunk_sizes, learning_rates, epoch_counts
            ):
                configs.append(ExperimentConfig(
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    model_type=model_type,
                    use_explicit_confounders=explicit_conf,
                    feature_extractor_type="hierarchical_cnn",
                    hcnn_chunk_size=cs,
                    learning_rate=lr,
                    epochs=ep,
                ))

        elif ext_type == "hierarchical_gru":
            # hierarchical_gru grid: chunk_sizes x confounders
            chunk_sizes = [256, 512]

            for (dataset_path, dataset_name), model_type, explicit_conf, cs, lr, ep in itertools.product(
                datasets, model_types, explicit_confounder_options, chunk_sizes, learning_rates, epoch_counts
            ):
                configs.append(ExperimentConfig(
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    model_type=model_type,
                    use_explicit_confounders=explicit_conf,
                    feature_extractor_type="hierarchical_gru",
                    hgru_chunk_size=cs,
                    learning_rate=lr,
                    epochs=ep,
                ))

        elif ext_type == "simple_cnn":
            # simple_cnn grid: max_lengths x confounders
            scnn_max_lengths = [5000, 10000, 25000]

            if filter_max_lengths:
                scnn_max_lengths = [m for m in scnn_max_lengths if m in filter_max_lengths]

            for (dataset_path, dataset_name), model_type, explicit_conf, max_len, lr, ep in itertools.product(
                datasets, model_types, explicit_confounder_options, scnn_max_lengths, learning_rates, epoch_counts
            ):
                configs.append(ExperimentConfig(
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    model_type=model_type,
                    use_explicit_confounders=explicit_conf,
                    feature_extractor_type="simple_cnn",
                    scnn_max_length=max_len,
                    learning_rate=lr,
                    epochs=ep,
                ))

    # Add best_attainable experiments (one per dataset, no GPU needed)
    if not filter_model_types or "best_attainable" in filter_model_types:
        for dataset_path, dataset_name in datasets:
            configs.append(ExperimentConfig(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                model_type="best_attainable",
                use_explicit_confounders=False,
            ))

    # Shuffle so patterns emerge early
    random.Random(42).shuffle(configs)

    return configs


def estimate_workers_per_gpu(
    device: str,
    max_cap: int = 50,
    per_worker_mb: int = 50,
) -> int:
    """Estimate how many concurrent experiment workers a GPU can support.

    After cache loading, queries free VRAM and divides by a conservative
    per-worker overhead estimate (model ~1MB + optimizer ~3MB + activations
    + batch tensors).

    Args:
        device: CUDA device string (e.g. "cuda:0").
        max_cap: Maximum workers regardless of VRAM.
        per_worker_mb: Estimated VRAM per worker in MB.

    Returns:
        Number of workers (clamped to [1, max_cap]).
    """
    try:
        free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
        free_mb = free_bytes / 1e6
        n = int(free_mb // per_worker_mb)
        n = max(1, min(n, max_cap))
        logger.info(f"  {device}: {free_mb:.0f} MB free, ~{per_worker_mb} MB/worker -> {n} workers")
        return n
    except Exception as e:
        logger.warning(f"  {device}: VRAM query failed ({e}), defaulting to 1 worker")
        return 1


def resolve_workers_per_gpu(
    workers_per_gpu_arg: str,
    device: str,
    use_cache: bool,
) -> int:
    """Resolve the --workers-per-gpu argument for a specific device.

    Returns 1 for non-cached mode (LLM loaded per experiment).
    """
    if not use_cache:
        return 1

    if workers_per_gpu_arg == "auto":
        return estimate_workers_per_gpu(device)
    else:
        return int(workers_per_gpu_arg)


def _open_cache_for_worker(cache_hash: str, cache_info: dict, cache_base_dir: Optional[str] = None) -> HiddenStateCache:
    """Open and preload a hidden state cache from disk in a worker process.

    Each worker process calls this independently to get its own cache handle.
    The OS page cache ensures that memmap reads are fast after the first load.
    """
    parquet_file = Path(cache_info['parquet_file'])
    cache_dir = cache_base_dir if cache_base_dir else str(parquet_file.parent / '.oci_cache')
    cache = HiddenStateCache(
        cache_dir=cache_dir,
        model_name=cache_info['model_name'],
        max_length=cache_info['max_length'],
        dataset_path=str(parquet_file),
        random_projection_dim=None,
        downprojection_dim=cache_info['downprojection_dim'],
        chat_template_prompt=cache_info.get('chat_template_prompt', None),
        chunk_size=cache_info.get('chunk_size', None),
        chunk_overlap=cache_info.get('chunk_overlap', None),
        max_chunks=cache_info.get('max_chunks', None),
    )
    cache.open()
    cache.preload_to_ram()
    return cache


def worker_process_fn(
    device: str,
    job_queue: mp.Queue,
    progress_queue: mp.Queue,
    output_dir: str,
    cache_hash: str,
    cache_info: Optional[dict],
    use_gpu_cache: bool,
    cache_base_dir: Optional[str] = None,
):
    """Worker process for a single GPU.

    Each process initializes its own CUDA context, opens the disk cache
    independently, and optionally loads a GPU store. This avoids GIL
    contention that serializes threading-based workers.
    """
    output_dir = Path(output_dir)
    torch.set_default_dtype(torch.float32)

    # Initialize cache in this process
    cache_registry = {}
    gpu_store_registry = {}

    if cache_info and cache_hash != "__no_cache__":
        cache = _open_cache_for_worker(cache_hash, cache_info, cache_base_dir=cache_base_dir)
        cache_registry[cache_hash] = cache

        if use_gpu_cache:
            store = load_single_gpu_store(cache, cache_info, device)
            if store is not None:
                gpu_store_registry = {cache_hash: store}

    logger.info(f"Worker process started on {device} (pid={os.getpid()})")

    while True:
        try:
            config = job_queue.get(timeout=2)
        except Exception:
            break

        config_hash = config.config_hash()

        try:
            result = run_single_experiment(
                config, device, output_dir, cache_registry, gpu_store_registry
            )

            result_file = output_dir / "results" / f"{config_hash}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            progress_queue.put(("done", config_hash, result))

        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            logger.error(
                f"Experiment {config_hash} FAILED "
                f"(extractor={config.feature_extractor_type}, "
                f"model={config.model_type}, ds={config.dataset_name}, "
                f"dp={config.flp_downprojection_dim}, "
                f"conf={config.use_explicit_confounders}): {error_msg}\n{tb}"
            )
            error_result = {
                'config': asdict(config),
                'error': error_msg,
                'skipped': True,
            }
            result_file = output_dir / "results" / f"{config_hash}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, 'w') as f:
                json.dump(error_result, f, indent=2, default=str)

            progress_queue.put(("error", config_hash, error_result))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cleanup
    for store in gpu_store_registry.values():
        store.free()
    for c in cache_registry.values():
        c.close()

    logger.info(f"Worker process on {device} (pid={os.getpid()}) finished")


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
    """Worker thread to process experiments on a single GPU (legacy, used without --cache)."""
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
                error_msg = str(e)
                results_dict[config_hash] = {
                    'config': asdict(config),
                    'error': error_msg,
                    'skipped': True
                }
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Error: {error_msg[:50]}")
                tb = traceback.format_exc()
                logger.error(
                    f"Experiment {config_hash} FAILED "
                    f"(extractor={config.feature_extractor_type}, "
                    f"model={config.model_type}, ds={config.dataset_name}, "
                    f"dp={config.flp_downprojection_dim}, "
                    f"conf={config.use_explicit_confounders}): {error_msg}\n{tb}"
                )

        finally:
            job_queue.task_done()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_best_attainable_worker(args_tuple):
    """Top-level function for ProcessPoolExecutor (must be picklable).

    Runs a single best_attainable experiment and saves the result JSON.
    Returns (config_hash, result_dict).
    """
    config, output_dir_str, n_jobs = args_tuple
    output_dir = Path(output_dir_str)
    config_hash = config.config_hash()

    try:
        parquet_file = _resolve_parquet_file(config.dataset_path)
        if parquet_file is None:
            result = {
                'config': asdict(config),
                'error': f"Dataset not found in {config.dataset_path}",
                'skipped': True,
            }
        else:
            df = pd.read_parquet(parquet_file)
            ba_result = run_best_attainable_experiment(config, df, n_jobs=n_jobs)
            result = {
                'config': asdict(config),
                'metrics': ba_result['metrics'],
                'n_samples': ba_result['n_samples'],
                'skipped': False,
                'error': None,
            }
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"best_attainable {config_hash} FAILED: {e}\n{tb}")
        result = {
            'config': asdict(config),
            'error': str(e),
            'skipped': True,
        }

    result_file = output_dir / "results" / f"{config_hash}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return config_hash, result


def main():
    parser = argparse.ArgumentParser(
        description="Oracle experiment runner for CDT (causal_forest, rlearner, dragonnet, best_attainable)"
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
        required=True,
        help="Dataset directory paths (each must contain dataset.parquet or dataset_with_extraction.parquet)"
    )
    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        default=None,
        help="Filter model types (causal_forest, rlearner, dragonnet, best_attainable)"
    )
    parser.add_argument(
        "--max-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Filter max lengths (5000, 10000, 25000, 50000, 100000)"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Opt-in to pre-caching hidden states to disk (default: live LLM forward per batch)"
    )
    parser.add_argument(
        "--gpu-cache",
        action="store_true",
        help="Keep pre-computed hidden states in GPU VRAM instead of disk cache (implies --cache)"
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of repeats per experiment config with different random seeds (default: 10)"
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        default=["Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-0.8B", "google/medgemma-1.5-4b-it"],
        help="HuggingFace model name(s) for frozen LLM extractors. "
             "Multiple names create a grid dimension "
             "(default: Qwen3.5-0.8B-Base, Qwen3.5-0.8B, medgemma-1.5-4b-it)"
    )
    parser.add_argument(
        "--chat-template-prompt",
        type=str,
        default=None,
        help="Chat template prompt for instruct models. Wraps each text in the model's "
             "chat template with this prompt preceding the clinical text. (default: None = disabled)"
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=str,
        default="auto",
        help="Concurrent experiment workers per GPU: 'auto' (estimate from free VRAM) "
             "or an integer (default: auto). Only effective with --cache/--gpu-cache; "
             "non-cached mode always uses 1."
    )
    parser.add_argument(
        "--filter-extractor-types",
        type=str,
        nargs="+",
        default=None,
        help="Filter feature extractor types to include in the grid "
             "(frozen_llm_pooler, hierarchical_llm, hierarchical_cnn, "
             "hierarchical_gru, simple_cnn). Default: all."
    )

    def _parse_dp_dim(value: str) -> Optional[int]:
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--downprojection-dims values must be integers or 'none', got '{value}'"
            )

    parser.add_argument(
        "--downprojection-dims",
        type=_parse_dp_dim,
        nargs="+",
        default=None,
        help="Filter downprojection dims for frozen_llm_pooler / hierarchical_llm grids. "
             "Pass integers (e.g. 128 256) and/or 'none' for no downprojection "
             "(pool on full hidden_size). Default: all of [none, 128, 256, 512]."
    )

    args = parser.parse_args()

    # Validate --workers-per-gpu
    if args.workers_per_gpu != "auto":
        try:
            wpg = int(args.workers_per_gpu)
            if wpg < 1:
                parser.error("--workers-per-gpu must be >= 1")
        except ValueError:
            parser.error(f"--workers-per-gpu must be 'auto' or an integer, got '{args.workers_per_gpu}'")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save command line invocation
    cmdline_file = output_dir / "command_line.txt"
    cmdline_file.write_text(" ".join(sys.argv) + "\n")
    logger.info(f"Command line saved to {cmdline_file}")

    base_configs = generate_experiment_grid(
        dataset_paths=args.datasets,
        filter_model_types=args.model_types,
        filter_max_lengths=args.max_lengths,
        model_names=args.model_names,
        chat_template_prompt=args.chat_template_prompt,
        filter_extractor_types=args.filter_extractor_types,
        filter_downprojection_dims=args.downprojection_dims,
    )

    use_cache = args.cache or args.gpu_cache

    # Expand each config into N repeats with different repeat_index values
    configs = []
    for base_config in base_configs:
        for repeat_idx in range(args.n_repeats):
            config = deepcopy(base_config)
            config.repeat_index = repeat_idx
            config.n_folds = args.n_folds
            config.flp_cache_hidden_states = use_cache
            config.hlm_cache_hidden_states = use_cache
            configs.append(config)

    # Re-shuffle with repeats included
    random.Random(42).shuffle(configs)

    logger.info(f"Generated {len(base_configs)} base configs x {args.n_repeats} repeats = {len(configs)} experiments")
    logger.info(f"Mode: {'cached hidden states' if use_cache else 'live LLM forward per batch'}")

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

    if not pending_configs:
        logger.info("No experiments to run")
        return

    # Print experiment summary and ask for confirmation
    ba_pending = [c for c in pending_configs if c.model_type == "best_attainable"]
    gpu_pending = [c for c in pending_configs if c.model_type != "best_attainable"]

    model_type_summary = {}
    for c in pending_configs:
        model_type_summary[c.model_type] = model_type_summary.get(c.model_type, 0) + 1

    extractor_summary = {}
    for c in pending_configs:
        extractor_summary[c.feature_extractor_type] = extractor_summary.get(c.feature_extractor_type, 0) + 1

    llm_model_summary = {}
    for c in pending_configs:
        if c.feature_extractor_type in ('frozen_llm_pooler',):
            llm_model_summary[c.flp_model_name] = llm_model_summary.get(c.flp_model_name, 0) + 1
        elif c.feature_extractor_type in ('hierarchical_llm',):
            llm_model_summary[c.hlm_model_name] = llm_model_summary.get(c.hlm_model_name, 0) + 1

    dataset_names = sorted(set(c.dataset_name for c in pending_configs))

    print(f"\n{'='*60}")
    print(f"Experiment Grid Summary")
    print(f"{'='*60}")
    print(f"Total experiments to run: {len(pending_configs)}")
    if completed_hashes:
        print(f"  Already completed (skipped): {len(completed_hashes)}")
    print(f"  best_attainable (CPU): {len(ba_pending)}")
    print(f"  GPU experiments: {len(gpu_pending)}")
    print(f"\nModel types: {', '.join(f'{k}({v})' for k, v in sorted(model_type_summary.items()))}")
    print(f"Extractors:  {', '.join(f'{k}({v})' for k, v in sorted(extractor_summary.items()))}")
    if llm_model_summary:
        print(f"Frozen LLMs: {', '.join(f'{k}({v})' for k, v in sorted(llm_model_summary.items()))}")
    lr_values = sorted(set(c.learning_rate for c in pending_configs))
    epoch_values = sorted(set(c.epochs for c in pending_configs))
    print(f"Datasets:    {', '.join(dataset_names)}")
    print(f"LR values:   {', '.join(str(v) for v in lr_values)}")
    print(f"Epoch counts:{', '.join(str(v) for v in epoch_values)}")
    print(f"Repeats:     {args.n_repeats}")
    print(f"{'='*60}")

    # response = input("Proceed? [y/N] ").strip()
    # if response.lower() not in ('y', 'yes'):
    #     print("Aborted.")
    #     return

    # Separate best_attainable (CPU-only) from GPU experiments
    ba_configs = [c for c in pending_configs if c.model_type == "best_attainable"]
    gpu_configs = [c for c in pending_configs if c.model_type != "best_attainable"]

    # Run best_attainable experiments in parallel on CPU
    if ba_configs:
        n_cpu_workers = min(len(ba_configs), max(1, os.cpu_count() // 2))
        # Limit per-worker sklearn threads to avoid oversubscription
        n_jobs_per_worker = max(1, os.cpu_count() // n_cpu_workers)
        logger.info(f"Running {len(ba_configs)} best_attainable experiments "
                    f"in parallel ({n_cpu_workers} CPU workers, "
                    f"{n_jobs_per_worker} threads each)")
        ba_progress = tqdm(total=len(ba_configs), desc="best_attainable (CPU)")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpu_workers) as pool:
            futures = {
                pool.submit(
                    _run_best_attainable_worker,
                    (config, str(output_dir), n_jobs_per_worker),
                ): config
                for config in ba_configs
            }
            for future in concurrent.futures.as_completed(futures):
                config_hash, result = future.result()
                results_dict[config_hash] = result
                ba_progress.update(1)
                if not result.get('skipped'):
                    metrics = result.get('metrics', {})
                    ba_progress.set_postfix_str(
                        f"ITE corr: {metrics.get('ite_corr', 'N/A'):.3f}"
                    )
        ba_progress.close()
        logger.info(f"Completed {len(ba_configs)} best_attainable experiments")

    if not gpu_configs:
        logger.info("No GPU experiments to run")
        return

    # Cache directory lives in the experiment output directory
    cache_base_dir = str(output_dir / '.oci_cache')

    # Group GPU experiments by cache key for sequential cache processing
    cache_groups = group_configs_by_cache_key(gpu_configs, use_cache)

    if not use_cache and args.workers_per_gpu != "auto" and int(args.workers_per_gpu) > 1:
        logger.warning("--workers-per-gpu > 1 requires --cache or --gpu-cache (LLM uses most VRAM); forcing 1")
        args.workers_per_gpu = "1"

    # Resolve workers per GPU for each device (used in multiprocessing path)
    # Non-cached mode: force 1 (each experiment loads the full LLM)
    # Cached mode (disk or GPU): spawn multiple workers per GPU
    # Note: with --gpu-cache, each worker process loads its own GPU store copy.
    # The per-worker VRAM check in load_single_gpu_store() will fall back to
    # disk cache if there isn't enough free VRAM, so this is safe.
    if use_cache:
        wpg_per_device = {
            d: resolve_workers_per_gpu(args.workers_per_gpu, d, use_cache)
            for d in args.devices
        }
        if args.gpu_cache and any(w > 1 for w in wpg_per_device.values()):
            logger.info("Note: with --gpu-cache, each worker loads its own GPU store copy. "
                        "Workers that can't fit the store will fall back to disk cache.")
    else:
        wpg_per_device = {d: 1 for d in args.devices}

    total_workers = sum(wpg_per_device.values())

    logger.info(f"Running {len(gpu_configs)} GPU experiments in {len(cache_groups)} cache group(s) "
                f"on {len(args.devices)} GPU(s) (workers-per-gpu: {args.workers_per_gpu})")
    for d, w in wpg_per_device.items():
        logger.info(f"  {d}: {w} worker(s)")
    if use_cache:
        logger.info(f"Using multiprocessing ({total_workers} total processes, avoids GIL contention)")
    else:
        logger.info("Using threading (non-cached mode, LLM loaded per experiment)")

    progress_bar = tqdm(total=len(gpu_configs), desc="Experiments")

    for group_idx, (cache_hash, cache_info, group_configs) in enumerate(cache_groups):
        if not group_configs:
            continue

        # Log cache group info
        if use_cache and cache_hash != "__no_cache__":
            logger.info(f"\n{'='*60}")
            logger.info(f"Cache group {group_idx+1}/{len(cache_groups)}: {cache_hash}")
            logger.info(f"  max_length={cache_info.get('max_length')}, "
                        f"dp_dim={cache_info.get('downprojection_dim')}, "
                        f"dataset={cache_info.get('dataset_name')}")
            logger.info(f"  {len(group_configs)} experiment(s) in this group")
            logger.info(f"{'='*60}")

        if use_cache and cache_hash != "__no_cache__":
            # === MULTIPROCESSING PATH (cached mode) ===
            # 1. Precompute cache in main process (multi-GPU LLM inference)
            cache = precompute_single_cache(cache_info, args.devices, cache_base_dir=cache_base_dir)
            torch.set_default_dtype(torch.float32)
            cache.close()  # Close in main process; workers reopen independently
            del cache

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 2. Serialize cache_info for worker processes (ensure Path -> str)
            serializable_cache_info = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in cache_info.items()
            }

            # 3. Create multiprocessing queues
            ctx = mp.get_context('spawn')
            job_queue = ctx.Queue()
            progress_queue = ctx.Queue()

            for config in group_configs:
                job_queue.put(config)

            # 4. Spawn worker processes (workers_per_gpu per GPU)
            processes = []
            for device in args.devices:
                n_workers = wpg_per_device[device]
                for worker_idx in range(n_workers):
                    p = ctx.Process(
                        target=worker_process_fn,
                        args=(device, job_queue, progress_queue, str(output_dir),
                              cache_hash, serializable_cache_info, args.gpu_cache,
                              cache_base_dir),
                        name=f"worker-{device}-{worker_idx}",
                    )
                    p.start()
                    processes.append(p)

            logger.info(f"Spawned {len(processes)} worker processes "
                        f"across {len(args.devices)} GPU(s)")

            # 5. Monitor progress from main process
            completed_in_group = 0
            expected = len(group_configs)
            while completed_in_group < expected:
                # Check for dead workers
                alive = [p for p in processes if p.is_alive()]
                if not alive and completed_in_group < expected:
                    logger.error(f"All workers died with {expected - completed_in_group} "
                                 f"experiments remaining")
                    break

                try:
                    status, config_hash, result = progress_queue.get(timeout=5)
                    results_dict[config_hash] = result
                    completed_in_group += 1
                    progress_bar.update(1)

                    if status == "done" and not result.get('skipped'):
                        metrics = result.get('metrics', {})
                        progress_bar.set_postfix_str(
                            f"{result.get('config', {}).get('model_type', '?')} "
                            f"ITE corr: {metrics.get('ite_corr', 'N/A'):.3f}"
                        )
                    elif result.get('skipped'):
                        progress_bar.set_postfix_str(
                            f"Skipped: {result.get('error', 'unknown')[:30]}"
                        )
                except Exception:
                    pass  # timeout, retry

            # 6. Join workers
            for p in processes:
                p.join(timeout=30)
                if p.is_alive():
                    logger.warning(f"Worker {p.name} did not exit cleanly, terminating")
                    p.terminate()

        else:
            # === THREADING PATH (non-cached / live LLM mode) ===
            cache_registry = {}
            lock = threading.Lock()

            job_queue_t = queue.Queue()
            for config in group_configs:
                job_queue_t.put(config)

            threads = []
            for device in args.devices:
                t = threading.Thread(
                    target=worker_thread,
                    args=(device, job_queue_t, results_dict, output_dir, lock, progress_bar,
                          cache_registry, {}),
                    name=f"worker-{device}"
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        # Group by config excluding repeat_index to aggregate across repeats
        group_cols = ['dataset_name', 'feature_extractor_type', 'model_type',
                      'flp_model_name', 'hlm_model_name',
                      'flp_max_length', 'flp_downprojection_dim',
                      'use_explicit_confounders', 'learning_rate', 'epochs']
        # Only group by columns that exist in the results
        group_cols = [c for c in group_cols if c in results_df.columns]

        metric_agg = {}
        for metric in ['ite_corr', 'ite_spearman_corr', 'ate_bias', 'propensity_auroc',
                        'ite_mse', 'ite_mae', 'ci_coverage', 'mean_ci_width']:
            if metric in results_df.columns:
                metric_agg[metric] = ['mean', 'std']

        summary = results_df.groupby(group_cols).agg(metric_agg)
        logger.info(f"\nSummary (mean +/- std across repeats):\n{summary}")

        summary.to_csv(output_dir / "summary.csv")

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Total experiments: {len(results_dict)}, "
               f"Successful: {len(all_results)}, "
               f"Skipped/Failed: {len(results_dict) - len(all_results)}")


if __name__ == "__main__":
    main()
