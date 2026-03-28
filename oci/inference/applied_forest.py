# oci/inference/applied_forest.py
"""Applied causal inference using two-stage neural + causal forest approach."""

import gc
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from ..config import (
    AppliedInferenceConfig,
    normalize_feature_extractor_type,
    TRAINABLE_EXTRACTOR_TYPES,
    CACHEABLE_EXTRACTOR_TYPES,
)
from ..models.causal_text_forest import CausalTextForest
from ..data import (
    ClinicalTextDataset,
    collate_batch,
    CachedHiddenStateDataset,
    collate_cached_batch,
    prepare_cached_batch,
)
from ..models.hidden_state_cache import HiddenStateCache
from ..utils import cuda_cleanup, get_memory_info


logger = logging.getLogger(__name__)


def run_applied_inference_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    num_workers: int = 1,
    verbose: bool = True,
    gpu_ids: Optional[List[int]] = None,
    explicit_confounder_columns: Optional[List[str]] = None,
) -> None:
    """
    Run applied causal inference using two-stage neural + causal forest approach.

    For each CV fold:
        1. Train feature extractor (propensity + outcome loss)
        2. Extract features for train and test
        3. Train causal forest on train features
        4. Predict ITE on test features

    Args:
        dataset: DataFrame with clinical text, outcomes, and treatments
        config: Configuration for applied inference
        output_path: Path to save predictions
        device: PyTorch device
        num_workers: Number of parallel workers (not used - sequential for memory)
        verbose: Print detailed logs
        gpu_ids: Optional list of GPU IDs for multi-GPU hidden state precomputation
    """
    logger.info("=" * 80)
    logger.info("APPLIED CAUSAL INFERENCE (CAUSAL FOREST)")
    logger.info("=" * 80)
    logger.info("Two-stage approach: Neural feature extraction + Causal Forest")

    # Pre-compute and cache LLM hidden states for cacheable extractors
    hidden_state_cache = None
    gpu_store = None
    arch_config = config.architecture
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'frozen_llm_pooler')
    )
    if feature_extractor_type in CACHEABLE_EXTRACTOR_TYPES:
        # Resolve extractor-specific config prefix (flp_ or hlm_)
        _prefix = "hlm" if feature_extractor_type == "hierarchical_llm" else "flp"
        _freeze = getattr(arch_config, f'{_prefix}_freeze_llm', True)
        _cache_enabled = getattr(arch_config, f'{_prefix}_cache_hidden_states', True)
        _gpu_cache = getattr(arch_config, f'{_prefix}_gpu_cache', False)

        dataset_path = config.dataset_path
        model_name = getattr(arch_config, f'{_prefix}_model_name', 'Qwen/Qwen3-0.6B-Base')
        max_length = getattr(arch_config, f'{_prefix}_max_length', 8192) if _prefix == "flp" else None
        batch_size = config.training.batch_size
        _downprojection_dim = getattr(arch_config, f'{_prefix}_downprojection_dim', None)
        _chat_template_prompt = getattr(arch_config, f'{_prefix}_chat_template_prompt', None)
        _random_projection_dim = getattr(arch_config, 'flp_random_projection_dim', None) if _prefix == "flp" else None

        # Reset index for consistent cache indices
        dataset = dataset.reset_index(drop=True)
        all_texts = dataset[config.text_column].tolist()

        # Try GPU cache first if requested
        if _gpu_cache and _freeze and device.type == "cuda":
            from ..models.gpu_hidden_state_store import GPUHiddenStateStore
            try:
                estimated_gb = GPUHiddenStateStore.estimate_vram_gb(
                    all_texts, model_name, max_length,
                    downprojection_dim=_downprojection_dim,
                    chat_template_prompt=_chat_template_prompt,
                )
                free_vram_gb = torch.cuda.mem_get_info(device)[0] / 1e9
                if estimated_gb < free_vram_gb * 0.8:
                    logger.info(
                        f"GPU cache: estimated {estimated_gb:.2f} GB, "
                        f"free VRAM {free_vram_gb:.1f} GB -- using GPU cache"
                    )
                    gpu_store = GPUHiddenStateStore()
                    gpu_store.precompute(
                        all_texts, model_name, max_length, device,
                        batch_size=batch_size,
                        downprojection_dim=_downprojection_dim,
                        chat_template_prompt=_chat_template_prompt,
                    )
                else:
                    logger.warning(
                        f"GPU cache needs ~{estimated_gb:.1f} GB but only "
                        f"{free_vram_gb:.1f} GB free. Falling back to disk cache."
                    )
            except Exception as e:
                logger.warning(f"GPU cache failed: {e}. Falling back to disk cache.")
                if gpu_store is not None:
                    gpu_store.free()
                    gpu_store = None

        # Fall back to disk cache if GPU cache not available
        if gpu_store is None and _cache_enabled and _freeze:
            cache_dir = str(Path(dataset_path).parent / ".oci_cache")
            cache_kwargs = dict(
                cache_dir=cache_dir,
                model_name=model_name,
                max_length=max_length,
                dataset_path=dataset_path,
                random_projection_dim=_random_projection_dim,
                downprojection_dim=_downprojection_dim,
                chat_template_prompt=_chat_template_prompt,
            )
            # Remove None max_length for hierarchical_llm (uses chunk_size instead)
            if max_length is None:
                del cache_kwargs['max_length']
            hidden_state_cache = HiddenStateCache(**cache_kwargs)

            if not hidden_state_cache.is_valid(len(dataset)):
                logger.info(f"Pre-computing {feature_extractor_type} hidden states for disk caching...")
                try:
                    # Use multi-GPU precomputation when multiple GPUs available
                    precompute_devices = [device]
                    if gpu_ids and device.type == "cuda":
                        precompute_devices = [torch.device(f"cuda:{i}") for i in gpu_ids]
                    if len(precompute_devices) > 1:
                        logger.info(f"Using {len(precompute_devices)} GPUs for parallel precomputation")
                        hidden_state_cache.precompute_multi_gpu(
                            all_texts, precompute_devices, batch_size=batch_size
                        )
                    else:
                        hidden_state_cache.precompute(all_texts, device, batch_size=batch_size)
                except Exception as e:
                    logger.warning(f"Hidden state caching failed: {e}. Falling back to non-cached mode.")
                    hidden_state_cache = None
            else:
                logger.info("Reusing existing hidden state cache")

            if hidden_state_cache is not None:
                hidden_state_cache.open()
                hidden_state_cache.preload_to_ram()
        elif gpu_store is None and _cache_enabled and not _freeze:
            logger.warning(
                f"{_prefix}_cache_hidden_states=True but {_prefix}_freeze_llm=False. "
                "Caching is only supported with frozen LLM. Skipping cache."
            )

    # Determine mode
    if config.cv_folds > 1:
        _run_cv_inference_forest(
            dataset, config, output_path, device, verbose,
            hidden_state_cache=hidden_state_cache,
            gpu_store=gpu_store
        )
    else:
        _run_fixed_split_inference_forest(
            dataset, config, output_path, device, verbose,
            hidden_state_cache=hidden_state_cache,
            gpu_store=gpu_store
        )

    # Cleanup
    if hidden_state_cache is not None:
        hidden_state_cache.close()
    if gpu_store is not None:
        gpu_store.free()


def _run_cv_inference_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    verbose: bool = True,
    hidden_state_cache: Optional[HiddenStateCache] = None,
    gpu_store=None
) -> None:
    """Run K-Fold Cross-Validation inference with causal forest."""
    k = config.cv_folds
    logger.info(f"Starting {k}-Fold Cross-Validation on {len(dataset)} samples")

    # Reset index (may already be reset if cache was initialized)
    dataset = dataset.reset_index(drop=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = list(kf.split(dataset))

    all_predictions = []
    all_training_logs = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"\n{'=' * 40}")
        logger.info(f"FOLD {fold + 1}/{k}")
        logger.info(f"{'=' * 40}")

        train_df = dataset.iloc[train_idx]
        test_df = dataset.iloc[test_idx]

        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

        # Train model and get predictions
        preds_df, history = _process_fold_forest(
            fold, train_df, test_df, config, device, verbose,
            hidden_state_cache=hidden_state_cache,
            train_indices=train_idx,
            test_indices=test_idx,
            gpu_store=gpu_store
        )

        # Add fold info
        for entry in history:
            entry['fold'] = fold + 1

        all_predictions.append(preds_df)
        all_training_logs.extend(history)

        # Cleanup
        gc.collect()
        cuda_cleanup()
        logger.info(f"FOLD {fold + 1} complete | {get_memory_info()}")

    # Combine predictions and save
    results_df = pd.concat(all_predictions).sort_index()
    _save_and_summarize_forest(results_df, output_path)

    # Save training logs
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(all_training_logs).to_csv(log_path, index=False)
    logger.info(f"Training logs saved to: {log_path}")


def _process_fold_forest(
    fold: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device,
    verbose: bool = True,
    hidden_state_cache: Optional[HiddenStateCache] = None,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    gpu_store=None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Process a single fold with causal forest."""
    arch_config = config.architecture
    train_config = config.training

    # Get feature extractor type
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'frozen_llm_pooler')
    )

    # Create model (with skip_llm if cache available for cacheable extractors)
    _using_cache = hidden_state_cache is not None or gpu_store is not None
    _cached_hidden_size = (
        gpu_store.hidden_size if gpu_store is not None
        else hidden_state_cache.hidden_size if hidden_state_cache is not None
        else 0
    )
    outcome_type = getattr(config, 'outcome_type', 'binary')
    model = _create_causal_forest_model(
        arch_config, device, outcome_type=outcome_type,
        flp_skip_llm=_using_cache and feature_extractor_type == "frozen_llm_pooler",
        flp_cached_hidden_size=(
            _cached_hidden_size if feature_extractor_type == "frozen_llm_pooler" else 0
        ),
        flp_downprojection_dim=(
            None if (_using_cache and feature_extractor_type == "frozen_llm_pooler")
            else getattr(arch_config, 'flp_downprojection_dim', None)
        ),
        hlm_skip_llm=_using_cache and feature_extractor_type == "hierarchical_llm",
        hlm_cached_hidden_size=(
            _cached_hidden_size if feature_extractor_type == "hierarchical_llm" else 0
        ),
        hlm_downprojection_dim=(
            None if (_using_cache and feature_extractor_type == "hierarchical_llm")
            else getattr(arch_config, 'hlm_downprojection_dim', None)
        ),
    )
    logger.info(f"Created CausalTextForest with {feature_extractor_type} extractor")
    if gpu_store is not None:
        logger.info(f"Using GPU-resident hidden states ({feature_extractor_type} LLM not loaded)")
    elif hidden_state_cache is not None:
        logger.info(f"Using cached hidden states ({feature_extractor_type} LLM not loaded)")

    # fit_tokenizer for trainable extractors (hierarchical_cnn, hierarchical_gru, simple_cnn)
    if feature_extractor_type in TRAINABLE_EXTRACTOR_TYPES:
        train_texts = train_df[config.text_column].tolist()
        logger.info(f"Fitting tokenizer for {extractor_type} on {len(train_texts)} texts...")
        model.fit_tokenizer(train_texts)

    # Stage 1: Train representation
    logger.info("\n--- Stage 1: Training representation ---")
    history = _train_representation(
        model, train_df, test_df, config, device, verbose,
        hidden_state_cache=hidden_state_cache,
        train_indices=train_indices,
        val_indices=test_indices,
        gpu_store=gpu_store
    )

    # Create DataLoaders for Stage 2 feature extraction and prediction
    if gpu_store is not None and train_indices is not None and test_indices is not None:
        # GPU cache mode: cache_index path
        train_dataset = CachedHiddenStateDataset(
            data=train_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=train_indices,
        )
        test_dataset = CachedHiddenStateDataset(
            data=test_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=test_indices,
        )
        collate_fn = collate_cached_batch
    elif hidden_state_cache is not None and train_indices is not None and test_indices is not None:
        cache_hs = hidden_state_cache.hidden_states_array
        cache_mask = hidden_state_cache.attention_mask_array
        train_dataset = CachedHiddenStateDataset(
            data=train_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=train_indices,
            cache_hidden_states=cache_hs,
            cache_attention_masks=cache_mask,
        )
        test_dataset = CachedHiddenStateDataset(
            data=test_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=test_indices,
            cache_hidden_states=cache_hs,
            cache_attention_masks=cache_mask,
        )
        collate_fn = collate_cached_batch
    else:
        train_dataset = ClinicalTextDataset(
            data=train_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )
        test_dataset = ClinicalTextDataset(
            data=test_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )
        collate_fn = collate_batch

    use_cached_mode = hidden_state_cache is not None and train_indices is not None
    if gpu_store is not None:
        dl_kwargs = {}
    elif use_cached_mode:
        dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True)
    else:
        dl_kwargs = {}

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs
    )

    # Stage 2: Train causal forest
    # Hidden states already loaded by DataLoader; prepare_cached_batch called in extract_features
    logger.info("\n--- Stage 2: Training causal forest ---")
    train_T = train_df[config.treatment_column].values
    train_Y = train_df[config.outcome_column].values
    model.train_causal_forest(train_loader, train_T, train_Y, gpu_store=gpu_store)

    # Predict on test
    logger.info("\n--- Generating predictions on test set ---")
    preds = model.predict(test_loader, return_ci=True, gpu_store=gpu_store)

    # Build predictions DataFrame
    preds_df = test_df.copy()
    preds_df['pred_y0_prob'] = preds['pred_y0_prob']
    preds_df['pred_y1_prob'] = preds['pred_y1_prob']
    preds_df['pred_ite_prob'] = preds['pred_ite_prob']
    preds_df['pred_propensity_prob'] = preds['pred_propensity_prob']
    preds_df['cv_fold'] = fold + 1

    # Add confidence intervals if available
    if 'tau_lower' in preds:
        preds_df['pred_ite_lower'] = preds['tau_lower']
        preds_df['pred_ite_upper'] = preds['tau_upper']

    # Cleanup
    model.cpu()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return preds_df, history


def _run_fixed_split_inference_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    verbose: bool = True,
    hidden_state_cache: Optional[HiddenStateCache] = None,
    gpu_store=None
) -> None:
    """Run inference using fixed train/val/test splits with causal forest."""
    logger.info("Running Fixed Split Inference (Train/Val/Test)")

    # Split data
    train_df = dataset[dataset[config.split_column] == 'train'].copy()
    val_df = dataset[dataset[config.split_column] == 'val'].copy()
    test_df = dataset[dataset[config.split_column] == 'test'].copy()

    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build index mappings for cache (dataset was reset_index in run_applied_inference_forest)
    train_indices = None
    val_indices = None
    test_indices = None
    if hidden_state_cache is not None or gpu_store is not None:
        train_mask = dataset[config.split_column] == 'train'
        val_mask = dataset[config.split_column] == 'val'
        test_mask = dataset[config.split_column] == 'test'
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

    arch_config = config.architecture
    train_config = config.training

    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'frozen_llm_pooler')
    )

    # Create model (with skip_llm if cache available for cacheable extractors)
    _using_cache = hidden_state_cache is not None or gpu_store is not None
    _cached_hidden_size = (
        gpu_store.hidden_size if gpu_store is not None
        else hidden_state_cache.hidden_size if hidden_state_cache is not None
        else 0
    )
    outcome_type = getattr(config, 'outcome_type', 'binary')
    model = _create_causal_forest_model(
        arch_config, device, outcome_type=outcome_type,
        flp_skip_llm=_using_cache and feature_extractor_type == "frozen_llm_pooler",
        flp_cached_hidden_size=(
            _cached_hidden_size if feature_extractor_type == "frozen_llm_pooler" else 0
        ),
        flp_downprojection_dim=(
            None if (_using_cache and feature_extractor_type == "frozen_llm_pooler")
            else getattr(arch_config, 'flp_downprojection_dim', None)
        ),
        hlm_skip_llm=_using_cache and feature_extractor_type == "hierarchical_llm",
        hlm_cached_hidden_size=(
            _cached_hidden_size if feature_extractor_type == "hierarchical_llm" else 0
        ),
        hlm_downprojection_dim=(
            None if (_using_cache and feature_extractor_type == "hierarchical_llm")
            else getattr(arch_config, 'hlm_downprojection_dim', None)
        ),
    )

    # fit_tokenizer for trainable extractors (hierarchical_cnn, hierarchical_gru, simple_cnn)
    if feature_extractor_type in TRAINABLE_EXTRACTOR_TYPES:
        train_texts = train_df[config.text_column].tolist()
        logger.info(f"Fitting tokenizer for {feature_extractor_type} on {len(train_texts)} texts...")
        model.fit_tokenizer(train_texts)

    # Stage 1: Train representation
    logger.info("\n--- Stage 1: Training representation ---")
    history = _train_representation(
        model, train_df, val_df, config, device, verbose,
        hidden_state_cache=hidden_state_cache,
        train_indices=train_indices,
        val_indices=val_indices,
        gpu_store=gpu_store
    )

    # Save training logs
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)

    # Stage 2: Train causal forest on full train + val
    logger.info("\n--- Stage 2: Training causal forest ---")
    combined_df = pd.concat([train_df, val_df])

    if gpu_store is not None and train_indices is not None and val_indices is not None:
        combined_indices = np.concatenate([train_indices, val_indices])
        combined_dataset = CachedHiddenStateDataset(
            data=combined_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=combined_indices,
        )
        combined_collate_fn = collate_cached_batch
    elif hidden_state_cache is not None and train_indices is not None and val_indices is not None:
        cache_hs = hidden_state_cache.hidden_states_array
        cache_mask = hidden_state_cache.attention_mask_array
        combined_indices = np.concatenate([train_indices, val_indices])
        combined_dataset = CachedHiddenStateDataset(
            data=combined_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=combined_indices,
            cache_hidden_states=cache_hs,
            cache_attention_masks=cache_mask,
        )
        combined_collate_fn = collate_cached_batch
    else:
        combined_dataset = ClinicalTextDataset(
            data=combined_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )
        combined_collate_fn = collate_batch

    use_cached_combined = hidden_state_cache is not None and train_indices is not None
    if gpu_store is not None:
        dl_kwargs_combined = {}
    elif use_cached_combined:
        dl_kwargs_combined = dict(num_workers=2, persistent_workers=True, pin_memory=True)
    else:
        dl_kwargs_combined = {}

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=combined_collate_fn,
        **dl_kwargs_combined
    )
    combined_T = combined_df[config.treatment_column].values
    combined_Y = combined_df[config.outcome_column].values
    model.train_causal_forest(combined_loader, combined_T, combined_Y, gpu_store=gpu_store)

    # Predict on test
    logger.info("\n--- Generating predictions on test set ---")
    if gpu_store is not None and test_indices is not None:
        test_dataset = CachedHiddenStateDataset(
            data=test_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=test_indices,
        )
        test_collate_fn = collate_cached_batch
    elif hidden_state_cache is not None and test_indices is not None:
        cache_hs = hidden_state_cache.hidden_states_array
        cache_mask = hidden_state_cache.attention_mask_array
        test_dataset = CachedHiddenStateDataset(
            data=test_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=test_indices,
            cache_hidden_states=cache_hs,
            cache_attention_masks=cache_mask,
        )
        test_collate_fn = collate_cached_batch
    else:
        test_dataset = ClinicalTextDataset(
            data=test_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )
        test_collate_fn = collate_batch

    use_cached_test = hidden_state_cache is not None and test_indices is not None
    if gpu_store is not None:
        dl_kwargs_test = {}
    elif use_cached_test:
        dl_kwargs_test = dict(num_workers=2, persistent_workers=True, pin_memory=True)
    else:
        dl_kwargs_test = {}

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=test_collate_fn,
        **dl_kwargs_test
    )
    preds = model.predict(test_loader, return_ci=True, gpu_store=gpu_store)

    # Build results DataFrame
    results_df = test_df.copy()
    results_df['pred_y0_prob'] = preds['pred_y0_prob']
    results_df['pred_y1_prob'] = preds['pred_y1_prob']
    results_df['pred_ite_prob'] = preds['pred_ite_prob']
    results_df['pred_propensity_prob'] = preds['pred_propensity_prob']

    if 'tau_lower' in preds:
        results_df['pred_ite_lower'] = preds['tau_lower']
        results_df['pred_ite_upper'] = preds['tau_upper']

    _save_and_summarize_forest(results_df, output_path)


def _create_causal_forest_model(
    arch_config,
    device: torch.device,
    outcome_type: str = "binary",
    flp_skip_llm: bool = False,
    flp_cached_hidden_size: int = 0,
    flp_downprojection_dim: Optional[int] = None,
    hlm_skip_llm: bool = False,
    hlm_cached_hidden_size: int = 0,
    hlm_downprojection_dim: Optional[int] = None,
) -> CausalTextForest:
    """Create CausalTextForest model from config."""
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'frozen_llm_pooler')
    )

    # Get causal forest config
    cf_config = getattr(arch_config, 'causal_forest', None)
    if cf_config is None:
        # Use defaults
        cf_n_estimators = 100
        cf_max_depth = None
        cf_min_samples_leaf = 5
        cf_max_features = "sqrt"
        cf_honest = True
        cf_inference = True
    else:
        cf_n_estimators = getattr(cf_config, 'n_estimators', 100)
        cf_max_depth = getattr(cf_config, 'max_depth', None)
        cf_min_samples_leaf = getattr(cf_config, 'min_samples_leaf', 5)
        cf_max_features = getattr(cf_config, 'max_features', "sqrt")
        cf_honest = getattr(cf_config, 'honest', True)
        cf_inference = getattr(cf_config, 'inference', True)

    # R-learner representation training options
    cf_use_rlearner_representation = getattr(cf_config, 'use_rlearner_representation', False) if cf_config else False
    cf_gamma_rlearner = getattr(cf_config, 'gamma_rlearner', 1.0) if cf_config else 1.0
    cf_rlearner_dual_extractors = getattr(cf_config, 'rlearner_dual_extractors', False) if cf_config else False

    model = CausalTextForest(
        feature_extractor_type=feature_extractor_type,
        # Frozen LLM Pooler args
        flp_model_name=getattr(arch_config, 'flp_model_name', 'Qwen/Qwen3-0.6B-Base'),
        flp_max_length=getattr(arch_config, 'flp_max_length', 8192),
        flp_freeze_llm=getattr(arch_config, 'flp_freeze_llm', True),
        flp_gated_attention_dim=getattr(arch_config, 'flp_gated_attention_dim', 128),
        flp_projection_dim=getattr(arch_config, 'flp_projection_dim', 128),
        flp_dropout=getattr(arch_config, 'flp_dropout', 0.1),
        flp_gradient_checkpointing=getattr(arch_config, 'flp_gradient_checkpointing', True),
        flp_downprojection_dim=flp_downprojection_dim,
        flp_skip_llm=flp_skip_llm,
        flp_cached_hidden_size=flp_cached_hidden_size,
        flp_chat_template_prompt=getattr(arch_config, 'flp_chat_template_prompt', None),
        # Hierarchical LLM args
        hlm_model_name=getattr(arch_config, 'hlm_model_name', 'Qwen/Qwen3-0.6B-Base'),
        hlm_chunk_size=getattr(arch_config, 'hlm_chunk_size', 2048),
        hlm_chunk_overlap=getattr(arch_config, 'hlm_chunk_overlap', 256),
        hlm_max_chunks=getattr(arch_config, 'hlm_max_chunks', 16),
        hlm_freeze_llm=getattr(arch_config, 'hlm_freeze_llm', True),
        hlm_gated_attention_dim=getattr(arch_config, 'hlm_gated_attention_dim', 128),
        hlm_projection_dim=getattr(arch_config, 'hlm_projection_dim', 128),
        hlm_dropout=getattr(arch_config, 'hlm_dropout', 0.1),
        hlm_gradient_checkpointing=getattr(arch_config, 'hlm_gradient_checkpointing', True),
        hlm_downprojection_dim=hlm_downprojection_dim,
        hlm_skip_llm=hlm_skip_llm,
        hlm_cached_hidden_size=hlm_cached_hidden_size,
        hlm_chat_template_prompt=getattr(arch_config, 'hlm_chat_template_prompt', None),
        # Hierarchical CNN args
        hcnn_embedding_dim=getattr(arch_config, 'hcnn_embedding_dim', 256),
        hcnn_conv_dim=getattr(arch_config, 'hcnn_conv_dim', 256),
        hcnn_kernel_size=getattr(arch_config, 'hcnn_kernel_size', 5),
        hcnn_num_conv_blocks=getattr(arch_config, 'hcnn_num_conv_blocks', 4),
        hcnn_chunk_size=getattr(arch_config, 'hcnn_chunk_size', 512),
        hcnn_chunk_overlap=getattr(arch_config, 'hcnn_chunk_overlap', 64),
        hcnn_max_chunks=getattr(arch_config, 'hcnn_max_chunks', 32),
        hcnn_vocab_size=getattr(arch_config, 'hcnn_vocab_size', 50000),
        hcnn_gated_attention_dim=getattr(arch_config, 'hcnn_gated_attention_dim', 128),
        hcnn_projection_dim=getattr(arch_config, 'hcnn_projection_dim', 128),
        hcnn_dropout=getattr(arch_config, 'hcnn_dropout', 0.1),
        # Hierarchical GRU args
        hgru_embedding_dim=getattr(arch_config, 'hgru_embedding_dim', 256),
        hgru_gru_hidden_dim=getattr(arch_config, 'hgru_gru_hidden_dim', 256),
        hgru_num_gru_layers=getattr(arch_config, 'hgru_num_gru_layers', 2),
        hgru_chunk_size=getattr(arch_config, 'hgru_chunk_size', 512),
        hgru_chunk_overlap=getattr(arch_config, 'hgru_chunk_overlap', 64),
        hgru_max_chunks=getattr(arch_config, 'hgru_max_chunks', 32),
        hgru_vocab_size=getattr(arch_config, 'hgru_vocab_size', 50000),
        hgru_gated_attention_dim=getattr(arch_config, 'hgru_gated_attention_dim', 128),
        hgru_projection_dim=getattr(arch_config, 'hgru_projection_dim', 128),
        hgru_dropout=getattr(arch_config, 'hgru_dropout', 0.1),
        # Simple CNN args
        scnn_embedding_dim=getattr(arch_config, 'scnn_embedding_dim', 256),
        scnn_conv_dim=getattr(arch_config, 'scnn_conv_dim', 256),
        scnn_kernel_size=getattr(arch_config, 'scnn_kernel_size', 5),
        scnn_num_conv_blocks=getattr(arch_config, 'scnn_num_conv_blocks', 4),
        scnn_max_length=getattr(arch_config, 'scnn_max_length', 10000),
        scnn_vocab_size=getattr(arch_config, 'scnn_vocab_size', 50000),
        scnn_gated_attention_dim=getattr(arch_config, 'scnn_gated_attention_dim', 128),
        scnn_projection_dim=getattr(arch_config, 'scnn_projection_dim', 128),
        scnn_dropout=getattr(arch_config, 'scnn_dropout', 0.1),
        # Head args
        representation_dim=getattr(arch_config, 'causal_head_representation_dim', 128),
        hidden_dim=getattr(arch_config, 'causal_head_hidden_outcome_dim', 64),
        dropout=getattr(arch_config, 'causal_head_dropout', 0.2),
        # Causal forest args
        cf_n_estimators=cf_n_estimators,
        cf_max_depth=cf_max_depth,
        cf_min_samples_leaf=cf_min_samples_leaf,
        cf_max_features=cf_max_features,
        cf_honest=cf_honest,
        cf_inference=cf_inference,
        # R-learner representation training
        cf_use_rlearner_representation=cf_use_rlearner_representation,
        cf_gamma_rlearner=cf_gamma_rlearner,
        cf_rlearner_dual_extractors=cf_rlearner_dual_extractors,
        # Device
        device=str(device),
        # Outcome type
        outcome_type=outcome_type
    )

    return model


def _train_representation(
    model: CausalTextForest,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device,
    verbose: bool = True,
    hidden_state_cache: Optional[HiddenStateCache] = None,
    train_indices: Optional[np.ndarray] = None,
    val_indices: Optional[np.ndarray] = None,
    gpu_store=None
) -> List[Dict[str, Any]]:
    """Train representation (Stage 1)."""
    train_config = config.training

    # Create datasets
    if gpu_store is not None and train_indices is not None and val_indices is not None:
        # GPU cache mode: cache_index path
        train_dataset = CachedHiddenStateDataset(
            data=train_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=train_indices,
        )
        val_dataset = CachedHiddenStateDataset(
            data=val_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=val_indices,
        )
        collate_fn = collate_cached_batch
        logger.info("Using GPU-resident hidden states for Stage 1 training")
    elif hidden_state_cache is not None and train_indices is not None and val_indices is not None:
        cache_hs = hidden_state_cache.hidden_states_array
        cache_mask = hidden_state_cache.attention_mask_array
        train_dataset = CachedHiddenStateDataset(
            data=train_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=train_indices,
            cache_hidden_states=cache_hs,
            cache_attention_masks=cache_mask,
        )
        val_dataset = CachedHiddenStateDataset(
            data=val_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            dataset_indices=val_indices,
            cache_hidden_states=cache_hs,
            cache_attention_masks=cache_mask,
        )
        collate_fn = collate_cached_batch
        logger.info("Using cached hidden state datasets for Stage 1 training")
    else:
        train_dataset = ClinicalTextDataset(
            data=train_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )
        val_dataset = ClinicalTextDataset(
            data=val_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )
        collate_fn = collate_batch

    use_cached_mode = hidden_state_cache is not None and train_indices is not None
    if gpu_store is not None:
        dl_kwargs = {}
    elif use_cached_mode:
        dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True)
    else:
        dl_kwargs = {}

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **dl_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dl_kwargs
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=getattr(train_config, 'weight_decay', 0.01)
    )

    # Scheduler
    if train_config.lr_schedule == "linear":
        total_steps = len(train_loader) * train_config.epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
    else:
        scheduler = None

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    history = []

    alpha_propensity = train_config.alpha_propensity
    label_smoothing = getattr(train_config, 'label_smoothing', 0.0)
    stop_grad_propensity = getattr(train_config, 'stop_grad_propensity', False)
    gradient_clip_norm = getattr(train_config, 'gradient_clip_norm', 0.0)

    # R-learner representation training: get gamma from model config
    gamma_rlearner = model.cf_gamma_rlearner if model.use_rlearner_representation else 0.0

    for epoch in range(train_config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_prop_loss = 0.0
        train_outcome_loss = 0.0
        train_r_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, disable=not verbose):
            batch['outcome'] = batch['outcome'].to(device)
            batch['treatment'] = batch['treatment'].to(device)

            prepare_cached_batch(batch, device, hidden_state_cache, gpu_store=gpu_store)

            optimizer.zero_grad()

            losses = model.train_representation_step(
                batch,
                alpha_propensity=alpha_propensity,
                gamma_rlearner=gamma_rlearner,
                label_smoothing=label_smoothing,
                stop_grad_propensity=stop_grad_propensity,
            )

            losses['loss'].backward()

            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_loss += losses['loss'].item()
            train_prop_loss += losses['propensity_loss'].item()
            train_outcome_loss += losses['outcome_loss'].item()
            train_r_loss += losses.get('r_loss', torch.tensor(0.0)).item()

        train_loss /= len(train_loader)
        train_prop_loss /= len(train_loader)
        train_outcome_loss /= len(train_loader)
        train_r_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_prop_loss = 0.0
        val_outcome_loss = 0.0
        val_r_loss = 0.0
        all_prop_logits = []
        all_outcome_logits = []
        all_treatments = []
        all_outcomes = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False, disable=not verbose):
                batch['outcome'] = batch['outcome'].to(device)
                batch['treatment'] = batch['treatment'].to(device)

                prepare_cached_batch(batch, device, hidden_state_cache, gpu_store=gpu_store)

                losses = model.train_representation_step(
                    batch,
                    alpha_propensity=alpha_propensity,
                    gamma_rlearner=gamma_rlearner,
                    stop_grad_propensity=stop_grad_propensity,
                )

                val_loss += losses['loss'].item()
                val_prop_loss += losses['propensity_loss'].item()
                val_outcome_loss += losses['outcome_loss'].item()
                val_r_loss += losses.get('r_loss', torch.tensor(0.0)).item()

                all_prop_logits.append(losses['propensity_logit'].cpu())
                all_outcome_logits.append(losses['outcome_logit'].cpu())
                all_treatments.append(batch['treatment'].cpu())
                all_outcomes.append(batch['outcome'].cpu())

        val_loss /= len(val_loader)
        val_prop_loss /= len(val_loader)
        val_outcome_loss /= len(val_loader)
        val_r_loss /= len(val_loader)

        # Compute metrics
        prop_scores = torch.sigmoid(torch.cat(all_prop_logits)).numpy().flatten()
        outcome_type = getattr(model, 'outcome_type', 'binary')
        if outcome_type == "continuous":
            outcome_scores = torch.cat(all_outcome_logits).numpy().flatten()
        else:
            outcome_scores = torch.sigmoid(torch.cat(all_outcome_logits)).numpy().flatten()
        treatments = torch.cat(all_treatments).numpy()
        outcomes = torch.cat(all_outcomes).numpy()

        try:
            val_auroc_prop = roc_auc_score(treatments, prop_scores)
        except:
            val_auroc_prop = None

        if outcome_type == "continuous":
            from sklearn.metrics import r2_score, mean_squared_error
            try:
                val_outcome_metric = r2_score(outcomes, outcome_scores) if len(outcomes) >= 2 else None
            except:
                val_outcome_metric = None
        else:
            try:
                val_outcome_metric = roc_auc_score(outcomes, outcome_scores)
            except:
                val_outcome_metric = None

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_propensity_loss': train_prop_loss,
            'train_outcome_loss': train_outcome_loss,
            'train_r_loss': train_r_loss,
            'val_loss': val_loss,
            'val_propensity_loss': val_prop_loss,
            'val_outcome_loss': val_outcome_loss,
            'val_r_loss': val_r_loss,
            'val_auroc_prop': val_auroc_prop,
            'val_auroc_outcome': val_outcome_metric,
        }
        history.append(epoch_log)

        if verbose:
            r_loss_str = f" | R-Loss: {train_r_loss:.4f}" if model.use_rlearner_representation else ""
            logger.info(
                f"Epoch {epoch+1}/{train_config.epochs} | "
                f"Train Loss: {train_loss:.4f}{r_loss_str} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUROC Prop: {f'{val_auroc_prop:.4f}' if val_auroc_prop else 'N/A'} | "
                f"Val AUROC Outcome: {f'{val_outcome_metric:.4f}' if val_outcome_metric else 'N/A'}"
            )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Cleanup
    del train_loader, val_loader, train_dataset, val_dataset
    del optimizer, best_model_state
    if scheduler is not None:
        del scheduler
    gc.collect()

    return history


def _save_and_summarize_forest(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save results and print summary for causal forest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)

    logger.info(f"Predictions saved to: {output_path}")
    logger.info("\nPrediction Summary (Causal Forest):")
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
        # Report CI coverage stats
        significant = (results_df['pred_ite_lower'] > 0) | (results_df['pred_ite_upper'] < 0)
        logger.info(f"    Significant effects (CI excludes 0): {significant.sum()} ({significant.mean()*100:.1f}%)")

    logger.info(f"  Mean predicted propensity: {results_df['pred_propensity_prob'].mean():.4f}")
