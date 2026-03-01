# cdt/inference/applied_forest.py
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

from ..config import AppliedInferenceConfig, normalize_feature_extractor_type
from ..models.causal_text_forest import CausalTextForest
from ..data import (
    ClinicalTextDataset,
    collate_batch,
    create_collator,
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
    verbose: bool = True
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
    """
    logger.info("=" * 80)
    logger.info("APPLIED CAUSAL INFERENCE (CAUSAL FOREST)")
    logger.info("=" * 80)
    logger.info("Two-stage approach: Neural feature extraction + Causal Forest")

    # Pre-compute and cache LLM hidden states for frozen_llm_pooler
    hidden_state_cache = None
    arch_config = config.architecture
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'gru_pool')
    )
    if feature_extractor_type == "frozen_llm_pooler":
        flp_freeze = getattr(arch_config, 'flp_freeze_llm', True)
        flp_cache_enabled = getattr(arch_config, 'flp_cache_hidden_states', True)

        if flp_cache_enabled and flp_freeze:
            dataset_path = config.dataset_path
            model_name = getattr(arch_config, 'flp_model_name', 'Qwen/Qwen3-0.6B-Base')
            max_length = getattr(arch_config, 'flp_max_length', 8192)

            cache_dir = str(Path(dataset_path).parent / ".cdt_cache")
            hidden_state_cache = HiddenStateCache(
                cache_dir=cache_dir,
                model_name=model_name,
                max_length=max_length,
                dataset_path=dataset_path,
            )

            # Reset index for consistent cache indices
            dataset = dataset.reset_index(drop=True)
            all_texts = dataset[config.text_column].tolist()

            if not hidden_state_cache.is_valid(len(dataset)):
                logger.info("Pre-computing LLM hidden states for caching...")
                batch_size = config.training.batch_size
                try:
                    hidden_state_cache.precompute(all_texts, device, batch_size=batch_size)
                except Exception as e:
                    logger.warning(f"Hidden state caching failed: {e}. Falling back to non-cached mode.")
                    hidden_state_cache = None
            else:
                logger.info("Reusing existing hidden state cache")

            if hidden_state_cache is not None:
                hidden_state_cache.open()
                hidden_state_cache.preload_to_ram()
        elif flp_cache_enabled and not flp_freeze:
            logger.warning(
                "flp_cache_hidden_states=True but flp_freeze_llm=False. "
                "Caching is only supported with frozen LLM. Skipping cache."
            )

    # Determine mode
    if config.cv_folds > 1:
        _run_cv_inference_forest(
            dataset, config, output_path, device, verbose,
            hidden_state_cache=hidden_state_cache
        )
    else:
        _run_fixed_split_inference_forest(
            dataset, config, output_path, device, verbose,
            hidden_state_cache=hidden_state_cache
        )

    # Cleanup cache
    if hidden_state_cache is not None:
        hidden_state_cache.close()


def _run_cv_inference_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    verbose: bool = True,
    hidden_state_cache: Optional[HiddenStateCache] = None
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
            test_indices=test_idx
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
    test_indices: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Process a single fold with causal forest."""
    arch_config = config.architecture
    train_config = config.training

    # Get feature extractor type
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'gru_pool')
    )

    # Create model (with skip_llm if cache available)
    outcome_type = getattr(config, 'outcome_type', 'binary')
    model = _create_causal_forest_model(
        arch_config, device, outcome_type=outcome_type,
        flp_skip_llm=hidden_state_cache is not None,
        flp_cached_hidden_size=hidden_state_cache.hidden_size if hidden_state_cache is not None else 0
    )
    logger.info(f"Created CausalTextForest with {feature_extractor_type.upper()} extractor")
    if hidden_state_cache is not None:
        logger.info("Using cached hidden states (LLM not loaded)")

    # Get training texts for tokenizer fitting
    train_texts = train_df[config.text_column].tolist()

    # Fit tokenizer if needed
    model.fit_tokenizer(train_texts)
    logger.info(f"Initialized feature extractor on {len(train_texts)} training texts")

    # Stage 1: Train representation
    logger.info("\n--- Stage 1: Training representation ---")
    history = _train_representation(
        model, train_df, test_df, config, device, verbose,
        hidden_state_cache=hidden_state_cache,
        train_indices=train_indices,
        val_indices=test_indices
    )

    # Create DataLoaders for Stage 2 feature extraction and prediction
    if hidden_state_cache is not None and train_indices is not None and test_indices is not None:
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
        collator = create_collator(model.feature_extractor)
        collate_fn = collator if collator is not None else collate_batch

    use_cached_mode = hidden_state_cache is not None and train_indices is not None
    dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True) if use_cached_mode else {}

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
    model.train_causal_forest(train_loader, train_T, train_Y)

    # Predict on test
    logger.info("\n--- Generating predictions on test set ---")
    preds = model.predict(test_loader, return_ci=True)

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
    hidden_state_cache: Optional[HiddenStateCache] = None
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
    if hidden_state_cache is not None:
        train_mask = dataset[config.split_column] == 'train'
        val_mask = dataset[config.split_column] == 'val'
        test_mask = dataset[config.split_column] == 'test'
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

    arch_config = config.architecture
    train_config = config.training

    # Create model (with skip_llm if cache available)
    outcome_type = getattr(config, 'outcome_type', 'binary')
    model = _create_causal_forest_model(
        arch_config, device, outcome_type=outcome_type,
        flp_skip_llm=hidden_state_cache is not None,
        flp_cached_hidden_size=hidden_state_cache.hidden_size if hidden_state_cache is not None else 0
    )

    # Get texts for tokenizer fitting
    train_texts = train_df[config.text_column].tolist()

    # Fit tokenizer
    model.fit_tokenizer(train_texts)

    # Stage 1: Train representation
    logger.info("\n--- Stage 1: Training representation ---")
    history = _train_representation(
        model, train_df, val_df, config, device, verbose,
        hidden_state_cache=hidden_state_cache,
        train_indices=train_indices,
        val_indices=val_indices
    )

    # Save training logs
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)

    # Stage 2: Train causal forest on full train + val
    logger.info("\n--- Stage 2: Training causal forest ---")
    combined_df = pd.concat([train_df, val_df])

    if hidden_state_cache is not None and train_indices is not None and val_indices is not None:
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
        collator = create_collator(model.feature_extractor)
        combined_collate_fn = collator if collator is not None else collate_batch

    use_cached_combined = hidden_state_cache is not None and train_indices is not None
    dl_kwargs_combined = dict(num_workers=2, persistent_workers=True, pin_memory=True) if use_cached_combined else {}

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=combined_collate_fn,
        **dl_kwargs_combined
    )
    combined_T = combined_df[config.treatment_column].values
    combined_Y = combined_df[config.outcome_column].values
    model.train_causal_forest(combined_loader, combined_T, combined_Y)

    # Predict on test
    logger.info("\n--- Generating predictions on test set ---")
    if hidden_state_cache is not None and test_indices is not None:
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
        collator = create_collator(model.feature_extractor)
        test_collate_fn = collator if collator is not None else collate_batch

    use_cached_test = hidden_state_cache is not None and test_indices is not None
    dl_kwargs_test = dict(num_workers=2, persistent_workers=True, pin_memory=True) if use_cached_test else {}

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=test_collate_fn,
        **dl_kwargs_test
    )
    preds = model.predict(test_loader, return_ci=True)

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
    flp_cached_hidden_size: int = 0
) -> CausalTextForest:
    """Create CausalTextForest model from config."""
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'gru_pool')
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
        # CNN args
        embedding_dim=getattr(arch_config, 'cnn_embedding_dim', 128),
        kernel_sizes=getattr(arch_config, 'cnn_kernel_sizes', [3, 4, 5, 7]),
        explicit_filter_concepts=getattr(arch_config, 'cnn_explicit_filter_concepts', None),
        num_kmeans_filters=getattr(arch_config, 'cnn_num_kmeans_filters', 64),
        num_random_filters=getattr(arch_config, 'cnn_num_random_filters', 0),
        cnn_dropout=getattr(arch_config, 'cnn_dropout', 0.1),
        max_length=getattr(arch_config, 'cnn_max_length', 2048),
        min_word_freq=getattr(arch_config, 'cnn_min_word_freq', 2),
        max_vocab_size=getattr(arch_config, 'cnn_max_vocab_size', 50000),
        # BERT args
        bert_model_name=getattr(arch_config, 'bert_model_name', 'bert-base-uncased'),
        bert_max_length=getattr(arch_config, 'bert_max_length', 512),
        bert_projection_dim=getattr(arch_config, 'bert_projection_dim', 128),
        bert_dropout=getattr(arch_config, 'bert_dropout', 0.1),
        bert_freeze_encoder=getattr(arch_config, 'bert_freeze_encoder', False),
        bert_gradient_checkpointing=getattr(arch_config, 'bert_gradient_checkpointing', False),
        # GRU args
        gru_hidden_dim=getattr(arch_config, 'gru_hidden_dim', 256),
        gru_num_layers=getattr(arch_config, 'gru_num_layers', 2),
        gru_dropout=getattr(arch_config, 'gru_dropout', 0.1),
        gru_bidirectional=getattr(arch_config, 'gru_bidirectional', True),
        gru_attention_dim=getattr(arch_config, 'gru_attention_dim', None),
        gru_projection_dim=getattr(arch_config, 'gru_projection_dim', 128),
        # Hierarchical Transformer args
        hier_transformer_sentence_model=getattr(arch_config, 'hier_transformer_sentence_model', 'prajjwal1/bert-tiny'),
        hier_transformer_freeze_sentence_encoder=getattr(arch_config, 'hier_transformer_freeze_sentence_encoder', True),
        hier_transformer_max_chunks=getattr(arch_config, 'hier_transformer_max_chunks', 100),
        hier_transformer_chunk_size=getattr(arch_config, 'hier_transformer_chunk_size', 128),
        hier_transformer_chunk_overlap=getattr(arch_config, 'hier_transformer_chunk_overlap', 32),
        hier_transformer_num_layers=getattr(arch_config, 'hier_transformer_num_layers', 2),
        hier_transformer_num_heads=getattr(arch_config, 'hier_transformer_num_heads', 4),
        hier_transformer_dim=getattr(arch_config, 'hier_transformer_dim', 256),
        hier_transformer_dropout=getattr(arch_config, 'hier_transformer_dropout', 0.1),
        hier_transformer_projection_dim=getattr(arch_config, 'hier_transformer_projection_dim', 128),
        # Gated MIL Hierarchical args
        gated_mil_sentence_model=getattr(arch_config, 'gated_mil_sentence_model', 'prajjwal1/bert-tiny'),
        gated_mil_freeze_sentence_encoder=getattr(arch_config, 'gated_mil_freeze_sentence_encoder', True),
        gated_mil_max_chunks=getattr(arch_config, 'gated_mil_max_chunks', 100),
        gated_mil_chunk_size=getattr(arch_config, 'gated_mil_chunk_size', 128),
        gated_mil_chunk_overlap=getattr(arch_config, 'gated_mil_chunk_overlap', 32),
        gated_mil_hidden_dim=getattr(arch_config, 'gated_mil_hidden_dim', 128),
        gated_mil_num_confounders=getattr(arch_config, 'gated_mil_num_confounders', 4),
        gated_mil_dropout=getattr(arch_config, 'gated_mil_dropout', 0.1),
        gated_mil_projection_dim=getattr(arch_config, 'gated_mil_projection_dim', 128),
        gated_mil_hierarchical=getattr(arch_config, 'gated_mil_hierarchical', False),
        gated_mil_token_hidden_dim=getattr(arch_config, 'gated_mil_token_hidden_dim', 64),
        gated_mil_use_mean_pooling=getattr(arch_config, 'gated_mil_use_mean_pooling', False),
        # GRU-Pool args
        gru_pool_embedding_dim=getattr(arch_config, 'gru_pool_embedding_dim', 128),
        gru_pool_gru_hidden_dim=getattr(arch_config, 'gru_pool_gru_hidden_dim', 128),
        gru_pool_gru_num_layers=getattr(arch_config, 'gru_pool_gru_num_layers', 1),
        gru_pool_gru_bidirectional=getattr(arch_config, 'gru_pool_gru_bidirectional', True),
        gru_pool_gru_dropout=getattr(arch_config, 'gru_pool_gru_dropout', 0.1),
        gru_pool_max_chunks=getattr(arch_config, 'gru_pool_max_chunks', 100),
        gru_pool_chunk_size=getattr(arch_config, 'gru_pool_chunk_size', 128),
        gru_pool_chunk_overlap=getattr(arch_config, 'gru_pool_chunk_overlap', 32),
        gru_pool_transformer_layers=getattr(arch_config, 'gru_pool_transformer_layers', 2),
        gru_pool_transformer_heads=getattr(arch_config, 'gru_pool_transformer_heads', 4),
        gru_pool_transformer_dim=getattr(arch_config, 'gru_pool_transformer_dim', 256),
        gru_pool_gated_attention_dim=getattr(arch_config, 'gru_pool_gated_attention_dim', 128),
        gru_pool_projection_dim=getattr(arch_config, 'gru_pool_projection_dim', 128),
        gru_pool_max_vocab=getattr(arch_config, 'gru_pool_max_vocab', 50000),
        gru_pool_min_word_freq=getattr(arch_config, 'gru_pool_min_word_freq', 2),
        # Transformer Pool args
        tp_embedding_dim=getattr(arch_config, 'tp_embedding_dim', 128),
        tp_token_transformer_layers=getattr(arch_config, 'tp_token_transformer_layers', 2),
        tp_token_transformer_heads=getattr(arch_config, 'tp_token_transformer_heads', 4),
        tp_token_transformer_dim=getattr(arch_config, 'tp_token_transformer_dim', 256),
        tp_token_transformer_dropout=getattr(arch_config, 'tp_token_transformer_dropout', 0.1),
        tp_chunk_transformer_layers=getattr(arch_config, 'tp_chunk_transformer_layers', 2),
        tp_chunk_transformer_heads=getattr(arch_config, 'tp_chunk_transformer_heads', 4),
        tp_chunk_transformer_dim=getattr(arch_config, 'tp_chunk_transformer_dim', 256),
        tp_chunk_transformer_dropout=getattr(arch_config, 'tp_chunk_transformer_dropout', 0.1),
        tp_gated_attention_dim=getattr(arch_config, 'tp_gated_attention_dim', 128),
        tp_projection_dim=getattr(arch_config, 'tp_projection_dim', 128),
        tp_chunk_size=getattr(arch_config, 'tp_chunk_size', 128),
        tp_chunk_overlap=getattr(arch_config, 'tp_chunk_overlap', 32),
        tp_max_chunks=getattr(arch_config, 'tp_max_chunks', 100),
        tp_max_vocab=getattr(arch_config, 'tp_max_vocab', 50000),
        tp_min_word_freq=getattr(arch_config, 'tp_min_word_freq', 2),
        # BERT Pool args
        bert_pool_sentence_model=getattr(arch_config, 'bert_pool_sentence_model', 'prajjwal1/bert-tiny'),
        bert_pool_freeze_sentence_encoder=getattr(arch_config, 'bert_pool_freeze_sentence_encoder', False),
        bert_pool_use_pretrained=getattr(arch_config, 'bert_pool_use_pretrained', True),
        bert_pool_max_chunks=getattr(arch_config, 'bert_pool_max_chunks', 100),
        bert_pool_chunk_size=getattr(arch_config, 'bert_pool_chunk_size', 128),
        bert_pool_chunk_overlap=getattr(arch_config, 'bert_pool_chunk_overlap', 32),
        bert_pool_transformer_layers=getattr(arch_config, 'bert_pool_transformer_layers', 2),
        bert_pool_transformer_heads=getattr(arch_config, 'bert_pool_transformer_heads', 4),
        bert_pool_transformer_dim=getattr(arch_config, 'bert_pool_transformer_dim', 256),
        bert_pool_transformer_dropout=getattr(arch_config, 'bert_pool_transformer_dropout', 0.1),
        bert_pool_gated_attention_dim=getattr(arch_config, 'bert_pool_gated_attention_dim', 128),
        bert_pool_projection_dim=getattr(arch_config, 'bert_pool_projection_dim', 128),
        # LLM args
        llm_model_name=getattr(arch_config, 'llm_model_name', 'Qwen/Qwen3-0.6B-Base'),
        llm_max_length=getattr(arch_config, 'llm_max_length', 8192),
        llm_projection_dim=getattr(arch_config, 'llm_projection_dim', 128),
        llm_dropout=getattr(arch_config, 'llm_dropout', 0.1),
        llm_gradient_checkpointing=getattr(arch_config, 'llm_gradient_checkpointing', True),
        llm_use_pretrained=getattr(arch_config, 'llm_use_pretrained', False),
        # Frozen LLM Pooler args
        flp_model_name=getattr(arch_config, 'flp_model_name', 'Qwen/Qwen3-0.6B-Base'),
        flp_max_length=getattr(arch_config, 'flp_max_length', 8192),
        flp_freeze_llm=getattr(arch_config, 'flp_freeze_llm', True),
        flp_gated_attention_dim=getattr(arch_config, 'flp_gated_attention_dim', 128),
        flp_projection_dim=getattr(arch_config, 'flp_projection_dim', 128),
        flp_dropout=getattr(arch_config, 'flp_dropout', 0.1),
        flp_gradient_checkpointing=getattr(arch_config, 'flp_gradient_checkpointing', True),
        flp_skip_llm=flp_skip_llm,
        flp_cached_hidden_size=flp_cached_hidden_size,
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
        # Numeric feature args
        numeric_features_enabled=getattr(arch_config, 'numeric_features_enabled', False),
        numeric_embedding_dim=getattr(arch_config, 'numeric_embedding_dim', 32),
        numeric_magnitude_bins=getattr(arch_config, 'numeric_magnitude_bins', 8),
        numeric_type_categories=getattr(arch_config, 'numeric_type_categories', 10),
        # Contrastive learning args
        contrastive_enabled=getattr(arch_config, 'contrastive_enabled', False),
        contrastive_num_clusters=getattr(arch_config, 'contrastive_num_clusters', 4),
        contrastive_temperature=getattr(arch_config, 'contrastive_temperature', 0.1),
        contrastive_label_mode=getattr(arch_config, 'contrastive_label_mode', 'joint'),
        contrastive_projection_dim=getattr(arch_config, 'contrastive_projection_dim', 64),
        contrastive_min_cluster_size=getattr(arch_config, 'contrastive_min_cluster_size', 2),
        contrastive_clustering_method=getattr(arch_config, 'contrastive_clustering_method', 'kmeans'),
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
    val_indices: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """Train representation (Stage 1)."""
    train_config = config.training

    # Create datasets
    if hidden_state_cache is not None and train_indices is not None and val_indices is not None:
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
        collator = create_collator(model.feature_extractor)
        collate_fn = collator if collator is not None else collate_batch

    use_cached_mode = hidden_state_cache is not None and train_indices is not None
    dl_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=True) if use_cached_mode else {}

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
    contrastive_weight = getattr(train_config, 'contrastive_weight', 0.1)

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

            prepare_cached_batch(batch, device, hidden_state_cache)

            optimizer.zero_grad()

            losses = model.train_representation_step(
                batch,
                alpha_propensity=alpha_propensity,
                gamma_rlearner=gamma_rlearner,
                label_smoothing=label_smoothing,
                stop_grad_propensity=stop_grad_propensity,
                contrastive_weight=contrastive_weight
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

                prepare_cached_batch(batch, device, hidden_state_cache)

                losses = model.train_representation_step(
                    batch,
                    alpha_propensity=alpha_propensity,
                    gamma_rlearner=gamma_rlearner,
                    stop_grad_propensity=stop_grad_propensity,
                    contrastive_weight=contrastive_weight
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
                f"Val AUROC Prop: {val_auroc_prop:.4f if val_auroc_prop else 'N/A'} | "
                f"Val AUROC Outcome: {val_auroc_outcome:.4f if val_auroc_outcome else 'N/A'}"
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
