# cdt/inference/matched_pair_applied.py
"""Applied causal inference using matched pair ITE estimation.

This module provides CV and fixed-split inference pipelines using the
matched pair approach:

1. Train propensity model on training data
2. Match patients by propensity score or embedding similarity
3. Train outcome/tau model on matched pairs only
4. Predict ITE for held-out test data

The key insight: propensity model and matching are done on TRAINING data only.
The learned tau model then predicts ITE on TEST data (which were never matched).
"""

import gc
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

from ..config import AppliedInferenceConfig, MatchedPairConfig
from ..models.matched_pair_ite import (
    PropensityMatchingModel,
    MatchedPairOutcomeModel,
    EnhancedMatchedPairOutcomeModel,
    CombinedMatchedPairModel,
    EndToEndMatchedPairModel,
    MeanEmbeddingITEModel
)
from ..training.matched_pair_training import (
    train_propensity_model,
    train_matched_pair_outcome_model,
    extract_all_representations,
    extract_propensity_scores,
    train_matched_pair_outcome_model_enhanced,
    train_end_to_end_matched_pair,
    train_mean_embedding_ite_model,
)
from ..matching import PropensityMatcher, match_by_cosine_similarity
from ..data import ClinicalTextDataset, collate_batch
from ..utils import cuda_cleanup, get_memory_info


logger = logging.getLogger(__name__)


def run_matched_pair_applied_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    matched_pair_config: MatchedPairConfig,
    output_path: Path,
    device: torch.device,
    gpu_ids: Optional[List[int]] = None,
    num_workers: int = 1
) -> None:
    """
    Run applied inference using matched pair ITE estimation.

    Mirrors the structure of run_applied_inference() but uses the
    two-stage propensity matching approach instead of DragonNet.

    Args:
        dataset: DataFrame with clinical text, outcomes, treatments
        config: AppliedInferenceConfig for data columns
        matched_pair_config: MatchedPairConfig for matched pair settings
        output_path: Path to save predictions
        device: PyTorch device
        gpu_ids: List of GPU IDs for parallel processing
        num_workers: Number of parallel workers
    """
    logger.info("=" * 80)
    logger.info("APPLIED CAUSAL INFERENCE (MATCHED PAIR ITE)")
    logger.info("=" * 80)

    # Use matched_pair_config column settings, falling back to config
    text_col = matched_pair_config.text_column or config.text_column
    treatment_col = matched_pair_config.treatment_column or config.treatment_column
    outcome_col = matched_pair_config.outcome_column or config.outcome_column

    # Check CV mode
    cv_folds = matched_pair_config.cv_folds
    if cv_folds > 1:
        _run_matched_pair_cv_inference(
            dataset, config, matched_pair_config, output_path,
            device, gpu_ids, num_workers
        )
    else:
        _run_matched_pair_fixed_split_inference(
            dataset, config, matched_pair_config, output_path, device
        )


def _run_matched_pair_cv_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    mp_config: MatchedPairConfig,
    output_path: Path,
    device: torch.device,
    gpu_ids: Optional[List[int]] = None,
    num_workers: int = 1
) -> None:
    """
    Run K-Fold Cross-Validation with matched pair estimation.

    For each fold:
    1. Train propensity model on training fold
    2. Match patients within training fold
    3. Train outcome/tau model on matched pairs
    4. Predict ITE for held-out test fold

    The key insight: propensity model and matching are done on TRAINING data only.
    The learned tau model then predicts ITE on TEST data (which were never matched).
    """
    k = mp_config.cv_folds
    logger.info(f"Starting {k}-Fold CV with Matched Pair ITE on {len(dataset)} samples")

    dataset = dataset.reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = list(kf.split(dataset))

    # Determine devices to use
    if gpu_ids:
        devices = [torch.device(f"cuda:{i}") for i in gpu_ids]
    else:
        devices = [device]

    if num_workers > 1:
        logger.info(f"Parallelizing across {num_workers} workers on devices: {devices}")
        results = Parallel(n_jobs=num_workers)(
            delayed(_process_matched_pair_fold)(
                fold, train_idx, test_idx, dataset, config, mp_config,
                devices[fold % len(devices)]
            )
            for fold, (train_idx, test_idx) in enumerate(splits)
        )
    else:
        results = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            results.append(_process_matched_pair_fold(
                fold, train_idx, test_idx, dataset, config, mp_config,
                devices[0]
            ))

    # Combine predictions
    all_predictions = [r[0] for r in results]
    all_training_logs = [log for r in results for log in r[1]]
    all_match_stats = [r[2] for r in results]

    results_df = pd.concat(all_predictions).sort_index()
    _save_matched_pair_results(results_df, all_match_stats, output_path)

    # Save training logs
    log_path = output_path.parent / "matched_pair_training_log.csv"
    pd.DataFrame(all_training_logs).to_csv(log_path, index=False)
    logger.info(f"Training logs saved to: {log_path}")


def _process_matched_pair_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    mp_config: MatchedPairConfig,
    device: torch.device
) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Process a single CV fold with matched pair estimation.

    If end_to_end_training is enabled, uses a single unified model with
    joint training. Otherwise, uses the 3-stage approach:

    3-Stage Approach:
    1. Train propensity model on training data
    2. Extract representations for training data
    3. Match patients within training data
    4. Train outcome/tau model on matched pairs only
    5. Predict ITE for test data

    End-to-End Approach:
    1. Create unified model
    2. Train with periodic re-matching
    3. Predict ITE for test data

    Returns:
        Tuple of (test_predictions_df, training_logs, match_statistics)
    """
    # Re-configure logger for worker process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"FOLD {fold + 1}: Matched Pair ITE starting on {device}")

    train_df = dataset.iloc[train_idx].reset_index(drop=True)
    test_df = dataset.iloc[test_idx]  # Keep original indices

    # Get column names
    text_col = mp_config.text_column
    treatment_col = mp_config.treatment_column
    outcome_col = mp_config.outcome_column

    # Branch based on training mode
    if mp_config.end_to_end_training:
        return _process_matched_pair_fold_e2e(
            fold, train_df, test_df, mp_config, device
        )

    # 3-Stage Approach (original implementation)
    # Use full training fold for propensity training (no internal validation split)
    # This implements pure n-fold CV where 100% of training fold is used for training

    # Step 1: Train propensity model
    logger.info(f"  Step 1: Training propensity model on {len(train_df)} samples")
    propensity_model = PropensityMatchingModel(
        sentence_model=mp_config.hier_transformer_sentence_model,
        freeze_sentence_encoder=mp_config.hier_transformer_freeze_sentence_encoder,
        max_sentences=mp_config.hier_transformer_max_sentences,
        max_sentence_length=mp_config.hier_transformer_max_sentence_length,
        transformer_dim=mp_config.hier_transformer_dim,
        num_transformer_layers=mp_config.hier_transformer_num_layers,
        num_attention_heads=mp_config.hier_transformer_num_heads,
        transformer_dropout=mp_config.hier_transformer_dropout,
        representation_dim=mp_config.representation_dim,
        joint_outcome_training=mp_config.joint_outcome_training,
        # Chunk encoder selection
        chunk_encoder=mp_config.chunk_encoder,
        # GRU-specific parameters
        gru_chunk_size=mp_config.gru_chunk_size,
        gru_chunk_overlap=mp_config.gru_chunk_overlap,
        gru_embedding_dim=mp_config.gru_embedding_dim,
        gru_hidden_dim=mp_config.gru_hidden_dim,
        gru_num_layers=mp_config.gru_num_layers,
        gru_max_vocab_size=mp_config.gru_max_vocab_size,
        gru_min_word_freq=mp_config.gru_min_word_freq,
        device=str(device)
    ).to(device)

    # Initialize
    propensity_model.fit_tokenizer(train_df[text_col].tolist())

    propensity_model, prop_history = train_propensity_model(
        propensity_model, train_df, None, mp_config, device  # val_df=None for pure CV
    )

    # Step 2: Extract representations and propensity scores for ALL training data
    logger.info(f"  Step 2: Extracting representations for {len(train_df)} training samples")
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

    # Step 3: Match patients within training data
    logger.info("  Step 3: Matching patients")
    treatment = train_df[treatment_col].values

    if mp_config.matching_method == "embedding":
        match_result = match_by_cosine_similarity(
            train_repr.numpy(), treatment,
            caliper=mp_config.caliper,
            method=mp_config.matching_algorithm
        )
    else:
        matcher = PropensityMatcher(
            method=mp_config.matching_algorithm,
            caliper=mp_config.caliper,
            caliper_scale=mp_config.caliper_scale
        )
        match_result = matcher.match(train_propensity, treatment)

    match_stats = {
        'fold': fold + 1,
        'n_train': len(train_df),
        'n_treated': match_result.n_treated,
        'n_control': match_result.n_control,
        'n_matched': len(match_result.matched_pairs),
        'match_rate': len(match_result.matched_pairs) / min(match_result.n_treated, match_result.n_control) \
            if min(match_result.n_treated, match_result.n_control) > 0 else 0.0,
        'mean_distance': match_result.distances.mean() if len(match_result.distances) > 0 else None,
        'unmatched_treated': match_result.n_unmatched_treated,
        'unmatched_control': match_result.n_unmatched_control
    }
    logger.info(f"    Matched {match_stats['n_matched']} pairs ({match_stats['match_rate']:.1%})")

    if len(match_result.matched_pairs) < 10:
        logger.warning(f"    Very few matched pairs ({len(match_result.matched_pairs)})! "
                      "Consider relaxing caliper or checking data balance.")

    # Step 4: Train outcome/tau model on matched pairs
    logger.info(f"  Step 4: Training outcome/tau model on {len(match_result.matched_pairs)} pairs")
    # Note: freezing is handled by train_matched_pair_outcome_model based on config.freeze_representation_stage2

    # Choose training approach based on config
    use_mean_ite = mp_config.use_mean_embedding_ite
    if use_mean_ite:
        logger.info(f"    Using mean-embedding ITE model")
        outcome_model, outcome_history = train_mean_embedding_ite_model(
            propensity_model, train_df, match_result.matched_pairs,
            mp_config, device
        )
    elif mp_config.use_cross_encoder:
        logger.info(f"    Using cross-encoder enhanced training")
        outcome_model, outcome_history = train_matched_pair_outcome_model_enhanced(
            propensity_model, train_df, match_result.matched_pairs,
            mp_config, device
        )
    else:
        outcome_model, outcome_history = train_matched_pair_outcome_model(
            propensity_model, train_df, match_result.matched_pairs,
            mp_config, device
        )

    # Combine logs
    all_logs = []
    for entry in prop_history:
        entry['fold'] = fold + 1
        entry['stage'] = 'propensity'
        all_logs.append(entry)
    for entry in outcome_history:
        entry['fold'] = fold + 1
        entry['stage'] = 'outcome_tau'
        all_logs.append(entry)

    # Step 5: Predict ITE for test fold
    logger.info(f"  Step 5: Predicting ITE for {len(test_df)} test samples")
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

        # Get potential outcomes and ITE from outcome model
        test_repr_device = test_repr.to(device)
        y0_prob, y1_prob, ite_prob = outcome_model.predict_potential_outcomes(test_repr_device)
        tau_logodds = outcome_model.predict_ite(test_repr_device)

    # Create predictions DataFrame
    preds_df = test_df.copy()
    preds_df['pred_propensity_prob'] = test_propensity
    preds_df['pred_tau_logodds'] = tau_logodds.cpu().numpy().flatten()
    preds_df['pred_y0_prob'] = y0_prob.cpu().numpy().flatten()
    preds_df['pred_y1_prob'] = y1_prob.cpu().numpy().flatten()
    preds_df['pred_ite_prob'] = ite_prob.cpu().numpy().flatten()
    preds_df['cv_fold'] = fold + 1

    # Cleanup
    propensity_model.cpu()
    outcome_model.cpu()
    del propensity_model, outcome_model, train_repr, test_repr
    gc.collect()
    cuda_cleanup()

    logger.info(f"FOLD {fold + 1} complete | {get_memory_info()}")
    return preds_df, all_logs, match_stats


def _process_matched_pair_fold_e2e(
    fold: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mp_config: MatchedPairConfig,
    device: torch.device
) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Process a single CV fold with end-to-end matched pair training.

    Uses a single unified EndToEndMatchedPairModel with joint training
    and periodic re-matching.

    Args:
        fold: Fold number (0-indexed)
        train_df: Training DataFrame (already reset index)
        test_df: Test DataFrame (original indices preserved)
        mp_config: MatchedPairConfig with e2e settings
        device: PyTorch device

    Returns:
        Tuple of (test_predictions_df, training_logs, match_statistics)
    """
    logger.info(f"  FOLD {fold + 1}: Using end-to-end training mode")

    text_col = mp_config.text_column
    treatment_col = mp_config.treatment_column
    outcome_col = mp_config.outcome_column

    # Split train into train/val for early stopping
    val_size = int(0.2 * len(train_df))
    val_df_inner = train_df.iloc[:val_size].reset_index(drop=True) if val_size > 0 else None
    train_df_inner = train_df.iloc[val_size:].reset_index(drop=True)

    # Create unified model
    model = EndToEndMatchedPairModel(
        sentence_model=mp_config.hier_transformer_sentence_model,
        freeze_sentence_encoder=mp_config.hier_transformer_freeze_sentence_encoder,
        max_sentences=mp_config.hier_transformer_max_sentences,
        max_sentence_length=mp_config.hier_transformer_max_sentence_length,
        transformer_dim=mp_config.hier_transformer_dim,
        num_transformer_layers=mp_config.hier_transformer_num_layers,
        num_attention_heads=mp_config.hier_transformer_num_heads,
        transformer_dropout=mp_config.hier_transformer_dropout,
        representation_dim=mp_config.representation_dim,
        hidden_outcome_dim=mp_config.hidden_outcome_dim,
        dropout=mp_config.dropout,
        device=str(device)
    ).to(device)

    # Initialize feature extractor
    model.fit_tokenizer(train_df_inner[text_col].tolist())

    # Train end-to-end
    model, history = train_end_to_end_matched_pair(
        model, train_df_inner, val_df_inner, mp_config, device
    )

    # Prepare logs
    all_logs = []
    for entry in history:
        entry['fold'] = fold + 1
        entry['stage'] = 'e2e'
        all_logs.append(entry)

    # Match statistics (from final epoch)
    match_stats = {
        'fold': fold + 1,
        'n_train': len(train_df),
        'training_mode': 'end_to_end',
        'final_n_matched_pairs': history[-1].get('n_matched_pairs', 0) if history else 0,
        'n_epochs': len(history),
    }

    # Predict on test data
    logger.info(f"  FOLD {fold + 1}: Predicting ITE for {len(test_df)} test samples")
    test_texts = test_df[text_col].tolist()

    model.eval()
    batch_size = mp_config.e2e_batch_size

    test_propensity = []
    test_y0 = []
    test_y1 = []
    test_ite = []
    test_tau = []

    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i + batch_size]

            # Get propensity
            prop = model.predict_propensity(batch_texts)
            test_propensity.append(prop.cpu().numpy())

            # Get potential outcomes
            y0, y1, ite = model.predict_potential_outcomes(batch_texts)
            test_y0.append(y0.cpu().numpy().flatten())
            test_y1.append(y1.cpu().numpy().flatten())
            test_ite.append(ite.cpu().numpy().flatten())

            # Get tau
            tau = model.predict_tau(batch_texts)
            test_tau.append(tau.cpu().numpy().flatten())

    test_propensity = np.concatenate(test_propensity)
    test_y0 = np.concatenate(test_y0)
    test_y1 = np.concatenate(test_y1)
    test_ite = np.concatenate(test_ite)
    test_tau = np.concatenate(test_tau)

    # Create predictions DataFrame
    preds_df = test_df.copy()
    preds_df['pred_propensity_prob'] = test_propensity
    preds_df['pred_tau_logodds'] = test_tau
    preds_df['pred_y0_prob'] = test_y0
    preds_df['pred_y1_prob'] = test_y1
    preds_df['pred_ite_prob'] = test_ite
    preds_df['cv_fold'] = fold + 1

    # Cleanup
    model.cpu()
    del model
    gc.collect()
    cuda_cleanup()

    logger.info(f"FOLD {fold + 1} (E2E) complete | {get_memory_info()}")
    return preds_df, all_logs, match_stats


def _run_matched_pair_fixed_split_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    mp_config: MatchedPairConfig,
    output_path: Path,
    device: torch.device
) -> None:
    """Run matched pair inference using fixed train/val/test splits."""
    logger.info("Running Fixed Split Matched Pair Inference")

    split_col = mp_config.split_column

    train_df = dataset[dataset[split_col] == 'train'].reset_index(drop=True)
    val_df = dataset[dataset[split_col] == 'val'].reset_index(drop=True)
    test_df = dataset[dataset[split_col] == 'test'].copy()

    # Combine train+val for propensity/matching training
    train_val_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Using {len(train_val_df)} samples for propensity training and matching")

    # Get column names
    text_col = mp_config.text_column
    treatment_col = mp_config.treatment_column
    outcome_col = mp_config.outcome_column

    # Step 1: Train propensity model
    logger.info(f"Step 1: Training propensity model on {len(train_df)} train + {len(val_df)} val samples")
    propensity_model = PropensityMatchingModel(
        sentence_model=mp_config.hier_transformer_sentence_model,
        freeze_sentence_encoder=mp_config.hier_transformer_freeze_sentence_encoder,
        max_sentences=mp_config.hier_transformer_max_sentences,
        max_sentence_length=mp_config.hier_transformer_max_sentence_length,
        transformer_dim=mp_config.hier_transformer_dim,
        num_transformer_layers=mp_config.hier_transformer_num_layers,
        num_attention_heads=mp_config.hier_transformer_num_heads,
        transformer_dropout=mp_config.hier_transformer_dropout,
        representation_dim=mp_config.representation_dim,
        joint_outcome_training=mp_config.joint_outcome_training,
        # Chunk encoder selection
        chunk_encoder=mp_config.chunk_encoder,
        # GRU-specific parameters
        gru_chunk_size=mp_config.gru_chunk_size,
        gru_chunk_overlap=mp_config.gru_chunk_overlap,
        gru_embedding_dim=mp_config.gru_embedding_dim,
        gru_hidden_dim=mp_config.gru_hidden_dim,
        gru_num_layers=mp_config.gru_num_layers,
        gru_max_vocab_size=mp_config.gru_max_vocab_size,
        gru_min_word_freq=mp_config.gru_min_word_freq,
        device=str(device)
    ).to(device)

    # Initialize
    propensity_model.fit_tokenizer(train_df[text_col].tolist())

    propensity_model, prop_history = train_propensity_model(
        propensity_model, train_df, val_df, mp_config, device
    )

    # Step 2: Extract representations
    logger.info(f"Step 2: Extracting representations for {len(train_val_df)} samples")
    train_texts = train_val_df[text_col].tolist()

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

    # Step 3: Match patients
    logger.info("Step 3: Matching patients")
    treatment = train_val_df[treatment_col].values

    if mp_config.matching_method == "embedding":
        match_result = match_by_cosine_similarity(
            train_repr.numpy(), treatment,
            caliper=mp_config.caliper,
            method=mp_config.matching_algorithm
        )
    else:
        matcher = PropensityMatcher(
            method=mp_config.matching_algorithm,
            caliper=mp_config.caliper,
            caliper_scale=mp_config.caliper_scale
        )
        match_result = matcher.match(train_propensity, treatment)

    match_stats = {
        'n_train': len(train_val_df),
        'n_treated': match_result.n_treated,
        'n_control': match_result.n_control,
        'n_matched': len(match_result.matched_pairs),
        'match_rate': len(match_result.matched_pairs) / min(match_result.n_treated, match_result.n_control) \
            if min(match_result.n_treated, match_result.n_control) > 0 else 0.0,
        'mean_distance': float(match_result.distances.mean()) if len(match_result.distances) > 0 else None
    }
    logger.info(f"  Matched {match_stats['n_matched']} pairs ({match_stats['match_rate']:.1%})")

    # Step 4: Train outcome/tau model
    logger.info(f"Step 4: Training outcome/tau model on {len(match_result.matched_pairs)} pairs")
    # Note: freezing is handled by train_matched_pair_outcome_model based on config.freeze_representation_stage2

    # Choose training approach based on config
    use_mean_ite = mp_config.use_mean_embedding_ite
    if use_mean_ite:
        logger.info(f"  Using mean-embedding ITE model")
        outcome_model, outcome_history = train_mean_embedding_ite_model(
            propensity_model, train_val_df, match_result.matched_pairs,
            mp_config, device
        )
    elif mp_config.use_cross_encoder:
        logger.info(f"  Using cross-encoder enhanced training")
        outcome_model, outcome_history = train_matched_pair_outcome_model_enhanced(
            propensity_model, train_val_df, match_result.matched_pairs,
            mp_config, device
        )
    else:
        outcome_model, outcome_history = train_matched_pair_outcome_model(
            propensity_model, train_val_df, match_result.matched_pairs,
            mp_config, device
        )

    # Save training logs
    log_path = output_path.parent / "matched_pair_training_log.csv"
    all_logs = []
    for entry in prop_history:
        entry['stage'] = 'propensity'
        all_logs.append(entry)
    for entry in outcome_history:
        entry['stage'] = 'outcome_tau'
        all_logs.append(entry)
    pd.DataFrame(all_logs).to_csv(log_path, index=False)

    # Step 5: Predict ITE for test set
    logger.info(f"Step 5: Predicting ITE for {len(test_df)} test samples")
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
        tau_logodds = outcome_model.predict_ite(test_repr_device)

    # Create predictions DataFrame
    results_df = test_df.copy()
    results_df['pred_propensity_prob'] = test_propensity
    results_df['pred_tau_logodds'] = tau_logodds.cpu().numpy().flatten()
    results_df['pred_y0_prob'] = y0_prob.cpu().numpy().flatten()
    results_df['pred_y1_prob'] = y1_prob.cpu().numpy().flatten()
    results_df['pred_ite_prob'] = ite_prob.cpu().numpy().flatten()

    _save_matched_pair_results(results_df, [match_stats], output_path)

    # Cleanup
    del propensity_model, outcome_model
    gc.collect()
    cuda_cleanup()


def _save_matched_pair_results(
    results_df: pd.DataFrame,
    match_stats: List[Dict],
    output_path: Path
) -> None:
    """Save predictions and match statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save predictions
    results_df.to_parquet(output_path)
    logger.info(f"Predictions saved to: {output_path}")

    # Save match statistics per fold
    match_df = pd.DataFrame(match_stats)
    match_path = output_path.parent / "match_statistics_by_fold.csv"
    match_df.to_csv(match_path, index=False)
    logger.info(f"Match statistics saved to: {match_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MATCHED PAIR ITE INFERENCE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(results_df)}")
    if 'match_rate' in match_df.columns:
        logger.info(f"Mean match rate across folds: {match_df['match_rate'].mean():.1%}")
    logger.info(f"Mean ITE (prob): {results_df['pred_ite_prob'].mean():.4f}")
    logger.info(f"Std ITE (prob): {results_df['pred_ite_prob'].std():.4f}")
    logger.info(f"Min ITE: {results_df['pred_ite_prob'].min():.4f}")
    logger.info(f"Max ITE: {results_df['pred_ite_prob'].max():.4f}")
    logger.info(f"Mean propensity: {results_df['pred_propensity_prob'].mean():.4f}")
