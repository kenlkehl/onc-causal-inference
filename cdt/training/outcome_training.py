# cdt/training/outcome_training.py
"""Outcome model training for assessing prognostic signal in data."""

import gc
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

from ..config import AppliedInferenceConfig, OutcomeModelConfig, normalize_feature_extractor_type
from ..models.outcome_model import OutcomeOnlyModel, create_outcome_model_from_config
from ..data import ClinicalTextDataset, collate_batch
from ..utils import cuda_cleanup, get_memory_info


logger = logging.getLogger(__name__)


def train_outcome_model_cv(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device,
    num_workers: int = 1,
    gpu_ids: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train outcome model using k-fold CV to generate out-of-sample scores.

    For each fold:
    1. Train outcome model on training fold
    2. Predict outcome probabilities on held-out fold

    This ensures each patient gets an outcome score from a model that
    did not include them in training, avoiding data leakage.

    Args:
        dataset: DataFrame with clinical text and outcome columns
        config: AppliedInferenceConfig with architecture and outcome_model settings
        device: PyTorch device
        num_workers: Number of parallel workers
        gpu_ids: List of GPU IDs for parallel processing

    Returns:
        Tuple of (DataFrame with 'outcome_score' column, training_log DataFrame)
    """
    outcome_config = config.outcome_model
    k = outcome_config.cv_folds

    logger.info(f"Training outcome model with {k}-fold CV on {len(dataset)} samples")

    # Reset index to ensure KFold works with indices
    dataset = dataset.reset_index(drop=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = list(kf.split(dataset))

    # Initialize outcome scores array
    outcome_scores = np.zeros(len(dataset))

    # Determine devices to use
    if gpu_ids and device.type == "cuda":
        devices = [torch.device(f"cuda:{i}") for i in gpu_ids]
    else:
        # MPS and CPU are single-device; ignore gpu_ids
        devices = [device]

    if num_workers > 1:
        logger.info(f"Parallelizing outcome CV across {num_workers} workers on devices: {devices}")

        results = Parallel(n_jobs=num_workers)(
            delayed(_process_outcome_fold)(
                fold, train_idx, test_idx, dataset, config,
                devices[fold % len(devices)]
            )
            for fold, (train_idx, test_idx) in enumerate(splits)
        )
    else:
        results = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            results.append(_process_outcome_fold(
                fold, train_idx, test_idx, dataset, config,
                devices[0]
            ))

    # Combine predictions from all folds
    all_history = []
    for test_idx, fold_scores, auroc, fold_history in results:
        outcome_scores[test_idx] = fold_scores
        all_history.extend(fold_history)
        logger.info(f"Fold outcome AUROC: {auroc:.4f}" if auroc else "Fold outcome AUROC: N/A")

    # Add outcome scores to dataset
    dataset = dataset.copy()
    dataset['outcome_score'] = outcome_scores

    # Create training log DataFrame
    training_log_df = pd.DataFrame(all_history)

    # Log summary statistics
    logger.info(f"Outcome score summary:")
    logger.info(f"  Mean: {outcome_scores.mean():.4f}")
    logger.info(f"  Std: {outcome_scores.std():.4f}")
    logger.info(f"  Min: {outcome_scores.min():.4f}")
    logger.info(f"  Max: {outcome_scores.max():.4f}")
    logger.info(f"  Median: {np.median(outcome_scores):.4f}")

    return dataset, training_log_df


def _process_outcome_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, Optional[float], List[Dict[str, Any]]]:
    """
    Process a single fold for outcome model training.

    Args:
        fold: Fold index
        train_idx: Training indices
        test_idx: Test indices
        dataset: Full dataset
        config: Configuration
        device: PyTorch device

    Returns:
        Tuple of (test_idx, outcome_scores, auroc, training_history)
    """
    # Re-configure logger for worker process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"Outcome FOLD {fold + 1} starting on {device}")

    outcome_config = config.outcome_model
    arch_config = config.architecture

    # Prepare data for this fold
    train_df = dataset.iloc[train_idx]
    test_df = dataset.iloc[test_idx]

    # Train outcome model
    model, fold_history = _train_outcome_model(
        train_df, test_df, config, outcome_config, arch_config, device
    )

    # Add fold number to each history entry
    for entry in fold_history:
        entry['fold'] = fold + 1

    # Predict on held-out fold
    outcome_scores = _predict_outcome(model, test_df, config, device)

    # Calculate AUROC if possible
    outcomes = test_df[config.outcome_column].values
    try:
        if len(np.unique(outcomes)) >= 2:
            auroc = roc_auc_score(outcomes, outcome_scores)
        else:
            auroc = None
    except Exception:
        auroc = None

    # Cleanup
    model.cpu()
    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    gc.collect()
    cuda_cleanup()

    logger.info(f"Outcome FOLD {fold + 1} complete | {get_memory_info()}")

    return test_idx, outcome_scores, auroc, fold_history


def _train_outcome_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: AppliedInferenceConfig,
    outcome_config: OutcomeModelConfig,
    arch_config,
    device: torch.device
) -> Tuple[OutcomeOnlyModel, List[Dict[str, Any]]]:
    """
    Train an outcome-only model.

    Args:
        train_df: Training data
        val_df: Validation data
        config: Applied inference config
        outcome_config: Outcome model config
        arch_config: Model architecture config
        device: PyTorch device

    Returns:
        Tuple of (trained OutcomeOnlyModel, training_history)
    """
    # Get feature extractor type (default to "cnn" for backward compatibility)
    # Normalize type (e.g., "modernbert" -> "bert")
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'cnn')
    )

    # Create outcome model
    model = create_outcome_model_from_config(
        arch_config=arch_config,
        representation_dim=arch_config.causal_head_representation_dim,
        device=device
    )

    train_texts = train_df[config.text_column].tolist()

    if feature_extractor_type == "cnn":
        # CNN-specific initialization
        model.fit_tokenizer(train_texts)
        logger.info(f"Fitted word tokenizer on {len(train_texts)} training texts")

        # Initialize embeddings from BERT if configured
        use_random_init = getattr(arch_config, 'cnn_use_random_embedding_init', False)
        if not use_random_init and getattr(arch_config, 'cnn_init_embeddings_from', None):
            model.feature_extractor.init_embeddings_from_bert(
                arch_config.cnn_init_embeddings_from,
                freeze=getattr(arch_config, 'cnn_freeze_embeddings', False)
            )

        # Initialize filters from explicit concepts and/or k-means
        if arch_config.cnn_explicit_filter_concepts or arch_config.cnn_num_kmeans_filters > 0:
            model.feature_extractor.init_filters(
                texts=train_texts,
                freeze=arch_config.cnn_freeze_filters
            )
    elif feature_extractor_type == "gru":
        # GRU-specific initialization
        model.fit_tokenizer(train_texts)
        logger.info(f"Fitted word tokenizer on {len(train_texts)} training texts")

        # Initialize embeddings from BERT if configured
        if getattr(arch_config, 'gru_init_embeddings_from', None):
            model.feature_extractor.init_embeddings_from_bert(
                arch_config.gru_init_embeddings_from,
                freeze=getattr(arch_config, 'gru_freeze_embeddings', False)
            )
    elif feature_extractor_type == "confounder":
        # Confounder extractor initialization
        # Check if GRU-based (requires fit_tokenizer)
        if getattr(arch_config, 'confounder_use_gru', False):
            model.fit_tokenizer(train_texts)
            logger.info(f"Fitted word tokenizer for GRU confounder extractor on {len(train_texts)} texts")
        else:
            # BERT-based or sentence-level: trigger lazy initialization
            model.fit_tokenizer(train_texts)  # No-op for pretrained encoders, triggers init
            logger.info("Using confounder feature extractor (pretrained encoder)")
    else:
        # BERT uses pretrained tokenizer, no fit_tokenizer needed
        logger.info(f"Using BERT feature extractor: {arch_config.bert_model_name}")

    # Create datasets
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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=outcome_config.outcome_batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=outcome_config.outcome_batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=outcome_config.outcome_learning_rate,
        weight_decay=1e-4
    )

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    history = []

    for epoch in range(outcome_config.outcome_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            batch['outcome'] = batch['outcome'].to(device)

            optimizer.zero_grad()
            losses = model.train_step(batch)
            losses['loss'].backward()
            optimizer.step()

            train_loss += losses['loss'].item()
            train_preds.append(torch.sigmoid(losses['y_logit']).flatten().cpu().numpy())
            train_labels.append(batch['outcome'].flatten().cpu().numpy())

        train_loss = train_loss / len(train_loader)

        # Flatten predictions and labels
        train_preds_flat = np.concatenate(train_preds)
        train_labels_flat = np.concatenate(train_labels)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch['outcome'] = batch['outcome'].to(device)
                losses = model.train_step(batch)
                val_loss += losses['loss'].item()
                val_preds.append(torch.sigmoid(losses['y_logit']).flatten().cpu().numpy())
                val_labels.append(batch['outcome'].flatten().cpu().numpy())

        val_loss = val_loss / len(val_loader)

        # Flatten predictions and labels
        val_preds_flat = np.concatenate(val_preds)
        val_labels_flat = np.concatenate(val_labels)

        # Calculate AUROCs
        try:
            if len(np.unique(train_labels_flat)) >= 2:
                train_auroc = roc_auc_score(train_labels_flat, train_preds_flat)
            else:
                train_auroc = None
        except Exception:
            train_auroc = None

        try:
            if len(np.unique(val_labels_flat)) >= 2:
                val_auroc = roc_auc_score(val_labels_flat, val_preds_flat)
            else:
                val_auroc = None
        except Exception:
            val_auroc = None

        # Record history
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'train_auroc': float(train_auroc) if train_auroc is not None else None,
            'val_auroc': float(val_auroc) if val_auroc is not None else None
        })

        # Log epoch metrics
        train_auroc_str = f"{train_auroc:.4f}" if train_auroc is not None else "N/A"
        val_auroc_str = f"{val_auroc:.4f}" if val_auroc is not None else "N/A"
        logger.info(f"  Epoch {epoch+1}/{outcome_config.outcome_epochs}: "
                   f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                   f"train_auroc={train_auroc_str}, val_auroc={val_auroc_str}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Cleanup
    del train_loader, val_loader, train_dataset, val_dataset, optimizer
    gc.collect()

    return model, history


def _predict_outcome(
    model: OutcomeOnlyModel,
    df: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device
) -> np.ndarray:
    """
    Predict outcome scores for a dataset.

    Args:
        model: Trained outcome model
        df: DataFrame with texts
        config: Configuration
        device: PyTorch device

    Returns:
        Array of outcome scores
    """
    dataset = ClinicalTextDataset(
        data=df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column
    )

    loader = DataLoader(
        dataset,
        batch_size=config.outcome_model.outcome_batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    model.eval()
    all_outcome = []

    with torch.no_grad():
        for batch in loader:
            texts = batch['texts']
            outcome_prob = model.predict(texts)
            all_outcome.append(outcome_prob.cpu().numpy())

    outcome_scores = np.concatenate(all_outcome)

    del loader, dataset
    gc.collect()

    return outcome_scores
