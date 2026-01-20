# cdt/inference/applied.py
"""Applied causal inference on real clinical data with CNN or BERT feature extraction."""

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
from joblib import Parallel, delayed

from ..config import AppliedInferenceConfig, normalize_feature_extractor_type
from ..models.causal_cnn import CausalCNNText
from ..data import (
    ClinicalTextDataset,
    collate_batch,
    load_dataset
)
from ..utils import cuda_cleanup, get_memory_info


logger = logging.getLogger(__name__)


def run_applied_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    cache=None,  # Kept for API compatibility, not used
    pretrained_weights_path: Optional[Path] = None,
    gpu_ids: Optional[List[int]] = None,
    num_workers: int = 1,
    save_filter_interpretations: bool = False,
    filter_interpretation_top_k: int = 10
) -> None:
    """
    Run applied causal inference on real data using CNN backbone.

    Args:
        dataset: DataFrame with clinical text, outcomes, and treatments
        config: Configuration for applied inference
        output_path: Path to save predictions
        device: PyTorch device
        cache: Unused, kept for API compatibility
        pretrained_weights_path: Unused, kept for API compatibility
        gpu_ids: List of GPU IDs for parallel processing
        num_workers: Number of parallel workers
        save_filter_interpretations: Whether to save filter interpretation analysis
        filter_interpretation_top_k: Number of top n-grams per filter to save
    """
    logger.info("=" * 80)
    logger.info("APPLIED CAUSAL INFERENCE (CNN)")
    logger.info("=" * 80)

    # Propensity trimming preprocessing (if enabled)
    trimming_stats = None
    if hasattr(config, 'propensity_trimming') and config.propensity_trimming.enabled:
        logger.info("=" * 80)
        logger.info("PROPENSITY-BASED DATASET TRIMMING")
        logger.info("=" * 80)

        from ..training.propensity_trimming import (
            train_propensity_model_cv, trim_by_propensity
        )

        # Train propensity model with CV to get out-of-sample scores
        dataset, propensity_training_log = train_propensity_model_cv(
            dataset, config, device, num_workers, gpu_ids
        )

        # Save propensity model training log
        training_log_path = output_path.parent / "propensity_trimming_training_log.csv"
        propensity_training_log.to_csv(training_log_path, index=False)
        logger.info(f"Propensity training log saved to: {training_log_path}")

        original_size = len(dataset)

        # Trim dataset
        dataset, trimming_stats = trim_by_propensity(
            dataset,
            config.propensity_trimming.min_propensity,
            config.propensity_trimming.max_propensity
        )

        logger.info(f"Dataset trimmed: {original_size} -> {len(dataset)} "
                   f"({trimming_stats['removed_low']} below min, "
                   f"{trimming_stats['removed_high']} above max)")

        # Save trimming stats
        trimming_stats_path = output_path.parent / "propensity_trimming_stats.json"
        with open(trimming_stats_path, 'w') as f:
            json.dump(trimming_stats, f, indent=2)
        logger.info(f"Trimming stats saved to: {trimming_stats_path}")

        logger.info("=" * 80)
        logger.info("CONTINUING WITH DRAGONNET TRAINING ON TRIMMED DATASET")
        logger.info("=" * 80)

    # Outcome model training (if enabled) - for assessing prognostic signal
    if hasattr(config, 'outcome_model') and config.outcome_model.enabled:
        logger.info("=" * 80)
        logger.info("OUTCOME MODEL TRAINING (PROGNOSTIC SIGNAL ASSESSMENT)")
        logger.info("=" * 80)

        from ..training.outcome_training import train_outcome_model_cv

        # Train outcome model with CV to get out-of-sample scores
        dataset, outcome_training_log = train_outcome_model_cv(
            dataset, config, device, num_workers, gpu_ids
        )

        # Save outcome model training log
        training_log_path = output_path.parent / "outcome_model_training_log.csv"
        outcome_training_log.to_csv(training_log_path, index=False)
        logger.info(f"Outcome model training log saved to: {training_log_path}")

        # Log summary of outcome prediction performance
        if 'val_auroc' in outcome_training_log.columns:
            mean_auroc = outcome_training_log['val_auroc'].dropna().mean()
            logger.info(f"Mean validation AUROC across folds: {mean_auroc:.4f}")

        logger.info("=" * 80)
        logger.info("CONTINUING WITH DRAGONNET TRAINING")
        logger.info("=" * 80)

    # Determine mode
    if config.cv_folds > 1:
        _run_cv_inference(
            dataset, config, output_path, device, gpu_ids, num_workers,
            save_filter_interpretations, filter_interpretation_top_k
        )
    else:
        _run_fixed_split_inference(
            dataset, config, output_path, device,
            save_filter_interpretations, filter_interpretation_top_k
        )


def _run_cv_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    gpu_ids: Optional[List[int]] = None,
    num_workers: int = 1,
    save_filter_interpretations: bool = False,
    filter_interpretation_top_k: int = 10
) -> None:
    """Run K-Fold Cross-Validation inference."""
    k = config.cv_folds
    logger.info(f"Starting {k}-Fold Cross-Validation on {len(dataset)} samples")

    # Reset index to ensure KFold works with indices
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
            delayed(_process_fold)(
                fold, train_idx, test_idx, dataset, config,
                devices[fold % len(devices)]
            )
            for fold, (train_idx, test_idx) in enumerate(splits)
        )
    else:
        results = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            results.append(_process_fold(
                fold, train_idx, test_idx, dataset, config,
                devices[0]
            ))

    # Unpack results
    all_predictions = [r[0] for r in results]
    all_training_logs = [log for r in results for log in r[1]]

    # Combine predictions and save
    results_df = pd.concat(all_predictions).sort_index()
    _save_and_summarize(results_df, output_path)

    # Save training logs
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(all_training_logs).to_csv(log_path, index=False)
    logger.info(f"Training logs saved to: {log_path}")

    # Save filter interpretations for the last fold if requested
    # (In CV mode, we train one more model on the last fold's training data for interpretation)
    if save_filter_interpretations:
        logger.info("Generating filter interpretations from final fold model...")
        last_fold = k - 1
        train_idx, _ = splits[last_fold]
        train_df = dataset.iloc[train_idx]
        val_df = dataset.iloc[splits[last_fold][1]]  # Use test as val for this

        # Train a model on the last fold for interpretation
        model, _ = _train_single_model(train_df, val_df, config, devices[0])
        train_texts = train_df[config.text_column].tolist()
        _save_filter_interpretations(
            model, train_texts, output_path.parent,
            top_k=filter_interpretation_top_k
        )

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _process_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Process a single fold (can be run in parallel)."""
    # Re-configure logger for worker process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"FOLD {fold + 1} starting on {device}")

    # 1. Prepare Data for this Fold
    train_df = dataset.iloc[train_idx]
    test_df = dataset.iloc[test_idx]

    # 2. Train Model on this fold
    model, history = _train_single_model(train_df, test_df, config, device)

    # Log History
    for entry in history:
        entry['fold'] = fold + 1

    # 3. Predict on Held-out Test fold
    preds = _predict_dataset(model, test_df, config, device)

    # 4. Store predictions with indices to reconstruct dataframe
    preds_df = test_df.copy()
    # Probability scale predictions (pred_* prefix indicates predicted values)
    preds_df['pred_y0_prob'] = preds['y0_prob']
    preds_df['pred_y1_prob'] = preds['y1_prob']
    preds_df['pred_ite_prob'] = preds['ite_prob']
    preds_df['pred_propensity_prob'] = preds['propensity_prob']
    preds_df['cv_fold'] = fold + 1

    # Aggressive GPU cleanup to prevent OOM across folds
    model.cpu()
    del model
    del preds
    del train_df, test_df

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()
    cuda_cleanup()

    logger.info(f"FOLD {fold + 1} complete | {get_memory_info()}")
    return preds_df, history


def _run_fixed_split_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    save_filter_interpretations: bool = False,
    filter_interpretation_top_k: int = 10
) -> None:
    """Run inference using fixed train/val/test splits."""
    logger.info("Running Fixed Split Inference (Train/Val/Test)")

    # Split data
    train_df = dataset[dataset[config.split_column] == 'train'].copy()
    val_df = dataset[dataset[config.split_column] == 'val'].copy()
    test_df = dataset[dataset[config.split_column] == 'test'].copy()

    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Train
    model, history = _train_single_model(train_df, val_df, config, device)

    # Save training logs
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    logger.info(f"Training logs saved to: {log_path}")

    # Save filter interpretations if requested
    if save_filter_interpretations:
        train_texts = train_df[config.text_column].tolist()
        _save_filter_interpretations(
            model, train_texts, output_path.parent,
            top_k=filter_interpretation_top_k
        )

    # Predict on Test
    logger.info("Generating predictions on test set...")
    preds = _predict_dataset(model, test_df, config, device)

    # Combine predictions (probability scale, pred_* prefix indicates predicted values)
    results_df = test_df.copy()
    results_df['pred_y0_prob'] = preds['y0_prob']
    results_df['pred_y1_prob'] = preds['y1_prob']
    results_df['pred_ite_prob'] = preds['ite_prob']
    results_df['pred_propensity_prob'] = preds['propensity_prob']

    _save_and_summarize(results_df, output_path)


def _train_single_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device
) -> Tuple[CausalCNNText, List[Dict[str, Any]]]:
    """Train a single model instance (CNN or BERT feature extractor)."""
    arch_config = config.architecture

    # Get feature extractor type (default to "cnn" for backward compatibility)
    # Normalize type (e.g., "modernbert" -> "bert")
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'cnn')
    )

    # Determine max_length and embedding_dim based on extractor type
    if feature_extractor_type == "gru":
        max_length = getattr(arch_config, 'gru_max_length', 8192)
        embedding_dim = getattr(arch_config, 'gru_embedding_dim', 256)
        min_word_freq = getattr(arch_config, 'gru_min_word_freq', 2)
        max_vocab_size = getattr(arch_config, 'gru_max_vocab_size', 50000)
    else:
        max_length = arch_config.cnn_max_length
        embedding_dim = arch_config.cnn_embedding_dim
        min_word_freq = getattr(arch_config, 'cnn_min_word_freq', 2)
        max_vocab_size = getattr(arch_config, 'cnn_max_vocab_size', 50000)

    # Create model with appropriate feature extractor
    model = CausalCNNText(
        feature_extractor_type=feature_extractor_type,
        # CNN/GRU shared args
        embedding_dim=embedding_dim,
        max_length=max_length,
        min_word_freq=min_word_freq,
        max_vocab_size=max_vocab_size,
        # CNN-specific args
        kernel_sizes=arch_config.cnn_kernel_sizes,
        explicit_filter_concepts=arch_config.cnn_explicit_filter_concepts,
        num_kmeans_filters=arch_config.cnn_num_kmeans_filters,
        num_random_filters=arch_config.cnn_num_random_filters,
        cnn_dropout=arch_config.cnn_dropout,
        projection_dim=arch_config.dragonnet_representation_dim,
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
        # DragonNet args
        dragonnet_representation_dim=arch_config.dragonnet_representation_dim,
        dragonnet_hidden_outcome_dim=arch_config.dragonnet_hidden_outcome_dim,
        dragonnet_dropout=getattr(arch_config, 'dragonnet_dropout', 0.2),
        device=str(device),
        model_type=arch_config.model_type
    )
    logger.info(f"Created model with {feature_extractor_type.upper()} feature extractor")

    train_texts = train_df[config.text_column].tolist()

    if feature_extractor_type == "cnn":
        # CNN-specific initialization
        # Fit word tokenizer on training texts
        model.fit_tokenizer(train_texts)
        logger.info(f"Fitted word tokenizer on {len(train_texts)} training texts")

        # Initialize embeddings from BERT if configured (unless random init is explicitly requested)
        use_random_init = getattr(arch_config, 'cnn_use_random_embedding_init', False)
        if not use_random_init and getattr(arch_config, 'cnn_init_embeddings_from', None):
            model.feature_extractor.init_embeddings_from_bert(
                arch_config.cnn_init_embeddings_from,
                freeze=getattr(arch_config, 'cnn_freeze_embeddings', False)
            )
        elif use_random_init:
            logger.info("Using random embedding initialization (cnn_use_random_embedding_init=True)")

        # Initialize filters from explicit concepts and/or k-means
        if arch_config.cnn_explicit_filter_concepts or arch_config.cnn_num_kmeans_filters > 0:
            model.feature_extractor.init_filters(
                texts=train_texts,
                freeze=arch_config.cnn_freeze_filters
            )
    elif feature_extractor_type == "gru":
        # GRU-specific initialization
        # Fit word tokenizer on training texts
        model.fit_tokenizer(train_texts)
        logger.info(f"Fitted word tokenizer on {len(train_texts)} training texts")

        # Initialize embeddings from BERT if configured
        if getattr(arch_config, 'gru_init_embeddings_from', None):
            model.feature_extractor.init_embeddings_from_bert(
                arch_config.gru_init_embeddings_from,
                freeze=getattr(arch_config, 'gru_freeze_embeddings', False)
            )
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
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    # Optimization
    train_config = config.training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=getattr(train_config, 'weight_decay', 0.01)
    )

    if train_config.lr_schedule == "linear":
        total_steps = len(train_loader) * train_config.epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
    else:
        scheduler = None

    # Training Loop
    best_val_loss = float('inf')
    best_model_state = None
    history = []

    for epoch in range(train_config.epochs):
        model.train()
        train_stats = _train_epoch(model, train_loader, optimizer, scheduler, device, train_config)

        model.eval()
        val_stats = _eval_epoch(model, val_loader, device, train_config)

        # Record history
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_stats['loss'],
            'train_auroc_y0': train_stats['auroc_y0'],
            'train_auroc_y1': train_stats['auroc_y1'],
            'train_auroc_prop': train_stats['auroc_prop'],
            'val_loss': val_stats['loss'],
            'val_auroc_y0': val_stats['auroc_y0'],
            'val_auroc_y1': val_stats['auroc_y1'],
            'val_auroc_prop': val_stats['auroc_prop'],
        }
        history.append(epoch_log)

        # Save best
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            best_model_state = model.state_dict()

    # Restore best
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Cleanup training artifacts
    del train_loader, val_loader, train_dataset, val_dataset
    del optimizer, best_model_state
    if scheduler is not None:
        del scheduler
    gc.collect()

    return model, history


def _predict_dataset(
    model: CausalCNNText,
    df: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device
) -> dict:
    """Generate predictions for a dataframe."""
    dataset = ClinicalTextDataset(
        data=df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column
    )

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    result = _generate_predictions(model, loader, device)

    del loader, dataset
    gc.collect()

    return result


def _train_epoch(
    model: CausalCNNText,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config
) -> dict:
    """Train for one epoch."""
    epoch_loss = 0.0
    all_targets = []
    all_treatments = []
    all_y0 = []
    all_y1 = []
    all_prop = []

    # Get regularization options from config
    label_smoothing = getattr(config, 'label_smoothing', 0.0)
    gradient_clip_norm = getattr(config, 'gradient_clip_norm', 0.0)
    gamma_rlearner = getattr(config, 'gamma_rlearner', 1.0)

    for batch in tqdm(loader, desc="Training", leave=False):
        # Move tensors to device
        batch['outcome'] = batch['outcome'].to(device)
        batch['treatment'] = batch['treatment'].to(device)
        # 'texts' stays as list of strings

        optimizer.zero_grad()

        losses = model.train_step(
            batch,
            alpha_propensity=config.alpha_propensity,
            beta_targreg=config.beta_targreg,
            gamma_rlearner=gamma_rlearner,
            label_smoothing=label_smoothing
        )

        losses['loss'].backward()

        # Gradient clipping (if enabled)
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_loss += losses['loss'].item()

        # Collect for metrics
        all_targets.append(batch['outcome'].detach().cpu())
        all_treatments.append(batch['treatment'].detach().cpu())
        all_y0.append(losses['y0_logit'].detach().cpu())
        all_y1.append(losses['y1_logit'].detach().cpu())
        all_prop.append(losses['t_logit'].detach().cpu())

    return _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop)


def _eval_epoch(
    model: CausalCNNText,
    loader: DataLoader,
    device: torch.device,
    config
) -> dict:
    """Evaluate for one epoch."""
    epoch_loss = 0.0
    all_targets = []
    all_treatments = []
    all_y0 = []
    all_y1 = []
    all_prop = []

    gamma_rlearner = getattr(config, 'gamma_rlearner', 1.0)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch['outcome'] = batch['outcome'].to(device)
            batch['treatment'] = batch['treatment'].to(device)

            losses = model.train_step(
                batch,
                alpha_propensity=config.alpha_propensity,
                beta_targreg=config.beta_targreg,
                gamma_rlearner=gamma_rlearner
            )

            epoch_loss += losses['loss'].item()

            all_targets.append(batch['outcome'].detach().cpu())
            all_treatments.append(batch['treatment'].detach().cpu())
            all_y0.append(losses['y0_logit'].detach().cpu())
            all_y1.append(losses['y1_logit'].detach().cpu())
            all_prop.append(losses['t_logit'].detach().cpu())

    return _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop)


def _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop):
    """Helper to compute AUROCs from collected batch outputs."""
    y_true = torch.cat(all_targets).numpy()
    t_true = torch.cat(all_treatments).numpy()
    y0_scores = torch.cat(all_y0).numpy()
    y1_scores = torch.cat(all_y1).numpy()
    prop_scores = torch.sigmoid(torch.cat(all_prop)).numpy()

    # Safe AUROC calculation
    def safe_auc(y, score):
        try:
            if len(np.unique(y)) < 2:
                return None
            return roc_auc_score(y, score)
        except Exception:
            return None

    # AUROC Y0 (on T=0 samples)
    mask0 = (t_true == 0)
    auroc_y0 = safe_auc(y_true[mask0], y0_scores[mask0]) if mask0.any() else None

    # AUROC Y1 (on T=1 samples)
    mask1 = (t_true == 1)
    auroc_y1 = safe_auc(y_true[mask1], y1_scores[mask1]) if mask1.any() else None

    # AUROC Propensity
    auroc_prop = safe_auc(t_true, prop_scores)

    return {
        'loss': epoch_loss / len(loader),
        'auroc_y0': auroc_y0,
        'auroc_y1': auroc_y1,
        'auroc_prop': auroc_prop
    }


def _generate_predictions(
    model: CausalCNNText,
    loader: DataLoader,
    device: torch.device
) -> dict:
    """Generate predictions on test set."""
    all_y0 = []
    all_y1 = []
    all_propensity = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            texts = batch['texts']

            preds = model.predict(texts)

            all_y0.append(preds['y0_logit'].cpu().numpy())
            all_y1.append(preds['y1_logit'].cpu().numpy())
            all_propensity.append(preds['t_logit'].cpu().numpy())

    y0_logit = np.concatenate(all_y0)
    y1_logit = np.concatenate(all_y1)
    propensity_logit = np.concatenate(all_propensity)
    ite_logit = y1_logit - y0_logit

    # Convert to probabilities using sigmoid
    y0_prob = 1.0 / (1.0 + np.exp(-y0_logit))
    y1_prob = 1.0 / (1.0 + np.exp(-y1_logit))
    propensity_prob = 1.0 / (1.0 + np.exp(-propensity_logit))
    ite_prob = y1_prob - y0_prob

    return {
        'y0_prob': y0_prob,
        'y1_prob': y1_prob,
        'propensity_prob': propensity_prob,
        'ite_prob': ite_prob
    }


def _save_and_summarize(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save results and print summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)

    logger.info(f"Predictions saved to: {output_path}")
    logger.info("\nPrediction Summary:")
    logger.info(f"  Samples: {len(results_df)}")
    logger.info("  Predicted ITE (probability scale):")
    logger.info(f"    Mean: {results_df['pred_ite_prob'].mean():.4f}")
    logger.info(f"    Std: {results_df['pred_ite_prob'].std():.4f}")
    logger.info(f"    Min: {results_df['pred_ite_prob'].min():.4f}")
    logger.info(f"    Max: {results_df['pred_ite_prob'].max():.4f}")
    logger.info(f"  Mean predicted propensity: {results_df['pred_propensity_prob'].mean():.4f}")


def _save_filter_interpretations(
    model: CausalCNNText,
    train_texts: List[str],
    output_dir: Path,
    top_k: int = 10
) -> None:
    """
    Generate and save filter interpretation analysis.

    Saves both a JSON file with structured data and a text summary.
    Only applicable for CNN feature extractors.

    Args:
        model: Trained CausalCNNText model
        train_texts: Training texts to analyze activations on
        output_dir: Directory to save interpretation files
        top_k: Number of top n-grams per filter
    """
    # Filter interpretations only available for CNN models
    if model.feature_extractor_type != "cnn":
        logger.info("Filter interpretations not available for BERT feature extractor (CNN-only feature)")
        return

    logger.info(f"Analyzing filter activations on {len(train_texts)} texts...")

    # Get structured interpretations
    interpretations = model.feature_extractor.interpret_filters(
        train_texts,
        top_k=top_k,
        batch_size=32
    )

    # Save JSON with full data
    json_path = output_dir / "filter_interpretations.json"
    with open(json_path, 'w') as f:
        json.dump(interpretations, f, indent=2)
    logger.info(f"Filter interpretations saved to: {json_path}")

    # Save human-readable summary
    summary = model.feature_extractor.get_filter_summary(
        train_texts,
        top_k=top_k,
        batch_size=32
    )
    summary_path = output_dir / "filter_interpretations_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Filter interpretation summary saved to: {summary_path}")
