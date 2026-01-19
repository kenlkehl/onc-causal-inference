# cdt/training/propensity_training.py
"""Training pipeline for propensity score models."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

from ..config import PropensityModelConfig, PropensityTrainingConfig
from ..models.propensity_model import PropensityModel
from ..data import ClinicalTextDataset, collate_batch, EmbeddingCache
from ..matching import PropensityMatcher, compute_balance_statistics, assess_overlap
from ..analysis import (
    estimate_att_matched,
    estimate_ate_ipw,
    estimate_ate_stratified,
    summarize_analysis
)
from ..utils import cuda_cleanup


logger = logging.getLogger(__name__)


def train_propensity_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: PropensityModelConfig,
    training_config: PropensityTrainingConfig,
    device: torch.device,
    cache: Optional[EmbeddingCache] = None
) -> Tuple[PropensityModel, List[Dict[str, Any]]]:
    """
    Train a propensity score model.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        config: Model architecture config
        training_config: Training hyperparameters
        device: Device to train on
        cache: Optional embedding cache

    Returns:
        Trained model and training history
    """
    # Create model
    model = PropensityModel(
        sentence_transformer_model_name=config.embedding_model_name,
        encoder_type=config.encoder_type,
        hidden_dim=config.hidden_dim,
        num_latent_confounders=config.num_latent_confounders,
        features_per_confounder=config.features_per_confounder,
        explicit_confounder_texts=config.explicit_confounder_texts,
        aggregator_mode=config.aggregator_mode,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        dropout=config.dropout,
        joint_outcome_prediction=config.joint_outcome_prediction,
        outcome_weight=config.outcome_weight,
        use_confounder_features=config.use_confounder_features,
        arctanh_transform=config.arctanh_transform,
        device=str(device)
    )

    # Create datasets
    train_dataset = ClinicalTextDataset(
        data=train_df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column,
        model=model.sentence_transformer_model,
        device=device,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        cache=cache
    )

    val_dataset = ClinicalTextDataset(
        data=val_df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column,
        model=model.sentence_transformer_model,
        device=device,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        cache=cache
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    # Learning rate scheduler
    if training_config.lr_schedule == "linear":
        total_steps = len(train_loader) * training_config.epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
    elif training_config.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.epochs
        )
    else:
        scheduler = None

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    history = []
    patience_counter = 0

    for epoch in range(training_config.epochs):
        # Training
        model.train()
        train_metrics = _train_epoch(
            model, train_loader, optimizer, scheduler, device
        )

        # Validation
        model.eval()
        val_metrics = _eval_epoch(model, val_loader, device)

        # Record history
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_propensity_loss': train_metrics['propensity_loss'],
            'train_outcome_loss': train_metrics.get('outcome_loss', 0),
            'train_auroc_prop': train_metrics['auroc_prop'],
            'val_loss': val_metrics['loss'],
            'val_propensity_loss': val_metrics['propensity_loss'],
            'val_outcome_loss': val_metrics.get('outcome_loss', 0),
            'val_auroc_prop': val_metrics['auroc_prop'],
            'lr': optimizer.param_groups[0]['lr']
        }
        history.append(epoch_log)

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{training_config.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val AUROC: {val_metrics['auroc_prop']:.4f}"
        )

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= training_config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def _train_epoch(
    model: PropensityModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    epoch_loss = 0.0
    epoch_prop_loss = 0.0
    epoch_out_loss = 0.0
    all_treatments = []
    all_propensities = []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        # Convert chunk embeddings to list format
        chunk_embeddings_list = [
            batch['chunk_embeddings'][i, :, :].contiguous()
            for i in range(batch['chunk_embeddings'].size(0))
        ]
        batch['chunk_embeddings'] = chunk_embeddings_list

        optimizer.zero_grad()
        losses = model.train_step(batch)
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_loss += losses['loss'].item()
        epoch_prop_loss += losses['propensity_loss'].item()
        if losses['outcome_loss'] is not None:
            epoch_out_loss += losses['outcome_loss'].item()

        all_treatments.append(batch['treatment'].cpu().numpy())
        all_propensities.append(torch.sigmoid(losses['propensity_logit']).cpu().numpy())

    # Compute AUROC
    treatments = np.concatenate(all_treatments)
    propensities = np.concatenate(all_propensities).flatten()

    try:
        auroc_prop = roc_auc_score(treatments, propensities)
    except ValueError:
        auroc_prop = 0.5

    return {
        'loss': epoch_loss / len(loader),
        'propensity_loss': epoch_prop_loss / len(loader),
        'outcome_loss': epoch_out_loss / len(loader),
        'auroc_prop': auroc_prop
    }


def _eval_epoch(
    model: PropensityModel,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate for one epoch."""
    epoch_loss = 0.0
    epoch_prop_loss = 0.0
    epoch_out_loss = 0.0
    all_treatments = []
    all_propensities = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            chunk_embeddings_list = [
                batch['chunk_embeddings'][i, :, :].contiguous()
                for i in range(batch['chunk_embeddings'].size(0))
            ]
            batch['chunk_embeddings'] = chunk_embeddings_list

            losses = model.train_step(batch)

            epoch_loss += losses['loss'].item()
            epoch_prop_loss += losses['propensity_loss'].item()
            if losses['outcome_loss'] is not None:
                epoch_out_loss += losses['outcome_loss'].item()

            all_treatments.append(batch['treatment'].cpu().numpy())
            all_propensities.append(torch.sigmoid(losses['propensity_logit']).cpu().numpy())

    treatments = np.concatenate(all_treatments)
    propensities = np.concatenate(all_propensities).flatten()

    try:
        auroc_prop = roc_auc_score(treatments, propensities)
    except ValueError:
        auroc_prop = 0.5

    return {
        'loss': epoch_loss / len(loader),
        'propensity_loss': epoch_prop_loss / len(loader),
        'outcome_loss': epoch_out_loss / len(loader),
        'auroc_prop': auroc_prop
    }


def predict_propensity_scores(
    model: PropensityModel,
    df: pd.DataFrame,
    config: PropensityModelConfig,
    device: torch.device,
    cache: Optional[EmbeddingCache] = None,
    batch_size: int = 8
) -> pd.DataFrame:
    """
    Generate propensity scores for a dataset.

    Args:
        model: Trained propensity model
        df: DataFrame with text data
        config: Model config
        device: Device for inference
        cache: Optional embedding cache
        batch_size: Batch size for inference

    Returns:
        DataFrame with propensity scores and optionally representations
    """
    dataset = ClinicalTextDataset(
        data=df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column,
        model=model.sentence_transformer_model,
        device=device,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        cache=cache
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    all_propensities = []
    all_representations = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            chunk_embeddings_list = [
                batch['chunk_embeddings'][i, :, :].to(device).contiguous()
                for i in range(batch['chunk_embeddings'].size(0))
            ]

            preds = model.predict(chunk_embeddings_list)
            all_propensities.append(preds['propensity'].cpu().numpy())
            all_representations.append(preds['representation'].cpu().numpy())

    result_df = df.copy()
    result_df['propensity_score'] = np.concatenate(all_propensities)

    return result_df, np.concatenate(all_representations)


def run_propensity_matching_pipeline(
    dataset: pd.DataFrame,
    config: PropensityModelConfig,
    training_config: PropensityTrainingConfig,
    matching_config: Dict[str, Any],
    output_path: Path,
    device: torch.device,
    cache: Optional[EmbeddingCache] = None,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Run complete propensity score matching pipeline.

    1. Train propensity model (with optional cross-validation)
    2. Generate propensity scores
    3. Perform matching
    4. Compute treatment effects and statistical inference

    Args:
        dataset: Full dataset
        config: Model architecture config
        training_config: Training hyperparameters
        matching_config: Matching algorithm parameters
        output_path: Output directory
        device: Device for training/inference
        cache: Optional embedding cache
        cv_folds: Number of CV folds (1 = no CV)

    Returns:
        Dictionary with all results
    """
    logger.info("=" * 80)
    logger.info("PROPENSITY SCORE MATCHING PIPELINE")
    logger.info("=" * 80)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Reset index
    dataset = dataset.reset_index(drop=True)

    if cv_folds > 1:
        # Cross-validation for propensity score estimation
        results = _run_cv_pipeline(
            dataset, config, training_config, matching_config,
            output_path, device, cache, cv_folds
        )
    else:
        # Single split
        results = _run_single_split_pipeline(
            dataset, config, training_config, matching_config,
            output_path, device, cache
        )

    return results


def _run_cv_pipeline(
    dataset: pd.DataFrame,
    config: PropensityModelConfig,
    training_config: PropensityTrainingConfig,
    matching_config: Dict[str, Any],
    output_path: Path,
    device: torch.device,
    cache: Optional[EmbeddingCache],
    cv_folds: int
) -> Dict[str, Any]:
    """Run pipeline with cross-validation for propensity estimation."""
    logger.info(f"Running {cv_folds}-fold cross-validation")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    all_propensities = np.zeros(len(dataset))
    all_representations = []
    all_histories = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        logger.info(f"\n--- FOLD {fold + 1}/{cv_folds} ---")

        train_val_df = dataset.iloc[train_idx]
        test_df = dataset.iloc[test_idx]

        # Split train into train/val
        fold_train_df, fold_val_df = train_test_split(
            train_val_df, test_size=0.1, random_state=42
        )

        # Train model
        model, history = train_propensity_model(
            fold_train_df, fold_val_df, config, training_config, device, cache
        )

        for h in history:
            h['fold'] = fold + 1
        all_histories.extend(history)

        # Predict on test fold
        test_df_with_ps, representations = predict_propensity_scores(
            model, test_df, config, device, cache, training_config.batch_size
        )

        all_propensities[test_idx] = test_df_with_ps['propensity_score'].values
        all_representations.append((test_idx, representations))

        # Cleanup
        del model
        cuda_cleanup()

    # Add propensity scores to dataset
    dataset = dataset.copy()
    dataset['propensity_score'] = all_propensities

    # Save training history
    history_df = pd.DataFrame(all_histories)
    history_df.to_csv(output_path / "training_history.csv", index=False)

    # Perform matching and analysis
    results = _perform_matching_and_analysis(
        dataset, config, matching_config, output_path
    )

    results['training_history'] = all_histories

    return results


def _run_single_split_pipeline(
    dataset: pd.DataFrame,
    config: PropensityModelConfig,
    training_config: PropensityTrainingConfig,
    matching_config: Dict[str, Any],
    output_path: Path,
    device: torch.device,
    cache: Optional[EmbeddingCache]
) -> Dict[str, Any]:
    """Run pipeline with single train/test split."""
    logger.info("Running with fixed split")

    # Split data
    if config.split_column in dataset.columns:
        train_df = dataset[dataset[config.split_column] == 'train'].copy()
        val_df = dataset[dataset[config.split_column] == 'val'].copy()
        test_df = dataset[dataset[config.split_column] == 'test'].copy()

        if len(val_df) == 0:
            # Create val from train
            train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    else:
        train_val_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Train model
    model, history = train_propensity_model(
        train_df, val_df, config, training_config, device, cache
    )

    # Save model
    model.save_checkpoint(str(output_path / "propensity_model.pt"))

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path / "training_history.csv", index=False)

    # Predict on full dataset for matching
    full_df_with_ps, representations = predict_propensity_scores(
        model, dataset, config, device, cache, training_config.batch_size
    )

    # Perform matching and analysis
    results = _perform_matching_and_analysis(
        full_df_with_ps, config, matching_config, output_path
    )

    results['training_history'] = history

    return results


def _perform_matching_and_analysis(
    dataset: pd.DataFrame,
    config: PropensityModelConfig,
    matching_config: Dict[str, Any],
    output_path: Path
) -> Dict[str, Any]:
    """Perform propensity score matching and statistical analysis."""
    logger.info("\n--- MATCHING AND ANALYSIS ---")

    propensity_scores = dataset['propensity_score'].values
    treatment = dataset[config.treatment_column].values
    outcomes = dataset[config.outcome_column].values

    # Assess overlap
    overlap = assess_overlap(propensity_scores, treatment)
    logger.info(f"Overlap coefficient: {overlap['overlap_coefficient']:.3f}")
    logger.info(f"PS range - Treated: [{overlap['ps_treated_mean']:.3f} +/- {overlap['ps_treated_std']:.3f}]")
    logger.info(f"PS range - Control: [{overlap['ps_control_mean']:.3f} +/- {overlap['ps_control_std']:.3f}]")

    # Perform matching
    matcher = PropensityMatcher(
        method=matching_config.get('method', 'nearest'),
        caliper=matching_config.get('caliper'),
        caliper_scale=matching_config.get('caliper_scale', 'std'),
        ratio=matching_config.get('ratio', 1),
        replacement=matching_config.get('replacement', False),
        random_state=42
    )

    match_result = matcher.match(propensity_scores, treatment)
    logger.info(f"Matching result: {match_result}")

    # Save matched pairs
    if len(match_result.matched_pairs) > 0:
        matched_df = pd.DataFrame({
            'treated_idx': match_result.matched_pairs[:, 0],
            'control_idx': match_result.matched_pairs[:, 1],
            'distance': match_result.distances
        })
        matched_df.to_csv(output_path / "matched_pairs.csv", index=False)

    # Compute balance statistics (using propensity score as the covariate for now)
    covariates = pd.DataFrame({'propensity_score': propensity_scores})
    balance_stats = compute_balance_statistics(covariates, treatment, match_result)
    balance_stats.to_csv(output_path / "balance_statistics.csv", index=False)
    logger.info(f"\nBalance Statistics:\n{balance_stats.to_string()}")

    # Statistical analysis
    analysis_results = summarize_analysis(
        outcomes, treatment, propensity_scores, match_result
    )

    # Log key results
    logger.info(f"\n--- TREATMENT EFFECT ESTIMATES ---")
    logger.info(f"Crude difference: {analysis_results['crude_difference']:.4f}")
    logger.info(f"IPW ATE: {analysis_results['ate_ipw']}")
    logger.info(f"Stratified ATE: {analysis_results['ate_stratified']}")

    if 'att_matched' in analysis_results:
        logger.info(f"Matched ATT: {analysis_results['att_matched']}")

    # Save predictions
    predictions_df = dataset.copy()
    predictions_df.to_parquet(output_path / "predictions.parquet", index=False)

    # Save summary
    summary = {
        'n_samples': len(dataset),
        'n_treated': int(np.sum(treatment)),
        'n_control': int(np.sum(1 - treatment)),
        'n_matched_pairs': len(match_result.matched_pairs),
        'overlap_coefficient': overlap['overlap_coefficient'],
        'crude_difference': analysis_results['crude_difference'],
        'ate_ipw_estimate': analysis_results['ate_ipw'].estimate,
        'ate_ipw_ci_lower': analysis_results['ate_ipw'].ci_lower,
        'ate_ipw_ci_upper': analysis_results['ate_ipw'].ci_upper,
        'ate_ipw_pvalue': analysis_results['ate_ipw'].p_value,
        'ate_stratified_estimate': analysis_results['ate_stratified'].estimate,
    }

    if 'att_matched' in analysis_results:
        summary.update({
            'att_matched_estimate': analysis_results['att_matched'].estimate,
            'att_matched_ci_lower': analysis_results['att_matched'].ci_lower,
            'att_matched_ci_upper': analysis_results['att_matched'].ci_upper,
            'att_matched_pvalue': analysis_results['att_matched'].p_value,
        })

    # Save sensitivity analysis if available
    if 'sensitivity_analysis' in analysis_results:
        sens_df = analysis_results['sensitivity_analysis']
        sens_df.to_csv(output_path / "sensitivity_analysis.csv", index=False)

    import json
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    return {
        'dataset': predictions_df,
        'match_result': match_result,
        'overlap': overlap,
        'balance_stats': balance_stats,
        'analysis': analysis_results,
        'summary': summary
    }
