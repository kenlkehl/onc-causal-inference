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
)
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

    # Determine mode
    if config.cv_folds > 1:
        _run_cv_inference_forest(
            dataset, config, output_path, device, verbose
        )
    else:
        _run_fixed_split_inference_forest(
            dataset, config, output_path, device, verbose
        )


def _run_cv_inference_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    verbose: bool = True
) -> None:
    """Run K-Fold Cross-Validation inference with causal forest."""
    k = config.cv_folds
    logger.info(f"Starting {k}-Fold Cross-Validation on {len(dataset)} samples")

    # Reset index
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
            fold, train_df, test_df, config, device, verbose
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
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Process a single fold with causal forest."""
    arch_config = config.architecture
    train_config = config.training

    # Get feature extractor type
    feature_extractor_type = normalize_feature_extractor_type(
        getattr(arch_config, 'feature_extractor_type', 'gru_pool')
    )

    # Create model
    model = _create_causal_forest_model(arch_config, device)
    logger.info(f"Created CausalTextForest with {feature_extractor_type.upper()} extractor")

    # Get training texts
    train_texts = train_df[config.text_column].tolist()
    test_texts = test_df[config.text_column].tolist()

    # Fit tokenizer if needed
    model.fit_tokenizer(train_texts)
    logger.info(f"Initialized feature extractor on {len(train_texts)} training texts")

    # Stage 1: Train representation
    logger.info("\n--- Stage 1: Training representation ---")
    history = _train_representation(
        model, train_df, test_df, config, device, verbose
    )

    # Stage 2: Train causal forest
    logger.info("\n--- Stage 2: Training causal forest ---")
    train_T = train_df[config.treatment_column].values
    train_Y = train_df[config.outcome_column].values
    model.train_causal_forest(train_texts, train_T, train_Y)

    # Predict on test
    logger.info("\n--- Generating predictions on test set ---")
    preds = model.predict(test_texts, return_ci=True)

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds_df, history


def _run_fixed_split_inference_forest(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    verbose: bool = True
) -> None:
    """Run inference using fixed train/val/test splits with causal forest."""
    logger.info("Running Fixed Split Inference (Train/Val/Test)")

    # Split data
    train_df = dataset[dataset[config.split_column] == 'train'].copy()
    val_df = dataset[dataset[config.split_column] == 'val'].copy()
    test_df = dataset[dataset[config.split_column] == 'test'].copy()

    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    arch_config = config.architecture

    # Create model
    model = _create_causal_forest_model(arch_config, device)

    # Get texts
    train_texts = train_df[config.text_column].tolist()
    test_texts = test_df[config.text_column].tolist()

    # Fit tokenizer
    model.fit_tokenizer(train_texts)

    # Stage 1: Train representation
    logger.info("\n--- Stage 1: Training representation ---")
    history = _train_representation(
        model, train_df, val_df, config, device, verbose
    )

    # Save training logs
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)

    # Stage 2: Train causal forest on full train + val
    logger.info("\n--- Stage 2: Training causal forest ---")
    combined_df = pd.concat([train_df, val_df])
    combined_texts = combined_df[config.text_column].tolist()
    combined_T = combined_df[config.treatment_column].values
    combined_Y = combined_df[config.outcome_column].values
    model.train_causal_forest(combined_texts, combined_T, combined_Y)

    # Predict on test
    logger.info("\n--- Generating predictions on test set ---")
    preds = model.predict(test_texts, return_ci=True)

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
    device: torch.device
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
        # Head args
        representation_dim=getattr(arch_config, 'dragonnet_representation_dim', 128),
        hidden_dim=getattr(arch_config, 'dragonnet_hidden_outcome_dim', 64),
        dropout=getattr(arch_config, 'dragonnet_dropout', 0.2),
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
        # Device
        device=str(device)
    )

    return model


def _train_representation(
    model: CausalTextForest,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Train representation (Stage 1)."""
    train_config = config.training

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_batch
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

            optimizer.zero_grad()

            losses = model.train_representation_step(
                batch,
                alpha_propensity=alpha_propensity,
                gamma_rlearner=gamma_rlearner,
                label_smoothing=label_smoothing,
                stop_grad_propensity=stop_grad_propensity
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

                losses = model.train_representation_step(
                    batch,
                    alpha_propensity=alpha_propensity,
                    gamma_rlearner=gamma_rlearner,
                    stop_grad_propensity=stop_grad_propensity
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

        # Compute AUROCs
        prop_scores = torch.sigmoid(torch.cat(all_prop_logits)).numpy().flatten()
        outcome_scores = torch.sigmoid(torch.cat(all_outcome_logits)).numpy().flatten()
        treatments = torch.cat(all_treatments).numpy()
        outcomes = torch.cat(all_outcomes).numpy()

        try:
            val_auroc_prop = roc_auc_score(treatments, prop_scores)
        except:
            val_auroc_prop = None

        try:
            val_auroc_outcome = roc_auc_score(outcomes, outcome_scores)
        except:
            val_auroc_outcome = None

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
            'val_auroc_outcome': val_auroc_outcome,
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
    logger.info("  Predicted ITE (τ):")
    logger.info(f"    Mean (ATE): {results_df['pred_ite_prob'].mean():.4f}")
    logger.info(f"    Std: {results_df['pred_ite_prob'].std():.4f}")
    logger.info(f"    Min: {results_df['pred_ite_prob'].min():.4f}")
    logger.info(f"    Max: {results_df['pred_ite_prob'].max():.4f}")

    if 'pred_ite_lower' in results_df.columns:
        # Report CI coverage stats
        significant = (results_df['pred_ite_lower'] > 0) | (results_df['pred_ite_upper'] < 0)
        logger.info(f"    Significant effects (CI excludes 0): {significant.sum()} ({significant.mean()*100:.1f}%)")

    logger.info(f"  Mean predicted propensity: {results_df['pred_propensity_prob'].mean():.4f}")
