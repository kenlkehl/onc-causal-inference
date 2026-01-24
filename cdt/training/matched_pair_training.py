# cdt/training/matched_pair_training.py
"""Training utilities for matched pair ITE estimation.

This module provides:
- MatchedPairDataset: PyTorch Dataset for matched pairs
- matched_pair_loss: Loss function for outcome/tau training
- train_propensity_model: Train the propensity model (Stage 1)
- train_matched_pair_outcome_model: Train outcome/tau on matched pairs (Stage 3)
"""

import gc
import logging
from typing import Optional, List, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from ..config import MatchedPairConfig
from ..models.matched_pair_ite import PropensityMatchingModel, MatchedPairOutcomeModel
from ..data import ClinicalTextDataset, collate_batch
from ..utils import cuda_cleanup


logger = logging.getLogger(__name__)


class MatchedPairDataset(Dataset):
    """
    Dataset of matched pairs for outcome/tau training.

    Each item is a (repr_U, repr_T, y_U, y_T) tuple where U is the untreated
    (control) patient and T is the treated patient.

    Args:
        representations: Tensor of all patient representations (N, D)
        outcomes: Array of binary outcomes (N,)
        matched_pairs: Array of (treated_idx, control_idx) pairs (M, 2)
                       Note: matched_pairs[:, 0] = treated, matched_pairs[:, 1] = control
    """

    def __init__(
        self,
        representations: torch.Tensor,
        outcomes: np.ndarray,
        matched_pairs: np.ndarray
    ):
        self.representations = representations
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32)
        self.matched_pairs = matched_pairs

        logger.info(f"MatchedPairDataset: {len(matched_pairs)} pairs from {len(representations)} patients")

    def __len__(self) -> int:
        return len(self.matched_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        treated_idx, control_idx = self.matched_pairs[idx]
        return {
            'repr_T': self.representations[treated_idx],
            'repr_U': self.representations[control_idx],
            'y_T': self.outcomes[treated_idx],
            'y_U': self.outcomes[control_idx]
        }


def matched_pair_loss(
    y_U_logit: torch.Tensor,
    y_T_logit: torch.Tensor,
    tau_pred: torch.Tensor,
    y_U: torch.Tensor,
    y_T: torch.Tensor,
    alpha_outcome: float = 1.0,
    beta_tau: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Compute matched pair loss.

    Loss = alpha * (BCE(y_U_pred, y_U) + BCE(y_T_pred, y_T))
         + beta * MSE(tau_pred, tau_target)

    where tau_target = logit(y_T_pred) - logit(y_U_pred) (detached)

    The tau target uses the model's own predictions (detached) as soft targets,
    providing smoother signal than raw binary outcomes.

    Args:
        y_U_logit: Predicted outcome logit for untreated (B, 1)
        y_T_logit: Predicted outcome logit for treated (B, 1)
        tau_pred: Predicted tau on log-odds scale (B, 1)
        y_U: Actual outcome for untreated (B,)
        y_T: Actual outcome for treated (B,)
        alpha_outcome: Weight for outcome loss
        beta_tau: Weight for tau loss

    Returns:
        Dictionary with loss components:
            - loss: Total loss
            - outcome_loss: Combined outcome BCE loss
            - outcome_loss_U: Outcome loss for untreated
            - outcome_loss_T: Outcome loss for treated
            - tau_loss: MSE loss for tau prediction
            - tau_pred_mean: Mean predicted tau
            - tau_target_mean: Mean tau target
    """
    # Outcome BCE loss (both patients)
    outcome_loss_U = F.binary_cross_entropy_with_logits(
        y_U_logit.squeeze(-1), y_U
    )
    outcome_loss_T = F.binary_cross_entropy_with_logits(
        y_T_logit.squeeze(-1), y_T
    )
    outcome_loss = outcome_loss_U + outcome_loss_T

    # Tau target: signed log-odds difference
    # Use model predictions (detached) as soft targets for smoother signal
    # This provides a continuous target rather than binary {-1, 0, 1}
    tau_target = y_T_logit.detach() - y_U_logit.detach()

    # Tau loss: MSE on log-odds scale
    tau_loss = F.mse_loss(tau_pred, tau_target)

    total_loss = alpha_outcome * outcome_loss + beta_tau * tau_loss

    return {
        'loss': total_loss,
        'outcome_loss': outcome_loss,
        'outcome_loss_U': outcome_loss_U,
        'outcome_loss_T': outcome_loss_T,
        'tau_loss': tau_loss,
        'tau_pred_mean': tau_pred.mean(),
        'tau_target_mean': tau_target.mean()
    }


def train_propensity_model(
    model: PropensityMatchingModel,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: MatchedPairConfig,
    device: torch.device
) -> Tuple[PropensityMatchingModel, List[Dict[str, Any]]]:
    """
    Train propensity model to convergence.

    Stage 1 of the matched pair pipeline. Trains the propensity model
    using binary cross-entropy for treatment prediction.

    Args:
        model: PropensityMatchingModel to train
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional). If None, trains for fixed epochs
            without early stopping.
        config: MatchedPairConfig with training settings
        device: PyTorch device

    Returns:
        Tuple of (trained_model, training_history)
    """
    logger.info(f"Training propensity model on {len(train_df)} samples")
    logger.info(f"  Epochs: {config.propensity_epochs}, LR: {config.propensity_lr}")
    if val_df is None:
        logger.info("  No validation set - training for fixed epochs")

    # Create datasets
    train_dataset = ClinicalTextDataset(
        data=train_df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.propensity_batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = None
    if val_df is not None:
        val_dataset = ClinicalTextDataset(
            data=val_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.propensity_batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.propensity_lr,
        weight_decay=0.01
    )

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(config.propensity_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for batch in tqdm(train_loader, desc=f"Propensity Epoch {epoch+1}", leave=False):
            texts = batch['texts']
            treatment = batch['treatment'].to(device)

            optimizer.zero_grad()

            logits = model(texts).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, treatment)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).detach().cpu())
            train_targets.append(treatment.detach().cpu())

        # Compute training metrics
        train_loss_avg = train_loss / len(train_loader)
        train_preds_cat = torch.cat(train_preds).numpy()
        train_targets_cat = torch.cat(train_targets).numpy()
        train_auroc = roc_auc_score(train_targets_cat, train_preds_cat) \
            if len(np.unique(train_targets_cat)) > 1 else None

        # Validation (if val_loader is available)
        val_loss_avg = None
        val_auroc = None

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Propensity Val", leave=False):
                    texts = batch['texts']
                    treatment = batch['treatment'].to(device)

                    logits = model(texts).squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, treatment)

                    val_loss += loss.item()
                    val_preds.append(torch.sigmoid(logits).cpu())
                    val_targets.append(treatment.cpu())

            val_loss_avg = val_loss / len(val_loader)
            val_preds_cat = torch.cat(val_preds).numpy()
            val_targets_cat = torch.cat(val_targets).numpy()
            val_auroc = roc_auc_score(val_targets_cat, val_preds_cat) \
                if len(np.unique(val_targets_cat)) > 1 else None

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss_avg,
            'val_loss': val_loss_avg,
            'train_auroc': train_auroc,
            'val_auroc': val_auroc
        })

        if val_loss_avg is not None:
            logger.info(f"  Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, "
                       f"val_loss={val_loss_avg:.4f}, val_auroc={val_auroc:.4f}")

            # Early stopping (only when validation is available)
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.propensity_early_stopping_patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break
        else:
            logger.info(f"  Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, "
                       f"train_auroc={train_auroc:.4f}")

    # Restore best model (only if we had validation-based early stopping)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Cleanup
    del train_loader
    if val_loader is not None:
        del val_loader
    del train_dataset
    gc.collect()

    return model, history


def train_matched_pair_outcome_model(
    propensity_model: PropensityMatchingModel,
    train_df: pd.DataFrame,
    matched_pairs: np.ndarray,
    config: MatchedPairConfig,
    device: torch.device
) -> Tuple[MatchedPairOutcomeModel, List[Dict[str, Any]]]:
    """
    Train outcome/tau model on matched pairs.

    Stage 3 of the matched pair pipeline:
    1. Extract frozen representations for all matched patients
    2. Create MatchedPairDataset
    3. Train MatchedPairOutcomeModel

    Args:
        propensity_model: Trained and frozen PropensityMatchingModel
        train_df: Training DataFrame (must contain all matched patient indices)
        matched_pairs: Array of (treated_idx, control_idx) pairs
        config: MatchedPairConfig with training settings
        device: PyTorch device

    Returns:
        Tuple of (trained_outcome_model, training_history)
    """
    logger.info(f"Training outcome/tau model on {len(matched_pairs)} matched pairs")

    # Ensure propensity model is frozen
    propensity_model.freeze_representation()
    propensity_model.eval()

    # Get all unique patient indices from matched pairs
    all_indices = np.unique(matched_pairs.flatten())

    # Extract texts and outcomes for matched patients
    texts = train_df.iloc[all_indices][config.text_column].tolist()
    outcomes = train_df.iloc[all_indices][config.outcome_column].values

    # Extract representations (frozen)
    logger.info(f"  Extracting representations for {len(all_indices)} patients...")
    representations = []
    batch_size = config.outcome_batch_size

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_repr = propensity_model.get_representation(batch_texts)
            representations.append(batch_repr.cpu())

    representations = torch.cat(representations, dim=0)  # (N, D)

    # Create index mapping: original_idx -> representation_idx
    idx_to_repr_idx = {orig_idx: i for i, orig_idx in enumerate(all_indices)}

    # Remap matched pairs to representation indices
    remapped_pairs = np.array([
        [idx_to_repr_idx[t], idx_to_repr_idx[c]]
        for t, c in matched_pairs
    ])

    # Create dataset
    pair_dataset = MatchedPairDataset(representations, outcomes, remapped_pairs)

    # Create dataloader
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=config.outcome_batch_size,
        shuffle=True
    )

    # Create outcome model
    outcome_model = MatchedPairOutcomeModel(
        representation_dim=config.representation_dim,
        hidden_dim=config.hidden_outcome_dim,
        dropout=config.dropout
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        outcome_model.parameters(),
        lr=config.outcome_lr,
        weight_decay=0.01
    )

    # Training loop
    history = []
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(config.outcome_epochs):
        outcome_model.train()
        epoch_losses = []

        for batch in tqdm(pair_loader, desc=f"Outcome Epoch {epoch+1}", leave=False):
            repr_U = batch['repr_U'].to(device)
            repr_T = batch['repr_T'].to(device)
            y_U = batch['y_U'].to(device)
            y_T = batch['y_T'].to(device)

            optimizer.zero_grad()

            y_U_logit, y_T_logit, tau_pred = outcome_model(repr_U, repr_T)

            losses = matched_pair_loss(
                y_U_logit, y_T_logit, tau_pred, y_U, y_T,
                alpha_outcome=config.alpha_outcome,
                beta_tau=config.beta_tau
            )

            losses['loss'].backward()
            optimizer.step()

            epoch_losses.append({k: v.item() for k, v in losses.items()})

        # Aggregate epoch losses
        epoch_summary = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        epoch_summary['epoch'] = epoch + 1
        history.append(epoch_summary)

        logger.info(f"  Epoch {epoch+1}: loss={epoch_summary['loss']:.4f}, "
                   f"outcome_loss={epoch_summary['outcome_loss']:.4f}, "
                   f"tau_loss={epoch_summary['tau_loss']:.4f}")

        # Track best
        if epoch_summary['loss'] < best_loss:
            best_loss = epoch_summary['loss']
            best_model_state = {k: v.cpu().clone() for k, v in outcome_model.state_dict().items()}

    # Restore best
    if best_model_state is not None:
        outcome_model.load_state_dict(best_model_state)

    # Cleanup
    del pair_loader, pair_dataset, representations
    gc.collect()

    return outcome_model, history


def extract_all_representations(
    propensity_model: PropensityMatchingModel,
    texts: List[str],
    batch_size: int = 32,
    device: torch.device = None
) -> torch.Tensor:
    """
    Extract representations for all texts using the propensity model.

    Args:
        propensity_model: Trained PropensityMatchingModel
        texts: List of document texts
        batch_size: Batch size for extraction
        device: PyTorch device

    Returns:
        Tensor of representations (N, D)
    """
    propensity_model.eval()
    representations = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_repr = propensity_model.get_representation(batch_texts)
            representations.append(batch_repr.cpu())

    return torch.cat(representations, dim=0)


def extract_propensity_scores(
    propensity_model: PropensityMatchingModel,
    texts: List[str],
    batch_size: int = 32,
    device: torch.device = None
) -> np.ndarray:
    """
    Extract propensity scores for all texts.

    Args:
        propensity_model: Trained PropensityMatchingModel
        texts: List of document texts
        batch_size: Batch size for extraction
        device: PyTorch device

    Returns:
        Array of propensity scores (N,)
    """
    propensity_model.eval()
    scores = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_scores = propensity_model.predict_propensity(batch_texts)
            scores.append(batch_scores.cpu().numpy())

    return np.concatenate(scores, axis=0)
