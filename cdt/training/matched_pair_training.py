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
from ..matching import PropensityMatcher, match_by_cosine_similarity


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

    When joint_outcome_training is enabled in config, also trains on outcome
    prediction to learn features that are true confounders (predictive of
    both treatment and outcome).

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
    joint_training = config.joint_outcome_training and model.joint_outcome_training
    logger.info(f"Training propensity model on {len(train_df)} samples")
    logger.info(f"  Epochs: {config.propensity_epochs}, LR: {config.propensity_lr}")
    logger.info(f"  Joint outcome training: {joint_training}")
    if joint_training:
        logger.info(f"    alpha_propensity={config.alpha_propensity_stage1}, "
                   f"alpha_outcome={config.alpha_outcome_stage1}")
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
        train_propensity_loss = 0.0
        train_outcome_loss = 0.0
        train_prop_preds, train_prop_targets = [], []
        train_out_preds, train_out_targets = [], []

        for batch in tqdm(train_loader, desc=f"Propensity Epoch {epoch+1}", leave=False):
            texts = batch['texts']
            treatment = batch['treatment'].to(device)
            outcome = batch['outcome'].to(device)

            optimizer.zero_grad()

            if joint_training:
                # Joint training: propensity + outcome
                t_logit, y_logit = model.forward_joint(texts)
                propensity_loss = F.binary_cross_entropy_with_logits(t_logit.squeeze(-1), treatment)
                outcome_loss = F.binary_cross_entropy_with_logits(y_logit.squeeze(-1), outcome)
                loss = (config.alpha_propensity_stage1 * propensity_loss +
                        config.alpha_outcome_stage1 * outcome_loss)

                train_propensity_loss += propensity_loss.item()
                train_outcome_loss += outcome_loss.item()
                train_out_preds.append(torch.sigmoid(y_logit).detach().cpu())
                train_out_targets.append(outcome.detach().cpu())
                train_prop_preds.append(torch.sigmoid(t_logit).detach().cpu())
            else:
                # Propensity only
                t_logit = model(texts)
                loss = F.binary_cross_entropy_with_logits(t_logit.squeeze(-1), treatment)
                train_propensity_loss += loss.item()
                train_prop_preds.append(torch.sigmoid(t_logit).detach().cpu())

            train_prop_targets.append(treatment.detach().cpu())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Compute training metrics
        n_batches = len(train_loader)
        train_loss_avg = train_loss / n_batches
        train_propensity_loss_avg = train_propensity_loss / n_batches
        train_outcome_loss_avg = train_outcome_loss / n_batches if joint_training else None

        train_prop_preds_cat = torch.cat(train_prop_preds).numpy()
        train_prop_targets_cat = torch.cat(train_prop_targets).numpy()
        train_propensity_auroc = roc_auc_score(train_prop_targets_cat, train_prop_preds_cat) \
            if len(np.unique(train_prop_targets_cat)) > 1 else None

        train_outcome_auroc = None
        if joint_training:
            train_out_preds_cat = torch.cat(train_out_preds).numpy()
            train_out_targets_cat = torch.cat(train_out_targets).numpy()
            train_outcome_auroc = roc_auc_score(train_out_targets_cat, train_out_preds_cat) \
                if len(np.unique(train_out_targets_cat)) > 1 else None

        # Validation (if val_loader is available)
        val_loss_avg = None
        val_propensity_loss_avg = None
        val_outcome_loss_avg = None
        val_propensity_auroc = None
        val_outcome_auroc = None

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_propensity_loss = 0.0
            val_outcome_loss = 0.0
            val_prop_preds, val_prop_targets = [], []
            val_out_preds, val_out_targets = [], []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Propensity Val", leave=False):
                    texts = batch['texts']
                    treatment = batch['treatment'].to(device)
                    outcome = batch['outcome'].to(device)

                    if joint_training:
                        t_logit, y_logit = model.forward_joint(texts)
                        propensity_loss = F.binary_cross_entropy_with_logits(t_logit.squeeze(-1), treatment)
                        outcome_loss = F.binary_cross_entropy_with_logits(y_logit.squeeze(-1), outcome)
                        loss = (config.alpha_propensity_stage1 * propensity_loss +
                                config.alpha_outcome_stage1 * outcome_loss)
                        val_propensity_loss += propensity_loss.item()
                        val_outcome_loss += outcome_loss.item()
                        val_out_preds.append(torch.sigmoid(y_logit).cpu())
                        val_out_targets.append(outcome.cpu())
                        val_prop_preds.append(torch.sigmoid(t_logit).cpu())
                    else:
                        t_logit = model(texts)
                        loss = F.binary_cross_entropy_with_logits(t_logit.squeeze(-1), treatment)
                        val_propensity_loss += loss.item()
                        val_prop_preds.append(torch.sigmoid(t_logit).cpu())

                    val_prop_targets.append(treatment.cpu())
                    val_loss += loss.item()

            n_val_batches = len(val_loader)
            val_loss_avg = val_loss / n_val_batches
            val_propensity_loss_avg = val_propensity_loss / n_val_batches
            val_outcome_loss_avg = val_outcome_loss / n_val_batches if joint_training else None

            val_prop_preds_cat = torch.cat(val_prop_preds).numpy()
            val_prop_targets_cat = torch.cat(val_prop_targets).numpy()
            val_propensity_auroc = roc_auc_score(val_prop_targets_cat, val_prop_preds_cat) \
                if len(np.unique(val_prop_targets_cat)) > 1 else None

            if joint_training:
                val_out_preds_cat = torch.cat(val_out_preds).numpy()
                val_out_targets_cat = torch.cat(val_out_targets).numpy()
                val_outcome_auroc = roc_auc_score(val_out_targets_cat, val_out_preds_cat) \
                    if len(np.unique(val_out_targets_cat)) > 1 else None

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss_avg,
            'train_propensity_loss': train_propensity_loss_avg,
            'train_outcome_loss': train_outcome_loss_avg,
            'train_propensity_auroc': train_propensity_auroc,
            'train_outcome_auroc': train_outcome_auroc,
            'val_loss': val_loss_avg,
            'val_propensity_loss': val_propensity_loss_avg,
            'val_outcome_loss': val_outcome_loss_avg,
            'val_propensity_auroc': val_propensity_auroc,
            'val_outcome_auroc': val_outcome_auroc
        })

        if val_loss_avg is not None:
            if joint_training:
                logger.info(f"  Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, "
                           f"val_loss={val_loss_avg:.4f}, "
                           f"val_prop_auroc={val_propensity_auroc:.4f}, "
                           f"val_out_auroc={val_outcome_auroc:.4f}")
            else:
                logger.info(f"  Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, "
                           f"val_loss={val_loss_avg:.4f}, val_auroc={val_propensity_auroc:.4f}")

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
            if joint_training:
                logger.info(f"  Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, "
                           f"prop_auroc={train_propensity_auroc:.4f}, "
                           f"out_auroc={train_outcome_auroc:.4f}")
            else:
                logger.info(f"  Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, "
                           f"train_auroc={train_propensity_auroc:.4f}")

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


class MatchedPairTextDataset(Dataset):
    """
    Dataset of matched pairs using text indices for on-the-fly representation.

    Used when freeze_representation_stage2=False and we need to compute
    fresh representations each batch to enable fine-tuning.

    Args:
        texts: List of all patient texts (indexed by original dataframe index)
        outcomes: Array of binary outcomes (indexed by original dataframe index)
        matched_pairs: Array of (treated_idx, control_idx) pairs using original indices
    """

    def __init__(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        matched_pairs: np.ndarray
    ):
        self.texts = texts
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32)
        self.matched_pairs = matched_pairs

        logger.info(f"MatchedPairTextDataset: {len(matched_pairs)} pairs")

    def __len__(self) -> int:
        return len(self.matched_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        treated_idx, control_idx = self.matched_pairs[idx]
        return {
            'text_T': self.texts[treated_idx],
            'text_U': self.texts[control_idx],
            'y_T': self.outcomes[treated_idx],
            'y_U': self.outcomes[control_idx]
        }


def collate_text_pairs(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for text pair batches."""
    return {
        'texts_T': [b['text_T'] for b in batch],
        'texts_U': [b['text_U'] for b in batch],
        'y_T': torch.stack([b['y_T'] for b in batch]),
        'y_U': torch.stack([b['y_U'] for b in batch])
    }


def _perform_rematching(
    propensity_model: PropensityMatchingModel,
    train_df: pd.DataFrame,
    config: MatchedPairConfig,
    device: torch.device
) -> np.ndarray:
    """
    Re-compute matching based on current representations.

    Called during dynamic re-matching when freeze_representation_stage2=False.
    Extracts fresh propensity scores or embeddings from the current model state
    and re-runs the matching algorithm.

    Args:
        propensity_model: Current PropensityMatchingModel (possibly updated during training)
        train_df: Training DataFrame with text and treatment columns
        config: MatchedPairConfig with matching settings
        device: PyTorch device

    Returns:
        New matched_pairs array of shape (n_pairs, 2) where each row is
        (treated_idx, control_idx)
    """
    propensity_model.eval()
    texts = train_df[config.text_column].tolist()
    treatment = train_df[config.treatment_column].values

    with torch.no_grad():
        if config.matching_method == "embedding":
            representations = extract_all_representations(
                propensity_model, texts, config.outcome_batch_size, device
            )
            match_result = match_by_cosine_similarity(
                representations.numpy(), treatment,
                caliper=config.caliper,
                method=config.matching_algorithm
            )
        else:
            propensity_scores = extract_propensity_scores(
                propensity_model, texts, config.outcome_batch_size, device
            )
            matcher = PropensityMatcher(
                method=config.matching_algorithm,
                caliper=config.caliper,
                caliper_scale=config.caliper_scale
            )
            match_result = matcher.match(propensity_scores, treatment)

    propensity_model.train()

    logger.info(f"    Re-matched {len(match_result.matched_pairs)} pairs "
                f"(mean dist: {match_result.distances.mean():.4f})")

    return match_result.matched_pairs


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
    1. Extract representations for all matched patients (frozen or trainable)
    2. Create MatchedPairDataset
    3. Train MatchedPairOutcomeModel

    When freeze_representation_stage2=True (default):
        - Representation is frozen and pre-extracted once
        - Only outcome model parameters are optimized

    When freeze_representation_stage2=False:
        - Representation remains trainable
        - Representations computed on-the-fly each batch
        - Both outcome model and propensity model parameters optimized

    Args:
        propensity_model: Trained PropensityMatchingModel
        train_df: Training DataFrame (must contain all matched patient indices)
        matched_pairs: Array of (treated_idx, control_idx) pairs
        config: MatchedPairConfig with training settings
        device: PyTorch device

    Returns:
        Tuple of (trained_outcome_model, training_history)
    """
    freeze_repr = config.freeze_representation_stage2
    logger.info(f"Training outcome/tau model on {len(matched_pairs)} matched pairs")
    logger.info(f"  Freeze representation: {freeze_repr}")

    # Log dynamic re-matching config
    if config.dynamic_rematching:
        if freeze_repr:
            logger.warning("  dynamic_rematching=True is ignored when freeze_representation_stage2=True")
        else:
            logger.info(f"  Dynamic re-matching: enabled (every {config.rematching_frequency} epochs, "
                       f"warmup={config.rematching_warmup_epochs})")

    # Get all unique patient indices from matched pairs
    all_indices = np.unique(matched_pairs.flatten())

    # Create outcome model
    outcome_model = MatchedPairOutcomeModel(
        representation_dim=config.representation_dim,
        hidden_dim=config.hidden_outcome_dim,
        dropout=config.dropout
    ).to(device)

    if freeze_repr:
        # Frozen mode: pre-extract representations once
        propensity_model.freeze_representation()
        propensity_model.eval()

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

        # Create dataset with pre-computed representations
        pair_dataset = MatchedPairDataset(representations, outcomes, remapped_pairs)
        pair_loader = DataLoader(
            pair_dataset,
            batch_size=config.outcome_batch_size,
            shuffle=True
        )

        # Optimizer: only outcome model
        optimizer = torch.optim.AdamW(
            outcome_model.parameters(),
            lr=config.outcome_lr,
            weight_decay=0.01
        )
    else:
        # Trainable mode: compute representations on-the-fly
        propensity_model.unfreeze_representation()
        propensity_model.train()

        # Store texts indexed by original dataframe position
        # We need to use iloc for matched_pairs which are row indices
        all_texts = train_df[config.text_column].tolist()
        all_outcomes = train_df[config.outcome_column].values

        # Create dataset with text indices
        pair_dataset = MatchedPairTextDataset(all_texts, all_outcomes, matched_pairs)
        pair_loader = DataLoader(
            pair_dataset,
            batch_size=config.outcome_batch_size,
            shuffle=True,
            collate_fn=collate_text_pairs
        )

        # Optimizer: both outcome model and propensity model
        optimizer = torch.optim.AdamW(
            list(outcome_model.parameters()) + list(propensity_model.parameters()),
            lr=config.outcome_lr,
            weight_decay=0.01
        )

    # Training loop
    history = []
    best_loss = float('inf')
    best_model_state = None
    best_propensity_state = None if freeze_repr else None

    for epoch in range(config.outcome_epochs):
        # Dynamic re-matching check (only when representation is trainable)
        if (not freeze_repr and
            config.dynamic_rematching and
            epoch >= config.rematching_warmup_epochs and
            epoch > 0 and
            epoch % config.rematching_frequency == 0):

            logger.info(f"  Epoch {epoch+1}: Re-matching patients...")
            matched_pairs = _perform_rematching(
                propensity_model, train_df, config, device
            )
            # Recreate dataset with new pairs
            pair_dataset = MatchedPairTextDataset(all_texts, all_outcomes, matched_pairs)
            pair_loader = DataLoader(
                pair_dataset,
                batch_size=config.outcome_batch_size,
                shuffle=True,
                collate_fn=collate_text_pairs
            )

        outcome_model.train()
        if not freeze_repr:
            propensity_model.train()

        epoch_losses = []

        for batch in tqdm(pair_loader, desc=f"Outcome Epoch {epoch+1}", leave=False):
            if freeze_repr:
                # Pre-computed representations
                repr_U = batch['repr_U'].to(device)
                repr_T = batch['repr_T'].to(device)
            else:
                # Compute fresh representations
                repr_T = propensity_model.get_representation(batch['texts_T'])
                repr_U = propensity_model.get_representation(batch['texts_U'])

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
            if not freeze_repr:
                best_propensity_state = {k: v.cpu().clone() for k, v in propensity_model.state_dict().items()}

    # Restore best
    if best_model_state is not None:
        outcome_model.load_state_dict(best_model_state)
    if best_propensity_state is not None:
        propensity_model.load_state_dict(best_propensity_state)

    # Cleanup
    del pair_loader, pair_dataset
    if freeze_repr:
        del representations
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
