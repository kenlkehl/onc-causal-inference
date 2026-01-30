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
from ..models.matched_pair_ite import (
    PropensityMatchingModel,
    MatchedPairOutcomeModel,
    EnhancedMatchedPairOutcomeModel,
    EndToEndMatchedPairModel,
    MeanEmbeddingITEModel,
    InstanceCausalHead
)
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


def mean_embedding_ite_loss(
    Y_U_logit: torch.Tensor,
    Y_T_logit: torch.Tensor,
    y_U: torch.Tensor,
    y_T: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Loss for mean-embedding ITE model.

    Simple supervised loss on both matched patients:
        L = BCE(sigmoid(Y_U_logit), y_U) + BCE(sigmoid(Y_T_logit), y_T)

    The model formulation is:
        mean_repr = (repr_T + repr_U) / 2
        base_logit = frozen_outcome_head(mean_repr)  [from Stage 1, frozen]
        ite_half = ite_head(mean_repr)               [trainable]
        Y_U_logit = base_logit - ite_half
        Y_T_logit = base_logit + ite_half

    This symmetric formulation ensures the ITE head learns the treatment effect
    as the adjustment around the frozen baseline outcome prediction.

    Args:
        Y_U_logit: Predicted outcome logit for untreated (B, 1)
        Y_T_logit: Predicted outcome logit for treated (B, 1)
        y_U: Actual outcome for untreated (B,)
        y_T: Actual outcome for treated (B,)

    Returns:
        Dictionary with loss components:
            - loss: Total loss (loss_U + loss_T)
            - loss_U: BCE loss for untreated patients
            - loss_T: BCE loss for treated patients
            - ite_half_mean: Mean predicted ite_half (implicit from logit difference)
    """
    loss_U = F.binary_cross_entropy_with_logits(Y_U_logit.squeeze(-1), y_U)
    loss_T = F.binary_cross_entropy_with_logits(Y_T_logit.squeeze(-1), y_T)

    total_loss = loss_U + loss_T

    # Compute implicit ite_half from the output (for logging)
    # ite_half = (Y_T_logit - Y_U_logit) / 2
    ite_half_mean = ((Y_T_logit - Y_U_logit) / 2).mean()

    return {
        'loss': total_loss,
        'loss_U': loss_U,
        'loss_T': loss_T,
        'ite_half_mean': ite_half_mean
    }


def compute_clam_loss(
    chunk_embeddings_list: List[torch.Tensor],
    attention_weights_list: List[torch.Tensor],
    treatments: torch.Tensor,
    outcomes: torch.Tensor,
    instance_head: InstanceCausalHead,
    num_instances: int = 5
) -> Dict[str, torch.Tensor]:
    """
    Compute CLAM instance-level loss on top-B attended chunks.

    This supervises the top-attended chunks with document-level labels,
    encouraging the model to attend to causally relevant text.

    Args:
        chunk_embeddings_list: List of (C_i, D) tensors - chunk embeddings per doc
        attention_weights_list: List of (C_i,) tensors - gated attention weights per doc
        treatments: (B,) - document-level treatment labels
        outcomes: (B,) - document-level outcome labels
        instance_head: InstanceCausalHead for predicting T and Y from chunks
        num_instances: Number of top-attended chunks to supervise per document

    Returns:
        Dictionary with:
            - clam_loss: Combined CLAM loss
            - clam_propensity_loss: Average propensity loss across instances
            - clam_outcome_loss: Average outcome loss across instances
    """
    total_t_loss = 0.0
    total_y_loss = 0.0
    n_docs = len(chunk_embeddings_list)
    n_valid = 0

    for i, (chunk_embs, attn_weights) in enumerate(zip(chunk_embeddings_list, attention_weights_list)):
        # Skip empty documents
        if chunk_embs.size(0) == 0:
            continue

        # Get top-B attended chunks
        B = min(num_instances, len(attn_weights))
        _, top_indices = torch.topk(attn_weights, B)
        top_chunks = chunk_embs[top_indices]  # (B, D)

        # Predict T and Y for top chunks
        t_logit, y_logit = instance_head(top_chunks)  # (B, 1), (B, 1)

        # Expand document-level labels to match chunk count
        t_label = treatments[i].expand(B)
        y_label = outcomes[i].expand(B)

        # Compute losses
        total_t_loss += F.binary_cross_entropy_with_logits(t_logit.squeeze(-1), t_label)
        total_y_loss += F.binary_cross_entropy_with_logits(y_logit.squeeze(-1), y_label)
        n_valid += 1

    # Average over valid documents
    if n_valid > 0:
        avg_t_loss = total_t_loss / n_valid
        avg_y_loss = total_y_loss / n_valid
    else:
        device = treatments.device
        avg_t_loss = torch.tensor(0.0, device=device)
        avg_y_loss = torch.tensor(0.0, device=device)

    return {
        'clam_loss': avg_t_loss + avg_y_loss,
        'clam_propensity_loss': avg_t_loss,
        'clam_outcome_loss': avg_y_loss
    }


def train_mean_embedding_ite_model(
    propensity_model: PropensityMatchingModel,
    train_df: pd.DataFrame,
    matched_pairs: np.ndarray,
    config: 'MatchedPairConfig',
    device: torch.device,
    instance_head: Optional[InstanceCausalHead] = None
) -> Tuple[MeanEmbeddingITEModel, List[Dict[str, Any]]]:
    """
    Train mean-embedding ITE model on matched pairs.

    This implements Stage 3 of the matched pair pipeline when
    use_mean_embedding_ite=True in config.

    When freeze_representation_stage2=True (default):
    1. Freeze propensity model completely (representation + propensity head + outcome head)
    2. Extract frozen outcome_head from propensity_model
    3. Create MeanEmbeddingITEModel with frozen outcome_head
    4. Pre-extract representations for all matched patients
    5. Train only the ite_head using mean of paired representations

    When freeze_representation_stage2=False:
    1. Keep propensity model trainable (unfreeze representation)
    2. Keep outcome_head trainable
    3. Create MeanEmbeddingITEModel without frozen outcome_head
    4. Compute representations on-the-fly each batch
    5. Train ite_head + propensity_model jointly
    6. Optionally add CLAM instance-level supervision

    Training loop:
    - For each (treated, untreated) pair:
      - mean_repr = (repr_T + repr_U) / 2
      - Y_U_logit, Y_T_logit, _ = model.forward_training(repr_U, repr_T, ...)
      - loss = BCE(Y_U_pred, y_U) + BCE(Y_T_pred, y_T)
      - backprop

    Prerequisites:
    - propensity_model must have been trained with joint_outcome_training=True
      (so that the outcome_head exists)

    Args:
        propensity_model: Trained PropensityMatchingModel with outcome_head
        train_df: Training DataFrame (must contain all matched patient indices)
        matched_pairs: Array of (treated_idx, control_idx) pairs
        config: MatchedPairConfig with training settings
        device: PyTorch device
        instance_head: Optional InstanceCausalHead from Stage 1 (for CLAM)

    Returns:
        Tuple of (trained_ite_model, training_history)

    Raises:
        ValueError: If propensity_model does not have an outcome_head
    """
    freeze_base = config.freeze_representation_stage2

    # Check if CLAM is enabled for Stage 3 (only when representation is trainable)
    clam_enabled = (config.clam_enabled and
                    not freeze_base and
                    config.chunk_encoder in ('bert_gated_pool', 'gru_pool') and
                    instance_head is not None)

    logger.info(f"Training mean-embedding ITE model on {len(matched_pairs)} matched pairs")
    logger.info(f"  Hidden dim: {config.mean_ite_hidden_dim}, Dropout: {config.mean_ite_dropout}")
    logger.info(f"  Freeze base model: {freeze_base}")
    if clam_enabled:
        logger.info(f"  CLAM instance supervision: enabled (weight={config.clam_instance_weight_stage3})")

    # Check that propensity model has outcome head
    if not propensity_model.joint_outcome_training or propensity_model.outcome_head is None:
        raise ValueError(
            "Mean-embedding ITE model requires joint_outcome_training=True in Stage 1. "
            "The propensity model must have an outcome_head."
        )

    if freeze_base:
        # === FROZEN MODE ===
        # Freeze propensity model completely
        propensity_model.freeze_representation()
        propensity_model.eval()
        for param in propensity_model.propensity_head.parameters():
            param.requires_grad = False
        for param in propensity_model.outcome_head.parameters():
            param.requires_grad = False

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

        # Create dataset with pre-computed representations
        pair_dataset = MatchedPairDataset(representations, outcomes, remapped_pairs)
        pair_loader = DataLoader(
            pair_dataset,
            batch_size=config.outcome_batch_size,
            shuffle=True
        )

        # Create mean-embedding ITE model with frozen outcome head
        frozen_outcome_head = propensity_model.outcome_head
        ite_model = MeanEmbeddingITEModel(
            representation_dim=config.representation_dim,
            hidden_dim=config.mean_ite_hidden_dim,
            dropout=config.mean_ite_dropout,
            frozen_outcome_head=frozen_outcome_head
        ).to(device)

        # Optimizer: only ite_head parameters
        optimizer = torch.optim.AdamW(
            ite_model.ite_head.parameters(),
            lr=config.outcome_lr,
            weight_decay=0.01
        )

    else:
        # === TRAINABLE MODE ===
        # Keep propensity model trainable
        propensity_model.unfreeze_representation()
        propensity_model.train()
        # outcome_head remains trainable (no freeze)

        # Store all texts and outcomes (indexed by original dataframe position)
        all_texts = train_df[config.text_column].tolist()
        all_outcomes = train_df[config.outcome_column].values

        # Create text-based dataset
        pair_dataset = MatchedPairTextDataset(all_texts, all_outcomes, matched_pairs)
        pair_loader = DataLoader(
            pair_dataset,
            batch_size=config.outcome_batch_size,
            shuffle=True,
            collate_fn=collate_text_pairs
        )

        # Create mean-embedding ITE model WITHOUT frozen outcome head
        ite_model = MeanEmbeddingITEModel(
            representation_dim=config.representation_dim,
            hidden_dim=config.mean_ite_hidden_dim,
            dropout=config.mean_ite_dropout,
            frozen_outcome_head=None  # Will use external outcome head
        ).to(device)

        # Optimizer: ite_head + propensity_model (lower LR for base model)
        # Include instance_head if CLAM is enabled
        param_groups = [
            {'params': ite_model.ite_head.parameters(), 'lr': config.outcome_lr},
            {'params': propensity_model.parameters(), 'lr': config.outcome_lr * 0.1}
        ]
        if clam_enabled and instance_head is not None:
            param_groups.append({'params': instance_head.parameters(), 'lr': config.outcome_lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # Training loop
    history = []
    best_loss = float('inf')
    best_model_state = None
    best_propensity_state = None

    for epoch in range(config.outcome_epochs):
        ite_model.train()
        if not freeze_base:
            propensity_model.train()
            if clam_enabled and instance_head is not None:
                instance_head.train()

        epoch_losses = []
        epoch_clam_losses = []

        for batch in tqdm(pair_loader, desc=f"MeanITE Epoch {epoch+1}", leave=False):
            if freeze_base:
                # Frozen mode: use pre-extracted representations
                repr_U = batch['repr_U'].to(device)
                repr_T = batch['repr_T'].to(device)
                chunk_embs_T = None
                chunk_embs_U = None
                attn_weights_T = None
                attn_weights_U = None
            else:
                # Trainable mode: compute representations on-the-fly
                if clam_enabled:
                    # Get representations with instance info for CLAM
                    repr_T, chunk_embs_T, attn_weights_T = propensity_model.forward_with_instances(batch['texts_T'])
                    repr_U, chunk_embs_U, attn_weights_U = propensity_model.forward_with_instances(batch['texts_U'])
                else:
                    repr_T = propensity_model.get_representation(batch['texts_T'])
                    repr_U = propensity_model.get_representation(batch['texts_U'])
                    chunk_embs_T = None
                    chunk_embs_U = None
                    attn_weights_T = None
                    attn_weights_U = None

            y_U = batch['y_U'].to(device)
            y_T = batch['y_T'].to(device)

            optimizer.zero_grad()

            if freeze_base:
                # Frozen mode: use frozen outcome head (internal)
                Y_U_logit, Y_T_logit, ite_half = ite_model.forward_training(repr_U, repr_T)
            else:
                # Trainable mode: use external (trainable) outcome head
                Y_U_logit, Y_T_logit, ite_half = ite_model.forward_training(
                    repr_U, repr_T,
                    external_outcome_head=propensity_model.outcome_head
                )

            losses = mean_embedding_ite_loss(Y_U_logit, Y_T_logit, y_U, y_T)

            # Add CLAM loss if enabled (trainable mode only)
            if clam_enabled and instance_head is not None and chunk_embs_T is not None:
                # Combine treated and untreated chunks for CLAM
                all_chunk_embs = chunk_embs_T + chunk_embs_U
                all_attn_weights = attn_weights_T + attn_weights_U
                treatments = torch.cat([torch.ones_like(y_T), torch.zeros_like(y_U)])
                outcomes = torch.cat([y_T, y_U])

                clam_losses = compute_clam_loss(
                    all_chunk_embs, all_attn_weights,
                    treatments, outcomes,
                    instance_head,
                    num_instances=config.clam_num_instances
                )
                total_loss = losses['loss'] + config.clam_instance_weight_stage3 * clam_losses['clam_loss']
                epoch_clam_losses.append(clam_losses['clam_loss'].item())
            else:
                total_loss = losses['loss']

            total_loss.backward()
            optimizer.step()

            epoch_losses.append({k: v.item() for k, v in losses.items()})

        # Aggregate epoch losses
        epoch_summary = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        epoch_summary['epoch'] = epoch + 1
        if epoch_clam_losses:
            epoch_summary['clam_loss'] = np.mean(epoch_clam_losses)
        history.append(epoch_summary)

        clam_str = f", clam_loss={epoch_summary['clam_loss']:.4f}" if 'clam_loss' in epoch_summary else ""
        logger.info(f"  Epoch {epoch+1}: loss={epoch_summary['loss']:.4f}, "
                   f"loss_U={epoch_summary['loss_U']:.4f}, "
                   f"loss_T={epoch_summary['loss_T']:.4f}, "
                   f"ite_half_mean={epoch_summary['ite_half_mean']:.4f}{clam_str}")

        # Track best
        if epoch_summary['loss'] < best_loss:
            best_loss = epoch_summary['loss']
            best_model_state = {k: v.cpu().clone() for k, v in ite_model.state_dict().items()}
            if not freeze_base:
                best_propensity_state = {k: v.cpu().clone() for k, v in propensity_model.state_dict().items()}

    # Restore best
    if best_model_state is not None:
        ite_model.load_state_dict(best_model_state)
    if best_propensity_state is not None:
        propensity_model.load_state_dict(best_propensity_state)

    # For unfrozen mode, we need to set the outcome head on the ite_model for inference
    if not freeze_base:
        ite_model.set_frozen_outcome_head(propensity_model.outcome_head)
        # Freeze it now for inference consistency
        for param in ite_model.frozen_outcome_head.parameters():
            param.requires_grad = False

    # Cleanup
    del pair_loader, pair_dataset
    if freeze_base:
        del representations
    gc.collect()

    logger.info(f"Mean-embedding ITE model training complete. Best loss: {best_loss:.4f}")
    return ite_model, history


def train_propensity_model(
    model: PropensityMatchingModel,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: MatchedPairConfig,
    device: torch.device,
    instance_head: Optional[InstanceCausalHead] = None
) -> Tuple[PropensityMatchingModel, List[Dict[str, Any]], Optional[InstanceCausalHead]]:
    """
    Train propensity model to convergence.

    Stage 1 of the matched pair pipeline. Trains the propensity model
    using binary cross-entropy for treatment prediction.

    When joint_outcome_training is enabled in config, also trains on outcome
    prediction to learn features that are true confounders (predictive of
    both treatment and outcome).

    When clam_enabled is True and using gated pool extractors, adds CLAM
    instance-level supervision on top-attended chunks.

    Args:
        model: PropensityMatchingModel to train
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional). If None, trains for fixed epochs
            without early stopping.
        config: MatchedPairConfig with training settings
        device: PyTorch device
        instance_head: Optional InstanceCausalHead for CLAM (created if None and clam_enabled)

    Returns:
        Tuple of (trained_model, training_history, instance_head)
    """
    joint_training = config.joint_outcome_training and model.joint_outcome_training

    # Check if CLAM is enabled and model supports it
    clam_enabled = config.clam_enabled and config.chunk_encoder in ('bert_gated_pool', 'gru_pool')
    if config.clam_enabled and config.chunk_encoder not in ('bert_gated_pool', 'gru_pool'):
        logger.warning("CLAM is only supported with chunk_encoder='bert_gated_pool' or 'gru_pool'. Disabling CLAM.")
        clam_enabled = False

    logger.info(f"Training propensity model on {len(train_df)} samples")
    logger.info(f"  Epochs: {config.propensity_epochs}, LR: {config.propensity_lr}")
    logger.info(f"  Joint outcome training: {joint_training}")
    logger.info(f"  CLAM instance supervision: {clam_enabled}")
    if joint_training:
        logger.info(f"    alpha_propensity={config.alpha_propensity_stage1}, "
                   f"alpha_outcome={config.alpha_outcome_stage1}")
    if clam_enabled:
        logger.info(f"    clam_num_instances={config.clam_num_instances}, "
                   f"clam_weight={config.clam_instance_weight_stage1}")
    if val_df is None:
        logger.info("  No validation set - training for fixed epochs")

    # Create CLAM instance head if needed
    if clam_enabled and instance_head is None:
        # Get transformer_dim from the feature extractor
        transformer_dim = model.feature_extractor.transformer_dim
        instance_head = InstanceCausalHead(
            input_dim=transformer_dim,
            hidden_dim=config.clam_instance_hidden_dim,
            dropout=config.dropout
        ).to(device)
        logger.info(f"  Created InstanceCausalHead with input_dim={transformer_dim}")

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

    # Optimizer - include instance_head if CLAM is enabled
    if clam_enabled and instance_head is not None:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(instance_head.parameters()),
            lr=config.propensity_lr,
            weight_decay=0.01
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.propensity_lr,
            weight_decay=0.01
        )

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    best_instance_head_state = None
    patience_counter = 0
    history = []

    for epoch in range(config.propensity_epochs):
        # Train
        model.train()
        if clam_enabled and instance_head is not None:
            instance_head.train()

        train_loss = 0.0
        train_propensity_loss = 0.0
        train_outcome_loss = 0.0
        train_clam_loss = 0.0
        train_prop_preds, train_prop_targets = [], []
        train_out_preds, train_out_targets = [], []

        for batch in tqdm(train_loader, desc=f"Propensity Epoch {epoch+1}", leave=False):
            texts = batch['texts']
            treatment = batch['treatment'].to(device)
            outcome = batch['outcome'].to(device)

            optimizer.zero_grad()

            # Use forward_with_instances if CLAM is enabled
            if clam_enabled:
                repr, chunk_embs_list, attn_weights_list = model.forward_with_instances(texts)
                t_logit = model.propensity_head(repr)
                y_logit = model.outcome_head(repr) if joint_training else None
            elif joint_training:
                # Joint training: propensity + outcome
                t_logit, y_logit = model.forward_joint(texts)
            else:
                # Propensity only
                t_logit = model(texts)
                y_logit = None

            # Compute main losses
            propensity_loss = F.binary_cross_entropy_with_logits(t_logit.squeeze(-1), treatment)
            if joint_training:
                outcome_loss = F.binary_cross_entropy_with_logits(y_logit.squeeze(-1), outcome)
                loss = (config.alpha_propensity_stage1 * propensity_loss +
                        config.alpha_outcome_stage1 * outcome_loss)
                train_outcome_loss += outcome_loss.item()
                train_out_preds.append(torch.sigmoid(y_logit).detach().cpu())
                train_out_targets.append(outcome.detach().cpu())
            else:
                loss = propensity_loss

            train_propensity_loss += propensity_loss.item()
            train_prop_preds.append(torch.sigmoid(t_logit).detach().cpu())
            train_prop_targets.append(treatment.detach().cpu())

            # Add CLAM loss if enabled
            if clam_enabled and instance_head is not None:
                clam_losses = compute_clam_loss(
                    chunk_embs_list, attn_weights_list,
                    treatment, outcome,
                    instance_head,
                    num_instances=config.clam_num_instances
                )
                loss = loss + config.clam_instance_weight_stage1 * clam_losses['clam_loss']
                train_clam_loss += clam_losses['clam_loss'].item()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Compute training metrics
        n_batches = len(train_loader)
        train_loss_avg = train_loss / n_batches
        train_propensity_loss_avg = train_propensity_loss / n_batches
        train_outcome_loss_avg = train_outcome_loss / n_batches if joint_training else None
        train_clam_loss_avg = train_clam_loss / n_batches if clam_enabled else None

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
            'train_clam_loss': train_clam_loss_avg,
            'train_propensity_auroc': train_propensity_auroc,
            'train_outcome_auroc': train_outcome_auroc,
            'val_loss': val_loss_avg,
            'val_propensity_loss': val_propensity_loss_avg,
            'val_outcome_loss': val_outcome_loss_avg,
            'val_propensity_auroc': val_propensity_auroc,
            'val_outcome_auroc': val_outcome_auroc
        })

        if val_loss_avg is not None:
            clam_str = f", clam_loss={train_clam_loss_avg:.4f}" if clam_enabled else ""
            if joint_training:
                logger.info(f"  Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, "
                           f"val_loss={val_loss_avg:.4f}, "
                           f"val_prop_auroc={val_propensity_auroc:.4f}, "
                           f"val_out_auroc={val_outcome_auroc:.4f}{clam_str}")
            else:
                logger.info(f"  Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, "
                           f"val_loss={val_loss_avg:.4f}, val_auroc={val_propensity_auroc:.4f}{clam_str}")

            # Early stopping (only when validation is available)
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if clam_enabled and instance_head is not None:
                    best_instance_head_state = {k: v.cpu().clone() for k, v in instance_head.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.propensity_early_stopping_patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break
        else:
            clam_str = f", clam_loss={train_clam_loss_avg:.4f}" if clam_enabled else ""
            if joint_training:
                logger.info(f"  Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, "
                           f"prop_auroc={train_propensity_auroc:.4f}, "
                           f"out_auroc={train_outcome_auroc:.4f}{clam_str}")
            else:
                logger.info(f"  Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, "
                           f"train_auroc={train_propensity_auroc:.4f}{clam_str}")

    # Restore best model (only if we had validation-based early stopping)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_instance_head_state is not None and instance_head is not None:
        instance_head.load_state_dict(best_instance_head_state)

    # Cleanup
    del train_loader
    if val_loader is not None:
        del val_loader
    del train_dataset
    gc.collect()

    return model, history, instance_head


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
                caliper_scale=config.caliper_scale,
                replacement=config.match_with_replacement
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


# =============================================================================
# ENHANCED CROSS-ENCODER TRAINING
# =============================================================================

def enhanced_matched_pair_loss(
    y_U_logit: torch.Tensor,
    y_T_logit: torch.Tensor,
    tau_pred: torch.Tensor,
    y_U: torch.Tensor,
    y_T: torch.Tensor,
    treatment_logit: Optional[torch.Tensor] = None,
    alpha_outcome: float = 1.0,
    beta_tau: float = 1.0,
    gamma_discrimination: float = 0.1,
    delta_consistency: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Enhanced matched pair loss with cross-encoder auxiliary losses.

    Loss = α * L_outcome + β * L_tau + γ * L_disc + δ * L_consistency

    Where:
    - L_outcome = BCE(σ(y_U_logit), y_U) + BCE(σ(y_T_logit), y_T)
    - L_tau = MSE(tau_pred, detach(y_T_logit - y_U_logit))
    - L_disc = BCE(σ(treatment_logit), 1) - encourages discriminative features
    - L_consistency = MSE(tau_pred, y_T_logit - y_U_logit) - tau ≈ implicit difference

    Args:
        y_U_logit: Predicted outcome logit for untreated (B, 1)
        y_T_logit: Predicted outcome logit for treated (B, 1)
        tau_pred: Predicted tau on log-odds scale (B, 1)
        y_U: Actual outcome for untreated (B,)
        y_T: Actual outcome for treated (B,)
        treatment_logit: Treatment discrimination logit from cross-encoder (B, 1)
        alpha_outcome: Weight for outcome loss
        beta_tau: Weight for tau loss
        gamma_discrimination: Weight for discrimination loss
        delta_consistency: Weight for consistency loss

    Returns:
        Dictionary with loss components
    """
    # Outcome BCE loss (both patients)
    outcome_loss_U = F.binary_cross_entropy_with_logits(
        y_U_logit.squeeze(-1), y_U
    )
    outcome_loss_T = F.binary_cross_entropy_with_logits(
        y_T_logit.squeeze(-1), y_T
    )
    outcome_loss = outcome_loss_U + outcome_loss_T

    # Tau target: signed log-odds difference (detached)
    tau_target = y_T_logit.detach() - y_U_logit.detach()
    tau_loss = F.mse_loss(tau_pred, tau_target)

    # Total loss starts with outcome + tau
    total_loss = alpha_outcome * outcome_loss + beta_tau * tau_loss

    # Optional: Discrimination loss (encourages cross-encoder to learn discriminative features)
    disc_loss = torch.tensor(0.0, device=y_U_logit.device)
    if treatment_logit is not None and gamma_discrimination > 0:
        # Target: treated patient should be identified (label=1)
        batch_size = treatment_logit.size(0)
        disc_target = torch.ones(batch_size, device=treatment_logit.device)
        disc_loss = F.binary_cross_entropy_with_logits(
            treatment_logit.squeeze(-1), disc_target
        )
        total_loss = total_loss + gamma_discrimination * disc_loss

    # Optional: Consistency loss (tau should match implicit outcome difference)
    consistency_loss = torch.tensor(0.0, device=y_U_logit.device)
    if delta_consistency > 0:
        # Non-detached: encourages tau to be consistent with outcome predictions
        implicit_tau = y_T_logit - y_U_logit
        consistency_loss = F.mse_loss(tau_pred, implicit_tau)
        total_loss = total_loss + delta_consistency * consistency_loss

    return {
        'loss': total_loss,
        'outcome_loss': outcome_loss,
        'outcome_loss_U': outcome_loss_U,
        'outcome_loss_T': outcome_loss_T,
        'tau_loss': tau_loss,
        'disc_loss': disc_loss,
        'consistency_loss': consistency_loss,
        'tau_pred_mean': tau_pred.mean(),
        'tau_target_mean': tau_target.mean()
    }


class MatchedPairSentenceDataset(Dataset):
    """
    Dataset of matched pairs with sentence embeddings for cross-encoder.

    Each item includes representations, outcomes, and sentence embeddings
    for both treated and untreated patients.

    Args:
        representations: Tensor of all patient representations (N, D)
        sentence_embeddings: List of sentence embedding tensors [(S_i, D), ...]
        outcomes: Array of binary outcomes (N,)
        matched_pairs: Array of (treated_idx, control_idx) pairs (M, 2)
    """

    def __init__(
        self,
        representations: torch.Tensor,
        sentence_embeddings: List[torch.Tensor],
        outcomes: np.ndarray,
        matched_pairs: np.ndarray
    ):
        self.representations = representations
        self.sentence_embeddings = sentence_embeddings
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32)
        self.matched_pairs = matched_pairs

        logger.info(f"MatchedPairSentenceDataset: {len(matched_pairs)} pairs, "
                   f"{len(representations)} patients")

    def __len__(self) -> int:
        return len(self.matched_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        treated_idx, control_idx = self.matched_pairs[idx]
        return {
            'repr_T': self.representations[treated_idx],
            'repr_U': self.representations[control_idx],
            'sent_T': self.sentence_embeddings[treated_idx],
            'sent_U': self.sentence_embeddings[control_idx],
            'y_T': self.outcomes[treated_idx],
            'y_U': self.outcomes[control_idx],
            'treated_idx': treated_idx,
            'control_idx': control_idx
        }


def collate_sentence_pairs(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for sentence pair batches."""
    return {
        'repr_T': torch.stack([b['repr_T'] for b in batch]),
        'repr_U': torch.stack([b['repr_U'] for b in batch]),
        'sent_T': [b['sent_T'] for b in batch],  # Variable length
        'sent_U': [b['sent_U'] for b in batch],  # Variable length
        'y_T': torch.stack([b['y_T'] for b in batch]),
        'y_U': torch.stack([b['y_U'] for b in batch]),
        'treated_idx': [b['treated_idx'] for b in batch],
        'control_idx': [b['control_idx'] for b in batch]
    }


def extract_sentence_embeddings(
    propensity_model: PropensityMatchingModel,
    texts: List[str],
    batch_size: int = 32,
    device: torch.device = None
) -> List[torch.Tensor]:
    """
    Extract sentence-level embeddings for cross-encoder input.

    Uses the propensity model's feature extractor to get sentence embeddings
    before the final pooling step.

    Args:
        propensity_model: Trained PropensityMatchingModel
        texts: List of document texts
        batch_size: Batch size for extraction
        device: PyTorch device

    Returns:
        List of sentence embedding tensors [(S_i, D), ...] for each text
    """
    propensity_model.eval()
    all_sent_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Get sentence embeddings from the hierarchical transformer
            # This accesses the feature extractor's sentence encoding
            extractor = propensity_model.feature_extractor

            for text in batch_texts:
                # Get sentence embeddings for single text
                # The HierarchicalTransformerExtractor stores sentence-level info
                sent_emb = extractor.get_sentence_embeddings([text])  # (1, S, D)
                if sent_emb is not None:
                    sent_emb = sent_emb.squeeze(0).cpu()  # (S, D)
                else:
                    # Fallback: use representation as single "sentence"
                    repr = propensity_model.get_representation([text])
                    sent_emb = repr.cpu()  # (1, D)

                all_sent_embeddings.append(sent_emb)

    return all_sent_embeddings


def train_matched_pair_outcome_model_enhanced(
    propensity_model: PropensityMatchingModel,
    train_df: pd.DataFrame,
    matched_pairs: np.ndarray,
    config: MatchedPairConfig,
    device: torch.device
) -> Tuple[EnhancedMatchedPairOutcomeModel, List[Dict[str, Any]]]:
    """
    Train enhanced outcome/tau model with cross-encoder on matched pairs.

    Stage 3 of the matched pair pipeline with cross-encoder support:
    1. Extract representations and sentence embeddings for all matched patients
    2. Create MatchedPairSentenceDataset
    3. Train EnhancedMatchedPairOutcomeModel with cross-encoder losses

    Args:
        propensity_model: Trained PropensityMatchingModel
        train_df: Training DataFrame (must contain all matched patient indices)
        matched_pairs: Array of (treated_idx, control_idx) pairs
        config: MatchedPairConfig with training settings
        device: PyTorch device

    Returns:
        Tuple of (trained_outcome_model, training_history)
    """
    logger.info(f"Training enhanced outcome/tau model with cross-encoder on {len(matched_pairs)} pairs")
    logger.info(f"  Cross-encoder: num_queries={config.cross_encoder_num_queries}, "
               f"num_heads={config.cross_encoder_num_heads}")
    logger.info(f"  Loss weights: gamma_discrimination={config.gamma_discrimination}, "
               f"delta_consistency={config.delta_consistency}")

    # Freeze representation for Stage 2
    propensity_model.freeze_representation()
    propensity_model.eval()

    # Get all unique patient indices from matched pairs
    all_indices = np.unique(matched_pairs.flatten())

    # Extract texts and outcomes for matched patients
    texts = train_df.iloc[all_indices][config.text_column].tolist()
    outcomes = train_df.iloc[all_indices][config.outcome_column].values

    # Extract representations
    logger.info(f"  Extracting representations for {len(all_indices)} patients...")
    representations = []
    batch_size = config.outcome_batch_size

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_repr = propensity_model.get_representation(batch_texts)
            representations.append(batch_repr.cpu())

    representations = torch.cat(representations, dim=0)  # (N, D)

    # Extract sentence embeddings for cross-encoder
    logger.info("  Extracting sentence embeddings...")
    sentence_embeddings = extract_sentence_embeddings(
        propensity_model, texts, batch_size, device
    )

    # Create index mapping: original_idx -> representation_idx
    idx_to_repr_idx = {orig_idx: i for i, orig_idx in enumerate(all_indices)}

    # Remap matched pairs to representation indices
    remapped_pairs = np.array([
        [idx_to_repr_idx[t], idx_to_repr_idx[c]]
        for t, c in matched_pairs
    ])

    # Create dataset with sentence embeddings
    pair_dataset = MatchedPairSentenceDataset(
        representations, sentence_embeddings, outcomes, remapped_pairs
    )
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=config.outcome_batch_size,
        shuffle=True,
        collate_fn=collate_sentence_pairs
    )

    # Create enhanced outcome model with cross-encoder
    outcome_model = EnhancedMatchedPairOutcomeModel(
        representation_dim=config.representation_dim,
        hidden_dim=config.hidden_outcome_dim,
        dropout=config.dropout,
        use_cross_encoder=True,
        cross_encoder_num_queries=config.cross_encoder_num_queries,
        cross_encoder_num_heads=config.cross_encoder_num_heads,
        cross_encoder_hidden_dim=config.cross_encoder_hidden_dim,
        cross_encoder_use_gating=config.cross_encoder_use_gating
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

        for batch in tqdm(pair_loader, desc=f"Enhanced Epoch {epoch+1}", leave=False):
            repr_U = batch['repr_U'].to(device)
            repr_T = batch['repr_T'].to(device)
            sent_U = [s.to(device) for s in batch['sent_U']]
            sent_T = [s.to(device) for s in batch['sent_T']]
            y_U = batch['y_U'].to(device)
            y_T = batch['y_T'].to(device)

            optimizer.zero_grad()

            # Forward pass with sentence embeddings
            y_U_logit, y_T_logit, tau_pred, _ = outcome_model(
                repr_U, repr_T, sent_U, sent_T, return_attention=False
            )

            # Get treatment discrimination logit (auxiliary task)
            treatment_logit = None
            if config.gamma_discrimination > 0:
                treatment_logit = outcome_model.predict_treatment_from_residual(sent_T, sent_U)

            # Compute enhanced loss
            losses = enhanced_matched_pair_loss(
                y_U_logit, y_T_logit, tau_pred, y_U, y_T,
                treatment_logit=treatment_logit,
                alpha_outcome=config.alpha_outcome,
                beta_tau=config.beta_tau,
                gamma_discrimination=config.gamma_discrimination,
                delta_consistency=config.delta_consistency
            )

            losses['loss'].backward()
            optimizer.step()

            epoch_losses.append({k: v.item() for k, v in losses.items()})

        # Aggregate epoch losses
        epoch_summary = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        epoch_summary['epoch'] = epoch + 1
        history.append(epoch_summary)

        logger.info(f"  Epoch {epoch+1}: loss={epoch_summary['loss']:.4f}, "
                   f"outcome={epoch_summary['outcome_loss']:.4f}, "
                   f"tau={epoch_summary['tau_loss']:.4f}, "
                   f"disc={epoch_summary['disc_loss']:.4f}, "
                   f"consist={epoch_summary['consistency_loss']:.4f}")

        # Track best
        if epoch_summary['loss'] < best_loss:
            best_loss = epoch_summary['loss']
            best_model_state = {k: v.cpu().clone() for k, v in outcome_model.state_dict().items()}

    # Restore best
    if best_model_state is not None:
        outcome_model.load_state_dict(best_model_state)

    # Cleanup
    del pair_loader, pair_dataset, representations, sentence_embeddings
    gc.collect()

    return outcome_model, history


# =============================================================================
# END-TO-END MATCHED PAIR TRAINING
# =============================================================================

def end_to_end_matched_pair_loss(
    output: Dict[str, torch.Tensor],
    y_U: torch.Tensor,
    y_T: torch.Tensor,
    alpha_propensity: float = 1.0,
    alpha_outcome: float = 1.0,
    beta_tau: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Compute loss for end-to-end matched pair training.

    Joint loss = α_prop * L_propensity + α_out * L_outcome + β * L_tau

    Where:
    - L_propensity = BCE(t_logit_T, 1) + BCE(t_logit_U, 0)
    - L_outcome = BCE(y_U_logit, y_U) + BCE(y_T_logit, y_T)
    - L_tau = MSE(tau_pred, detach(y_T_logit - y_U_logit))

    Args:
        output: Dictionary from EndToEndMatchedPairModel.forward_matched_pair()
        y_U: Actual outcome for untreated (B,)
        y_T: Actual outcome for treated (B,)
        alpha_propensity: Weight for propensity loss
        alpha_outcome: Weight for outcome loss
        beta_tau: Weight for tau loss

    Returns:
        Dictionary with loss components
    """
    batch_size = y_U.size(0)
    device = y_U.device

    # Propensity loss: T should be predicted as treated (1), U as control (0)
    t_target_T = torch.ones(batch_size, device=device)
    t_target_U = torch.zeros(batch_size, device=device)

    propensity_loss_T = F.binary_cross_entropy_with_logits(
        output['t_logit_T'].squeeze(-1), t_target_T
    )
    propensity_loss_U = F.binary_cross_entropy_with_logits(
        output['t_logit_U'].squeeze(-1), t_target_U
    )
    propensity_loss = propensity_loss_T + propensity_loss_U

    # Outcome loss: predict Y for both T and U
    outcome_loss_U = F.binary_cross_entropy_with_logits(
        output['y_U_logit'].squeeze(-1), y_U
    )
    outcome_loss_T = F.binary_cross_entropy_with_logits(
        output['y_T_logit'].squeeze(-1), y_T
    )
    outcome_loss = outcome_loss_U + outcome_loss_T

    # Tau loss: tau should match log-odds difference (detached for stability)
    tau_target = output['y_T_logit'].detach() - output['y_U_logit'].detach()
    tau_loss = F.mse_loss(output['tau_pred'], tau_target)

    # Total loss
    total_loss = (alpha_propensity * propensity_loss +
                  alpha_outcome * outcome_loss +
                  beta_tau * tau_loss)

    return {
        'loss': total_loss,
        'propensity_loss': propensity_loss,
        'propensity_loss_T': propensity_loss_T,
        'propensity_loss_U': propensity_loss_U,
        'outcome_loss': outcome_loss,
        'outcome_loss_U': outcome_loss_U,
        'outcome_loss_T': outcome_loss_T,
        'tau_loss': tau_loss,
        'tau_pred_mean': output['tau_pred'].mean(),
        'tau_target_mean': tau_target.mean()
    }


def _compute_initial_matches_e2e(
    model: EndToEndMatchedPairModel,
    train_df: pd.DataFrame,
    config: MatchedPairConfig,
    device: torch.device
) -> np.ndarray:
    """
    Compute initial matches for end-to-end training.

    Uses a relaxed caliper since the model is not yet trained.

    Args:
        model: EndToEndMatchedPairModel (may be random/untrained)
        train_df: Training DataFrame
        config: MatchedPairConfig
        device: PyTorch device

    Returns:
        matched_pairs array of shape (n_pairs, 2)
    """
    texts = train_df[config.text_column].tolist()
    treatment = train_df[config.treatment_column].values

    # Relaxed caliper for initial matching
    relaxed_caliper = config.caliper * config.e2e_initial_caliper_multiplier

    model.eval()
    with torch.no_grad():
        if config.e2e_initial_matching == "random":
            # Random matching: just pair treated/control randomly
            treated_idx = np.where(treatment == 1)[0]
            control_idx = np.where(treatment == 0)[0]
            np.random.shuffle(control_idx)
            n_pairs = min(len(treated_idx), len(control_idx))
            matched_pairs = np.column_stack([treated_idx[:n_pairs], control_idx[:n_pairs]])
            logger.info(f"  Initial random matching: {n_pairs} pairs")
        elif config.e2e_initial_matching == "embedding":
            # Embedding-based matching
            representations = []
            batch_size = config.e2e_batch_size
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_repr = model.get_representation(batch_texts)
                representations.append(batch_repr.cpu())
            representations = torch.cat(representations, dim=0)

            match_result = match_by_cosine_similarity(
                representations.numpy(), treatment,
                caliper=relaxed_caliper,
                method=config.matching_algorithm
            )
            matched_pairs = match_result.matched_pairs
            logger.info(f"  Initial embedding matching: {len(matched_pairs)} pairs "
                       f"(mean dist: {match_result.distances.mean():.4f})")
        else:
            # Propensity-based matching (default)
            propensity_scores = []
            batch_size = config.e2e_batch_size
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_scores = model.predict_propensity(batch_texts)
                propensity_scores.append(batch_scores.cpu().numpy())
            propensity_scores = np.concatenate(propensity_scores)

            matcher = PropensityMatcher(
                method=config.matching_algorithm,
                caliper=relaxed_caliper,
                caliper_scale=config.caliper_scale,
                replacement=config.match_with_replacement
            )
            match_result = matcher.match(propensity_scores, treatment)
            matched_pairs = match_result.matched_pairs
            logger.info(f"  Initial propensity matching: {len(matched_pairs)} pairs "
                       f"(mean dist: {match_result.distances.mean():.4f})")

    model.train()
    return matched_pairs


def _recompute_matches_e2e(
    model: EndToEndMatchedPairModel,
    train_df: pd.DataFrame,
    config: MatchedPairConfig,
    device: torch.device
) -> np.ndarray:
    """
    Re-compute matches based on current model state.

    Uses the configured caliper (not relaxed) since model is partially trained.

    Args:
        model: EndToEndMatchedPairModel (partially trained)
        train_df: Training DataFrame
        config: MatchedPairConfig
        device: PyTorch device

    Returns:
        matched_pairs array of shape (n_pairs, 2)
    """
    texts = train_df[config.text_column].tolist()
    treatment = train_df[config.treatment_column].values

    model.eval()
    with torch.no_grad():
        if config.matching_method == "embedding":
            representations = []
            batch_size = config.e2e_batch_size
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_repr = model.get_representation(batch_texts)
                representations.append(batch_repr.cpu())
            representations = torch.cat(representations, dim=0)

            match_result = match_by_cosine_similarity(
                representations.numpy(), treatment,
                caliper=config.caliper,
                method=config.matching_algorithm
            )
        else:
            propensity_scores = []
            batch_size = config.e2e_batch_size
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_scores = model.predict_propensity(batch_texts)
                propensity_scores.append(batch_scores.cpu().numpy())
            propensity_scores = np.concatenate(propensity_scores)

            matcher = PropensityMatcher(
                method=config.matching_algorithm,
                caliper=config.caliper,
                caliper_scale=config.caliper_scale,
                replacement=config.match_with_replacement
            )
            match_result = matcher.match(propensity_scores, treatment)

    model.train()

    logger.info(f"    Re-matched: {len(match_result.matched_pairs)} pairs "
               f"(mean dist: {match_result.distances.mean():.4f})")

    return match_result.matched_pairs


def train_end_to_end_matched_pair(
    model: EndToEndMatchedPairModel,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: MatchedPairConfig,
    device: torch.device
) -> Tuple[EndToEndMatchedPairModel, List[Dict[str, Any]]]:
    """
    End-to-end training with joint propensity + outcome + tau learning.

    Single model, single optimizer, periodic re-matching. Unlike the 3-stage
    approach, this jointly trains all heads from scratch, with matches
    recomputed periodically as the model improves.

    Training loop:
    1. Compute initial matches (from random/untrained model with relaxed caliper)
    2. For each epoch:
       a. If past warmup and epoch % rematching_frequency == 0: re-compute matches
       b. For each batch of matched pairs:
          - Forward pass through matched pair forward
          - Compute joint loss (propensity + outcome + tau)
          - Backward + optimizer step
       c. Validation metrics if val_df provided
       d. Early stopping check

    Args:
        model: EndToEndMatchedPairModel to train
        train_df: Training DataFrame with text, treatment, outcome columns
        val_df: Optional validation DataFrame for early stopping
        config: MatchedPairConfig with e2e training settings
        device: PyTorch device

    Returns:
        Tuple of (trained_model, training_history)
    """
    logger.info(f"End-to-end matched pair training on {len(train_df)} samples")
    logger.info(f"  Epochs: {config.e2e_epochs}, LR: {config.e2e_lr}")
    logger.info(f"  Loss weights: α_prop={config.e2e_alpha_propensity}, "
               f"α_out={config.e2e_alpha_outcome}, β_tau={config.e2e_beta_tau}")
    logger.info(f"  Re-matching: every {config.e2e_rematching_frequency} epochs "
               f"(warmup: {config.e2e_rematching_warmup_epochs})")
    logger.info(f"  Initial matching: {config.e2e_initial_matching} "
               f"(caliper multiplier: {config.e2e_initial_caliper_multiplier})")

    text_col = config.text_column
    treatment_col = config.treatment_column
    outcome_col = config.outcome_column

    # Store texts and outcomes
    all_texts = train_df[text_col].tolist()
    all_outcomes = train_df[outcome_col].values

    # Step 1: Compute initial matches
    logger.info("Computing initial matches from untrained model...")
    matched_pairs = _compute_initial_matches_e2e(model, train_df, config, device)

    if len(matched_pairs) < 10:
        logger.warning(f"Very few initial matched pairs ({len(matched_pairs)})! "
                      "Consider relaxing caliper or using random initial matching.")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.e2e_lr,
        weight_decay=0.01
    )

    # Learning rate scheduler
    total_steps = config.e2e_epochs
    warmup_steps = int(config.e2e_warmup_ratio * total_steps)

    if config.e2e_lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    elif config.e2e_lr_schedule == "linear":
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    else:
        scheduler = None

    # Training state
    history = []
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.e2e_epochs):
        # Check for re-matching
        if (epoch >= config.e2e_rematching_warmup_epochs and
            epoch > 0 and
            epoch % config.e2e_rematching_frequency == 0):
            logger.info(f"  Epoch {epoch+1}: Re-computing matches...")
            matched_pairs = _recompute_matches_e2e(model, train_df, config, device)

            if len(matched_pairs) < 10:
                logger.warning(f"  Very few matched pairs ({len(matched_pairs)}) after re-matching!")

        # Create dataset for current matches
        pair_dataset = MatchedPairTextDataset(all_texts, all_outcomes, matched_pairs)
        pair_loader = DataLoader(
            pair_dataset,
            batch_size=config.e2e_batch_size,
            shuffle=True,
            collate_fn=collate_text_pairs
        )

        # Training epoch
        model.train()
        epoch_losses = []

        for batch in tqdm(pair_loader, desc=f"E2E Epoch {epoch+1}", leave=False):
            texts_T = batch['texts_T']
            texts_U = batch['texts_U']
            y_T = batch['y_T'].to(device)
            y_U = batch['y_U'].to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model.forward_matched_pair(texts_T, texts_U)

            # Compute loss
            losses = end_to_end_matched_pair_loss(
                output, y_U, y_T,
                alpha_propensity=config.e2e_alpha_propensity,
                alpha_outcome=config.e2e_alpha_outcome,
                beta_tau=config.e2e_beta_tau
            )

            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append({k: v.item() for k, v in losses.items()})

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Aggregate epoch losses
        epoch_summary = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        epoch_summary['epoch'] = epoch + 1
        epoch_summary['n_matched_pairs'] = len(matched_pairs)
        epoch_summary['lr'] = optimizer.param_groups[0]['lr']

        # Validation (if provided)
        val_loss = None
        if val_df is not None:
            val_loss = _validate_e2e(model, val_df, config, device)
            epoch_summary['val_loss'] = val_loss

        history.append(epoch_summary)

        # Logging
        log_msg = (f"  Epoch {epoch+1}: loss={epoch_summary['loss']:.4f}, "
                  f"prop={epoch_summary['propensity_loss']:.4f}, "
                  f"out={epoch_summary['outcome_loss']:.4f}, "
                  f"tau={epoch_summary['tau_loss']:.4f}, "
                  f"pairs={epoch_summary['n_matched_pairs']}")
        if val_loss is not None:
            log_msg += f", val_loss={val_loss:.4f}"
        logger.info(log_msg)

        # Early stopping
        current_loss = val_loss if val_loss is not None else epoch_summary['loss']
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.e2e_early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        # Cleanup loader each epoch
        del pair_loader, pair_dataset

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from epoch with loss={best_loss:.4f}")

    gc.collect()
    return model, history


def _validate_e2e(
    model: EndToEndMatchedPairModel,
    val_df: pd.DataFrame,
    config: MatchedPairConfig,
    device: torch.device
) -> float:
    """
    Compute validation loss for end-to-end model.

    Computes propensity and outcome prediction loss on validation data
    (without matching, since validation data shouldn't be matched).

    Args:
        model: EndToEndMatchedPairModel
        val_df: Validation DataFrame
        config: MatchedPairConfig
        device: PyTorch device

    Returns:
        Validation loss (propensity + outcome)
    """
    model.eval()
    val_texts = val_df[config.text_column].tolist()
    val_treatment = val_df[config.treatment_column].values
    val_outcome = val_df[config.outcome_column].values

    total_loss = 0.0
    n_batches = 0
    batch_size = config.e2e_batch_size

    with torch.no_grad():
        for i in range(0, len(val_texts), batch_size):
            batch_texts = val_texts[i:i + batch_size]
            batch_treatment = torch.tensor(val_treatment[i:i + batch_size], dtype=torch.float32, device=device)
            batch_outcome = torch.tensor(val_outcome[i:i + batch_size], dtype=torch.float32, device=device)

            repr = model.get_representation(batch_texts)
            t_logit = model.propensity_head(repr)
            y_logit = model._outcome_forward(repr)

            prop_loss = F.binary_cross_entropy_with_logits(t_logit.squeeze(-1), batch_treatment)
            out_loss = F.binary_cross_entropy_with_logits(y_logit.squeeze(-1), batch_outcome)

            total_loss += (prop_loss.item() + out_loss.item())
            n_batches += 1

    model.train()
    return total_loss / n_batches if n_batches > 0 else float('inf')
