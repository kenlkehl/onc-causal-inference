#!/usr/bin/env python
"""Run remaining extractor + model combinations with multi-GPU support.

Runs oracle (patient_prompt) and clinical_text conditions only.
Skips LLM extraction experiments.

This script runs all missing combinations of feature extractors and modeling strategies
(dragonnet, rlearner, causal_forest, uplift, traditional_logreg) for oracle and clinical_text
conditions.

Missing combinations (26 total):
- CNN: causal_forest, uplift, traditional_logreg
- BERT: dragonnet, causal_forest, uplift, traditional_logreg
- GRU: dragonnet, rlearner, causal_forest, uplift, traditional_logreg
- Confounder: dragonnet, causal_forest, uplift, traditional_logreg
- Hierarchical Transformer: causal_forest, uplift, traditional_logreg
- Gated MIL Hierarchical: uplift, traditional_logreg
- GRU Transformer MIL: causal_forest, uplift, traditional_logreg
- GRU Pool: uplift, traditional_logreg

Usage:
    # Run all missing experiments on cuda:0
    python oracle_experiment_scripts/run_remaining_experiments.py \
        --dataset ../pcori_experiments/explicit_confounder_experiments_1-19-26/dataset_with_extraction.parquet \
        --output-dir ../pcori_experiments/explicit_confounder_experiments_1-19-26/remaining_experiments \
        --device cuda:0 \
        --epochs 20

    # Run specific experiments (for multi-GPU parallelism)
    python run_remaining_experiments.py --device cuda:0 --experiments bert_dragonnet gru_dragonnet &
    python run_remaining_experiments.py --device cuda:1 --experiments bert_causal_forest gru_rlearner &

    # List all available experiments
    python run_remaining_experiments.py --list-experiments
"""

import argparse
import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdt.models.causal_text import CausalText
from cdt.models.causal_text_forest import CausalTextForest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Definitions (26 missing combinations)
# =============================================================================

EXPERIMENTS = [
    # CNN experiments (3)
    {"extractor": "cnn", "model_type": "causal_forest", "name": "cnn_causal_forest"},
    {"extractor": "cnn", "model_type": "uplift", "name": "cnn_uplift"},
    {"extractor": "cnn", "model_type": "traditional_logreg", "name": "cnn_traditional_logreg"},

    # BERT experiments (4)
    {"extractor": "bert", "model_type": "dragonnet", "name": "bert_dragonnet"},
    {"extractor": "bert", "model_type": "causal_forest", "name": "bert_causal_forest"},
    {"extractor": "bert", "model_type": "uplift", "name": "bert_uplift"},
    {"extractor": "bert", "model_type": "traditional_logreg", "name": "bert_traditional_logreg"},

    # GRU experiments (5)
    {"extractor": "gru", "model_type": "dragonnet", "name": "gru_dragonnet"},
    {"extractor": "gru", "model_type": "rlearner", "name": "gru_rlearner"},
    {"extractor": "gru", "model_type": "causal_forest", "name": "gru_causal_forest"},
    {"extractor": "gru", "model_type": "uplift", "name": "gru_uplift"},
    {"extractor": "gru", "model_type": "traditional_logreg", "name": "gru_traditional_logreg"},

    # Confounder experiments (4)
    {"extractor": "confounder", "model_type": "dragonnet", "name": "confounder_dragonnet"},
    {"extractor": "confounder", "model_type": "causal_forest", "name": "confounder_causal_forest"},
    {"extractor": "confounder", "model_type": "uplift", "name": "confounder_uplift"},
    {"extractor": "confounder", "model_type": "traditional_logreg", "name": "confounder_traditional_logreg"},

    # Hierarchical Transformer experiments (3)
    {"extractor": "hierarchical_transformer", "model_type": "causal_forest", "name": "hier_transformer_causal_forest"},
    {"extractor": "hierarchical_transformer", "model_type": "uplift", "name": "hier_transformer_uplift"},
    {"extractor": "hierarchical_transformer", "model_type": "traditional_logreg", "name": "hier_transformer_traditional_logreg"},

    # Gated MIL Hierarchical experiments (2)
    {"extractor": "gated_mil_hierarchical", "model_type": "uplift", "name": "gated_mil_uplift"},
    {"extractor": "gated_mil_hierarchical", "model_type": "traditional_logreg", "name": "gated_mil_traditional_logreg"},

    # GRU Transformer MIL experiments (3)
    {"extractor": "gru_transformer_mil", "model_type": "causal_forest", "name": "gru_mil_causal_forest"},
    {"extractor": "gru_transformer_mil", "model_type": "uplift", "name": "gru_mil_uplift"},
    {"extractor": "gru_transformer_mil", "model_type": "traditional_logreg", "name": "gru_mil_traditional_logreg"},

    # GRU Pool experiments (2)
    {"extractor": "gru_pool", "model_type": "uplift", "name": "gru_pool_uplift"},
    {"extractor": "gru_pool", "model_type": "traditional_logreg", "name": "gru_pool_traditional_logreg"},
]

# Conditions to run for each experiment (oracle and clinical_text only, no LLM)
CONDITIONS = [
    {"name": "1_oracle", "text_column": "patient_prompt"},
    {"name": "2_clinical_text", "text_column": "clinical_text"},
]


def _get_device(device_str: str) -> torch.device:
    """Get device with MPS/CUDA/CPU fallback."""
    if device_str == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        return torch.device("cpu")
    return torch.device(device_str)


class TextDataset(Dataset):
    """Simple dataset for text + labels."""

    def __init__(
        self,
        texts: List[str],
        treatments: np.ndarray,
        outcomes: np.ndarray
    ):
        self.texts = texts
        self.treatments = torch.tensor(treatments, dtype=torch.float32)
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'texts': self.texts[idx],
            'treatment': self.treatments[idx],
            'outcome': self.outcomes[idx]
        }


def collate_text_batch(batch):
    """Collate function for text batches."""
    texts = [b['texts'] for b in batch]
    treatments = torch.stack([b['treatment'] for b in batch])
    outcomes = torch.stack([b['outcome'] for b in batch])

    return {
        'texts': texts,
        'treatment': treatments,
        'outcome': outcomes
    }


def compute_metrics(
    pred_ite: np.ndarray,
    true_ite: np.ndarray,
    pred_propensity: np.ndarray,
    true_treatment: np.ndarray,
    pred_y0: np.ndarray,
    pred_y1: np.ndarray,
    true_y0: np.ndarray,
    true_y1: np.ndarray,
    true_outcome: np.ndarray,
    tau_lower: Optional[np.ndarray] = None,
    tau_upper: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    metrics = {}

    # ITE metrics
    metrics['ite_mse'] = mean_squared_error(true_ite, pred_ite)
    metrics['ite_mae'] = mean_absolute_error(true_ite, pred_ite)
    metrics['ite_corr'], _ = stats.pearsonr(pred_ite, true_ite)
    metrics['ite_spearman_corr'], _ = stats.spearmanr(pred_ite, true_ite)
    metrics['ate_bias'] = abs(np.mean(pred_ite) - np.mean(true_ite))
    metrics['ate_pred'] = np.mean(pred_ite)
    metrics['ate_true'] = np.mean(true_ite)

    # Propensity metrics
    try:
        metrics['propensity_auroc'] = roc_auc_score(true_treatment, pred_propensity)
    except ValueError:
        metrics['propensity_auroc'] = np.nan

    # Outcome metrics (MSE against ground truth probabilities)
    metrics['y0_mse'] = mean_squared_error(true_y0, pred_y0)
    metrics['y1_mse'] = mean_squared_error(true_y1, pred_y1)

    # Outcome AUROC metrics (on factual outcomes only)
    untreated_mask = true_treatment == 0
    if untreated_mask.sum() > 0:
        try:
            metrics['y0_auroc'] = roc_auc_score(
                true_outcome[untreated_mask],
                pred_y0[untreated_mask]
            )
        except ValueError:
            metrics['y0_auroc'] = np.nan
    else:
        metrics['y0_auroc'] = np.nan

    treated_mask = true_treatment == 1
    if treated_mask.sum() > 0:
        try:
            metrics['y1_auroc'] = roc_auc_score(
                true_outcome[treated_mask],
                pred_y1[treated_mask]
            )
        except ValueError:
            metrics['y1_auroc'] = np.nan
    else:
        metrics['y1_auroc'] = np.nan

    # Confidence interval metrics (for causal forest)
    if tau_lower is not None and tau_upper is not None:
        coverage = np.mean((true_ite >= tau_lower) & (true_ite <= tau_upper))
        metrics['ci_coverage'] = coverage
        significant = (tau_lower > 0) | (tau_upper < 0)
        metrics['pct_significant'] = np.mean(significant)
        metrics['mean_ci_width'] = np.mean(tau_upper - tau_lower)

    return metrics


def create_model(
    extractor_type: str,
    model_type: str,
    device: torch.device,
    args: argparse.Namespace
) -> Tuple[Any, bool]:
    """Create a model based on extractor type and model type.

    Returns:
        Tuple of (model, requires_tokenizer_fit)
    """
    requires_fit_tokenizer = extractor_type in ["cnn", "gru", "gru_pool", "gru_transformer_mil"]
    # Confounder with GRU mode also requires tokenizer - but we use sentence-level by default

    common_kwargs = {
        "device": str(device),
        "causal_head_representation_dim": 128,
        "causal_head_hidden_outcome_dim": 64,
        "causal_head_dropout": 0.2,
    }

    if model_type == "causal_forest":
        # Use CausalTextForest for causal_forest model type
        model = _create_causal_forest_model(extractor_type, device, args)
        return model, requires_fit_tokenizer

    # For other model types, use CausalText
    if extractor_type == "cnn":
        model = CausalText(
            feature_extractor_type="cnn",
            model_type=model_type,
            embedding_dim=128,
            kernel_sizes=[3, 4, 5, 7],
            num_kmeans_filters=64,
            num_random_filters=0,
            cnn_dropout=0.1,
            max_length=2048,
            min_word_freq=2,
            max_vocab_size=50000,
            projection_dim=128,
            **common_kwargs
        )
    elif extractor_type == "bert":
        model = CausalText(
            feature_extractor_type="bert",
            model_type=model_type,
            bert_model_name="prajjwal1/bert-tiny",  # Fast for experiments
            bert_max_length=512,
            bert_projection_dim=128,
            bert_dropout=0.1,
            bert_freeze_encoder=False,
            **common_kwargs
        )
    elif extractor_type == "gru":
        model = CausalText(
            feature_extractor_type="gru",
            model_type=model_type,
            embedding_dim=128,
            gru_hidden_dim=128,
            gru_num_layers=2,
            gru_dropout=0.1,
            gru_bidirectional=True,
            gru_projection_dim=128,
            max_length=2048,
            min_word_freq=2,
            max_vocab_size=50000,
            **common_kwargs
        )
    elif extractor_type == "confounder":
        model = CausalText(
            feature_extractor_type="confounder",
            model_type=model_type,
            confounder_num_latents=4,
            confounder_value_dim=128,
            confounder_sentence_model="all-MiniLM-L6-v2",
            confounder_freeze_encoder=True,
            confounder_max_sentences=100,
            confounder_num_heads=4,
            confounder_num_iterations=2,
            confounder_sparse_attention=True,
            confounder_sparse_method="entmax",
            confounder_sparse_alpha=1.5,
            confounder_dropout=0.1,
            confounder_use_gru=False,  # Use sentence-level (no tokenizer needed)
            **common_kwargs
        )
        requires_fit_tokenizer = False  # Sentence-level mode
    elif extractor_type == "hierarchical_transformer":
        model = CausalText(
            feature_extractor_type="hierarchical_transformer",
            model_type=model_type,
            hier_transformer_sentence_model="prajjwal1/bert-tiny",
            hier_transformer_freeze_sentence_encoder=True,
            hier_transformer_max_chunks=100,
            hier_transformer_chunk_size=128,
            hier_transformer_chunk_overlap=32,
            hier_transformer_num_layers=2,
            hier_transformer_num_heads=4,
            hier_transformer_dim=256,
            hier_transformer_dropout=0.1,
            hier_transformer_projection_dim=128,
            **common_kwargs
        )
        requires_fit_tokenizer = False
    elif extractor_type == "gated_mil_hierarchical":
        model = CausalText(
            feature_extractor_type="gated_mil_hierarchical",
            model_type=model_type,
            gated_mil_sentence_model="prajjwal1/bert-tiny",
            gated_mil_freeze_sentence_encoder=True,
            gated_mil_max_chunks=100,
            gated_mil_chunk_size=128,
            gated_mil_chunk_overlap=32,
            gated_mil_hidden_dim=128,
            gated_mil_num_confounders=4,
            gated_mil_dropout=0.1,
            gated_mil_projection_dim=128,
            **common_kwargs
        )
        requires_fit_tokenizer = False
    elif extractor_type == "gru_transformer_mil":
        model = CausalText(
            feature_extractor_type="gru_transformer_mil",
            model_type=model_type,
            gru_mil_embedding_dim=128,
            gru_mil_gru_hidden_dim=128,
            gru_mil_gru_num_layers=1,
            gru_mil_gru_bidirectional=True,
            gru_mil_gru_dropout=0.1,
            gru_mil_max_chunks=100,
            gru_mil_chunk_size=128,
            gru_mil_chunk_overlap=32,
            gru_mil_transformer_layers=2,
            gru_mil_transformer_heads=4,
            gru_mil_transformer_dim=256,
            gru_mil_num_confounders=4,
            gru_mil_mil_hidden_dim=128,
            gru_mil_projection_dim=128,
            gru_mil_max_vocab=50000,
            gru_mil_min_word_freq=2,
            **common_kwargs
        )
    elif extractor_type == "gru_pool":
        model = CausalText(
            feature_extractor_type="gru_pool",
            model_type=model_type,
            gru_pool_embedding_dim=128,
            gru_pool_gru_hidden_dim=128,
            gru_pool_gru_num_layers=1,
            gru_pool_gru_bidirectional=True,
            gru_pool_gru_dropout=0.1,
            gru_pool_max_chunks=100,
            gru_pool_chunk_size=128,
            gru_pool_chunk_overlap=32,
            gru_pool_transformer_layers=2,
            gru_pool_transformer_heads=4,
            gru_pool_transformer_dim=256,
            gru_pool_gated_attention_dim=128,
            gru_pool_projection_dim=128,
            gru_pool_max_vocab=50000,
            gru_pool_min_word_freq=2,
            **common_kwargs
        )
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")

    return model, requires_fit_tokenizer


def _create_causal_forest_model(
    extractor_type: str,
    device: torch.device,
    args: argparse.Namespace
) -> CausalTextForest:
    """Create a CausalTextForest model with the specified extractor."""

    common_kwargs = {
        "device": str(device),
        "representation_dim": 128,
        "hidden_dim": 64,
        "dropout": 0.2,
        "cf_n_estimators": args.cf_n_estimators,
        "cf_min_samples_leaf": args.cf_min_samples_leaf,
        "cf_honest": True,
        "cf_inference": True,
        "cf_use_rlearner_representation": False,
    }

    if extractor_type == "cnn":
        return CausalTextForest(
            feature_extractor_type="cnn",
            embedding_dim=128,
            kernel_sizes=[3, 4, 5, 7],
            num_kmeans_filters=64,
            num_random_filters=0,
            cnn_dropout=0.1,
            max_length=2048,
            min_word_freq=2,
            max_vocab_size=50000,
            projection_dim=128,
            **common_kwargs
        )
    elif extractor_type == "bert":
        return CausalTextForest(
            feature_extractor_type="bert",
            bert_model_name="prajjwal1/bert-tiny",
            bert_max_length=512,
            bert_projection_dim=128,
            bert_dropout=0.1,
            bert_freeze_encoder=False,
            **common_kwargs
        )
    elif extractor_type == "gru":
        return CausalTextForest(
            feature_extractor_type="gru",
            embedding_dim=128,
            gru_hidden_dim=128,
            gru_num_layers=2,
            gru_dropout=0.1,
            gru_bidirectional=True,
            gru_projection_dim=128,
            max_length=2048,
            min_word_freq=2,
            max_vocab_size=50000,
            **common_kwargs
        )
    elif extractor_type == "confounder":
        return CausalTextForest(
            feature_extractor_type="confounder",
            confounder_num_latents=4,
            confounder_value_dim=128,
            confounder_sentence_model="all-MiniLM-L6-v2",
            confounder_freeze_encoder=True,
            confounder_max_sentences=100,
            confounder_num_heads=4,
            confounder_num_iterations=2,
            confounder_sparse_attention=True,
            confounder_sparse_method="entmax",
            confounder_sparse_alpha=1.5,
            confounder_dropout=0.1,
            confounder_use_gru=False,
            **common_kwargs
        )
    elif extractor_type == "hierarchical_transformer":
        return CausalTextForest(
            feature_extractor_type="hierarchical_transformer",
            hier_transformer_sentence_model="prajjwal1/bert-tiny",
            hier_transformer_freeze_sentence_encoder=True,
            hier_transformer_max_chunks=100,
            hier_transformer_chunk_size=128,
            hier_transformer_chunk_overlap=32,
            hier_transformer_num_layers=2,
            hier_transformer_num_heads=4,
            hier_transformer_dim=256,
            hier_transformer_dropout=0.1,
            hier_transformer_projection_dim=128,
            **common_kwargs
        )
    elif extractor_type == "gated_mil_hierarchical":
        return CausalTextForest(
            feature_extractor_type="gated_mil_hierarchical",
            gated_mil_sentence_model="prajjwal1/bert-tiny",
            gated_mil_freeze_sentence_encoder=True,
            gated_mil_max_chunks=100,
            gated_mil_chunk_size=128,
            gated_mil_chunk_overlap=32,
            gated_mil_hidden_dim=128,
            gated_mil_num_confounders=4,
            gated_mil_dropout=0.1,
            gated_mil_projection_dim=128,
            **common_kwargs
        )
    elif extractor_type == "gru_transformer_mil":
        return CausalTextForest(
            feature_extractor_type="gru_transformer_mil",
            gru_mil_embedding_dim=128,
            gru_mil_gru_hidden_dim=128,
            gru_mil_gru_num_layers=1,
            gru_mil_gru_bidirectional=True,
            gru_mil_gru_dropout=0.1,
            gru_mil_max_chunks=100,
            gru_mil_chunk_size=128,
            gru_mil_chunk_overlap=32,
            gru_mil_transformer_layers=2,
            gru_mil_transformer_heads=4,
            gru_mil_transformer_dim=256,
            gru_mil_num_confounders=4,
            gru_mil_mil_hidden_dim=128,
            gru_mil_projection_dim=128,
            gru_mil_max_vocab=50000,
            gru_mil_min_word_freq=2,
            **common_kwargs
        )
    elif extractor_type == "gru_pool":
        return CausalTextForest(
            feature_extractor_type="gru_pool",
            gru_pool_embedding_dim=128,
            gru_pool_gru_hidden_dim=128,
            gru_pool_gru_num_layers=1,
            gru_pool_gru_bidirectional=True,
            gru_pool_gru_dropout=0.1,
            gru_pool_max_chunks=100,
            gru_pool_chunk_size=128,
            gru_pool_chunk_overlap=32,
            gru_pool_transformer_layers=2,
            gru_pool_transformer_heads=4,
            gru_pool_transformer_dim=256,
            gru_pool_gated_attention_dim=128,
            gru_pool_projection_dim=128,
            gru_pool_max_vocab=50000,
            gru_pool_min_word_freq=2,
            **common_kwargs
        )
    else:
        raise ValueError(f"Unknown extractor type for causal forest: {extractor_type}")


def train_causal_text_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model: CausalText,
    text_column: str,
    model_type: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gamma_rlearner: float = 1.0,
    beta_targreg: float = 0.1,
) -> List[Dict]:
    """Train a CausalText model for one fold."""

    train_dataset = TextDataset(
        texts=train_df[text_column].tolist(),
        treatments=train_df['treatment_indicator'].values,
        outcomes=train_df['outcome_indicator'].values
    )
    val_dataset = TextDataset(
        texts=val_df[text_column].tolist(),
        treatments=val_df['treatment_indicator'].values,
        outcomes=val_df['outcome_indicator'].values
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_text_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_text_batch
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch['treatment'] = batch['treatment'].to(device)
            batch['outcome'] = batch['outcome'].to(device)

            optimizer.zero_grad()

            if model_type == "rlearner":
                losses = model.train_step(
                    batch,
                    alpha_propensity=1.0,
                    gamma_rlearner=gamma_rlearner,
                )
            else:
                losses = model.train_step(
                    batch,
                    alpha_propensity=1.0,
                    beta_targreg=beta_targreg,
                )

            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += losses['loss'].item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)

                if model_type == "rlearner":
                    losses = model.train_step(
                        batch,
                        alpha_propensity=1.0,
                        gamma_rlearner=gamma_rlearner,
                    )
                else:
                    losses = model.train_step(
                        batch,
                        alpha_propensity=1.0,
                        beta_targreg=beta_targreg,
                    )
                val_loss += losses['loss'].item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0]
        })

        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    return history


def train_causal_forest_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model: CausalTextForest,
    text_column: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> List[Dict]:
    """Train a CausalTextForest model for one fold."""

    train_dataset = TextDataset(
        texts=train_df[text_column].tolist(),
        treatments=train_df['treatment_indicator'].values,
        outcomes=train_df['outcome_indicator'].values
    )
    val_dataset = TextDataset(
        texts=val_df[text_column].tolist(),
        treatments=val_df['treatment_indicator'].values,
        outcomes=val_df['outcome_indicator'].values
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_text_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_text_batch
    )

    # Stage 1: Train representation
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch['treatment'] = batch['treatment'].to(device)
            batch['outcome'] = batch['outcome'].to(device)

            optimizer.zero_grad()
            losses = model.train_representation_step(batch, alpha_propensity=1.0)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += losses['loss'].item()

        scheduler.step()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)
                losses = model.train_representation_step(batch, alpha_propensity=1.0)
                val_loss += losses['loss'].item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0]
        })

        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    # Stage 2: Train causal forest on combined data
    logger.info("    Training causal forest on extracted features...")
    combined_df = pd.concat([train_df, val_df])
    combined_texts = combined_df[text_column].tolist()
    combined_T = combined_df['treatment_indicator'].values
    combined_Y = combined_df['outcome_indicator'].values

    model.train_causal_forest(combined_texts, combined_T, combined_Y, batch_size=batch_size)

    return history


def predict_causal_text(
    model: CausalText,
    df: pd.DataFrame,
    text_column: str,
    device: torch.device,
    batch_size: int
) -> Dict[str, np.ndarray]:
    """Generate predictions from CausalText model."""
    model.eval()
    texts = df[text_column].tolist()

    all_y0 = []
    all_y1 = []
    all_prop = []
    all_tau = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            preds = model.predict(batch_texts)
            all_y0.append(preds['y0_prob'].cpu().numpy())
            all_y1.append(preds['y1_prob'].cpu().numpy())
            all_prop.append(preds['propensity'].cpu().numpy())
            all_tau.append(preds['tau_pred'].cpu().numpy())

    return {
        'y0_prob': np.concatenate(all_y0),
        'y1_prob': np.concatenate(all_y1),
        'propensity': np.concatenate(all_prop),
        'tau_pred': np.concatenate(all_tau),
        'ite_prob': np.concatenate(all_y1) - np.concatenate(all_y0)
    }


def predict_causal_forest(
    model: CausalTextForest,
    df: pd.DataFrame,
    text_column: str,
    batch_size: int
) -> Dict[str, np.ndarray]:
    """Generate predictions from CausalTextForest model."""
    texts = df[text_column].tolist()
    preds = model.predict(texts, batch_size=batch_size, return_ci=True)

    return {
        'y0_prob': preds['pred_y0_prob'],
        'y1_prob': preds['pred_y1_prob'],
        'propensity': preds['pred_propensity_prob'],
        'tau_pred': preds['tau_pred'],
        'ite_prob': preds['pred_ite_prob'],
        'tau_lower': preds.get('tau_lower'),
        'tau_upper': preds.get('tau_upper')
    }


def run_single_condition(
    df: pd.DataFrame,
    extractor_type: str,
    model_type: str,
    text_column: str,
    condition_name: str,
    device: torch.device,
    n_folds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    args: argparse.Namespace
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run cross-validation for one experimental condition."""

    logger.info(f"  Running condition: {condition_name}")
    logger.info(f"    Text column: {text_column}")

    if text_column not in df.columns:
        logger.warning(f"    Skipping: column '{text_column}' not found")
        return pd.DataFrame(), {}

    df = df.reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        logger.info(f"    Fold {fold + 1}/{n_folds}")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Create fresh model for each fold
        model, requires_fit_tokenizer = create_model(extractor_type, model_type, device, args)

        # Fit tokenizer if needed
        if requires_fit_tokenizer:
            train_texts = train_df[text_column].tolist()
            model.fit_tokenizer(train_texts)

        # Train and predict based on model type
        if model_type == "causal_forest":
            history = train_causal_forest_model(
                train_df, test_df, model, text_column,
                device, epochs, batch_size, learning_rate
            )
            preds = predict_causal_forest(model, test_df, text_column, batch_size)
        else:
            history = train_causal_text_model(
                train_df, test_df, model, text_column, model_type,
                device, epochs, batch_size, learning_rate
            )
            preds = predict_causal_text(model, test_df, text_column, device, batch_size)

        # Store predictions
        fold_preds = test_df.copy()
        fold_preds['pred_y0_prob'] = preds['y0_prob']
        fold_preds['pred_y1_prob'] = preds['y1_prob']
        fold_preds['pred_ite_prob'] = preds['ite_prob']
        fold_preds['pred_propensity'] = preds['propensity']
        fold_preds['pred_tau'] = preds['tau_pred']
        fold_preds['cv_fold'] = fold + 1

        if preds.get('tau_lower') is not None:
            fold_preds['pred_tau_lower'] = preds['tau_lower']
            fold_preds['pred_tau_upper'] = preds['tau_upper']

        all_predictions.append(fold_preds)

        # Cleanup
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    if not all_predictions:
        return pd.DataFrame(), {}

    results_df = pd.concat(all_predictions).sort_index()

    # Compute metrics
    metrics = compute_metrics(
        pred_ite=results_df['pred_ite_prob'].values,
        true_ite=results_df['true_ite_prob'].values,
        pred_propensity=results_df['pred_propensity'].values,
        true_treatment=results_df['treatment_indicator'].values,
        pred_y0=results_df['pred_y0_prob'].values,
        pred_y1=results_df['pred_y1_prob'].values,
        true_y0=results_df['true_y0_prob'].values,
        true_y1=results_df['true_y1_prob'].values,
        true_outcome=results_df['outcome_indicator'].values,
        tau_lower=results_df.get('pred_tau_lower', pd.Series([None])).values if 'pred_tau_lower' in results_df.columns else None,
        tau_upper=results_df.get('pred_tau_upper', pd.Series([None])).values if 'pred_tau_upper' in results_df.columns else None
    )

    logger.info(f"    Results: ITE Corr={metrics['ite_corr']:.4f}, ITE Rank Corr={metrics['ite_spearman_corr']:.4f}, ATE Bias={metrics['ate_bias']:.4f}")

    return results_df, metrics


def run_experiment(
    df: pd.DataFrame,
    experiment: Dict[str, str],
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Run a single experiment (extractor + model_type) across all conditions."""

    exp_name = experiment['name']
    extractor_type = experiment['extractor']
    model_type = experiment['model_type']

    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT: {exp_name}")
    logger.info(f"  Extractor: {extractor_type}")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"{'='*70}")

    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for condition in CONDITIONS:
        cond_name = condition['name']
        text_column = condition['text_column']

        try:
            results_df, metrics = run_single_condition(
                df=df,
                extractor_type=extractor_type,
                model_type=model_type,
                text_column=text_column,
                condition_name=cond_name,
                device=device,
                n_folds=args.n_folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                args=args
            )

            if metrics:
                all_metrics[cond_name] = metrics

                # Save condition results
                cond_dir = exp_dir / cond_name
                cond_dir.mkdir(exist_ok=True)
                results_df.to_parquet(cond_dir / "predictions.parquet", index=False)

        except Exception as e:
            logger.error(f"  Error in condition {cond_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save experiment metrics summary
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df.index.name = 'condition'
        metrics_df.to_csv(exp_dir / "metrics_summary.csv")

        logger.info(f"\n  Experiment {exp_name} summary:")
        logger.info(f"\n{metrics_df[['ite_corr', 'ite_spearman_corr', 'ate_bias', 'propensity_auroc']].to_string()}")

    # Save experiment config
    config_info = {
        'experiment_name': exp_name,
        'extractor_type': extractor_type,
        'model_type': model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'n_folds': args.n_folds,
        'device': str(device),
        'conditions_run': list(all_metrics.keys())
    }
    with open(exp_dir / "experiment_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run remaining extractor + model combinations"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=False,
        help="Path to dataset parquet file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./remaining_experiment_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, mps, cpu, etc.)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--cf-n-estimators",
        type=int,
        default=200,
        help="Number of trees in causal forest (must be divisible by 4)"
    )
    parser.add_argument(
        "--cf-min-samples-leaf",
        type=int,
        default=5,
        help="Minimum samples per leaf in causal forest"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to run (e.g., bert_dragonnet gru_rlearner)"
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all available experiments and exit"
    )

    args = parser.parse_args()

    # List experiments mode
    if args.list_experiments:
        print("\nAvailable experiments (26 total):\n")
        print(f"{'Name':<35} {'Extractor':<25} {'Model Type':<20}")
        print("-" * 80)
        for exp in EXPERIMENTS:
            print(f"{exp['name']:<35} {exp['extractor']:<25} {exp['model_type']:<20}")
        print("\nExample multi-GPU usage:")
        print("  python run_remaining_experiments.py --device cuda:0 --experiments bert_dragonnet gru_dragonnet &")
        print("  python run_remaining_experiments.py --device cuda:1 --experiments bert_causal_forest gru_rlearner &")
        return

    # Require dataset for actual runs
    if not args.dataset:
        parser.error("--dataset is required (use --list-experiments to see available experiments)")

    # Ensure cf_n_estimators is divisible by 4
    if args.cf_n_estimators % 4 != 0:
        args.cf_n_estimators = (args.cf_n_estimators // 4 + 1) * 4
        logger.warning(f"Adjusted cf_n_estimators to {args.cf_n_estimators} (must be divisible by 4)")

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load dataset
    df = pd.read_parquet(args.dataset)
    logger.info(f"Loaded {len(df)} samples from {args.dataset}")

    # Check required columns
    required_cols = ['treatment_indicator', 'outcome_indicator', 'true_ite_prob', 'true_y0_prob', 'true_y1_prob']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check text columns
    for cond in CONDITIONS:
        col = cond['text_column']
        if col in df.columns:
            logger.info(f"  Found text column: {col}")
        else:
            logger.warning(f"  Text column not found: {col}")

    # Filter experiments if specified
    experiments_to_run = EXPERIMENTS
    if args.experiments:
        experiments_to_run = [e for e in EXPERIMENTS if e['name'] in args.experiments]
        if not experiments_to_run:
            logger.error(f"No matching experiments found. Available: {[e['name'] for e in EXPERIMENTS]}")
            return
        logger.info(f"Running {len(experiments_to_run)} selected experiments")
    else:
        logger.info(f"Running all {len(experiments_to_run)} experiments")

    # Run experiments
    all_results = {}

    for experiment in experiments_to_run:
        try:
            metrics = run_experiment(df, experiment, device, output_dir, args)
            all_results[experiment['name']] = metrics
        except Exception as e:
            logger.error(f"Error in experiment {experiment['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save overall summary
    if all_results:
        # Flatten results for summary
        summary_rows = []
        for exp_name, conditions in all_results.items():
            for cond_name, metrics in conditions.items():
                row = {'experiment': exp_name, 'condition': cond_name}
                row.update(metrics)
                summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(output_dir / "overall_summary.csv", index=False)

            logger.info("\n" + "=" * 80)
            logger.info("OVERALL RESULTS SUMMARY")
            logger.info("=" * 80)
            logger.info(f"\n{summary_df[['experiment', 'condition', 'ite_corr', 'ite_spearman_corr', 'ate_bias']].to_string()}")

    # Save run config
    run_config = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'n_folds': args.n_folds,
        'cf_n_estimators': args.cf_n_estimators,
        'cf_min_samples_leaf': args.cf_min_samples_leaf,
        'device': str(device),
        'experiments_run': [e['name'] for e in experiments_to_run if e['name'] in all_results]
    }
    with open(output_dir / "run_config.json", 'w') as f:
        json.dump(run_config, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
