#!/usr/bin/env python
"""Hierarchical Confounder Extractor R-Learner experiment for synthetic clinical text.

This script runs experimental conditions using the HierarchicalConfounderExtractor
as the feature extractor, testing whether sparse hierarchical attention improves
ITE estimation from clinical text compared to BERT or CNN approaches.

Key features:
- Uses HierarchicalConfounderExtractor with token-level attention
- 1 explicit confounder ("number of metastatic sites") for guided attention
- 8 latent confounders for discovering other relevant signals
- Sparse attention (entmax) to focus on relevant sentences
- R-Learner objective for direct treatment effect optimization

Conditions tested:
1. Oracle: patient_prompt (explicit confounder, upper bound)
2. Baseline frozen: clinical_text with frozen token encoder
3. Fine-tuned: clinical_text with fine-tuned token encoder
4. LLM-extract-only: Extracted categorical features (MLP)
5. LLM-extract-combined: Text + extracted features (hybrid)
6. LLM-extract-as-text: Structured text from extraction

Usage:
    # Single GPU:
    python oracle_experiment_scripts/run_concept_aware_experiment_rlearner_confounders.py \
        --dataset path/to/dataset.parquet \
        --output-dir results/confounder_rlearner \
        --device cuda:0

    # Parallel across two GPUs (run two processes):
    python run_concept_aware_experiment_rlearner_confounders.py --device cuda:0 --conditions 1_oracle 2_baseline_frozen 3_fine_tuned &
    python run_concept_aware_experiment_rlearner_confounders.py --device cuda:1 --conditions 4_llm_extract_only 5_llm_extract_combined 6_llm_extract_as_text &
"""

import argparse
import gc
import json
import logging
from dataclasses import dataclass
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

from cdt.models import CausalText
from mlp_dragonnet import MLPDragonNet, CategoricalEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


# Default token encoder for hierarchical mode
DEFAULT_TOKEN_ENCODER = "distilbert-base-uncased"

# Explicit confounder text - guides attention to metastatic site mentions
EXPLICIT_CONFOUNDER_TEXTS = [
    "number of metastatic sites"
]

# Number of latent confounders to learn
NUM_LATENT_CONFOUNDERS = 8


@dataclass
class ExperimentConfig:
    """Configuration for an experimental condition."""
    name: str
    text_column: str
    freeze_encoder: bool = False
    use_llm_extraction: bool = False
    llm_extraction_only: bool = False
    llm_combined: bool = False
    llm_as_text: bool = False


# Define experimental conditions
EXPERIMENT_CONDITIONS = [
    ExperimentConfig(
        name="1_oracle",
        text_column="patient_prompt",
        freeze_encoder=False
    ),
    ExperimentConfig(
        name="2_baseline_frozen",
        text_column="clinical_text",
        freeze_encoder=True  # Frozen encoder = baseline
    ),
    ExperimentConfig(
        name="3_fine_tuned",
        text_column="clinical_text",
        freeze_encoder=False  # Fine-tuned = main condition
    ),
    ExperimentConfig(
        name="4_llm_extract_only",
        text_column="clinical_text",  # Not used
        use_llm_extraction=True,
        llm_extraction_only=True
    ),
    ExperimentConfig(
        name="5_llm_extract_combined",
        text_column="clinical_text",
        freeze_encoder=False,
        use_llm_extraction=True,
        llm_combined=True
    ),
    ExperimentConfig(
        name="6_llm_extract_as_text",
        text_column="llm_structured_text",
        freeze_encoder=False,
        llm_as_text=True
    ),
]


class TextDataset(Dataset):
    """Simple dataset for text + labels."""

    def __init__(
        self,
        texts: List[str],
        treatments: np.ndarray,
        outcomes: np.ndarray,
        auxiliary_features: Optional[torch.Tensor] = None
    ):
        self.texts = texts
        self.treatments = torch.tensor(treatments, dtype=torch.float32)
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32)
        self.auxiliary_features = auxiliary_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            'texts': self.texts[idx],
            'treatment': self.treatments[idx],
            'outcome': self.outcomes[idx]
        }
        if self.auxiliary_features is not None:
            item['auxiliary_features'] = self.auxiliary_features[idx]
        return item


class CategoricalDataset(Dataset):
    """Dataset for categorical features only."""

    def __init__(
        self,
        features: torch.Tensor,
        treatments: np.ndarray,
        outcomes: np.ndarray
    ):
        self.features = features
        self.treatments = torch.tensor(treatments, dtype=torch.float32)
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'treatment': self.treatments[idx],
            'outcome': self.outcomes[idx]
        }


def collate_text_batch(batch):
    """Collate function for text batches."""
    texts = [b['texts'] for b in batch]
    treatments = torch.stack([b['treatment'] for b in batch])
    outcomes = torch.stack([b['outcome'] for b in batch])

    result = {
        'texts': texts,
        'treatment': treatments,
        'outcome': outcomes
    }

    if 'auxiliary_features' in batch[0]:
        aux = torch.stack([b['auxiliary_features'] for b in batch])
        result['auxiliary_features'] = aux

    return result


def collate_categorical_batch(batch):
    """Collate function for categorical batches."""
    features = torch.stack([b['features'] for b in batch])
    treatments = torch.stack([b['treatment'] for b in batch])
    outcomes = torch.stack([b['outcome'] for b in batch])

    return {
        'features': features,
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
    true_outcome: np.ndarray
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
    # y0_auroc: evaluate on untreated (T=0), where outcome_indicator = Y(0)
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

    # y1_auroc: evaluate on treated (T=1), where outcome_indicator = Y(1)
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

    return metrics


def train_confounder_rlearner_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gamma_rlearner: float = 1.0,
    token_encoder: str = DEFAULT_TOKEN_ENCODER,
    num_latent_confounders: int = NUM_LATENT_CONFOUNDERS,
    explicit_confounder_texts: List[str] = EXPLICIT_CONFOUNDER_TEXTS,
    encoder: Optional[CategoricalEncoder] = None,
    gradient_accumulation_steps: int = 1
) -> Tuple[CausalText, List[Dict]]:
    """Train a Hierarchical Confounder Extractor R-Learner model for one fold."""
    text_column = config.text_column

    # Prepare auxiliary features if needed
    auxiliary_dim = 0
    train_aux = None
    val_aux = None

    if config.llm_combined and encoder is not None:
        auxiliary_dim = encoder.num_categories
        train_aux = encoder.transform(
            train_df['llm_extracted_metastatic_sites'].tolist(),
            device=device
        )
        val_aux = encoder.transform(
            val_df['llm_extracted_metastatic_sites'].tolist(),
            device=device
        )

    # Create model with Hierarchical Confounder Extractor and R-Learner architecture
    model = CausalText(
        feature_extractor_type="confounder",
        model_type="rlearner",
        # Hierarchical confounder extractor settings
        confounder_hierarchical=True,
        confounder_token_encoder=token_encoder,
        confounder_freeze_token_encoder=config.freeze_encoder,
        confounder_max_sentence_tokens=128,
        # Confounder architecture
        confounder_num_latents=num_latent_confounders,
        confounder_explicit_texts=explicit_confounder_texts,
        confounder_value_dim=128,
        confounder_max_sentences=100,
        confounder_num_heads=4,
        # Sparse attention settings
        confounder_sparse_attention=True,
        confounder_sparse_method="entmax",
        confounder_sparse_alpha=1.5,
        confounder_dropout=0.1,
        # Causal head parameters
        causal_head_representation_dim=128,
        causal_head_hidden_outcome_dim=64,
        causal_head_dropout=0.2,
        device=str(device),
        auxiliary_dim=auxiliary_dim
    )

    # No fit_tokenizer needed for confounder extractor

    # Create datasets
    train_texts = train_df[text_column].tolist()
    train_dataset = TextDataset(
        texts=train_texts,
        treatments=train_df['treatment_indicator'].values,
        outcomes=train_df['outcome_indicator'].values,
        auxiliary_features=train_aux
    )

    val_dataset = TextDataset(
        texts=val_df[text_column].tolist(),
        treatments=val_df['treatment_indicator'].values,
        outcomes=val_df['outcome_indicator'].values,
        auxiliary_features=val_aux
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

    # Training with AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Linear warmup then decay
    num_training_steps = len(train_loader) * epochs // gradient_accumulation_steps
    num_warmup_steps = num_training_steps // 10

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_state = None
    history = []
    global_step = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_r_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            batch['treatment'] = batch['treatment'].to(device)
            batch['outcome'] = batch['outcome'].to(device)

            losses = model.train_step(
                batch,
                alpha_propensity=1.0,
                gamma_rlearner=gamma_rlearner
            )

            # Scale loss for gradient accumulation
            loss = losses['loss'] / gradient_accumulation_steps
            loss.backward()

            train_loss += losses['loss'].item()
            train_r_loss += losses['r_loss'].item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Validate
        model.eval()
        val_loss = 0.0
        val_r_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)
                losses = model.train_step(
                    batch,
                    alpha_propensity=1.0,
                    gamma_rlearner=gamma_rlearner
                )
                val_loss += losses['loss'].item()
                val_r_loss += losses['r_loss'].item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_r_loss /= len(train_loader)
        val_r_loss /= len(val_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_r_loss': train_r_loss,
            'val_r_loss': val_r_loss,
            'lr': scheduler.get_last_lr()[0]
        })

        logger.info(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, r_loss={val_r_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    return model, history


def train_mlp_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    encoder: CategoricalEncoder,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float
) -> Tuple[MLPDragonNet, List[Dict]]:
    """Train MLP model for LLM-extract-only condition."""
    # Encode categorical features
    train_features = encoder.transform(
        train_df['llm_extracted_metastatic_sites'].tolist(),
        device=device
    )
    val_features = encoder.transform(
        val_df['llm_extracted_metastatic_sites'].tolist(),
        device=device
    )

    model = MLPDragonNet(
        input_dim=encoder.num_categories,
        hidden_dims=[32, 32],
        causal_head_representation_dim=32,
        causal_head_hidden_outcome_dim=16,
        causal_head_dropout=0.2,
        device=str(device)
    )

    train_dataset = CategoricalDataset(
        features=train_features,
        treatments=train_df['treatment_indicator'].values,
        outcomes=train_df['outcome_indicator'].values
    )

    val_dataset = CategoricalDataset(
        features=val_features,
        treatments=val_df['treatment_indicator'].values,
        outcomes=val_df['outcome_indicator'].values
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_categorical_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_categorical_batch
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_val_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch['features'] = batch['features'].to(device)
            batch['treatment'] = batch['treatment'].to(device)
            batch['outcome'] = batch['outcome'].to(device)

            optimizer.zero_grad()
            losses = model.train_step(batch, alpha_propensity=1.0, beta_targreg=0.1)
            losses['loss'].backward()
            optimizer.step()
            train_loss += losses['loss'].item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch['features'] = batch['features'].to(device)
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)
                losses = model.train_step(batch, alpha_propensity=1.0, beta_targreg=0.1)
                val_loss += losses['loss'].item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)

    return model, history


def predict_confounder_rlearner(
    model: CausalText,
    df: pd.DataFrame,
    text_column: str,
    device: torch.device,
    batch_size: int,
    encoder: Optional[CategoricalEncoder] = None,
    use_auxiliary: bool = False
) -> Dict[str, np.ndarray]:
    """Generate predictions from Confounder R-Learner model."""
    model.eval()

    texts = df[text_column].tolist()
    all_y0 = []
    all_y1 = []
    all_prop = []
    all_tau = []

    aux_features = None
    if use_auxiliary and encoder is not None:
        aux_features = encoder.transform(
            df['llm_extracted_metastatic_sites'].tolist(),
            device=device
        )

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_aux = None
            if aux_features is not None:
                batch_aux = aux_features[i:i + batch_size]

            preds = model.predict(batch_texts, batch_aux)
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


def predict_mlp(
    model: MLPDragonNet,
    df: pd.DataFrame,
    encoder: CategoricalEncoder,
    device: torch.device,
    batch_size: int
) -> Dict[str, np.ndarray]:
    """Generate predictions from MLP model."""
    model.eval()

    features = encoder.transform(
        df['llm_extracted_metastatic_sites'].tolist(),
        device=device
    )

    all_y0 = []
    all_y1 = []
    all_prop = []

    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]
            preds = model.predict(batch_features)
            all_y0.append(preds['y0_prob'].cpu().numpy())
            all_y1.append(preds['y1_prob'].cpu().numpy())
            all_prop.append(preds['propensity'].cpu().numpy())

    return {
        'y0_prob': np.concatenate(all_y0),
        'y1_prob': np.concatenate(all_y1),
        'propensity': np.concatenate(all_prop),
        'ite_prob': np.concatenate(all_y1) - np.concatenate(all_y0)
    }


def run_condition(
    df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    n_folds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gamma_rlearner: float = 1.0,
    token_encoder: str = DEFAULT_TOKEN_ENCODER,
    num_latent_confounders: int = NUM_LATENT_CONFOUNDERS,
    explicit_confounder_texts: List[str] = EXPLICIT_CONFOUNDER_TEXTS,
    encoder: Optional[CategoricalEncoder] = None,
    gradient_accumulation_steps: int = 1
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run cross-validation for one experimental condition."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running condition: {config.name} (Hierarchical Confounder R-Learner)")
    logger.info(f"{'='*60}")

    df = df.reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        logger.info(f"  Fold {fold + 1}/{n_folds}")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        if config.llm_extraction_only:
            # MLP on categorical features only
            model, _ = train_mlp_model(
                train_df, test_df, encoder, device,
                epochs, batch_size * 4, learning_rate * 10  # MLP can use larger batch/LR
            )
            preds = predict_mlp(model, test_df, encoder, device, batch_size * 4)
        else:
            # Hierarchical Confounder Extractor R-Learner
            model, _ = train_confounder_rlearner_model(
                train_df, test_df, config, device,
                epochs, batch_size, learning_rate,
                gamma_rlearner=gamma_rlearner,
                token_encoder=token_encoder,
                num_latent_confounders=num_latent_confounders,
                explicit_confounder_texts=explicit_confounder_texts,
                encoder=encoder if config.llm_combined else None,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            preds = predict_confounder_rlearner(
                model, test_df, config.text_column, device, batch_size,
                encoder=encoder if config.llm_combined else None,
                use_auxiliary=config.llm_combined
            )

        # Store predictions
        fold_preds = test_df.copy()
        fold_preds['pred_y0_prob'] = preds['y0_prob']
        fold_preds['pred_y1_prob'] = preds['y1_prob']
        fold_preds['pred_ite_prob'] = preds['ite_prob']
        fold_preds['pred_propensity'] = preds['propensity']
        if 'tau_pred' in preds:
            fold_preds['pred_tau'] = preds['tau_pred']
        fold_preds['cv_fold'] = fold + 1

        all_predictions.append(fold_preds)

        # Cleanup
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # Combine predictions
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
        true_outcome=results_df['outcome_indicator'].values
    )

    logger.info(f"  Results for {config.name} (Hierarchical Confounder R-Learner):")
    logger.info(f"    ITE MSE: {metrics['ite_mse']:.4f}")
    logger.info(f"    ITE MAE: {metrics['ite_mae']:.4f}")
    logger.info(f"    ITE Correlation: {metrics['ite_corr']:.4f}")
    logger.info(f"    ITE Rank Corr: {metrics['ite_spearman_corr']:.4f}")
    logger.info(f"    ATE Bias: {metrics['ate_bias']:.4f}")
    logger.info(f"    Propensity AUROC: {metrics['propensity_auroc']:.4f}")
    logger.info(f"    Y0 AUROC (T=0): {metrics['y0_auroc']:.4f}")
    logger.info(f"    Y1 AUROC (T=1): {metrics['y1_auroc']:.4f}")

    return results_df, metrics


def create_llm_structured_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create structured text from LLM-extracted values."""
    df = df.copy()

    def make_structured(extracted_value):
        if pd.isna(extracted_value) or extracted_value == "unknown":
            return "Number of metastatic sites: unknown"
        return f"Number of metastatic sites: {extracted_value}"

    df['llm_structured_text'] = df['llm_extracted_metastatic_sites'].apply(make_structured)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run Hierarchical Confounder R-Learner experiment for clinical text ITE estimation"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Path to dataset parquet file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/confounder_rlearner_experiment",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, cpu, etc.)"
    )
    parser.add_argument(
        "--token-encoder",
        type=str,
        default=DEFAULT_TOKEN_ENCODER,
        help="HuggingFace model for token encoding (e.g., distilbert-base-uncased, emilyalsentzer/Bio_ClinicalBERT)"
    )
    parser.add_argument(
        "--num-latent-confounders",
        type=int,
        default=NUM_LATENT_CONFOUNDERS,
        help="Number of latent confounder vectors"
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
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (effective batch = batch_size * this)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--gamma-rlearner",
        type=float,
        default=1.0,
        help="Weight for R-learner loss"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="Specific conditions to run (e.g., 1_oracle 3_fine_tuned)"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Token encoder: {args.token_encoder}")
    logger.info(f"Latent confounders: {args.num_latent_confounders}")
    logger.info(f"Explicit confounders: {EXPLICIT_CONFOUNDER_TEXTS}")

    # Load dataset
    df = pd.read_parquet(args.dataset)
    logger.info(f"Loaded {len(df)} samples from {args.dataset}")

    # Check for required columns
    required_cols = ['clinical_text', 'patient_prompt', 'treatment_indicator',
                     'outcome_indicator', 'true_ite_prob', 'true_y0_prob', 'true_y1_prob']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check for LLM extraction column
    has_llm_extraction = 'llm_extracted_metastatic_sites' in df.columns

    # Create structured text for condition 6
    if has_llm_extraction:
        df = create_llm_structured_text(df)

    # Setup categorical encoder
    encoder = None
    if has_llm_extraction:
        encoder = CategoricalEncoder(categories=["1", "2", "3", "4_or_more"])
        encoder.fit(df['llm_extracted_metastatic_sites'].tolist())

    # Filter conditions if specified
    conditions = EXPERIMENT_CONDITIONS
    if args.conditions:
        conditions = [c for c in conditions if c.name in args.conditions]
        logger.info(f"Running {len(conditions)} selected conditions: {[c.name for c in conditions]}")

    # Skip LLM conditions if extraction not available
    if not has_llm_extraction:
        conditions = [c for c in conditions if not c.use_llm_extraction and not c.llm_as_text]
        logger.warning("LLM extraction column not found. Skipping LLM-based conditions.")

    # Run all conditions
    all_metrics = {}
    all_predictions = {}

    for config in conditions:
        try:
            results_df, metrics = run_condition(
                df, config, device,
                n_folds=args.n_folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gamma_rlearner=args.gamma_rlearner,
                token_encoder=args.token_encoder,
                num_latent_confounders=args.num_latent_confounders,
                explicit_confounder_texts=EXPLICIT_CONFOUNDER_TEXTS,
                encoder=encoder,
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )

            all_metrics[config.name] = metrics
            all_predictions[config.name] = results_df

            # Save condition results
            condition_dir = output_dir / config.name
            condition_dir.mkdir(exist_ok=True)
            results_df.to_parquet(condition_dir / "predictions.parquet", index=False)

        except Exception as e:
            logger.error(f"Error running condition {config.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save combined metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df.index.name = 'condition'
        metrics_df.to_csv(output_dir / "metrics_summary.csv")

        # Print summary table
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT RESULTS SUMMARY (Hierarchical Confounder R-Learner)")
        logger.info("=" * 80)
        logger.info("\n" + metrics_df.to_string())

    # Save config
    config_info = {
        'dataset': args.dataset,
        'model_type': 'hierarchical_confounder_rlearner',
        'token_encoder': args.token_encoder,
        'num_latent_confounders': args.num_latent_confounders,
        'explicit_confounder_texts': EXPLICIT_CONFOUNDER_TEXTS,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'gamma_rlearner': args.gamma_rlearner,
        'n_folds': args.n_folds,
        'device': str(device),
        'conditions_run': [c.name for c in conditions]
    }
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
