#!/usr/bin/env python
"""Causal Forest experiment for synthetic clinical text.

This script runs experiments using the two-stage CausalTextForest approach:
- Stage 1: Train neural feature extractor (GRU-Pool) with propensity + outcome loss
- Stage 2: Train CausalForestDML on extracted features for ITE estimation

Key advantages of causal forest approach:
- Doubly-robust estimation (robust to misspecification of either nuisance model)
- Built-in confidence intervals for treatment effects
- No gradient competition between representation learning and effect estimation
- Theoretical guarantees from the causal forest literature

Experimental Conditions:
1. Oracle (patient_prompt) - structured ground truth with confounders
2. Realistic (clinical_text) - natural clinical language
3. LLM-Extracted (llm_structured_text) - LLM extracted confounders as text

Usage:
    # Basic usage (no R-learner loss)
    python oracle_experiment_scripts/run_causal_forest_experiment.py \
        --dataset example_synthetic_data_one_confounder/dataset_with_extraction.parquet \
        --output-dir ../pcori_experiments/causal_forest \
        --device cuda:0 \
        --n-folds 5 \
        --cf-n-estimators 100 \
        --epochs 20

    # With R-learner representation training using shared features
    # (adds τ head and R-loss to Stage 1, same extractor for nuisance and effect)
    python oracle_experiment_scripts/run_causal_forest_experiment.py \
        --dataset example_synthetic_data_one_confounder/dataset_with_extraction.parquet \
        --output-dir ../pcori_experiments/causal_forest_rlearner_shared \
        --device cuda:0 \
        --rlearner-mode shared \
        --gamma-rlearner 1.0 \
        --cf-n-estimators 200 \
        --epochs 20

    # With R-learner representation training using dual extractors
    # (separate extractors for nuisance e(X),m(X) and effect τ(X))
    python oracle_experiment_scripts/run_causal_forest_experiment.py \
        --dataset example_synthetic_data_one_confounder/dataset_with_extraction.parquet \
        --output-dir ../pcori_experiments/causal_forest_rlearner_dual \
        --device cuda:0 \
        --rlearner-mode dual \
        --gamma-rlearner 1.0 \
        --cf-n-estimators 200 \
        --epochs 20



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

from cdt.models.causal_text_forest import CausalTextForest

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


@dataclass
class ExperimentConfig:
    """Configuration for an experimental condition."""
    name: str
    text_column: str
    # GRU-Pool architecture params
    embedding_dim: int = 128
    gru_hidden_dim: int = 128
    gru_num_layers: int = 1
    max_chunks: int = 100
    chunk_size: int = 128
    chunk_overlap: int = 32
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_dim: int = 256
    gated_attention_dim: int = 128
    projection_dim: int = 128
    # Causal forest params
    cf_n_estimators: int = 100
    cf_min_samples_leaf: int = 5
    cf_honest: bool = True


# Define experimental conditions
EXPERIMENT_CONDITIONS = [
    ExperimentConfig(
        name="1_oracle_patient_prompt",
        text_column="patient_prompt",
    ),
    ExperimentConfig(
         name="2_realistic_clinical_text",
         text_column="clinical_text",
    ),    
    ExperimentConfig(
        name="3_llm_extracted",
        text_column="llm_structured_text", 
    )


]



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

    # Confidence interval coverage (if available)
    if tau_lower is not None and tau_upper is not None:
        # Check how many true ITEs fall within predicted CI
        coverage = np.mean((true_ite >= tau_lower) & (true_ite <= tau_upper))
        metrics['ci_coverage'] = coverage

        # Proportion of significant effects (CI excludes 0)
        significant = (tau_lower > 0) | (tau_upper < 0)
        metrics['pct_significant'] = np.mean(significant)

        # Mean CI width
        metrics['mean_ci_width'] = np.mean(tau_upper - tau_lower)

    return metrics


def train_causal_forest_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    stop_grad_propensity: bool = False,
    use_rlearner_representation: bool = False,
    rlearner_dual_extractors: bool = False,
    gamma_rlearner: float = 1.0,
    numeric_features: bool = False
) -> Tuple[CausalTextForest, List[Dict]]:
    """Train a CausalTextForest model for one fold."""
    text_column = config.text_column

    # Create model
    model = CausalTextForest(
        feature_extractor_type="gru_pool",
        # GRU-Pool settings
        gru_pool_embedding_dim=config.embedding_dim,
        gru_pool_gru_hidden_dim=config.gru_hidden_dim,
        gru_pool_gru_num_layers=config.gru_num_layers,
        gru_pool_gru_bidirectional=True,
        gru_pool_gru_dropout=0.1,
        gru_pool_max_chunks=config.max_chunks,
        gru_pool_chunk_size=config.chunk_size,
        gru_pool_chunk_overlap=config.chunk_overlap,
        gru_pool_transformer_layers=config.transformer_layers,
        gru_pool_transformer_heads=config.transformer_heads,
        gru_pool_transformer_dim=config.transformer_dim,
        gru_pool_gated_attention_dim=config.gated_attention_dim,
        gru_pool_projection_dim=config.projection_dim,
        gru_pool_max_vocab=50000,
        gru_pool_min_word_freq=2,
        # Simple heads for representation learning
        representation_dim=128,
        hidden_dim=64,
        dropout=0.2,
        # Causal forest settings
        cf_n_estimators=config.cf_n_estimators,
        cf_min_samples_leaf=config.cf_min_samples_leaf,
        cf_honest=config.cf_honest,
        cf_inference=True,
        # R-learner representation training
        cf_use_rlearner_representation=use_rlearner_representation,
        cf_gamma_rlearner=gamma_rlearner,
        cf_rlearner_dual_extractors=rlearner_dual_extractors,
        numeric_features_enabled=numeric_features,
        device=str(device)
    )

    # Fit tokenizer
    train_texts = train_df[text_column].tolist()
    model.fit_tokenizer(train_texts)

    # Create datasets
    train_dataset = TextDataset(
        texts=train_texts,
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

    # Compute effective gamma for R-learner loss
    effective_gamma = gamma_rlearner if use_rlearner_representation else 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_prop_loss = 0.0
        train_outcome_loss = 0.0
        train_r_loss = 0.0

        for batch in train_loader:
            batch['treatment'] = batch['treatment'].to(device)
            batch['outcome'] = batch['outcome'].to(device)

            optimizer.zero_grad()

            losses = model.train_representation_step(
                batch,
                alpha_propensity=1.0,
                gamma_rlearner=effective_gamma,
                stop_grad_propensity=stop_grad_propensity
            )

            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += losses['loss'].item()
            train_prop_loss += losses['propensity_loss'].item()
            train_outcome_loss += losses['outcome_loss'].item()
            train_r_loss += losses.get('r_loss', torch.tensor(0.0)).item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_prop_loss = 0.0
        val_outcome_loss = 0.0
        val_r_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)

                losses = model.train_representation_step(
                    batch,
                    alpha_propensity=1.0,
                    gamma_rlearner=effective_gamma,
                    stop_grad_propensity=stop_grad_propensity
                )

                val_loss += losses['loss'].item()
                val_prop_loss += losses['propensity_loss'].item()
                val_outcome_loss += losses['outcome_loss'].item()
                val_r_loss += losses.get('r_loss', torch.tensor(0.0)).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_prop_loss /= len(train_loader)
        val_prop_loss /= len(val_loader)
        train_outcome_loss /= len(train_loader)
        val_outcome_loss /= len(val_loader)
        train_r_loss /= len(train_loader)
        val_r_loss /= len(val_loader)

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_propensity_loss': train_prop_loss,
            'val_propensity_loss': val_prop_loss,
            'train_outcome_loss': train_outcome_loss,
            'val_outcome_loss': val_outcome_loss,
            'lr': scheduler.get_last_lr()[0]
        }
        if use_rlearner_representation:
            epoch_log['train_r_loss'] = train_r_loss
            epoch_log['val_r_loss'] = val_r_loss
        history.append(epoch_log)

        if (epoch + 1) % 10 == 0:
            r_loss_str = f", r_loss={train_r_loss:.4f}" if use_rlearner_representation else ""
            logger.info(f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                       f"val_loss={val_loss:.4f}{r_loss_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    # Stage 2: Train causal forest on combined train + val
    logger.info("    Training causal forest on extracted features...")
    combined_df = pd.concat([train_df, val_df])
    combined_texts = combined_df[text_column].tolist()
    combined_T = combined_df['treatment_indicator'].values
    combined_Y = combined_df['outcome_indicator'].values

    model.train_causal_forest(combined_texts, combined_T, combined_Y, batch_size=batch_size)

    return model, history


def predict_model(
    model: CausalTextForest,
    df: pd.DataFrame,
    text_column: str,
    batch_size: int
) -> Dict[str, np.ndarray]:
    """Generate predictions from model."""
    texts = df[text_column].tolist()

    # Get predictions with confidence intervals
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


def run_condition(
    df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    n_folds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Optional[Path] = None,
    stop_grad_propensity: bool = False,
    use_rlearner_representation: bool = False,
    rlearner_dual_extractors: bool = False,
    gamma_rlearner: float = 1.0,
    numeric_features: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run cross-validation for one experimental condition."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running condition: {config.name}")
    logger.info(f"  Text column: {config.text_column}")
    logger.info(f"  Causal forest: {config.cf_n_estimators} trees, honest={config.cf_honest}")
    logger.info(f"  GRU-Pool: embed={config.embedding_dim}, hidden={config.gru_hidden_dim}")
    logger.info(f"  Stop grad propensity: {stop_grad_propensity}")
    if use_rlearner_representation:
        mode_str = "dual extractors" if rlearner_dual_extractors else "shared"
        logger.info(f"  R-learner representation: {mode_str}")
        logger.info(f"  gamma_rlearner: {gamma_rlearner}")
    else:
        logger.info(f"  R-learner representation: none")
    logger.info(f"{'='*60}")

    # Check if text column exists
    if config.text_column not in df.columns:
        logger.warning(f"  Skipping condition: column '{config.text_column}' not found in dataset")
        return pd.DataFrame(), {}

    # Reset index
    df = df.reset_index(drop=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        logger.info(f"  Fold {fold + 1}/{n_folds}")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model, history = train_causal_forest_model(
            train_df, test_df, config, device,
            epochs, batch_size, learning_rate,
            stop_grad_propensity=stop_grad_propensity,
            use_rlearner_representation=use_rlearner_representation,
            rlearner_dual_extractors=rlearner_dual_extractors,
            gamma_rlearner=gamma_rlearner,
            numeric_features=numeric_features
        )

        preds = predict_model(model, test_df, config.text_column, batch_size)

        # Store predictions
        fold_preds = test_df.copy()
        fold_preds['pred_y0_prob'] = preds['y0_prob']
        fold_preds['pred_y1_prob'] = preds['y1_prob']
        fold_preds['pred_ite_prob'] = preds['ite_prob']
        fold_preds['pred_propensity'] = preds['propensity']
        fold_preds['pred_tau'] = preds['tau_pred']
        fold_preds['cv_fold'] = fold + 1

        if preds['tau_lower'] is not None:
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

    # Combine predictions
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

    logger.info(f"  Results for {config.name}:")
    logger.info(f"    ITE MSE: {metrics['ite_mse']:.4f}")
    logger.info(f"    ITE MAE: {metrics['ite_mae']:.4f}")
    logger.info(f"    ITE Correlation: {metrics['ite_corr']:.4f}")
    logger.info(f"    ITE Rank Corr: {metrics['ite_spearman_corr']:.4f}")
    logger.info(f"    ATE Bias: {metrics['ate_bias']:.4f}")
    logger.info(f"    ATE Predicted: {metrics['ate_pred']:.4f}")
    logger.info(f"    ATE True: {metrics['ate_true']:.4f}")
    logger.info(f"    Propensity AUROC: {metrics['propensity_auroc']:.4f}")
    if 'ci_coverage' in metrics:
        logger.info(f"    CI Coverage: {metrics['ci_coverage']:.4f}")
        logger.info(f"    Mean CI Width: {metrics['mean_ci_width']:.4f}")
        logger.info(f"    Pct Significant: {metrics['pct_significant']:.4f}")

    return results_df, metrics


def create_llm_structured_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create structured text from LLM extracted values."""
    df = df.copy()

    # Check if llm_extracted_metastatic_sites exists
    if 'llm_extracted_metastatic_sites' in df.columns:
        # Create a text representation of the extracted confounder
        df['llm_structured_text'] = df['llm_extracted_metastatic_sites'].apply(
            lambda x: f"Number of metastatic sites: {x}. This patient has {x} sites of metastatic disease."
        )
        logger.info("Created llm_structured_text from llm_extracted_metastatic_sites")
    else:
        logger.warning("llm_extracted_metastatic_sites column not found")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run Causal Forest experiment with GRU-Pool feature extraction"
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
        default="./causal_forest_experiment_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)"
    )
    # GRU-Pool architecture parameters
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Word embedding dimension"
    )
    parser.add_argument(
        "--gru-hidden-dim",
        type=int,
        default=128,
        help="GRU hidden dimension per direction"
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=2,
        help="Number of cross-chunk transformer layers"
    )
    parser.add_argument(
        "--transformer-dim",
        type=int,
        default=256,
        help="Transformer hidden dimension"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Tokens per chunk"
    )
    # Causal forest parameters
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
        "--cf-honest",
        action="store_true",
        default=True,
        help="Use honest estimation in causal forest"
    )
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of representation training epochs"
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
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="Specific conditions to run (e.g., 1_oracle_patient_prompt)"
    )
    parser.add_argument(
        "--stop-grad-propensity",
        action="store_true",
        help="Detach features before propensity loss"
    )
    parser.add_argument(
        "--rlearner-mode",
        type=str,
        choices=["none", "shared", "dual"],
        default="none",
        help="R-learner representation training mode: "
             "'none' (default) = no R-learner loss, "
             "'shared' = R-learner loss with shared feature extractor, "
             "'dual' = R-learner loss with separate extractors for nuisance/effect"
    )
    parser.add_argument(
        "--gamma-rlearner",
        type=float,
        default=1.0,
        help="Weight for R-learner loss (only used when --rlearner-mode is 'shared' or 'dual')"
    )
    parser.add_argument(
        "--numeric-features",
        action="store_true",
        default=False,
        help="Enable magnitude-aware numeric feature extraction from clinical text"
    )

    args = parser.parse_args()

    # Ensure n_estimators is divisible by 4 (econml requirement)
    if args.cf_n_estimators % 4 != 0:
        args.cf_n_estimators = (args.cf_n_estimators // 4 + 1) * 4
        logger.warning(f"Adjusted cf_n_estimators to {args.cf_n_estimators} (must be divisible by 4)")

    # Parse R-learner mode into flags
    use_rlearner_representation = args.rlearner_mode in ("shared", "dual")
    rlearner_dual_extractors = args.rlearner_mode == "dual"

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Using CausalTextForest (GRU-Pool + CausalForestDML)")
    logger.info(f"  Embedding dim: {args.embedding_dim}")
    logger.info(f"  GRU hidden dim: {args.gru_hidden_dim} x 2 directions")
    logger.info(f"  Transformer layers: {args.transformer_layers}")
    logger.info(f"  Transformer dim: {args.transformer_dim}")
    logger.info(f"  Causal forest: {args.cf_n_estimators} trees")
    logger.info(f"  stop_grad_propensity: {args.stop_grad_propensity}")
    logger.info(f"  R-learner mode: {args.rlearner_mode}")
    if use_rlearner_representation:
        logger.info(f"    use_rlearner_representation: True")
        logger.info(f"    dual_extractors: {rlearner_dual_extractors}")
        logger.info(f"    gamma_rlearner: {args.gamma_rlearner}")
    logger.info(f"  Numeric features: {args.numeric_features}")

    # Load dataset
    df = pd.read_parquet(args.dataset)
    logger.info(f"Loaded {len(df)} samples from {args.dataset}")

    # Create LLM structured text column
    df = create_llm_structured_text(df)

    # Check for required columns
    required_cols = ['treatment_indicator', 'outcome_indicator', 'true_ite_prob', 'true_y0_prob', 'true_y1_prob']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check available text columns
    for col in ['patient_prompt', 'clinical_text', 'llm_structured_text']:
        if col in df.columns:
            logger.info(f"  Found text column: {col}")
        else:
            logger.warning(f"  Text column not found: {col}")

    # Update experiment configs with command-line parameters
    conditions = []
    for config in EXPERIMENT_CONDITIONS:
        # Skip conditions with unavailable text columns
        if config.text_column not in df.columns:
            logger.warning(f"Skipping condition {config.name}: column '{config.text_column}' not in dataset")
            continue

        # Create new config with updated parameters
        new_config = ExperimentConfig(
            name=config.name,
            text_column=config.text_column,
            embedding_dim=args.embedding_dim,
            gru_hidden_dim=args.gru_hidden_dim,
            transformer_layers=args.transformer_layers,
            transformer_dim=args.transformer_dim,
            chunk_size=args.chunk_size,
            cf_n_estimators=args.cf_n_estimators,
            cf_min_samples_leaf=args.cf_min_samples_leaf,
            cf_honest=args.cf_honest
        )
        conditions.append(new_config)

    # Filter conditions if specified
    if args.conditions:
        conditions = [c for c in conditions if c.name in args.conditions]
        logger.info(f"Running {len(conditions)} selected conditions")
    else:
        logger.info(f"Running {len(conditions)} available conditions")

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
                output_dir=output_dir,
                stop_grad_propensity=args.stop_grad_propensity,
                use_rlearner_representation=use_rlearner_representation,
                rlearner_dual_extractors=rlearner_dual_extractors,
                gamma_rlearner=args.gamma_rlearner,
                numeric_features=args.numeric_features
            )

            if metrics:
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
        logger.info("EXPERIMENT RESULTS SUMMARY (Causal Forest)")
        logger.info("=" * 80)
        logger.info("\n" + metrics_df.to_string())
    else:
        logger.warning("No successful experiments completed")

    # Save config
    config_info = {
        'dataset': args.dataset,
        'model_type': 'causal_forest',
        'feature_extractor_type': 'gru_pool',
        'embedding_dim': args.embedding_dim,
        'gru_hidden_dim': args.gru_hidden_dim,
        'transformer_layers': args.transformer_layers,
        'transformer_dim': args.transformer_dim,
        'chunk_size': args.chunk_size,
        'cf_n_estimators': args.cf_n_estimators,
        'cf_min_samples_leaf': args.cf_min_samples_leaf,
        'cf_honest': args.cf_honest,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'stop_grad_propensity': args.stop_grad_propensity,
        'rlearner_mode': args.rlearner_mode,
        'use_rlearner_representation': use_rlearner_representation,
        'rlearner_dual_extractors': rlearner_dual_extractors,
        'gamma_rlearner': args.gamma_rlearner,
        'numeric_features': args.numeric_features,
        'n_folds': args.n_folds,
        'device': str(device),
        'conditions_run': [c.name for c in conditions if c.name in all_metrics]
    }
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
