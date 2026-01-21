#!/usr/bin/env python
"""GRU Hierarchical Confounder Extractor R-Learner experiment for synthetic clinical text.

This script tests whether learning confounder extraction from scratch
(via GRU + causal objective) improves ITE estimation compared to
pretrained BERT-based approaches.

Key features:
- Uses GRUHierarchicalConfounderExtractor (learns from scratch)
- All parameters (embeddings, GRU, attention, latent confounders) optimize together
- Sparse attention (entmax) to focus on relevant sentences
- R-Learner objective for direct treatment effect optimization
- No pretrained encoder - all parameters learn together via causal loss

Conditions tested:
1. Oracle: patient_prompt (explicit confounder, upper bound)
2. GRU from scratch: clinical_text with GRU confounder extractor
3. LLM-extract-only: Extracted categorical features (MLP baseline)

Usage:
    python oracle_experiment_scripts/run_gru_confounder_experiment.py \
        --dataset path/to/dataset.parquet \
        --output-dir results/gru_confounder \
        --device cuda:0
"""

import argparse
import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from cdt.models.mlp_dragonnet import MLPDragonNet, CategoricalEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Number of latent confounders to learn
NUM_LATENT_CONFOUNDERS = 8


@dataclass
class ExperimentConfig:
    """Configuration for an experimental condition."""
    name: str
    text_column: str
    use_llm_extraction: bool = False
    llm_extraction_only: bool = False


# Define experimental conditions
EXPERIMENT_CONDITIONS = [
    ExperimentConfig(
        name="1_oracle",
        text_column="patient_prompt"
    ),
    ExperimentConfig(
        name="2_gru_from_scratch",
        text_column="clinical_text"
    ),
    ExperimentConfig(
        name="3_llm_extract_only",
        text_column="clinical_text",  # Not used
        use_llm_extraction=True,
        llm_extraction_only=True
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
    true_y1: np.ndarray
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    metrics = {}

    # ITE metrics
    metrics['ite_mse'] = mean_squared_error(true_ite, pred_ite)
    metrics['ite_mae'] = mean_absolute_error(true_ite, pred_ite)
    metrics['ite_corr'], _ = stats.pearsonr(pred_ite, true_ite)
    metrics['ate_bias'] = abs(np.mean(pred_ite) - np.mean(true_ite))
    metrics['ate_pred'] = np.mean(pred_ite)
    metrics['ate_true'] = np.mean(true_ite)

    # Propensity metrics
    try:
        metrics['propensity_auroc'] = roc_auc_score(true_treatment, pred_propensity)
    except ValueError:
        metrics['propensity_auroc'] = np.nan

    # Outcome metrics
    metrics['y0_mse'] = mean_squared_error(true_y0, pred_y0)
    metrics['y1_mse'] = mean_squared_error(true_y1, pred_y1)

    return metrics


def train_gru_confounder_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gamma_rlearner: float = 1.0,
    num_latent_confounders: int = NUM_LATENT_CONFOUNDERS,
    gru_embedding_dim: int = 128,
    gru_hidden_dim: int = 128,
    gru_num_layers: int = 1,
    gradient_accumulation_steps: int = 1
) -> Tuple[CausalText, List[Dict]]:
    """Train a GRU Hierarchical Confounder Extractor R-Learner model for one fold."""
    text_column = config.text_column

    # Get training texts for tokenizer fitting
    train_texts = train_df[text_column].tolist()

    # Create model with GRU-based Confounder Extractor
    model = CausalText(
        feature_extractor_type="confounder",
        model_type="rlearner",
        # Enable GRU-based confounder extractor (learns from scratch)
        confounder_use_gru=True,
        confounder_gru_embedding_dim=gru_embedding_dim,
        confounder_gru_hidden_dim=gru_hidden_dim,
        confounder_gru_num_layers=gru_num_layers,
        confounder_gru_bidirectional=True,
        confounder_gru_dropout=0.1,
        confounder_gru_min_word_freq=1,  # Lower for synthetic data
        confounder_gru_max_sentence_length=128,
        # Confounder architecture
        confounder_num_latents=num_latent_confounders,
        confounder_value_dim=128,
        confounder_max_sentences=100,
        confounder_num_heads=4,
        # Sparse attention settings
        confounder_sparse_attention=True,
        confounder_sparse_method="entmax",
        confounder_sparse_alpha=1.5,
        confounder_dropout=0.1,
        # DragonNet/R-Learner head parameters
        dragonnet_representation_dim=128,
        dragonnet_hidden_outcome_dim=64,
        dragonnet_dropout=0.2,
        device=str(device)
    )

    # IMPORTANT: GRU mode requires tokenizer fitting
    model.fit_tokenizer(train_texts)
    logger.info(f"Fitted word tokenizer on {len(train_texts)} training texts")

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
        dragonnet_representation_dim=32,
        dragonnet_hidden_outcome_dim=16,
        dragonnet_dropout=0.2,
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


def predict_gru_confounder(
    model: CausalText,
    df: pd.DataFrame,
    text_column: str,
    device: torch.device,
    batch_size: int
) -> Dict[str, np.ndarray]:
    """Generate predictions from GRU Confounder R-Learner model."""
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
    num_latent_confounders: int = NUM_LATENT_CONFOUNDERS,
    gru_embedding_dim: int = 128,
    gru_hidden_dim: int = 128,
    gru_num_layers: int = 1,
    encoder: Optional[CategoricalEncoder] = None,
    gradient_accumulation_steps: int = 1
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run cross-validation for one experimental condition."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running condition: {config.name} (GRU Confounder R-Learner)")
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
            # GRU Confounder Extractor R-Learner
            model, _ = train_gru_confounder_model(
                train_df, test_df, config, device,
                epochs, batch_size, learning_rate,
                gamma_rlearner=gamma_rlearner,
                num_latent_confounders=num_latent_confounders,
                gru_embedding_dim=gru_embedding_dim,
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            preds = predict_gru_confounder(
                model, test_df, config.text_column, device, batch_size
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        true_y1=results_df['true_y1_prob'].values
    )

    logger.info(f"  Results for {config.name} (GRU Confounder R-Learner):")
    logger.info(f"    ITE MSE: {metrics['ite_mse']:.4f}")
    logger.info(f"    ITE MAE: {metrics['ite_mae']:.4f}")
    logger.info(f"    ITE Correlation: {metrics['ite_corr']:.4f}")
    logger.info(f"    ATE Bias: {metrics['ate_bias']:.4f}")
    logger.info(f"    Propensity AUROC: {metrics['propensity_auroc']:.4f}")

    return results_df, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run GRU Confounder R-Learner experiment for clinical text ITE estimation"
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
        default="results/gru_confounder_experiment",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, cpu, etc.)"
    )
    parser.add_argument(
        "--num-latent-confounders",
        type=int,
        default=NUM_LATENT_CONFOUNDERS,
        help="Number of latent confounder vectors"
    )
    parser.add_argument(
        "--gru-embedding-dim",
        type=int,
        default=128,
        help="Word embedding dimension for GRU"
    )
    parser.add_argument(
        "--gru-hidden-dim",
        type=int,
        default=128,
        help="GRU hidden state dimension per direction"
    )
    parser.add_argument(
        "--gru-num-layers",
        type=int,
        default=1,
        help="Number of stacked GRU layers"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
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
        help="Specific conditions to run (e.g., 1_oracle 2_gru_from_scratch)"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"GRU embedding dim: {args.gru_embedding_dim}")
    logger.info(f"GRU hidden dim: {args.gru_hidden_dim}")
    logger.info(f"GRU num layers: {args.gru_num_layers}")
    logger.info(f"Latent confounders: {args.num_latent_confounders}")

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
        conditions = [c for c in conditions if not c.use_llm_extraction]
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
                num_latent_confounders=args.num_latent_confounders,
                gru_embedding_dim=args.gru_embedding_dim,
                gru_hidden_dim=args.gru_hidden_dim,
                gru_num_layers=args.gru_num_layers,
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
        logger.info("EXPERIMENT RESULTS SUMMARY (GRU Confounder R-Learner)")
        logger.info("=" * 80)
        logger.info("\n" + metrics_df.to_string())

    # Save config
    config_info = {
        'dataset': args.dataset,
        'model_type': 'gru_confounder_rlearner',
        'gru_embedding_dim': args.gru_embedding_dim,
        'gru_hidden_dim': args.gru_hidden_dim,
        'gru_num_layers': args.gru_num_layers,
        'num_latent_confounders': args.num_latent_confounders,
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
