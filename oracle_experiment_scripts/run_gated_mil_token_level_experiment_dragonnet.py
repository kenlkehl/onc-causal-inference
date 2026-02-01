#!/usr/bin/env python
"""Gated MIL Token-Level Hierarchical experiment using DragonNet for synthetic clinical text.

This script runs experiments using the GatedMILHierarchicalExtractor with TOKEN-LEVEL
gated pooling enabled (`gated_mil_hierarchical=True`), combined with DragonNet
architecture for causal inference with propensity score regularization.

Token-level mode preserves fine-grained distinctions that sentence-level [CLS]
embeddings may lose, such as:
- "ECOG performance status 0" vs "ECOG performance status 2"
- "no metastatic disease" vs "metastatic disease"
- "stable disease" vs "progressive disease"

Architecture with token-level gating:
1. Split text into sentences
2. Encode each sentence with BERT, keeping ALL token embeddings (not just [CLS])
3. Token-level gated pooling: Each of K confounder queries attends to tokens,
   creating K confounder-specific sentence representations per sentence
4. Sentence-level gated MIL attention: Each confounder's view is aggregated
5. Task-specific weighting of shared confounders (propensity, y0, y1)

Key insight: Token-level gated attention creates K DIFFERENT representations per
sentence. Confounder 1 might focus on "metastatic" while confounder 2 focuses on
"PS 2". This preserves fine-grained signal that gets lost in a single [CLS] embedding.

DragonNet outputs:
- y0_logit: Predicted outcome under control (T=0)
- y1_logit: Predicted outcome under treatment (T=1)
- t_logit: Treatment propensity logit
- ITE = sigmoid(y1_logit) - sigmoid(y0_logit)

Experimental Conditions:
1. Oracle: patient_prompt (structured ground truth) with token-level gated MIL
2. Clinical Text: clinical_text (natural language) with token-level gated MIL
3. LLM-as-Text: llm_structured_text (LLM extracted, converted to text)

Usage:
    python oracle_experiment_scripts/run_gated_mil_token_level_experiment_dragonnet.py \
        --dataset ../pcori_experiments/explicit_confounder_experiments_1-19-26/dataset_with_extraction.parquet \
        --output-dir ../pcori_experiments/explicit_confounder_experiments_1-19-26/token_mil_dragonnet_experiment \
        --device cuda:0 \
        --epochs 25

Comparison with sentence-level:
    To compare, run both experiments:
    1. run_gated_mil_hierarchical_experiment.py (sentence-level, default)
    2. run_gated_mil_token_level_experiment_dragonnet.py (token-level, this script)
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

from cdt.models.causal_text import CausalText

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
    sentence_model: str = "prajjwal1/bert-tiny"
    freeze_sentence_encoder: bool = False  # Fine-tune by default
    max_sentences: int = 100
    max_sentence_length: int = 128
    mil_hidden_dim: int = 128
    num_confounders: int = 4
    projection_dim: int = 128
    # Token-level settings
    hierarchical: bool = True  # ENABLED for token-level gating
    token_hidden_dim: int = 64


# Define experimental conditions
EXPERIMENT_CONDITIONS = [
    ExperimentConfig(
        name="1_oracle_token_level",
        text_column="patient_prompt",
    ),
    ExperimentConfig(
        name="2_clinical_text_token_level",
        text_column="clinical_text",
    ),
    ExperimentConfig(
        name="3_llm_as_text_token_level",
        text_column="llm_structured_text",
    ),
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
    true_outcome: np.ndarray
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


def train_gated_mil_token_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta_targreg: float = 0.1,
    stop_grad_propensity: bool = False,
    attention_entropy_weight: float = 0.0,
    use_mean_pooling: bool = False
) -> Tuple[CausalText, List[Dict]]:
    """Train a Gated MIL Token-Level Hierarchical DragonNet model for one fold."""
    text_column = config.text_column

    # Create model with Gated MIL Hierarchical + Token-Level + DragonNet
    model = CausalText(
        feature_extractor_type="gated_mil_hierarchical",
        model_type="dragonnet",
        # Gated MIL Hierarchical settings
        gated_mil_sentence_model=config.sentence_model,
        gated_mil_freeze_sentence_encoder=config.freeze_sentence_encoder,
        gated_mil_max_sentences=config.max_sentences,
        gated_mil_max_sentence_length=config.max_sentence_length,
        gated_mil_hidden_dim=config.mil_hidden_dim,
        gated_mil_num_confounders=config.num_confounders,
        gated_mil_dropout=0.1,
        gated_mil_projection_dim=config.projection_dim,
        # TOKEN-LEVEL GATING ENABLED
        gated_mil_hierarchical=config.hierarchical,  # True for token-level
        gated_mil_token_hidden_dim=config.token_hidden_dim,
        gated_mil_use_mean_pooling=use_mean_pooling,
        # DragonNet head settings
        dragonnet_representation_dim=128,
        dragonnet_hidden_outcome_dim=64,
        dragonnet_dropout=0.2,
        device=str(device)
    )

    # Trigger lazy initialization (no-op for pretrained encoders)
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

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_targreg_loss = 0.0
        train_entropy_loss = 0.0
        for batch in train_loader:
            batch['treatment'] = batch['treatment'].to(device)
            batch['outcome'] = batch['outcome'].to(device)

            optimizer.zero_grad()
            losses = model.train_step(
                batch,
                alpha_propensity=1.0,
                beta_targreg=beta_targreg,
                stop_grad_propensity=stop_grad_propensity,
                attention_entropy_weight=attention_entropy_weight
            )
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += losses['loss'].item()
            train_targreg_loss += losses.get('targreg_loss', 0.0)
            if isinstance(train_targreg_loss, torch.Tensor):
                train_targreg_loss = train_targreg_loss.item()
            if 'entropy_loss' in losses:
                train_entropy_loss += losses['entropy_loss'].item() if hasattr(losses['entropy_loss'], 'item') else losses['entropy_loss']

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_targreg_loss = 0.0
        val_entropy_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)
                losses = model.train_step(
                    batch,
                    alpha_propensity=1.0,
                    beta_targreg=beta_targreg,
                    stop_grad_propensity=stop_grad_propensity,
                    attention_entropy_weight=attention_entropy_weight
                )
                val_loss += losses['loss'].item()
                targreg = losses.get('targreg_loss', 0.0)
                if isinstance(targreg, torch.Tensor):
                    targreg = targreg.item()
                val_targreg_loss += targreg
                if 'entropy_loss' in losses:
                    val_entropy_loss += losses['entropy_loss'].item() if hasattr(losses['entropy_loss'], 'item') else losses['entropy_loss']

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_targreg_loss /= len(train_loader)
        val_targreg_loss /= len(val_loader)
        if attention_entropy_weight > 0:
            train_entropy_loss /= len(train_loader)
            val_entropy_loss /= len(val_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_targreg_loss': train_targreg_loss,
            'val_targreg_loss': val_targreg_loss,
            'train_entropy_loss': train_entropy_loss if attention_entropy_weight > 0 else None,
            'val_entropy_loss': val_entropy_loss if attention_entropy_weight > 0 else None,
            'lr': scheduler.get_last_lr()[0]
        })

        if (epoch + 1) % 10 == 0:
            log_msg = f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_targreg={train_targreg_loss:.4f}"
            if attention_entropy_weight > 0:
                log_msg += f", entropy_loss={train_entropy_loss:.4f}"
            logger.info(log_msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    return model, history


def predict_model(
    model: CausalText,
    df: pd.DataFrame,
    text_column: str,
    device: torch.device,
    batch_size: int
) -> Dict[str, np.ndarray]:
    """Generate predictions from model."""
    model.eval()

    texts = df[text_column].tolist()
    all_y0 = []
    all_y1 = []
    all_prop = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            preds = model.predict(batch_texts)
            all_y0.append(preds['y0_prob'].cpu().numpy())
            all_y1.append(preds['y1_prob'].cpu().numpy())
            all_prop.append(preds['propensity'].cpu().numpy())

    y0_prob = np.concatenate(all_y0)
    y1_prob = np.concatenate(all_y1)

    return {
        'y0_prob': y0_prob,
        'y1_prob': y1_prob,
        'propensity': np.concatenate(all_prop),
        'ite_prob': y1_prob - y0_prob
    }


def run_condition(
    df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    n_folds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta_targreg: float = 0.1,
    save_attention: bool = False,
    output_dir: Optional[Path] = None,
    stop_grad_propensity: bool = False,
    attention_entropy_weight: float = 0.0,
    use_mean_pooling: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run cross-validation for one experimental condition."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running condition: {config.name}")
    logger.info(f"  Text column: {config.text_column}")
    logger.info(f"  Sentence model: {config.sentence_model}")
    logger.info(f"  Freeze encoder: {config.freeze_sentence_encoder}")
    logger.info(f"  Num confounders: {config.num_confounders}")
    logger.info(f"  Token-level hierarchical: {config.hierarchical}")  # Key difference
    logger.info(f"  Token hidden dim: {config.token_hidden_dim}")
    logger.info(f"  Stop grad propensity: {stop_grad_propensity}")
    logger.info(f"  Attention entropy weight: {attention_entropy_weight}")
    logger.info(f"  Use mean pooling: {use_mean_pooling}")
    logger.info(f"{'='*60}")

    # Reset index
    df = df.reset_index(drop=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_predictions = []
    last_model = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        logger.info(f"  Fold {fold + 1}/{n_folds}")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model, history = train_gated_mil_token_model(
            train_df, test_df, config, device,
            epochs, batch_size, learning_rate,
            beta_targreg=beta_targreg,
            stop_grad_propensity=stop_grad_propensity,
            attention_entropy_weight=attention_entropy_weight,
            use_mean_pooling=use_mean_pooling
        )
        preds = predict_model(model, test_df, config.text_column, device, batch_size)

        # Store predictions
        fold_preds = test_df.copy()
        fold_preds['pred_y0_prob'] = preds['y0_prob']
        fold_preds['pred_y1_prob'] = preds['y1_prob']
        fold_preds['pred_ite_prob'] = preds['ite_prob']
        fold_preds['pred_propensity'] = preds['propensity']
        fold_preds['cv_fold'] = fold + 1

        all_predictions.append(fold_preds)

        # Keep last model for attention visualization
        if fold == n_folds - 1:
            last_model = model
        else:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save attention weights from last fold if requested
    if save_attention and last_model is not None and output_dir is not None:
        logger.info("  Saving attention interpretations from last fold...")
        try:
            sample_texts = df[config.text_column].head(20).tolist()
            interpretations = last_model.feature_extractor.interpret_attention(sample_texts, top_k=5)
            task_weights = last_model.feature_extractor.get_task_weights()

            interp_path = output_dir / f"{config.name}_attention_interpretations.json"
            with open(interp_path, 'w') as f:
                json.dump({
                    'interpretations': interpretations,
                    'task_weights': task_weights,
                    'token_level_enabled': True
                }, f, indent=2, default=str)
            logger.info(f"  Saved attention interpretations to: {interp_path}")
        except Exception as e:
            logger.warning(f"  Failed to save attention interpretations: {e}")

    # Cleanup last model
    if last_model is not None:
        del last_model
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

    logger.info(f"  Results for {config.name}:")
    logger.info(f"    ITE MSE: {metrics['ite_mse']:.4f}")
    logger.info(f"    ITE MAE: {metrics['ite_mae']:.4f}")
    logger.info(f"    ITE Correlation: {metrics['ite_corr']:.4f}")
    logger.info(f"    ATE Bias: {metrics['ate_bias']:.4f}")
    logger.info(f"    ATE Predicted: {metrics['ate_pred']:.4f}")
    logger.info(f"    ATE True: {metrics['ate_true']:.4f}")
    logger.info(f"    Propensity AUROC: {metrics['propensity_auroc']:.4f}")
    logger.info(f"    Y0 AUROC (T=0): {metrics['y0_auroc']:.4f}")
    logger.info(f"    Y1 AUROC (T=1): {metrics['y1_auroc']:.4f}")

    return results_df, metrics


def create_llm_structured_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create structured text from LLM-extracted values (condition 3)."""
    df = df.copy()

    def make_structured(extracted_value):
        if pd.isna(extracted_value) or extracted_value == "unknown":
            return "Number of metastatic sites: unknown"
        return f"Number of metastatic sites: {extracted_value}"

    df['llm_structured_text'] = df['llm_extracted_metastatic_sites'].apply(make_structured)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run Gated MIL Token-Level Hierarchical experiment with DragonNet"
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
        default="results/gated_mil_token_level_dragonnet_experiment",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--sentence-model",
        type=str,
        default="prajjwal1/bert-tiny",
        help="Sentence encoder model (e.g., prajjwal1/bert-tiny, prajjwal1/bert-small)"
    )
    parser.add_argument(
        "--num-confounders",
        type=int,
        default=4,
        help="Number of confounder queries (K)"
    )
    parser.add_argument(
        "--mil-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for gated MIL attention"
    )
    parser.add_argument(
        "--token-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for token-level gated attention"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        "--beta-targreg",
        type=float,
        default=0.1,
        help="Weight for targeted regularization loss"
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
        help="Specific conditions to run (e.g., 1_oracle_token_level 2_clinical_text_token_level)"
    )
    parser.add_argument(
        "--skip-llm-condition",
        action="store_true",
        help="Skip condition 3 that requires LLM extractions"
    )
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save attention interpretations for analysis"
    )
    parser.add_argument(
        "--stop-grad-propensity",
        action="store_true",
        help="Detach features before propensity loss (prevents propensity from dominating representation)"
    )
    parser.add_argument(
        "--attention-entropy-weight",
        type=float,
        default=0.0,
        help="Weight for attention entropy regularization (encourages focused attention)"
    )
    parser.add_argument(
        "--use-mean-pooling",
        action="store_true",
        help="Use mean pooling instead of [CLS] token for sentence embeddings"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Using Gated MIL Token-Level Hierarchical with DragonNet")
    logger.info(f"  Sentence model: {args.sentence_model}")
    logger.info(f"  Num confounders: {args.num_confounders}")
    logger.info(f"  MIL hidden dim: {args.mil_hidden_dim}")
    logger.info(f"  Token hidden dim: {args.token_hidden_dim}")
    logger.info(f"  Token-level gating: ENABLED")
    logger.info(f"  Freeze encoder: False (fine-tuning enabled)")
    logger.info(f"  beta_targreg: {args.beta_targreg}")
    logger.info(f"  stop_grad_propensity: {args.stop_grad_propensity}")
    logger.info(f"  attention_entropy_weight: {args.attention_entropy_weight}")
    logger.info(f"  use_mean_pooling: {args.use_mean_pooling}")

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

    # Create structured text for condition 3
    if has_llm_extraction:
        df = create_llm_structured_text(df)

    # Update experiment configs with command-line parameters
    conditions = []
    for config in EXPERIMENT_CONDITIONS:
        # Create new config with updated parameters
        new_config = ExperimentConfig(
            name=config.name,
            text_column=config.text_column,
            sentence_model=args.sentence_model,
            freeze_sentence_encoder=False,  # Always fine-tune as requested
            max_sentences=config.max_sentences,
            max_sentence_length=config.max_sentence_length,
            mil_hidden_dim=args.mil_hidden_dim,
            num_confounders=args.num_confounders,
            projection_dim=config.projection_dim,
            # Token-level settings
            hierarchical=True,  # Always enabled in this script
            token_hidden_dim=args.token_hidden_dim
        )
        conditions.append(new_config)

    # Filter conditions if specified
    if args.conditions:
        conditions = [c for c in conditions if c.name in args.conditions]
        logger.info(f"Running {len(conditions)} selected conditions")

    # Skip LLM condition if requested or extraction not available
    if args.skip_llm_condition or not has_llm_extraction:
        conditions = [c for c in conditions if c.name != "3_llm_as_text_token_level"]
        if not has_llm_extraction:
            logger.warning("LLM extraction column not found. Skipping condition 3.")

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
                beta_targreg=args.beta_targreg,
                save_attention=args.save_attention,
                output_dir=output_dir,
                stop_grad_propensity=args.stop_grad_propensity,
                attention_entropy_weight=args.attention_entropy_weight,
                use_mean_pooling=args.use_mean_pooling
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
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = 'condition'
    metrics_df.to_csv(output_dir / "metrics_summary.csv")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT RESULTS SUMMARY (Gated MIL Token-Level Hierarchical + DragonNet)")
    logger.info("=" * 80)
    logger.info("\n" + metrics_df.to_string())

    # Save config
    config_info = {
        'dataset': args.dataset,
        'feature_extractor_type': 'gated_mil_hierarchical',
        'model_type': 'dragonnet',
        'sentence_model': args.sentence_model,
        'num_confounders': args.num_confounders,
        'mil_hidden_dim': args.mil_hidden_dim,
        'token_hidden_dim': args.token_hidden_dim,
        'gated_mil_hierarchical': True,  # Token-level enabled
        'freeze_sentence_encoder': False,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'beta_targreg': args.beta_targreg,
        'stop_grad_propensity': args.stop_grad_propensity,
        'attention_entropy_weight': args.attention_entropy_weight,
        'use_mean_pooling': args.use_mean_pooling,
        'n_folds': args.n_folds,
        'device': str(device),
        'conditions_run': [c.name for c in conditions]
    }
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
