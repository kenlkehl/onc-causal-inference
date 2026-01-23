#!/usr/bin/env python
"""Task-Specific Multi-Head Aggregation experiment for confounder vectors.

This script tests the task-specific multi-head aggregation mechanism that
allows propensity and outcome tasks to weight confounders differently.

Key features tested:
- Task-specific aggregation: Different learnable queries for propensity vs outcome
- Dimensionality reduction: K*D -> 2*D (e.g., 2048 -> 512 with K=8, D=256)
- Patient-specific weighting: Attention computed from each patient's confounder values
- Interpretability: Which confounders each task finds most important

Conditions tested:
1. Oracle: patient_prompt (explicit confounder, upper bound)
2. Task-specific GRU: clinical_text with task-specific aggregation (new method)
3. Varying K: Different numbers of latent confounders (4, 8, 16)
4. LLM-extract-only: Extracted categorical features (MLP baseline)

Output:
- Per-condition predictions and metrics
- Task-specific confounder weight analysis
- Visualization of propensity vs outcome confounder preferences

Usage:
    python oracle_experiment_scripts/run_task_specific_aggregation_experiment.py \
        --dataset path/to/dataset.parquet \
        --output-dir results/task_specific_aggregation \
        --device cuda:0
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
from cdt.models.mlp_dragonnet import MLPDragonNet, CategoricalEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experimental condition."""
    name: str
    text_column: str
    num_latent_confounders: int = 8
    use_llm_extraction: bool = False
    llm_extraction_only: bool = False
    description: str = ""


# Define experimental conditions
EXPERIMENT_CONDITIONS = [
    ExperimentConfig(
        name="1_oracle",
        text_column="patient_prompt",
        num_latent_confounders=8,
        description="Oracle: patient_prompt with explicit confounders (upper bound)"
    ),
    ExperimentConfig(
        name="2_task_specific_K4",
        text_column="clinical_text",
        num_latent_confounders=4,
        description="Task-specific aggregation with K=4 latent confounders"
    ),
    ExperimentConfig(
        name="3_task_specific_K8",
        text_column="clinical_text",
        num_latent_confounders=8,
        description="Task-specific aggregation with K=8 latent confounders (default)"
    ),
    ExperimentConfig(
        name="4_task_specific_K16",
        text_column="clinical_text",
        num_latent_confounders=16,
        description="Task-specific aggregation with K=16 latent confounders"
    ),
    ExperimentConfig(
        name="5_llm_extract_only",
        text_column="clinical_text",  # Not used
        use_llm_extraction=True,
        llm_extraction_only=True,
        description="MLP baseline using LLM-extracted categorical features"
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


def analyze_task_weights(
    model: CausalText,
    texts: List[str],
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Analyze task-specific confounder weights learned by the model.

    Returns statistics about how propensity vs outcome tasks weight confounders.
    """
    model.eval()

    all_prop_weights = []
    all_out_weights = []

    with torch.no_grad():
        for i in range(0, min(len(texts), 100), batch_size):  # Sample up to 100 texts
            batch_texts = texts[i:i + batch_size]
            _, attention_info = model.feature_extractor(batch_texts, return_attention=True)

            for info in attention_info:
                if 'propensity_confounder_weights' in info:
                    all_prop_weights.append(info['propensity_confounder_weights'].numpy())
                if 'outcome_confounder_weights' in info:
                    all_out_weights.append(info['outcome_confounder_weights'].numpy())

    if not all_prop_weights:
        return {}

    prop_weights = np.stack(all_prop_weights)  # (N, K)
    out_weights = np.stack(all_out_weights)    # (N, K)

    # Compute statistics
    analysis = {
        'num_confounders': prop_weights.shape[1],
        'num_samples_analyzed': prop_weights.shape[0],

        # Mean weights per confounder (which confounders are most important on average)
        'propensity_mean_weights': prop_weights.mean(axis=0).tolist(),
        'outcome_mean_weights': out_weights.mean(axis=0).tolist(),

        # Std of weights (how much do weights vary across patients)
        'propensity_std_weights': prop_weights.std(axis=0).tolist(),
        'outcome_std_weights': out_weights.std(axis=0).tolist(),

        # Correlation between propensity and outcome weights (per sample)
        'weight_correlations': [
            stats.pearsonr(prop_weights[i], out_weights[i])[0]
            for i in range(prop_weights.shape[0])
        ],

        # Entropy of weight distributions (higher = more uniform, lower = more concentrated)
        'propensity_entropy_mean': float(np.mean([
            stats.entropy(w + 1e-10) for w in prop_weights
        ])),
        'outcome_entropy_mean': float(np.mean([
            stats.entropy(w + 1e-10) for w in out_weights
        ])),

        # Which confounders differ most between tasks
        'weight_difference_per_confounder': np.abs(
            prop_weights.mean(axis=0) - out_weights.mean(axis=0)
        ).tolist(),
    }

    # Summary: average correlation
    analysis['mean_weight_correlation'] = float(np.mean(analysis['weight_correlations']))

    return analysis


def analyze_attention_interpretations(
    model: CausalText,
    texts: List[str],
    output_dir: Path,
    top_k: int = 5,
    max_samples: int = 50
) -> Dict[str, Any]:
    """
    Generate and save detailed attention interpretation analysis.

    Analyzes:
    - Which sentences each confounder attends to (sentence-level weights)
    - How attention patterns differ across patients
    - Common sentence patterns attended by each confounder

    Args:
        model: Trained CausalText model with confounder extractor
        texts: List of texts to analyze
        output_dir: Directory to save interpretation files
        top_k: Number of top-attended sentences per confounder to report
        max_samples: Maximum number of samples to analyze

    Returns:
        Summary statistics about attention patterns
    """
    model.eval()

    # Limit samples for efficiency
    texts_to_analyze = texts[:max_samples]
    logger.info(f"  Analyzing attention interpretations on {len(texts_to_analyze)} texts...")

    # Get interpretations using the model's interpret_attention method
    interpretations = model.feature_extractor.interpret_attention(
        texts_to_analyze,
        top_k=top_k
    )

    # Save full interpretations as JSON
    interp_path = output_dir / "attention_interpretations.json"
    with open(interp_path, 'w') as f:
        json.dump(interpretations, f, indent=2, default=str)
    logger.info(f"    Saved attention interpretations to: {interp_path}")

    # Get detailed attention info including task-specific weights
    all_attention_data = []
    with torch.no_grad():
        batch_size = 8
        for i in range(0, len(texts_to_analyze), batch_size):
            batch_texts = texts_to_analyze[i:i + batch_size]
            _, attention_info = model.feature_extractor(batch_texts, return_attention=True)

            for j, info in enumerate(attention_info):
                sample_data = {
                    'sample_idx': i + j,
                    'text_preview': batch_texts[j][:200] + '...' if len(batch_texts[j]) > 200 else batch_texts[j],
                    'sentences': info.get('sentences', []),
                    'sentence_weights': info['sentence_weights'].numpy().tolist() if 'sentence_weights' in info else None,
                }

                if 'propensity_confounder_weights' in info:
                    sample_data['propensity_confounder_weights'] = info['propensity_confounder_weights'].numpy().tolist()
                if 'outcome_confounder_weights' in info:
                    sample_data['outcome_confounder_weights'] = info['outcome_confounder_weights'].numpy().tolist()

                all_attention_data.append(sample_data)

    # Save detailed attention data
    detailed_path = output_dir / "attention_detailed.json"
    with open(detailed_path, 'w') as f:
        json.dump(all_attention_data, f, indent=2, default=str)
    logger.info(f"    Saved detailed attention data to: {detailed_path}")

    # Aggregate patterns across documents
    confounder_patterns = {}
    for doc_interp in interpretations:
        for conf_name, attended_sentences in doc_interp.items():
            if conf_name not in confounder_patterns:
                confounder_patterns[conf_name] = []
            for sent_info in attended_sentences:
                confounder_patterns[conf_name].append({
                    'sentence': sent_info['sentence'],
                    'attention': sent_info['attention']
                })

    # Find most commonly attended sentence patterns per confounder
    pattern_summary = {}
    for conf_name, patterns in confounder_patterns.items():
        # Sort by attention weight
        sorted_patterns = sorted(patterns, key=lambda x: -x['attention'])

        # Get unique sentences with highest attention
        seen = set()
        unique_top = []
        for p in sorted_patterns:
            sent_key = p['sentence'][:50]  # Use first 50 chars as key
            if sent_key not in seen:
                unique_top.append(p)
                seen.add(sent_key)
            if len(unique_top) >= 10:
                break

        pattern_summary[conf_name] = {
            'num_attended_sentences': len(patterns),
            'mean_attention': float(np.mean([p['attention'] for p in patterns])),
            'max_attention': float(max(p['attention'] for p in patterns)) if patterns else 0,
            'top_sentences': unique_top[:5]
        }

    # Save pattern summary
    pattern_path = output_dir / "attention_patterns.json"
    with open(pattern_path, 'w') as f:
        json.dump(pattern_summary, f, indent=2, default=str)
    logger.info(f"    Saved attention patterns to: {pattern_path}")

    # Generate human-readable summary
    summary_lines = [
        "=" * 80,
        "ATTENTION INTERPRETATION SUMMARY",
        "=" * 80,
        f"\nAnalyzed {len(texts_to_analyze)} documents",
        f"Top {top_k} attended sentences shown per confounder\n",
    ]

    for conf_name in sorted(pattern_summary.keys()):
        info = pattern_summary[conf_name]
        summary_lines.append(f"\n--- {conf_name} ---")
        summary_lines.append(f"  Mean attention: {info['mean_attention']:.4f}")
        summary_lines.append(f"  Max attention: {info['max_attention']:.4f}")
        summary_lines.append(f"  Total attended sentences: {info['num_attended_sentences']}")
        summary_lines.append("  Top sentences:")
        for sent_info in info['top_sentences']:
            sent_preview = sent_info['sentence'][:80]
            summary_lines.append(f"    [{sent_info['attention']:.3f}] {sent_preview}...")

    summary_path = output_dir / "attention_interpretation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    logger.info(f"    Saved interpretation summary to: {summary_path}")

    # Return summary statistics
    return {
        'num_samples_analyzed': len(texts_to_analyze),
        'num_confounders': len(pattern_summary),
        'confounder_names': list(pattern_summary.keys()),
        'mean_attention_per_confounder': {
            name: info['mean_attention'] for name, info in pattern_summary.items()
        }
    }


def train_task_specific_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gamma_rlearner: float = 1.0,
    gru_embedding_dim: int = 128,
    gru_hidden_dim: int = 128,
    gru_num_layers: int = 1,
    gradient_accumulation_steps: int = 1
) -> Tuple[CausalText, List[Dict]]:
    """Train a GRU Hierarchical Confounder Extractor with task-specific aggregation."""
    text_column = config.text_column
    num_latent_confounders = config.num_latent_confounders

    # Get training texts for tokenizer fitting
    train_texts = train_df[text_column].tolist()

    # Create model with GRU-based Confounder Extractor (uses task-specific aggregation)
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
    logger.info(f"Using K={num_latent_confounders} latent confounders with task-specific aggregation")

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

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
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


def predict_task_specific(
    model: CausalText,
    df: pd.DataFrame,
    text_column: str,
    device: torch.device,
    batch_size: int
) -> Dict[str, np.ndarray]:
    """Generate predictions from task-specific aggregation model."""
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
    gru_embedding_dim: int = 128,
    gru_hidden_dim: int = 128,
    gru_num_layers: int = 1,
    encoder: Optional[CategoricalEncoder] = None,
    gradient_accumulation_steps: int = 1,
    analyze_weights: bool = True,
    analyze_attention: bool = True,
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, float], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run cross-validation for one experimental condition.

    Returns:
        results_df: Predictions DataFrame
        metrics: Evaluation metrics
        weight_analysis: Task-specific weight analysis (if analyze_weights=True)
        attention_analysis: Attention interpretation analysis (if analyze_attention=True)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running condition: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info(f"{'='*60}")

    df = df.reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_predictions = []
    all_weight_analyses = []

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
            weight_analysis = None
        else:
            # Task-specific aggregation model
            model, _ = train_task_specific_model(
                train_df, test_df, config, device,
                epochs, batch_size, learning_rate,
                gamma_rlearner=gamma_rlearner,
                gru_embedding_dim=gru_embedding_dim,
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            preds = predict_task_specific(
                model, test_df, config.text_column, device, batch_size
            )

            # Analyze task-specific weights
            if analyze_weights:
                weight_analysis = analyze_task_weights(
                    model,
                    test_df[config.text_column].tolist(),
                    batch_size=batch_size
                )
                weight_analysis['fold'] = fold + 1
                all_weight_analyses.append(weight_analysis)

                logger.info(f"    Weight analysis: mean correlation between tasks = {weight_analysis.get('mean_weight_correlation', 'N/A'):.3f}")

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
        true_y1=results_df['true_y1_prob'].values,
        true_outcome=results_df['outcome_indicator'].values
    )

    logger.info(f"  Results for {config.name}:")
    logger.info(f"    ITE MSE: {metrics['ite_mse']:.4f}")
    logger.info(f"    ITE MAE: {metrics['ite_mae']:.4f}")
    logger.info(f"    ITE Correlation: {metrics['ite_corr']:.4f}")
    logger.info(f"    ATE Bias: {metrics['ate_bias']:.4f}")
    logger.info(f"    Propensity AUROC: {metrics['propensity_auroc']:.4f}")
    logger.info(f"    Y0 AUROC (T=0): {metrics['y0_auroc']:.4f}")
    logger.info(f"    Y1 AUROC (T=1): {metrics['y1_auroc']:.4f}")

    # Aggregate weight analyses across folds
    combined_weight_analysis = None
    if all_weight_analyses:
        combined_weight_analysis = {
            'num_confounders': all_weight_analyses[0]['num_confounders'],
            'mean_weight_correlation': float(np.mean([
                wa['mean_weight_correlation'] for wa in all_weight_analyses
            ])),
            'propensity_entropy_mean': float(np.mean([
                wa['propensity_entropy_mean'] for wa in all_weight_analyses
            ])),
            'outcome_entropy_mean': float(np.mean([
                wa['outcome_entropy_mean'] for wa in all_weight_analyses
            ])),
            'per_fold_analyses': all_weight_analyses
        }

    # Attention interpretation analysis (train one model on last fold for interpretation)
    attention_analysis = None
    if analyze_attention and not config.llm_extraction_only and output_dir is not None:
        logger.info("  Running attention interpretation analysis...")

        # Use last fold's train/test split
        splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=42).split(df))
        last_fold = n_folds - 1
        train_idx, test_idx = splits[last_fold]
        train_df = df.iloc[train_idx]
        val_df = df.iloc[test_idx]

        # Train a model for interpretation
        model, _ = train_task_specific_model(
            train_df, val_df, config, device,
            epochs, batch_size, learning_rate,
            gamma_rlearner=gamma_rlearner,
            gru_embedding_dim=gru_embedding_dim,
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Create condition output directory
        condition_output_dir = output_dir / config.name
        condition_output_dir.mkdir(parents=True, exist_ok=True)

        # Analyze attention patterns
        attention_analysis = analyze_attention_interpretations(
            model,
            val_df[config.text_column].tolist(),
            condition_output_dir,
            top_k=5,
            max_samples=50
        )

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results_df, metrics, combined_weight_analysis, attention_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Test task-specific multi-head aggregation for confounder vectors"
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
        default="results/task_specific_aggregation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, cpu, etc.)"
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
        help="Specific conditions to run (e.g., 1_oracle 3_task_specific_K8)"
    )
    parser.add_argument(
        "--skip-weight-analysis",
        action="store_true",
        help="Skip task-specific weight analysis (faster)"
    )
    parser.add_argument(
        "--skip-attention-analysis",
        action="store_true",
        help="Skip attention interpretation analysis (faster, less interpretability output)"
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
    all_weight_analyses = {}
    all_attention_analyses = {}

    for config in conditions:
        try:
            results_df, metrics, weight_analysis, attention_analysis = run_condition(
                df, config, device,
                n_folds=args.n_folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gamma_rlearner=args.gamma_rlearner,
                gru_embedding_dim=args.gru_embedding_dim,
                gru_hidden_dim=args.gru_hidden_dim,
                gru_num_layers=args.gru_num_layers,
                encoder=encoder,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                analyze_weights=not args.skip_weight_analysis,
                analyze_attention=not args.skip_attention_analysis,
                output_dir=output_dir
            )

            all_metrics[config.name] = metrics
            all_predictions[config.name] = results_df
            if weight_analysis:
                all_weight_analyses[config.name] = weight_analysis
            if attention_analysis:
                all_attention_analyses[config.name] = attention_analysis

            # Save condition results
            condition_dir = output_dir / config.name
            condition_dir.mkdir(exist_ok=True)
            results_df.to_parquet(condition_dir / "predictions.parquet", index=False)

            if weight_analysis:
                with open(condition_dir / "task_weight_analysis.json", 'w') as f:
                    json.dump(weight_analysis, f, indent=2, default=str)

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
        logger.info("EXPERIMENT RESULTS SUMMARY (Task-Specific Aggregation)")
        logger.info("=" * 80)
        logger.info("\n" + metrics_df.to_string())

    # Save weight analysis summary
    if all_weight_analyses:
        weight_summary = {
            condition: {
                'num_confounders': wa['num_confounders'],
                'mean_weight_correlation': wa['mean_weight_correlation'],
                'propensity_entropy': wa['propensity_entropy_mean'],
                'outcome_entropy': wa['outcome_entropy_mean']
            }
            for condition, wa in all_weight_analyses.items()
        }

        with open(output_dir / "weight_analysis_summary.json", 'w') as f:
            json.dump(weight_summary, f, indent=2)

        logger.info("\n" + "=" * 80)
        logger.info("TASK-SPECIFIC WEIGHT ANALYSIS SUMMARY")
        logger.info("=" * 80)
        for condition, summary in weight_summary.items():
            logger.info(f"  {condition}:")
            logger.info(f"    K={summary['num_confounders']}, "
                       f"task correlation={summary['mean_weight_correlation']:.3f}, "
                       f"prop_entropy={summary['propensity_entropy']:.3f}, "
                       f"out_entropy={summary['outcome_entropy']:.3f}")

    # Save attention analysis summary
    if all_attention_analyses:
        with open(output_dir / "attention_analysis_summary.json", 'w') as f:
            json.dump(all_attention_analyses, f, indent=2, default=str)

        logger.info("\n" + "=" * 80)
        logger.info("ATTENTION INTERPRETATION SUMMARY")
        logger.info("=" * 80)
        for condition, analysis in all_attention_analyses.items():
            logger.info(f"  {condition}:")
            logger.info(f"    Samples analyzed: {analysis['num_samples_analyzed']}")
            logger.info(f"    Num confounders: {analysis['num_confounders']}")
            if 'mean_attention_per_confounder' in analysis:
                for conf, attn in analysis['mean_attention_per_confounder'].items():
                    logger.info(f"      {conf}: mean_attn={attn:.4f}")

    # Save config
    config_info = {
        'dataset': args.dataset,
        'model_type': 'task_specific_aggregation',
        'gru_embedding_dim': args.gru_embedding_dim,
        'gru_hidden_dim': args.gru_hidden_dim,
        'gru_num_layers': args.gru_num_layers,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'gamma_rlearner': args.gamma_rlearner,
        'n_folds': args.n_folds,
        'device': str(device),
        'conditions_run': [c.name for c in conditions],
        'analyze_weights': not args.skip_weight_analysis,
        'analyze_attention': not args.skip_attention_analysis
    }
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
