#!/usr/bin/env python
"""3-Way Task-Specific Aggregation experiment comparing DragonNet vs R-Learner.

This script tests the 3-way task-specific aggregation mechanism that allows
different prediction tasks to weight confounders differently:

- R-Learner mode: propensity, outcome m(X), and treatment effect tau(X) queries
- DragonNet mode: propensity, Y0, and Y1 queries

Key features tested:
- 3-way aggregation: Different learnable queries for propensity vs outcome vs treatment effect
- Dimensionality reduction: K*D -> 3*D (e.g., 1024 -> 384 with K=8, D=128)
- Patient-specific weighting: Attention computed from each patient's confounder values
- Model comparison: R-Learner (direct tau) vs DragonNet (y0/y1 potential outcomes)
- Interpretability: Which confounders each task finds most important

Conditions tested:
1. Oracle R-Learner: patient_prompt with explicit confounders (upper bound)
2. Oracle DragonNet: patient_prompt with DragonNet architecture
3-4. R-Learner with 3-way aggregation (K=4, K=8)
5-6. DragonNet with 3-way aggregation (K=4, K=8)
7. LLM-extract-only: Extracted categorical features (MLP baseline)

Output:
- Per-condition predictions and metrics
- 3-way task-specific confounder weight analysis
- Comparison of prognostic vs effect-modifying confounders
- Visualization of propensity vs outcome vs tau/y1 confounder preferences

Usage:
    python oracle_experiment_scripts/run_all_tasks_specific_aggregation_experiment.py \
        --dataset path/to/dataset.parquet \
        --output-dir results/3way_aggregation \
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
    model_type: str = "rlearner"  # "rlearner" or "dragonnet"
    num_latent_confounders: int = 8
    use_llm_extraction: bool = False
    llm_extraction_only: bool = False
    description: str = ""


# Define experimental conditions - comparing R-Learner vs DragonNet with 3-way aggregation
EXPERIMENT_CONDITIONS = [
    # Oracle baselines (use patient_prompt with explicit confounders)
    ExperimentConfig(
        name="1_oracle_rlearner",
        text_column="patient_prompt",
        model_type="rlearner",
        num_latent_confounders=8,
        description="Oracle R-Learner: patient_prompt with propensity/outcome/tau queries"
    ),
    ExperimentConfig(
        name="2_oracle_dragonnet",
        text_column="patient_prompt",
        model_type="dragonnet",
        num_latent_confounders=8,
        description="Oracle DragonNet: patient_prompt with propensity/y0/y1 queries"
    ),

    # R-Learner with 3-way aggregation (propensity + outcome + tau)
    ExperimentConfig(
        name="3_rlearner_K4",
        text_column="clinical_text",
        model_type="rlearner",
        num_latent_confounders=4,
        description="R-Learner 3-way aggregation with K=4 latent confounders"
    ),
    ExperimentConfig(
        name="4_rlearner_K8",
        text_column="clinical_text",
        model_type="rlearner",
        num_latent_confounders=8,
        description="R-Learner 3-way aggregation with K=8 latent confounders (default)"
    ),

    # DragonNet with 3-way aggregation (propensity + y0 + y1)
    ExperimentConfig(
        name="5_dragonnet_K4",
        text_column="clinical_text",
        model_type="dragonnet",
        num_latent_confounders=4,
        description="DragonNet 3-way aggregation with K=4 latent confounders"
    ),
    ExperimentConfig(
        name="6_dragonnet_K8",
        text_column="clinical_text",
        model_type="dragonnet",
        num_latent_confounders=8,
        description="DragonNet 3-way aggregation with K=8 latent confounders"
    ),

    # MLP baseline using LLM-extracted features
    ExperimentConfig(
        name="7_llm_extract_only",
        text_column="clinical_text",  # Not used
        model_type="dragonnet",  # MLP uses DragonNet head
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


def analyze_task_weights_3way(
    model: CausalText,
    texts: List[str],
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Analyze 3-way task-specific confounder weights learned by the model.

    For R-Learner: propensity, outcome, tau weights
    For DragonNet: propensity, y0, y1 weights

    Returns statistics about how different tasks weight confounders.
    """
    model.eval()
    model_type = model.model_type

    all_prop_weights = []
    all_weight2 = []  # outcome (R-Learner) or y0 (DragonNet)
    all_weight3 = []  # tau (R-Learner) or y1 (DragonNet)

    with torch.no_grad():
        for i in range(0, min(len(texts), 100), batch_size):
            batch_texts = texts[i:i + batch_size]
            _, attention_info = model.feature_extractor(batch_texts, return_attention=True)

            for info in attention_info:
                if 'propensity_confounder_weights' in info:
                    all_prop_weights.append(info['propensity_confounder_weights'].numpy())

                if model_type == "rlearner":
                    if 'outcome_confounder_weights' in info:
                        all_weight2.append(info['outcome_confounder_weights'].numpy())
                    if 'tau_confounder_weights' in info:
                        all_weight3.append(info['tau_confounder_weights'].numpy())
                else:  # dragonnet
                    if 'y0_confounder_weights' in info:
                        all_weight2.append(info['y0_confounder_weights'].numpy())
                    if 'y1_confounder_weights' in info:
                        all_weight3.append(info['y1_confounder_weights'].numpy())

    if not all_prop_weights or not all_weight2 or not all_weight3:
        return {}

    prop_weights = np.stack(all_prop_weights)  # (N, K)
    weight2 = np.stack(all_weight2)            # (N, K)
    weight3 = np.stack(all_weight3)            # (N, K)

    # Name the weights based on model type
    if model_type == "rlearner":
        weight2_name = "outcome"
        weight3_name = "tau"
    else:
        weight2_name = "y0"
        weight3_name = "y1"

    # Compute per-sample correlations between all pairs
    prop_weight2_corrs = []
    prop_weight3_corrs = []
    weight2_weight3_corrs = []

    for i in range(prop_weights.shape[0]):
        c1, _ = stats.pearsonr(prop_weights[i], weight2[i])
        c2, _ = stats.pearsonr(prop_weights[i], weight3[i])
        c3, _ = stats.pearsonr(weight2[i], weight3[i])
        prop_weight2_corrs.append(c1)
        prop_weight3_corrs.append(c2)
        weight2_weight3_corrs.append(c3)

    # Build analysis dict
    analysis = {
        'model_type': model_type,
        'num_confounders': prop_weights.shape[1],
        'num_samples_analyzed': prop_weights.shape[0],

        # Mean weights per confounder
        'propensity_mean_weights': prop_weights.mean(axis=0).tolist(),
        f'{weight2_name}_mean_weights': weight2.mean(axis=0).tolist(),
        f'{weight3_name}_mean_weights': weight3.mean(axis=0).tolist(),

        # Std of weights
        'propensity_std_weights': prop_weights.std(axis=0).tolist(),
        f'{weight2_name}_std_weights': weight2.std(axis=0).tolist(),
        f'{weight3_name}_std_weights': weight3.std(axis=0).tolist(),

        # 3-way correlations (per sample)
        f'prop_{weight2_name}_correlations': prop_weight2_corrs,
        f'prop_{weight3_name}_correlations': prop_weight3_corrs,
        f'{weight2_name}_{weight3_name}_correlations': weight2_weight3_corrs,

        # Mean correlations
        f'mean_prop_{weight2_name}_correlation': float(np.nanmean(prop_weight2_corrs)),
        f'mean_prop_{weight3_name}_correlation': float(np.nanmean(prop_weight3_corrs)),
        f'mean_{weight2_name}_{weight3_name}_correlation': float(np.nanmean(weight2_weight3_corrs)),

        # Entropy of weight distributions
        'propensity_entropy_mean': float(np.mean([
            stats.entropy(w + 1e-10) for w in prop_weights
        ])),
        f'{weight2_name}_entropy_mean': float(np.mean([
            stats.entropy(w + 1e-10) for w in weight2
        ])),
        f'{weight3_name}_entropy_mean': float(np.mean([
            stats.entropy(w + 1e-10) for w in weight3
        ])),

        # Which confounders differ most between tasks
        f'prop_{weight2_name}_diff_per_confounder': np.abs(
            prop_weights.mean(axis=0) - weight2.mean(axis=0)
        ).tolist(),
        f'prop_{weight3_name}_diff_per_confounder': np.abs(
            prop_weights.mean(axis=0) - weight3.mean(axis=0)
        ).tolist(),
        f'{weight2_name}_{weight3_name}_diff_per_confounder': np.abs(
            weight2.mean(axis=0) - weight3.mean(axis=0)
        ).tolist(),
    }

    # Identify effect modifier confounders
    # For R-Learner: high tau weight but different from outcome weight
    # For DragonNet: different y0 vs y1 weights
    if model_type == "rlearner":
        # Confounders with high tau weight relative to outcome
        tau_vs_outcome = weight3.mean(axis=0) - weight2.mean(axis=0)
        analysis['effect_modifier_scores'] = tau_vs_outcome.tolist()
        analysis['top_effect_modifiers'] = np.argsort(-np.abs(tau_vs_outcome))[:3].tolist()
    else:
        # Confounders with different y0 vs y1 weights
        y1_vs_y0 = weight3.mean(axis=0) - weight2.mean(axis=0)
        analysis['treatment_modifier_scores'] = y1_vs_y0.tolist()
        analysis['top_treatment_modifiers'] = np.argsort(-np.abs(y1_vs_y0))[:3].tolist()

    # Identify prognostic confounders (high outcome/y0 weight, low propensity weight)
    prognostic_scores = weight2.mean(axis=0) - prop_weights.mean(axis=0)
    analysis['prognostic_scores'] = prognostic_scores.tolist()
    analysis['top_prognostic'] = np.argsort(-prognostic_scores)[:3].tolist()

    # Identify confounding confounders (high propensity weight)
    analysis['top_confounders'] = np.argsort(-prop_weights.mean(axis=0))[:3].tolist()

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
    - Which sentences each confounder attends to
    - 3-way task-specific weights (propensity vs outcome/y0 vs tau/y1)
    - Common sentence patterns attended by each confounder

    Args:
        model: Trained CausalText model with confounder extractor
        texts: List of texts to analyze
        output_dir: Directory to save interpretation files
        top_k: Number of top-attended sentences per confounder
        max_samples: Maximum number of samples to analyze

    Returns:
        Summary statistics about attention patterns
    """
    model.eval()
    model_type = model.model_type

    texts_to_analyze = texts[:max_samples]
    logger.info(f"  Analyzing attention interpretations on {len(texts_to_analyze)} texts...")

    # Get interpretations
    interpretations = model.feature_extractor.interpret_attention(
        texts_to_analyze,
        top_k=top_k
    )

    # Save full interpretations
    interp_path = output_dir / "attention_interpretations.json"
    with open(interp_path, 'w') as f:
        json.dump(interpretations, f, indent=2, default=str)
    logger.info(f"    Saved attention interpretations to: {interp_path}")

    # Get detailed attention info including 3-way task-specific weights
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

                # Add propensity weights
                if 'propensity_confounder_weights' in info:
                    sample_data['propensity_confounder_weights'] = info['propensity_confounder_weights'].numpy().tolist()

                # Add model-type-specific weights
                if model_type == "rlearner":
                    if 'outcome_confounder_weights' in info:
                        sample_data['outcome_confounder_weights'] = info['outcome_confounder_weights'].numpy().tolist()
                    if 'tau_confounder_weights' in info:
                        sample_data['tau_confounder_weights'] = info['tau_confounder_weights'].numpy().tolist()
                else:  # dragonnet
                    if 'y0_confounder_weights' in info:
                        sample_data['y0_confounder_weights'] = info['y0_confounder_weights'].numpy().tolist()
                    if 'y1_confounder_weights' in info:
                        sample_data['y1_confounder_weights'] = info['y1_confounder_weights'].numpy().tolist()

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
        sorted_patterns = sorted(patterns, key=lambda x: -x['attention'])

        seen = set()
        unique_top = []
        for p in sorted_patterns:
            sent_key = p['sentence'][:50]
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
        f"ATTENTION INTERPRETATION SUMMARY ({model_type.upper()} mode)",
        "=" * 80,
        f"\nAnalyzed {len(texts_to_analyze)} documents",
        f"Top {top_k} attended sentences shown per confounder",
        f"\n3-way aggregation: propensity + {'outcome/tau' if model_type == 'rlearner' else 'y0/y1'}\n",
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

    return {
        'model_type': model_type,
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
    beta_targreg: float = 0.1,
    gru_embedding_dim: int = 128,
    gru_hidden_dim: int = 128,
    gru_num_layers: int = 1,
    gradient_accumulation_steps: int = 1
) -> Tuple[CausalText, List[Dict]]:
    """Train a GRU Hierarchical Confounder Extractor with 3-way task-specific aggregation."""
    text_column = config.text_column
    num_latent_confounders = config.num_latent_confounders
    model_type = config.model_type

    train_texts = train_df[text_column].tolist()

    # Create model with specified model_type (R-Learner or DragonNet)
    model = CausalText(
        feature_extractor_type="confounder",
        model_type=model_type,  # "rlearner" or "dragonnet"
        # Enable GRU-based confounder extractor
        confounder_use_gru=True,
        confounder_gru_embedding_dim=gru_embedding_dim,
        confounder_gru_hidden_dim=gru_hidden_dim,
        confounder_gru_num_layers=gru_num_layers,
        confounder_gru_bidirectional=True,
        confounder_gru_dropout=0.1,
        confounder_gru_min_word_freq=1,
        confounder_gru_max_sentence_length=128,
        # Confounder architecture
        confounder_num_latents=num_latent_confounders,
        confounder_value_dim=128,
        confounder_max_sentences=100,
        confounder_num_heads=4,
        # Sparse attention
        confounder_sparse_attention=True,
        confounder_sparse_method="entmax",
        confounder_sparse_alpha=1.5,
        confounder_dropout=0.1,
        # Causal head
        dragonnet_representation_dim=128,
        dragonnet_hidden_outcome_dim=64,
        dragonnet_dropout=0.2,
        device=str(device)
    )

    # GRU mode requires tokenizer fitting
    model.fit_tokenizer(train_texts)
    logger.info(f"Fitted tokenizer on {len(train_texts)} texts")
    logger.info(f"Model type: {model_type}, K={num_latent_confounders} latent confounders (3-way aggregation)")

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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Scheduler
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
        model.train()
        train_loss = 0.0
        train_causal_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
            batch['treatment'] = batch['treatment'].to(device)
            batch['outcome'] = batch['outcome'].to(device)

            if model_type == "rlearner":
                losses = model.train_step(
                    batch,
                    alpha_propensity=1.0,
                    gamma_rlearner=gamma_rlearner
                )
                causal_loss = losses.get('r_loss', 0.0)
            else:  # dragonnet
                losses = model.train_step(
                    batch,
                    alpha_propensity=1.0,
                    beta_targreg=beta_targreg
                )
                causal_loss = losses.get('targreg_loss', 0.0)

            loss = losses['loss'] / gradient_accumulation_steps
            loss.backward()

            train_loss += losses['loss'].item()
            train_causal_loss += causal_loss if isinstance(causal_loss, float) else causal_loss.item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Validate
        model.eval()
        val_loss = 0.0
        val_causal_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch['treatment'] = batch['treatment'].to(device)
                batch['outcome'] = batch['outcome'].to(device)

                if model_type == "rlearner":
                    losses = model.train_step(
                        batch,
                        alpha_propensity=1.0,
                        gamma_rlearner=gamma_rlearner
                    )
                    causal_loss = losses.get('r_loss', 0.0)
                else:
                    losses = model.train_step(
                        batch,
                        alpha_propensity=1.0,
                        beta_targreg=beta_targreg
                    )
                    causal_loss = losses.get('targreg_loss', 0.0)

                val_loss += losses['loss'].item()
                val_causal_loss += causal_loss if isinstance(causal_loss, float) else causal_loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_causal_loss /= len(train_loader)
        val_causal_loss /= len(val_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_causal_loss': train_causal_loss,
            'val_causal_loss': val_causal_loss,
            'lr': scheduler.get_last_lr()[0]
        })

        causal_name = 'r_loss' if model_type == 'rlearner' else 'targreg_loss'
        logger.info(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, {causal_name}={val_causal_loss:.4f}")

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
            if 'tau_pred' in preds:
                all_tau.append(preds['tau_pred'].cpu().numpy())

    result = {
        'y0_prob': np.concatenate(all_y0),
        'y1_prob': np.concatenate(all_y1),
        'propensity': np.concatenate(all_prop),
        'ite_prob': np.concatenate(all_y1) - np.concatenate(all_y0)
    }

    if all_tau:
        result['tau_pred'] = np.concatenate(all_tau)

    return result


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
    beta_targreg: float = 0.1,
    gru_embedding_dim: int = 128,
    gru_hidden_dim: int = 128,
    gru_num_layers: int = 1,
    encoder: Optional[CategoricalEncoder] = None,
    gradient_accumulation_steps: int = 1,
    analyze_weights: bool = True,
    analyze_attention: bool = True,
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, float], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run cross-validation for one experimental condition with 3-way aggregation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running condition: {config.name}")
    logger.info(f"Model type: {config.model_type}")
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
            model, _ = train_mlp_model(
                train_df, test_df, encoder, device,
                epochs, batch_size * 4, learning_rate * 10
            )
            preds = predict_mlp(model, test_df, encoder, device, batch_size * 4)
            weight_analysis = None
        else:
            model, _ = train_task_specific_model(
                train_df, test_df, config, device,
                epochs, batch_size, learning_rate,
                gamma_rlearner=gamma_rlearner,
                beta_targreg=beta_targreg,
                gru_embedding_dim=gru_embedding_dim,
                gru_hidden_dim=gru_hidden_dim,
                gru_num_layers=gru_num_layers,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            preds = predict_task_specific(
                model, test_df, config.text_column, device, batch_size
            )

            # Analyze 3-way task-specific weights
            if analyze_weights:
                weight_analysis = analyze_task_weights_3way(
                    model,
                    test_df[config.text_column].tolist(),
                    batch_size=batch_size
                )
                weight_analysis['fold'] = fold + 1
                all_weight_analyses.append(weight_analysis)

                # Log key correlations based on model type
                if config.model_type == "rlearner":
                    logger.info(f"    3-way correlations: prop-outcome={weight_analysis.get('mean_prop_outcome_correlation', 'N/A'):.3f}, "
                               f"prop-tau={weight_analysis.get('mean_prop_tau_correlation', 'N/A'):.3f}, "
                               f"outcome-tau={weight_analysis.get('mean_outcome_tau_correlation', 'N/A'):.3f}")
                else:
                    logger.info(f"    3-way correlations: prop-y0={weight_analysis.get('mean_prop_y0_correlation', 'N/A'):.3f}, "
                               f"prop-y1={weight_analysis.get('mean_prop_y1_correlation', 'N/A'):.3f}, "
                               f"y0-y1={weight_analysis.get('mean_y0_y1_correlation', 'N/A'):.3f}")

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

    logger.info(f"  Results for {config.name} ({config.model_type}):")
    logger.info(f"    ITE MSE: {metrics['ite_mse']:.4f}")
    logger.info(f"    ITE MAE: {metrics['ite_mae']:.4f}")
    logger.info(f"    ITE Correlation: {metrics['ite_corr']:.4f}")
    logger.info(f"    ATE Bias: {metrics['ate_bias']:.4f}")
    logger.info(f"    Propensity AUROC: {metrics['propensity_auroc']:.4f}")
    logger.info(f"    Y0 AUROC (T=0): {metrics['y0_auroc']:.4f}")
    logger.info(f"    Y1 AUROC (T=1): {metrics['y1_auroc']:.4f}")

    # Aggregate weight analyses
    combined_weight_analysis = None
    if all_weight_analyses:
        first_wa = all_weight_analyses[0]
        combined_weight_analysis = {
            'model_type': config.model_type,
            'num_confounders': first_wa.get('num_confounders'),
            'per_fold_analyses': all_weight_analyses
        }

        # Aggregate 3-way correlations based on model type
        if config.model_type == "rlearner":
            combined_weight_analysis['mean_prop_outcome_correlation'] = float(np.mean([
                wa.get('mean_prop_outcome_correlation', 0) for wa in all_weight_analyses
            ]))
            combined_weight_analysis['mean_prop_tau_correlation'] = float(np.mean([
                wa.get('mean_prop_tau_correlation', 0) for wa in all_weight_analyses
            ]))
            combined_weight_analysis['mean_outcome_tau_correlation'] = float(np.mean([
                wa.get('mean_outcome_tau_correlation', 0) for wa in all_weight_analyses
            ]))
        else:
            combined_weight_analysis['mean_prop_y0_correlation'] = float(np.mean([
                wa.get('mean_prop_y0_correlation', 0) for wa in all_weight_analyses
            ]))
            combined_weight_analysis['mean_prop_y1_correlation'] = float(np.mean([
                wa.get('mean_prop_y1_correlation', 0) for wa in all_weight_analyses
            ]))
            combined_weight_analysis['mean_y0_y1_correlation'] = float(np.mean([
                wa.get('mean_y0_y1_correlation', 0) for wa in all_weight_analyses
            ]))

        # Aggregate entropy
        combined_weight_analysis['propensity_entropy_mean'] = float(np.mean([
            wa.get('propensity_entropy_mean', 0) for wa in all_weight_analyses
        ]))

    # Attention interpretation analysis
    attention_analysis = None
    if analyze_attention and not config.llm_extraction_only and output_dir is not None:
        logger.info("  Running attention interpretation analysis...")

        splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=42).split(df))
        last_fold = n_folds - 1
        train_idx, test_idx = splits[last_fold]
        train_df = df.iloc[train_idx]
        val_df = df.iloc[test_idx]

        model, _ = train_task_specific_model(
            train_df, val_df, config, device,
            epochs, batch_size, learning_rate,
            gamma_rlearner=gamma_rlearner,
            beta_targreg=beta_targreg,
            gru_embedding_dim=gru_embedding_dim,
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        condition_output_dir = output_dir / config.name
        condition_output_dir.mkdir(parents=True, exist_ok=True)

        attention_analysis = analyze_attention_interpretations(
            model,
            val_df[config.text_column].tolist(),
            condition_output_dir,
            top_k=5,
            max_samples=50
        )

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    return results_df, metrics, combined_weight_analysis, attention_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Test 3-way task-specific aggregation comparing R-Learner vs DragonNet"
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
        default="results/3way_aggregation",
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
        help="Gradient accumulation steps"
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
        "--beta-targreg",
        type=float,
        default=0.1,
        help="Weight for DragonNet targeted regularization"
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
        help="Specific conditions to run"
    )
    parser.add_argument(
        "--skip-weight-analysis",
        action="store_true",
        help="Skip 3-way task-specific weight analysis"
    )
    parser.add_argument(
        "--skip-attention-analysis",
        action="store_true",
        help="Skip attention interpretation analysis"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Testing 3-way task-specific aggregation")
    logger.info(f"  R-Learner: propensity + outcome + tau queries")
    logger.info(f"  DragonNet: propensity + y0 + y1 queries")

    # Load dataset
    df = pd.read_parquet(args.dataset)
    logger.info(f"Loaded {len(df)} samples from {args.dataset}")

    # Check required columns
    required_cols = ['clinical_text', 'patient_prompt', 'treatment_indicator',
                     'outcome_indicator', 'true_ite_prob', 'true_y0_prob', 'true_y1_prob']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check for LLM extraction
    has_llm_extraction = 'llm_extracted_metastatic_sites' in df.columns

    encoder = None
    if has_llm_extraction:
        encoder = CategoricalEncoder(categories=["1", "2", "3", "4_or_more"])
        encoder.fit(df['llm_extracted_metastatic_sites'].tolist())

    # Filter conditions
    conditions = EXPERIMENT_CONDITIONS
    if args.conditions:
        conditions = [c for c in conditions if c.name in args.conditions]
        logger.info(f"Running {len(conditions)} selected conditions: {[c.name for c in conditions]}")

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
                beta_targreg=args.beta_targreg,
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

        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT RESULTS SUMMARY (3-Way Task-Specific Aggregation)")
        logger.info("=" * 80)
        logger.info("\n" + metrics_df.to_string())

    # Save 3-way weight analysis summary
    if all_weight_analyses:
        weight_summary = {}
        for condition, wa in all_weight_analyses.items():
            summary = {
                'model_type': wa.get('model_type'),
                'num_confounders': wa.get('num_confounders'),
                'propensity_entropy': wa.get('propensity_entropy_mean')
            }
            # Add model-specific correlations
            if wa.get('model_type') == 'rlearner':
                summary['prop_outcome_corr'] = wa.get('mean_prop_outcome_correlation')
                summary['prop_tau_corr'] = wa.get('mean_prop_tau_correlation')
                summary['outcome_tau_corr'] = wa.get('mean_outcome_tau_correlation')
            else:
                summary['prop_y0_corr'] = wa.get('mean_prop_y0_correlation')
                summary['prop_y1_corr'] = wa.get('mean_prop_y1_correlation')
                summary['y0_y1_corr'] = wa.get('mean_y0_y1_correlation')
            weight_summary[condition] = summary

        with open(output_dir / "weight_analysis_summary.json", 'w') as f:
            json.dump(weight_summary, f, indent=2)

        logger.info("\n" + "=" * 80)
        logger.info("3-WAY TASK-SPECIFIC WEIGHT ANALYSIS SUMMARY")
        logger.info("=" * 80)
        for condition, summary in weight_summary.items():
            model_type = summary.get('model_type', 'unknown')
            logger.info(f"  {condition} ({model_type}):")
            if model_type == 'rlearner':
                logger.info(f"    K={summary['num_confounders']}, "
                           f"prop-out={summary.get('prop_outcome_corr', 'N/A'):.3f}, "
                           f"prop-tau={summary.get('prop_tau_corr', 'N/A'):.3f}, "
                           f"out-tau={summary.get('outcome_tau_corr', 'N/A'):.3f}")
            else:
                logger.info(f"    K={summary['num_confounders']}, "
                           f"prop-y0={summary.get('prop_y0_corr', 'N/A'):.3f}, "
                           f"prop-y1={summary.get('prop_y1_corr', 'N/A'):.3f}, "
                           f"y0-y1={summary.get('y0_y1_corr', 'N/A'):.3f}")

    # Save attention analysis summary
    if all_attention_analyses:
        with open(output_dir / "attention_analysis_summary.json", 'w') as f:
            json.dump(all_attention_analyses, f, indent=2, default=str)

        logger.info("\n" + "=" * 80)
        logger.info("ATTENTION INTERPRETATION SUMMARY")
        logger.info("=" * 80)
        for condition, analysis in all_attention_analyses.items():
            logger.info(f"  {condition} ({analysis.get('model_type', 'unknown')}):")
            logger.info(f"    Samples analyzed: {analysis['num_samples_analyzed']}")
            logger.info(f"    Num confounders: {analysis['num_confounders']}")

    # Save experiment config
    config_info = {
        'dataset': args.dataset,
        'experiment_type': '3way_task_specific_aggregation',
        'model_types_tested': ['rlearner', 'dragonnet'],
        'aggregation_queries': {
            'rlearner': ['propensity', 'outcome', 'tau'],
            'dragonnet': ['propensity', 'y0', 'y1']
        },
        'gru_embedding_dim': args.gru_embedding_dim,
        'gru_hidden_dim': args.gru_hidden_dim,
        'gru_num_layers': args.gru_num_layers,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'gamma_rlearner': args.gamma_rlearner,
        'beta_targreg': args.beta_targreg,
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
