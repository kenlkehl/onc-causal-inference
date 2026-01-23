#!/usr/bin/env python
"""Concept-aware CNN filter experiment for synthetic clinical text.

This script runs 8 experimental conditions to evaluate different strategies
for ITE estimation on synthetic clinical text:

End-to-End CNN Approaches:
1. Oracle: patient_prompt with k-means filters (upper bound)
2. Baseline: clinical_text with random filters only
3. K-means: clinical_text with k-means filters
4. Concept-Aware: clinical_text with semantic concept filters
5. Concept + K-means: clinical_text with both semantic and k-means filters

Two-Stage LLM Extraction Approaches:
6. LLM-Extract-Only: Train on extracted categorical features only
7. LLM-Extract-Combined: Text features + extracted features (hybrid)
8. LLM-Extract-as-Text: Convert extraction to structured text like patient_prompt

Usage:
    python oracle_experiment_scripts/run_concept_aware_experiment.py \
        --dataset ../pcori_experiments/explicit_confounder_experiments_1-19-26/dataset_with_extraction.parquet \
        --output-dir rlearner_results/concept_aware_experiment \
        --device cuda:0 \
        --epochs 50
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

from cdt.models.causal_cnn import CausalCNNText
from cdt.models.mlp_dragonnet import MLPDragonNet, CategoricalEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Semantic concepts for metastatic site count
METASTATIC_SITE_CONCEPTS = {
    "3": [  # 3-gram concepts
        "single metastatic site",
        "one metastatic lesion",
        "solitary bone metastasis",
        "isolated liver metastasis",
        "oligometastatic disease noted",
        "limited metastatic burden",
        "multiple metastatic sites",
        "widespread metastatic disease",
        "extensive bone involvement",
        "diffuse hepatic metastases",
    ],
    "4": [  # 4-gram concepts
        "metastatic to the bone",
        "metastatic to the liver",
        "metastases involving the bone",
        "metastases involving the liver",
        "metastatic disease in bone",
        "metastatic lesions in liver",
        "osseous metastatic disease noted",
        "hepatic metastatic disease noted",
    ],
    "5": [  # 5-gram concepts
        "metastatic disease involving bone and",
        "metastatic lesions noted in the",
        "widespread metastatic disease with involvement",
        "multiple sites of metastatic disease",
        "extensive metastatic burden with lesions",
    ],
}


@dataclass
class ExperimentConfig:
    """Configuration for an experimental condition."""
    name: str
    text_column: str
    use_kmeans: bool = True
    use_semantic_concepts: bool = False
    use_random_only: bool = False
    use_llm_extraction: bool = False
    llm_extraction_only: bool = False
    llm_combined: bool = False
    llm_as_text: bool = False
    num_kmeans_filters: int = 64
    num_random_filters: int = 0


# Define all 8 experimental conditions
EXPERIMENT_CONDITIONS = [
    ExperimentConfig(
        name="1_oracle",
        text_column="patient_prompt",
        use_kmeans=True,
        use_semantic_concepts=False,
        num_kmeans_filters=64
    ),
    ExperimentConfig(
        name="2_baseline_random",
        text_column="clinical_text",
        use_kmeans=False,
        use_random_only=True,
        num_random_filters=64
    ),
    ExperimentConfig(
        name="3_kmeans_only",
        text_column="clinical_text",
        use_kmeans=True,
        use_semantic_concepts=False,
        num_kmeans_filters=64
    ),
    ExperimentConfig(
        name="4_concept_aware",
        text_column="clinical_text",
        use_kmeans=False,
        use_semantic_concepts=True,
        num_random_filters=32  # Some random to fill out
    ),
    ExperimentConfig(
        name="5_concept_plus_kmeans",
        text_column="clinical_text",
        use_kmeans=True,
        use_semantic_concepts=True,
        num_kmeans_filters=32
    ),
    ExperimentConfig(
        name="6_llm_extract_only",
        text_column="clinical_text",  # Not used, but required
        use_llm_extraction=True,
        llm_extraction_only=True
    ),
    ExperimentConfig(
        name="7_llm_extract_combined",
        text_column="clinical_text",
        use_kmeans=True,
        use_llm_extraction=True,
        llm_combined=True,
        num_kmeans_filters=64
    ),
    ExperimentConfig(
        name="8_llm_extract_as_text",
        text_column="llm_structured_text",  # Will be created
        use_kmeans=True,
        use_semantic_concepts=False,
        llm_as_text=True,
        num_kmeans_filters=64
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
    """Dataset for categorical features only (condition 6)."""

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


def train_cnn_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    encoder: Optional[CategoricalEncoder] = None
) -> Tuple[CausalCNNText, List[Dict]]:
    """Train a CNN-based model for one fold."""
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

    # Configure filter initialization
    explicit_concepts = METASTATIC_SITE_CONCEPTS if config.use_semantic_concepts else None
    num_kmeans = config.num_kmeans_filters if config.use_kmeans else 0
    num_random = config.num_random_filters if config.use_random_only else 0

    # Create model
    model = CausalCNNText(
        feature_extractor_type="cnn",
        embedding_dim=128,
        kernel_sizes=[3, 4, 5, 7],
        explicit_filter_concepts=explicit_concepts,
        num_kmeans_filters=num_kmeans,
        num_random_filters=num_random,
        projection_dim=128,
        cnn_dropout=0.1,
        max_length=2048,
        dragonnet_representation_dim=128,
        dragonnet_hidden_outcome_dim=64,
        dragonnet_dropout=0.2,
        device=str(device),
        auxiliary_dim=auxiliary_dim
    )

    # Fit tokenizer
    train_texts = train_df[text_column].tolist()
    model.fit_tokenizer(train_texts)

    # Initialize embeddings from BERT
    model.feature_extractor.init_embeddings_from_bert(
        "emilyalsentzer/Bio_ClinicalBERT",
        freeze=False
    )

    # Initialize filters
    if config.use_semantic_concepts or config.use_kmeans:
        model.feature_extractor.init_filters(train_texts, freeze=False)

    # Create datasets
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

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
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
            losses = model.train_step(batch, alpha_propensity=1.0, beta_targreg=0.1)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += losses['loss'].item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
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


def train_mlp_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    encoder: CategoricalEncoder,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float
) -> Tuple[MLPDragonNet, List[Dict]]:
    """Train MLP model for condition 6 (LLM-extract-only)."""
    # Encode categorical features
    train_features = encoder.transform(
        train_df['llm_extracted_metastatic_sites'].tolist(),
        device=device
    )
    val_features = encoder.transform(
        val_df['llm_extracted_metastatic_sites'].tolist(),
        device=device
    )

    # Create model
    model = MLPDragonNet(
        input_dim=encoder.num_categories,
        hidden_dims=[32, 32],
        dragonnet_representation_dim=32,
        dragonnet_hidden_outcome_dim=16,
        dragonnet_dropout=0.2,
        device=str(device)
    )

    # Create datasets
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

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_val_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        # Train
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

        # Validate
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


def predict_cnn(
    model: CausalCNNText,
    df: pd.DataFrame,
    text_column: str,
    device: torch.device,
    batch_size: int,
    encoder: Optional[CategoricalEncoder] = None,
    use_auxiliary: bool = False
) -> Dict[str, np.ndarray]:
    """Generate predictions from CNN model."""
    model.eval()

    texts = df[text_column].tolist()
    all_y0 = []
    all_y1 = []
    all_prop = []

    # Prepare auxiliary features if needed
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

    return {
        'y0_prob': np.concatenate(all_y0),
        'y1_prob': np.concatenate(all_y1),
        'propensity': np.concatenate(all_prop),
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
    encoder: Optional[CategoricalEncoder] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run cross-validation for one experimental condition."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running condition: {config.name}")
    logger.info(f"{'='*60}")

    # Reset index
    df = df.reset_index(drop=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        logger.info(f"  Fold {fold + 1}/{n_folds}")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        if config.llm_extraction_only:
            # Condition 6: MLP on categorical features only
            model, _ = train_mlp_model(
                train_df, test_df, encoder, device,
                epochs, batch_size, learning_rate
            )
            preds = predict_mlp(model, test_df, encoder, device, batch_size)
        else:
            # CNN-based conditions
            model, _ = train_cnn_model(
                train_df, test_df, config, device,
                epochs, batch_size, learning_rate,
                encoder=encoder if config.llm_combined else None
            )
            preds = predict_cnn(
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

    return results_df, metrics


def create_llm_structured_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create structured text from LLM-extracted values (condition 8)."""
    df = df.copy()

    def make_structured(extracted_value):
        if pd.isna(extracted_value) or extracted_value == "unknown":
            return "Number of metastatic sites: unknown"
        return f"Number of metastatic sites: {extracted_value}"

    df['llm_structured_text'] = df['llm_extracted_metastatic_sites'].apply(make_structured)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run concept-aware CNN experiment")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Path to dataset parquet file (with LLM extractions for conditions 6-8)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/concept_aware_experiment",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)"
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
        default=16,
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
        help="Specific conditions to run (e.g., 1_oracle 4_concept_aware)"
    )
    parser.add_argument(
        "--skip-llm-conditions",
        action="store_true",
        help="Skip conditions 6-8 that require LLM extractions"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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

    # Create structured text for condition 8
    if has_llm_extraction:
        df = create_llm_structured_text(df)

    # Setup categorical encoder for LLM conditions
    encoder = None
    if has_llm_extraction:
        encoder = CategoricalEncoder(categories=["1", "2", "3", "4_or_more"])
        encoder.fit(df['llm_extracted_metastatic_sites'].tolist())

    # Filter conditions if specified
    conditions = EXPERIMENT_CONDITIONS
    if args.conditions:
        conditions = [c for c in conditions if c.name in args.conditions]
        logger.info(f"Running {len(conditions)} selected conditions")

    # Skip LLM conditions if requested or extraction not available
    if args.skip_llm_conditions or not has_llm_extraction:
        conditions = [c for c in conditions if not c.use_llm_extraction]
        if not has_llm_extraction:
            logger.warning("LLM extraction column not found. Skipping conditions 6-8.")
            logger.warning("Run scripts/extract_confounders_llm.py first to enable these conditions.")

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
                encoder=encoder
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
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info("\n" + metrics_df.to_string())

    # Save config
    config_info = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'n_folds': args.n_folds,
        'device': str(device),
        'conditions_run': [c.name for c in conditions]
    }
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
