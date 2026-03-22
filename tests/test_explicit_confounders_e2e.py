#!/usr/bin/env python
"""
End-to-end test for explicit confounder extraction feature.

This script tests the complete pipeline:
1. LLM-based confounder extraction using vLLM (both server and python_api modes)
2. Confounder featurization
3. GRU-Pool feature extractor with DragonNet causal head
4. GRU-Pool feature extractor with Causal Forest

Uses a 100-patient sample from synthetic_data/example_synthetic_datasets/(one_confounder/ten_confounders) dataset
and the Qwen/Qwen3.5-0.8B-Base for extraction.

Usage:
    # Test python_api mode (default)
    python tests/test_explicit_confounders_e2e.py --mode python_api

    # Test server mode (requires running vLLM server)
    python tests/test_explicit_confounders_e2e.py --mode server --server-url http://localhost:8000/v1

    # Skip extraction (use cached results)
    python tests/test_explicit_confounders_e2e.py --skip-extraction

    # Full test with both DragonNet and Causal Forest
    python tests/test_explicit_confounders_e2e.py --mode python_api --test-all
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import ExplicitConfounderSpec, ExplicitConfounderExtractionConfig
from oci.data import ClinicalTextDataset, collate_batch
from oci.extraction import VLLMConfounderExtractor, ExtractionCache
from oci.models import CausalText, ExplicitConfounderFeaturizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFOUNDER SPECIFICATIONS (from metadata.json)
# ============================================================================

def get_confounder_specs() -> List[ExplicitConfounderSpec]:
    """Get confounder specifications from the synthetic dataset metadata."""

    # for one confounder
    return [
        ExplicitConfounderSpec(
            name="metastatic_site_count",
            type="categorical",
            categories=["1", "2", "3", "4_or_more"],
            description="Number of metastatic sites at treatment initiation, categorized to reflect disease burden"
            )
    ]
    # for ten confounders
    return [
        ExplicitConfounderSpec(
            name="age",
            type="continuous",
            description="Patient age in years at initiation of first-line therapy"
        ),
        ExplicitConfounderSpec(
            name="ecog_performance_status",
            type="categorical",
            categories=["0", "1", "2", "3"],
            description="Eastern Cooperative Oncology Group performance status score at baseline"
        ),
        ExplicitConfounderSpec(
            name="menopausal_status",
            type="categorical",
            categories=["pre-menopausal", "peri-menopausal", "post-menopausal"],
            description="Menopausal state influencing choice of endocrine backbone"
        ),
        ExplicitConfounderSpec(
            name="visceral_metastasis_site",
            type="categorical",
            categories=["bone-only", "liver", "lung", "multiple_viscera"],
            description="Anatomical location of metastatic disease at treatment start"
        ),
        ExplicitConfounderSpec(
            name="prior_adjuvant_endocrine_therapy",
            type="categorical",
            categories=["none", "tamoxifen", "aromatase_inhibitor", "sequential"],
            description="Type of adjuvant endocrine therapy received before metastatic recurrence"
        ),
        ExplicitConfounderSpec(
            name="baseline_neutrophil_lymphocyte_ratio",
            type="continuous",
            description="Pre-treatment NLR, a marker of systemic inflammation"
        ),
        ExplicitConfounderSpec(
            name="charlson_comorbidity_index",
            type="continuous",
            description="Weighted score summarizing the burden of baseline comorbid conditions"
        ),
        ExplicitConfounderSpec(
            name="baseline_alt_level",
            type="continuous",
            description="Serum alanine aminotransferase (ALT) measured in U/L prior to therapy"
        ),
        ExplicitConfounderSpec(
            name="her2_status",
            type="categorical",
            categories=["negative", "positive", "equivocal"],
            description="HER2 receptor status of the tumor"
        ),
        ExplicitConfounderSpec(
            name="insurance_type",
            type="categorical",
            categories=["private", "medicare", "medicaid", "uninsured"],
            description="Primary payer category"
        ),
    ]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_sample_dataset(
    dataset_path: str = "synthetic_data/example_synthetic_datasets/one_confounder/dataset.parquet",
    sample_size: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """Load a sample of the dataset."""
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_parquet(dataset_path)

    # Sample
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        logger.info(f"Sampled {sample_size} patients")

    logger.info(f"Dataset: {len(df)} patients, treatment rate: {df['treatment_indicator'].mean():.2%}, "
                f"outcome rate: {df['outcome_indicator'].mean():.2%}")

    return df


# ============================================================================
# EXTRACTION
# ============================================================================

def run_extraction(
    df: pd.DataFrame,
    specs: List[ExplicitConfounderSpec],
    mode: str = "python_api",
    server_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen3.5-0.8B-Base",
    tensor_parallel_size: int = 1,
    cache_dir: Optional[str] = None,
    skip_cache: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """Run LLM-based confounder extraction.

    Args:
        df: DataFrame with clinical_text column
        specs: List of confounder specifications
        mode: "server" or "python_api"
        server_url: vLLM server URL (for server mode)
        model_name: Model name for extraction
        tensor_parallel_size: Number of GPUs for tensor parallelism
        cache_dir: Directory for cache files
        skip_cache: If True, don't use cache

    Returns:
        Tuple of (enriched DataFrame, confounder column names)
    """
    logger.info("=" * 60)
    logger.info("EXPLICIT CONFOUNDER EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Extracting {len(specs)} confounders: {[s.name for s in specs]}")

    # Check cache
    cache = ExtractionCache(cache_dir=cache_dir)
    cache_config = {
        'confounders': specs,
        'vllm_model_name': model_name,
        'extraction_temperature': 0.0,
        'extraction_max_tokens': 1024,
    }

    if not skip_cache:
        cached_df = cache.load_if_valid(
            "test_extraction_cache",
            cache_config,
            expected_rows=len(df)
        )
        if cached_df is not None:
            logger.info("Using cached extraction results")
            confounder_columns = [f"explicit_conf_{s.name}" for s in specs]
            for col in cached_df.columns:
                df[col] = cached_df[col].values
            return df, confounder_columns

    # Run extraction
    texts = df['clinical_text'].tolist()

    extractor = VLLMConfounderExtractor(
        specs=specs,
        mode=mode,
        server_url=server_url,
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_retries=3,
        temperature=0.0,
        max_tokens=1024
    )

    start_time = time.time()
    try:
        extracted_df = extractor.extract_to_dataframe(texts, batch_size=32)
    finally:
        extractor.cleanup()
    extraction_time = time.time() - start_time

    # Merge into original dataframe
    confounder_columns = [f"explicit_conf_{s.name}" for s in specs]
    for col in extracted_df.columns:
        df[col] = extracted_df[col].values

    # Log statistics
    logger.info(f"Extraction completed in {extraction_time:.1f}s ({extraction_time/len(texts):.2f}s/sample)")
    for spec in specs:
        col = f"explicit_conf_{spec.name}"
        missing_col = f"{col}_missing"
        if missing_col in df.columns:
            missing_count = df[missing_col].sum()
            logger.info(f"  {spec.name}: {len(df) - missing_count}/{len(df)} extracted "
                       f"({missing_count} missing)")

    # Save cache
    cache.save("test_extraction_cache", cache_config, extracted_df)

    return df, confounder_columns


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate_dragonnet(
    df: pd.DataFrame,
    confounder_columns: List[str],
    specs: List[ExplicitConfounderSpec],
    device: str = "cuda:0",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-5
) -> Dict:
    """Train and evaluate GRU-Pool + DragonNet with explicit confounders.

    Args:
        df: DataFrame with clinical text and confounder columns
        confounder_columns: List of confounder column names
        specs: List of confounder specifications
        device: Device to use
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        Dictionary with evaluation results
    """
    logger.info("=" * 60)
    logger.info("TRAINING: GRU-Pool + DragonNet")
    logger.info("=" * 60)

    # Split data (80% train, 20% test)
    n_train = int(0.8 * len(df))
    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df = df.iloc[n_train:].reset_index(drop=True)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Create model with explicit confounders
    model = CausalText(
        feature_extractor_type="gru_pool",
        gru_pool_embedding_dim=128,
        gru_pool_gru_hidden_dim=128,
        gru_pool_transformer_layers=4,
        gru_pool_max_chunks=100,
        gru_pool_chunk_size=128,
        gru_pool_chunk_overlap=32,
        gru_pool_projection_dim=128,
        gru_pool_max_vocab=10000,
        gru_pool_min_word_freq=2,
        causal_head_representation_dim=128,
        causal_head_hidden_outcome_dim=64,
        causal_head_dropout=0.0,
        explicit_confounder_specs=specs,
        explicit_confounder_hidden_dim=64,
        explicit_confounder_dropout=0.0,
        model_type="rlearner",
        device=device
    )

    # Fit tokenizer
    train_texts = train_df['clinical_text'].tolist()
    model.fit_tokenizer(train_texts)
    logger.info(f"Fitted tokenizer on {len(train_texts)} texts")

    # Fit explicit confounder featurizer (using spec.name keys, not prefixed column names)
    train_confounder_values = []
    for idx in range(len(train_df)):
        row_values = {}
        for spec in specs:
            col = f"explicit_conf_{spec.name}"
            row_values[spec.name] = train_df[col].iloc[idx] if col in train_df.columns else None
            missing_col = f"{col}_missing"
            if missing_col in train_df.columns:
                row_values[f"{spec.name}_missing"] = train_df[missing_col].iloc[idx]
            else:
                row_values[f"{spec.name}_missing"] = row_values[spec.name] is None
        train_confounder_values.append(row_values)
    model.fit_explicit_confounder_featurizer(train_confounder_values)
    logger.info("Fitted explicit confounder featurizer")

    # Create datasets
    train_dataset = ClinicalTextDataset(
        data=train_df,
        text_column='clinical_text',
        outcome_column='outcome_indicator',
        treatment_column='treatment_indicator',
        explicit_confounder_columns=confounder_columns
    )

    test_dataset = ClinicalTextDataset(
        data=test_df,
        text_column='clinical_text',
        outcome_column='outcome_indicator',
        treatment_column='treatment_indicator',
        explicit_confounder_columns=confounder_columns
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch['outcome'] = batch['outcome'].to(device)
            batch['treatment'] = batch['treatment'].to(device)

            optimizer.zero_grad()
            losses = model.train_step(batch, alpha_propensity=1.0, beta_targreg=0.1)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += losses['loss'].item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_outcomes = []
    all_treatments = []
    all_true_ite = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            texts = batch['texts']
            explicit_confounder_values = batch.get('explicit_confounder_values')

            preds = model.predict(texts, explicit_confounder_values=explicit_confounder_values)

            all_preds.append({
                'y0_prob': preds['y0_prob'].cpu().numpy(),
                'y1_prob': preds['y1_prob'].cpu().numpy(),
                'propensity': preds['propensity'].cpu().numpy()
            })
            all_outcomes.append(batch['outcome'].numpy())
            all_treatments.append(batch['treatment'].numpy())

    # Concatenate predictions
    y0_prob = np.concatenate([p['y0_prob'] for p in all_preds])
    y1_prob = np.concatenate([p['y1_prob'] for p in all_preds])
    propensity = np.concatenate([p['propensity'] for p in all_preds])
    outcomes = np.concatenate(all_outcomes)
    treatments = np.concatenate(all_treatments)

    # Get true ITE from test_df
    true_ite = test_df['true_ite_prob'].values
    pred_ite = y1_prob - y0_prob

    # Compute metrics
    results = {
        'model_type': 'dragonnet',
        'final_train_loss': train_losses[-1],
        'propensity_auroc': roc_auc_score(treatments, propensity) if len(np.unique(treatments)) > 1 else None,
        'ite_correlation': np.corrcoef(true_ite, pred_ite)[0, 1],
        'ite_mae': np.mean(np.abs(true_ite - pred_ite)),
        'mean_pred_ite': np.mean(pred_ite),
        'std_pred_ite': np.std(pred_ite),
        'mean_true_ite': np.mean(true_ite),
        'std_true_ite': np.std(true_ite),
    }

    # Outcome AUROC by treatment arm
    if len(np.unique(outcomes[treatments == 0])) > 1:
        results['y0_auroc'] = roc_auc_score(outcomes[treatments == 0], y0_prob[treatments == 0])
    if len(np.unique(outcomes[treatments == 1])) > 1:
        results['y1_auroc'] = roc_auc_score(outcomes[treatments == 1], y1_prob[treatments == 1])

    logger.info("Results:")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def train_and_evaluate_causal_forest(
    df: pd.DataFrame,
    confounder_columns: List[str],
    specs: List[ExplicitConfounderSpec],
    device: str = "cuda:0",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-5
) -> Dict:
    """Train and evaluate GRU-Pool + Causal Forest with explicit confounders.

    Uses two-stage training:
    1. Train neural network for feature extraction (using CausalTextForest.train_step)
    2. Train causal forest on extracted neural features + raw confounder features

    CausalTextForest now natively supports explicit confounders - it automatically
    concatenates raw confounder features to neural features.

    Args:
        df: DataFrame with clinical text and confounder columns
        confounder_columns: List of confounder column names
        specs: List of confounder specifications
        device: Device to use
        epochs: Number of training epochs for stage 1
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        Dictionary with evaluation results
    """
    logger.info("=" * 60)
    logger.info("TRAINING: GRU-Pool + Causal Forest")
    logger.info("=" * 60)

    # Check if econml is available
    try:
        from oci.models import CausalTextForest, ECONML_AVAILABLE
        if not ECONML_AVAILABLE:
            logger.warning("econml not available, skipping causal forest test")
            return {'model_type': 'causal_forest', 'error': 'econml not available'}
    except ImportError:
        logger.warning("CausalTextForest not available, skipping causal forest test")
        return {'model_type': 'causal_forest', 'error': 'import error'}

    # Split data
    n_train = int(0.8 * len(df))
    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df = df.iloc[n_train:].reset_index(drop=True)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Create model WITH explicit confounder specs
    model = CausalTextForest(
        feature_extractor_type="frozen_llm_pooler",
        flp_projection_dim=128,
        representation_dim=128,
        hidden_dim=64,
        dropout=0.0,
        cf_n_estimators=100,
        cf_min_samples_leaf=5,
        cf_honest=True,
        cf_inference=True,
        cf_use_rlearner_representation=True,
        explicit_confounder_specs=specs,  # Native support for explicit confounders
        device=device
    )

    # Create datasets with explicit confounders
    train_dataset = ClinicalTextDataset(
        data=train_df,
        text_column='clinical_text',
        outcome_column='outcome_indicator',
        treatment_column='treatment_indicator',
        explicit_confounder_columns=confounder_columns
    )

    test_dataset = ClinicalTextDataset(
        data=test_df,
        text_column='clinical_text',
        outcome_column='outcome_indicator',
        treatment_column='treatment_indicator',
        explicit_confounder_columns=confounder_columns
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # Get confounder values for fitting normalization stats
    train_conf_values = [train_dataset[i]['explicit_confounder_values'] for i in range(len(train_dataset))]

    # Fit the MLP featurizer for Stage 1 training (learns jointly with text extractor)
    model.fit_explicit_confounder_featurizer(train_conf_values)
    logger.info("Fitted ExplicitConfounderFeaturizer for Stage 1 training")

    # Fit raw confounder stats for Stage 2 causal forest
    model.fit_explicit_confounders(train_conf_values)
    logger.info("Fitted explicit confounder normalization stats for causal forest")

    # Stage 1: Train neural network for feature extraction
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    logger.info("Stage 1: Training neural feature extractor...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch['outcome'] = batch['outcome'].to(device)
            batch['treatment'] = batch['treatment'].to(device)

            optimizer.zero_grad()
            losses = model.train_step(batch, alpha_propensity=1.0)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += losses['loss'].item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f}")

    # Stage 2: Train causal forest on neural features + raw confounder features
    # The model now handles the concatenation internally
    logger.info("Stage 2: Training causal forest on neural features + raw confounder features...")
    model.eval()

    T_train = train_df['treatment_indicator'].values
    Y_train = train_df['outcome_indicator'].values

    # Use the model's built-in train_causal_forest with confounder values
    model.train_causal_forest(
        texts=train_texts,
        T=T_train,
        Y=Y_train,
        batch_size=batch_size,
        explicit_confounder_values=train_conf_values
    )
    logger.info("Causal forest trained")

    # Evaluation on test set using model.predict() with confounder values
    test_texts = test_df['clinical_text'].tolist()
    test_conf_values = [test_dataset[i]['explicit_confounder_values'] for i in range(len(test_dataset))]

    preds = model.predict(
        texts=test_texts,
        batch_size=batch_size,
        return_ci=True,
        alpha=0.05,
        explicit_confounder_values=test_conf_values
    )

    tau_pred = preds['tau_pred']
    tau_lower = preds.get('tau_lower', np.full_like(tau_pred, np.nan))
    tau_upper = preds.get('tau_upper', np.full_like(tau_pred, np.nan))
    true_ite = test_df['true_ite_prob'].values

    # Compute metrics
    results = {
        'model_type': 'causal_forest',
        'ite_correlation': np.corrcoef(true_ite, tau_pred)[0, 1],
        'ite_mae': np.mean(np.abs(true_ite - tau_pred)),
        'mean_pred_ite': np.mean(tau_pred),
        'std_pred_ite': np.std(tau_pred),
        'mean_true_ite': np.mean(true_ite),
        'std_true_ite': np.std(true_ite),
        'ci_coverage': np.mean((true_ite >= tau_lower.flatten()) & (true_ite <= tau_upper.flatten())),
        'mean_ci_width': np.mean(tau_upper - tau_lower),
    }

    logger.info("Results:")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-end test for explicit confounder extraction")
    parser.add_argument("--mode", choices=["server", "python_api"], default="python_api",
                        help="vLLM mode for extraction")
    parser.add_argument("--server-url", default="http://localhost:8000/v1",
                        help="vLLM server URL (for server mode)")
    parser.add_argument("--model-name", default="google/medgemma-1.5-4b-it",
                        help="Model name for extraction")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of patients to sample")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extraction, use cached results")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Don't use cache for extraction")
    parser.add_argument("--test-all", action="store_true",
                        help="Test both DragonNet and Causal Forest")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--output-dir", default="./test_outputs",
                        help="Output directory for results")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXPLICIT CONFOUNDER EXTRACTION E2E TEST")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Device: {args.device}")

    # Get confounder specs
    specs = get_confounder_specs()
    logger.info(f"Confounders: {[s.name for s in specs]}")

    # Load dataset
    df = load_sample_dataset(sample_size=args.sample_size)

    # Run extraction
    if not args.skip_extraction:
        df, confounder_columns = run_extraction(
            df,
            specs,
            mode=args.mode,
            server_url=args.server_url,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            cache_dir=str(output_dir),
            skip_cache=args.skip_cache
        )
    else:
        logger.info("Skipping extraction, loading from cache...")
        cache = ExtractionCache(cache_dir=str(output_dir))
        cache_config = {
            'confounders': specs,
            'vllm_model_name': args.model_name,
            'extraction_temperature': 0.0,
            'extraction_max_tokens': 1024,
        }
        cached_df = cache.load_if_valid("test_extraction_cache", cache_config, expected_rows=len(df))
        if cached_df is None:
            logger.error("No cached extraction results found. Run without --skip-extraction first.")
            return
        confounder_columns = [f"explicit_conf_{s.name}" for s in specs]
        for col in cached_df.columns:
            df[col] = cached_df[col].values

    # Save enriched dataset
    enriched_path = output_dir / "enriched_dataset.parquet"
    df.to_parquet(enriched_path)
    logger.info(f"Saved enriched dataset to {enriched_path}")

    # Train and evaluate models
    results = {}

    #Test DragonNet
    results['dragonnet'] = train_and_evaluate_dragonnet(
        df,
        confounder_columns,
        specs,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Test Causal Forest (if requested)
    if args.test_all:
        results['causal_forest'] = train_and_evaluate_causal_forest(
            df,
            confounder_columns,
            specs,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {results_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for model_name, model_results in results.items():
        logger.info(f"\n{model_name.upper()}:")
        if 'error' in model_results:
            logger.info(f"  Error: {model_results['error']}")
        else:
            if 'ite_correlation' in model_results:
                logger.info(f"  ITE Correlation: {model_results['ite_correlation']:.4f}")
            if 'ite_mae' in model_results:
                logger.info(f"  ITE MAE: {model_results['ite_mae']:.4f}")
            if 'propensity_auroc' in model_results:
                logger.info(f"  Propensity AUROC: {model_results['propensity_auroc']:.4f}")

    logger.info("\nTest completed successfully!")


if __name__ == "__main__":
    main()
