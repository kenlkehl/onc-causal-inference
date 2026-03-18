#!/usr/bin/env python
"""Semi-synthetic experiment runner for OCI.

Uses real clinical text but simulated treatments/outcomes with known true ITEs.
Measures how well text extractors capture confounding beyond explicitly specified
confounders.

Two equation modes:
- "random": LLM generates regression equations with random coefficients (stress-testing)
- "fitted": Equations fit to real T/Y using extracted confounders (stability analysis)

Usage:
    # Random mode: M=5 DGPs, N=5 repeats each
    python oracle_experiment_scripts/run_semisynthetic_experiments.py \
        --dataset-path /path/to/real/dataset.parquet \
        --clinical-question "Compare pembrolizumab vs docetaxel for advanced NSCLC" \
        --output-dir ../pcori_experiments/semisynthetic \
        --equation-mode random \
        --num-dgps 5 --num-repeats 5 \
        --devices cuda:0 cuda:1

    # Fitted mode: learn equations from real T/Y, measure stability
    python oracle_experiment_scripts/run_semisynthetic_experiments.py \
        --dataset-path /path/to/real/dataset.parquet \
        --clinical-question "Compare pembrolizumab vs docetaxel for advanced NSCLC" \
        --output-dir ../pcori_experiments/semisynthetic_fitted \
        --equation-mode fitted \
        --num-dgps 5 --num-repeats 10 \
        --devices cuda:0 cuda:1

        python oracle_experiment_scripts/run_semisynthetic_experiments.py --dataset-
path ./synthetic_data/example_synthetic_datasets/ten_confounders_nsclc/dataset.parquet --clinical-question "Compare gemcitabine to vinorelbine for advanced NSCLC
" --output-dir ../pcori_experiments/production_ten_confounders_nsclc/semisynthetic --equation-mode fitted --num-dgps 10 --num-repeats 10 --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

    # Quick test
    python oracle_experiment_scripts/run_semisynthetic_experiments.py \
        --dataset-path /path/to/real/dataset.parquet \
        --clinical-question "Compare drug A vs drug B for condition C" \
        --output-dir /tmp/semisynthetic_test \
        --equation-mode random \
        --num-dgps 1 --num-repeats 1 --epochs 3 --max-length 5000 \
        --devices cuda:0
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import traceback
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from semisynthetic_dgp import (
    SemiSyntheticDGPConfig,
    SemiSyntheticDGP,
    generate_dgp,
    generate_and_extract_confounders,
    simulate_outcomes,
    select_confounder_subset,
    save_dgp_metadata,
)
from run_oracle_experiments import (
    ExperimentConfig,
    compute_metrics,
    run_causal_forest_experiment,
    run_best_attainable_experiment,
    build_confounder_values_from_columns,
    _common_model_kwargs,
    _create_datasets_and_loaders,
    precompute_single_cache,
)

from oci.models.hidden_state_cache import HiddenStateCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confounder-only forest runner (adapted from run_best_attainable_experiment)
# ---------------------------------------------------------------------------

def run_confounder_forest_arm(
    df: pd.DataFrame,
    confounder_specs,
    n_folds: int,
    seed: int,
    cf_n_estimators: int = 200,
    cf_min_samples_leaf: int = 5,
) -> Dict[str, Any]:
    """Run confounder-only CausalForestDML with K-fold CV.

    Uses extracted confounders (explicit_conf_* columns) as features.
    No neural network, no text features.
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import KFold
    from oci.models.explicit_confounder_featurizer import get_raw_confounder_features

    if not confounder_specs:
        return {'error': 'No confounders specified', 'skipped': True}

    spec_names = [s.name for s in confounder_specs]
    df = df.reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Build confounder value dicts from explicit_conf_* columns
        train_values = build_confounder_values_from_columns(
            train_df, spec_names, "explicit_conf"
        )
        test_values = build_confounder_values_from_columns(
            test_df, spec_names, "explicit_conf"
        )

        # Compute normalization stats from train fold
        continuous_means = {}
        continuous_stds = {}
        for spec in confounder_specs:
            if spec.type == "continuous":
                vals = []
                for v in train_values:
                    val = v.get(spec.name)
                    miss = v.get(f"{spec.name}_missing", val is None)
                    if not miss and val is not None:
                        try:
                            vals.append(float(val))
                        except (ValueError, TypeError):
                            pass
                if vals:
                    continuous_means[spec.name] = sum(vals) / len(vals)
                    variance = sum((x - continuous_means[spec.name]) ** 2 for x in vals) / len(vals)
                    continuous_stds[spec.name] = max(variance ** 0.5, 1e-6)
                else:
                    continuous_means[spec.name] = 0.0
                    continuous_stds[spec.name] = 1.0

        train_features, feature_names = get_raw_confounder_features(
            train_values, confounder_specs,
            continuous_means=continuous_means,
            continuous_stds=continuous_stds,
        )
        test_features, _ = get_raw_confounder_features(
            test_values, confounder_specs,
            continuous_means=continuous_means,
            continuous_stds=continuous_stds,
        )

        X_train = np.array(train_features, dtype=np.float64)
        X_test = np.array(test_features, dtype=np.float64)
        T_train = train_df['treatment_indicator'].values.astype(np.float64)
        Y_train = train_df['outcome_indicator'].values.astype(np.float64)

        cf = CausalForestDML(
            model_t=RandomForestClassifier(
                n_estimators=max(50, cf_n_estimators // 2),
                min_samples_leaf=cf_min_samples_leaf,
                random_state=seed, n_jobs=-1,
            ),
            model_y=RandomForestRegressor(
                n_estimators=max(50, cf_n_estimators // 2),
                min_samples_leaf=cf_min_samples_leaf,
                random_state=seed, n_jobs=-1,
            ),
            discrete_treatment=True,
            n_estimators=cf_n_estimators,
            min_samples_leaf=cf_min_samples_leaf,
            max_depth=None,
            honest=True,
            inference=True,
            random_state=seed,
            n_jobs=-1,
        )
        cf.fit(Y_train, T_train, X=X_train)

        tau_pred = cf.effect(X_test).flatten()
        tau_lower, tau_upper = cf.effect_interval(X_test, alpha=0.05)
        tau_lower = tau_lower.flatten()
        tau_upper = tau_upper.flatten()

        # Propensity and outcome predictions for metrics
        rf_prop = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_prop.fit(X_train, T_train.astype(int))
        pred_propensity = rf_prop.predict_proba(X_test)[:, 1]

        rf_out = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_out.fit(X_train, Y_train.astype(int))
        pred_outcome = rf_out.predict_proba(X_test)[:, 1]

        pred_y0 = pred_outcome - tau_pred * pred_propensity
        pred_y1 = pred_y0 + tau_pred

        fold_preds = test_df.copy()
        fold_preds['pred_y0_prob'] = pred_y0
        fold_preds['pred_y1_prob'] = pred_y1
        fold_preds['pred_ite_prob'] = tau_pred
        fold_preds['pred_propensity'] = pred_propensity
        fold_preds['pred_tau_lower'] = tau_lower
        fold_preds['pred_tau_upper'] = tau_upper
        fold_preds['cv_fold'] = fold + 1
        all_predictions.append(fold_preds)

    results_df = pd.concat(all_predictions).sort_index()

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
        tau_lower=results_df['pred_tau_lower'].values,
        tau_upper=results_df['pred_tau_upper'].values,
    )

    return {'metrics': metrics, 'n_samples': len(results_df)}


# ---------------------------------------------------------------------------
# Text + confounder forest runner
# ---------------------------------------------------------------------------

def run_text_forest_arm(
    df: pd.DataFrame,
    confounder_specs,
    confounder_cols: List[str],
    seed: int,
    device: str,
    flp_model_name: str,
    flp_max_length: int,
    flp_downprojection_dim: int,
    flp_projection_dim: int = 128,
    flp_gated_attention_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    n_folds: int = 5,
    cf_n_estimators: int = 200,
    cf_min_samples_leaf: int = 5,
    gamma_rlearner: float = 1.0,
    hidden_state_cache=None,
    gpu_store=None,
) -> Dict[str, Any]:
    """Run causal forest with text features + optional confounders.

    Wraps run_causal_forest_experiment with the right config.
    """
    config = ExperimentConfig(
        dataset_path="",  # Not used by the experiment runner directly
        dataset_name="semisynthetic",
        model_type="causal_forest",
        use_explicit_confounders=bool(confounder_specs),
        repeat_index=seed - 42,  # For KFold seeding
        flp_model_name=flp_model_name,
        flp_max_length=flp_max_length,
        flp_downprojection_dim=flp_downprojection_dim,
        flp_projection_dim=flp_projection_dim,
        flp_gated_attention_dim=flp_gated_attention_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_folds=n_folds,
        cf_n_estimators=cf_n_estimators,
        cf_min_samples_leaf=cf_min_samples_leaf,
        gamma_rlearner=gamma_rlearner,
    )

    device_obj = torch.device(device)

    return run_causal_forest_experiment(
        config=config,
        device=device_obj,
        df=df,
        confounder_specs=confounder_specs if confounder_specs else None,
        confounder_cols=confounder_cols if confounder_cols else None,
        gpu_store=gpu_store,
        hidden_state_cache=hidden_state_cache,
        cache_registry=None,
        gpu_store_registry=None,
    )


# ---------------------------------------------------------------------------
# Prepare simulation dataset
# ---------------------------------------------------------------------------

def prepare_simulation_dataset(
    dgp: SemiSyntheticDGP,
    real_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Create a dataset with real text + simulated T/Y from the DGP.

    Simulates new treatment/outcome each call (different Bernoulli draws).
    """
    sim_df = simulate_outcomes(
        dgp.characteristics,
        dgp.confounders,
        dgp.summary_stats,
        dgp.treatment_equation,
        dgp.outcome_equation,
        seed=seed,
    )

    # Combine: real text + extraction columns + simulated T/Y/ITE
    result = real_df[['clinical_text']].copy()
    result = result.reset_index(drop=True)

    # Add extraction columns
    for col in dgp.extracted_df.columns:
        result[col] = dgp.extracted_df[col].values

    # Replace treatment/outcome with simulated values
    for col in sim_df.columns:
        result[col] = sim_df[col].values

    return result


def get_confounder_cols(
    specs, extracted_df: pd.DataFrame
) -> Optional[List[str]]:
    """Get explicit_conf_* column names for the given specs."""
    if not specs:
        return None
    cols = []
    for s in specs:
        col = f"explicit_conf_{s.name}"
        miss_col = f"explicit_conf_{s.name}_missing"
        if col in extracted_df.columns:
            cols.append(col)
        if miss_col in extracted_df.columns:
            cols.append(miss_col)
    return cols if cols else None


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_result(
    result: Dict[str, Any],
    dgp_index: int,
    repeat_index: int,
    arm: str,
    uses_text: bool,
    n_confounders_used: int,
    n_confounders_total: int,
    confounder_fraction: float,
    equation_mode: str,
    dgp: SemiSyntheticDGP,
    sim_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Format experiment result for saving."""
    n_patients = len(sim_df)
    total_missing = 0
    for spec in dgp.confounder_specs:
        miss_col = f"explicit_conf_{spec.name}_missing"
        if miss_col in dgp.extracted_df.columns:
            total_missing += dgp.extracted_df[miss_col].sum()
    missingness_rate = total_missing / (n_patients * len(dgp.confounder_specs)) if dgp.confounder_specs else 0

    return {
        "dgp_index": dgp_index,
        "repeat_index": repeat_index,
        "arm": arm,
        "uses_text": uses_text,
        "n_confounders_used": n_confounders_used,
        "n_confounders_total": n_confounders_total,
        "confounder_fraction": confounder_fraction,
        "equation_mode": equation_mode,
        "metrics": result.get('metrics', {}),
        "n_samples": result.get('n_samples', n_patients),
        "dgp_stats": {
            "extraction_missingness_rate": float(missingness_rate),
            "simulated_treatment_rate": float(sim_df['treatment_indicator'].mean()),
            "simulated_outcome_rate": float(sim_df['outcome_indicator'].mean()),
            "true_ate": float(sim_df['true_ite_prob'].mean()),
            "true_ite_std": float(sim_df['true_ite_prob'].std()),
        },
    }


def save_result(result: Dict[str, Any], output_dir: Path):
    """Save a single experiment result."""
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"dgp{result['dgp_index']}_repeat{result['repeat_index']}"
        f"_{result['arm']}.json"
    )
    path = results_dir / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def is_result_done(output_dir: Path, dgp_index: int, repeat_index: int, arm: str) -> bool:
    """Check if a result already exists (for resume support)."""
    filename = f"dgp{dgp_index}_repeat{repeat_index}_{arm}.json"
    return (output_dir / "results" / filename).exists()


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiments(
    dataset_path: str,
    clinical_question: str,
    output_dir: str,
    equation_mode: str = "random",
    num_dgps: int = 5,
    num_repeats: int = 5,
    num_confounders: int = 10,
    confounder_fractions: List[float] = None,
    vary_confounders_per_dgp: Optional[bool] = None,
    # Model hyperparameters
    flp_model_name: str = "Qwen/Qwen3.5-0.8B-Base",
    flp_max_length: int = 10000,
    flp_downprojection_dim: int = 256,
    flp_projection_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    n_folds: int = 5,
    cf_n_estimators: int = 200,
    cf_min_samples_leaf: int = 5,
    # DGP params
    treatment_effect_prob: float = 0.10,
    target_treatment_rate: float = 0.5,
    target_control_outcome_rate: float = 0.2,
    # vLLM (used for both confounder extraction and generation)
    vllm_mode: str = "server",
    vllm_server_url: str = "http://localhost:8000/v1",
    vllm_model_name: str = "openai/gpt-oss-120b",
    vllm_tensor_parallel_size: int = 1,
    vllm_download_dir: Optional[str] = None,
    vllm_max_model_len: int = 120000,
    vllm_max_tokens: int = 5000,
    # Infrastructure
    devices: List[str] = None,
    cache: bool = False,
    gpu_cache: bool = False,
    resume: bool = False,
):
    """Main experiment runner."""
    if confounder_fractions is None:
        confounder_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    if devices is None:
        devices = ["cuda:0"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build DGP config
    dgp_config = SemiSyntheticDGPConfig(
        clinical_question=clinical_question,
        num_confounders=num_confounders,
        equation_mode=equation_mode,
        vary_confounders_per_dgp=vary_confounders_per_dgp,
        treatment_effect_prob=treatment_effect_prob,
        target_treatment_rate=target_treatment_rate,
        target_control_outcome_rate=target_control_outcome_rate,
        vllm_mode=vllm_mode,
        vllm_server_url=vllm_server_url,
        vllm_model_name=vllm_model_name,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_download_dir=vllm_download_dir,
        vllm_max_model_len=vllm_max_model_len,
        vllm_max_tokens=vllm_max_tokens,
    )

    # Save config
    config_dict = {
        "dataset_path": dataset_path,
        "clinical_question": clinical_question,
        "equation_mode": equation_mode,
        "num_dgps": num_dgps,
        "num_repeats": num_repeats,
        "confounder_fractions": confounder_fractions,
        "dgp_config": asdict(dgp_config),
        "flp_model_name": flp_model_name,
        "flp_max_length": flp_max_length,
        "flp_downprojection_dim": flp_downprojection_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_folds": n_folds,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # Load real dataset
    dp = Path(dataset_path)
    if dp.is_dir():
        parquet_file = dp / "dataset.parquet"
    else:
        parquet_file = dp
    df = pd.read_parquet(parquet_file)
    texts = df['clinical_text'].tolist()
    logger.info(f"Loaded dataset: {len(df)} patients from {parquet_file}")

    # Pre-cache frozen LLM hidden states (shared across all DGPs/repeats)
    device = devices[0]
    hidden_state_cache = None
    gpu_store = None

    if cache or gpu_cache:
        cache_dir = str(parquet_file.parent / '.oci_cache')
        hs_cache = HiddenStateCache(
            cache_dir=cache_dir,
            model_name=flp_model_name,
            max_length=flp_max_length,
            dataset_path=str(parquet_file),
            downprojection_dim=flp_downprojection_dim,
        )
        if not hs_cache.is_cached():
            logger.info("Pre-computing hidden state cache...")
            cache_info = dict(
                parquet_file=parquet_file,
                model_name=flp_model_name,
                max_length=flp_max_length,
                batch_size=batch_size,
                downprojection_dim=flp_downprojection_dim,
            )
            hs_cache = precompute_single_cache(cache_info, devices)
        else:
            hs_cache.load()
            logger.info(f"Loaded existing hidden state cache: {hs_cache.num_samples} samples")

        if gpu_cache:
            from oci.models.gpu_hidden_state_store import GPUHiddenStateStore
            gpu_store = GPUHiddenStateStore.from_cache(hs_cache, device=device)
            logger.info(f"GPU cache loaded: {gpu_store.num_samples} samples on {device}")
        else:
            hidden_state_cache = hs_cache

    # Generate shared confounders if vary_confounders is False
    shared_confounders = None
    shared_specs = None
    shared_extracted_df = None

    if not dgp_config.should_vary_confounders:
        logger.info("Generating shared confounders (vary_confounders=False)...")
        shared_confounders, shared_specs, shared_extracted_df = \
            generate_and_extract_confounders(
                dgp_config, texts, dataset_path, dgp_index=0,
                cache_dir=str(output_path),
            )

    # Experiment loop
    all_results = []
    total_arms = 0

    for dgp_idx in range(num_dgps):
        logger.info(f"\n{'='*60}")
        logger.info(f"DGP {dgp_idx + 1}/{num_dgps}")
        logger.info(f"{'='*60}")

        # Generate DGP
        dgp = generate_dgp(
            dgp_config, texts, df, dataset_path, dgp_idx,
            cache_dir=str(output_path),
            confounders=shared_confounders,
            specs=shared_specs,
            extracted_df=shared_extracted_df,
        )
        save_dgp_metadata(dgp, output_path, dgp_idx)

        for repeat_idx in range(num_repeats):
            seed = 42 + dgp_idx * 1000 + repeat_idx
            logger.info(f"\n--- DGP {dgp_idx}, Repeat {repeat_idx + 1}/{num_repeats} (seed={seed}) ---")

            # Simulate T/Y
            sim_df = prepare_simulation_dataset(dgp, df, seed)

            # --- Confounder-only arms ---
            for fraction in confounder_fractions:
                n_subset = round(len(dgp.confounder_specs) * fraction)
                subset_specs = select_confounder_subset(
                    dgp.confounder_specs, n_subset, seed=seed
                )
                arm_name = f"confounder_forest_{fraction:.2f}"

                if resume and is_result_done(output_path, dgp_idx, repeat_idx, arm_name):
                    logger.info(f"  Skipping {arm_name} (already done)")
                    continue

                if n_subset == 0:
                    # No confounders = no features -> skip
                    logger.info(f"  Skipping {arm_name} (0 confounders)")
                    continue

                logger.info(f"  Running {arm_name} ({n_subset} confounders)...")
                try:
                    result = run_confounder_forest_arm(
                        sim_df, subset_specs,
                        n_folds=n_folds, seed=seed,
                        cf_n_estimators=cf_n_estimators,
                        cf_min_samples_leaf=cf_min_samples_leaf,
                    )
                    formatted = format_result(
                        result, dgp_idx, repeat_idx, arm_name,
                        uses_text=False, n_confounders_used=n_subset,
                        n_confounders_total=len(dgp.confounder_specs),
                        confounder_fraction=fraction,
                        equation_mode=equation_mode, dgp=dgp, sim_df=sim_df,
                    )
                    save_result(formatted, output_path)
                    all_results.append(formatted)
                    total_arms += 1
                    ite_corr = result.get('metrics', {}).get('ite_corr', 'N/A')
                    logger.info(f"    ITE corr: {ite_corr}")
                except Exception as e:
                    logger.error(f"    Failed: {e}")
                    traceback.print_exc()

            # --- Best attainable (all confounders) ---
            arm_name = "best_attainable"
            if not (resume and is_result_done(output_path, dgp_idx, repeat_idx, arm_name)):
                logger.info(f"  Running {arm_name} (all {len(dgp.confounder_specs)} confounders)...")
                try:
                    result = run_confounder_forest_arm(
                        sim_df, dgp.confounder_specs,
                        n_folds=n_folds, seed=seed,
                        cf_n_estimators=cf_n_estimators,
                        cf_min_samples_leaf=cf_min_samples_leaf,
                    )
                    formatted = format_result(
                        result, dgp_idx, repeat_idx, arm_name,
                        uses_text=False,
                        n_confounders_used=len(dgp.confounder_specs),
                        n_confounders_total=len(dgp.confounder_specs),
                        confounder_fraction=1.0,
                        equation_mode=equation_mode, dgp=dgp, sim_df=sim_df,
                    )
                    save_result(formatted, output_path)
                    all_results.append(formatted)
                    total_arms += 1
                    ite_corr = result.get('metrics', {}).get('ite_corr', 'N/A')
                    logger.info(f"    ITE corr: {ite_corr}")
                except Exception as e:
                    logger.error(f"    Failed: {e}")
                    traceback.print_exc()

            # --- Text + confounder arms ---
            for fraction in confounder_fractions:
                n_subset = round(len(dgp.confounder_specs) * fraction)
                subset_specs = select_confounder_subset(
                    dgp.confounder_specs, n_subset, seed=seed
                )
                arm_name = f"text_forest_{fraction:.2f}"

                if resume and is_result_done(output_path, dgp_idx, repeat_idx, arm_name):
                    logger.info(f"  Skipping {arm_name} (already done)")
                    continue

                confounder_cols = get_confounder_cols(
                    subset_specs, dgp.extracted_df
                )

                logger.info(
                    f"  Running {arm_name} (text + {n_subset} confounders) "
                    f"on {device}..."
                )
                try:
                    result = run_text_forest_arm(
                        df=sim_df,
                        confounder_specs=subset_specs if subset_specs else None,
                        confounder_cols=confounder_cols,
                        seed=seed,
                        device=device,
                        flp_model_name=flp_model_name,
                        flp_max_length=flp_max_length,
                        flp_downprojection_dim=flp_downprojection_dim,
                        flp_projection_dim=flp_projection_dim,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        n_folds=n_folds,
                        cf_n_estimators=cf_n_estimators,
                        cf_min_samples_leaf=cf_min_samples_leaf,
                        hidden_state_cache=hidden_state_cache,
                        gpu_store=gpu_store,
                    )
                    formatted = format_result(
                        result, dgp_idx, repeat_idx, arm_name,
                        uses_text=True, n_confounders_used=n_subset,
                        n_confounders_total=len(dgp.confounder_specs),
                        confounder_fraction=fraction,
                        equation_mode=equation_mode, dgp=dgp, sim_df=sim_df,
                    )
                    save_result(formatted, output_path)
                    all_results.append(formatted)
                    total_arms += 1
                    ite_corr = result.get('metrics', {}).get('ite_corr', 'N/A')
                    logger.info(f"    ITE corr: {ite_corr}")
                except Exception as e:
                    logger.error(f"    Failed: {e}")
                    traceback.print_exc()

                # Clean up GPU memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Aggregate results
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed {total_arms} experiment arms")
    logger.info(f"{'='*60}")

    if all_results:
        summary_rows = []
        for r in all_results:
            row = {
                'dgp_index': r['dgp_index'],
                'repeat_index': r['repeat_index'],
                'arm': r['arm'],
                'uses_text': r['uses_text'],
                'n_confounders_used': r['n_confounders_used'],
                'confounder_fraction': r['confounder_fraction'],
                'equation_mode': r['equation_mode'],
            }
            row.update(r.get('metrics', {}))
            row.update({f"dgp_{k}": v for k, v in r.get('dgp_stats', {}).items()})
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_path / "summary.csv", index=False)
        logger.info(f"Summary saved to {output_path / 'summary.csv'}")

        # Print summary table
        print("\n=== Summary ===")
        for arm_type in sorted(summary_df['arm'].unique()):
            arm_data = summary_df[summary_df['arm'] == arm_type]
            mean_corr = arm_data['ite_corr'].mean()
            std_corr = arm_data['ite_corr'].std()
            print(f"  {arm_type:40s}  ITE corr: {mean_corr:.3f} +/- {std_corr:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semi-synthetic experiment runner for OCI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-path", required=True,
                        help="Path to real dataset (parquet file or directory)")
    parser.add_argument("--clinical-question", required=True,
                        help="Clinical question for confounder generation")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for results")
    parser.add_argument("--equation-mode", default="random", choices=["random", "fitted"],
                        help="DGP equation mode (default: random)")
    parser.add_argument("--num-dgps", type=int, default=5,
                        help="Number of DGPs to generate (M)")
    parser.add_argument("--num-repeats", type=int, default=5,
                        help="Number of repeats per DGP (N)")
    parser.add_argument("--num-confounders", type=int, default=10,
                        help="Number of confounders per DGP")
    parser.add_argument("--confounder-fractions", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Confounder subset fractions to test")
    parser.add_argument("--vary-confounders-per-dgp", type=lambda x: x.lower() == 'true',
                        default=None,
                        help="Whether to vary confounders per DGP (default: mode-dependent)")

    # Model hyperparameters
    parser.add_argument("--flp-model-name", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--max-length", type=int, default=10000)
    parser.add_argument("--downprojection-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cf-n-estimators", type=int, default=200)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=5)

    # DGP params
    parser.add_argument("--treatment-effect-prob", type=float, default=0.10)
    parser.add_argument("--target-treatment-rate", type=float, default=0.5)
    parser.add_argument("--target-control-outcome-rate", type=float, default=0.2)

    # vLLM (used for both confounder extraction and generation)
    parser.add_argument("--vllm-mode", default="server",
                        choices=["server", "start_server", "python_api"])
    parser.add_argument("--vllm-server-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model-name", default="openai/gpt-oss-120b")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (start_server/python_api)")
    parser.add_argument("--vllm-download-dir", default=None,
                        help="Model download directory (start_server/python_api)")
    parser.add_argument("--vllm-max-model-len", type=int, default=120000,
                        help="Maximum model context length (start_server/python_api)")
    parser.add_argument("--vllm-max-tokens", type=int, default=5000,
                        help="Maximum new tokens per LLM request")

    # Infrastructure
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--cache", action="store_true",
                        help="Pre-cache hidden states to disk")
    parser.add_argument("--gpu-cache", action="store_true",
                        help="Keep hidden states on GPU VRAM")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip completed arms)")

    args = parser.parse_args()

    run_experiments(
        dataset_path=args.dataset_path,
        clinical_question=args.clinical_question,
        output_dir=args.output_dir,
        equation_mode=args.equation_mode,
        num_dgps=args.num_dgps,
        num_repeats=args.num_repeats,
        num_confounders=args.num_confounders,
        confounder_fractions=args.confounder_fractions,
        vary_confounders_per_dgp=args.vary_confounders_per_dgp,
        flp_model_name=args.flp_model_name,
        flp_max_length=args.max_length,
        flp_downprojection_dim=args.downprojection_dim,
        flp_projection_dim=args.projection_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_folds=args.n_folds,
        cf_n_estimators=args.cf_n_estimators,
        cf_min_samples_leaf=args.cf_min_samples_leaf,
        treatment_effect_prob=args.treatment_effect_prob,
        target_treatment_rate=args.target_treatment_rate,
        target_control_outcome_rate=args.target_control_outcome_rate,
        vllm_mode=args.vllm_mode,
        vllm_server_url=args.vllm_server_url,
        vllm_model_name=args.vllm_model_name,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_download_dir=args.vllm_download_dir,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_tokens=args.vllm_max_tokens,
        devices=args.devices,
        cache=args.cache,
        gpu_cache=args.gpu_cache,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
